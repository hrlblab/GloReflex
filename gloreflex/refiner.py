"""SAM2.1 refiner: LoRA on the Hiera image encoder + fully trainable mask decoder.

The refiner takes an image patch and the current mask (as a dense mask prompt)
and returns a refined mask. Clicks are optional: they are used while training
and while collecting experience, but never at deployment.

* Image encoder (Hiera): base weights frozen, LoRA (rank 16, alpha 32,
  dropout 0.1) on every ``trunk.blocks.*.attn.{qkv, proj}`` linear layer.
* Prompt encoder: frozen.
* Mask decoder: fully trainable, including the IoU prediction head.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import CROP

CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_b+.yaml"
# Path of the SAM2.1 Hiera-B+ checkpoint (sam2.1_hiera_base_plus.pt); override with
# the SAM2_CKPT environment variable.
DEFAULT_CKPT_PATH = Path(os.environ.get("SAM2_CKPT", "checkpoints/sam2.1_hiera_base_plus.pt"))
DEFAULT_LORA_TARGETS = (
    r"image_encoder\.trunk\.blocks\.\d+\.attn\.qkv$",
    r"image_encoder\.trunk\.blocks\.\d+\.attn\.proj$",
)
IMAGE_SIZE = 1024          # SAM2 input resolution
MASK_LOGIT_SIZE = 256      # SAM2 low-resolution mask resolution


# ----------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """``nn.Linear`` with a frozen base weight and a trainable low-rank update:
    ``y = W x + b + (B A dropout(x)) * alpha / rank``."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        device, dtype = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + delta * self.scaling


def _resolve(model: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(model: nn.Module, target_patterns: Iterable[str] = DEFAULT_LORA_TARGETS, *,
                rank: int = 16, alpha: float = 32.0, dropout: float = 0.1) -> list[str]:
    patterns = [re.compile(p) for p in target_patterns]
    replaced: list[str] = []
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear) or not any(p.search(name) for p in patterns):
            continue
        parent, attr = _resolve(model, name)
        setattr(parent, attr, LoRALinear(mod, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(name)
    return replaced


def configure_finetune(model: nn.Module, *, lora_rank: int = 16, lora_alpha: float = 32.0,
                       lora_dropout: float = 0.1,
                       lora_targets: Iterable[str] = DEFAULT_LORA_TARGETS) -> dict:
    """Freeze everything, inject LoRA into the Hiera attention, unfreeze the mask decoder."""
    for p in model.parameters():
        p.requires_grad = False
    replaced = inject_lora(model, lora_targets, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
    for p in model.sam_mask_decoder.parameters():
        p.requires_grad = True
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"lora_targets_replaced": len(replaced), "n_params_total": int(n_total),
            "n_params_trainable": int(n_train), "trainable_pct": float(n_train) / float(n_total)}


def build_refiner(ckpt_dir: str | Path | None, device) -> torch.nn.Module:
    """SAM2.1 Hiera-B+ with LoRA; loads a fine-tuned state from ``ckpt_dir`` if given."""
    from sam2.build_sam import build_sam2
    model = build_sam2(CONFIG_NAME, ckpt_path=str(DEFAULT_CKPT_PATH), device=str(device))
    configure_finetune(model)
    if ckpt_dir is not None:
        load_finetuned_state(model, Path(ckpt_dir))
    return model


def sam_transforms():
    from sam2.utils.transforms import SAM2Transforms
    return SAM2Transforms(resolution=IMAGE_SIZE, mask_threshold=0.0,
                          max_hole_area=0.0, max_sprinkle_area=0.0)


# ----------------------------------------------------------------------------- forward
def _bb_feat_sizes_for(model: nn.Module) -> list[tuple[int, int]]:
    img_size = model.image_size if hasattr(model, "image_size") else 1024
    return [(img_size // 4,) * 2, (img_size // 8,) * 2, (img_size // 16,) * 2]


def encode_image(model: nn.Module, image_batch: torch.Tensor):
    """Image encoder forward with gradients (mirrors ``SAM2ImagePredictor.set_image``)."""
    backbone_out = model.forward_image(image_batch)
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
    bb_sizes = _bb_feat_sizes_for(model)
    B = image_batch.shape[0]
    feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size)
             for feat, feat_size in zip(vision_feats[::-1], bb_sizes[::-1])][::-1]
    return feats[-1], feats[:-1]


def forward(model, image, coords, labels, mask_logits):
    """One refinement forward pass. ``coords`` may be ``None`` (mask-only prompt).
    Returns ``(low_res_logits (B, 1, 256, 256), iou_pred (B, 1))``."""
    emb, hi = encode_image(model, image)
    pts = None if coords is None else (coords, labels)
    sparse, dense = model.sam_prompt_encoder(points=pts, boxes=None, masks=mask_logits)
    low, iou_pred, _, _ = model.sam_mask_decoder(
        image_embeddings=emb, image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False, repeat_image=False, high_res_features=hi)
    return low, iou_pred


def mask_to_logits(m: np.ndarray) -> np.ndarray:
    """Binary mask -> (1, 256, 256) mask-prompt logits (+-20)."""
    b = (m > 0).astype(np.uint8)
    r = cv2.resize(b, (MASK_LOGIT_SIZE, MASK_LOGIT_SIZE), interpolation=cv2.INTER_NEAREST)
    return np.where(r > 0, 20.0, -20.0).astype(np.float32)[None]


def to_target(m: np.ndarray) -> np.ndarray:
    b = (m > 0).astype(np.uint8)
    return cv2.resize(b, (MASK_LOGIT_SIZE, MASK_LOGIT_SIZE),
                      interpolation=cv2.INTER_NEAREST).astype(np.float32)[None]


def refine(model, img, cur, coords, labels, dev, amp):
    """One refinement step on a batch. ``coords=None`` means mask-only.
    Returns ``(new masks (B, CROP, CROP) uint8, iou_pred (B,))``."""
    mi = torch.from_numpy(np.stack([mask_to_logits(c) for c in cur])).to(dev)
    with torch.no_grad(), amp:
        low, iou = forward(model, img, coords, labels, mi)
    up = F.interpolate(low.float(), size=(CROP, CROP), mode="bilinear", align_corners=False)[:, 0] > 0
    return up.cpu().numpy().astype(np.uint8), iou.float().squeeze(-1).cpu().numpy()


# ----------------------------------------------------------------------------- losses
def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, *, eps: float = 1e-6):
    p = torch.sigmoid(logits).flatten(1)
    t = targets.flatten(1)
    dice = (2 * (p * t).sum(dim=1) + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)
    return 1.0 - dice.mean()


def bce_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


# ----------------------------------------------------------------------------- save / load
def state_paths(out_dir: Path) -> dict[str, Path]:
    return {"lora": out_dir / "lora_adapter.pt", "decoder": out_dir / "mask_decoder.pt",
            "meta": out_dir / "meta.json"}


def save_finetuned_state(model: nn.Module, out_dir: Path, extra_meta: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = state_paths(out_dir)
    lora_state = {n: p.detach().cpu() for n, p in model.named_parameters()
                  if p.requires_grad and ("lora_A" in n or "lora_B" in n)}
    torch.save(lora_state, paths["lora"])
    torch.save(model.sam_mask_decoder.state_dict(), paths["decoder"])
    meta = {"n_lora_params": sum(t.numel() for t in lora_state.values()),
            "decoder_n_params": sum(p.numel() for p in model.sam_mask_decoder.parameters())}
    meta.update(extra_meta or {})
    json.dump(meta, open(paths["meta"], "w"), indent=2)


def load_finetuned_state(model: nn.Module, in_dir: Path, *, strict_decoder: bool = True) -> dict:
    """Load LoRA + decoder weights into a model prepared with :func:`configure_finetune`."""
    paths = state_paths(Path(in_dir))
    lora_state = torch.load(paths["lora"], map_location="cpu")
    own = dict(model.named_parameters())
    missing = []
    for k, v in lora_state.items():
        if k not in own:
            missing.append(k); continue
        own[k].data.copy_(v.to(own[k].device))
    model.sam_mask_decoder.load_state_dict(torch.load(paths["decoder"], map_location="cpu"),
                                           strict=strict_decoder)
    meta = json.load(open(paths["meta"])) if paths["meta"].exists() else {}
    meta["missing_lora_keys"] = missing
    return meta

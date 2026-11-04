"""State encoder: overlay rendering -> frozen DINOv3 -> trained MLP head.

A state is an image patch with its current mask. The mask is drawn on the patch
(green fill at alpha 0.65 plus an 8-px red contour), the overlay is encoded by a
frozen DINOv3 ViT-B/16 (CLS token + four quadrant-pooled patch tokens, 5 x 768
dims), and a two-layer MLP maps the feature to a 128-d L2-normalised embedding.

The head is trained with a supervised contrastive loss plus prediction losses
(glomerulus vs. non-glomerulus, Dice regression and Dice-bin classification,
Dice-change regression and Dice-change-bin classification).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINOV3_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"   # gated on Hugging Face


# ----------------------------------------------------------------------------- overlay
def render_overlay(crop: np.ndarray, mask: np.ndarray, *, color=(0, 255, 0), alpha: float = 0.65,
                   boundary_color=(255, 0, 0), boundary_thickness: int = 8) -> np.ndarray:
    """Draw ``mask`` on ``crop`` (both HxW): tinted interior + thick contour."""
    if crop.dtype != np.uint8:
        crop = crop.astype(np.uint8)
    out = crop.copy()
    fg = mask > 0
    if fg.any():
        color_arr = np.array(color, dtype=np.float32)
        out[fg] = (alpha * color_arr + (1.0 - alpha) * out[fg].astype(np.float32)).clip(0, 255).astype(np.uint8)
    bin_mask = ((mask > 0).astype(np.uint8)) * 255
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, boundary_color, thickness=boundary_thickness)
    return out


# ----------------------------------------------------------------------------- backbone
@dataclass(frozen=True)
class BackboneSpec:
    name: str
    feature_dim: int
    image_size: int
    mean: tuple
    std: tuple


class DinoV3FeatureExtractor(nn.Module):
    """Frozen DINOv3 ViT returning ``(B, 5 * D)`` = CLS + four quadrant patch means."""

    def __init__(self, model_id: str = DINOV3_MODEL_ID, image_size: int = 512):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        cfg = self.model.config
        self.embed_dim = int(getattr(cfg, "hidden_size"))
        self.patch_size = int(getattr(cfg, "patch_size", 16))
        self.n_register = int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.image_size = image_size
        self.grid = image_size // self.patch_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hs = self.model(pixel_values=pixel_values).last_hidden_state   # (B, 1+R+P, D)
            cls = hs[:, 0, :]
            patches = hs[:, 1 + self.n_register:, :]
            B, P, D = patches.shape
            g = self.grid
            patches = patches[:, : g * g, :].reshape(B, g, g, D)
            h = g // 2 + (g % 2)
            q_tl = patches[:, :h, :h, :].mean(dim=(1, 2))
            q_tr = patches[:, :h, h:, :].mean(dim=(1, 2))
            q_bl = patches[:, h:, :h, :].mean(dim=(1, 2))
            q_br = patches[:, h:, h:, :].mean(dim=(1, 2))
            return torch.cat([cls, q_tl, q_tr, q_bl, q_br], dim=1)


def make_backbone(name: str = "dinov3") -> tuple[nn.Module, BackboneSpec]:
    if name not in ("dinov3", "dinov3_vitb16"):
        raise ValueError(f"unknown backbone: {name}")
    img = 512
    ext = DinoV3FeatureExtractor(image_size=img)
    try:
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID)
        mean, std = tuple(proc.image_mean), tuple(proc.image_std)
    except Exception:  # noqa: BLE001
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    return ext, BackboneSpec("dinov3", 5 * ext.embed_dim, img, mean, std)


def overlay_tensor(crop: np.ndarray, mask01: np.ndarray, spec: BackboneSpec) -> np.ndarray:
    """Normalised CHW float32 input for the backbone. ``mask01`` is a {0, 1} or {0, 255} mask."""
    from PIL import Image
    ov = render_overlay(crop, mask01)
    ov = np.asarray(Image.fromarray(ov).resize((spec.image_size,) * 2, Image.BILINEAR))
    mean = np.asarray(spec.mean, np.float32).reshape(3, 1, 1)
    std = np.asarray(spec.std, np.float32).reshape(3, 1, 1)
    return (ov.astype(np.float32).transpose(2, 0, 1) / 255.0 - mean) / std


# ----------------------------------------------------------------------------- head
N_DICE_BINS = 10
# Dice-change bins follow the decision semantics: the sign near zero is noise; what
# matters is whether a gain worth another refinement step remains.
DELTA_EDGES = np.array([-1e9, -0.02, -0.005, 0.0, 0.005, 0.02, 0.05, 0.15, 1e9])
N_DELTA_BINS = len(DELTA_EDGES) - 1


def dice_bin(y):
    return np.clip((y * N_DICE_BINS).astype(np.int64), 0, N_DICE_BINS - 1)


def delta_bin(y):
    return np.clip(np.digitize(y, DELTA_EDGES[1:-1]), 0, N_DELTA_BINS - 1)


class StateHead(nn.Module):
    """Two-layer MLP projection to a 128-d embedding plus prediction heads."""

    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim))
        self.classifier = nn.Linear(out_dim, 2)          # glomerulus vs. non-glomerulus
        self.dice_head = nn.Linear(out_dim, 1)           # Dice of the current mask
        self.delta_head = nn.Linear(out_dim, 1)          # Dice change after one more step
        self.dicebin = nn.Linear(out_dim, N_DICE_BINS)
        self.deltabin = nn.Linear(out_dim, N_DELTA_BINS)

    def forward(self, x):
        h = self.proj(x)
        return (F.normalize(h, p=2, dim=1), self.classifier(h),
                torch.sigmoid(self.dice_head(h)).squeeze(-1),
                self.delta_head(h).squeeze(-1),
                self.dicebin(h), self.deltabin(h))


def supcon_loss(z: torch.Tensor, label: torch.Tensor, group_id: list[str], *,
                temperature: float = 0.1, hard_neg_weight: float = 3.0) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020) in which negatives that share
    ``group_id`` with the anchor (same detection, other label) are up-weighted."""
    device = z.device
    B = z.shape[0]
    sim = (z @ z.t()) / temperature
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    label = label.to(device)
    pos_mask = (label.unsqueeze(0) == label.unsqueeze(1)) & (~self_mask)
    gid = torch.tensor([hash(c) for c in group_id], device=device, dtype=torch.long)
    hard_neg_mask = (gid.unsqueeze(0) == gid.unsqueeze(1)) & (label.unsqueeze(0) != label.unsqueeze(1))
    log_w = torch.zeros_like(sim)
    if hard_neg_weight != 1.0:
        log_w = log_w + hard_neg_mask.float() * torch.log(
            torch.tensor(hard_neg_weight, device=device, dtype=sim.dtype))
    log_denom = torch.logsumexp(sim + log_w, dim=1)
    pos_count = pos_mask.float().sum(dim=1)
    valid = pos_count > 0
    if not valid.any():
        return sim.new_zeros(())
    pos_sum = sim.masked_fill(~pos_mask, 0.0).sum(dim=1)
    per_anchor = pos_sum / pos_count - log_denom
    return -per_anchor[valid].mean()


def head_losses(model: StateHead, xb, y_cls, y_d, y_dd, sid, dev, w: dict):
    """Total training loss of the head on one batch (labels are numpy arrays; NaN = no label)."""
    z, logit, dp, ddp, dbl, ddbl = model(xb)
    cls = torch.from_numpy(y_cls).to(dev)
    sup = supcon_loss(z, cls, list(sid))
    ce = F.cross_entropy(logit, cls)
    md = torch.from_numpy(np.isfinite(y_d)).to(dev)
    mdd = torch.from_numpy(np.isfinite(y_dd)).to(dev)
    ld = (F.mse_loss(dp[md], torch.from_numpy(np.nan_to_num(y_d)).to(dev)[md])
          if md.any() else dp.sum() * 0)
    ldd = (F.mse_loss(ddp[mdd], torch.from_numpy(np.nan_to_num(y_dd)).to(dev)[mdd])
           if mdd.any() else ddp.sum() * 0)
    ldb = (F.cross_entropy(dbl[md], torch.from_numpy(dice_bin(np.nan_to_num(y_d))).to(dev)[md])
           if md.any() else dbl.sum() * 0)
    lddb = (F.cross_entropy(ddbl[mdd], torch.from_numpy(delta_bin(np.nan_to_num(y_dd))).to(dev)[mdd])
            if mdd.any() else ddbl.sum() * 0)
    loss = (w["supcon_w"] * sup + w["ce_w"] * ce + w["dice_w"] * ld + w["delta_w"] * ldd
            + w["dicebin_w"] * ldb + w["deltabin_w"] * lddb)
    parts = {"supcon": sup, "ce": ce, "dice": ld, "delta": ldd, "dicebin": ldb, "deltabin": lddb}
    return loss, parts


@torch.no_grad()
def embed_features(model: StateHead, X: np.ndarray, dev, chunk: int = 4096) -> np.ndarray:
    """128-d embeddings of cached backbone features."""
    Xt = torch.from_numpy(X)
    return torch.cat([model(Xt[i:i + chunk].to(dev))[0].cpu()
                      for i in range(0, len(Xt), chunk)]).numpy().astype(np.float32)

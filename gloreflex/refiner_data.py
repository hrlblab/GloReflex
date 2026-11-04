"""Training data for the refiner and click sampling.

A row is a (patch, detector mask, GT mask) triplet. Clicks are sampled on the fly
from the error region between the current mask and the GT:

* clicks are drawn uniformly from the error region, with a jitter that scales
  with the size of the error component; a jittered click that leaves the region
  of its polarity falls back to the original point, so no click is mislabelled;
* 7% of clicks are redundant but correctly labelled (inside an already correct
  foreground or background region), so that a click does not imply that an
  already correct region must change.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def sample_click(prior: np.ndarray, gt: np.ndarray, rng: np.random.Generator,
                 p_redundant: float = 0.07, jitter_frac: float = 0.15):
    """Sample one click from the error region of ``prior`` w.r.t. ``gt``.

    Returns ``(x, y, label, kind)`` with ``label`` 1 = should be foreground,
    0 = should be background, or ``None`` if there is nowhere to click.
    """
    p, g = prior > 0, gt > 0
    if rng.random() < p_redundant:
        if rng.random() < 0.5:
            reg, lab, kind = p & g, 1, "redundant_pos"
        else:
            reg, lab, kind = ~p & ~g, 0, "redundant_neg"
    else:
        fn, fp = g & ~p, p & ~g
        if fn.sum() == 0 and fp.sum() == 0:
            reg, lab, kind = p & g, 1, "no_error_pos"
            if reg.sum() == 0:
                return None
        elif fp.sum() == 0 or (fn.sum() > 0 and rng.random() < fn.sum() / (fn.sum() + fp.sum())):
            reg, lab, kind = fn, 1, "error_pos"
        else:
            reg, lab, kind = fp, 0, "error_neg"
    ys, xs = np.nonzero(reg)
    if len(ys) == 0:
        return None
    i = int(rng.integers(len(ys)))
    x, y = int(xs[i]), int(ys[i])
    n, lbl = cv2.connectedComponents(reg.astype(np.uint8))
    area = int((lbl == lbl[y, x]).sum())
    sig = max(jitter_frac * np.sqrt(max(area, 1)), 0.0)
    if sig > 0.5:
        jx = int(round(x + rng.normal(0, sig))); jy = int(round(y + rng.normal(0, sig)))
        H, W = reg.shape
        if 0 <= jx < W and 0 <= jy < H and reg[jy, jx]:
            x, y = jx, jy
    return x, y, lab, kind


class RefinerDataset(Dataset):
    """Rows of the refiner training manifest. Rows without GT (``has_gt=False``,
    pure false positives) only supervise the IoU head (target 0)."""

    def __init__(self, rows, transforms, aug: bool = True, seed: int = 2026):
        self.rows, self.tf, self.aug = rows, transforms, aug
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        crop = np.array(Image.open(r["crop_path"]).convert("RGB"))
        prior = np.array(Image.open(r["prior_path"]).convert("L"))
        has_gt = bool(r["has_gt"])
        gt = np.array(Image.open(r["gt_path"]).convert("L")) if has_gt else np.zeros(prior.shape, np.uint8)

        if self.aug:
            if rng.random() < 0.5:
                crop, prior, gt = crop[:, ::-1], prior[:, ::-1], gt[:, ::-1]
            if rng.random() < 0.5:
                crop, prior, gt = crop[::-1], prior[::-1], gt[::-1]
            if rng.random() < 0.5:                       # transpose (90-degree family)
                crop, prior, gt = crop.transpose(1, 0, 2), prior.T, gt.T
            if rng.random() < 0.5:                       # scale jitter (covers re-crop rescaling)
                z = float(rng.uniform(1.0, 1.3)); H, W = prior.shape
                h, w = int(H / z), int(W / z)
                y0, x0 = (H - h) // 2, (W - w) // 2
                crop = cv2.resize(np.ascontiguousarray(crop[y0:y0 + h, x0:x0 + w]), (W, H),
                                  interpolation=cv2.INTER_LINEAR)
                prior = cv2.resize(np.ascontiguousarray(prior[y0:y0 + h, x0:x0 + w]), (W, H),
                                   interpolation=cv2.INTER_NEAREST)
                gt = cv2.resize(np.ascontiguousarray(gt[y0:y0 + h, x0:x0 + w]), (W, H),
                                interpolation=cv2.INTER_NEAREST)
            if rng.random() < 0.5:                       # mild stain jitter
                f = rng.uniform(0.92, 1.08, size=3)
                crop = np.clip(np.ascontiguousarray(crop).astype(np.float32) * f, 0, 255).astype(np.uint8)
        crop = np.ascontiguousarray(crop); prior = np.ascontiguousarray(prior)
        gt = np.ascontiguousarray(gt)
        return {
            "image": self.tf(Image.fromarray(crop)),
            "prior": torch.from_numpy((prior > 0).astype(np.uint8)),
            "gt": torch.from_numpy((gt > 0).astype(np.uint8)),
            "has_gt": torch.tensor(1 if has_gt else 0, dtype=torch.uint8),
            "sample_id": r["sample_id"],
            "wsi_id": r["wsi_id"],
            "source": r["source"],
            "prior_dice": float(r["prior_dice"]) if r["prior_dice"] == r["prior_dice"] else -1.0,
        }

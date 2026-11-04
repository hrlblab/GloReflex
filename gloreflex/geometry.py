"""Patch geometry, mask placement and detection-to-GT matching.

Every detection is examined in a square patch centred on its box with side
``1.5 * max(box width, box height)`` (:func:`square_crop_box`). Patches are
resampled to ``CROP x CROP`` pixels; masks live in that patch frame.
"""
from __future__ import annotations

import cv2
import numpy as np
from pycocotools import mask as mask_util

CROP = 1024            # patch resolution seen by the refiner and the encoder
MARGIN = 1.5           # patch side = MARGIN * longest box side
IOU_TP = 0.5           # box IoU for a true positive
IOU_NEAR = 0.1         # box IoU below which a detection touches no glomerulus


def square_crop_box(cx: float, cy: float, x0, y0, x1, y1,
                    margin_factor: float = MARGIN) -> tuple[int, int, int, int]:
    """Square patch centred on (cx, cy) with side ``margin_factor * max(box dims)``.
    Returns ``(x0, y0, side, side)`` in level-0 pixels."""
    side = max(x1 - x0, y1 - y0) * margin_factor
    half = side / 2.0
    cx0 = int(round(cx - half))
    cy0 = int(round(cy - half))
    s = int(round(side))
    return cx0, cy0, s, s


def detection_patch(pred) -> tuple[int, int, int]:
    """``(x0, y0, side)`` of the examination patch of a detection."""
    x0, y0, x1, y1 = [float(v) for v in pred["bbox"]]
    cx0, cy0, side, _ = square_crop_box((x0 + x1) / 2, (y0 + y1) / 2, x0, y0, x1, y1, MARGIN)
    return cx0, cy0, side


def paste_mask(pm: np.ndarray, x0: float, y0: float, cx0: int, cy0: int, side: int) -> np.ndarray:
    """Paste a bbox-local mask (top-left at level-0 ``(x0, y0)``) into a
    ``side x side`` window whose top-left is ``(cx0, cy0)``."""
    full = np.zeros((side, side), np.uint8)
    bx0, by0 = int(round(x0)) - cx0, int(round(y0)) - cy0
    h, w = pm.shape
    ys, xs = slice(max(by0, 0), min(by0 + h, side)), slice(max(bx0, 0), min(bx0 + w, side))
    sy, sx = slice(max(-by0, 0), max(-by0, 0) + (ys.stop - ys.start)), \
        slice(max(-bx0, 0), max(-bx0, 0) + (xs.stop - xs.start))
    if ys.stop > ys.start and xs.stop > xs.start:
        full[ys, xs] = pm[sy, sx]
    return full


def place_prior(pm: np.ndarray, x0, y0, cx0, cy0, side) -> np.ndarray:
    """Detector mask in the ``CROP x CROP`` patch frame."""
    return cv2.resize(paste_mask(pm, x0, y0, cx0, cy0, side), (CROP, CROP),
                      interpolation=cv2.INTER_NEAREST)


def to_wsi_rle(m: np.ndarray, cx0: int, cy0: int, side: int, W: int, H: int):
    """Patch-frame mask -> level-0 ``{"bbox", "seg"}`` with a tight box (``None`` if empty)."""
    if not m.any():
        return None
    up = cv2.resize(m.astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(up)
    gx0, gx1 = cx0 + int(xs.min()), cx0 + int(xs.max()) + 1
    gy0, gy1 = cy0 + int(ys.min()), cy0 + int(ys.max()) + 1
    gx0, gy0 = max(gx0, 0), max(gy0, 0); gx1, gy1 = min(gx1, W), min(gy1, H)
    if gx1 <= gx0 or gy1 <= gy0:
        return None
    sub = up[gy0 - cy0:gy1 - cy0, gx0 - cx0:gx1 - cx0]
    r = mask_util.encode(np.asfortranarray(sub.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii")
    return {"bbox": [gx0, gy0, gx1, gy1], "seg": r}


def poly_mask(inst, ox, oy, W, H) -> np.ndarray:
    """Rasterise a GT polygon (with holes) into a ``W x H`` window at ``(ox, oy)``."""
    m = np.zeros((H, W), np.uint8)
    e = np.asarray(inst["exterior"], np.float64)
    if e.ndim != 2 or len(e) < 3:
        return m
    cv2.fillPoly(m, [np.round(e - [ox, oy]).astype(np.int32)], 1)
    for h in (inst.get("holes") or []):
        ha = np.asarray(h, np.float64)
        if ha.ndim == 2 and len(ha) >= 3:
            cv2.fillPoly(m, [np.round(ha - [ox, oy]).astype(np.int32)], 0)
    return m


def poly_in_window(inst, cx0: int, cy0: int, side: int) -> np.ndarray:
    """GT polygon rasterised into a patch window and resampled to ``CROP x CROP``."""
    m = np.zeros((side, side), np.uint8)
    ext = np.asarray(inst["exterior"], np.float64)
    if ext.ndim != 2 or len(ext) < 3:
        return cv2.resize(m, (CROP, CROP), interpolation=cv2.INTER_NEAREST)
    cv2.fillPoly(m, [np.round(ext - [cx0, cy0]).astype(np.int32)], 1)
    for h in (inst.get("holes") or []):
        ha = np.asarray(h, np.float64)
        if ha.ndim == 2 and len(ha) >= 3:
            cv2.fillPoly(m, [np.round(ha - [cx0, cy0]).astype(np.int32)], 0)
    return cv2.resize(m, (CROP, CROP), interpolation=cv2.INTER_NEAREST)


def poly_area(inst) -> float:
    """Area (level-0 pixels) of a GT instance, holes excluded."""
    ext = np.asarray(inst["exterior"], np.float64)
    if ext.ndim != 2 or len(ext) < 3:
        return 0.0
    a = abs(cv2.contourArea(ext.astype(np.float32)))
    for h in (inst.get("holes") or []):
        ha = np.asarray(h, np.float64)
        if ha.ndim == 2 and len(ha) >= 3:
            a -= abs(cv2.contourArea(ha.astype(np.float32)))
    return float(max(a, 0.0))


def mask_aspect(m: np.ndarray) -> float:
    """Aspect ratio of the rotated minimum-area rectangle of the largest component."""
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return float("nan")
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) <= 0:
        return float("nan")
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    return float(max(w, h) / min(w, h)) if min(w, h) > 1e-6 else float("nan")


def dice_np(a: np.ndarray, b: np.ndarray, empty: float = 1.0) -> float:
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / s) if s else empty


# ----------------------------------------------------------------------------- matching
def bbox_iou_mat(pb: np.ndarray, gb: np.ndarray) -> np.ndarray:
    """Pairwise box IoU, ``pb`` (n, 4) and ``gb`` (m, 4) in x0y0x1y1."""
    if len(pb) == 0 or len(gb) == 0:
        return np.zeros((len(pb), len(gb)))
    x0 = np.maximum(pb[:, None, 0], gb[None, :, 0]); y0 = np.maximum(pb[:, None, 1], gb[None, :, 1])
    x1 = np.minimum(pb[:, None, 2], gb[None, :, 2]); y1 = np.minimum(pb[:, None, 3], gb[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ap = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])
    ag = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
    return inter / np.maximum(ap[:, None] + ag[None, :] - inter, 1e-9)


def greedy_match(M: np.ndarray, thr: float = IOU_TP) -> list[int]:
    """WSI-level greedy matching of score-sorted detections (rows of ``M``) to GT
    (columns): each GT is used once. Returns the matched GT index per row (-1 if none)."""
    used, match = set(), [-1] * M.shape[0]
    for k in range(M.shape[0]):
        bj, bi = -1, thr
        for j in np.nonzero(M[k] >= thr)[0]:
            if j not in used and M[k, j] >= bi:
                bj, bi = int(j), M[k, j]
        if bj >= 0:
            used.add(bj); match[k] = bj
    return match


def categorize(M: np.ndarray, k: int, match: list[int]) -> tuple[str, int]:
    """Relation of detection ``k`` to the GT:

    * ``tp``        matched;
    * ``duplicate`` unmatched but box IoU >= 0.5 with some GT (a second box on a glomerulus);
    * ``nearmiss``  0.1 <= best box IoU < 0.5;
    * ``fp``        best box IoU < 0.1.

    Returns ``(category, gt_index)``; ``gt_index`` is the matched GT, or the best
    overlapping GT for duplicates / near misses, or -1.
    """
    best_iou = float(M[k].max()) if M.shape[1] else 0.0
    best_j = int(M[k].argmax()) if M.shape[1] else -1
    if match[k] >= 0:
        return "tp", match[k]
    if best_iou >= IOU_TP:
        return "duplicate", best_j
    if best_iou >= IOU_NEAR:
        return "nearmiss", best_j
    return "fp", -1

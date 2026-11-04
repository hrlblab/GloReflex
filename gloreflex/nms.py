"""Slide-level mask NMS applied to the accepted masks of a WSI.

Refinement and re-crop can turn several overlapping detections (near misses,
boxes cut by a tile border) into the same complete glomerulus. The accepted
masks of a WSI are therefore merged by greedy mask NMS in descending detector
score: a mask is dropped if its mask IoU with an already kept mask exceeds 0.5.

Because the greedy order is by score, a low-score prediction can only be
suppressed by a higher-score one; running NMS on the predictions with score
>= 0.3 gives the same result on that subset as running it on all predictions.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mu

from .metrics import pred_mask_on

NMS_IOU = 0.5
PAIR_LO = 0.3          # pairs whose IoU upper bound is <= this are never rasterised


def _rle(p):
    s = p["seg"]; c = s["counts"]
    return {"size": s["size"], "counts": c.encode("ascii") if isinstance(c, str) else c}


def pair_ious(preds: list[dict], lo: float = PAIR_LO) -> dict[int, list[tuple[int, float]]]:
    """Adjacency ``{i: [(j, iou), ...]}`` of all pairs with mask IoU > ``lo``.
    Pairs are pre-filtered with the bound IoU <= box intersection / max(mask area)."""
    n = len(preds)
    adj: dict[int, list[tuple[int, float]]] = {}
    if n < 2:
        return adj
    bb = np.array([[int(round(v)) for v in p["bbox"]] for p in preds], np.int64)
    area = np.array([float(mu.area(_rle(p))) for p in preds])
    ii, jj = [], []
    for s0 in range(0, n, 1024):
        r = bb[s0:s0 + 1024]
        x0 = np.maximum(r[:, None, 0], bb[None, :, 0]); y0 = np.maximum(r[:, None, 1], bb[None, :, 1])
        x1 = np.minimum(r[:, None, 2], bb[None, :, 2]); y1 = np.minimum(r[:, None, 3], bb[None, :, 3])
        inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        ub = inter / np.maximum(np.maximum(area[s0:s0 + 1024, None], area[None, :]), 1.0)
        a, b = np.nonzero(ub > lo)
        a = a + s0; m = b > a
        ii += a[m].tolist(); jj += b[m].tolist()
    for i, j in zip(ii, jj):
        a, b = preds[i]["bbox"], preds[j]["bbox"]
        ux0, uy0 = int(min(a[0], b[0])), int(min(a[1], b[1]))
        ux1, uy1 = int(np.ceil(max(a[2], b[2]))), int(np.ceil(max(a[3], b[3])))
        win = (ux0, uy0, ux1, uy1)
        ma, mb = pred_mask_on(win, preds[i]), pred_mask_on(win, preds[j])
        it = int(np.logical_and(ma, mb).sum()); un = int(ma.sum()) + int(mb.sum()) - it
        v = it / un if un > 0 else 0.0
        if v > lo:
            adj.setdefault(i, []).append((j, v)); adj.setdefault(j, []).append((i, v))
    return adj


def nms(scores, sel, adj, thr: float = NMS_IOU) -> list[int]:
    """Greedy NMS over the candidate indices ``sel`` (stable sort by descending score).
    Returns the kept indices in ascending order."""
    sel = np.asarray(sel, int)
    if not len(sel):
        return []
    sc = np.asarray(scores, float)[sel]
    order = sel[np.argsort(-sc, kind="stable")]
    kept = set()
    for i in order.tolist():
        if not any(j in kept and v > thr for j, v in adj.get(i, ())):
            kept.add(i)
    return sorted(kept)


def mask_nms(preds: list[dict], thr: float = NMS_IOU) -> list[dict]:
    """Apply mask NMS to all predictions of one WSI, keeping the original order."""
    adj = pair_ious(preds, lo=thr)
    keep = nms([float(p["score"]) for p in preds], range(len(preds)), adj, thr)
    return [preds[i] for i in keep]

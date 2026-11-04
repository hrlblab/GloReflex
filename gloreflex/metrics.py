"""WSI-level instance detection / segmentation scoring (framework agnostic).

Ground truth
    A COCO-style JSON with one image per WSI (``wsi_id``, ``source_dataset``,
    ``width``, ``height``) and one annotation per glomerulus (``bbox`` in
    ``[x, y, w, h]`` level-0 pixels, ``wsi_instance_id``). The polygon of each
    instance is read from ``<poly_root>/<wsi_id>/<wsi_instance_id>.json`` with
    keys ``exterior`` and ``holes``.

Predictions
    One JSON per WSI, ``<preds_dir>/<wsi_id>.json``::

        {"wsi_id": ..., "preds": [{"bbox": [x0, y0, x1, y1], "score": float,
                                   "seg": {"size": [h, w], "counts": str}}]}

    ``seg`` is a COCO RLE local to the (rounded) bbox window.

Metrics
    * recall / precision / F1 at an operating score threshold: greedy matching in
      descending score order, a prediction is a true positive if its box overlaps
      an unmatched GT box with IoU >= 0.5;
    * mean Dice / IoU of the matched pairs;
    * COCO-style box and mask AP (101-point interpolation, IoU 0.50:0.95), with
      detections pooled over all WSIs of a source.

Masks are only rasterised on the union window of each overlapping pair, so
gigapixel slides never have to be held in memory.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
from pycocotools import mask as mask_util

IOU_THRS = np.round(np.arange(0.5, 1.0, 0.05), 2)
POOLED = "POOLED"


# ----------------------------------------------------------------------------- masks
def poly_to_mask(exterior, holes, ox, oy, W, H):
    m = np.zeros((H, W), np.uint8)
    e = np.asarray(exterior, np.float64)
    if e.ndim != 2 or len(e) < 3:
        return m
    e[:, 0] -= ox; e[:, 1] -= oy
    cv2.fillPoly(m, [np.round(e).astype(np.int32)], 1)
    for h in (holes or []):
        ha = np.asarray(h, np.float64)
        if ha.ndim != 2 or len(ha) < 3:
            continue
        ha[:, 0] -= ox; ha[:, 1] -= oy
        cv2.fillPoly(m, [np.round(ha).astype(np.int32)], 0)
    return m


def bbox_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def pred_mask_on(window, pred):
    """Rasterise a prediction's RLE mask into ``window = (x0, y0, x1, y1)``."""
    ux0, uy0, ux1, uy1 = window
    W, H = ux1 - ux0, uy1 - uy0
    m = np.zeros((H, W), np.uint8)
    rle = dict(pred["seg"])
    if isinstance(rle["counts"], str):
        rle = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
    pm = mask_util.decode(rle)
    px0, py0 = int(round(pred["bbox"][0])) - ux0, int(round(pred["bbox"][1])) - uy0
    h, w = pm.shape
    y0d, x0d = max(0, py0), max(0, px0)
    y1d, x1d = min(H, py0 + h), min(W, px0 + w)
    if y1d > y0d and x1d > x0d:
        m[y0d:y1d, x0d:x1d] = pm[y0d - py0:y1d - py0, x0d - px0:x1d - px0]
    return m


def mask_iou_dice(pred, gt):
    """Mask IoU and Dice between a prediction and a GT instance (``bbox``, ``exterior``, ``holes``)."""
    pb, gb = pred["bbox"], gt["bbox"]
    ux0, uy0 = int(min(pb[0], gb[0])), int(min(pb[1], gb[1]))
    ux1, uy1 = int(np.ceil(max(pb[2], gb[2]))), int(np.ceil(max(pb[3], gb[3])))
    if ux1 <= ux0 or uy1 <= uy0:
        return 0.0, 0.0
    win = (ux0, uy0, ux1, uy1)
    pm = pred_mask_on(win, pred)
    gm = poly_to_mask(gt["exterior"], gt["holes"], ux0, uy0, ux1 - ux0, uy1 - uy0)
    inter = int(np.logical_and(pm, gm).sum())
    ps, gs = int(pm.sum()), int(gm.sum())
    union = ps + gs - inter
    iou = inter / union if union > 0 else 0.0
    dice = 2 * inter / (ps + gs) if (ps + gs) > 0 else 0.0
    return iou, dice


# ----------------------------------------------------------------------------- loading
def load_gt(gt_json, poly_root):
    """``{wsi_id: {"source", "width", "height", "inst": [{"bbox", "exterior", "holes"}]}}``."""
    tj = json.load(open(gt_json))
    img = {im["id"]: im for im in tj["images"]}
    wsis = {}
    for im in tj["images"]:
        wsis[im["wsi_id"]] = {"source": im["source_dataset"], "width": im["width"],
                              "height": im["height"], "inst": []}
    for ann in tj["annotations"]:
        im = img[ann["image_id"]]
        wid = im["wsi_id"]; iid = ann["wsi_instance_id"]
        x, y, w, h = ann["bbox"]
        pf = f"{poly_root}/{wid}/{iid}.json"
        if not os.path.exists(pf):
            continue
        pg = json.load(open(pf))
        wsis[wid]["inst"].append({"bbox": [x, y, x + w, y + h],
                                  "exterior": pg.get("exterior"), "holes": pg.get("holes")})
    return wsis


def load_preds(preds_dir, wid):
    pf = f"{preds_dir}/{wid}.json"
    if not os.path.exists(pf):
        return []
    return json.load(open(pf)).get("preds", [])


def decode_seg(pred) -> np.ndarray:
    """Decode a prediction's bbox-local RLE mask to a uint8 array."""
    s = pred["seg"]; c = s["counts"]
    return mask_util.decode({"size": s["size"],
                             "counts": c.encode("ascii") if isinstance(c, str) else c}).astype(np.uint8)


# ----------------------------------------------------------------------------- matching
def iou_matrices(preds, gts):
    n, m = len(preds), len(gts)
    bb = np.zeros((n, m)); mk = np.zeros((n, m)); dc = np.zeros((n, m))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            b = bbox_iou(p["bbox"], g["bbox"])
            bb[i, j] = b
            if b > 0:
                mk[i, j], dc[i, j] = mask_iou_dice(p, g)
    return bb, mk, dc


def coco_ap(records, n_gt, iou_thr):
    """AP from pooled ``(score, is_tp)`` records with 101-point interpolation."""
    if n_gt == 0:
        return float("nan")
    if not records:
        return 0.0
    records = sorted(records, key=lambda r: -r[0])
    tp = np.array([r[1] for r in records], dtype=float)
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p / 101
    return ap


def match_at(iou_mat, scores, n_gt, thr):
    """Greedy matching in descending score order; returns ``[(score, is_tp)]``."""
    order = np.argsort(-np.asarray(scores))
    gt_used = np.zeros(iou_mat.shape[1], dtype=bool)
    out = []
    for i in order:
        j_best, v_best = -1, thr
        row = iou_mat[i]
        for j in range(len(gt_used)):
            if not gt_used[j] and row[j] >= v_best:
                v_best, j_best = row[j], j
        if j_best >= 0:
            gt_used[j_best] = True
            out.append((scores[i], 1.0))
        else:
            out.append((scores[i], 0.0))
    return out


def _empty_acc():
    return {"n_gt": 0, "bbox": {t: [] for t in IOU_THRS}, "mask": {t: [] for t in IOU_THRS},
            "tp": 0, "fp": 0, "fn": 0, "dice": [], "iou": []}


def score(preds_dir, gt, sources, score_thr):
    """Accumulate per-source statistics for every WSI in ``gt`` whose source is in ``sources``."""
    src_set = set(sources)
    acc = {s: _empty_acc() for s in src_set}
    for wid, g in gt.items():
        s = g["source"]
        if s not in src_set:
            continue
        gts = g["inst"]
        preds = list(load_preds(preds_dir, wid))
        a = acc[s]
        a["n_gt"] += len(gts)
        if len(gts) == 0 and len(preds) == 0:
            continue
        scores = [p["score"] for p in preds]
        if len(preds) and len(gts):
            bb, mk, dc = iou_matrices(preds, gts)
        else:
            bb = np.zeros((len(preds), len(gts))); mk = bb.copy(); dc = bb.copy()
        for t in IOU_THRS:                                   # AP uses all predictions
            if len(preds):
                a["bbox"][t] += match_at(bb, scores, len(gts), t)
                a["mask"][t] += match_at(mk, scores, len(gts), t)
        keep = [i for i, sc in enumerate(scores) if sc >= score_thr]
        if keep and len(gts):
            sub_scores = [scores[i] for i in keep]
            order = np.argsort(-np.asarray(sub_scores))
            gt_used = np.zeros(len(gts), dtype=bool)
            tp = 0
            for oi in order:
                pi = keep[oi]
                row = bb[pi]
                j_best, v_best = -1, 0.5
                for j in range(len(gts)):
                    if not gt_used[j] and row[j] >= v_best:
                        v_best, j_best = row[j], j
                if j_best >= 0:
                    gt_used[j_best] = True
                    tp += 1
                    a["dice"].append(dc[pi, j_best])
                    a["iou"].append(mk[pi, j_best])
            a["tp"] += tp
            a["fp"] += len(keep) - tp
            a["fn"] += len(gts) - tp
        else:
            a["fp"] += len(keep)
            a["fn"] += len(gts)
    return acc


def summarize(acc):
    """Per-source rows plus a pooled row (key :data:`POOLED`)."""
    rows = {}
    pool = _empty_acc()
    for s, a in acc.items():
        pool["n_gt"] += a["n_gt"]; pool["tp"] += a["tp"]; pool["fp"] += a["fp"]; pool["fn"] += a["fn"]
        pool["dice"] += a["dice"]; pool["iou"] += a["iou"]
        for t in IOU_THRS:
            pool["bbox"][t] += a["bbox"][t]; pool["mask"][t] += a["mask"][t]
    for name, a in list(acc.items()) + [(POOLED, pool)]:
        def ap(kind):
            return {float(t): coco_ap(a[kind][t], a["n_gt"], t) for t in IOU_THRS}
        bap, mapp = ap("bbox"), ap("mask")
        tp, fp, fn = a["tp"], a["fp"], a["fn"]
        R = tp / (tp + fn) if tp + fn else float("nan")
        P = tp / (tp + fp) if tp + fp else float("nan")
        F1 = 2 * P * R / (P + R) if P and R and not (np.isnan(P) or np.isnan(R)) and (P + R) > 0 else float("nan")
        rows[name] = {
            "n_gt": a["n_gt"], "tp": tp, "fp": fp, "fn": fn,
            "recall": R, "precision": P, "f1": F1,
            "bbox_mAP": float(np.nanmean(list(bap.values()))), "bbox_AP50": bap[0.5], "bbox_AP75": bap[0.75],
            "mask_mAP": float(np.nanmean(list(mapp.values()))), "mask_AP50": mapp[0.5], "mask_AP75": mapp[0.75],
            "dice_mean": float(np.mean(a["dice"])) if a["dice"] else float("nan"),
            "dice_median": float(np.median(a["dice"])) if a["dice"] else float("nan"),
            "iou_mean": float(np.mean(a["iou"])) if a["iou"] else float("nan"),
            "n_matched": len(a["dice"]),
        }
    return rows

#!/usr/bin/env python
"""Relate every training detection to the ground truth.

Detections with score >= 0.3, sorted by descending score, are matched to GT by
WSI-level greedy matching (box IoU >= 0.5). Each detection receives a category
(``tp`` / ``duplicate`` / ``nearmiss`` / ``fp``, see
:func:`gloreflex.geometry.categorize`) and, when it overlaps a GT instance, three
reachability quantities computed on level-0 pixels in its examination patch:

* ``prior_on_gt``     = area(mask ∩ GT) / area(mask)       does the mask lie on the glomerulus?
* ``gt_in_view_frac`` = area(GT ∩ patch) / area(GT)        how much of the glomerulus is in view?
* ``gt_crop_cover``   = area(GT ∩ patch) / area(patch)

    python tools/experience/label_detections.py --preds-dir <train_preds03> \
        --gt-json data/train_wsi.json --poly-root data/polygons --split train --out <dir>/labels_train.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gloreflex.geometry import (bbox_iou_mat, categorize, detection_patch, greedy_match,
                                paste_mask, poly_mask)
from gloreflex.metrics import decode_seg, load_gt, load_preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--poly-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--sources", default=None, help="comma-separated source_dataset filter")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = load_gt(args.gt_json, args.poly_root)
    if args.sources:
        keep = set(args.sources.split(","))
        gt = {w: g for w, g in gt.items() if g["source"] in keep}
    rows, t0 = [], time.time()
    for i, (wid, g) in enumerate(sorted(gt.items()), 1):
        preds = [p for p in load_preds(args.preds_dir, wid) if p["score"] >= args.score_thr]
        if not preds:
            continue
        preds.sort(key=lambda p: -p["score"])
        M = bbox_iou_mat(np.array([p["bbox"] for p in preds], float),
                         np.array([inst["bbox"] for inst in g["inst"]], float))
        match = greedy_match(M)
        for k, p in enumerate(preds):
            cat, gt_j = categorize(M, k, match)
            cx0, cy0, side = detection_patch(p)
            x0, y0 = float(p["bbox"][0]), float(p["bbox"][1])
            prior = paste_mask(decode_seg(p), x0, y0, cx0, cy0, side)
            pa = int(prior.sum())
            r = {"wsi_id": wid, "source": g["source"], "split": args.split, "pred_idx": k,
                 "category": cat, "score": float(p["score"]),
                 "best_iou": float(M[k].max()) if M.shape[1] else 0.0,
                 "prior_area": pa, "crop_side": side,
                 "prior_on_gt": np.nan, "gt_in_view_frac": np.nan, "gt_crop_cover": np.nan, "dice": np.nan}
            if gt_j >= 0 and pa > 0:
                # rasterise the GT once in a window covering both the patch and the GT box,
                # so the in-view part and the whole instance come from the same raster
                inst = g["inst"][gt_j]
                gx0, gy0, gx1, gy1 = inst["bbox"]
                ux0 = min(cx0, int(np.floor(gx0)) - 1); uy0 = min(cy0, int(np.floor(gy0)) - 1)
                ux1 = max(cx0 + side, int(np.ceil(gx1)) + 1)
                uy1 = max(cy0 + side, int(np.ceil(gy1)) + 1)
                gm_all = poly_mask(inst, ux0, uy0, int(ux1 - ux0), int(uy1 - uy0))
                gf = int(gm_all.sum())
                sx0, sy0 = cx0 - ux0, cy0 - uy0
                gm_view = gm_all[sy0:sy0 + side, sx0:sx0 + side]
                ga_view = int(gm_view.sum())
                inter = int((prior.astype(bool) & gm_view.astype(bool)).sum())
                r["prior_on_gt"] = inter / max(pa, 1)
                r["gt_in_view_frac"] = ga_view / max(gf, 1)
                r["gt_crop_cover"] = ga_view / float(side * side)
                r["dice"] = 2.0 * inter / max(pa + ga_view, 1)
            rows.append(r)
        if i % 20 == 0 or i == len(gt):
            print(f"[{i}/{len(gt)}] {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows\n{df.category.value_counts().to_string()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

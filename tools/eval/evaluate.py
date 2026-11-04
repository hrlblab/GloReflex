#!/usr/bin/env python
"""Score one or more prediction directories against the WSI-level ground truth.

    python tools/eval/evaluate.py --gt-json data/test.json --poly-root data/polygons \
        --preds ours=outputs/mask_rcnn/corrected/wsi_preds baseline=outputs/mask_rcnn/stitched/wsi_preds \
        --out results/mask_rcnn.csv

Reports recall / precision / F1 at score >= 0.3, mask and box AP (0.50:0.95) and the
mean Dice of matched instances, per source and pooled.
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from gloreflex.metrics import POOLED, load_gt, score, summarize

COLS = ["recall", "precision", "f1", "mask_mAP", "bbox_mAP", "dice_mean", "dice_median",
        "tp", "fp", "fn", "n_gt"]
GT, SRC, THR = None, None, None


def _init(gt, src, thr):
    global GT, SRC, THR
    GT, SRC, THR = gt, src, thr


def _score(item):
    name, d = item
    rows = summarize(score(str(d), GT, SRC, THR))
    return [{"name": name, "source": s, **{c: r[c] for c in COLS}} for s, r in rows.items()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--poly-root", required=True)
    ap.add_argument("--preds", nargs="+", required=True, help="NAME=DIR pairs (or plain DIRs)")
    ap.add_argument("--sources", default=None, help="comma-separated source_dataset filter")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="CSV with one row per (name, source)")
    a = ap.parse_args()

    gt = load_gt(a.gt_json, a.poly_root)
    src = a.sources.split(",") if a.sources else sorted({g["source"] for g in gt.values()})
    items = [tuple(p.split("=", 1)) if "=" in p else (Path(p).parent.name, p) for p in a.preds]
    with Pool(min(a.workers, len(items)), initializer=_init, initargs=(gt, src, a.score_thr)) as pool:
        df = pd.DataFrame([r for rows in pool.map(_score, items, chunksize=1) for r in rows])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
    show = df[df.source == POOLED][["name", "recall", "precision", "f1", "mask_mAP", "dice_mean"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}" if not np.isnan(x) else "nan"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Baseline: detector output + cross-tile stitching + geometric filter + score threshold + mask NMS.

1. Predictions with score >= 0.3 whose masks overlap by at least half of the
   smaller mask are merged (union of masks, maximum score); see
   :mod:`gloreflex.stitch`.
2. The geometric filter (aspect <= 3.0, size >= 50 µm) is applied to the
   stitched masks.
3. The score threshold ``--score-thr`` is applied.
4. The same final mask NMS as GloReflex (IoU 0.5, by score).

    python tools/eval/stitch_baseline.py --preds-dir outputs/<det>/test/wsi_preds \
        --feats outputs/<det>/feats_test/det_features.csv \
        --score-thr 0.5 --out outputs/<det>/baseline_stitched
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from gloreflex.geometry import mask_aspect
from gloreflex.metrics import decode_seg
from gloreflex.nms import mask_nms
from gloreflex.stitch import STITCH_T, stitch

GEOM = (3.0, 50.0)


def one(args):
    f, mpp, thr, out, T, nms_iou = args
    j = json.load(open(f))
    st = stitch(j["preds"], T)
    keep = []
    for p in st:
        m = decode_seg(p)
        a = mask_aspect(m); s = 2.0 * np.sqrt(max(int(m.sum()), 1) / np.pi) * mpp
        if a == a and a <= GEOM[0] and s >= GEOM[1] and p["score"] >= thr:
            keep.append(p)
    if nms_iou > 0:
        keep = mask_nms(keep, nms_iou)
    for name, pr in (("stitched", st), ("baseline", keep)):
        d = Path(out) / name / "wsi_preds"; d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{f.name}.tmp"; json.dump({**j, "preds": pr}, open(tmp, "w")); os.replace(tmp, d / f.name)
    return len(j["preds"]), len(keep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--feats", required=True, help="det_features.csv (provides the slide mpp)")
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--stitch-t", type=float, default=STITCH_T)
    ap.add_argument("--nms-iou", type=float, default=0.5, help="final mask NMS; 0 disables it")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True, help="writes <out>/stitched and <out>/baseline")
    a = ap.parse_args()
    thr = a.score_thr
    mpp = pd.read_csv(a.feats).groupby("wsi_id").mpp.first()
    tasks = [(f, float(mpp[f.stem]), thr, a.out, a.stitch_t, a.nms_iou)
             for f in sorted(Path(a.preds_dir).glob("*.json")) if f.stem in mpp.index]
    with Pool(a.workers) as p:
        res = np.array(p.map(one, tasks, chunksize=1))
    print(f"{len(tasks)} WSIs: {res[:, 0].sum()} predictions -> {res[:, 1].sum()} after stitching, "
          f"geometry, score >= {thr} and NMS -> {a.out}/baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""RLE instance COCO -> polygon COCO (MaskDINO's LSJ mapper reads polygons only).

Each RLE mask becomes its external contours (absolute tile coordinates); bbox and
area are recomputed from the mask. Masks without a valid contour are dropped.

    python tools/data/coco_rle_to_polygon.py --in data/instseg_coco/train.json --out data/instseg_coco_poly/train.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

MIN_PTS, MIN_AREA = 3, 4


def rle_to_polys(seg):
    rle = dict(seg)
    if isinstance(rle["counts"], str):
        rle = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
    m = mask_util.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    if int(m.sum()) < MIN_AREA:
        return None
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [c.reshape(-1, 2).reshape(-1).astype(np.float64).tolist() for c in cnts
             if cv2.contourArea(c) >= MIN_AREA and len(c.reshape(-1, 2)) >= MIN_PTS]
    if not polys:
        return None
    ys, xs = np.where(m)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return polys, [x0, y0, x1 - x0, y1 - y0], float(int(m.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    coco = json.load(open(a.inp))
    anns = []
    for ann in coco["annotations"]:
        r = rle_to_polys(ann["segmentation"])
        if r is not None:
            anns.append({**ann, "segmentation": r[0], "bbox": r[1], "area": r[2], "iscrowd": 0})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({**{k: v for k, v in coco.items() if k != "annotations"}, "annotations": anns}, open(a.out, "w"))
    print(f"{len(coco['annotations'])} -> {len(anns)} annotations -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

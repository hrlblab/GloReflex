#!/usr/bin/env python
"""Tile training / validation WSIs into a single-class instance-segmentation COCO.

Recipe used for all detectors: level-0 tiles of 1024 x 1024 with stride 768;
instances whose box clipped to the tile is thinner than 8 px are dropped; tiles
without instances are kept with probability 0.1 and only if at least 10% of their
pixels are not near-white. Masks are rasterised from the polygons
and stored as tile-local RLE.

Input: a WSI-level JSON (see README, "Data format") and the polygon directory.
Output::

    <out>/images/<wsi_id>/tile_<x0>_<y0>.png
    <out>/instseg_coco/<split>.json          (file_name relative to <out>)

    python tools/data/build_instseg_tiles.py --gt-json data/train_wsi.json --poly-root data/polygons \
        --split train --out data
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

from gloreflex.metrics import poly_to_mask
from gloreflex.wsi import not_white_frac, open_slide, read_region_rgb

TILE, STRIDE = 1024, 768
EMPTY_KEEP_PROB = 0.1
TISSUE_GATE = 0.10
MIN_BOX = 8
SEED = 20260610


def grid_starts(n: int, tile: int, stride: int) -> list[int]:
    last = max(0, n - tile)
    xs = list(range(0, last + 1, stride))
    if n > tile and xs[-1] != last:
        xs.append(last)
    return xs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--poly-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default=None, help="comma-separated source_dataset filter")
    args = ap.parse_args()

    out = Path(args.out)
    tj = json.load(open(args.gt_json))
    imgs = tj["images"]
    if args.sources:
        imgs = [im for im in imgs if im["source_dataset"] in set(args.sources.split(","))]
    anns_by_img = {}
    for a in tj["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    rng = random.Random(SEED)
    images, annotations = [], []
    for wi, im in enumerate(imgs, 1):
        wid = im["wsi_id"]
        insts = []
        for a in anns_by_img.get(im["id"], []):
            pf = Path(args.poly_root) / wid / f"{a['wsi_instance_id']}.json"
            if pf.exists():
                x, y, w, h = a["bbox"]
                insts.append((x, y, x + w, y + h, a["wsi_instance_id"], json.load(open(pf))))
        slide = open_slide(im["file_name"])
        W, H = slide.dimensions
        (out / "images" / wid).mkdir(parents=True, exist_ok=True)
        for ty0 in grid_starts(H, TILE, STRIDE):
            for tx0 in grid_starts(W, TILE, STRIDE):
                tx1, ty1 = tx0 + TILE, ty0 + TILE
                in_tile = []
                for x0, y0, x1, y1, iid, pg in insts:
                    if x1 <= tx0 or x0 >= tx1 or y1 <= ty0 or y0 >= ty1:
                        continue
                    if min(int(x1), tx1) - max(int(x0), tx0) < MIN_BOX or min(int(y1), ty1) - max(int(y0), ty0) < MIN_BOX:
                        continue
                    in_tile.append((iid, pg))
                if not in_tile and rng.random() >= EMPTY_KEEP_PROB:
                    continue
                region = read_region_rgb(slide, tx0, ty0, TILE, TILE)
                if not in_tile and not_white_frac(region) < TISSUE_GATE:
                    continue
                fname = f"images/{wid}/tile_{tx0}_{ty0}.png"
                Image.fromarray(region).save(out / fname)
                img_id = len(images) + 1
                images.append({"id": img_id, "file_name": fname, "width": TILE, "height": TILE, "wsi_id": wid,
                               "source_dataset": im["source_dataset"], "tile_x0": tx0, "tile_y0": ty0})
                for iid, pg in in_tile:
                    m = poly_to_mask(pg.get("exterior"), pg.get("holes"), tx0, ty0, TILE, TILE)
                    if int(m.sum()) == 0:
                        continue
                    rle = mask_util.encode(np.asfortranarray(m))
                    rle["counts"] = rle["counts"].decode("ascii")
                    ys, xs = np.where(m)
                    bx0, by0 = int(xs.min()), int(ys.min())
                    annotations.append({"id": len(annotations) + 1, "image_id": img_id, "category_id": 1,
                                        "bbox": [bx0, by0, int(xs.max() - bx0 + 1), int(ys.max() - by0 + 1)],
                                        "area": int(m.sum()), "iscrowd": 0, "segmentation": rle,
                                        "wsi_instance_id": iid})
        slide.close()
        print(f"[{wi}/{len(imgs)}] {wid}: {len(images)} tiles / {len(annotations)} instances so far", flush=True)
    (out / "instseg_coco").mkdir(parents=True, exist_ok=True)
    json.dump({"images": images, "annotations": annotations,
               "categories": [{"id": 1, "name": "glomerulus"}]},
              open(out / "instseg_coco" / f"{args.split}.json", "w"))
    print(f"wrote {out}/instseg_coco/{args.split}.json: {len(images)} tiles, {len(annotations)} instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

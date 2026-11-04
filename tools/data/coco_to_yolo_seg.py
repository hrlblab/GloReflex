#!/usr/bin/env python
"""Instance-segmentation COCO (tile-local RLE) -> ultralytics YOLO-seg layout.

Each instance becomes the largest external contour of its mask, simplified with
``approxPolyDP`` (epsilon = 0.4% of the perimeter) and normalised by the tile size.
Images are symlinked.

    <out>/images/{train,val}/<wsi_id>/<tile>.png -> <img-root>/<file_name>
    <out>/labels/{train,val}/<wsi_id>/<tile>.txt   ("0 x1 y1 ... xn yn")
    <out>/dataset.yaml

    python tools/data/coco_to_yolo_seg.py --coco-root data/instseg_coco --img-root data --out data/instseg_yolo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

MIN_CONTOUR_PTS = 3
MIN_AREA = 4


def rle_to_polygon(seg, W, H, eps_frac=0.004):
    rle = dict(seg)
    if isinstance(rle["counts"], str):
        rle = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
    m = mask_util.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    if int(m.sum()) < MIN_AREA:
        return None
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None
    approx = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True)
    if len(approx) < MIN_CONTOUR_PTS:
        approx = c
    pts = approx.reshape(-1, 2).astype(np.float64)
    pts[:, 0] = np.clip(pts[:, 0] / W, 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / H, 0.0, 1.0)
    return pts.reshape(-1).tolist()


def convert_split(coco_json, img_root, out_dir, split):
    coco = json.load(open(coco_json))
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_to_yolo = {c["id"]: i for i, c in enumerate(cats)}
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    n_img = n_ann = 0
    for im in coco["images"]:
        rel = im["file_name"]
        wsi = im.get("wsi_id") or Path(rel).parts[-2]
        tile = Path(rel).name
        (out_dir / "images" / split / wsi).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split / wsi).mkdir(parents=True, exist_ok=True)
        link = out_dir / "images" / split / wsi / tile
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to((img_root / rel).resolve())
        lines = []
        for a in anns_by_img.get(im["id"], []):
            cls = cat_to_yolo.get(a["category_id"])
            poly = rle_to_polygon(a["segmentation"], im["width"], im["height"]) if cls is not None else None
            if poly is not None:
                lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in poly)); n_ann += 1
        (out_dir / "labels" / split / wsi / (tile.rsplit(".", 1)[0] + ".txt")).write_text("\n".join(lines))
        n_img += 1
    return n_img, n_ann, [c["name"] for c in cats]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", required=True, help="directory with train.json / val.json")
    ap.add_argument("--img-root", required=True, help="directory the COCO file_name is relative to")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    coco_root, img_root, out = Path(args.coco_root), Path(args.img_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = None
    for split in ("train", "val"):
        ni, na, names = convert_split(coco_root / f"{split}.json", img_root, out, split)
        print(f"{split}: {ni} images / {na} polygons")
    (out / "dataset.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
                                      f"nc: {len(names)}\nnames:\n" +
                                      "\n".join(f"  {i}: {n}" for i, n in enumerate(names)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

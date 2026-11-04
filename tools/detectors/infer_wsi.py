#!/usr/bin/env python
"""Tiled WSI inference for a trained instance-segmentation detector.

Each WSI is cut into 1024 x 1024 tiles with stride 768 (256-px overlap). The
detector's stock inference runs on every tile that passes the tissue gate, every
predicted instance mask is lifted to level-0 coordinates, and duplicates in the
tile overlaps are merged by box NMS whose kept box absorbs (mask union) the boxes
it suppresses. The output is one JSON per WSI:

    <out>/wsi_preds/<wsi_id>.json =
      {"wsi_id", "source_dataset", "width", "height",
       "preds": [{"bbox": [x0, y0, x1, y1], "score": float,
                  "seg": {"size": [h, w], "counts": str}}]}     # sorted by score

The box of a detection is the tight box of its mask. ``--conf-min`` is kept low so
that AP can be computed; the operating threshold is applied by the scorer.

Frameworks: ``mmdet`` (Mask R-CNN, SOLOv2, Mask2Former), ``yolo`` (YOLO11-seg),
``maskdino`` (MaskDINO / detectron2; set ``MASKDINO_ROOT`` to the MaskDINO repo).

    python tools/detectors/infer_wsi.py --framework mmdet \
        --config tools/detectors/configs/mask_rcnn_glo.py --weights work_dirs/mask_rcnn/epoch_12.pth \
        --gt-json data/test.json --out outputs/mask_rcnn/test --device 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

from gloreflex.wsi import TileReader, not_white_frac


def tile_offsets(W: int, H: int, tile: int, stride: int):
    last_x, last_y = max(0, W - tile), max(0, H - tile)
    xs = list(range(0, last_x + 1, stride))
    if xs and xs[-1] != last_x and W > tile:
        xs.append(last_x)
    ys = list(range(0, last_y + 1, stride))
    if ys and ys[-1] != last_y and H > tile:
        ys.append(last_y)
    return (xs or [0]), (ys or [0])


def mask_to_pred(mask_tile: np.ndarray, score: float, tx0: int, ty0: int):
    """Tile-local boolean mask -> level-0 prediction with a tight box (None if empty)."""
    ys, xs = np.where(mask_tile)
    if xs.size == 0:
        return None
    lx0, lx1 = int(xs.min()), int(xs.max()) + 1
    ly0, ly1 = int(ys.min()), int(ys.max()) + 1
    crop = mask_tile[ly0:ly1, lx0:lx1].astype(np.uint8)
    ox, oy = lx0 + tx0, ly0 + ty0
    return {"bbox": [ox, oy, ox + crop.shape[1], oy + crop.shape[0]],
            "score": float(score), "mask": crop, "ox": ox, "oy": oy}


def _union_two(i, j):
    nx0 = min(i["bbox"][0], j["bbox"][0]); ny0 = min(i["bbox"][1], j["bbox"][1])
    nx1 = max(i["bbox"][2], j["bbox"][2]); ny1 = max(i["bbox"][3], j["bbox"][3])
    canvas = np.zeros((ny1 - ny0, nx1 - nx0), np.uint8)
    for p in (i, j):
        h, w = p["mask"].shape
        ry, rx = p["oy"] - ny0, p["ox"] - nx0
        canvas[ry:ry + h, rx:rx + w] |= p["mask"]
    return mask_to_pred(canvas.astype(bool), max(i["score"], j["score"]), nx0, ny0)


def mask_nms_union(preds, iou_thr: float = 0.5):
    """Greedy box NMS in descending score; each kept prediction absorbs the masks of
    the predictions it suppresses."""
    n = len(preds)
    if n == 0:
        return preds
    boxes = np.array([p["bbox"] for p in preds], dtype=np.float64)
    scores = np.array([p["score"] for p in preds], dtype=np.float64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    done = np.zeros(n, dtype=bool)
    groups = []
    for i in order:
        if done[i]:
            continue
        xx1 = np.maximum(x1[i], x1); yy1 = np.maximum(y1[i], y1)
        xx2 = np.minimum(x2[i], x2); yy2 = np.minimum(y2[i], y2)
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas - inter + 1e-9)
        members = np.where((iou >= iou_thr) & (~done))[0]
        members = members[members != i]
        done[members] = True
        done[i] = True
        groups.append((int(i), members.tolist()))
    out = []
    for ki, members in groups:
        cur = preds[ki]
        for m in members:
            cur = _union_two(cur, preds[m])
        out.append(cur)
    return out


def iterate_tiles(svs_path, tile, stride, tissue_gate, max_tiles=None):
    """Yield ``(tx0, ty0, rgb_tile)`` for the tiles that pass the tissue gate."""
    reader = TileReader(svs_path)
    W, H = reader.dimensions
    xs, ys = tile_offsets(W, H, tile, stride)
    n_done = 0
    try:
        for ty0 in ys:
            for tx0 in xs:
                region = reader.read_rgb(tx0, ty0, tile, tile)
                n_done += 1
                if not (tissue_gate > 0 and not_white_frac(region) < tissue_gate):
                    yield tx0, ty0, region
                if max_tiles and n_done >= max_tiles:
                    return
    finally:
        reader.close()


def predict_mmdet(model, region, conf_min):
    from mmdet.apis import inference_detector
    inst = inference_detector(model, region[:, :, ::-1].copy()).pred_instances   # mmdet expects BGR
    if len(inst) == 0 or "masks" not in inst:
        return []
    scores = inst.scores.cpu().numpy()
    sel = scores >= conf_min
    if not sel.any():
        return []
    masks = inst.masks.cpu().numpy()
    return list(zip(masks[sel], scores[sel]))


def predict_yolo(model, region, conf_min, device, tile):
    import cv2
    r = model.predict(region[:, :, ::-1], conf=conf_min, device=device, retina_masks=True, verbose=False)[0]
    if r.masks is None or len(r.masks) == 0:
        return []
    out = []
    for m, sc in zip(r.masks.data.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        mb = m > 0.5
        if mb.shape[0] != tile or mb.shape[1] != tile:
            mb = cv2.resize(m.astype(np.float32), (tile, tile)) > 0.5
        out.append((mb, sc))
    return out


def predict_maskdino(predictor, region, conf_min):
    inst = predictor(region[:, :, ::-1])["instances"].to("cpu")       # BGR in
    if not (len(inst) and inst.has("pred_masks")):
        return []
    masks, scores = inst.pred_masks.numpy(), inst.scores.numpy()
    sel = scores >= conf_min
    return list(zip(masks[sel], scores[sel]))


def preds_to_json(preds, wsi_id, source, W, H):
    out = []
    for p in preds:
        rle = mask_util.encode(np.asfortranarray(p["mask"]))
        out.append({"bbox": [float(v) for v in p["bbox"]], "score": float(p["score"]),
                    "seg": {"size": [int(rle["size"][0]), int(rle["size"][1])],
                            "counts": rle["counts"].decode("ascii")}})
    return {"wsi_id": wsi_id, "source_dataset": source, "width": int(W), "height": int(H), "preds": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--framework", default="mmdet", choices=["mmdet", "yolo", "maskdino"])
    ap.add_argument("--config", default=None, help="mmdet .py or MaskDINO .yaml config (not used for yolo)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gt-json", required=True, help="WSI-level JSON listing the slides to process")
    ap.add_argument("--out", required=True, help="predictions are written to <out>/wsi_preds/")
    ap.add_argument("--sources", default=None, help="comma-separated source_dataset filter")
    ap.add_argument("--wsi-ids", nargs="*", default=None)
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=768)
    ap.add_argument("--conf-min", type=float, default=0.05)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--tissue-gate", type=float, default=0.10)
    ap.add_argument("--max-tiles-per-wsi", type=int, default=None)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    preds_dir = Path(args.out) / "wsi_preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    imgs = json.load(open(args.gt_json))["images"]
    if args.sources:
        imgs = [im for im in imgs if im["source_dataset"] in set(args.sources.split(","))]
    if args.wsi_ids:
        imgs = [im for im in imgs if im["wsi_id"] in set(args.wsi_ids)]
    if not imgs:
        print("no WSI selected", file=sys.stderr)
        return 1
    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device

    if args.framework == "mmdet":
        from mmengine.config import Config
        from mmengine.utils import import_modules_from_strings
        from mmdet.apis import init_detector
        ci = Config.fromfile(args.config).get("custom_imports")
        if ci:
            import_modules_from_strings(**ci)
        model = init_detector(args.config, args.weights, device=device)
        predict = lambda region: predict_mmdet(model, region, args.conf_min)            # noqa: E731
    elif args.framework == "yolo":
        from ultralytics import YOLO
        model = YOLO(args.weights)
        predict = lambda region: predict_yolo(model, region, args.conf_min, args.device,  # noqa: E731
                                              args.tile_size)
    else:
        if os.environ.get("MASKDINO_ROOT"):
            sys.path.insert(0, os.environ["MASKDINO_ROOT"])
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.projects.deeplab import add_deeplab_config
        from maskdino import add_maskdino_config
        cfg = get_cfg(); add_deeplab_config(cfg); add_maskdino_config(cfg)
        cfg.merge_from_file(args.config)
        cfg.MODEL.WEIGHTS = args.weights
        cfg.MODEL.DEVICE = device
        cfg.freeze()
        predictor = DefaultPredictor(cfg)
        predict = lambda region: predict_maskdino(predictor, region, args.conf_min)     # noqa: E731

    t0 = time.time()
    for k, im in enumerate(imgs, 1):
        wid = im["wsi_id"]
        out_json = preds_dir / f"{wid}.json"
        if out_json.exists():
            continue
        t1 = time.time()
        raw = []
        for tx0, ty0, region in iterate_tiles(im["file_name"], args.tile_size, args.stride,
                                              args.tissue_gate, args.max_tiles_per_wsi):
            for m, sc in predict(region):
                p = mask_to_pred(m, float(sc), tx0, ty0)
                if p is not None:
                    raw.append(p)
        merged = mask_nms_union(raw, args.nms_iou)
        out_json.write_text(json.dumps(preds_to_json(merged, wid, im["source_dataset"], im["width"], im["height"])))
        print(f"[{k}/{len(imgs)}] {wid}: raw {len(raw)} -> merged {len(merged)} ({time.time() - t1:.0f}s)",
              flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {preds_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

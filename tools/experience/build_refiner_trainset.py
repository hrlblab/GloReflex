#!/usr/bin/env python
"""Materialise the selected training detections as (patch, detector mask, GT) files.

* patch  -- square window of side 1.5 x the longest side of the **detection** box,
            resampled to 1024 x 1024 (identical to what GloReflex sees at deployment);
* prior  -- the detector's own mask in that window;
* gt     -- the matched GT polygon rasterised into the same window (only the part
            inside the window for near misses).

Writes ``<out>/{crops,prior,gt}/`` and ``<out>/manifest.csv``.

    python tools/experience/build_refiner_trainset.py --preds-dir <train_preds03> \
        --gt-json data/train_wsi.json --poly-root data/polygons --rows <dir>/rows_train.csv \
        --out <data_dir>/train
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from gloreflex.geometry import (CROP, bbox_iou_mat, categorize, detection_patch, dice_np,
                                greedy_match, place_prior, poly_in_window)
from gloreflex.metrics import decode_seg, load_gt, load_preds
from gloreflex.wsi import open_slide, read_region_rgb, resolve_mpp

WHITE_THR = 200


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--poly-root", required=True)
    ap.add_argument("--rows", required=True,
                    help="CSV of the selected training detections with columns wsi_id, pred_idx, score, "
                         "use_refiner, quality_head_only, retrieval_label, zone, aspect, size_um")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("crops", "prior", "gt"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    gt = load_gt(args.gt_json, args.poly_root)
    meta = {im["wsi_id"]: im for im in json.load(open(args.gt_json))["images"]}
    sel = pd.read_csv(args.rows)
    want = {(r.wsi_id, int(r.pred_idx)): r for r in sel.itertuples(index=False)}
    gt = {w: g for w, g in gt.items() if w in set(sel.wsi_id)}

    rows, t0 = [], time.time()
    for i, (wid, g) in enumerate(sorted(gt.items()), 1):
        preds = [p for p in load_preds(args.preds_dir, wid) if p["score"] >= args.score_thr]
        if not preds:
            continue
        preds.sort(key=lambda p: -p["score"])
        M = bbox_iou_mat(np.array([p["bbox"] for p in preds], float),
                         np.array([inst["bbox"] for inst in g["inst"]], float))
        match = greedy_match(M)
        im = meta[wid]
        mpp, mpp_src = resolve_mpp(im["file_name"], im.get("mpp_x"))
        slide = open_slide(im["file_name"])
        for k, p in enumerate(preds):
            if (wid, k) not in want:
                continue
            r_sel = want[(wid, k)]
            assert abs(float(p["score"]) - float(r_sel.score)) < 1e-9, f"pred_idx misaligned at {wid} {k}"
            cat, gt_j = categorize(M, k, match)
            x0, y0 = float(p["bbox"][0]), float(p["bbox"][1])
            cx0, cy0, side = detection_patch(p)
            crop = read_region_rgb(slide, cx0, cy0, side, side)
            if crop.shape[0] != side or crop.shape[1] != side:
                pad = np.full((side, side, 3), 255, np.uint8)
                pad[:crop.shape[0], :crop.shape[1]] = crop; crop = pad
            crop_r = cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA)
            prior = place_prior(decode_seg(p), x0, y0, cx0, cy0, side)
            gtm = poly_in_window(g["inst"][gt_j], cx0, cy0, side) if gt_j >= 0 else None

            sid = f"{wid}__d{k:05d}"
            Image.fromarray(crop_r).save(out / "crops" / f"{sid}.jpg", quality=args.jpeg_quality)
            Image.fromarray(prior * 255).save(out / "prior" / f"{sid}.png")
            if gtm is not None:
                Image.fromarray(gtm * 255).save(out / "gt" / f"{sid}.png")
            inside = prior.astype(bool)
            rows.append({
                "sample_id": sid, "wsi_id": wid, "source": g["source"], "split": args.split,
                "category": cat, "score": float(p["score"]),
                "best_iou": float(M[k].max()) if M.shape[1] else 0.0,
                "prior_dice": dice_np(prior, gtm, empty=0.0) if gtm is not None else np.nan,
                "crop_path": str(out / "crops" / f"{sid}.jpg"),
                "prior_path": str(out / "prior" / f"{sid}.png"),
                "gt_path": str(out / "gt" / f"{sid}.png") if gtm is not None else "",
                "cx0": cx0, "cy0": cy0, "side": side, "mpp": mpp, "mpp_src": mpp_src,
                "prior_area_px": int(prior.sum()),
                "white_frac": float((crop_r[inside].min(axis=1) > WHITE_THR).mean()) if inside.any() else np.nan,
                "use_refiner": bool(r_sel.use_refiner), "quality_head_only": bool(r_sel.quality_head_only),
                "retrieval_label": r_sel.retrieval_label, "zone": r_sel.zone,
                "aspect": float(r_sel.aspect), "size_um": float(r_sel.size_um),
            })
        try:
            slide.close()
        except Exception:  # noqa: BLE001
            pass
        print(f"[{i}/{len(gt)}] {wid} ({time.time() - t0:.0f}s)", flush=True)

    got = {(r["wsi_id"], int(r["sample_id"].rsplit("__d", 1)[1])) for r in rows}
    missing = set(want) - got
    assert not missing, f"{len(missing)} selected rows were not materialised, e.g. {sorted(missing)[:3]}"
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    print(f"wrote {out}/manifest.csv: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

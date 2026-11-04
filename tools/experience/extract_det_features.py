#!/usr/bin/env python
"""Per-detection mask descriptors used by the geometric filter.

For every prediction of every WSI (sorted by descending score; ``pred_idx`` is
the index in that order):

* ``aspect``     -- aspect ratio of the rotated minimum-area rectangle of the
                    largest mask component;
* ``size_um``    -- equivalent diameter ``2 * sqrt(area / pi) * mpp`` in µm
                    (NaN when the slide resolution is unknown);
* ``white_frac`` -- fraction of mask pixels with min(R, G, B) > 200.

When ground truth is available the best box IoU / mask Dice to any GT instance
are stored as well (diagnostics only; nothing downstream uses them). Ground truth
is not needed at test time: without ``--poly-root`` only the slide list is read.

    python tools/experience/extract_det_features.py --preds-dir <wsi_preds> \
        --gt-json data/test.json --poly-root data/polygons --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gloreflex.geometry import bbox_iou_mat, mask_aspect
from gloreflex.metrics import decode_seg, load_gt, load_preds, mask_iou_dice
from gloreflex.wsi import open_slide, read_region_rgb, resolve_mpp

WHITE_THR = 200
CAND_KEEP = 0.1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--poly-root", default=None, help="GT polygons (optional; only for diagnostics)")
    ap.add_argument("--sources", default=None, help="comma-separated source_dataset filter")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = {im["wsi_id"]: im for im in json.load(open(args.gt_json))["images"]}
    gt = (load_gt(args.gt_json, args.poly_root) if args.poly_root else
          {w: {"source": im["source_dataset"], "inst": []} for w, im in meta.items()})
    if args.sources:
        srcs = set(args.sources.split(","))
        gt = {w: g for w, g in gt.items() if g["source"] in srcs}

    rows, t0 = [], time.time()
    for i, (wid, g) in enumerate(sorted(gt.items()), 1):
        preds = load_preds(args.preds_dir, wid)
        if not preds:
            continue
        preds.sort(key=lambda p: -p["score"])
        pb = np.array([p["bbox"] for p in preds], float)
        gb = np.array([inst["bbox"] for inst in g["inst"]], float)
        M = bbox_iou_mat(pb, gb)
        im = meta[wid]
        mpp, mpp_src = resolve_mpp(im["file_name"], im.get("mpp_x"))
        slide = open_slide(im["file_name"])
        for k, p in enumerate(preds):
            m = decode_seg(p)
            area_px = int(m.sum())
            x0, y0, x1, y1 = [int(v) for v in p["bbox"]]
            reg = read_region_rgb(slide, x0, y0, max(1, x1 - x0), max(1, y1 - y0))
            mm = m
            if reg.shape[:2] != mm.shape:
                hh, ww = min(reg.shape[0], mm.shape[0]), min(reg.shape[1], mm.shape[1])
                reg, mm = reg[:hh, :ww], mm[:hh, :ww]
            ins = mm.astype(bool)
            white = float((reg[ins].min(axis=1) > WHITE_THR).mean()) if ins.any() else np.nan
            cands = [(int(j), float(M[k, j]), mask_iou_dice(p, g["inst"][int(j)])[1])
                     for j in np.nonzero(M[k] >= CAND_KEEP)[0]]
            rows.append({"wsi_id": wid, "source": g["source"], "pred_idx": k,
                         "score": float(p["score"]), "aspect": mask_aspect(m),
                         "size_um": (2.0 * np.sqrt(max(area_px, 1) / np.pi) * mpp if mpp is not None else np.nan),
                         "mpp": mpp, "mpp_src": mpp_src, "white_frac": white, "area_px": area_px,
                         "best_iou": max((c[1] for c in cands), default=0.0),
                         "best_dice": max((c[2] for c in cands), default=0.0)})
        try:
            slide.close()
        except Exception:  # noqa: BLE001
            pass
        print(f"[{i}/{len(gt)}] {wid}: {len(preds)} predictions ({time.time() - t0:.0f}s)", flush=True)

    pd.DataFrame(rows).to_csv(out / "det_features.csv", index=False)
    print(f"wrote {out}/det_features.csv ({len(rows)} detections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

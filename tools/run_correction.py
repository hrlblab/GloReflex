#!/usr/bin/env python
"""Run GloReflex on the frozen predictions of a detector.

For each WSI::

    detector predictions -> geometric filter (aspect <= 3.0, size >= 50 µm)
      -> for every remaining detection: state s_0 = (1.5x patch, detector mask)
         repeat for t = 0..T (T = 5):
           embed s_t, retrieve k = 15 neighbours from the experience bank,
           read out P(glomerulus) and the expected Dice change d_t
           t = 0 and P(glomerulus) < tau_g   -> discard
           d_t <= tau_s or t = T             -> accept the current mask
           otherwise                         -> refine (SAM2, mask prompt only);
               if the refined mask touches the patch border (> 1% of its contour),
               enlarge the patch on the touched sides and re-read it (at most twice)
      -> merge the accepted masks of the WSI by mask NMS (IoU > 0.5, by detector score)

No ground truth is used. ``tau_g`` (``--fp-thresh``) and ``tau_s`` (``--stop-delta``) have
fixed defaults; set them for your own data.

Every output prediction carries ``pred_idx`` (index in the detector's file),
``p_glom0`` (P(glomerulus) at t = 0), ``p_glom``, ``pred_dice``, ``n_iter`` and
``recrop_generation``. With ``--nms-iou 0`` the output before the final NMS is written.

    python tools/run_correction.py --bank <weights>/bank --head <weights>/state_head \
        --refiner <weights>/refiner --preds-dir outputs/<det>/test/wsi_preds \
        --feats outputs/<det>/feats_test/det_features.csv --gt-json data/test.json \
        --out outputs/<det>/corrected
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from gloreflex import refiner as ft
from gloreflex.bank import ExperienceBank
from gloreflex.encoder import StateHead, make_backbone, overlay_tensor
from gloreflex.geometry import CROP, detection_patch, place_prior, to_wsi_rle
from gloreflex.metrics import decode_seg
from gloreflex.nms import mask_nms
from gloreflex.recrop import MAX_RECROP, border_touch_frac, mask_stats, plan_recrop, read_crop, remap_mask, touch_sides
from gloreflex.refiner import refine
from gloreflex.wsi import open_slide

MAX_ITER = 5
FP_THRESH = 0.5          # tau_g
STOP_DELTA = 0.005       # tau_s
GEOM_ASPECT_MAX = 3.0
GEOM_SIZE_MIN_UM = 50.0
RECROP_TOUCH = 0.01


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--refiner", required=True)
    ap.add_argument("--preds-dir", required=True, help="detector predictions (<wsi_id>.json, sorted by score)")
    ap.add_argument("--feats", required=True, help="det_features.csv of the same predictions")
    ap.add_argument("--gt-json", required=True, help="WSI list (annotations are not used)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wsi-ids", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--fp-thresh", type=float, default=FP_THRESH, help="tau_g: discard threshold on P(glomerulus)")
    ap.add_argument("--stop-delta", type=float, default=STOP_DELTA, help="tau_s: stop threshold on the expected Dice change")
    ap.add_argument("--nms-iou", type=float, default=0.5, help="final mask NMS IoU; 0 disables it")
    args = ap.parse_args()

    out = Path(args.out); (out / "wsi_preds").mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    bank = ExperienceBank(args.bank, dev)
    FP_T, STOP = args.fp_thresh, args.stop_delta
    TOUCH, AMAX, SMIN = RECROP_TOUCH, GEOM_ASPECT_MAX, GEOM_SIZE_MIN_UM
    print(f"tau_g {FP_T} | tau_s {STOP} | re-crop touch > {TOUCH} | geometry aspect <= {AMAX}, "
          f"size >= {SMIN} um | NMS IoU {args.nms_iou}", flush=True)
    json.dump({"fp_thresh": FP_T, "stop_delta": STOP, "nms_iou": args.nms_iou, "bank": args.bank,
               "head": args.head, "refiner": args.refiner}, open(out / "config.json", "w"), indent=1)

    ext, spec = make_backbone("dinov3"); ext = ext.to(dev).eval()
    hd = StateHead(spec.feature_dim).to(dev)
    hd.load_state_dict(torch.load(f"{args.head}/head.pt", map_location=dev)); hd.eval()
    sam = ft.build_refiner(args.refiner, dev); sam.eval()
    tf = ft.sam_transforms()
    amp = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    def embed(crops, masks):
        x = np.stack([overlay_tensor(c, (m * 255).astype(np.uint8), spec) for c, m in zip(crops, masks)])
        with torch.no_grad(), amp:
            z, *_ = hd(ext(torch.from_numpy(x).to(dev)).float())
        return z

    feats = pd.read_csv(args.feats)
    meta = {im["wsi_id"]: im for im in json.load(open(args.gt_json))["images"]}
    ids = args.wsi_ids or sorted(meta)
    stats_all, t0 = [], time.time()
    for wi, wid in enumerate(ids):
        op = out / "wsi_preds" / f"{wid}.json"
        if op.exists():
            continue
        im = meta[wid]
        pr = json.load(open(f"{args.preds_dir}/{wid}.json"))
        pr = pr["preds"] if isinstance(pr, dict) else pr
        fw = feats[feats.wsi_id == wid].set_index("pred_idx")
        keep = [k for k in range(len(pr)) if k in fw.index
                and fw.loc[k, "aspect"] <= AMAX and fw.loc[k, "size_um"] >= SMIN]
        slide = open_slide(im["file_name"])
        W, H = im["width"], im["height"]
        kept, disc = [], 0
        for s in range(0, len(keep), args.batch_size):
            ks = keep[s:s + args.batch_size]
            geo, crops, priors, scores, ioup, niter, p0 = [], [], [], [], [], [], []
            for k in ks:
                p = pr[k]; x0, y0 = float(p["bbox"][0]), float(p["bbox"][1])
                cx0, cy0, side = detection_patch(p)
                geo.append([cx0, cy0, side, 0])
                crops.append(read_crop(slide, cx0, cy0, side))
                priors.append(place_prior(decode_seg(p), x0, y0, cx0, cy0, side))
                scores.append(float(p["score"]))
                ioup.append(np.nan); niter.append(0); p0.append(np.nan)
            alive = list(range(len(ks)))
            for it in range(MAX_ITER + 1):
                if not alive:
                    break
                rows = []
                for j in alive:
                    mpp = float(fw.loc[ks[j], "mpp"])
                    st = mask_stats(priors[j], crops[j], mpp, geo[j][2])
                    rows.append({"prior_border_touch_frac": st["border_touch_frac"],
                                 "prior_aspect": st["aspect"], "prior_white_frac": st["white_frac"],
                                 "prior_size_um": st["size_um"], "mpp_eff": mpp * geo[j][2] / CROP,
                                 "det_score": scores[j], "n_clicks_so_far": 0,
                                 "iou_pred_prior": ioup[j], "iter": it, "recrop_generation": geo[j][3],
                                 "log_area": float(np.log1p(st["area_px"]))})
                pred_dice, pred_delta, p_glom = bank.query(embed([crops[j] for j in alive],
                                                                 [priors[j] for j in alive]),
                                                           bank.standardize(rows))
                nxt = []
                for a, j in enumerate(alive):
                    if it == 0:
                        p0[j] = float(p_glom[a])
                        if p_glom[a] < FP_T:                         # discard (first examination only)
                            priors[j] = None; disc += 1; continue
                    if it == MAX_ITER or pred_delta[a] <= STOP:      # accept
                        kept.append((priors[j].copy(), list(geo[j]), scores[j], float(pred_dice[a]),
                                     float(p_glom[a]), niter[j], int(ks[j]), p0[j]))
                        continue
                    nxt.append(j)                                    # refine
                if not nxt:
                    alive = []; continue
                imgs = torch.stack([tf(Image.fromarray(crops[j])) for j in nxt]).to(dev)
                nm, ni_ = refine(sam, imgs, [priors[j] for j in nxt], None, None, dev, amp)
                for t_, j in enumerate(nxt):
                    new = nm[t_]; ioup[j] = float(ni_[t_]); niter[j] += 1
                    cx0, cy0, side, gen = geo[j]
                    tfrac = border_touch_frac(new)
                    if tfrac > TOUCH and gen < MAX_RECROP:                # re-crop
                        pl = plan_recrop(tfrac, touch_sides(new), cx0, cy0, side, 0, 0)
                        if pl is not None:
                            nx0, ny0, ns, _ = pl
                            crops[j] = read_crop(slide, nx0, ny0, ns)
                            priors[j] = remap_mask(new, cx0, cy0, side, nx0, ny0, ns)
                            geo[j] = [nx0, ny0, ns, gen + 1]
                            continue
                    priors[j] = new
                alive = nxt
        preds_out, n_empty = [], 0
        for msk, (cx0, cy0, side, gen), sc_, q, pgv, ni2, pidx, pg0 in kept:
            r = to_wsi_rle(msk, cx0, cy0, side, W, H)
            if r is None:
                n_empty += 1
                continue
            preds_out.append({**r, "score": float(sc_), "pred_dice": q, "p_glom": pgv,
                              "recrop_generation": int(gen), "n_iter": int(ni2),
                              "pred_idx": int(pidx), "p_glom0": float(pg0)})
        n_pre = len(preds_out)
        if args.nms_iou > 0:
            preds_out = mask_nms(preds_out, args.nms_iou)
        json.dump({"wsi_id": wid, "preds": preds_out}, open(op, "w"))
        stats_all.append({"wsi_id": wid, "raw": len(pr), "geom_keep": len(keep), "discarded": disc,
                          "empty": n_empty, "accepted": n_pre, "after_nms": len(preds_out)})
        print(f"[{wi + 1}/{len(ids)}] {wid}: {len(pr)} -> geometry {len(keep)} -> discarded {disc} -> "
              f"accepted {n_pre} -> NMS {len(preds_out)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if stats_all:
        tag = os.environ.get("RUN_TAG", os.environ.get("CUDA_VISIBLE_DEVICES", "0")).replace(",", "_")
        pd.DataFrame(stats_all).to_csv(out / f"run_stats_{tag}.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

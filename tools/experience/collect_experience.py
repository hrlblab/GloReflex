#!/usr/bin/env python
"""Collect the experience bank: run the refiner on the training detections and
record every visited state.

Two branches, each with a protocol that is identical for all detections it
covers (so that no record feature can leak the label through the protocol):

* ``probe`` (all selected rows, glomeruli and non-glomerular detections alike):
  a single refinement step with one click at the mask's interior point farthest
  from its border (GT-free). Provides ``label_is_glom``.
* ``traj`` (rows with ``use_refiner``, i.e. glomeruli): a fixed number of rounds
  (5) without early stopping, so that steps without gain are recorded too. Every
  round adds one click sampled from the current error region w.r.t. the GT; after
  each round, a refined mask that touches the patch border triggers a re-crop
  (at most two). Labels use the frame-independent Dice against the complete GT
  instance (``label_dice``) and its change after the step (``label_delta_dice``),
  so states before and after a re-crop are comparable.

Records are flushed to ``<out>/shards/`` (resumable) and merged into
``<out>/experience.parquet``.

    python tools/experience/collect_experience.py --data <data_dir>/train --ckpt <weights>/refiner \
        --preds-dir <train_preds03> --gt-json data/train_wsi.json --poly-root data/polygons \
        --out <data_dir>/experience --amp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from gloreflex import refiner as ft
from gloreflex.geometry import (CROP, IOU_TP, bbox_iou_mat, dice_np, greedy_match, poly_area,
                                poly_in_window)
from gloreflex.metrics import load_gt
from gloreflex.recrop import mask_stats, plan_recrop, read_crop, remap_mask
from gloreflex.refiner import IMAGE_SIZE, refine
from gloreflex.refiner_data import RefinerDataset, sample_click
from gloreflex.wsi import open_slide


def dice_full(mask: np.ndarray, gt_view: np.ndarray, side: int, gt_area_wsi: float) -> float:
    """Dice of a patch-frame mask against the complete GT instance (level-0 pixels),
    including the part of the instance outside the patch."""
    if gt_area_wsi <= 0:
        return float("nan")
    k = (side / CROP) ** 2
    inter = float((mask.astype(bool) & gt_view.astype(bool)).sum()) * k
    m = float(mask.sum()) * k
    return float(2.0 * inter / (m + gt_area_wsi)) if (m + gt_area_wsi) > 0 else 0.0


def match_gt(preds, insts) -> dict[int, int]:
    """Same greedy matching as ``build_refiner_trainset.py``; unmatched -> best-IoU GT."""
    M = bbox_iou_mat(np.array([p["bbox"] for p in preds], float),
                     np.array([i["bbox"] for i in insts], float))
    if M.size == 0:
        return {k: -1 for k in range(len(preds))}
    match = greedy_match(M, IOU_TP)
    return {k: (match[k] if match[k] >= 0 else int(M[k].argmax())) for k in range(len(preds))}


def center_click(m: np.ndarray) -> tuple[int, int]:
    """Interior point of the mask farthest from its border (GT-free)."""
    if not m.any():
        return CROP // 2, CROP // 2
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    i = int(d.argmax())
    return int(i % CROP), int(i // CROP)


def click_record(x: int, y: int, m: np.ndarray, cx0: int, cy0: int, side: int) -> dict:
    """A click in patch, level-0 and mask-relative coordinates."""
    d = {"x": int(x), "y": int(y),
         "wsi_x": float(cx0 + x * side / CROP), "wsi_y": float(cy0 + y * side / CROP),
         "nx": np.nan, "ny": np.nan, "r_signed": np.nan, "theta": np.nan}
    if not m.any():
        return d
    ys, xs = np.nonzero(m)
    bw, bh = max(xs.max() - xs.min(), 1), max(ys.max() - ys.min(), 1)
    d["nx"] = float((x - (xs.min() + xs.max()) / 2) / bw)
    d["ny"] = float((y - (ys.min() + ys.max()) / 2) / bh)
    u = m.astype(np.uint8)
    din = cv2.distanceTransform(u, cv2.DIST_L2, 3)
    dout = cv2.distanceTransform(1 - u, cv2.DIST_L2, 3)
    r_eff = float(np.sqrt(max(int(m.sum()), 1) / np.pi))
    d["r_signed"] = float((dout[y, x] - din[y, x]) / r_eff)
    d["theta"] = float(np.arctan2(y - float(ys.mean()), x - float(xs.mean())))
    return d


def _flush(recs, shard_dir: Path) -> None:
    if recs:
        i = len(list(shard_dir.glob("*.parquet")))
        pd.DataFrame(recs).to_parquet(shard_dir / f"shard_{i:05d}.parquet", index=False)


def _merge(out: Path) -> pd.DataFrame:
    """Merge the shards (de-duplicated by record_id) and remove orphaned image files."""
    df = pd.concat([pd.read_parquet(f) for f in sorted((out / "shards").glob("*.parquet"))], ignore_index=True)
    df = df.drop_duplicates(subset=["record_id"], keep="last").reset_index(drop=True)
    ref = {Path(x).name for x in df.refined_mask_path} | {Path(x).name for x in df.prior_path}
    ref |= {Path(x).name for x in df.crop_path}
    for sub in ("masks", "crops"):
        for f in (out / sub).glob("*"):
            if f.name not in ref:
                f.unlink()
    df.to_parquet(out / "experience.parquet", index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory with manifest.csv (build_refiner_trainset.py)")
    ap.add_argument("--ckpt", required=True, help="fine-tuned refiner directory")
    ap.add_argument("--preds-dir", required=True, help="the predictions the manifest was built from")
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--poly-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=5, help="fixed number of traj rounds (no early stop)")
    ap.add_argument("--recrop-thresh", type=float, default=0.01, help="border-touch fraction that triggers a re-crop")
    ap.add_argument("--max-recrop", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--flush-every", type=int, default=2000)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda")
    out = Path(args.out)
    if not args.resume and out.exists() and any(f.is_file() for f in out.rglob("*")):
        raise SystemExit(f"output directory is not empty: {out} (remove it or pass --resume)")
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    shard_dir = out / "shards"; shard_dir.mkdir(parents=True, exist_ok=True)

    model = ft.build_refiner(args.ckpt, dev); model.eval()
    tf = ft.sam_transforms()
    amp = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.amp else torch.amp.autocast("cuda", enabled=False)

    man = pd.read_csv(Path(args.data) / "manifest.csv")
    if args.max_rows:
        man = man.head(args.max_rows)
    if args.resume:
        done = set()
        for f in sorted(shard_dir.glob("*.parquet")):
            done |= set(pd.read_parquet(f, columns=["sample_id"]).sample_id.unique())
        man = man[~man.sample_id.isin(done)]
        if man.empty:
            _merge(out); return 0
    man["has_gt"] = man.use_refiner
    meta = {r.sample_id: r for r in man.itertuples(index=False)}
    dl = DataLoader(RefinerDataset(man.to_dict("records"), tf, aug=False, seed=args.seed),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"{len(man)} rows: probe for all, traj for {int(man.use_refiner.sum())} x {args.iters} rounds", flush=True)

    # GT polygons and the detection -> GT matching (needed for the frame-independent Dice
    # and for re-rasterising the GT after a re-crop)
    gt = load_gt(args.gt_json, args.poly_root)
    wsi_file = {im["wsi_id"]: im["file_name"] for im in json.load(open(args.gt_json))["images"]}
    gtpoly, gtidx = {}, {}
    for wid in sorted(man.wsi_id.unique()):
        gtpoly[wid] = gt[wid]["inst"]
        pr = json.load(open(f"{args.preds_dir}/{wid}.json"))
        pr = pr["preds"] if isinstance(pr, dict) else pr
        for k, gj in match_gt(pr, gtpoly[wid]).items():
            gtidx[(wid, k)] = gj
    chk = man[man.use_refiner].sample(min(60, int(man.use_refiner.sum())), random_state=7)
    for r in chk.itertuples(index=False):        # the re-derived GT must equal the stored GT masks
        gj = gtidx.get((r.wsi_id, int(r.sample_id.rsplit("__d", 1)[1])), -1)
        a = poly_in_window(gtpoly[r.wsi_id][gj], int(r.cx0), int(r.cy0), int(r.side)) > 0 if gj >= 0 else None
        b = np.asarray(Image.open(r.gt_path)) > 127
        u = (a | b).sum() if a is not None else 0
        assert a is not None and (not u or (a & b).sum() / u >= 0.99), \
            "GT re-matching disagrees with the stored GT masks (wrong --preds-dir?)"

    slides = {}

    def get_slide(wid):
        if wid not in slides:
            if len(slides) >= 2:
                slides.clear()
            try:
                slides[wid] = open_slide(wsi_file[wid])
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] cannot open {wid}: {e}", flush=True); slides[wid] = None
        return slides[wid]

    def gt_index(sid):
        m = meta[sid]
        return gtidx.get((m.wsi_id, int(sid.rsplit("__d", 1)[1])), -1)

    def base(sid, kind, it, parent):
        m = meta[sid]
        return {"record_id": f"{sid}__{kind}{it}", "sample_id": sid, "wsi_id": m.wsi_id,
                "source": m.source, "traj_id": f"{sid}__{kind}", "iter": it, "parent_record_id": parent,
                "record_kind": kind, "det_score": float(m.score),
                "crop_path": m.crop_path, "cx0": int(m.cx0), "cy0": int(m.cy0),
                "side": int(m.side), "mpp": float(m.mpp), "mpp_eff": float(m.mpp) * int(m.side) / CROP,
                "recrop_generation": 0, "recrop_reason": "none", "expand_dirs": "",
                "did_recrop": False, "new_crop_box": "",
                "category": m.category, "zone": m.zone, "use_refiner": bool(m.use_refiner),
                "quality_head_only": bool(m.quality_head_only),
                "label_is_glom": "glom" if m.retrieval_label == "glom" else "non_glom"}

    rng = np.random.default_rng(args.seed)
    recs, t0, n = [], time.time(), 0
    for batch in dl:
        img = batch["image"].to(dev, non_blocking=True)
        pri = batch["prior"].numpy(); gtn = batch["gt"].numpy()
        hg = batch["has_gt"].numpy().astype(bool)
        sids = batch["sample_id"]
        crops = [np.asarray(Image.open(meta[s].crop_path).convert("RGB")) for s in sids]

        # ---- probe: one step with a GT-free centre click ----
        cc = [center_click(p) for p in pri]
        co = torch.tensor([[[x / CROP * IMAGE_SIZE, y / CROP * IMAGE_SIZE]] for x, y in cc],
                          dtype=torch.float32, device=dev)
        la = torch.ones((len(pri), 1), dtype=torch.int64, device=dev)
        pm, pi = refine(model, img, list(pri), co, la, dev, amp)
        for b, sid in enumerate(sids):
            m = meta[sid]; side = int(m.side)
            r = base(sid, "probe", 0, "")
            ps = mask_stats(pri[b], crops[b], float(m.mpp), side)
            rs = mask_stats(pm[b], crops[b], float(m.mpp), side)
            mp = f"{out}/masks/{sid}__probe0.png"
            Image.fromarray(pm[b] * 255).save(mp)
            cf = click_record(cc[b][0], cc[b][1], pri[b], int(m.cx0), int(m.cy0), side)
            r.update({"prior_path": m.prior_path, "refined_mask_path": mp,
                      "n_clicks_so_far": 0, "click_history": "[]",
                      "clicks_applied": json.dumps([{**cf, "label": 1}]),
                      "click_source": "center_probe", "refiner_mode": "mask+clicks",
                      "iou_pred_prior": np.nan, "iou_pred_refined": float(pi[b]),
                      **{f"prior_{k}": v for k, v in ps.items()},
                      **{f"refined_{k}": v for k, v in rs.items()},
                      "dice_prior": dice_np(pri[b], gtn[b]) if hg[b] else np.nan,
                      "dice_refined": dice_np(pm[b], gtn[b]) if hg[b] else np.nan,
                      "terminated": True, "termination_reason": "probe_single_step"})
            r["delta_dice"] = (r["dice_refined"] - r["dice_prior"]) if hg[b] else np.nan
            # probe records only carry the glomerulus label; Dice labels come from traj
            r.update({"label_dice": np.nan, "label_delta_dice": np.nan, "dice_full_prior": np.nan,
                      "dice_full_refined": np.nan, "delta_dice_full": np.nan, "gt_full_area_wsi": np.nan})
            recs.append(r)

        # ---- traj: fixed-length rollout with error-region clicks and re-crop ----
        idx = [b for b in range(len(pri)) if hg[b]]
        if idx:
            cur = [pri[b].copy() for b in idx]
            gtc = [gtn[b].copy() for b in idx]
            cim = [crops[b] for b in idx]
            itens = [img[b] for b in idx]
            geo = [(int(meta[sids[b]].cx0), int(meta[sids[b]].cy0), int(meta[sids[b]].side)) for b in idx]
            gen = [0] * len(idx)
            gfa = [poly_area(gtpoly[meta[sids[b]].wsi_id][gt_index(sids[b])]) if gt_index(sids[b]) >= 0 else 0.0
                   for b in idx]
            hist = [[] for _ in idx]
            iou_prev = [np.nan] * len(idx)
            stale = [False] * len(idx)
            parent = ["" for _ in idx]
            prev_mask_path = [meta[sids[b]].prior_path for b in idx]
            cur_crop_path = [meta[sids[b]].crop_path for b in idx]
            next_prior_path = [None] * len(idx)
            for it in range(args.iters):
                cs = [sample_click(cur[j], gtc[j], rng) for j in range(len(idx))]
                nmax = max(1, max(len(h) for h in hist) + 1)
                co = np.zeros((len(idx), nmax, 2), np.float32)
                la = np.full((len(idx), nmax), -1, np.int64)
                for j in range(len(idx)):
                    allc = hist[j] + ([cs[j]] if cs[j] else [])
                    for k, c in enumerate(allc[:nmax]):
                        co[j, k] = [c[0] / CROP * IMAGE_SIZE, c[1] / CROP * IMAGE_SIZE]
                        la[j, k] = c[2]
                nm, ni = refine(model, torch.stack(itens), cur, torch.from_numpy(co).to(dev),
                                torch.from_numpy(la).to(dev), dev, amp)
                for j, b in enumerate(idx):
                    sid = sids[b]; m = meta[sid]
                    cx0, cy0, side = geo[j]
                    r = base(sid, "traj", it, parent[j])
                    r.update({"cx0": cx0, "cy0": cy0, "side": side, "recrop_generation": gen[j],
                              "crop_path": cur_crop_path[j], "mpp_eff": float(m.mpp) * side / CROP})
                    ps = mask_stats(cur[j], cim[j], float(m.mpp), side)
                    rs = mask_stats(nm[j], cim[j], float(m.mpp), side)
                    mp = f"{out}/masks/{sid}__traj{it}.png"
                    Image.fromarray(nm[j] * 255).save(mp)
                    cf = click_record(cs[j][0], cs[j][1], cur[j], cx0, cy0, side) if cs[j] else None
                    hist_snap = list(hist[j])                 # this record is in the old frame
                    d0, d1 = dice_np(cur[j], gtc[j]), dice_np(nm[j], gtc[j])
                    df0 = dice_full(cur[j], gtc[j], side, gfa[j])
                    df1 = dice_full(nm[j], gtc[j], side, gfa[j])

                    did, newbox, dirs, reason = False, "", "", "none"
                    if (args.recrop_thresh > 0 and it < args.iters - 1 and gen[j] < args.max_recrop
                            and rs["border_touch_frac"] > args.recrop_thresh):
                        pl = plan_recrop(rs["border_touch_frac"], rs["touch_sides"], cx0, cy0, side)
                        sl = get_slide(m.wsi_id) if pl is not None else None
                        if pl is not None and sl is not None:
                            nx0, ny0, ns, dirs = pl
                            cim[j] = read_crop(sl, nx0, ny0, ns)
                            itens[j] = tf(Image.fromarray(cim[j])).to(dev)
                            cur[j] = remap_mask(nm[j], cx0, cy0, side, nx0, ny0, ns)
                            gj = gt_index(sid)
                            gtc[j] = (poly_in_window(gtpoly[m.wsi_id][gj], nx0, ny0, ns)
                                      if gj >= 0 else np.zeros((CROP, CROP), np.uint8))

                            def to_new(c):
                                return (int(round((cx0 + c[0] * side / CROP - nx0) / ns * CROP)),
                                        int(round((cy0 + c[1] * side / CROP - ny0) / ns * CROP)), c[2], c[3])
                            hist[j] = [to_new(c) for c in hist[j]]
                            if cs[j]:
                                cs[j] = to_new(cs[j])
                            geo[j] = (nx0, ny0, ns); gen[j] += 1
                            cpn = f"{out}/crops/{sid}__g{gen[j]}.jpg"
                            Image.fromarray(cim[j]).save(cpn, quality=92)
                            pmn = f"{out}/masks/{sid}__traj{it}_g{gen[j]}.png"
                            Image.fromarray(cur[j] * 255).save(pmn)
                            cur_crop_path[j] = cpn
                            next_prior_path[j] = pmn
                            did, newbox, reason = True, f"{nx0},{ny0},{ns}", "border_touch"

                    r.update({"prior_path": prev_mask_path[j], "refined_mask_path": mp,
                              "n_clicks_so_far": len(hist_snap),
                              "click_history": json.dumps([{"x": c[0], "y": c[1], "label": c[2], "kind": c[3]}
                                                           for c in hist_snap]),
                              "clicks_applied": json.dumps([{**cf, "label": cs[j][2], "kind": cs[j][3]}] if cf else []),
                              "click_source": cs[j][3] if cs[j] else "none",
                              "refiner_mode": "mask+clicks" if cs[j] else "mask_only",
                              "iou_pred_prior": iou_prev[j], "iou_pred_prior_stale": bool(stale[j]),
                              "iou_pred_refined": float(ni[j]),
                              **{f"prior_{k}": v for k, v in ps.items()},
                              **{f"refined_{k}": v for k, v in rs.items()},
                              "dice_prior": d0, "dice_refined": d1, "delta_dice": d1 - d0,
                              "dice_full_prior": df0, "dice_full_refined": df1,
                              "delta_dice_full": df1 - df0, "gt_full_area_wsi": gfa[j],
                              "label_dice": df0, "label_delta_dice": df1 - df0,
                              "did_recrop": did, "new_crop_box": newbox, "expand_dirs": dirs,
                              "recrop_reason": reason, "terminated": it == args.iters - 1,
                              "termination_reason": "max_iter" if it == args.iters - 1 else ""})
                    recs.append(r)
                    parent[j] = r["record_id"]; iou_prev[j] = float(ni[j]); stale[j] = did
                    prev_mask_path[j] = next_prior_path[j] or mp
                    next_prior_path[j] = None
                    if cs[j]:
                        hist[j].append(cs[j])
                    if not did:
                        cur[j] = nm[j]
        n += len(sids)
        if len(recs) >= args.flush_every:
            _flush(recs, shard_dir); recs = []
        if n % 400 < args.batch_size:
            el = time.time() - t0
            print(f"  {n}/{len(man)} rows  ETA {(len(man) - n) / max(n / el, 1e-9) / 60:.1f} min", flush=True)

    _flush(recs, shard_dir)
    df = _merge(out)
    print(f"{len(df)} records -> {out}/experience.parquet ({(time.time() - t0) / 60:.1f} min)\n"
          f"  by kind: {df.record_kind.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

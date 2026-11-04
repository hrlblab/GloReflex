#!/usr/bin/env python
"""Paired slide-level bootstrap confidence intervals.

Every prediction directory is scored WSI by WSI with the official scorer; WSIs are
then resampled with replacement, with the same resampling indices for all rows
(paired). Recall, precision, F1 and Dice use 10,000 resamples, mask AP 2,000.
A difference is significant if its 95% interval excludes zero.

    python tools/eval/bootstrap_ci.py --gt-json data/test.json --poly-root data/polygons \
        --rows ours=<dir> baseline=<dir> --pairs ours:baseline --out results/ci
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from gloreflex.metrics import IOU_THRS, POOLED, coco_ap, load_gt, score, summarize

GT, SRC, THR = None, None, None
_PER, _WIDS, _IDM = None, None, None


def _init(gt, src, thr):
    global GT, SRC, THR
    GT, SRC, THR = gt, src, thr


def per_wsi(args):
    name, d = args
    out = {}
    for wid, g in GT.items():
        acc = score(str(d), {wid: g}, SRC, THR)[g["source"]]
        out[wid] = {"tp": acc["tp"], "fp": acc["fp"], "fn": acc["fn"], "n_gt": acc["n_gt"],
                    "dice_sum": float(np.sum(acc["dice"])), "n_dice": len(acc["dice"]),
                    "mask": {t: acc["mask"][t] for t in IOU_THRS}}
    pooled = summarize(score(str(d), GT, SRC, THR))[POOLED]
    return name, out, pooled


def _boot_map(n):
    per_n = _PER[n]
    out = []
    for ii in _IDM:
        ng = sum(per_n[_WIDS[i]]["n_gt"] for i in ii)
        out.append(float(np.mean([coco_ap([r for i in ii for r in per_n[_WIDS[i]]["mask"][t]], ng, t)
                                  for t in IOU_THRS])))
    return n, np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--poly-root", required=True)
    ap.add_argument("--rows", nargs="+", required=True, help="NAME=DIR pairs")
    ap.add_argument("--pairs", nargs="*", default=[], help="A:B pairs (difference A - B)")
    ap.add_argument("--sources", default=None)
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-boot-map", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gt = load_gt(a.gt_json, a.poly_root)
    src = a.sources.split(",") if a.sources else sorted({g["source"] for g in gt.values()})
    gt = {w: g for w, g in gt.items() if g["source"] in src}
    wids = sorted(gt)
    rows = dict(r.split("=", 1) for r in a.rows)
    with Pool(a.workers, initializer=_init, initargs=(gt, src, a.score_thr)) as pool:
        res = pool.map(per_wsi, list(rows.items()), chunksize=1)
    per = {}
    for name, o, pooled in res:
        s = tuple(sum(o[w][k] for w in wids) for k in ("tp", "fp", "fn"))
        assert s == (pooled["tp"], pooled["fp"], pooled["fn"]), f"per-WSI totals disagree with the scorer ({name})"
        per[name] = o

    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, len(wids), size=(a.n_boot, len(wids)))
    M = {n: {k: np.array([per[n][w][k] for w in wids], float) for k in ("tp", "fp", "fn", "dice_sum", "n_dice")}
         for n in per}

    def stats(n, ii):
        m = M[n]; tp, fp, fn = m["tp"][ii].sum(-1), m["fp"][ii].sum(-1), m["fn"][ii].sum(-1)
        R = tp / np.maximum(tp + fn, 1); P = tp / np.maximum(tp + fp, 1)
        return {"recall": R, "precision": P, "f1": 2 * P * R / np.maximum(P + R, 1e-12),
                "dice": m["dice_sum"][ii].sum(-1) / np.maximum(m["n_dice"][ii].sum(-1), 1)}

    full = np.arange(len(wids))
    boot = {n: stats(n, idx) for n in per}
    point = {n: {k: float(np.ravel(v)[0]) for k, v in stats(n, full[None, :]).items()} for n in per}
    global _PER, _WIDS, _IDM
    _PER, _WIDS, _IDM = per, wids, np.vstack([full, idx[:a.n_boot_map]])     # row 0 = point estimate
    with Pool(a.workers) as pool:
        mb = dict(pool.map(_boot_map, list(per), chunksize=1))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ci = []
    for n in per:
        r = {"row": n, "mask_mAP": mb[n][0]}
        lo, hi = np.percentile(mb[n][1:], [2.5, 97.5]); r.update({"mask_mAP_lo": lo, "mask_mAP_hi": hi})
        for k in ("recall", "precision", "f1", "dice"):
            lo, hi = np.percentile(boot[n][k], [2.5, 97.5])
            r.update({k: point[n][k], f"{k}_lo": lo, f"{k}_hi": hi})
        ci.append(r)
    pd.DataFrame(ci).to_csv(out / "ci.csv", index=False)
    pr = []
    for p in a.pairs:
        x, y = p.split(":", 1)
        r = {"pair": f"{x} - {y}"}
        for k in ("recall", "precision", "f1", "dice"):
            d = boot[x][k] - boot[y][k]
            r.update({k: point[x][k] - point[y][k], f"{k}_lo": np.percentile(d, 2.5), f"{k}_hi": np.percentile(d, 97.5)})
        d = mb[x][1:] - mb[y][1:]
        r.update({"mask_mAP": mb[x][0] - mb[y][0], "mask_mAP_lo": np.percentile(d, 2.5),
                  "mask_mAP_hi": np.percentile(d, 97.5)})
        pr.append(r)
    pd.DataFrame(pr).to_csv(out / "paired.csv", index=False)
    K = ("recall", "precision", "f1", "mask_mAP", "dice")
    for r in pr:
        print(f"{r['pair']}: " + "  ".join(f"{k} {r[k]:+.3f} [{r[k + '_lo']:+.3f}, {r[k + '_hi']:+.3f}]" for k in K))
    return 0


if __name__ == "__main__":
    sys.exit(main())

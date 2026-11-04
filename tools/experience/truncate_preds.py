#!/usr/bin/env python
"""Keep only the score >= 0.3 prefix of each WSI's predictions (train / val slides).

Detector outputs are sorted by descending score, so the prefix keeps every
``pred_idx`` unchanged; all downstream files are aligned by ``pred_idx``. The
operating-point metrics only use predictions with score >= 0.3, so truncation
does not change them but removes most of the compute for dense detectors.

    python tools/experience/truncate_preds.py --src outputs/<det>/train/wsi_preds \
        --dst outputs/<det>/train_preds03 --gt-json data/train_wsi.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--sources", default=None, help="comma-separated source_dataset filter")
    ap.add_argument("--thr", type=float, default=0.3)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    srcs = set(args.sources.split(",")) if args.sources else None
    ids = sorted(im["wsi_id"] for im in json.load(open(args.gt_json))["images"]
                 if srcs is None or im["source_dataset"] in srcs)
    n_all = n_kept = 0
    for wid in ids:
        f = src / f"{wid}.json"
        if not f.exists():
            raise SystemExit(f"missing prediction file: {f}")
        d = json.load(open(f))
        preds = d["preds"] if isinstance(d, dict) else d
        s = [float(p["score"]) for p in preds]
        if any(s[i] < s[i + 1] for i in range(len(s) - 1)):
            raise SystemExit(f"{wid}: predictions are not sorted by score")
        n = next((i for i, v in enumerate(s) if v < args.thr), len(s))
        out = dict(d) if isinstance(d, dict) else {"wsi_id": wid}
        out["preds"] = preds[:n]
        out["truncated_from"] = len(preds)
        out["score_thr"] = args.thr
        tmp = dst / f".{wid}.json.tmp"
        tmp.write_text(json.dumps(out))
        os.replace(tmp, dst / f"{wid}.json")
        n_all += len(preds); n_kept += n
    print(f"{len(ids)} WSI: {n_all} -> {n_kept} predictions -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Frozen DINOv3 features of every experience record.

A record is embedded by the state *before* its action, i.e. its patch with the
mask it started from (``crop_path`` + ``prior_path``) -- the same thing GloReflex
embeds when it queries the bank at deployment. Writes ``features.npy``
(N x 3840) and ``index.parquet`` (labels and descriptors, row-aligned).

    python tools/experience/embed_states.py --experience <data_dir>/experience/experience.parquet \
        --out <data_dir>/embed --amp
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from gloreflex.encoder import make_backbone, overlay_tensor

INDEX_COLS = ["record_id", "sample_id", "wsi_id", "source", "record_kind", "iter",
              "recrop_generation", "label_is_glom", "label_dice", "label_delta_dice",
              "dice_full_prior", "dice_full_refined", "delta_dice_full",
              "prior_border_touch_frac", "prior_area_px", "prior_aspect", "prior_white_frac",
              "prior_size_um", "mpp_eff", "det_score", "n_clicks_so_far",
              "iou_pred_prior", "iou_pred_refined", "category", "zone", "did_recrop"]


class OverlayDS(Dataset):
    def __init__(self, rows, spec):
        self.rows, self.spec = rows, spec

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        crop = np.asarray(Image.open(r["crop_path"]).convert("RGB"))
        mask = np.asarray(Image.open(r["prior_path"]).convert("L"))
        return torch.from_numpy(overlay_tensor(crop, mask, self.spec))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experience", required=True, help="experience.parquet from collect_experience.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.experience)
    if args.max_rows:
        df = df.head(args.max_rows)
    dev = torch.device("cuda")
    ext, spec = make_backbone("dinov3")
    ext = ext.to(dev).eval()
    dl = DataLoader(OverlayDS(df.to_dict("records"), spec), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    amp = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.amp else torch.amp.autocast("cuda", enabled=False)
    feats, t0 = [], time.time()
    with torch.no_grad():
        for i, x in enumerate(dl):
            with amp:
                feats.append(ext(x.to(dev, non_blocking=True)).float().cpu().numpy())
            if i % 100 == 0:
                print(f"  {min((i + 1) * args.batch_size, len(df))}/{len(df)}", flush=True)
    F = np.concatenate(feats).astype(np.float32)
    assert len(F) == len(df)
    np.save(out / "features.npy", F)
    df[INDEX_COLS].to_parquet(out / "index.parquet", index=False)
    print(f"{F.shape} -> {out} ({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

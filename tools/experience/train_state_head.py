#!/usr/bin/env python
"""Train the state-encoder MLP head on cached DINOv3 features (backbone frozen).

loss = 0.2 * SupCon (grouped by detection) + CE(glomerulus vs. non-glomerulus)
     + 5 * MSE(Dice) + 5 * MSE(Dice change) + 3 * CE(Dice bin) + 3 * CE(Dice-change bin)

Dice terms use only the ``traj`` records that carry Dice labels; ``probe`` records
only enter the classification and contrastive terms. Different rounds of one
detection share the SupCon group, so near-duplicates are never used as negatives.

A fraction of the training WSIs (``--val-frac``, 20% by default) is held out from
head training. For the zero-shot setting, pass the embeddings of several detectors.

    python tools/experience/train_state_head.py --emb <data_dir>/embed --out <weights>/state_head
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gloreflex.bank import load_embeddings
from gloreflex.encoder import StateHead, embed_features, head_losses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="+", required=True, help="one or more embed_states.py output dirs")
    ap.add_argument("--emb-tags", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--supcon-w", type=float, default=0.2)
    ap.add_argument("--ce-w", type=float, default=1.0)
    ap.add_argument("--dice-w", type=float, default=5.0)
    ap.add_argument("--delta-w", type=float, default=5.0)
    ap.add_argument("--dicebin-w", type=float, default=3.0)
    ap.add_argument("--deltabin-w", type=float, default=3.0)
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of WSIs held out")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    X, ix = load_embeddings(args.emb, args.emb_tags)
    rng = np.random.default_rng(args.seed)
    uw = pd.unique(ix.wsi_id)
    hold = set(rng.choice(uw, max(1, int(len(uw) * args.val_frac)), replace=False).tolist())
    trn = ~ix.wsi_id.isin(hold).to_numpy()
    print(f"{X.shape} | train {int(trn.sum())} rows / held out {int((~trn).sum())} rows ({len(hold)} WSIs)",
          flush=True)

    y_cls = (ix.label_is_glom.to_numpy() == "glom").astype(np.int64)
    y_d = ix.label_dice.to_numpy().astype(np.float32)
    y_dd = ix.label_delta_dice.to_numpy().astype(np.float32)
    sid = ix.sample_id.to_numpy()
    w = {k: getattr(args, k) for k in ("supcon_w", "ce_w", "dice_w", "delta_w", "dicebin_w", "deltabin_w")}

    Xt = torch.from_numpy(X)
    torch.manual_seed(args.seed)
    model = StateHead(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    idx_tr = np.nonzero(trn)[0]
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, len(idx_tr) // args.batch_size))
    log, t0 = [], time.time()
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(idx_tr)
        for s in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            b = perm[s:s + args.batch_size]
            loss, parts = head_losses(model, Xt[b].to(dev), y_cls[b], y_d[b], y_dd[b], sid[b], dev, w)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            log.append({"epoch": ep, "loss": float(loss), **{k: float(v) for k, v in parts.items()}})
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:>3} loss {loss.item():.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    model.eval()
    Z = embed_features(model, X, dev)
    np.save(out / "z.npy", Z)
    torch.save(model.state_dict(), out / "head.pt")
    json.dump({"holdout_wsi": sorted(hold), "in_dim": int(X.shape[1]), "emb": [str(e) for e in args.emb],
               "args": vars(args), "n": int(len(X))}, open(out / "meta.json", "w"), indent=1)
    json.dump(log, open(out / "train_log.json", "w"))
    print(f"z {Z.shape} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

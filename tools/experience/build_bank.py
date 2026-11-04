#!/usr/bin/env python
"""Package the experience bank for deployment and train the read-out models.

Outputs (``--out``)::

    bank_z.npy                   (N, 128) embeddings of all records
    bank_meta.parquet            record kind, labels, descriptors
    readout_{dice,delta,fp}.pkl  gradient-boosting read-outs
    bank.json                    k and the descriptor statistics

Read-out inputs = leave-one-WSI-out k-NN statistics of the record + standardised
descriptors. ``dice`` / ``delta`` regress the Dice and the Dice change of ``traj``
records; ``fp`` classifies ``probe`` records as glomerulus / non-glomerulus.

Several ``--emb`` directories build one bank from the experience of several
detectors (zero-shot setting); ``--head`` must then be a head trained on them.

    python tools/experience/build_bank.py --emb <data_dir>/embed --head <weights>/state_head \
        --out <weights>/bank
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from gloreflex.bank import SCALARS, knn_readout, load_embeddings, prep_scalars
from gloreflex.encoder import StateHead, embed_features


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="+", required=True)
    ap.add_argument("--emb-tags", nargs="*", default=None)
    ap.add_argument("--head", required=True, help="train_state_head.py output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    X, ix = load_embeddings(args.emb, args.emb_tags)
    zp = Path(args.head) / "z.npy"
    if zp.exists() and len(np.load(zp, mmap_mode="r")) == len(ix):
        Z = np.load(zp).astype(np.float32)
    else:
        head = StateHead(X.shape[1]).to(dev)
        head.load_state_dict(torch.load(Path(args.head) / "head.pt", map_location=dev)); head.eval()
        Z = embed_features(head, X, dev)
    del X
    S, s_mu, s_sd = prep_scalars(ix, return_stats=True)
    tr_ = (ix.record_kind == "traj").to_numpy()
    pb = (ix.record_kind == "probe").to_numpy()
    wsi = ix.wsi_id.to_numpy()
    print(f"bank {len(ix)} records", flush=True)

    from sklearn.ensemble import HistGradientBoostingClassifier as C
    from sklearn.ensemble import HistGradientBoostingRegressor as R
    d, dd = knn_readout(Z[tr_], wsi[tr_], ix.label_dice.to_numpy()[tr_], args.k, dev,
                        y2=ix.label_delta_dice.to_numpy()[tr_])
    Xt = np.hstack([d[:, None], dd[:, None], S[tr_]])
    models = {}
    for nm, y in [("dice", ix.label_dice.to_numpy()[tr_]), ("delta", ix.label_delta_dice.to_numpy()[tr_])]:
        m = np.isfinite(y)
        models[nm] = R(max_iter=250, random_state=0).fit(Xt[m], y[m])
    yg = (ix.label_is_glom.to_numpy()[pb] == "glom").astype(int)
    ng = knn_readout(Z[pb], wsi[pb], yg.astype(float), args.k, dev)
    Xf = np.hstack([ng[:, None], S[pb]])
    models["fp"] = C(max_iter=250, random_state=0).fit(Xf, yg)

    np.save(out / "bank_z.npy", Z.astype(np.float32))
    cols = ["record_id", "sample_id", "wsi_id", "source", "record_kind", "iter",
            "label_dice", "label_delta_dice", "label_is_glom"] + SCALARS + ["prior_area_px", "emb_tag"]
    cols = [c for c in dict.fromkeys(cols) if c in ix.columns]
    ix[cols].to_parquet(out / "bank_meta.parquet", index=False)
    for k_, m_ in models.items():
        pickle.dump(m_, open(out / f"readout_{k_}.pkl", "wb"))
    json.dump({"k": args.k, "scalars": SCALARS, "scalar_mu": s_mu.tolist(), "scalar_sd": s_sd.tolist(),
               "emb": [str(e) for e in args.emb], "head": str(args.head)},
              open(out / "bank.json", "w"), indent=1)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

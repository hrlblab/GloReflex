"""Experience bank: k-NN retrieval and read-out models.

The bank holds one record per state visited while refining the training-slide
detections. Two kinds of records exist:

* ``traj``  -- states of the multi-round refinement trajectories of glomeruli,
  labelled with the frame-independent Dice of the current mask (``label_dice``)
  and the Dice change after one more refinement step (``label_delta_dice``);
* ``probe`` -- one state per training detection (glomeruli and non-glomerular
  detections alike), labelled glomerulus / non-glomerulus (``label_is_glom``).

At query time the k nearest neighbours (cosine similarity, softmax weights with
temperature 1/20) give the similarity-weighted Dice and Dice change of the
``traj`` neighbours and the glomerulus fraction of the ``probe`` neighbours.
Gradient-boosting read-outs combine these neighbour statistics with simple mask
descriptors (:data:`SCALARS`) into the glomerulus probability and the expected
Dice change used by the decision rule.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCALARS = ["prior_border_touch_frac", "prior_aspect", "prior_white_frac",
           "prior_size_um", "mpp_eff", "det_score", "n_clicks_so_far",
           "iou_pred_prior", "iter", "recrop_generation", "log_area"]
SIM_TEMP = 20.0


# ----------------------------------------------------------------------------- loading
def load_embeddings(dirs: list, tags: list | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Load one or more embedding directories. With several directories, ``sample_id``
    is prefixed with the directory tag so detections of different detectors never collide."""
    Xs, ixs = [], []
    for i, d in enumerate(dirs):
        d = Path(d)
        X = np.load(d / "features.npy")
        ix = pd.read_parquet(d / "index.parquet").copy()
        assert len(X) == len(ix), f"{d}: {len(X)} features vs {len(ix)} index rows"
        ix["emb_tag"] = tags[i] if tags else d.parent.name
        if len(dirs) > 1:
            ix["sample_id"] = ix["emb_tag"] + "::" + ix["sample_id"].astype(str)
        Xs.append(X.astype(np.float32, copy=False)); ixs.append(ix)
    X = Xs[0] if len(Xs) == 1 else np.concatenate(Xs, axis=0)
    ix = ixs[0] if len(ixs) == 1 else pd.concat(ixs, ignore_index=True)
    return np.ascontiguousarray(X), ix.reset_index(drop=True)


# ----------------------------------------------------------------------------- descriptors
def scalar_matrix(ix: pd.DataFrame) -> np.ndarray:
    d = ix.copy()
    d["log_area"] = np.log1p(d.prior_area_px.astype(float))
    X = d[SCALARS].astype(float).to_numpy()
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def prep_scalars(ix: pd.DataFrame, stats=None, return_stats: bool = False):
    """Standardise the descriptors. At deployment the training-side ``(mu, sd)``
    stored in ``bank.json`` must be used (tree split points are absolute)."""
    X = scalar_matrix(ix)
    if stats is None:
        mu, sd = X.mean(0), X.std(0) + 1e-6
    else:
        mu, sd = np.asarray(stats[0], float), np.asarray(stats[1], float)
    Z = ((X - mu) / sd).astype(np.float32)
    return (Z, mu, sd) if return_stats else Z


# ----------------------------------------------------------------------------- k-NN
def knn_readout(emb, wsi, y, k, dev, chunk=2048, y2=None):
    """Leave-one-WSI-out similarity-weighted k-NN read-out within one set of
    L2-normalised embeddings (neighbours from the query's own WSI are excluded)."""
    E = torch.from_numpy(emb).to(dev)
    W = torch.from_numpy(pd.factorize(wsi)[0]).to(dev)
    Y = torch.from_numpy(np.nan_to_num(y, nan=0.0)).float().to(dev)
    Yv = torch.from_numpy(~np.isnan(y)).float().to(dev)
    Y2 = torch.from_numpy(np.nan_to_num(y2, nan=0.0)).float().to(dev) if y2 is not None else None
    out, out2 = [], []
    for s in range(0, len(E), chunk):
        sim = E[s:s + chunk] @ E.T
        sim[W[s:s + chunk, None] == W[None, :]] = -2.0
        v, i = sim.topk(k, dim=1)
        w = torch.softmax(v * SIM_TEMP, dim=1) * Yv[i]
        ws = w.sum(1).clamp_min(1e-6)
        out.append(((w * Y[i]).sum(1) / ws).cpu().numpy())
        if Y2 is not None:
            out2.append(((w * Y2[i]).sum(1) / ws).cpu().numpy())
    return (np.concatenate(out), np.concatenate(out2)) if Y2 is not None else np.concatenate(out)


def knn_query_bank(qz, qw, bz, bw, y, k, dev, y2=None, chunk=None):
    """Query-vs-bank version of :func:`knn_readout` (neighbours on the query's WSI excluded).
    Returns ``(readout[, readout2], max_similarity)``."""
    if chunk is None:
        chunk = int(max(64, min(2048, 2e8 // max(len(bz), 1))))
    codes, _ = pd.factorize(np.concatenate([np.asarray(qw), np.asarray(bw)]))
    qc = torch.from_numpy(codes[:len(qw)]).to(dev); bc = torch.from_numpy(codes[len(qw):]).to(dev)
    Bz = torch.from_numpy(np.ascontiguousarray(bz)).to(dev)
    Y = torch.from_numpy(np.nan_to_num(y, nan=0.0)).float().to(dev)
    Yv = torch.from_numpy(~np.isnan(y)).float().to(dev)
    Y2 = torch.from_numpy(np.nan_to_num(y2, nan=0.0)).float().to(dev) if y2 is not None else None
    out, out2, smax = [], [], []
    for s in range(0, len(qz), chunk):
        q = torch.from_numpy(np.ascontiguousarray(qz[s:s + chunk])).to(dev)
        sim = q @ Bz.T
        sim[qc[s:s + chunk, None] == bc[None, :]] = -2.0
        v, i = sim.topk(min(k, sim.shape[1]), dim=1)
        w = torch.softmax(v * SIM_TEMP, dim=1) * Yv[i]
        ws = w.sum(1).clamp_min(1e-6)
        out.append(((w * Y[i]).sum(1) / ws).cpu().numpy())
        if Y2 is not None:
            out2.append(((w * Y2[i]).sum(1) / ws).cpu().numpy())
        smax.append(v[:, 0].cpu().numpy())
    r = [np.concatenate(out)]
    if Y2 is not None:
        r.append(np.concatenate(out2))
    r.append(np.concatenate(smax))
    return r


# ----------------------------------------------------------------------------- deployment
class ExperienceBank:
    """Deployed bank: embeddings, labels and read-out models.

    Directory layout (written by ``tools/experience/build_bank.py``)::

        bank_z.npy  bank_meta.parquet  readout_{dice,delta,fp}.pkl  bank.json
    """

    def __init__(self, bank_dir: str | Path, dev):
        bank_dir = Path(bank_dir)
        self.cfg = json.load(open(bank_dir / "bank.json"))
        bm = pd.read_parquet(bank_dir / "bank_meta.parquet")
        self.bz = torch.from_numpy(np.load(bank_dir / "bank_z.npy")).to(dev)
        b_tr = (bm.record_kind == "traj").to_numpy(); b_pb = (bm.record_kind == "probe").to_numpy()
        self.b_dice = torch.from_numpy(np.nan_to_num(bm.label_dice.to_numpy(), nan=0.0)).float().to(dev)
        self.b_dv = torch.from_numpy(np.isfinite(bm.label_dice.to_numpy())).float().to(dev)
        self.b_del = torch.from_numpy(np.nan_to_num(bm.label_delta_dice.to_numpy(), nan=0.0)).float().to(dev)
        self.b_glom = torch.from_numpy((bm.label_is_glom.to_numpy() == "glom").astype(np.float32)).to(dev)
        self.zt = torch.from_numpy(np.nonzero(b_tr)[0]).to(dev)
        self.zp = torch.from_numpy(np.nonzero(b_pb)[0]).to(dev)
        self.ro = {k: pickle.load(open(bank_dir / f"readout_{k}.pkl", "rb")) for k in ("dice", "delta", "fp")}
        self.s_mu = np.asarray(self.cfg["scalar_mu"], np.float64)
        self.s_sd = np.asarray(self.cfg["scalar_sd"], np.float64)
        assert len(self.s_mu) == len(SCALARS), "descriptor statistics do not match SCALARS"
        self.k = int(self.cfg["k"])

    def standardize(self, rows: list[dict]) -> np.ndarray:
        dfx = pd.DataFrame(rows)[SCALARS].astype(float).to_numpy()
        dfx = np.nan_to_num(dfx, nan=0.0, posinf=0.0, neginf=0.0)
        return ((dfx - self.s_mu) / self.s_sd).astype(np.float32)

    def query(self, z: torch.Tensor, scal: np.ndarray):
        """Return ``(predicted Dice, predicted Dice change, P(glomerulus))`` per state."""
        k = self.k
        sim_t = z @ self.bz[self.zt].T; sim_p = z @ self.bz[self.zp].T
        vt, it_ = sim_t.topk(k, 1); vp, ip = sim_p.topk(k, 1)
        wt = torch.softmax(vt * SIM_TEMP, 1) * self.b_dv[self.zt][it_]
        nd = ((wt * self.b_dice[self.zt][it_]).sum(1) / wt.sum(1).clamp_min(1e-6)).cpu().numpy()
        ndd = ((wt * self.b_del[self.zt][it_]).sum(1) / wt.sum(1).clamp_min(1e-6)).cpu().numpy()
        wp = torch.softmax(vp * SIM_TEMP, 1)
        ng = ((wp * self.b_glom[self.zp][ip]).sum(1)).cpu().numpy()
        Xr = np.hstack([nd[:, None], ndd[:, None], scal])
        Xf = np.hstack([ng[:, None], scal])
        return (self.ro["dice"].predict(Xr), self.ro["delta"].predict(Xr),
                self.ro["fp"].predict_proba(Xf)[:, 1])

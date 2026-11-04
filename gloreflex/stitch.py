"""Cross-tile stitching used by the baselines.

A glomerulus cut by a tile border yields partial masks in neighbouring tiles that
overlap each other too little for NMS to remove them. Stitching merges every pair
of predictions (score >= ``score_thr``) whose mask intersection covers at least
``T`` of the smaller mask; each connected group is replaced by the union of its
masks with the maximum score. Predictions below ``score_thr`` are left untouched.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mu

STITCH_T = 0.5


def _dec(p):
    c = p["seg"]["counts"]
    return mu.decode({"size": p["seg"]["size"],
                      "counts": c.encode("ascii") if isinstance(c, str) else c}).astype(bool)


def _enc(m):
    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    return {"size": [int(m.shape[0]), int(m.shape[1])], "counts": r["counts"].decode("ascii")}


def stitch(preds: list[dict], T: float = STITCH_T, score_thr: float = 0.3) -> list[dict]:
    hi = [i for i, p in enumerate(preds) if p["score"] >= score_thr]
    M = {i: _dec(preds[i]) for i in hi}
    org = {i: (int(round(preds[i]["bbox"][0])), int(round(preds[i]["bbox"][1]))) for i in hi}
    area = {i: int(M[i].sum()) for i in hi}
    par = {i: i for i in hi}

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i

    bb = np.array([preds[i]["bbox"] for i in hi], float).reshape(-1, 4)
    for a in range(len(hi)):
        ov = np.where((bb[a + 1:, 0] < bb[a, 2]) & (bb[a + 1:, 2] > bb[a, 0])
                      & (bb[a + 1:, 1] < bb[a, 3]) & (bb[a + 1:, 3] > bb[a, 1]))[0] + a + 1
        i = hi[a]
        for bidx in ov:
            j = hi[bidx]
            (xi, yi), (xj, yj) = org[i], org[j]
            hi_, wi_ = M[i].shape; hj, wj = M[j].shape
            x0, y0 = max(xi, xj), max(yi, yj); x1, y1 = min(xi + wi_, xj + wj), min(yi + hi_, yj + hj)
            if x1 <= x0 or y1 <= y0:
                continue
            inter = int((M[i][y0 - yi:y1 - yi, x0 - xi:x1 - xi] & M[j][y0 - yj:y1 - yj, x0 - xj:x1 - xj]).sum())
            if inter >= T * max(1, min(area[i], area[j])):
                par[find(i)] = find(j)
    groups = {}
    for i in hi:
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        if len(g) == 1:
            out.append(preds[g[0]]); continue
        x0 = min(org[i][0] for i in g); y0 = min(org[i][1] for i in g)
        x1 = max(org[i][0] + M[i].shape[1] for i in g); y1 = max(org[i][1] + M[i].shape[0] for i in g)
        c = np.zeros((y1 - y0, x1 - x0), bool)
        for i in g:
            c[org[i][1] - y0:org[i][1] - y0 + M[i].shape[0], org[i][0] - x0:org[i][0] - x0 + M[i].shape[1]] |= M[i]
        ys, xs = np.where(c)
        c = c[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        ox, oy = x0 + int(xs.min()), y0 + int(ys.min())
        out.append({"bbox": [ox, oy, ox + c.shape[1], oy + c.shape[0]],
                    "score": max(preds[i]["score"] for i in g), "seg": _enc(c), "n_merged": len(g)})
    out += [p for p in preds if p["score"] < score_thr]
    return sorted(out, key=lambda p: -p["score"])

"""Border-touch detection and asymmetric re-crop.

When a glomerulus extends beyond its patch, the refined mask touches the patch
border. If more than ``touch_thresh`` (1%) of the mask contour lies on the
border, the patch is extended on the touched sides only, by 10-40% of its size
depending on the touched fraction, padded back to a square, re-read from the
WSI, and the current mask is mapped into the new frame.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import CROP
from .wsi import read_region_rgb

WHITE_THR = 200        # a pixel is "white" if min(R, G, B) > WHITE_THR
MAX_RECROP = 2         # at most two re-crops per detection


def border_touch_frac(m: np.ndarray) -> float:
    """Fraction of the mask perimeter that lies on the patch border.

    The mask is zero-padded before the morphological gradient: with OpenCV's
    default border handling, mask pixels on the image edge would not be eroded
    and the border segment of the contour would be invisible.
    """
    mp = np.pad(m.astype(np.uint8), 1)
    b = (cv2.morphologyEx(mp, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)[1:-1, 1:-1]
    per = int(b.sum())
    if per == 0:
        return 0.0
    e = np.zeros_like(b); e[0, :] = e[-1, :] = e[:, 0] = e[:, -1] = True
    return float((b & e).sum()) / per


def touch_sides(m: np.ndarray) -> str:
    """Patch sides touched by the mask, e.g. ``"top|left"``."""
    s = []
    if m[0, :].any(): s.append("top")
    if m[-1, :].any(): s.append("bottom")
    if m[:, 0].any(): s.append("left")
    if m[:, -1].any(): s.append("right")
    return "|".join(s)


def plan_recrop(touch: float, sides: str, cx0: int, cy0: int, side: int,
                W: int = 0, H: int = 0) -> tuple[int, int, int, str] | None:
    """New patch ``(x0, y0, side, sides)`` extended on the touched sides only.

    The extension is ``clip(0.08 + 1.5 * touch, 0.10, 0.40) * side`` per touched
    side; the box is then padded symmetrically along its shorter dimension to a
    square. Returns ``None`` if nothing is touched or the patch would not grow
    (or would grow more than 4x).
    """
    if not sides:
        return None
    a = float(np.clip(0.08 + 1.5 * touch, 0.10, 0.40)) * side
    x0, y0, x1, y1 = cx0, cy0, cx0 + side, cy0 + side
    if "left" in sides:   x0 -= a
    if "right" in sides:  x1 += a
    if "top" in sides:    y0 -= a
    if "bottom" in sides: y1 += a
    w, h = x1 - x0, y1 - y0
    n = max(w, h)
    x0 -= (n - w) / 2; y0 -= (n - h) / 2
    nx0, ny0, ns = int(round(x0)), int(round(y0)), int(round(n))
    if ns <= side or ns > 4 * side:
        return None
    return nx0, ny0, ns, sides


def remap_mask(m: np.ndarray, cx0: int, cy0: int, side: int,
               nx0: int, ny0: int, ns: int) -> np.ndarray:
    """Map a mask from the old patch frame into the new one (both ``CROP x CROP``)."""
    out = np.zeros((CROP, CROP), np.uint8)
    k = side / ns
    w = max(int(round(CROP * k)), 1)
    ox = int(round((cx0 - nx0) / ns * CROP)); oy = int(round((cy0 - ny0) / ns * CROP))
    r = cv2.resize(m.astype(np.uint8), (w, w), interpolation=cv2.INTER_NEAREST)
    xs, ys = max(ox, 0), max(oy, 0)
    xe, ye = min(ox + w, CROP), min(oy + w, CROP)
    if xe > xs and ye > ys:
        out[ys:ye, xs:xe] = r[ys - oy:ye - oy, xs - ox:xe - ox]
    return out


def read_crop(slide, cx0: int, cy0: int, side: int) -> np.ndarray:
    """Read a level-0 square patch (white-padded outside the slide) at ``CROP x CROP``."""
    reg = read_region_rgb(slide, cx0, cy0, side, side)
    if reg.shape[0] != side or reg.shape[1] != side:
        pad = np.full((side, side, 3), 255, np.uint8)
        pad[:reg.shape[0], :reg.shape[1]] = reg; reg = pad
    return cv2.resize(reg, (CROP, CROP), interpolation=cv2.INTER_AREA)


def mask_stats(m: np.ndarray, crop: np.ndarray, mpp: float, side: int) -> dict:
    """Simple mask descriptors visible at deployment: area, size (µm), aspect,
    fraction of white pixels under the mask, border-touch fraction."""
    a = int(m.sum())
    out = {"area_px": a, "size_um": np.nan, "aspect": np.nan, "white_frac": np.nan,
           "border_touch_frac": border_touch_frac(m), "touch_sides": touch_sides(m)}
    if a == 0:
        return out
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs:
        (_, _), (w, h), _ = cv2.minAreaRect(max(cs, key=cv2.contourArea))
        if min(w, h) > 0:
            out["aspect"] = float(max(w, h) / min(w, h))
            if mpp == mpp:
                out["size_um"] = float(max(w, h) * (side / CROP) * mpp)
    b = m.astype(bool)
    out["white_frac"] = float((crop[b].min(axis=1) > WHITE_THR).mean())
    return out

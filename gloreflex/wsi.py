"""Whole-slide image access.

Two readers are provided:

* :func:`open_slide` returns an openslide-compatible object (``read_region``,
  ``dimensions``, ``level_dimensions``, ``close``). It is used by the correction
  layer to read patches around detections.
* :class:`TileReader` is a minimal level-0 tile reader used for tiled detector
  inference.

Both handle planar (channel-first) OME-TIFF files, which openslide reads as a
single grayscale plane, by falling back to ``tifffile``.

The module also contains the micron-per-pixel lookup.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# --------------------------------------------------------------------------- #
# planar OME-TIFF support
# --------------------------------------------------------------------------- #
def is_planar_ometiff(path: str | Path) -> bool:
    """True for a .tif/.tiff whose first series is channel-first (C before Y)."""
    p = str(path)
    if not p.lower().endswith((".tif", ".tiff")):
        return False
    try:
        import tifffile
        with tifffile.TiffFile(p) as tf:
            ax = tf.series[0].axes
        return "C" in ax and "Y" in ax and ax.index("C") < ax.index("Y")
    except Exception:  # noqa: BLE001
        return False


def spatial_hw(series) -> tuple[int, int]:
    """(H, W) of a tifffile series, whatever the axis order or labelling."""
    keep = [(a, s) for a, s in zip(series.axes, list(series.shape))
            if not (a in "TZ" and s == 1)]
    ci = next((k for k, (a, _) in enumerate(keep) if a in "CS"), None)
    if ci is None:
        ci = next((k for k, (_, s) in enumerate(keep) if s in (3, 4)), None)
    spat = [s for k, (_, s) in enumerate(keep) if k != ci]
    if len(spat) != 2:
        raise ValueError(f"cannot derive H,W from axes={series.axes} shape={series.shape}")
    return int(spat[0]), int(spat[1])


class PlanarSlide:
    """openslide-compatible reader for (planar or multi-dimensional) OME-TIFF.

    Level-0 windows are served from a lazily loaded HWC array.
    """

    def __init__(self, path: str):
        import tifffile
        self.path = str(path)
        self._tf = tifffile.TiffFile(self.path)
        self._series = self._tf.series[0]
        self._levels = list(self._series.levels)          # large -> small
        h0, w0 = spatial_hw(self._levels[0])
        self.W, self.H = int(w0), int(h0)
        self._l0 = None
        self.properties: dict = {}

    @staticmethod
    def _to_hwc(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        while arr.ndim > 3:                               # squeeze T/Z
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (3, 4):      # CHW (planar)
            arr = np.transpose(arr[:3], (1, 2, 0))
        elif arr.ndim == 3 and arr.shape[2] >= 3:         # HWC
            arr = arr[:, :, :3]
        elif arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return np.ascontiguousarray(arr.astype(np.uint8))

    @property
    def level_count(self) -> int:
        return len(self._levels)

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        out = []
        for lv in self._levels:
            h, w = spatial_hw(lv)
            out.append((int(w), int(h)))
        return out

    @property
    def dimensions(self) -> tuple[int, int]:
        return (self.W, self.H)

    def _level0_hwc(self) -> np.ndarray:
        if self._l0 is None:
            self._l0 = self._to_hwc(self._levels[0].asarray())
        return self._l0

    def read_region(self, location, level, size):
        """location=(x, y) in level-0 pixels, size=(w, h) in level pixels -> RGB PIL image."""
        x0, y0 = int(location[0]), int(location[1])
        w, h = int(size[0]), int(size[1])
        lw, lh = self.level_dimensions[level]
        ds = self.W / lw
        lx, ly = int(round(x0 / ds)), int(round(y0 / ds))
        arr = self._level0_hwc() if level == 0 else self._to_hwc(self._levels[level].asarray())
        tile = arr[ly:ly + h, lx:lx + w]
        if tile.shape[0] != h or tile.shape[1] != w:
            pad = np.zeros((h, w, 3), np.uint8)
            pad[:tile.shape[0], :tile.shape[1]] = tile
            tile = pad
        return Image.fromarray(tile, mode="RGB")

    def close(self) -> None:
        self._tf.close()


def open_slide(path: str | Path):
    """Open a WSI: openslide when possible, :class:`PlanarSlide` for planar/multi-dim TIFF."""
    import openslide
    path = str(path)
    if is_planar_ometiff(path):
        return PlanarSlide(path)
    try:
        return openslide.OpenSlide(path)
    except Exception:  # noqa: BLE001 -- multi-dim / interleaved OME-TIFF
        if path.lower().endswith((".tif", ".tiff")):
            return PlanarSlide(path)
        raise


def read_region_rgb(slide, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """Read a level-0 RGB region (x0, y0 in level-0 pixels) as HWC uint8."""
    img = slide.read_region((int(x0), int(y0)), 0, (int(w), int(h))).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# tile reader for detector inference
# --------------------------------------------------------------------------- #
def _planar_level0_rgb(path: str) -> np.ndarray:
    """Read the full level 0 of a planar / multi-dim OME-TIFF as HWC uint8 RGB."""
    import tifffile
    with tifffile.TiffFile(path) as tf:
        arr = np.asarray(tf.series[0].asarray())
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.transpose(arr[:3], (1, 2, 0))
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    elif arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return np.ascontiguousarray(arr.astype(np.uint8))


class TileReader:
    """Level-0 RGB tile reader: openslide for regular slides, a full tifffile load
    for planar OME-TIFF and for files openslide cannot open."""

    def __init__(self, path: str):
        self.path = path
        self._planar = None
        self._slide = None
        if is_planar_ometiff(path):
            self._planar = _planar_level0_rgb(path)
            self.H, self.W = self._planar.shape[:2]
            return
        try:
            import openslide
            self._slide = openslide.OpenSlide(path)
            self.W, self.H = self._slide.dimensions
        except Exception:  # noqa: BLE001
            self._planar = _planar_level0_rgb(path)
            self.H, self.W = self._planar.shape[:2]

    @property
    def dimensions(self) -> tuple[int, int]:
        return (self.W, self.H)

    def read_rgb(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        if self._planar is not None:
            tile = self._planar[y:y + h, x:x + w]
            if tile.shape[0] != h or tile.shape[1] != w:
                pad = np.zeros((h, w, 3), np.uint8)
                pad[:tile.shape[0], :tile.shape[1]] = tile[:, :, :3]
                tile = pad
            return tile
        reg = self._slide.read_region((x, y), 0, (w, h)).convert("RGB")
        return np.asarray(reg)

    def close(self) -> None:
        if self._slide is not None:
            self._slide.close()


def not_white_frac(rgb: np.ndarray, white_v: int = 230) -> float:
    """Fraction of pixels that are not near-white (max channel < ``white_v``)."""
    return float((rgb.max(axis=2) < white_v).mean())


# --------------------------------------------------------------------------- #
# microns per pixel
# --------------------------------------------------------------------------- #
def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def resolve_mpp(path: str, given_mpp=None) -> tuple[float | None, str]:
    """Return ``(mpp, source)``. Tries the given value, openslide metadata and the
    OME-XML ``PhysicalSizeX``; returns ``(None, "unknown")`` instead of guessing."""
    if _ok(given_mpp):
        return float(given_mpp), "given"
    try:
        import openslide
        s = openslide.OpenSlide(path)
        v = s.properties.get("openslide.mpp-x")
        s.close()
        if v:
            return float(v), "openslide"
    except Exception:  # noqa: BLE001
        pass
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            d = tf.pages[0].description or ""
        m = re.search(r'PhysicalSizeX="([\d.]+)"', d)
        u = re.search(r'PhysicalSizeXUnit="([^"]+)"', d)
        if m and (u is None or u.group(1) in ("µm", "um", "micron")):
            return float(m.group(1)), "ome-xml"
    except Exception:  # noqa: BLE001
        pass
    return None, "unknown"

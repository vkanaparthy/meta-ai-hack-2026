"""Elevation model mosaic and terrain-skyline rendering.

Tiles: Copernicus GLO-30 DEM (download with scripts/download_dem.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

R_EARTH = 6_371_000.0
REFRACTION_K = 0.13          # standard atmospheric refraction coefficient


@dataclass
class DEM:
    z: np.ndarray            # elevation grid, metres, row 0 = north edge
    lat_max: float           # latitude of row 0 (pixel centre)
    lon_min: float           # longitude of column 0 (pixel centre)
    step_deg: float          # grid spacing, degrees

    @classmethod
    def from_tiles(cls, paths: list[Path], step_arcsec: float = 2.0) -> "DEM":
        """Mosaic GeoTIFF tiles onto a regular lat/lon grid (nearest/average resample)."""
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.merge import merge

        srcs = [rasterio.open(p) for p in paths]
        try:
            step = step_arcsec / 3600
            arr, transform = merge(srcs, res=(step, step), resampling=Resampling.average)
        finally:
            for s in srcs:
                s.close()
        z = arr[0].astype(np.float32)
        z[(z < -500) | (z > 9000)] = np.nan
        return cls(z, transform.f + transform.e / 2, transform.c + transform.a / 2, step)

    @classmethod
    def load(cls, path: Path) -> "DEM":
        d = np.load(path)
        return cls(d["z"], float(d["lat_max"]), float(d["lon_min"]), float(d["step_deg"]))

    def save(self, path: Path) -> None:
        np.savez_compressed(path, z=self.z, lat_max=self.lat_max, lon_min=self.lon_min,
                            step_deg=self.step_deg)

    def elevation(self, lat, lon):
        """Bilinear elevation; NaN outside the grid."""
        r = (self.lat_max - np.asarray(lat, float)) / self.step_deg
        c = (np.asarray(lon, float) - self.lon_min) / self.step_deg
        r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
        ok = (r0 >= 0) & (c0 >= 0) & (r0 < self.z.shape[0] - 1) & (c0 < self.z.shape[1] - 1)
        r0c, c0c = np.where(ok, r0, 0), np.where(ok, c0, 0)
        fr, fc = r - r0c, c - c0c
        z = self.z
        val = (z[r0c, c0c] * (1 - fr) * (1 - fc) + z[r0c + 1, c0c] * fr * (1 - fc)
               + z[r0c, c0c + 1] * (1 - fr) * fc + z[r0c + 1, c0c + 1] * fr * fc)
        return np.where(ok, val, np.nan)

    def snap_to_summit(self, lat: float, lon: float, radius_m: float = 600.0):
        """Highest DEM point within radius (camera sites sit on summits).
        Returns (lat, lon, elevation, moved_m)."""
        spacing_m = min(30.0, self.step_deg * 111_000)
        n = int(radius_m / spacing_m) + 1
        dl = np.arange(-n, n + 1) * spacing_m / 111_000
        LA, LO = np.meshgrid(lat + dl, lon + dl / np.cos(np.radians(lat)), indexing="ij")
        dist = np.hypot((LA - lat) * 111_000, (LO - lon) * 111_000 * np.cos(np.radians(lat)))
        z = np.where(dist <= radius_m, self.elevation(LA, LO), np.nan)
        if np.all(np.isnan(z)):
            return lat, lon, float("nan"), 0.0
        i = np.unravel_index(np.nanargmax(z), z.shape)
        return float(LA[i]), float(LO[i]), float(z[i]), float(dist[i])


def skyline(dem: DEM, lat: float, lon: float, h_cam: float, az_deg: np.ndarray,
            d_min: float = 150.0, d_max: float = 120_000.0, growth: float = 1.006,
            min_elevation_deg: float = -10.0):
    """Terrain skyline seen from (lat, lon, h_cam): for each azimuth, the highest
    elevation angle of any terrain point, and that point's distance (metres)."""
    n = int(np.log(d_max / d_min) / np.log(growth)) + 1
    d = d_min * growth ** np.arange(n)
    az = np.radians(np.asarray(az_deg, float))
    best_el = np.full(az.shape, np.radians(min_elevation_deg))
    best_d = np.full(az.shape, np.nan)
    coslat = np.cos(np.radians(lat))
    for chunk in np.array_split(np.arange(n), max(1, n // 100)):
        dd = d[chunk][None, :]
        la = lat + np.degrees(dd * np.cos(az)[:, None] / R_EARTH)
        lo = lon + np.degrees(dd * np.sin(az)[:, None] / (R_EARTH * coslat))
        z = dem.elevation(la, lo)
        drop = dd ** 2 * (1 - REFRACTION_K) / (2 * R_EARTH)
        el = np.arctan2(z - drop - h_cam, dd)
        el = np.where(np.isnan(el), -np.inf, el)
        j = np.argmax(el, axis=1)
        e = el[np.arange(len(az)), j]
        better = e > best_el
        best_el = np.where(better, e, best_el)
        best_d = np.where(better, dd[0, j], best_d)
    return np.degrees(best_el), best_d

"""GOES-18 ABI fire detections (L2 FDCC: CONUS, every 5 minutes, about 2 km pixels).

Only the fire codes NOAA recommends for users who need few false alarms are kept:
10 processed, 11 saturated, and their temporally filtered versions 30 and 31.
Pixel positions are not terrain-corrected, so in mountains a detection can sit a few km
off the true fire; matching allows for that (see GOES_PIXEL_M).
"""
from __future__ import annotations

import numpy as np

FIRE_CODES = {10: "processed", 11: "saturated", 30: "processed (filtered)", 31: "saturated (filtered)"}
GOES_PIXEL_M = 3000            # ~2 km at nadir, ~3 km over Southern California, plus terrain offset


def fixed_grid_to_latlon(x, y, h_sat: float, r_eq: float, r_pol: float, lon0_deg: float):
    """GOES-R ABI fixed-grid scan angles (radians) to geodetic lat/lon (degrees).
    Formula from the GOES-R Product User Guide (section 4.2.8.1)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    H = h_sat + r_eq
    lam0 = np.radians(lon0_deg)
    a = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + (r_eq ** 2 / r_pol ** 2) * np.sin(y) ** 2)
    b = -2 * H * np.cos(x) * np.cos(y)
    c = H ** 2 - r_eq ** 2
    disc = b ** 2 - 4 * a * c
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(disc)) / (2 * a)
        sx = rs * np.cos(x) * np.cos(y)
        sy = -rs * np.sin(x)
        sz = rs * np.cos(x) * np.sin(y)
        lat = np.degrees(np.arctan((r_eq ** 2 / r_pol ** 2) * sz / np.sqrt((H - sx) ** 2 + sy ** 2)))
        lon = np.degrees(lam0 - np.arctan(sy / (H - sx)))
    return np.where(disc >= 0, lat, np.nan), np.where(disc >= 0, lon, np.nan)


def _scaled(ds):
    v = ds[()].astype(float)
    fill = ds.attrs.get("_FillValue")
    if fill is not None:
        v = np.where(ds[()] == np.asarray(fill).ravel()[0], np.nan, v)
    sf = ds.attrs.get("scale_factor")
    off = ds.attrs.get("add_offset")
    if sf is not None:
        v = v * float(np.asarray(sf).ravel()[0])
    if off is not None:
        v = v + float(np.asarray(off).ravel()[0])
    return v


def read_fires(path, box: tuple[float, float, float, float]) -> list[dict]:
    """Fire pixels inside box (west, south, east, north) from one FDCC NetCDF file."""
    import h5py

    west, south, east, north = box
    with h5py.File(path, "r") as f:
        mask = f["Mask"][()]
        rows, cols = np.nonzero(np.isin(mask, list(FIRE_CODES)))
        if rows.size == 0:
            return []
        x = _scaled(f["x"])[cols]
        y = _scaled(f["y"])[rows]
        p = f["goes_imager_projection"].attrs
        g = lambda k: float(np.asarray(p[k]).ravel()[0])
        lat, lon = fixed_grid_to_latlon(x, y, g("perspective_point_height"), g("semi_major_axis"),
                                        g("semi_minor_axis"), g("longitude_of_projection_origin"))
        power = _scaled(f["Power"])[rows, cols] if "Power" in f else np.full(rows.shape, np.nan)
        out = []
        for i in range(rows.size):
            if south <= lat[i] <= north and west <= lon[i] <= east:
                out.append({"lat": round(float(lat[i]), 5), "lon": round(float(lon[i]), 5),
                            "mask": int(mask[rows[i], cols[i]]),
                            "frp_mw": None if np.isnan(power[i]) else round(float(power[i]), 1)})
        return out

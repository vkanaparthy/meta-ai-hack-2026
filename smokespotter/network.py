"""What-if geometry for a camera network: who can see a point, and how precisely bearings from a
set of cameras would fix it. Used for offline studies only (it takes the true point as input)."""
from __future__ import annotations

import itertools
import math

import numpy as np

from .geo import bearing_deg, haversine_m

R_EARTH = 6_371_000.0
K_REFRACTION = 0.13
CHI2_90_2D = 4.605          # 90% quantile of chi-square with 2 degrees of freedom


def line_of_sight(dem, cam_lat, cam_lon, cam_h, tgt_lat, tgt_lon, tgt_h, n: int = 240) -> bool:
    """True if no terrain blocks the straight line from camera to target (heights in metres above
    sea level), with earth curvature and standard refraction."""
    d = haversine_m(cam_lat, cam_lon, tgt_lat, tgt_lon)
    s = np.linspace(0, 1, n)[1:-1]
    s = s[s * d > 300]                     # the mast's own hilltop doesn't block its view
    lats = cam_lat + (tgt_lat - cam_lat) * s
    lons = cam_lon + (tgt_lon - cam_lon) * s
    z = dem.elevation(lats, lons)
    if np.all(np.isnan(z)):
        return False
    r_eff = R_EARTH / (1 - K_REFRACTION)
    bulge = (s * d) * ((1 - s) * d) / (2 * r_eff)
    line = cam_h + (tgt_h - cam_h) * s
    return bool(np.all(np.nan_to_num(z, nan=-1e9) + bulge <= line))


def fix_radius_m(cams: list[tuple[float, float]], tgt: tuple[float, float], sigma_deg: float = 1.0) -> float:
    """90% error radius (worst axis) of a bearings-only fix from these camera positions,
    linearised at the target. Infinite if the bearings can't fix it (one camera, or all in line)."""
    if len(cams) < 2:
        return math.inf
    J = np.zeros((2, 2))
    for la, lo in cams:
        d = haversine_m(la, lo, *tgt)
        b = math.radians(bearing_deg(la, lo, *tgt))
        u = np.array([math.cos(b), -math.sin(b)])            # east, north components across the ray
        w = 1.0 / (d * math.radians(sigma_deg)) ** 2
        J += w * np.outer(u, u)
    ev = np.linalg.eigvalsh(J)
    if ev[0] <= 1e-18:
        return math.inf
    return math.sqrt(CHI2_90_2D / ev[0])


def best_subset(cams: list[tuple[float, float]], tgt, k: int, sigma_deg: float = 1.0):
    """The k cameras giving the smallest fix radius (exhaustive for k=2, greedy beyond)."""
    if len(cams) < k:
        return None, math.inf
    pair = min(itertools.combinations(range(len(cams)), 2),
               key=lambda p: fix_radius_m([cams[i] for i in p], tgt, sigma_deg))
    chosen = list(pair)
    while len(chosen) < k:
        nxt = min((i for i in range(len(cams)) if i not in chosen),
                  key=lambda i: fix_radius_m([cams[j] for j in chosen + [i]], tgt, sigma_deg))
        chosen.append(nxt)
    return chosen, fix_radius_m([cams[i] for i in chosen], tgt, sigma_deg)

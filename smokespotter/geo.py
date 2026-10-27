"""Bearings from pixels and multi-camera triangulation.

Uses a local tangent-plane approximation, accurate to well under 1% over the
tens of kilometres between HPWREN stations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EARTH_R = 6_371_000.0


def pixel_to_bearing(x: float, width: int, heading_deg: float, hfov_deg: float) -> float:
    """Compass bearing (deg) of image column ``x`` for a level, fixed camera."""
    f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    offset = math.degrees(math.atan((x - width / 2) / f))
    return (heading_deg + offset) % 360


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class Observation:
    lat: float
    lon: float
    bearing_deg: float
    weight: float = 1.0


@dataclass(frozen=True)
class Fix:
    lat: float
    lon: float
    error_m: float      # RMS perpendicular distance of the fix from the bearing lines
    n_obs: int


def triangulate(obs: list[Observation]) -> Fix:
    """Least-squares point closest to all bearing rays."""
    if len(obs) < 2:
        raise ValueError("need at least two observations")
    lat0 = sum(o.lat for o in obs) / len(obs)
    lon0 = sum(o.lon for o in obs) / len(obs)
    kx = math.radians(1) * EARTH_R * math.cos(math.radians(lat0))
    ky = math.radians(1) * EARTH_R

    A = np.zeros((2, 2))
    b = np.zeros(2)
    lines = []
    for o in obs:
        p = np.array([(o.lon - lon0) * kx, (o.lat - lat0) * ky])
        t = math.radians(o.bearing_deg)
        d = np.array([math.sin(t), math.cos(t)])          # east, north
        n = np.eye(2) - np.outer(d, d)                     # projector onto line normal
        A += o.weight * n
        b += o.weight * n @ p
        lines.append((p, d, n))
    if abs(np.linalg.det(A)) < 1e-9:
        raise ValueError("bearings are parallel; cannot intersect")
    x = np.linalg.solve(A, b)

    for p, d, _ in lines:
        if np.dot(x - p, d) < 0:
            raise ValueError("solution lies behind a camera; check headings")
    resid = [float(np.linalg.norm(n @ (x - p))) for p, _, n in lines]
    err = math.sqrt(sum(r * r for r in resid) / len(resid))
    return Fix(lat0 + x[1] / ky, lon0 + x[0] / kx, err, len(obs))

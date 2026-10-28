"""Cross-camera association and triangulation of smoke plumes.

Each plume track contributes one bearing (its plume-base column over the first
few frames, where the base marks the origin). Tracks from different stations
are combined; the combination whose bearings agree best, using the most
stations, becomes the fire location.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass

from .geo import Fix, Observation, bearing_deg, haversine_m, triangulate


@dataclass(frozen=True)
class TrackBearing:
    camera_id: str          # e.g. "hp-s"
    station: str
    track_id: str
    lat: float
    lon: float
    bearing_deg: float
    available_epoch: int    # when this bearing is first usable (3rd frame of the track)
    edge_clipped: bool = False
    sigma_deg: float = 1.0  # 1-sigma bearing uncertainty


@dataclass(frozen=True)
class Location:
    fix: Fix
    members: tuple[TrackBearing, ...]
    epoch: int              # time the fix became possible
    uncertainty_m: float    # 90th percentile error from simulated bearing noise


def _obs(members, noise=None):
    return [Observation(m.lat, m.lon, m.bearing_deg + (noise[i] if noise else 0.0), 1.0 / m.sigma_deg ** 2)
            for i, m in enumerate(members)]


def angular_misfit(members, fix) -> list[float]:
    """Each bearing's deviation from the direction to the fix, in sigmas."""
    out = []
    for m in members:
        d = (bearing_deg(m.lat, m.lon, fix.lat, fix.lon) - m.bearing_deg + 180) % 360 - 180
        out.append(d / m.sigma_deg)
    return out


def uncertainty_m(members: tuple[TrackBearing, ...], n: int = 300, seed: int = 0) -> float:
    rng = random.Random(seed)
    base = triangulate(_obs(members))
    dists = []
    for _ in range(n):
        try:
            f = triangulate(_obs(members, [rng.gauss(0, m.sigma_deg) for m in members]))
        except ValueError:
            continue
        dists.append(haversine_m(base.lat, base.lon, f.lat, f.lon))
    if not dists:
        return float("inf")
    dists.sort()
    return dists[int(0.9 * (len(dists) - 1))]


def merge_fragments(tracks: list[TrackBearing], max_sep_deg: float = 4.0,
                    max_dt_s: int = 600) -> list[TrackBearing]:
    """SAM sometimes splits one plume into several masks. Within a camera, tracks
    whose bearings are within max_sep_deg and that appear within max_dt_s of each
    other are treated as one plume; the earliest one (closest to the origin) is kept."""
    out: list[TrackBearing] = []
    for t in sorted(tracks, key=lambda t: t.available_epoch):
        dup = any(o.camera_id == t.camera_id
                  and abs((o.bearing_deg - t.bearing_deg + 180) % 360 - 180) <= max_sep_deg
                  and abs(o.available_epoch - t.available_epoch) <= max_dt_s for o in out)
        if not dup:
            out.append(t)
    return out


OUTLIER_LOGP = math.log(1 / 90.0)     # an unrelated bearing: uniform over a ~90 degree view


def set_loglik(members, mis) -> float:
    """Log-likelihood of treating ``members`` as inliers (Gaussian bearing errors)."""
    return sum(-0.5 * x * x - math.log(m.sigma_deg * math.sqrt(2 * math.pi))
               for m, x in zip(members, mis))


def best_location(tracks: list[TrackBearing], n_sigma: float = 4.0,
                  max_range_m: float = 80_000.0) -> Location | None:
    """Robust choice of which tracks belong to one fire. Each station that saw a
    plume either contributes one inlier track (Gaussian error with its sigma) or
    counts as an outlier (uniform). Picks the subset with the highest
    likelihood, so a camera that disagrees with the rest is dropped instead of
    dragging the fix. Hard limits: every inlier within n_sigma, fix in range."""
    by_station: dict[str, list[TrackBearing]] = {}
    for t in merge_fragments(tracks):
        by_station.setdefault(t.station, []).append(t)
    stations = sorted(by_station)
    best: tuple | None = None
    for k in range(len(stations), 1, -1):
        for subset in itertools.combinations(stations, k):
            for members in itertools.product(*(by_station[s] for s in subset)):
                try:
                    fix = triangulate(_obs(members))
                except ValueError:
                    continue
                mis = angular_misfit(members, fix)
                if any(abs(x) > n_sigma for x in mis):
                    continue
                if any(haversine_m(m.lat, m.lon, fix.lat, fix.lon) > max_range_m for m in members):
                    continue
                ll = set_loglik(members, mis) + (len(stations) - k) * OUTLIER_LOGP
                if best is None or ll > best[0]:
                    best = (ll, fix, members)
    if best is None:
        return None
    _, fix, members = best
    return Location(fix, tuple(members), max(m.available_epoch for m in members),
                    uncertainty_m(tuple(members)))


def first_location(tracks: list[TrackBearing], **kw) -> Location | None:
    """Earliest time a consistent fix exists, using only tracks available by then."""
    for t in sorted({t.available_epoch for t in tracks}):
        loc = best_location([x for x in tracks if x.available_epoch <= t], **kw)
        if loc is not None:
            return loc
    return None


def to_geojson(loc: Location, truth: tuple[float, float] | None = None,
               line_km: float = 40.0) -> dict:
    feats = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [loc.fix.lon, loc.fix.lat]},
        "properties": {"kind": "fire_fix", "uncertainty_m": round(loc.uncertainty_m),
                       "line_residual_m": round(loc.fix.error_m), "epoch": loc.epoch,
                       "cameras": [m.camera_id for m in loc.members]},
    }]
    for m in loc.members:
        d = line_km * 1000 / 6_371_000
        lat1, lon1, b = math.radians(m.lat), math.radians(m.lon), math.radians(m.bearing_deg)
        lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(b))
        lon2 = lon1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(lat1),
                                 math.cos(d) - math.sin(lat1) * math.sin(lat2))
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[m.lon, m.lat], [math.degrees(lon2), math.degrees(lat2)]]},
            "properties": {"kind": "bearing", "camera": m.camera_id, "track": m.track_id,
                           "bearing_deg": round(m.bearing_deg, 2), "sigma_deg": m.sigma_deg,
                           "edge_clipped": m.edge_clipped},
        })
    if truth:
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": [truth[1], truth[0]]},
                      "properties": {"kind": "official_ignition"}})
    return {"type": "FeatureCollection", "features": feats}

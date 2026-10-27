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

from .geo import Fix, Observation, haversine_m, triangulate


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


@dataclass(frozen=True)
class Location:
    fix: Fix
    members: tuple[TrackBearing, ...]
    epoch: int              # time the fix became possible
    uncertainty_m: float    # 90th percentile error from simulated bearing noise


def uncertainty_m(members: tuple[TrackBearing, ...], sigma_deg: float = 1.0,
                  n: int = 300, seed: int = 0) -> float:
    rng = random.Random(seed)
    base = triangulate([Observation(m.lat, m.lon, m.bearing_deg) for m in members])
    dists = []
    for _ in range(n):
        try:
            f = triangulate([Observation(m.lat, m.lon, m.bearing_deg + rng.gauss(0, sigma_deg))
                             for m in members])
        except ValueError:
            continue
        dists.append(haversine_m(base.lat, base.lon, f.lat, f.lon))
    if not dists:
        return float("inf")
    dists.sort()
    return dists[int(0.9 * (len(dists) - 1))]


def best_location(tracks: list[TrackBearing], max_resid_m: float = 400.0,
                  max_range_m: float = 80_000.0, sigma_deg: float = 1.0) -> Location | None:
    """Try every set of tracks with at most one per station; keep consistent ones
    (all bearing lines pass within max_resid_m of the fix, fix within range of
    every camera); prefer more stations, then lower residual."""
    by_station: dict[str, list[TrackBearing]] = {}
    for t in tracks:
        by_station.setdefault(t.station, []).append(t)
    stations = sorted(by_station)
    best: tuple | None = None
    for k in range(len(stations), 1, -1):
        for subset in itertools.combinations(stations, k):
            for members in itertools.product(*(by_station[s] for s in subset)):
                try:
                    fix = triangulate([Observation(m.lat, m.lon, m.bearing_deg) for m in members])
                except ValueError:
                    continue
                if fix.error_m > max_resid_m:
                    continue
                if any(haversine_m(m.lat, m.lon, fix.lat, fix.lon) > max_range_m for m in members):
                    continue
                key = (-k, sum(m.edge_clipped for m in members), fix.error_m)
                if best is None or key < best[0]:
                    best = (key, fix, members)
        if best is not None:
            break       # found a consistent set with k stations; fewer can't beat it
    if best is None:
        return None
    _, fix, members = best
    return Location(fix, tuple(members), max(m.available_epoch for m in members),
                    uncertainty_m(tuple(members), sigma_deg))


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
                           "bearing_deg": round(m.bearing_deg, 2), "edge_clipped": m.edge_clipped},
        })
    if truth:
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": [truth[1], truth[0]]},
                      "properties": {"kind": "official_ignition"}})
    return {"type": "FeatureCollection", "features": feats}

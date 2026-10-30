"""NASA FIRMS satellite hotspots as a second, independent source of evidence.

Two kinds of source:
  * FIRMS: polar-orbiting VIIRS and MODIS, fine pixels but only a few passes a day. In a replay a
    hotspot becomes visible at its observation time plus FIRMS_LATENCY_S (NASA's ultra-real-time
    feed for the US and Canada publishes within about a minute).
  * GOES-18 ABI: geostationary, a CONUS scan every 5 minutes, coarse (~3 km here). A pixel becomes
    visible at the file's measured publication time in NOAA's public bucket.
"""
from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass

from .config import ROOT, load_env

DEFAULT_LATENCY_S = 60


@dataclass(frozen=True)
class Hotspot:
    lat: float
    lon: float
    acq_epoch: int               # satellite observation time
    available_epoch: int         # when a live system would have it
    satellite: str               # e.g. "N20 VIIRS", "GOES-18 ABI"
    confidence: str
    frp: float                   # fire radiative power, MW
    pixel_m: float = 375         # footprint: VIIRS 375 m, MODIS 1 km, GOES ~3 km here
    persistent: bool = False     # GOES pixel already hot before the replay (solar farm etc.)


def latency_s() -> int:
    import os
    load_env()
    return int(os.getenv("FIRMS_LATENCY_S", DEFAULT_LATENCY_S))


def load_hotspots(fire: str, t_start: int | None = None) -> list[Hotspot]:
    """FIRMS (polar) hotspots plus GOES-18 (geostationary) fire pixels, if downloaded."""
    return sorted(_firms(fire) + _goes(fire, t_start), key=lambda h: h.available_epoch)


def _goes(fire: str, t_start: int | None) -> list[Hotspot]:
    from .goes import FIRE_CODES, GOES_PIXEL_M

    path = ROOT / "data" / "goes" / f"{fire}.csv"
    if not path.exists():
        return []
    rows = list(csv.DictReader(open(path)))
    key = lambda r: (round(float(r["lat"]), 2), round(float(r["lon"]), 2))
    before = {key(r) for r in rows if t_start is not None and int(r["scan_start"]) < t_start}
    return [Hotspot(float(r["lat"]), float(r["lon"]), int(r["scan_start"]), int(r["available"]),
                    f"{r.get('sat') or 'GOES-18'} ABI",
                    FIRE_CODES.get(int(r["mask"]), r["mask"]), float(r["frp_mw"] or 0), GOES_PIXEL_M,
                    key(r) in before) for r in rows]


def _firms(fire: str) -> list[Hotspot]:
    path = ROOT / "data" / "firms" / f"{fire}.csv"
    if not path.exists():
        return []
    lag = latency_s()
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            at = dt.datetime.strptime(r["acq_date"] + r["acq_time"].zfill(4), "%Y-%m-%d%H%M")
            e = int(at.replace(tzinfo=dt.timezone.utc).timestamp())       # FIRMS times are UTC
            try:
                frp = float(r.get("frp") or 0)
            except ValueError:
                frp = 0.0
            inst = r.get("instrument", "")
            out.append(Hotspot(float(r["latitude"]), float(r["longitude"]), e, e + lag,
                               f"{r.get('satellite', '')} {inst}".strip(),
                               str(r.get("confidence", "")), frp, 1000 if inst == "MODIS" else 375))
    return out

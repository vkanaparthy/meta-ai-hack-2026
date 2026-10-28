"""Nearest named places, for plain-language alert locations."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache

from .config import ROOT
from .geo import bearing_deg, haversine_m

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def compass(bearing: float) -> str:
    return COMPASS[int((bearing % 360) / 22.5 + 0.5) % 16]


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float
    kind: str
    population: int


@lru_cache(maxsize=1)
def load_places() -> tuple[Place, ...]:
    path = ROOT / "data" / "places.csv"
    if not path.exists():
        return ()
    with open(path) as fh:
        return tuple(Place(r["name"], float(r["lat"]), float(r["lon"]), r["kind"], int(r["population"] or 0))
                     for r in csv.DictReader(fh))


def nearby(lat: float, lon: float, k: int = 3, places=None) -> list[dict]:
    """The k nearest towns, plus the nearest peak if one is within 5 km. Each item
    says where the point is FROM the place, e.g. 6.2 km NW of Julian."""
    places = load_places() if places is None else places
    scored = sorted(((haversine_m(p.lat, p.lon, lat, lon), p) for p in places), key=lambda x: x[0])
    towns = [(d, p) for d, p in scored if p.kind == "town"][:k]
    peaks = [(d, p) for d, p in scored if p.kind == "peak" and d < 5000][:1]
    out = []
    for d, p in towns + peaks:
        direction = compass(bearing_deg(p.lat, p.lon, lat, lon))
        out.append({"name": p.name, "kind": p.kind, "distance_km": round(d / 1000, 1),
                    "direction": direction,
                    "phrase": f"{d / 1000:.1f} km {direction} of {p.name}" if d >= 500 else f"at {p.name}"})
    return out

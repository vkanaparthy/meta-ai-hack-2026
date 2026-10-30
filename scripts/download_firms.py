"""Download NASA FIRMS satellite hotspots around each replayed fire (for the cameras-vs-satellites
comparison). Needs a free MAP_KEY: FIRMS_MAP_KEY in .env.

  python scripts/download_firms.py                      # the six replayed fires
  python scripts/download_firms.py --fire Kitchenfire

One request per sensor per day (the fire's day and the two after), one at a time with a pause.
Standard-processing data (_SP) first; near-real-time (_NRT) when SP isn't published yet.
The search box surrounds the fire's cameras, not the official origin.
Writes data/firms/<fire>.csv. Times in FIRMS are UTC (acq_date, acq_time HHMM).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smokespotter.config import ROOT, setting  # noqa: E402

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
SENSORS = ["VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21", "MODIS"]
FIRES = ["Kitchenfire", "ResortFire", "JunctionFire", "GroveFire", "AirportFire", "BridgeFire"]
MARGIN_DEG = 0.4       # around the fire's cameras: roughly how far they can see
PAUSE_S = 1.5


def camera_box(fire: str) -> str:
    """Bounding box of the fire's camera stations plus a margin. Built from the cameras, not the
    official origin, so the agent's satellite view doesn't leak where the fire really was."""
    def rows(name):
        with open(ROOT / "data" / name) as fh:
            return list(csv.DictReader(l for l in fh if not l.startswith("#")))
    st = {r["station"]: r for r in rows("stations.csv")}
    pts = [(float(st[r["station"]]["lat"]), float(st[r["station"]]["lon"]))
           for r in rows("fires.csv") if r["fire"] == fire and r["station"] in st]
    lats, lons = [p[0] for p in pts], [p[1] for p in pts]
    return (f"{min(lons) - MARGIN_DEG:.3f},{min(lats) - MARGIN_DEG:.3f},"
            f"{max(lons) + MARGIN_DEG:.3f},{max(lats) + MARGIN_DEG:.3f}")


def ignitions() -> dict:
    with open(ROOT / "data" / "ignitions.csv") as fh:
        return {r["fire"]: r for r in csv.DictReader(l for l in fh if not l.startswith("#"))}


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "firefinder-hackathon/0.1 (research)"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(10 * (attempt + 1)); continue
            raise
    return ""


def fetch(key: str, sensor: str, box: str, day: str) -> tuple[str, list[dict]]:
    """Try standard processing, then near-real-time. Returns (source, rows)."""
    for source in (f"{sensor}_SP", f"{sensor}_NRT"):
        text = get(f"{API}/{key}/{source}/{box}/1/{day}")
        time.sleep(PAUSE_S)
        head = text.lstrip()[:200].lower()
        if "invalid" in head and "key" in head:
            raise SystemExit(f"FIRMS rejected the MAP_KEY: {text.strip()[:200]}")
        if not head.startswith("latitude"):
            continue                      # not published for this source/date, or an error message
        rows = list(csv.DictReader(io.StringIO(text)))
        return source, rows               # header present: an empty list means "no hotspots"
    return "", []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="append")
    ap.add_argument("--days", type=int, default=3, help="the fire's day plus N-1 following days")
    args = ap.parse_args()
    key = setting("FIRMS_MAP_KEY")
    ign = ignitions()
    out_dir = ROOT / "data" / "firms"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fire in args.fire or FIRES:
        r = ign[fire]                     # only the date is used from here
        box = camera_box(fire)
        start = dt.date.fromisoformat(r["date"])
        rows, used = [], []
        for i in range(args.days):
            day = (start + dt.timedelta(days=i)).isoformat()
            for sensor in SENSORS:
                source, got = fetch(key, sensor, box, day)
                if source:
                    used.append(f"{day} {source}:{len(got)}")
                for g in got:
                    g["source"] = source
                rows += got
        cols = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument", "confidence",
                "frp", "daynight", "source"]
        with open(out_dir / f"{fire}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(rows, key=lambda g: (g["acq_date"], g["acq_time"])))
        print(f"{fire}: {len(rows)} hotspots  ({'; '.join(used) or 'no source returned data'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

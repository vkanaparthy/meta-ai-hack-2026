"""Do NOAA HMS analyst-drawn smoke polygons cover our fires? (display/scoring only: HMS is drawn
after the fact, so the agent never sees it.)

  python scripts/check_hms_smoke.py

Downloads the daily smoke KML for each fire's date and the day after (UTC), keeps polygons within
--near-km of the official origin, prints a table, and saves them to data/hms/<fire>.geojson.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smokespotter.geo import haversine_m  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/KML"
FIRES = ["Kitchenfire", "ResortFire", "JunctionFire", "GroveFire", "AirportFire", "BridgeFire"]


def fetch(day: dt.date) -> bytes | None:
    cache = ROOT / "data" / "hms" / "raw" / f"hms_smoke{day:%Y%m%d}.kml"
    if cache.exists():
        return cache.read_bytes()
    url = f"{BASE}/{day:%Y}/{day:%m}/hms_smoke{day:%Y%m%d}.kml"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "firefinder-hackathon/0.1 (research)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        print(f"  {url}: HTTP {e.code}")
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    time.sleep(1.0)
    return data


def strip_ns(root):
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def polygons(kml: bytes) -> list[dict]:
    root = strip_ns(ET.fromstring(kml))
    out = []
    def walk(node, folder=""):
        for child in node:
            if child.tag == "Folder":
                walk(child, (child.findtext("name") or folder))
            elif child.tag == "Document":
                walk(child, folder)
            elif child.tag == "Placemark":
                text = " ".join(t for t in [child.findtext("name"), child.findtext("description")] if t)
                for ed in child.iter("Data"):
                    text += f" {ed.get('name')}: {ed.findtext('value')}"
                for sd in child.iter("SimpleData"):
                    text += f" {sd.get('name')}: {sd.text}"
                for ring in child.iter("outerBoundaryIs"):
                    coords = (ring.findtext(".//coordinates") or "").split()
                    pts = [(float(c.split(",")[1]), float(c.split(",")[0])) for c in coords if "," in c]
                    if len(pts) >= 3:
                        out.append({"folder": folder, "info": re.sub(r"<[^>]+>", " ", text), "ring": pts})
    walk(root)
    return out


def inside(lat, lon, ring) -> bool:
    hit = False
    for (y1, x1), (y2, x2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-12) + x1:
            hit = not hit
    return hit


def density(p) -> str:
    s = (p["folder"] + " " + p["info"]).lower()
    for d in ("heavy", "medium", "light"):
        if d in s:
            return d
    m = re.search(r"density\W+(\d+)", s)
    return {"5": "light", "16": "medium", "27": "heavy"}.get(m.group(1), m.group(1)) if m else "?"


def times(p) -> str:
    m = re.findall(r"(?:start|end)\w*\W+([\d :]{4,})", p["info"], flags=re.I)
    return " - ".join(x.strip() for x in m[:2]) if m else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--near-km", type=float, default=15.0)
    args = ap.parse_args()
    with open(ROOT / "data" / "ignitions.csv") as fh:
        ign = {r["fire"]: r for r in csv.DictReader(l for l in fh if not l.startswith("#"))}
    print(f"{'fire':<13}{'HMS day':<10}{'density':<9}{'covers origin':<15}{'nearest km':>11}  times (UTC, HHMM or YYYYDDD HHMM)")
    for fire in FIRES:
        r = ign[fire]
        lat, lon = float(r["lat"]), float(r["lon"])
        d0 = dt.date.fromisoformat(r["date"])
        feats, found = [], False
        for day in (d0, d0 + dt.timedelta(days=1)):
            data = fetch(day)
            if not data:
                continue
            for p in polygons(data):
                near = min(haversine_m(lat, lon, a, b) for a, b in p["ring"]) / 1000
                cov = inside(lat, lon, p["ring"])
                if cov or near <= args.near_km:
                    found = True
                    print(f"{fire:<13}{day:%m-%d}     {density(p):<9}{('yes' if cov else 'no'):<15}"
                          f"{(0.0 if cov else near):>11.1f}  {times(p)}")
                    feats.append({"type": "Feature", "properties": {"day": day.isoformat(), "density": density(p),
                                  "covers_origin": cov, "info": p["info"][:300]},
                                  "geometry": {"type": "Polygon", "coordinates": [[[b, a] for a, b in p["ring"]]]}})
        if not found:
            print(f"{fire:<13}{'-':<10}{'-':<9}{'no polygon within ' + str(int(args.near_km)) + ' km':<26}")
        (ROOT / "data" / "hms" / f"{fire}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    print("\nwrote data/hms/<fire>.geojson (polygons near each fire; display only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

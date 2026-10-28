"""Download GeoNames place names for plain-language alert locations
("6 km NW of Julian") and keep those inside the camera region.

  python scripts/download_places.py           # cities500: towns with 500+ people (~10 MB)
  python scripts/download_places.py --full    # all US + Mexico features: small communities, peaks (~100+ MB)

Data: GeoNames (https://www.geonames.org), CC BY 4.0.
"""
from __future__ import annotations

import argparse
import csv
import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BBOX = (31.5, 34.6, -118.6, -115.2)          # lat_min, lat_max, lon_min, lon_max
KEEP = {("P", None), ("T", "MT"), ("T", "PK")}  # populated places, mountains, peaks


def fetch(name: str) -> bytes:
    url = f"https://download.geonames.org/export/dump/{name}"
    req = urllib.request.Request(url, headers={"User-Agent": "smokespotter-hackathon/0.1"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    files = ["US.zip", "MX.zip"] if args.full else ["cities500.zip"]
    rows = []
    for f in files:
        print(f"downloading {f} ...", flush=True)
        z = zipfile.ZipFile(io.BytesIO(fetch(f)))
        txt = next(n for n in z.namelist() if n.endswith(".txt") and not n.startswith("readme"))
        with z.open(txt) as fh:
            for line in io.TextIOWrapper(fh, encoding="utf-8"):
                c = line.rstrip("\n").split("\t")
                if len(c) < 15:
                    continue
                lat, lon = float(c[4]), float(c[5])
                if not (BBOX[0] <= lat <= BBOX[1] and BBOX[2] <= lon <= BBOX[3]):
                    continue
                cls, code = c[6], c[7]
                if (cls, None) not in KEEP and (cls, code) not in KEEP:
                    continue
                rows.append({"name": c[1], "lat": lat, "lon": lon,
                             "kind": "peak" if cls == "T" else "town",
                             "population": int(c[14] or 0), "country": c[8]})
    out = ROOT / "data" / "places.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "lat", "lon", "kind", "population", "country"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out.relative_to(ROOT)}: {len(rows)} places")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

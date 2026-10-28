"""Download Copernicus GLO-30 elevation tiles around the camera stations and
build one mosaic for skyline calibration (data/dem/region.npz).

  python scripts/download_dem.py               # tiles within 100 km of all stations
  python scripts/download_dem.py --radius-km 80

Data: Copernicus DEM GLO-30, (c) DLR e.V. 2010-2014 and (c) Airbus Defence and
Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_name(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def needed_tiles(radius_km: float) -> list[tuple[int, int]]:
    tiles = set()
    with open(ROOT / "data" / "stations.csv") as fh:
        for r in csv.DictReader(fh):
            lat, lon = float(r["lat"]), float(r["lon"])
            dlat = radius_km / 111.0
            dlon = radius_km / (111.0 * math.cos(math.radians(lat)))
            for la in range(math.floor(lat - dlat), math.floor(lat + dlat) + 1):
                for lo in range(math.floor(lon - dlon), math.floor(lon + dlon) + 1):
                    tiles.add((la, lo))
    return sorted(tiles)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-km", type=float, default=100.0)
    ap.add_argument("--arcsec", type=float, default=3.0, help="mosaic resolution")
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()
    out_dir = ROOT / "data" / "dem"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for la, lo in needed_tiles(args.radius_km):
        name = tile_name(la, lo)
        dest = out_dir / f"{name}.tif"
        if not dest.exists():
            url = f"{BASE}/{name}/{name}.tif"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "smokespotter-hackathon/0.1"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
                print(f"{name}: {len(data) / 1e6:.1f} MB")
                time.sleep(args.delay)
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):          # open ocean: no tile exists
                    print(f"{name}: no tile (ocean)")
                    continue
                raise
        paths.append(dest)

    from smokespotter.dem import DEM

    print(f"Building mosaic from {len(paths)} tiles at {args.arcsec} arcsec...")
    dem = DEM.from_tiles(paths, step_arcsec=args.arcsec)
    dem.save(out_dir / "region.npz")
    print(f"Wrote data/dem/region.npz ({dem.z.shape[0]}x{dem.z.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Refine camera-station positions by fitting all of a station's cameras to the
terrain skyline together (near ridges shift with position, far ones don't).

  python scripts/refine_stations.py --station vo
  python scripts/refine_stations.py --station hp --radius 600

Writes data/stations_refined.csv and updates outputs/<seq>/camera_model.json
for that station's sequences (position + refitted orientation, shared lens).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from smokespotter.camera import CameraModel  # noqa: E402
from smokespotter.dem import DEM, skyline  # noqa: E402
from smokespotter.frames import load_sequence  # noqa: E402
from smokespotter.skyline import SkylineSamples, edge_map, quality, refine_lsq  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAST_M = 10.0
REFINED = ROOT / "data" / "stations_refined.csv"


def rows(name):
    with open(ROOT / "data" / name) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def load_items(station: str):
    items = []
    for r in rows("fires.csv"):
        if r["station"] != station:
            continue
        mpath = ROOT / "outputs" / r["sequence"] / "camera_model.json"
        if not mpath.exists():
            continue
        full = CameraModel.load(mpath)
        frames = load_sequence(ROOT / "data" / "raw" / r["sequence"])
        emap, scale = edge_map([Image.open(f.path) for f in frames[:5]])
        work = replace(full, f_px=full.f_px / scale, width=emap.shape[1], height=emap.shape[0])
        items.append({"seq": r["sequence"], "full": full, "work": work, "emap": emap,
                      "scale": scale, "meta": json.loads(mpath.read_text())})
    return items


def sectors(items, margin=12.0):
    az = []
    for it in items:
        c = it["work"]
        half = c.hfov_deg / 2 + margin
        az.append(np.arange(c.heading_deg - half, c.heading_deg + half, 0.1) % 360)
    return np.unique(np.round(np.concatenate(az), 1))


def evaluate(dem, items, lat, lon, az):
    h = float(dem.elevation(lat, lon)) + MAST_M
    el, dist = skyline(dem, lat, lon, h, az, growth=1.01)
    sky = SkylineSamples(az, el, dist)
    total, fits = 0.0, []
    for it in items:
        c0 = replace(it["work"], lat=lat, lon=lon, height_m=h)
        lo = np.array([c0.heading_deg - 10, -15, -8, c0.f_px - 1e-6, c0.k1 - 1e-6])
        hi = np.array([c0.heading_deg + 10, 15, 8, c0.f_px + 1e-6, c0.k1 + 1e-6])
        c = refine_lsq(c0, sky, it["emap"], lo, hi, windows=(8, 4, 3))
        q = quality(c, sky, it["emap"])
        resid = q["median_abs_px"] if q["median_abs_px"] is not None else 20.0
        total += resid + 5.0 * (1.0 - q["edge_coverage"])
        fits.append((c, q))
    return total / len(items), h, fits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", required=True)
    ap.add_argument("--radius", type=float, default=600.0, help="search radius, metres")
    args = ap.parse_args()
    t0 = time.time()
    dem = DEM.load(ROOT / "data" / "dem" / "region.npz")
    items = load_items(args.station)
    if not items:
        print(f"No calibrated sequences for station {args.station}")
        return 1
    st = {r["station"]: r for r in rows("stations.csv")}[args.station]
    lat0, lon0 = float(st["lat"]), float(st["lon"])
    az = sectors(items)
    kx = 111_000 * math.cos(math.radians(lat0))

    start = (items[0]["full"].lat, items[0]["full"].lon)
    best = (evaluate(dem, items, *start, az), start)
    print(f"{args.station}: {len(items)} camera sequence(s); current position score {best[0][0]:.2f}", flush=True)
    for step, span in ((200.0, args.radius), (60.0, 180.0), (20.0, 60.0)):
        clat, clon = best[1]
        offs = np.arange(-span, span + 1e-9, step)
        for dy in offs:
            for dx in offs:
                lat, lon = clat + dy / 111_000, clon + dx / kx
                if math.hypot((lat - lat0) * 111_000, (lon - lon0) * kx) > args.radius + 1:
                    continue
                res = evaluate(dem, items, lat, lon, az)
                if res[0] < best[0][0]:
                    best = (res, (lat, lon))
        print(f"  step {step:.0f} m: best score {best[0][0]:.2f} at {best[1][0]:.5f}, {best[1][1]:.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    (score, h, fits), (lat, lon) = best
    moved = math.hypot((lat - lat0) * 111_000, (lon - lon0) * kx)
    print(f"Refined {args.station}: {lat:.5f}, {lon:.5f}, camera height {h:.0f} m "
          f"({moved:.0f} m from listed coordinates)")
    for it, (c, q) in zip(items, fits):
        before = it["meta"]["quality"]
        print(f"  {it['seq']:<38} heading {it['work'].heading_deg:7.2f} -> {c.heading_deg:7.2f}  "
              f"resid {before['median_abs_px']} -> {q['median_abs_px']} px, "
              f"coverage {before['edge_coverage']:.0%} -> {q['edge_coverage']:.0%}")
        full = replace(c, f_px=c.f_px * it["scale"], width=it["full"].width, height=it["full"].height)
        meta = {k: v for k, v in it["meta"].items() if k not in CameraModel.__dataclass_fields__}
        meta.update(quality=q, station_refined=True)
        full.save(ROOT / "outputs" / it["seq"] / "camera_model.json", **meta)

    existing = {r["station"]: r for r in csv.DictReader(open(REFINED))} if REFINED.exists() else {}
    existing[args.station] = {"station": args.station, "lat": f"{lat:.6f}", "lon": f"{lon:.6f}",
                              "height_m": f"{h:.1f}", "moved_m": f"{moved:.0f}", "score": f"{score:.3f}",
                              "n_sequences": len(items)}
    with open(REFINED, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(existing.values())).keys()))
        w.writeheader()
        w.writerows(existing.values())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

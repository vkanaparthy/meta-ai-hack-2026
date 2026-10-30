"""What if FireFinder used every camera in the Cal OES / ALERTCalifornia network?

For each replayed fire: which cameras within --range-km have a terrain-clear line of sight to a
plume --plume-m above the official origin, and how precise a bearing fix (90% radius) would our
cameras, the best 2 or 3 network cameras, and all visible cameras give, with the same --sigma-deg
bearing error everywhere. Geometry only: it assumes each camera spots and calibrates the plume.
It uses the official origins, so it is an offline study, never an input to the agent.
Visibility check: with a 500 m plume, the line-of-sight model predicts 14 of the 16 camera views of
our fires that really saw smoke (its one false "visible" is Bridge, hidden by cloud).

  python scripts/download_caloes_cameras.py
  python scripts/network_whatif.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smokespotter.dem import DEM  # noqa: E402
from smokespotter.geo import haversine_m  # noqa: E402
from smokespotter.network import best_subset, fix_radius_m, line_of_sight  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIRES = ["Kitchenfire", "ResortFire", "JunctionFire", "GroveFire", "AirportFire", "BridgeFire"]
OUR_RESULT = {"Kitchenfire": "confirmed, 119 m", "ResortFire": "confirmed, 1.0 km", "JunctionFire": "confirmed, 1.9 km",
              "GroveFire": "no pin", "AirportFire": "no pin", "BridgeFire": "missed"}
MAST_M = 10.0


def rows(name):
    with open(ROOT / "data" / name) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def network_cameras(type_filter: str) -> list[dict]:
    gj = json.loads((ROOT / "data" / "cameras" / "caloes_webcams.geojson").read_text())
    cams = []
    for f in gj["features"]:
        g, p = f.get("geometry") or {}, f.get("properties") or {}
        if g.get("type") != "Point":
            continue
        kind = str(p.get("CameraType") or "").strip()
        if type_filter and not any(t.strip().lower() == kind.lower() for t in type_filter.split(",")):
            continue
        lon, lat = g["coordinates"][:2]
        name = str(p.get("Location") or p.get("Camera_ID") or "")
        cams.append({"lat": float(lat), "lon": float(lon), "kind": kind, "name": name})
    # one entry per mast: cameras on the same site share a viewpoint
    sites = []
    for c in cams:
        if not any(haversine_m(c["lat"], c["lon"], s["lat"], s["lon"]) < 300 for s in sites):
            sites.append(c)
    return sites


def fmt(r):
    return "-" if r == math.inf else (f"{r / 1000:.1f} km" if r >= 1000 else f"{r:.0f} m")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--range-km", type=float, default=45.0)
    ap.add_argument("--plume-m", type=float, default=500.0,
                    help="plume height for visibility; 500 m reproduces 14 of 16 real camera sightings of our fires")
    ap.add_argument("--sigma-deg", type=float, default=1.0)
    ap.add_argument("--type", default="ALERTCalifornia,Fire",
                    help="CameraType values to keep, comma-separated ('' = all, including traffic cameras)")
    args = ap.parse_args()
    dem = DEM.load(ROOT / "data" / "dem" / "region.npz")
    ign = {r["fire"]: r for r in rows("ignitions.csv")}
    st = {r["station"]: r for r in rows("stations.csv")}
    net = network_cameras(args.type)
    print(f"{len(net)} network camera sites (type filter: {args.type!r})\n")

    def visible(cams, tgt, z_tgt):
        out = []
        for c in cams:
            d = haversine_m(c["lat"], c["lon"], *tgt)
            if d > args.range_km * 1000:
                continue
            # listed coordinates are approximate; cameras sit on summits, as in calibration
            la, lo, zc, _ = dem.snap_to_summit(c["lat"], c["lon"])
            h = float(zc) + MAST_M
            if not math.isnan(h) and line_of_sight(dem, la, lo, h, *tgt, z_tgt + args.plume_m):
                out.append({**c, "km": round(d / 1000, 1)})
        return out

    report = {}
    print(f"{'fire':<13}{'our result':<19}{'our cams in view':<18}{'our fix':>9}   {'network in view':<16}"
          f"{'best 2':>9}{'best 3':>9}{'all':>9}")
    for fire in FIRES:
        r = ign[fire]
        tgt = (float(r["lat"]), float(r["lon"]))
        z = float(dem.elevation(*tgt))
        ours_all = [{"lat": float(st[s]["lat"]), "lon": float(st[s]["lon"]), "name": s}
                    for s in sorted({x["station"] for x in rows("fires.csv") if x["fire"] == fire}) if s in st]
        ours = visible(ours_all, tgt, z)
        seen = visible(net, tgt, z)
        pts = [(c["lat"], c["lon"]) for c in seen]
        r_ours = fix_radius_m([(c["lat"], c["lon"]) for c in ours], tgt, args.sigma_deg)
        i2, r2 = best_subset(pts, tgt, 2, args.sigma_deg)
        i3, r3 = best_subset(pts, tgt, 3, args.sigma_deg)
        r_all = fix_radius_m(pts, tgt, args.sigma_deg)
        report[fire] = {"our_result": OUR_RESULT[fire], "our_stations": [c["name"] for c in ours_all],
                        "our_stations_in_view": [c["name"] for c in ours], "our_fix_m": r_ours,
                        "network_sites_in_view": len(seen), "best2_m": r2, "best3_m": r3, "all_m": r_all,
                        "best3": [seen[i] for i in (i3 or [])]}
        print(f"{fire:<13}{OUR_RESULT[fire]:<19}{f'{len(ours)} of {len(ours_all)}':<18}{fmt(r_ours):>9}   "
              f"{len(seen):<16}{fmt(r2):>9}{fmt(r3):>9}{fmt(r_all):>9}")
    out = ROOT / "outputs" / "network_whatif.json"
    out.write_text(json.dumps(report, indent=2, default=lambda v: None if v == math.inf else v))
    print(f"\nfix = 90% error radius with {args.sigma_deg} deg bearing error, plume {args.plume_m:.0f} m above the "
          f"origin, cameras within {args.range_km:.0f} km with a clear line of sight. Geometry only.\nwrote {out.relative_to(ROOT)}")
    for fire, v in report.items():
        names = ", ".join(f"{c['name'] or '?'} ({c['km']} km)" for c in v["best3"])
        print(f"  {fire}: best 3 network cameras: {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

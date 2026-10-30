"""Cameras vs satellites: when did FireFinder alert, and when did the first NASA FIRMS hotspot
appear near the fire?

  python scripts/compare_firms.py

Uses the live Muse Spark 1.3 replays (outputs/<fire>_timeline.json) and data/firms/<fire>.csv.
A hotspot counts if it lies within --radius-km of the official origin (scoring only) and is
acquired no earlier than --before-min minutes before the official start (official start times are
often report times: Suomi NPP saw the Airport Fire 8 minutes before its listed start). Writes outputs/firms_comparison.json.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smokespotter.geo import haversine_m  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ZoneInfo("America/Los_Angeles")
FIRES = ["Kitchenfire", "ResortFire", "JunctionFire", "GroveFire", "AirportFire", "BridgeFire"]


def official_start(r: dict) -> int | None:
    """Epoch of 'start HH:MM' (local) in the ignitions notes, if given."""
    import re
    m = re.search(r"start (\d{1,2}):(\d{2})", r.get("notes", ""))
    if not m:
        return None
    d = dt.date.fromisoformat(r["date"])
    return int(dt.datetime(d.year, d.month, d.day, int(m[1]), int(m[2]), tzinfo=LOCAL).timestamp())


def hhmm(epoch: int) -> str:
    return dt.datetime.fromtimestamp(epoch, LOCAL).strftime("%a %H:%M %Z")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-km", type=float, default=5.0)
    ap.add_argument("--before-min", type=float, default=60.0)
    args = ap.parse_args()
    with open(ROOT / "data" / "ignitions.csv") as fh:
        ign = {r["fire"]: r for r in csv.DictReader(l for l in fh if not l.startswith("#"))}
    out = {}
    print(f"{'fire':<13}{'official start':<17}{'FireFinder first alert':<26}{'located':<27}{'first satellite hotspot':<30}{'lead':>8}")
    for fire in FIRES:
        r = ign[fire]
        lat, lon = float(r["lat"]), float(r["lon"])
        t0 = official_start(r)
        tl = ROOT / "outputs" / f"{fire}_timeline.json"
        alerts = json.loads(tl.read_text())["alerts"] if tl.exists() else []
        first = alerts[0] if alerts else None
        loc = next((a for a in alerts if a.get("location")), None)
        hot = []
        fp = ROOT / "data" / "firms" / f"{fire}.csv"
        if fp.exists():
            for h in csv.DictReader(open(fp)):
                at = dt.datetime.strptime(h["acq_date"] + h["acq_time"].zfill(4), "%Y-%m-%d%H%M").replace(
                    tzinfo=dt.timezone.utc)
                e = int(at.timestamp())
                d = haversine_m(float(h["latitude"]), float(h["longitude"]), lat, lon)
                if d <= args.radius_km * 1000 and (t0 is None or e >= t0 - args.before_min * 60):
                    hot.append((e, round(d), h))
        hot.sort(key=lambda x: x[0])
        sat = hot[0] if hot else None
        # GOES-18 (geostationary): first fresh fire pixel near the origin, by its publication time
        goes_first = None
        gp = ROOT / "data" / "goes" / f"{fire}.csv"
        if gp.exists():
            g = list(csv.DictReader(open(gp)))
            if g:
                px = lambda r: (round(float(r["lat"]), 2), round(float(r["lon"]), 2))
                cutoff = (t0 - args.before_min * 60) if t0 else min(int(r["scan_start"]) for r in g)
                base = {px(r) for r in g if int(r["scan_start"]) < cutoff}    # already hot before the fire
                for r in sorted(g, key=lambda r: int(r["scan_start"])):
                    d = haversine_m(float(r["lat"]), float(r["lon"]), lat, lon)
                    if px(r) not in base and d <= (args.radius_km + 3) * 1000:
                        goes_first = (int(r["scan_start"]), int(r["available"]), round(d / 1000, 1))
                        break
        ref = loc or first
        lead_h = round((sat[0] - ref["epoch"]) / 3600, 2) if (sat and ref) else None
        out[fire] = {
            "official_start": hhmm(t0) if t0 else None,
            "firefinder_first_alert": f"{hhmm(first['epoch'])} {first['level']}" if first else None,
            "firefinder_located": f"{hhmm(loc['epoch'])} {loc['level']}" if loc else None,
            "first_hotspot": hhmm(sat[0]) if sat else None,
            "first_hotspot_sensor": f"{sat[2]['satellite']} {sat[2]['instrument']}" if sat else None,
            "first_hotspot_km_from_origin": round(sat[1] / 1000, 1) if sat else None,
            "hotspots_within_radius": len(hot),
            "lead_hours_vs_first_hotspot": lead_h,
            "firms_file": fp.exists(),
            "goes_first_scan": hhmm(goes_first[0]) if goes_first else None,
            "goes_published": hhmm(goes_first[1]) if goes_first else None,
            "goes_km_from_origin": goes_first[2] if goes_first else None,
            "goes_file": gp.exists(),
        }
        o = out[fire]
        print(f"{fire:<13}{(o['official_start'] or '-'):<17}{(o['firefinder_first_alert'] or 'none'):<26}"
              f"{(o['firefinder_located'] or 'not located'):<27}"
              f"{((o['first_hotspot'] or ('none' if fp.exists() else 'no data')) + (' ' + o['first_hotspot_sensor'] if sat else '')):<30}"
              f"{(f'{lead_h * 60:+.0f} min' if lead_h is not None and abs(lead_h) < 1 else (f'{lead_h:+.1f} h' if lead_h is not None else '-')):>9}"
              f"   GOES-18: {(o['goes_first_scan'] + ' (published ' + o['goes_published'] + ')') if goes_first else ('none in replay window' if gp.exists() else 'no data')}")
    (ROOT / "outputs" / "firms_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\nlead = first satellite hotspot minus FireFinder's first located alert (or first alert); "
          f"positive means FireFinder was earlier.\nwrote outputs/firms_comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

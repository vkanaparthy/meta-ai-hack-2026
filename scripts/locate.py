"""Locate a fire from SAM tracks in several cameras.

  python scripts/locate.py --fire JunctionFire
  python scripts/locate.py --fire ResortFire --sigma 1.5

Needs run_sam.py results for the fire's cameras and calibrated cameras in
data/cameras.csv. Prints when the first fix was possible, where, its
uncertainty, and the distance to the official ignition point if known.
Writes outputs/<fire>_location.geojson.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smokespotter.calibrate import NOMINAL  # noqa: E402
from smokespotter.frames import load_sequence  # noqa: E402
from smokespotter.geo import haversine_m, pixel_to_bearing  # noqa: E402
from smokespotter.locate import TrackBearing, best_location, first_location, to_geojson  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def rows(name: str) -> list[dict]:
    path = ROOT / "data" / name
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", required=True)
    ap.add_argument("--sigma", type=float, default=1.0, help="bearing uncertainty, degrees")
    ap.add_argument("--max-resid", type=float, default=400.0, help="metres, line agreement threshold")
    args = ap.parse_args()

    stations = {r["station"]: r for r in rows("stations.csv")}
    cams = {r["camera_id"]: r for r in rows("cameras.csv")}
    # A station's mount rotation applies to all its cameras: derive headings for
    # cameras at calibrated stations that calibration never saw directly.
    offsets, hfovs = {}, []
    for c in cams.values():
        if c["direction"] in NOMINAL:
            off = (float(c["heading_deg"]) - NOMINAL[c["direction"]] + 180) % 360 - 180
            offsets.setdefault(c["station"], off)
            hfovs.append(float(c["hfov_deg"]))
    ignition = {r["fire"]: r for r in rows("ignitions.csv")}.get(args.fire)
    seqs = [r for r in rows("fires.csv") if r["fire"] == args.fire]
    if not seqs:
        print(f"{args.fire} not in data/fires.csv")
        return 1

    tracks, first_visible, skipped, calib_fires = [], [], [], set()
    for r in seqs:
        cam = f"{r['station']}-{r['direction']}"
        if cam not in cams and r["station"] in offsets and r["direction"] in NOMINAL and hfovs:
            cams[cam] = {"camera_id": cam, "station": r["station"], "direction": r["direction"],
                         "heading_deg": (NOMINAL[r["direction"]] + offsets[r["station"]]) % 360,
                         "hfov_deg": hfovs[0], "notes": "derived from station offset"}
        res_path = ROOT / "outputs" / r["sequence"] / "results.json"
        if cam not in cams or not res_path.exists() or r["station"] not in stations:
            skipped.append(f"{cam} ({'not calibrated' if cam not in cams else 'no SAM results'})")
            continue
        c, st = cams[cam], stations[r["station"]]
        calib_fires.update(c.get("notes", "").replace("fires:", "").split(";")[0].strip().split("+"))
        frames = load_sequence(ROOT / "data" / "raw" / r["sequence"])
        first_visible += [f.epoch for f in frames if f.offset >= 0][:1]
        res = json.loads(res_path.read_text())
        dets = {}
        for d in res["detections"]:
            dets.setdefault(d["object_id"], []).append(d)
        for t in res.get("track_summaries", []):
            ds = sorted(dets[t["object_id"]], key=lambda d: d["frame"])
            x = statistics.median(t["base_x_first3"])
            bearing = pixel_to_bearing(x, int(res["image_width"]), float(c["heading_deg"]), float(c["hfov_deg"]))
            avail = frames[ds[min(2, len(ds) - 1)]["frame"]].epoch
            tracks.append(TrackBearing(cam, r["station"], t["object_id"], float(st["lat"]), float(st["lon"]),
                                       bearing, avail, bool(t["edge_clipped"])))

    if args.fire in calib_fires:
        print(f"NOTE: {args.fire} was used to calibrate these cameras, so location accuracy here is "
              f"not an independent test.\n")
    print(f"{len(tracks)} plume track(s) from {len({t.camera_id for t in tracks})} camera(s)")
    for s in skipped:
        print(f"  skipped {s}")
    t0 = min(first_visible) if first_visible else None

    def when(epoch):
        if t0 is None:
            return dt.datetime.utcfromtimestamp(epoch).strftime("%H:%M UTC")
        return f"+{(epoch - t0) / 60:.1f} min after first visible smoke"

    kw = dict(max_resid_m=args.max_resid, sigma_deg=args.sigma)
    first = first_location(tracks, **kw)
    final = best_location(tracks, **kw)
    if final is None:
        print("No consistent fix (need plumes from at least two stations whose bearings agree).")
        return 1
    truth = (float(ignition["lat"]), float(ignition["lon"])) if ignition else None
    for label, loc in (("First fix", first), ("Best fix", final)):
        if loc is None:
            continue
        line = (f"{label}: {loc.fix.lat:.5f}, {loc.fix.lon:.5f}  ({when(loc.epoch)})\n"
                f"  cameras: {', '.join(m.camera_id + ('*' if m.edge_clipped else '') for m in loc.members)}\n"
                f"  uncertainty (90%): {loc.uncertainty_m:,.0f} m; lines agree within {loc.fix.error_m:,.0f} m")
        if truth:
            line += f"\n  distance to official ignition point: {haversine_m(loc.fix.lat, loc.fix.lon, *truth):,.0f} m"
        print(line)
    unused = [t for t in tracks if t not in final.members]
    if unused:
        print("Tracks not matched to this fire: " +
              ", ".join(f"{t.camera_id} {t.track_id} ({t.bearing_deg:.1f} deg)" for t in unused))
    print("(* = plume clipped at image edge)")
    out = ROOT / "outputs" / f"{args.fire}_location.geojson"
    out.write_text(json.dumps(to_geojson(final, truth), indent=2))
    print(f"Wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

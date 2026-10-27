"""Fit camera headings and field of view from fires with known origins.

Uses the plume base column SAM found in each camera (outputs/<seq>/results.json,
from run_sam.py) and the official ignition point (data/ignitions.csv), then
writes data/cameras.csv.

  python scripts/calibrate.py --fire JunctionFire
  python scripts/calibrate.py --fire JunctionFire --fire Kitchenfire
  python scripts/calibrate.py --fire JunctionFire --manual hp-e=2950 --manual vo-n=410

--manual sets a camera's plume column by hand (read it off an image), which
overrides SAM. Keep the fires you report location accuracy on OUT of
calibration, or the result is circular.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from smokespotter.calibrate import CalObs, NOMINAL, _pixel_angle, camera_table, fit  # noqa: E402
from smokespotter.frames import load_sequence  # noqa: E402
from smokespotter.geo import bearing_deg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def rows(name: str) -> list[dict]:
    with open(ROOT / "data" / name) as fh:
        return [r for r in csv.DictReader(l for l in fh if not l.startswith("#"))]


def plume_x(seq: str, min_frames: int = 5) -> tuple[float | None, bool]:
    """(median plume-base column over the first 3 frames, edge_clipped) for the
    fire's plume: the earliest track that starts AFTER ignition (a track already
    present before ignition can't be this fire) and persists min_frames."""
    path = ROOT / "outputs" / seq / "results.json"
    if not path.exists():
        return None, False
    res = json.loads(path.read_text())
    sums = res.get("track_summaries")
    if sums:
        frames = load_sequence(ROOT / "data" / "raw" / seq)
        new = [t for t in sums if frames[t["first_frame"]].offset >= 0 and t["n_frames"] >= min_frames]
        if not new:
            return None, False
        t = min(new, key=lambda t: (t["first_frame"], -t["n_frames"]))
        return statistics.median(t["base_x_first3"]), bool(t["edge_clipped"])
    dets = res.get("detections", [])
    tracks: dict[str, list[dict]] = {}
    for d in dets:
        tracks.setdefault(d["object_id"], []).append(d)
    if not tracks:
        return None, False
    best = max(tracks.values(), key=len)
    first = sorted(best, key=lambda d: d["frame"])[:3]
    return statistics.median(d["base_x"] for d in first), False


def image_width(seq: str) -> int | None:
    frames = load_sequence(ROOT / "data" / "raw" / seq) if (ROOT / "data" / "raw" / seq).exists() else []
    return Image.open(frames[0].path).width if frames else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="append", required=True)
    ap.add_argument("--manual", action="append", default=[], help="camera=column, e.g. hp-e=2950")
    ap.add_argument("--default-hfov", type=float, default=100.0,
                    help="used only if no station has two views of a fire")
    ap.add_argument("--group", choices=["station", "camera"], default="station",
                    help="one rotation per station (default) or per camera")
    ap.add_argument("--dry-run", action="store_true", help="print the fit but don't write cameras.csv")
    ap.add_argument("--skip-edge", action="store_true",
                    help="ignore views where the plume is clipped at the image edge")
    args = ap.parse_args()
    manual = {k: float(v) for k, v in (m.split("=") for m in args.manual)}

    stations = {r["station"]: r for r in rows("stations.csv")}
    ignitions = {r["fire"]: r for r in rows("ignitions.csv")}
    fires = rows("fires.csv")

    obs, notes, all_cams, edges = [], [], [], set()
    for fire in args.fire:
        if fire not in ignitions:
            print(f"{fire}: no entry in data/ignitions.csv, skipping", file=sys.stderr)
            continue
        ig = ignitions[fire]
        for r in (r for r in fires if r["fire"] == fire):
            if r["direction"] not in NOMINAL or r["station"] not in stations:
                continue
            cam = f"{r['station']}-{r['direction']}"
            all_cams.append(cam)
            st = stations[r["station"]]
            true_b = bearing_deg(float(st["lat"]), float(st["lon"]), float(ig["lat"]), float(ig["lon"]))
            width = image_width(r["sequence"])
            sam_x, edge = plume_x(r["sequence"])
            x = manual.get(cam, sam_x)
            if cam in manual:
                edge = False
            if edge:
                edges.add(cam)
                if args.skip_edge:
                    notes.append(f"  {cam:<10} skipped: plume clipped at image edge")
                    continue
            if width is None or x is None:
                notes.append(f"  {cam:<10} true bearing {true_b:6.1f}  (no {'frames' if width is None else 'plume column'} yet)")
                continue
            obs.append(CalObs(cam, x, width, true_b))

    if not obs:
        print("No usable observations. Run run_sam.py on these fires first, or pass --manual.")
        for n in notes:
            print(n)
        return 1

    hfov, offsets, rms, identified = fit(obs, default_hfov=args.default_hfov, group=args.group)
    print(f"Field of view: {hfov:.1f} deg ({'fitted' if identified else 'assumed, not identifiable from this data'})")
    print(f"RMS bearing error: {rms:.2f} deg over {len(obs)} views\n")
    print(f"  {'camera':<10} {'column':>7} {'true':>7} {'heading':>8} {'error':>6}")
    table = {c.camera_id: c for c in camera_table(obs, hfov, offsets, all_cams, args.group)}
    for o in obs:
        c = table[o.camera_id]
        pred = (c.heading_deg + _pixel_angle(o.x, o.width, hfov)) % 360
        err = (pred - o.true_bearing + 180) % 360 - 180
        flag = "  edge" if o.camera_id in edges else ""
        print(f"  {o.camera_id:<10} {o.x:7.0f} {o.true_bearing:7.1f} {c.heading_deg:8.1f} {err:+6.2f}{flag}")
    for n in notes:
        print(n)

    if args.dry_run:
        return 0
    # merge into data/cameras.csv, keeping cameras calibrated by other runs
    path = ROOT / "data" / "cameras.csv"
    existing = {r["camera_id"]: r for r in rows("cameras.csv")} if path.exists() else {}
    widths = {o.camera_id: o.width for o in obs}
    for cam, c in table.items():
        st, d = cam.rsplit("-", 1)
        existing[cam] = {
            "camera_id": cam, "station": st, "direction": d,
            "heading_deg": c.heading_deg, "hfov_deg": c.hfov_deg,
            "image_width_px": widths.get(cam, existing.get(cam, {}).get("image_width_px", "")),
            "calibrated": "yes" if c.n_obs else "station-offset",
            "notes": f"fires: {'+'.join(args.fire)}; {args.group} fit; residual {c.residual_deg} deg" if c.n_obs
                     else f"offset from other cameras at {st}",
        }
    with open(path, "w", newline="") as fh:
        fh.write("# heading_deg = compass bearing of the image centre column; hfov_deg = horizontal field of view.\n"
                 "# Written by scripts/calibrate.py from fires with known ignition points.\n")
        w = csv.DictWriter(fh, fieldnames=["camera_id", "station", "direction", "heading_deg", "hfov_deg",
                                           "image_width_px", "calibrated", "notes"])
        w.writeheader()
        for cam in sorted(existing):
            w.writerow(existing[cam])
    print(f"\nWrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Calibrate each camera sequence against the terrain skyline (no fires needed).

  python scripts/calibrate_skyline.py --all
  python scripts/calibrate_skyline.py --fire ResortFire --fire JunctionFire
  python scripts/calibrate_skyline.py data/raw/20241108_ResortFire_*

Pass 1 fits heading, pitch, roll, focal length and distortion per sequence.
Pass 2 (default when 3+ good fits) fixes the lens to the median of the good
fits (same camera model everywhere) and refits orientation only.
Writes outputs/<sequence>/camera_model.json and skyline_fit.jpg (check it!).

Stages can be run separately (each is resumable):
  --stage 1      per-sequence free-lens fits      (saves camera_model_free.json)
  --stage joint  shared lens from the good stage-1 fits (saves data/lens.json)
  --stage 2      orientation-only refits with the shared lens
Needs data/dem/region.npz from scripts/download_dem.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from smokespotter.calibrate import NOMINAL  # noqa: E402
from smokespotter.camera import CameraModel  # noqa: E402
from smokespotter.dem import DEM, skyline  # noqa: E402
from smokespotter.frames import load_sequence  # noqa: E402
from smokespotter.skyline import SkylineSamples, edge_map, fit, fit_shared_lens, predicted_rows  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAST_M = 10.0
HFOV_PRIOR = 104.0


def rows(name):
    with open(ROOT / "data" / name) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def station_skyline(dem: DEM, station: dict, cache_dir: Path):
    """Snap the station to the nearby summit and render its 360-degree skyline (cached)."""
    lat, lon, z, moved = dem.snap_to_summit(float(station["lat"]), float(station["lon"]))
    h = z + MAST_M
    path = cache_dir / f"{station['station']}.npz"
    if path.exists():
        d = np.load(path)
        if np.allclose([d["lat"], d["lon"], d["h"]], [lat, lon, h]):
            return SkylineSamples(d["az"], d["el"], d["dist"]), (lat, lon, h, moved)
    az = np.arange(0, 360, 0.05)
    el, dist = skyline(dem, lat, lon, h, az)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, az=az, el=el, dist=dist, lat=lat, lon=lon, h=h)
    return SkylineSamples(az, el, dist), (lat, lon, h, moved)


def calibrate(seq: dict, dem: DEM, stations: dict, lens=None) -> dict | None:
    folder = ROOT / "data" / "raw" / seq["sequence"]
    frames = load_sequence(folder)
    if len(frames) < 3 or seq["direction"] not in NOMINAL or seq["station"] not in stations:
        return None
    sky, (lat, lon, h, moved) = station_skyline(dem, stations[seq["station"]], ROOT / "outputs" / "skyline")
    imgs = [Image.open(f.path) for f in frames[:5]]          # pre-ignition frames
    W, H = imgs[0].size
    emap, scale = edge_map(imgs)
    w_work, h_work = emap.shape[1], emap.shape[0]
    if lens:
        f0, k1, k2 = lens["f_norm"] * w_work, lens["k1"], lens.get("k2", 0.0)
    else:
        f0, k1, k2 = (w_work / 2) / np.tan(np.radians(HFOV_PRIOR / 2)), 0.0, 0.0
    base = CameraModel(lat, lon, h, NOMINAL[seq["direction"]], 0.0, 0.0, f0, k1, w_work, h_work, k2=k2)
    cam, q = fit(emap, sky, base, fix_lens=lens is not None)

    out = ROOT / "outputs" / seq["sequence"]
    out.mkdir(parents=True, exist_ok=True)
    img = imgs[0].convert("RGB").resize((w_work, h_work))
    xs = np.arange(0, w_work, 1.0)
    yp = predicted_rows(cam, sky, xs)
    pts = [(x, y) for x, y in zip(xs, yp) if np.isfinite(y)]
    ImageDraw.Draw(img).line(pts, fill=(255, 0, 255), width=2)
    img.save(out / "skyline_fit.jpg", quality=88)

    full = replace(cam, f_px=cam.f_px * scale, width=W, height=H)
    meta = dict(quality=q, station_moved_m=round(moved), lens="shared" if lens else "free",
                f_norm=cam.f_px / w_work)
    full.save(out / "camera_model.json", **meta)
    if not lens:
        full.save(out / "camera_model_free.json", **meta)
    return {"seq": seq, "cam": full, "q": q, "moved": moved, "f_norm": cam.f_px / w_work,
            "emap": emap, "sky": sky, "cam_work": cam}


def report(r: dict) -> str:
    c, q = r["cam"], r["q"]
    cam_id = f"{r['seq']['station']}-{r['seq']['direction']}"
    flag = "" if good(r) else "  <-- weak fit, check skyline_fit.jpg"
    return (f"  {r['seq']['fire']:<14} {cam_id:<7} heading {c.heading_deg:6.2f}  pitch {c.pitch_deg:+5.2f}  "
            f"roll {c.roll_deg:+5.2f}  fov {c.hfov_deg:6.2f}  k1 {c.k1:+.3f} k2 {c.k2:+.3f}  "
            f"resid {q['median_abs_px']} px  coverage {q['edge_coverage']:.0%}{flag}")


def good(r: dict) -> bool:
    q = r["q"]
    return q["median_abs_px"] is not None and q["median_abs_px"] <= 2.0 and q["edge_coverage"] >= 0.6


def load_free_fit(seq: dict, dem: DEM, stations: dict) -> dict | None:
    """Rebuild a stage-1 result (edge map, skyline, working-res camera) from disk."""
    path = ROOT / "outputs" / seq["sequence"] / "camera_model_free.json"
    if not path.exists():
        return None
    meta = json.loads(path.read_text())
    full = CameraModel.load(path)
    frames = load_sequence(ROOT / "data" / "raw" / seq["sequence"])
    emap, scale = edge_map([Image.open(f.path) for f in frames[:5]])
    sky, _ = station_skyline(dem, stations[seq["station"]], ROOT / "outputs" / "skyline")
    work = replace(full, f_px=full.f_px / scale, width=emap.shape[1], height=emap.shape[0])
    return {"seq": seq, "cam": full, "q": meta["quality"], "moved": meta["station_moved_m"],
            "f_norm": meta["f_norm"], "emap": emap, "sky": sky, "cam_work": work}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="*", type=Path)
    ap.add_argument("--fire", action="append", default=[])
    ap.add_argument("--all", action="store_true", help="every downloaded sequence in fires.csv")
    ap.add_argument("--free-lens", action="store_true", help="stop after stage 1")
    ap.add_argument("--stage", choices=["all", "1", "joint", "2"], default="all")
    args = ap.parse_args()

    dem_path = ROOT / "data" / "dem" / "region.npz"
    if not dem_path.exists():
        print("Missing data/dem/region.npz: run scripts/download_dem.py first")
        return 1
    dem = DEM.load(dem_path)
    stations = {r["station"]: r for r in rows("stations.csv")}
    names = {f.name for f in args.folders}
    seqs = [r for r in rows("fires.csv")
            if (ROOT / "data" / "raw" / r["sequence"]).exists()
            and (args.all or r["fire"] in args.fire or r["sequence"] in names)]
    if not seqs:
        print("No matching downloaded sequences.")
        return 1

    results = []
    if args.stage in ("all", "1"):
        print(f"Stage 1: free lens, {len(seqs)} sequence(s)")
        for s in seqs:
            r = calibrate(s, dem, stations)
            if r:
                results.append(r)
                print(report(r), flush=True)
                if r["moved"] > 300:
                    print(f"    note: station position snapped {r['moved']:.0f} m to the nearby summit")
        if args.stage == "1" or args.free_lens:
            return 0

    lens_path = ROOT / "data" / "lens.json"
    if args.stage in ("all", "joint"):
        if not results:
            results = [r for r in (load_free_fit(s, dem, stations) for s in seqs) if r]
        good_fits = [r for r in results if good(r)]
        # drop lens outliers (f/k1 trade-off gone wrong) before pooling
        med = float(np.median([r["cam"].hfov_deg for r in good_fits])) if good_fits else 0.0
        for r in [r for r in good_fits if abs(r["cam"].hfov_deg - med) > 6.0]:
            print(f"  excluding {r['seq']['sequence']} from lens fit (fov {r['cam'].hfov_deg:.1f} vs median {med:.1f})")
        good_fits = [r for r in good_fits if abs(r["cam"].hfov_deg - med) <= 6.0]
        if len(good_fits) < 3:
            print(f"Only {len(good_fits)} good stage-1 fits; need 3 for a shared lens.")
            return 1
        print(f"\nJoint lens fit over {len(good_fits)} good sequences (focal length, k1, k2 shared)...", flush=True)
        lens, _ = fit_shared_lens([(r["emap"], r["sky"], r["cam_work"]) for r in good_fits])
        print(f"Shared lens: fov {lens['hfov_deg']:.2f} deg, k1 {lens['k1']:+.3f}, k2 {lens['k2']:+.3f}")
        lens_path.write_text(json.dumps({**lens, "n_fits": len(good_fits),
                                         "sequences": [r["seq"]["sequence"] for r in good_fits]}, indent=2))
        if args.stage == "joint":
            return 0

    if not lens_path.exists():
        print("No data/lens.json yet: run --stage joint first")
        return 1
    lens = json.loads(lens_path.read_text())
    print(f"Stage 2: shared lens (fov {lens['hfov_deg']:.2f}), orientation only")
    for s in seqs:
        r = calibrate(s, dem, stations, lens=lens)
        if r:
            print(report(r), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

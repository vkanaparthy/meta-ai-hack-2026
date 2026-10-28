"""Diagnostic: per camera, bearing error of the fire's own plume track vs the
official ignition point, under the free-lens and shared-lens calibrations.
The plume track = earliest track starting after ignition (diagnostic only)."""
import csv, json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smokespotter.camera import CameraModel
from smokespotter.frames import load_sequence
from smokespotter.geo import bearing_deg

ROOT = Path(__file__).resolve().parent.parent
ign = {r["fire"]: r for r in csv.DictReader(open(ROOT / "data/ignitions.csv"))}
print(f"{'sequence':<34} {'x%':>4} {'free err':>9} {'(resid)':>8} {'shared err':>10} {'(resid)':>8}")
for r in csv.DictReader(open(ROOT / "data/fires.csv")):
    out = ROOT / "outputs" / r["sequence"]
    if r["fire"] not in ign or not (out / "results.json").exists() or not (out / "camera_model_free.json").exists():
        continue
    res = json.loads((out / "results.json").read_text())
    frames = load_sequence(ROOT / "data/raw" / r["sequence"])
    cands = [t for t in res["track_summaries"] if frames[t["first_frame"]].offset >= 0 and t["n_frames"] >= 5]
    if not cands:
        continue
    t = min(cands, key=lambda t: t["first_frame"])
    ds = sorted([d for d in res["detections"] if d["object_id"] == t["object_id"]], key=lambda d: d["frame"])[:3]
    x = statistics.median(t["base_x_first3"]); y = statistics.median(d["y1"] - 1 for d in ds)
    row = []
    for name in ("camera_model_free.json", "camera_model.json"):
        m = CameraModel.load(out / name); meta = json.loads((out / name).read_text())
        b = float(m.pixel_to_azel(x, y)[0])
        tb = bearing_deg(m.lat, m.lon, float(ign[r["fire"]]["lat"]), float(ign[r["fire"]]["lon"]))
        row += [(b - tb + 180) % 360 - 180, meta["quality"]["median_abs_px"]]
    print(f"{r['sequence']:<34} {x / res['image_width']:4.0%} {row[0]:+9.2f} {row[1]:>8} {row[2]:+10.2f} {row[3]:>8}")

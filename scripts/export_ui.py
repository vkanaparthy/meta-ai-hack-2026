"""Package agent timelines for the replay page (ui/index.html).

  python scripts/export_ui.py                     # every fire with a timeline
  python scripts/export_ui.py --fire Kitchenfire

Writes ui/data/fires.json, ui/data/<fire>.json and camera thumbnails under
ui/data/<fire>/<camera>/. Serve with:  python -m http.server -d ui 8000
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from smokespotter.camera import CameraModel  # noqa: E402
from smokespotter.desk import hhmm  # noqa: E402
from smokespotter.frames import load_sequence  # noqa: E402
from smokespotter.observations import load_fire  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "ui" / "data"
THUMB_W = 480
LEAD_S = 10 * 60          # replay starts this long before the first visible smoke


def rows(name):
    with open(ROOT / "data" / name) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def timeline_path(fire: str) -> Path | None:
    for name in (f"{fire}_timeline.json", f"{fire}_timeline_autopilot.json"):
        p = ROOT / "outputs" / name
        if p.exists():
            return p
    return None


def short(call: dict) -> str:
    """One readable line per tool call for the reasoning log."""
    t, a, r = call["tool"], call["args"], call["result"]
    if t == "_agent_says":
        return str(r)
    if isinstance(r, dict) and r.get("error"):
        return f"{t}: {r['error']}"
    if t == "get_situation":
        n = len(r.get("plume_tracks", []))
        return f"Checked the board: {n} plume track{'s' if n != 1 else ''} in view"
    if t == "verify_track":
        return f"Checked {a.get('track_id')}: {r.get('state')} ({r.get('confidence')}). {r.get('reason', '')}"
    if t == "recheck_track":
        if not r.get("consistent"):
            return (f"Considered rechecking {a.get('track_id')}: it points {abs(r.get('difference_deg', 0)):.0f}° "
                    f"away from the fire, so it was left out")
        outcome = str(r.get("result", "")).split(";")[0].split(":")[0]
        return f"Rechecked {a.get('track_id')} against the fire estimate: {outcome}. {r.get('reason', '')}"
    if t == "locate":
        if r.get("located"):
            place = r["nearby"][0]["phrase"] if r.get("nearby") else f"{r['lat']}, {r['lon']}"
            return (f"Located the fire {place}, ±{r['uncertainty_m']} m, from "
                    f"{', '.join(r['cameras'])}")
        return f"Tried to locate: {r.get('reason')}"
    if t == "issue_alert":
        return f"Issued a {a.get('level')} alert" if r.get("issued") else f"Alert refused: {r.get('error')}"
    return f"{t}"


def export(fire: str) -> dict | None:
    tl_path = timeline_path(fire)
    if tl_path is None:
        print(f"{fire}: no timeline; run scripts/run_agent.py first")
        return None
    tl = json.loads(tl_path.read_text())
    obs = load_fire(fire)
    names = {r["station"]: r["name"] for r in rows("stations.csv")}
    out_dir = UI / fire
    t0 = (obs.t_first_visible or obs.t_start) - LEAD_S

    cameras = {}
    for r in rows("fires.csv"):
        if r["fire"] != fire:
            continue
        cam_id = f"{r['station']}-{r['direction']}"
        mpath = ROOT / "outputs" / r["sequence"] / "camera_model.json"
        raw = ROOT / "data" / "raw" / r["sequence"]
        if not mpath.exists() or not raw.exists():
            continue
        m = CameraModel.load(mpath)
        frames = [f for f in load_sequence(raw) if f.epoch >= t0 - 60]
        (out_dir / cam_id).mkdir(parents=True, exist_ok=True)
        flist = []
        for f in frames:
            name = f"{f.epoch}.jpg"
            dest = out_dir / cam_id / name
            if not dest.exists():
                with Image.open(f.path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((THUMB_W, THUMB_W))
                    im.save(dest, quality=68, optimize=True)
            flist.append([f.epoch, f"{fire}/{cam_id}/{name}"])
        cameras[cam_id] = {"id": cam_id, "station": r["station"], "station_name": names.get(r["station"], r["station"]),
                           "lat": m.lat, "lon": m.lon, "heading": round(m.heading_deg, 2),
                           "hfov": round(m.hfov_deg, 2), "frames": flist,
                           "scale": THUMB_W / m.width}

    tracks = []
    for t in obs.tracks:
        cam = cameras.get(t.camera_id)
        s = cam["scale"] if cam else 1.0
        tracks.append({"id": t.track_id, "camera": t.camera_id, "lat": t.lat, "lon": t.lon,
                       "bearing": round(t.bearing_deg, 2), "sigma": t.sigma_deg, "edge": t.edge_clipped,
                       "first": t.first_epoch, "bearing_ready": t.bearing_epoch, "verify_ready": t.verify_epoch,
                       "boxes": [[t.frames[d.frame].epoch, round(d.x0 * s), round(d.y0 * s),
                                  round(d.x1 * s), round(d.y1 * s)] for d in t.dets]})

    states, locations, log = [], [], []
    for step in tl["timeline"]:
        e = step["epoch"]
        entry = {"epoch": e, "time": step["time"], "trigger": step["trigger"], "lines": []}
        for c in step["calls"]:
            r = c["result"]
            if c["tool"] == "verify_track" and isinstance(r, dict) and r.get("state"):
                states.append({"epoch": e, "track": r["track_id"], "state": r["state"],
                               "confidence": r.get("confidence"), "reason": r.get("reason")})
            if c["tool"] == "recheck_track" and isinstance(r, dict) and "corroborated" in str(r.get("result", "")):
                states.append({"epoch": e, "track": r["track_id"], "state": "corroborated",
                               "confidence": None, "reason": r.get("reason")})
            if c["tool"] == "locate" and isinstance(r, dict) and r.get("located"):
                locations.append({"epoch": e, "lat": r["lat"], "lon": r["lon"], "uncertainty_m": r["uncertainty_m"],
                                  "cameras": r["cameras"], "tracks": r["track_ids"],
                                  "place": r["nearby"][0]["phrase"] if r.get("nearby") else None})
            entry["lines"].append({"tool": c["tool"], "text": short(c)})
        log.append(entry)

    alerts = [{k: a.get(k) for k in ("level", "time", "epoch", "headline", "message", "location")}
              for a in tl["alerts"]]
    ign = {r["fire"]: r for r in rows("ignitions.csv")}.get(fire, {})
    label = fire.replace("Day2", "").replace("fire", " Fire").replace("Fire", " Fire").replace("  ", " ").strip()
    if fire.endswith("Day2"):
        label += ", day 2"
    data = {"fire": fire, "label": label,
            "driver": tl["driver"], "t0": t0, "t1": obs.t_end, "t_first_visible": obs.t_first_visible,
            "first_visible_time": hhmm(obs.t_first_visible) if obs.t_first_visible else None,
            "cameras": list(cameras.values()), "tracks": tracks, "states": states, "locations": locations,
            "alerts": alerts, "log": log, "score": tl["score"],
            "truth": {"lat": float(ign["lat"]), "lon": float(ign["lon"]), "location": ign.get("location"),
                      "source": ign.get("source")} if ign else None}
    (UI / f"{fire}.json").write_text(json.dumps(data))
    print(f"{fire}: {len(cameras)} cameras, {sum(len(c['frames']) for c in cameras.values())} frames, "
          f"{len(alerts)} alerts ({tl['driver']})")
    return {"fire": fire, "label": label, "date": next((r["date"] for r in rows("fires.csv") if r["fire"] == fire), ""),
            "alerts": [a["level"] for a in alerts], "score": tl["score"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="append")
    args = ap.parse_args()
    UI.mkdir(parents=True, exist_ok=True)
    fires = args.fire or sorted({r["fire"] for r in rows("fires.csv") if timeline_path(r["fire"])})
    index_path = UI / "fires.json"
    index = {f["fire"]: f for f in (json.loads(index_path.read_text()) if index_path.exists() else [])}
    for f in fires:
        item = export(f)
        if item:
            index[f] = item
    index_path.write_text(json.dumps(sorted(index.values(), key=lambda x: x["date"], reverse=True), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

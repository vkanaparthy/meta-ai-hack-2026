"""Run SAM 3.1 (and optionally Muse Spark) on camera sequences and score them.

  python scripts/run_sam.py data/raw/20260629_JunctionFire_*
  python scripts/run_sam.py --verify data/raw/20260629_JunctionFire_*

Per camera it prints time-to-detect (seconds after the labelled first visible
smoke) and false alarms before ignition, writes outputs/<sequence>/results.json
and an overlay image of the first detection. SAM output and Muse verdicts are
cached, so reruns don't cost API calls (use --refresh to redo).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from smokespotter.frames import load_sequence  # noqa: E402
from dataclasses import replace  # noqa: E402

from smokespotter.sam import Detection, parse_detections, segment_video  # noqa: E402
from smokespotter.video import frames_to_mp4  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIDEO_MAX_WIDTH = 1536


def tiles_for(w: int, h: int, n: int, band: str) -> list[tuple[int, int, int, int]]:
    """n side-by-side crops of the horizontal band ``band`` ("top:bottom" as
    fractions of image height), each overlapping its neighbour by 10%."""
    top, bottom = (float(v) for v in band.split(":"))
    y0, y1 = int(h * top) // 2 * 2, int(h * bottom) // 2 * 2
    if n <= 1:
        return [(0, y0, w, y1 - y0)]
    tw = int(w / n * 1.1) // 2 * 2
    step = (w - tw) / (n - 1)
    return [(int(i * step) // 2 * 2, y0, tw, y1 - y0) for i in range(n)]


def tracks_of(dets: list[Detection]) -> dict[str, list[Detection]]:
    t = defaultdict(list)
    for d in dets:
        t[d.object_id].append(d)
    return {k: sorted(v, key=lambda d: d.frame) for k, v in t.items()}


def overlap(a: Detection, b: Detection) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    smaller = min((a.x1 - a.x0) * (a.y1 - a.y0), (b.x1 - b.x0) * (b.y1 - b.y0))
    return inter / smaller if smaller else 0.0   # overlap of the smaller box


def dedupe_tiles(tracks: dict[str, list[Detection]]) -> dict[str, list[Detection]]:
    """Drop a track that duplicates a longer one from another tile (tile overlap)."""
    keep = dict(tracks)
    for a in sorted(tracks, key=lambda k: len(tracks[k])):
        if a not in keep:
            continue
        for b in keep:
            if b == a or b.split(":")[0] == a.split(":")[0] or len(keep[b]) < len(keep[a]):
                continue
            fb = {d.frame: d for d in keep[b]}
            shared = [(d, fb[d.frame]) for d in keep[a] if d.frame in fb]
            if len(shared) >= len(keep[a]) / 2 and sum(overlap(x, y) > 0.5 for x, y in shared) >= len(shared) / 2:
                del keep[a]
                break
    return keep


def track_summary(oid: str, ds: list[Detection], width: int) -> dict:
    first = ds[:3]
    margin = 0.01 * width
    return {
        "object_id": oid, "first_frame": ds[0].frame, "last_frame": ds[-1].frame, "n_frames": len(ds),
        "confidence": ds[0].confidence,
        "base_x_first3": [round(d.base_x, 1) for d in first],
        "edge_clipped": any(d.x0 <= margin or d.x1 >= width - margin for d in first),
    }


def union_box(ds: list[Detection]) -> tuple[float, float, float, float]:
    return (min(d.x0 for d in ds), min(d.y0 for d in ds),
            max(d.x1 for d in ds), max(d.y1 for d in ds))


def score(frames, dets) -> dict:
    hit = sorted({d.frame for d in dets})
    false_alarms = sum(1 for i in hit if not frames[i].has_smoke)
    first = next((frames[i] for i in hit if frames[i].has_smoke), None)
    pre = sum(1 for f in frames if not f.has_smoke)
    return {
        "detected": first is not None,
        "time_to_detect_s": first.offset if first else None,
        "false_alarm_frames": false_alarms,
        "pre_ignition_frames": pre,
        "first_detection_frame": frames.index(first) if first else None,
    }


def overlay(frame_path: Path, dets: list[Detection], out: Path) -> None:
    img = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w = max(2, img.width // 600)
    for d in dets:
        draw.rectangle([d.x0, d.y0, d.x1, d.y1], outline=(255, 80, 40), width=w)
        draw.line([d.base_x, d.y0, d.base_x, d.y1], fill=(255, 220, 0), width=w)
        label = f"id {d.object_id}" + (f"  {d.confidence:.2f}" if d.confidence is not None else "")
        draw.text((d.x0, max(0, d.y0 - 14 * w)), label, fill=(255, 80, 40))
    img.thumbnail((1600, 1600))
    img.save(out, quality=88)


def run(folder: Path, args) -> dict | None:
    frames = load_sequence(folder)
    if len(frames) < 2:
        print(f"{folder.name}: not enough frames")
        return None
    out = ROOT / "outputs" / folder.name
    out.mkdir(parents=True, exist_ok=True)
    with Image.open(frames[0].path) as im:
        orig_w, orig_h = im.size
    dets = []
    for i, (cx, cy, cw, ch) in enumerate(tiles_for(orig_w, orig_h, args.tiles, args.band)):
        tag = f"{args.prompt.replace(' ', '_')}" + (f"_t{args.tiles}x{i}_b{args.band}" if args.tiles > 1 or args.band != "0:1" else "")
        raw_path = out / f"sam_{tag}.txt"
        scale = cw / min(VIDEO_MAX_WIDTH, cw)
        if args.refresh or not raw_path.exists():
            crop = None if (cw, ch) == (orig_w, orig_h) else (cx, cy, cw, ch)
            mp4 = frames_to_mp4(frames, out / f"clip_{i}.mp4", max_width=VIDEO_MAX_WIDTH, crop=crop)
            print(f"{folder.name}: running SAM 3.1 on {len(frames)} frames, tile {i + 1} "
                  f"({cw}x{ch} at x={cx}, y={cy})...", flush=True)
            raw_path.write_text(segment_video(mp4, args.prompt))
        for d in parse_detections(raw_path.read_text(), scale):
            dets.append(replace(d, object_id=f"{i}:{d.object_id}", x0=d.x0 + cx, x1=d.x1 + cx,
                                y0=d.y0 + cy, y1=d.y1 + cy, base_x=d.base_x + cx))
    dets = [d for d in dets if d.frame < len(frames)
            and (d.confidence is None or d.confidence >= args.min_conf)
            and d.area_px >= args.min_area]
    tracks = {k: v for k, v in tracks_of(dets).items() if len(v) >= args.min_frames}
    tracks = dedupe_tiles(tracks)
    sam_only = score(frames, [d for v in tracks.values() for d in v])
    result = {"sequence": folder.name, "frames": len(frames), "tracks": len(tracks),
              "sam": sam_only}

    if args.verify:
        from smokespotter.muse import verify_track

        cache_path = out / "verdicts.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() and not args.refresh else {}
        ranked = sorted(tracks, key=lambda k: -len(tracks[k]))[: args.max_verify]
        kept = []
        for oid in ranked:
            ds = tracks[oid]
            if oid not in cache:
                picks = [ds[0], ds[len(ds) // 2], ds[-1]]
                v = verify_track([frames[d.frame].path for d in picks], union_box(ds))
                cache[oid] = {"is_smoke": v.is_smoke, "confidence": v.confidence, "reason": v.reason}
            if cache[oid]["is_smoke"]:
                kept.append(oid)
        cache_path.write_text(json.dumps(cache, indent=2))
        tracks = {k: tracks[k] for k in kept}
        result["verified"] = score(frames, [d for v in tracks.values() for d in v])
        result["verdicts"] = cache

    final = result.get("verified", sam_only)
    if final["first_detection_frame"] is not None:
        i = final["first_detection_frame"]
        overlay(frames[i].path, [d for v in tracks.values() for d in v if d.frame == i],
                out / "first_detection.jpg")
    result["image_width"] = orig_w
    result["track_summaries"] = [track_summary(k, v, orig_w) for k, v in tracks.items()]
    result["detections"] = [d.to_dict() for v in tracks.values() for d in v]
    (out / "results.json").write_text(json.dumps(result, indent=2))

    def line(tag, s):
        when = f"{s['time_to_detect_s']:+d}s" if s["detected"] else "missed"
        return (f"  {tag:<9} detected {when}; false alarms "
                f"{s['false_alarm_frames']}/{s['pre_ignition_frames']} pre-ignition frames")
    print(f"{folder.name}: {len(tracks)} track(s)")
    for t in result["track_summaries"]:
        print(f"    track {t['object_id']}: frames {t['first_frame']}-{t['last_frame']}, "
              f"plume base x~{t['base_x_first3'][0]:.0f}" + ("  [edge-clipped]" if t["edge_clipped"] else ""))
    print(line("SAM", sam_only))
    if "verified" in result:
        print(line("+Muse", result["verified"]))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--prompt", default="smoke", help="SAM text prompt (one noun phrase)")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--min-area", type=float, default=200, help="min mask area, original px")
    ap.add_argument("--min-frames", type=int, default=3, help="track must persist this many frames")
    ap.add_argument("--tiles", type=int, default=1,
                    help="split each frame into N full-resolution crops (more API calls, sees smaller plumes)")
    ap.add_argument("--band", default="0:1",
                    help="vertical band to keep, as fractions top:bottom, e.g. 0.3:0.7 for the horizon")
    ap.add_argument("--verify", action="store_true", help="check tracks with Muse Spark")
    ap.add_argument("--max-verify", type=int, default=5, help="max Muse calls per camera")
    ap.add_argument("--refresh", action="store_true", help="ignore cached API results")
    args = ap.parse_args()
    for f in args.folders:
        try:
            run(f, args)
        except Exception as e:
            print(f"{f.name}: FAILED ({type(e).__name__}: {e})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

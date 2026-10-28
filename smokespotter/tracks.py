"""Plume tracks: tiling, grouping SAM detections into tracks, de-duplication,
and loading cached SAM results for a camera sequence."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from .sam import Detection, parse_detections

VIDEO_MAX_WIDTH = 1536
DEFAULTS = dict(prompt="smoke", tiles=2, band="0.3:0.7", min_conf=0.3, min_area=200.0, min_frames=3)


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


def raw_tag(prompt: str, tiles: int, band: str, i: int) -> str:
    return f"{prompt.replace(' ', '_')}" + (f"_t{tiles}x{i}_b{band}" if tiles > 1 or band != "0:1" else "")


def detections_from_cache(out_dir: Path, width: int, height: int, n_frames: int,
                          prompt: str = "smoke", tiles: int = 2, band: str = "0.3:0.7",
                          min_conf: float = 0.3, min_area: float = 200.0) -> list[Detection] | None:
    """Detections in original-image pixels from cached SAM output (None if not cached)."""
    dets = []
    for i, (cx, cy, cw, ch) in enumerate(tiles_for(width, height, tiles, band)):
        raw = out_dir / f"sam_{raw_tag(prompt, tiles, band, i)}.txt"
        if not raw.exists():
            return None
        for d in parse_detections(raw.read_text(), cw / min(VIDEO_MAX_WIDTH, cw)):
            dets.append(replace(d, object_id=f"{i}:{d.object_id}", x0=d.x0 + cx, x1=d.x1 + cx,
                                y0=d.y0 + cy, y1=d.y1 + cy, base_x=d.base_x + cx))
    return [d for d in dets if d.frame < n_frames
            and (d.confidence is None or d.confidence >= min_conf) and d.area_px >= min_area]


def build_tracks(dets: list[Detection], min_frames: int = 3) -> dict[str, list[Detection]]:
    tracks = {k: v for k, v in tracks_of(dets).items() if len(v) >= min_frames}
    return dedupe_tiles(tracks)

"""Latency and cost per call, measured the same way for every model.

  python scripts/benchmark_latency.py --provider meta --provider open --provider glimmer --sam

Smoke check: the same --n crops (real plumes from the replayed fires) go to each provider one at a
time, so the timings are comparable. --sam also runs SAM 3.1 once on a 10-frame clip cut exactly
like the pipeline's (one tile, 1-minute frames), without touching any cache.
Every call is logged to outputs/usage_<provider>.jsonl; run scripts/usage_report.py for the table.
Cost: about 10 smoke checks per provider (well under $0.10) and one SAM call.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smokespotter.models import set_provider  # noqa: E402
from smokespotter.observations import load_fire  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIRES = ["Kitchenfire", "ResortFire", "JunctionFire", "GroveFire", "AirportFire"]


def crops(n: int):
    """n real-plume tracks, spread across fires, in a fixed order."""
    picks, per_fire = [], []
    for fire in FIRES:
        per_fire.append([t for t in load_fire(fire).tracks if t.frames[t.dets[0].frame].offset >= 0])
    i = 0
    while len(picks) < n and any(per_fire):
        lst = per_fire[i % len(per_fire)]
        if lst:
            picks.append(lst.pop(0))
        i += 1
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", action="append", choices=["meta", "open", "glimmer"])
    ap.add_argument("--n", type=int, default=10, help="smoke checks per provider")
    ap.add_argument("--sam", action="store_true", help="also time one SAM 3.1 call (Meta API)")
    args = ap.parse_args()
    out = {}
    tracks = crops(args.n)
    for prov in args.provider or []:
        set_provider(prov)
        from smokespotter import muse
        secs = []
        for t in tracks:
            t0 = time.time()
            muse.verify_track(t.verify_frames(), t.verify_box(), at_edge=t.edge_clipped)
            secs.append(time.time() - t0)
            print(f"  {prov:<8} {t.track_id:<16} {secs[-1]:5.1f} s")
        out[prov] = {"checks": len(secs), "median_s": round(statistics.median(secs), 1),
                     "max_s": round(max(secs), 1)}
    if args.sam:
        set_provider("meta")
        from smokespotter.frames import load_sequence
        from smokespotter.sam import segment_video
        from smokespotter.tracks import VIDEO_MAX_WIDTH, tiles_for
        from smokespotter.video import frames_to_mp4
        frames = load_sequence(ROOT / "data" / "raw" / "20240701_Kitchenfire_pi-e-mobo-c")[40:50]
        from PIL import Image
        with Image.open(frames[0].path) as im:
            W, H = im.size
        crop = tiles_for(W, H, 2, "0.3:0.7")[1]
        with tempfile.TemporaryDirectory() as tmp:
            mp4 = frames_to_mp4(frames, Path(tmp) / "bench.mp4", max_width=VIDEO_MAX_WIDTH, crop=crop)
            t0 = time.time()
            raw = segment_video(mp4)
            dt = time.time() - t0
        out["sam_10_frames"] = {"seconds": round(dt, 1), "output_chars": len(raw)}
        print(f"  SAM 3.1  10-frame tile clip  {dt:5.1f} s")
    (ROOT / "outputs" / "latency_benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("details per call: python scripts/usage_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

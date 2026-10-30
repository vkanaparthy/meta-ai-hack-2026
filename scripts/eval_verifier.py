"""Stress-test the smoke verifier on every plume track SAM found, not just the ones the agent uses.

Tracks that start before smoke is first visible on that camera cannot be this fire's smoke, so
they measure false alarms. Hard negatives add more: the same crop of the same view 15 and 25
minutes before that plume appeared. Only crops whose three frames all come before smoke was first
visible on that camera (same_view_before_smoke) count as negatives; later ones may show the fire. Each track is checked --repeats times to measure how stable the
verdict is, and compared with the cached Muse Spark 1.3 verdict where there is one.

  python scripts/eval_verifier.py --provider open --repeats 3
  python scripts/eval_verifier.py --provider open --fire Kitchenfire --limit 10   # quick look

Writes outputs/verifier_eval_<provider>.json (rows + summary). Rerunning resumes: rows already
done are kept.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smokespotter.models import BudgetExceeded, set_provider, spent_usd  # noqa: E402
from smokespotter.observations import load_fire  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def fires() -> list[str]:
    with open(ROOT / "data" / "fires.csv") as fh:
        rows = list(csv.DictReader(l for l in fh if not l.startswith("#")))
    return sorted({r["fire"] for r in rows})


def summarize(rows: list[dict]) -> dict:
    out = {}
    for group in ("after_smoke", "before_smoke", "same_view_before_smoke", "same_view_fire_burning"):
        g = [r for r in rows if r["group"] == group and r["votes"]]
        if not g:
            continue
        maj = [sum(r["votes"]) * 2 > len(r["votes"]) for r in g]
        stable = [len(set(r["votes"])) == 1 for r in g]
        ref = [(m, r["meta_1_3"]) for m, r in zip(maj, g) if r["meta_1_3"] is not None]
        out[group] = {
            "tracks": len(g),
            "judged_smoke": sum(maj),
            "judged_smoke_rate": round(sum(maj) / len(g), 3),
            "unanimous_across_repeats": round(sum(stable) / len(g), 3),
            "agree_with_muse_1_3": round(sum(a == b for a, b in ref) / len(ref), 3) if ref else None,
            "compared_with_muse_1_3": len(ref),
            "median_confidence": round(statistics.median(c for r in g for c in r["confidences"]), 2),
        }
    replies = [x for r in rows for x in r["reasons"]]
    out["replies"] = {"total": len(replies),
                      "unparseable": sum(x.startswith("unparseable") for x in replies)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["meta", "open"], default="open")
    ap.add_argument("--fire", action="append", help="default: every fire in data/fires.csv")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, help="max tracks per fire")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--earlier", type=int, nargs="*", default=[15, 25],
                    help="hard negatives: minutes before each plume (empty to skip)")
    args = ap.parse_args()
    set_provider(args.provider)
    from smokespotter import muse

    out_path = ROOT / "outputs" / f"verifier_eval_{args.provider}.json"
    done = {r["track"]: r for r in json.loads(out_path.read_text())["rows"]} if out_path.exists() else {}
    for r in done.values():
        # empty or cut-off replies are a transport problem, not a verdict: drop and redo them
        keep = [i for i, x in enumerate(r["reasons"]) if not x.startswith("unparseable")]
        for k in ("votes", "confidences", "reasons"):
            r[k] = [r[k][i] for i in keep]
        if r["group"] == "same_view_earlier":
            r["group"] = "same_view_before_smoke" if r["minutes_from_first_smoke"] < 0 else "same_view_fire_burning"

    jobs = []
    for fire in args.fire or fires():
        try:
            obs = load_fire(fire)
        except ValueError:
            continue
        set_provider("meta")            # reference verdicts come from the Muse Spark 1.3 cache
        ref = {t.track_id: t.verdict for t in load_fire(fire).tracks}
        set_provider(args.provider)
        tracks = obs.tracks[: args.limit] if args.limit else obs.tracks
        for t in tracks:
            key = f"{fire}:{t.track_id}"
            first = t.frames[t.dets[0].frame]
            row = done.get(key) or {
                "track": key, "fire": fire, "camera": t.camera_id, "frames": len(t.dets),
                "group": "before_smoke" if first.offset < 0 else "after_smoke",
                "minutes_from_first_smoke": round(first.offset / 60, 1),
                "edge_clipped": t.edge_clipped,
                "meta_1_3": (ref.get(t.track_id) or {}).get("is_smoke"),
                "votes": [], "confidences": [], "reasons": []}
            done[key] = row
            frames, box = t.verify_frames(), t.verify_box()
            for _ in range(max(0, args.repeats - len(row["votes"]))):
                jobs.append((row, frames, box, t.edge_clipped))
            # hard negatives: same box, same camera, before this plume existed
            i0 = t.dets[0].frame
            for back in args.earlier:
                j = i0 - back
                if j < 0 or j + 4 >= i0:
                    continue
                nkey = f"{key}@-{back}min"
                nrow = done.get(nkey) or {
                    "track": nkey, "fire": fire, "camera": t.camera_id, "frames": 3,
                    "group": None,
                    "minutes_from_first_smoke": round(t.frames[j].offset / 60, 1),
                    "edge_clipped": t.edge_clipped, "meta_1_3": None,
                    "votes": [], "confidences": [], "reasons": []}
                # a negative only if ALL three frames are before smoke was first visible on this camera
                nrow["group"] = "same_view_before_smoke" if t.frames[j + 4].offset < 0 else "same_view_fire_burning"
                nrow["last_frame_minutes_from_first_smoke"] = round(t.frames[j + 4].offset / 60, 1)
                done[nkey] = nrow
                nframes = [t.frames[j].path, t.frames[j + 2].path, t.frames[j + 4].path]
                for _ in range(max(0, args.repeats - len(nrow["votes"]))):
                    jobs.append((nrow, nframes, box, t.edge_clipped))

    print(f"{len(done)} tracks, {len(jobs)} checks to run on {args.provider}")

    def check(frames, box, edge):
        return muse.verify_track(frames, box, at_edge=edge)

    def save():
        rows = sorted(done.values(), key=lambda r: r["track"])
        out_path.write_text(json.dumps({"provider": args.provider, "summary": summarize(rows), "rows": rows}, indent=2))

    n = 0
    try:
        with ThreadPoolExecutor(args.workers) as pool:
            futs = {pool.submit(check, fr, bx, ed): row for row, fr, bx, ed in jobs}
            for f in as_completed(futs):
                row = futs[f]
                try:
                    v = f.result()
                except BudgetExceeded:
                    raise
                except Exception as e:      # one bad call should not lose the run
                    print(f"  {row['track']}: error {e}")
                    continue
                row["votes"].append(v.is_smoke)
                row["confidences"].append(v.confidence)
                row["reasons"].append(v.reason)
                n += 1
                if n % 20 == 0:
                    save()
                    extra = f", spend ${spent_usd():.2f}" if args.provider == "open" else ""
                    print(f"  {n}/{len(jobs)} checks{extra}")
    except BudgetExceeded as e:
        print(f"stopped: {e}")
    save()
    print(json.dumps(summarize(list(done.values())), indent=2))
    if args.provider == "open":
        print(f"open-weights spend so far: ${spent_usd():.2f}")
    print(f"wrote {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

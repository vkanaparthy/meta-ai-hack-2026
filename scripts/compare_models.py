"""Scorecard: Muse Spark 1.3 (Meta API) vs Muse Spark Open (open weights) on the same replays.

  python scripts/compare_models.py

Reads outputs/<fire>_timeline.json (1.3) and outputs/<fire>_timeline_open[_rN].json (open),
plus outputs/verifier_eval_open.json if present. Writes outputs/model_comparison.json.
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEVELS = ["spotted", "provisional", "confirmed"]


def summarize(t: dict) -> dict:
    levels = [a["level"] for a in t["alerts"]]
    top = max((LEVELS.index(l) for l in levels if l in LEVELS), default=-1)
    tools = [c for s in t["timeline"] for c in s["calls"] if not c["tool"].startswith("_")]
    refused = [c for c in tools if c["tool"] == "issue_alert" and isinstance(c["result"], dict)
               and c["result"].get("error")]
    sc = t.get("score", {})
    return {"top_level": LEVELS[top] if top >= 0 else "none", "alerts": len(levels),
            "final_error_m": sc.get("final_error_m"), "final_uncertainty_m": sc.get("final_uncertainty_m"),
            "spotted_min": sc.get("spotted_min_after_first_smoke"),
            "tool_calls": len(tools), "alerts_refused_by_guardrail": len(refused)}


def main() -> int:
    out = {"fires": {}}
    fires = sorted({Path(p).name.split("_timeline")[0] for p in glob.glob(str(ROOT / "outputs" / "*_timeline*.json"))})
    print(f"{'fire':<14}{'model':<10}{'runs':>5}  {'top level (runs)':<34}{'error m (median)':>17}{'refused':>9}")
    for fire in fires:
        rows = {}
        meta = ROOT / "outputs" / f"{fire}_timeline.json"
        if meta.exists():
            rows["1.3"] = [summarize(json.loads(meta.read_text()))]
        opens = sorted(glob.glob(str(ROOT / "outputs" / f"{fire}_timeline_open*.json")))
        if opens:
            rows["open"] = [summarize(json.loads(Path(p).read_text())) for p in opens]
        out["fires"][fire] = rows
        for model, rs in rows.items():
            tops = {}
            for r in rs:
                tops[r["top_level"]] = tops.get(r["top_level"], 0) + 1
            errs = [r["final_error_m"] for r in rs if r["final_error_m"] is not None]
            err = f"{statistics.median(errs):,.0f}" if errs else "-"
            print(f"{fire:<14}{model:<10}{len(rs):>5}  {', '.join(f'{k} x{v}' for k, v in tops.items()):<34}"
                  f"{err:>17}{sum(r['alerts_refused_by_guardrail'] for r in rs):>9}")
    ev = ROOT / "outputs" / "verifier_eval_open.json"
    if ev.exists():
        out["verifier_open"] = json.loads(ev.read_text())["summary"]
        print("\nverifier (open weights):")
        print(json.dumps(out["verifier_open"], indent=2))
    evg = ROOT / "outputs" / "verifier_eval_glimmer.json"
    if evg.exists():
        out["verifier_glimmer"] = json.loads(evg.read_text())["summary"]
        print("\nverifier (Muse Glimmer, local):")
        print(json.dumps(out["verifier_glimmer"], indent=2))
    usage = ROOT / "outputs" / "usage_open.jsonl"
    if usage.exists():
        rows = [json.loads(l) for l in usage.read_text().splitlines() if l.strip()]
        by = {}
        for r in rows:
            b = by.setdefault(r["purpose"], {"calls": 0, "usd": 0.0})
            b["calls"] += 1
            b["usd"] = round(b["usd"] + r["usd"], 4)
        out["open_weights_spend"] = by
        print(f"\nopen-weights spend: ${sum(b['usd'] for b in by.values()):.2f}  {by}")
    (ROOT / "outputs" / "model_comparison.json").write_text(json.dumps(out, indent=2))
    print("wrote outputs/model_comparison.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Latency and cost of every logged model call, per provider and purpose.

  python scripts/usage_report.py

Reads outputs/usage_<provider>.jsonl (written automatically by every model call).
Writes outputs/usage_report.json.
"""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABEL = {"meta": "Meta Model API", "open": "Muse Spark Open (Baseten)", "glimmer": "Muse Glimmer (local)"}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main() -> int:
    report = {}
    print(f"{'provider':<27}{'purpose':<13}{'calls':>6}{'p50 s':>8}{'p90 s':>8}{'in tok':>9}{'out tok':>9}"
          f"{'$/call':>10}{'total $':>9}")
    for path in sorted(glob.glob(str(ROOT / "outputs" / "usage_*.jsonl"))):
        prov = Path(path).stem.removeprefix("usage_")
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        by: dict[str, list[dict]] = {}
        for r in rows:
            by.setdefault(r["purpose"], []).append(r)
        for purpose, rs in sorted(by.items()):
            secs = [r["seconds"] for r in rs if r.get("seconds") is not None]
            usd = [r["usd"] for r in rs if r.get("usd") is not None]
            row = {"calls": len(rs),
                   "p50_s": round(statistics.median(secs), 1) if secs else None,
                   "p90_s": round(pct(secs, 0.9), 1) if secs else None,
                   "timed_calls": len(secs),
                   "mean_in_tokens": round(statistics.mean(r["in"] for r in rs)),
                   "mean_out_tokens": round(statistics.mean(r["out"] for r in rs)),
                   "usd_per_call": round(statistics.mean(usd), 5) if usd else None,
                   "usd_total": round(sum(usd), 2) if usd else None}
            report.setdefault(prov, {})[purpose] = row
            f = lambda v, fmt: (fmt.format(v) if v is not None else "-")
            print(f"{LABEL.get(prov, prov):<27}{purpose:<13}{row['calls']:>6}{f(row['p50_s'], '{:.1f}'):>8}"
                  f"{f(row['p90_s'], '{:.1f}'):>8}{row['mean_in_tokens']:>9}{row['mean_out_tokens']:>9}"
                  f"{f(row['usd_per_call'], '${:.4f}'):>10}{f(row['usd_total'], '${:.2f}'):>9}")
    (ROOT / "outputs" / "usage_report.json").write_text(json.dumps(report, indent=2))
    print("\nSAM 3.1 has no published per-token price, so its cost shows as '-'. Calls logged before "
          "latency metering have no timing (timed_calls in the JSON).\nwrote outputs/usage_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

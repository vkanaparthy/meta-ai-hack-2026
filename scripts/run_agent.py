"""Replay a fire as if live with the dispatch agent.

  python scripts/run_agent.py --fire Kitchenfire                        # Muse Spark 1.3 agent (Meta API)
  python scripts/run_agent.py --fire Kitchenfire --provider open        # Muse Spark Open (open weights)
  python scripts/run_agent.py --fire Kitchenfire --provider open --runs 5
  python scripts/run_agent.py --fire Kitchenfire --autopilot            # fixed policy, cached verdicts, no API

Writes outputs/<fire>_timeline[_open][_rN].json (every tool call and alert) and prints the alerts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smokespotter.agent import Autopilot, make_agent, run_replay  # noqa: E402
from smokespotter.desk import FireDesk, cached_backend, live_backend  # noqa: E402
from smokespotter.models import BudgetExceeded, set_provider, spent_usd  # noqa: E402
from smokespotter.observations import load_fire  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="append", required=True)
    ap.add_argument("--provider", choices=["meta", "open"], help="Muse provider (default: MUSE_PROVIDER or meta)")
    ap.add_argument("--runs", type=int, default=1, help="repeat each replay N times (model output varies)")
    ap.add_argument("--autopilot", action="store_true", help="fixed policy instead of Muse Spark")
    ap.add_argument("--offline", action="store_true", help="cached verdicts only, no live smoke checks")
    ap.add_argument("--verbose", action="store_true", help="print every tool call")
    args = ap.parse_args()
    if args.provider:
        set_provider(args.provider)
    from smokespotter.models import provider
    prov = provider()
    for fire in args.fire:
        for run in range(1, args.runs + 1):
            obs = load_fire(fire)      # fresh each run: verdicts cached by earlier runs are reused
            backend = cached_backend() if (args.offline or args.autopilot) else live_backend()
            desk = FireDesk(obs, backend)
            driver = Autopilot() if args.autopilot else make_agent()
            try:
                result = run_replay(obs, desk, driver)
            except BudgetExceeded as e:
                print(f"stopped: {e}")
                return 1
            suffix = "_autopilot" if args.autopilot else ("_open" if prov == "open" else "")
            if args.runs > 1:
                suffix += f"_r{run}"
            name = f"{fire}_timeline{suffix}.json"
            (ROOT / "outputs" / name).write_text(json.dumps(result, indent=2, default=str))
            print(f"=== {fire} run {run}/{args.runs} ({result['driver']}, model: {result.get('model')}, "
                  f"smoke checks: {result['muse']})")
            for step in result["timeline"]:
                if args.verbose:
                    print(f"  [{step['time']}] {'; '.join(step['trigger'])}")
                    for c in step["calls"]:
                        r = c["result"]
                        short = r if isinstance(r, str) else json.dumps(r, default=str)[:160]
                        print(f"      {c['tool']}({json.dumps(c['args'])[:60]}) -> {short}")
            for a in result["alerts"]:
                loc = a.get("location")
                where = f" @ {loc['lat']}, {loc['lon']} +-{loc['uncertainty_m']} m" if loc else ""
                print(f"  {a['time']}  {a['level'].upper():<11} {a['headline']}{where}")
                print(f"         {a['message']}")
            print(f"  score: {json.dumps(result['score'])}")
            print(f"  wrote outputs/{name}")
            if prov == "open":
                print(f"  open-weights spend so far: ${spent_usd():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

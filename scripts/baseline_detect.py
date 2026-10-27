"""Day-one baseline: how soon does simple change detection flag smoke?

Prints, per sequence, the first frame flagged and its offset from the labelled
first-visible-smoke time, plus false alarms before ignition. SAM 3.1 + Muse
should beat this.

  python scripts/baseline_detect.py data/raw/20260629_JunctionFire_hp-e-mobo-c
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smokespotter.detect import change_candidates, load_gray  # noqa: E402
from smokespotter.frames import load_sequence  # noqa: E402


def run(folder: str, window: int = 5) -> None:
    frames = load_sequence(folder)
    if not frames:
        print(f"{folder}: no frames found")
        return
    history, false_alarms, first_hit = [], 0, None
    for f in frames:
        img = load_gray(f.path)
        boxes = change_candidates(history[-window:], img)
        if boxes:
            if f.has_smoke and first_hit is None:
                first_hit = f
            elif not f.has_smoke:
                false_alarms += 1
        history.append(img)
    name = Path(folder).name
    if first_hit:
        print(f"{name}: detected at {first_hit.offset:+d}s after first visible smoke; "
              f"{false_alarms} false alarms before ignition")
    else:
        print(f"{name}: missed; {false_alarms} false alarms before ignition")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        run(arg)

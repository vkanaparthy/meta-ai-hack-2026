"""Load FIgLib-style image sequences.

FIgLib file names look like ``<epoch_seconds>_<offset_seconds>.jpg`` where the
offset is relative to when the plume first became visible (negative = before).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_NAME = re.compile(r"^(?P<epoch>\d{9,11})_(?P<offset>[+-]?\d+)\.jpe?g$", re.IGNORECASE)


@dataclass(frozen=True)
class Frame:
    path: Path
    epoch: int          # capture time, unix seconds (UTC)
    offset: int         # seconds from first visible smoke (ground truth label)

    @property
    def has_smoke(self) -> bool:
        return self.offset >= 0


def parse_frame_name(name: str) -> tuple[int, int] | None:
    m = _NAME.match(name)
    if not m:
        return None
    return int(m["epoch"]), int(m["offset"])


def load_sequence(folder: str | Path) -> list[Frame]:
    """Return frames in capture order. Files that don't match the pattern are skipped."""
    frames = []
    for p in Path(folder).iterdir():
        parsed = parse_frame_name(p.name)
        if parsed:
            frames.append(Frame(p, *parsed))
    return sorted(frames, key=lambda f: f.epoch)

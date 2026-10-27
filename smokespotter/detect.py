"""Smoke candidate detection.

Stage 1 (cheap, here): frame differencing against a rolling background finds
regions that changed. Stage 2 (SAM 3.1, via ``Segmenter``) outlines and tracks
the plume inside those regions. Stage 3 (Muse) judges smoke vs fog/haze/glare.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int
    score: float

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2


def load_gray(path, max_width: int = 1024) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)))
    return np.asarray(img, dtype=np.float32)


def change_candidates(history: Sequence[np.ndarray], current: np.ndarray,
                      thresh: float = 18.0, min_pixels: int = 40,
                      sky_fraction: float = 0.75) -> list[Box]:
    """Bounding box of pixels that differ from the median of recent frames.

    Only the top ``sky_fraction`` of the image is searched, where early plumes
    rise above the ridgeline. Returns at most one coarse box; SAM refines it.
    """
    if not history:
        return []
    bg = np.median(np.stack(history), axis=0)
    diff = np.abs(current - bg)
    h = int(current.shape[0] * sky_fraction)
    mask = diff[:h] > thresh
    if mask.sum() < min_pixels:
        return []
    ys, xs = np.nonzero(mask)
    return [Box(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()),
                float(mask.sum()))]


class Segmenter(Protocol):
    """Interface for SAM 3.1 (wired up once Meta Model API access is live)."""

    def track(self, frames: Sequence[str], prompt: str,
              hint: Box | None = None) -> list[np.ndarray]:
        """Return one boolean mask per frame for the tracked object."""
        ...

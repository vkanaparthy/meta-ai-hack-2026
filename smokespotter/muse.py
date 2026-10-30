"""Muse Spark checks whether a SAM track is really smoke (not fog, haze,
cloud shadow or glare) by looking at crops from several moments."""
from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .models import ask_vision

PROMPT = """You are verifying a wildfire smoke detection from a fixed mountaintop camera.
The images are crops of the SAME region at successive times (earliest first), about a minute apart.
Decide whether the highlighted region is a new wildfire smoke plume.
Smoke rises from one point near the ground, grows and drifts over time.
Look-alikes to reject: fog, low cloud, haze layers, dust, sun glare, lens flare, cloud shadows.
Reply with JSON only: {"is_smoke": true|false, "confidence": 0.0-1.0, "reason": "<one sentence>"}"""

EDGE_NOTE = """
Note: this region touches the edge of the camera's field of view, so the plume may be
cut off: its ground source or part of its column can lie outside the image. Do not
reject it only because the base or full column is not visible; judge by whether the
visible part grows, rises and drifts like smoke over time."""


@dataclass(frozen=True)
class Verdict:
    is_smoke: bool
    confidence: float
    reason: str


def _crop_data_url(path: Path, box: tuple[float, float, float, float], pad: float = 1.0,
                   max_side: int = 768) -> str:
    img = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = box
    w, h = max(x1 - x0, 32), max(y1 - y0, 32)
    crop = img.crop((max(0, x0 - pad * w), max(0, y0 - pad * h),
                     min(img.width, x1 + pad * w), min(img.height, y1 + pad * h)))
    crop.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_verdict(text: str) -> Verdict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return Verdict(False, 0.0, f"unparseable reply: {text[:120]}")
    try:
        d = json.loads(m.group(0))
        return Verdict(bool(d.get("is_smoke")), float(d.get("confidence", 0.0)),
                       str(d.get("reason", "")))
    except (ValueError, TypeError):
        return Verdict(False, 0.0, f"unparseable reply: {text[:120]}")


def _ask_verdict(prompt: str, images: list[str], purpose: str, tries: int = 3) -> Verdict:
    """Ask, and ask again if the reply is empty or cut off (no parseable JSON)."""
    v = Verdict(False, 0.0, "unparseable reply: ")
    for _ in range(tries):
        v = parse_verdict(ask_vision(prompt, images, purpose=purpose))
        if not v.reason.startswith("unparseable reply"):
            return v
    return v


def verify_track(frame_paths: list[Path], box: tuple[float, float, float, float],
                 at_edge: bool = False) -> Verdict:
    """frame_paths: 2-4 frames of the track, earliest first. box: union box in original pixels.
    at_edge: the region touches the image border (plume may be cut off)."""
    return _ask_verdict(PROMPT + (EDGE_NOTE if at_edge else ""),
                        [_crop_data_url(p, box) for p in frame_paths], "verify")


RECHECK_PROMPT = """You are re-examining a camera view that an earlier check doubted.
Independent evidence: {context}
The images are crops from this camera at successive times (earliest first), about a minute apart,
centred on the region that lines up with that fire.
Is there wildfire smoke here consistent with that fire? The plume may be small, distant, partly
hidden behind terrain, or cut off at the image edge. Do not say yes just because other cameras
saw a fire; say yes only if this view shows smoke (a grey/brown haze or column that is not a
cloud, glare or ordinary haze layer).
Reply with JSON only: {{"is_smoke": true|false, "confidence": 0.0-1.0, "reason": "<one sentence>"}}"""


def recheck_track(frame_paths: list[Path], box: tuple[float, float, float, float], context: str) -> Verdict:
    return _ask_verdict(RECHECK_PROMPT.format(context=context),
                        [_crop_data_url(p, box, pad=1.5) for p in frame_paths], "recheck")

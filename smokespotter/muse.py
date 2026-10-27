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

from .models import client, reasoning_model

PROMPT = """You are verifying a wildfire smoke detection from a fixed mountaintop camera.
The images are crops of the SAME region at successive times (earliest first), about a minute apart.
Decide whether the highlighted region is a new wildfire smoke plume.
Smoke rises from one point near the ground, grows and drifts over time.
Look-alikes to reject: fog, low cloud, haze layers, dust, sun glare, lens flare, cloud shadows.
Reply with JSON only: {"is_smoke": true|false, "confidence": 0.0-1.0, "reason": "<one sentence>"}"""


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


def verify_track(frame_paths: list[Path], box: tuple[float, float, float, float]) -> Verdict:
    """frame_paths: 2-4 frames of the track, earliest first. box: union box in original pixels."""
    content = [{"type": "input_text", "text": PROMPT}]
    content += [{"type": "input_image", "image_url": _crop_data_url(p, box)} for p in frame_paths]
    resp = client().responses.create(
        model=reasoning_model(),
        input=[{"type": "message", "role": "user", "content": content}],
    )
    return parse_verdict(resp.output_text)

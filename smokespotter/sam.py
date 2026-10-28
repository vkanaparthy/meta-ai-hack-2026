"""SAM 3.1 video segmentation through the Meta Model API.

One request per camera: the image sequence is sent as a short MP4 and SAM
tracks every object matching the text prompt across frames.
Docs: https://dev.meta.ai/docs/sam/segmenting
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from meta_sam_parser import (
    CompletedOutcome,
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    video_segmentation_format,
)

from .models import client, sam_model


@dataclass(frozen=True)
class Detection:
    frame: int              # index into the camera's frame list
    object_id: str          # stable across frames within one request
    x0: float               # box in ORIGINAL image pixels
    y0: float
    x1: float
    y1: float
    confidence: float | None
    area_px: float          # mask area in original pixels
    base_x: float           # mean column of the plume's lowest rows (origin estimate)

    def to_dict(self) -> dict:
        return asdict(self)


def segment_video(mp4: Path, prompt: str = "smoke") -> str:
    """Upload the clip, run SAM 3.1, return the raw output text."""
    c = client()
    with open(mp4, "rb") as fh:
        uploaded = c.files.create(file=fh, purpose="user_data")
    chunks: list[str] = []
    with c.responses.stream(
        model=sam_model(),
        input=[{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_video", "file_id": uploaded.id},
            ],
        }],
        metadata={"include_confidence": "true", "mask_encoding": "one_bit"},
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)
    return "".join(chunks)


def parse_detections(raw: str, scale: float = 1.0) -> list[Detection]:
    """Parse SAM output with Meta's parser; ``scale`` = original width / video width."""
    if not raw.strip():
        return []
    parser = video_segmentation_format().create_parser()
    parser.push(raw if raw.endswith("\n") else raw + "\n")
    records = parser.finish(CompletedOutcome()).result.records

    boxes: dict[tuple[int, str], SegmentationBoxRecord] = {}
    masks: dict[tuple[int, str], SegmentationMaskRecord] = {}
    for r in records:
        frame = r.frame.frame_index if getattr(r, "frame", None) else 0
        if isinstance(r, SegmentationBoxRecord):
            boxes[(frame, r.object_id)] = r
        elif isinstance(r, SegmentationMaskRecord):
            masks[(frame, r.object_id)] = r

    out = []
    for key, b in sorted(boxes.items()):
        frame, oid = key
        area, base_x = _mask_stats(masks.get(key), b)
        out.append(Detection(
            frame=frame, object_id=oid,
            x0=b.left * scale, y0=b.top * scale, x1=b.right * scale, y1=b.bottom * scale,
            confidence=b.confidence, area_px=area * scale * scale, base_x=base_x * scale,
        ))
    return out


def _mask_stats(m: SegmentationMaskRecord | None, b: SegmentationBoxRecord) -> tuple[float, float]:
    """Area and plume-base column in video pixels. Falls back to the box."""
    if m is None:
        return (b.right - b.left) * (b.bottom - b.top), (b.left + b.right) / 2
    import numpy as np

    H, W = m.mask.height, m.mask.width
    grid = np.frombuffer(decode_mask_to_raster(m.mask), dtype=np.uint8).reshape(H, W) > 0
    bx = m.bounds
    sx = (bx.right - bx.left) / W if W else 1.0
    sy = (bx.bottom - bx.top) / H if H else 1.0
    rows = np.flatnonzero(grid.any(axis=1))
    if rows.size == 0:
        return 0.0, (b.left + b.right) / 2
    # lowest 15% of occupied rows ~ where the plume leaves the ground
    base_rows = rows[-max(1, rows.size * 15 // 100):]
    cols = np.nonzero(grid[base_rows])[1]
    area = float(grid.sum()) * sx * sy
    return area, bx.left + (float(cols.mean()) + 0.5) * sx

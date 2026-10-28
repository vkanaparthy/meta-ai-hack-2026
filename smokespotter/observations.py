"""Everything the agent can observe about a fire, built from cached results:
each camera's plume tracks with bearing, uncertainty, timing and Muse verdict."""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .camera import CameraModel
from .config import ROOT
from .frames import Frame, load_sequence
from .locate import TrackBearing
from .sam import Detection
from .tracks import build_tracks, detections_from_cache, track_summary, union_box


@dataclass
class CameraTrack:
    track_id: str               # "<camera>/<sam object id>", unique within a fire
    camera_id: str
    station: str
    sequence: str
    oid: str
    dets: list[Detection]
    frames: list[Frame] = field(repr=False)
    lat: float
    lon: float
    bearing_deg: float
    sigma_deg: float
    edge_clipped: bool
    first_epoch: int            # first frame the plume is segmented
    bearing_epoch: int          # 3rd frame: plume base estimate usable
    verify_epoch: int           # 5th frame: enough history for Muse to judge
    verdict: dict | None        # cached Muse verdict, if any

    def bearing(self) -> TrackBearing:
        return TrackBearing(self.camera_id, self.station, self.track_id, self.lat, self.lon,
                            self.bearing_deg, self.bearing_epoch, self.edge_clipped, self.sigma_deg)

    def verify_frames(self) -> list[Path]:
        ds = self.dets
        return [self.frames[d.frame].path for d in (ds[0], ds[min(2, len(ds) - 1)], ds[min(4, len(ds) - 1)])]

    def verify_box(self):
        ds = self.dets
        return union_box([ds[0], ds[min(2, len(ds) - 1)], ds[min(4, len(ds) - 1)]])


@dataclass
class FireObservations:
    fire: str
    tracks: list[CameraTrack]
    cameras: list[str]                  # every camera with footage, incl. ones with no plume
    t_start: int
    t_end: int
    t_first_visible: int | None         # labelled first visible smoke (for scoring only)
    ignition: tuple[float, float] | None = None   # official point (for scoring only)


def _rows(name: str) -> list[dict]:
    path = ROOT / "data" / name
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def sigma_for(quality: dict, xfrac: float) -> float:
    """Bearing uncertainty from calibration quality and position in the frame."""
    weak = (quality.get("median_abs_px") or 99) > 2.0 or (quality.get("edge_coverage") or 0) < 0.75
    if weak:
        return 6.0
    return 2.5 if (xfrac < 0.10 or xfrac > 0.88) else 1.0


def load_fire(fire: str) -> FireObservations:
    seqs = [r for r in _rows("fires.csv") if r["fire"] == fire]
    if not seqs:
        raise ValueError(f"{fire} not in data/fires.csv")
    tracks, cameras, starts, ends, visible = [], [], [], [], []
    for r in seqs:
        cam_id = f"{r['station']}-{r['direction']}"
        raw = ROOT / "data" / "raw" / r["sequence"]
        out = ROOT / "outputs" / r["sequence"]
        model_path = out / "camera_model.json"
        if not raw.exists() or not model_path.exists():
            continue
        frames = load_sequence(raw)
        if not frames:
            continue
        cameras.append(cam_id)
        starts.append(frames[0].epoch)
        ends.append(frames[-1].epoch)
        visible += [f.epoch for f in frames if f.offset >= 0][:1]
        with Image.open(frames[0].path) as im:
            W, H = im.size
        dets = detections_from_cache(out, W, H, len(frames))
        if dets is None:
            continue
        model = CameraModel.load(model_path)
        quality = json.loads(model_path.read_text()).get("quality", {})
        verdicts = json.loads((out / "verdicts.json").read_text()) if (out / "verdicts.json").exists() else {}
        for oid, ds in build_tracks(dets).items():
            ds = sorted(ds, key=lambda d: d.frame)
            summ = track_summary(oid, ds, W)
            x = statistics.median(summ["base_x_first3"])
            y = statistics.median(d.y1 - 1 for d in ds[:3])
            tracks.append(CameraTrack(
                track_id=f"{cam_id}/{oid}", camera_id=cam_id, station=r["station"], sequence=r["sequence"],
                oid=oid, dets=ds, frames=frames, lat=model.lat, lon=model.lon,
                bearing_deg=float(model.pixel_to_azel(x, y)[0]), sigma_deg=sigma_for(quality, x / W),
                edge_clipped=bool(summ["edge_clipped"]),
                first_epoch=frames[ds[0].frame].epoch,
                bearing_epoch=frames[ds[min(2, len(ds) - 1)].frame].epoch,
                verify_epoch=frames[ds[min(4, len(ds) - 1)].frame].epoch,
                verdict=verdicts.get(oid)))
    ign = {r["fire"]: r for r in _rows("ignitions.csv")}.get(fire)
    return FireObservations(
        fire=fire, tracks=sorted(tracks, key=lambda t: t.first_epoch), cameras=sorted(cameras),
        t_start=min(starts), t_end=max(ends), t_first_visible=min(visible) if visible else None,
        ignition=(float(ign["lat"]), float(ign["lon"])) if ign else None)

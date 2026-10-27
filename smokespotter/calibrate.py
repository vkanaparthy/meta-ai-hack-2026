"""Camera heading and field-of-view calibration from fires with known origins.

Model: every camera at a station is mounted at its nominal direction
(n=0, e=90, s=180, w=270) plus one shared rotation offset for that station's
mount; all cameras share one horizontal field of view (same camera model).

    bearing(x) = nominal + offset[station] + atan((x - W/2) / f),  f = (W/2) / tan(hfov/2)

Each observation is "the plume base was at column x in camera C, and the true
bearing from C's station to the known ignition point is B". Two cameras at the
same station seeing the same fire (e.g. hp-e and hp-s) pin down the field of
view; one observation per station pins down its offset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

NOMINAL = {"n": 0.0, "e": 90.0, "s": 180.0, "w": 270.0}


@dataclass(frozen=True)
class CalObs:
    camera_id: str      # "<station>-<direction>", e.g. "hp-e"
    x: float            # plume base column, original image pixels
    width: int          # image width, pixels
    true_bearing: float


@dataclass(frozen=True)
class CameraCal:
    camera_id: str
    heading_deg: float
    hfov_deg: float
    residual_deg: float | None   # this camera's mean |error|; None if uncalibrated
    n_obs: int


def split_camera(camera_id: str) -> tuple[str, str]:
    station, _, direction = camera_id.rpartition("-")
    return station, direction


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _pixel_angle(x: float, width: int, hfov: float) -> float:
    f = (width / 2) / math.tan(math.radians(hfov) / 2)
    return math.degrees(math.atan((x - width / 2) / f))


def _circular_mean(angles: list[float]) -> float:
    s = sum(math.sin(math.radians(a)) for a in angles)
    c = sum(math.cos(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(s, c))


def _key(camera_id: str, group: str) -> str:
    return camera_id if group == "camera" else split_camera(camera_id)[0]


def _offsets(obs: list[CalObs], hfov: float, group: str = "station") -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for o in obs:
        _, d = split_camera(o.camera_id)
        rest = o.true_bearing - NOMINAL[d] - _pixel_angle(o.x, o.width, hfov)
        per.setdefault(_key(o.camera_id, group), []).append(_wrap(rest))
    return {k: _circular_mean(v) for k, v in per.items()}


def _residuals(obs: list[CalObs], hfov: float, offsets: dict[str, float],
               group: str = "station") -> list[float]:
    out = []
    for o in obs:
        _, d = split_camera(o.camera_id)
        pred = NOMINAL[d] + offsets[_key(o.camera_id, group)] + _pixel_angle(o.x, o.width, hfov)
        out.append(_wrap(pred - o.true_bearing))
    return out


def fit(obs: list[CalObs], hfov_range=(60.0, 150.0), step: float = 0.1,
        default_hfov: float | None = None,
        group: str = "station") -> tuple[float, dict[str, float], float, bool]:
    """Return (hfov, offsets, RMS residual deg, hfov_identified).

    group="station": one rotation per station (cameras assumed exactly 90 deg apart).
    group="camera":  one rotation per camera; needs a camera with 2+ views at
                     different columns to identify the field of view."""
    if not obs:
        raise ValueError("no calibration observations")
    for o in obs:
        if split_camera(o.camera_id)[1] not in NOMINAL:
            raise ValueError(f"camera {o.camera_id!r} needs a n/e/s/w direction suffix")

    keys = [_key(o.camera_id, group) for o in obs]
    identified = len(keys) > len(set(keys))   # some group has 2+ observations
    if not identified:
        hfov = default_hfov or 100.0
        return hfov, _offsets(obs, hfov, group), 0.0, False

    best = None
    h = hfov_range[0]
    while h <= hfov_range[1] + 1e-9:
        offs = _offsets(obs, h, group)
        res = _residuals(obs, h, offs, group)
        rms = math.sqrt(sum(r * r for r in res) / len(res))
        if best is None or rms < best[2]:
            best = (h, offs, rms)
        h += step
    return best[0], best[1], best[2], True


def camera_table(obs: list[CalObs], hfov: float, offsets: dict[str, float],
                 cameras: list[str], group: str = "station") -> list[CameraCal]:
    if group == "camera":   # cameras without views inherit their station's mean rotation
        by_station: dict[str, list[float]] = {}
        for k, v in offsets.items():
            by_station.setdefault(split_camera(k)[0], []).append(v)
        offsets = {**{s: _circular_mean(v) for s, v in by_station.items()}, **offsets}
    res = dict()
    for o, r in zip(obs, _residuals(obs, hfov, offsets, group)):
        res.setdefault(o.camera_id, []).append(abs(r))
    out = []
    for cam in sorted(set(cameras)):
        station, d = split_camera(cam)
        if d not in NOMINAL:
            continue
        off = offsets.get(cam) if group == "camera" and cam in offsets else offsets.get(station)
        rs = res.get(cam)
        out.append(CameraCal(
            camera_id=cam,
            heading_deg=round((NOMINAL[d] + (off or 0.0)) % 360, 2),
            hfov_deg=round(hfov, 2),
            residual_deg=round(sum(rs) / len(rs), 2) if rs else None,
            n_obs=len(rs) if rs else 0,
        ))
    return out

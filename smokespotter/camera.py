"""Full camera model: orientation (heading, pitch, roll), focal length and one
radial distortion term. Maps (azimuth, elevation) <-> image pixels.

World frame is local East-North-Up at the camera. Image x grows right, y down.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraModel:
    lat: float
    lon: float
    height_m: float          # camera height above sea level
    heading_deg: float       # azimuth of the optical axis, clockwise from north
    pitch_deg: float         # elevation of the optical axis (positive = up)
    roll_deg: float          # rotation about the optical axis (positive = clockwise in image)
    f_px: float              # focal length, pixels
    k1: float                # radial distortion: r_d = r (1 + k1 r^2 + k2 r^4), r in focal units
    width: int
    height: int
    k2: float = 0.0

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.width / 2 / self.f_px)))

    def _axes(self):
        h, p, r = np.radians([self.heading_deg, self.pitch_deg, self.roll_deg])
        fwd = np.array([np.cos(p) * np.sin(h), np.cos(p) * np.cos(h), np.sin(p)])
        right0 = np.array([np.cos(h), -np.sin(h), 0.0])
        up0 = np.cross(right0, fwd)
        right = np.cos(r) * right0 - np.sin(r) * up0
        up = np.sin(r) * right0 + np.cos(r) * up0
        return fwd, right, up

    def project(self, az_deg, el_deg):
        """Directions -> pixel (x, y). Points behind the camera get NaN."""
        az, el = np.radians(np.asarray(az_deg, float)), np.radians(np.asarray(el_deg, float))
        v = np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], axis=-1)
        fwd, right, up = self._axes()
        z = v @ fwd
        u = np.where(z > 1e-6, (v @ right) / np.where(z > 1e-6, z, 1), np.nan)
        w = np.where(z > 1e-6, (v @ up) / np.where(z > 1e-6, z, 1), np.nan)
        r2 = u * u + w * w
        # keep only directions that land within ~1.3x the image half-diagonal and
        # where the distortion is still monotonic (beyond that it folds back)
        r2_max = 1.3 ** 2 * ((self.width / 2) ** 2 + (self.height / 2) ** 2) / self.f_px ** 2
        valid = (r2 <= r2_max) & (1 + 3 * self.k1 * r2 + 5 * self.k2 * r2 * r2 > 0.2)
        s = 1 + self.k1 * r2 + self.k2 * r2 * r2
        x = self.width / 2 + self.f_px * u * s
        y = self.height / 2 - self.f_px * w * s
        return np.where(valid, x, np.nan), np.where(valid, y, np.nan)

    def pixel_to_azel(self, x, y):
        """Pixel (x, y) -> (azimuth, elevation) in degrees."""
        ud = (np.asarray(x, float) - self.width / 2) / self.f_px
        wd = (self.height / 2 - np.asarray(y, float)) / self.f_px
        u, w = ud, wd
        for _ in range(30):                       # invert the distortion
            r2 = u * u + w * w
            s = 1 + self.k1 * r2 + self.k2 * r2 * r2
            u, w = ud / s, wd / s
        fwd, right, up = self._axes()
        v = fwd + np.multiply.outer(u, right) + np.multiply.outer(w, up)
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)
        az = np.degrees(np.arctan2(v[..., 0], v[..., 1])) % 360
        el = np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))
        return az, el

    def save(self, path: Path, **extra) -> None:
        path.write_text(json.dumps({**asdict(self), **extra}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "CameraModel":
        d = json.loads(Path(path).read_text())
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

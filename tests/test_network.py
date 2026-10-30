"""Geometry of the network what-if study."""
import math

import numpy as np

from smokespotter.network import fix_radius_m, line_of_sight


class FlatDEM:
    def __init__(self, z=0.0, ridge=None):
        self.zv, self.ridge = z, ridge

    def elevation(self, lat, lon):
        z = np.full(np.shape(lat), self.zv, float)
        if self.ridge:
            lon0, h = self.ridge
            z[np.abs(np.asarray(lon) - lon0) < 0.01] = h
        return z


def test_perpendicular_beats_nearly_in_line():
    tgt = (33.0, -117.0)
    square = fix_radius_m([(33.0, -117.3), (33.25, -117.0)], tgt)       # 90 degrees apart
    inline = fix_radius_m([(33.0, -117.3), (33.0, -117.25)], tgt)       # same direction
    assert square < 1500 and (inline == math.inf or inline > 20 * square)


def test_one_camera_cannot_fix():
    assert fix_radius_m([(33.0, -117.3)], (33.0, -117.0)) == math.inf


def test_ridge_blocks_view():
    cam, tgt = (33.0, -117.3), (33.0, -117.0)
    assert line_of_sight(FlatDEM(), *cam, 20, *tgt, 150)
    assert not line_of_sight(FlatDEM(ridge=(-117.15, 2000)), *cam, 20, *tgt, 150)

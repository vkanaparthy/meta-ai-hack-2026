import math

import pytest

from smokespotter.geo import Observation, haversine_m, pixel_to_bearing, triangulate


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360


FIRE = (33.25, -116.60)
CAMS = [(33.35, -116.70), (33.40, -116.50), (33.15, -116.75)]


def test_three_cameras_recover_fire():
    obs = [Observation(la, lo, bearing(la, lo, *FIRE)) for la, lo in CAMS]
    fix = triangulate(obs)
    assert haversine_m(fix.lat, fix.lon, *FIRE) < 150
    assert fix.n_obs == 3


def test_two_cameras_with_heading_error():
    obs = [Observation(la, lo, bearing(la, lo, *FIRE) + 1.0) for la, lo in CAMS[:2]]
    fix = triangulate(obs)
    # a 1 degree error at ~15 km is a few hundred metres
    assert haversine_m(fix.lat, fix.lon, *FIRE) < 800


def test_parallel_bearings_rejected():
    with pytest.raises(ValueError):
        triangulate([Observation(33.0, -116.0, 90), Observation(33.1, -116.0, 90)])


def test_pixel_to_bearing_centre_and_edges():
    assert pixel_to_bearing(1536, 3072, 90, 100) == pytest.approx(90)
    assert pixel_to_bearing(0, 3072, 90, 100) == pytest.approx(40)
    assert pixel_to_bearing(3072, 3072, 0, 100) == pytest.approx(50)

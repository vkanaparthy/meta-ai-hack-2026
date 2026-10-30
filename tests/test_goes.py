"""GOES fixed-grid navigation against the worked example in the GOES-R Product User Guide."""
from smokespotter.goes import fixed_grid_to_latlon


def test_pug_example_point():
    # PUG Vol. 5, 4.2.8.1: x = -0.024052, y = 0.095340 rad for GOES-East (lon0 -75)
    lat, lon = fixed_grid_to_latlon(-0.024052, 0.095340, 35786023.0, 6378137.0, 6356752.31414, -75.0)
    assert abs(float(lat) - 33.846162) < 1e-3 and abs(float(lon) - (-84.690932)) < 1e-3


def test_off_disk_is_nan():
    lat, lon = fixed_grid_to_latlon(0.2, 0.2, 35786023.0, 6378137.0, 6356752.31414, -137.0)
    assert lat != lat and lon != lon

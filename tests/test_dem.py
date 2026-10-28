import numpy as np
import rasterio
from rasterio.transform import from_origin

from smokespotter.dem import DEM


def write_tile(path, lat0, lon0, n=360):
    """1x1 degree tile whose elevation = 1000*lat_frac + 100*lon_frac (pixel centres)."""
    step = 1.0 / n
    rows = lat0 + 1 - step * (np.arange(n) + 0.5)
    cols = lon0 + step * (np.arange(n) + 0.5)
    z = (1000 * (rows[:, None] - 32) + 100 * (cols[None, :] + 118)).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(lon0, lat0 + 1, step, step)) as dst:
        dst.write(z, 1)


def test_mosaic_and_lookup(tmp_path):
    paths = []
    for lat0, lon0 in [(32, -118), (32, -117), (33, -118), (33, -117)]:
        p = tmp_path / f"t{lat0}{lon0}.tif"
        write_tile(p, lat0, lon0)
        paths.append(p)
    dem = DEM.from_tiles(paths, step_arcsec=20)
    for lat, lon in [(32.5, -117.5), (33.7, -116.3), (33.01, -117.99)]:
        expect = 1000 * (lat - 32) + 100 * (lon + 118)
        assert abs(float(dem.elevation(lat, lon)) - expect) < 10
    dem.save(tmp_path / "m.npz")
    again = DEM.load(tmp_path / "m.npz")
    assert np.isnan(again.elevation(40.0, -117.0))
    lat, lon, z, moved = again.snap_to_summit(33.5, -116.5, radius_m=500)
    assert lat > 33.5 and lon > -116.5 and moved <= 500

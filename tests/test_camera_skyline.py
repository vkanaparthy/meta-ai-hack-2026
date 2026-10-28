import numpy as np
from PIL import Image

from smokespotter.camera import CameraModel
from smokespotter.dem import DEM, skyline
from smokespotter.skyline import SkylineSamples, edge_map, fit, predicted_rows

CAM_LAT, CAM_LON = 33.16, -116.81


def synthetic_dem(seed=3):
    step = 15 / 3600
    lats = np.arange(34.3, 32.0, -step)
    lons = np.arange(-118.2, -115.4, step)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    rng = np.random.default_rng(seed)
    z = np.full(LA.shape, 200.0)
    for _ in range(160):
        la, lo = rng.uniform(32.1, 34.2), rng.uniform(-118.1, -115.5)
        h, s = rng.uniform(300, 2600), rng.uniform(2, 9) / 111
        z += h * np.exp(-((LA - la) ** 2 + ((LO - lo) * 0.84) ** 2) / (2 * s * s))
    z += 1100 * np.exp(-((LA - CAM_LAT) ** 2 + ((LO - CAM_LON) * 0.84) ** 2) / (2 * (1.2 / 111) ** 2))
    return DEM(z.astype(np.float32), float(lats[0]), float(lons[0]), step)


def test_camera_roundtrip():
    cam = CameraModel(33, -116, 1000, 47.0, -2.0, 1.5, 1150.0, -0.08, 3072, 2048)
    az, el = np.array([20.0, 47.0, 80.0]), np.array([-3.0, 0.0, 4.0])
    x, y = cam.project(az, el)
    az2, el2 = cam.pixel_to_azel(x, y)
    assert np.allclose(az2, az, atol=1e-6) and np.allclose(el2, el, atol=1e-6)


def test_fit_recovers_orientation_and_lens():
    dem = synthetic_dem()
    lat, lon, z, _ = dem.snap_to_summit(CAM_LAT, CAM_LON)
    h_cam = z + 10
    az = np.arange(0, 180, 0.1)
    el, dist = skyline(dem, lat, lon, h_cam, az, d_max=100_000)
    sky = SkylineSamples(az, el, dist)

    W, H = 1536, 1024
    f_true = (W / 2) / np.tan(np.radians(99 / 2))
    truth = CameraModel(lat, lon, h_cam, 90 + 7.3, -2.1, 1.2, f_true, -0.08, W, H)
    rows = predicted_rows(truth, sky, np.arange(W, dtype=float))
    yy = np.arange(H)[:, None]
    img = np.where(yy < rows[None, :], 1.0, 0.0)[..., None] * np.array([120, 170, 230]) \
        + np.where(yy >= rows[None, :], 1.0, 0.0)[..., None] * np.array([95, 90, 70])
    img[700:, 1300:1450] = [40, 60, 30]                 # foreground tree
    img += np.random.default_rng(0).normal(0, 6, img.shape)
    emap, scale = edge_map([Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))])
    assert scale == 1.0

    f_nom = (W / 2) / np.tan(np.radians(104 / 2))
    base = CameraModel(lat, lon, h_cam, 90.0, 0.0, 0.0, f_nom, 0.0, W, H)
    cam, q = fit(emap, sky, base)

    # what matters downstream: bearings across the image
    xs = np.linspace(100, W - 100, 9)
    for y in (300, 500):
        a_true, _ = truth.pixel_to_azel(xs, np.full_like(xs, y))
        a_fit, _ = cam.pixel_to_azel(xs, np.full_like(xs, y))
        err = (a_fit - a_true + 180) % 360 - 180
        assert np.max(np.abs(err)) < 0.3, err
    assert q["median_abs_px"] is not None and q["median_abs_px"] <= 1.5


def test_shared_lens_joint_fit_recovers_k2():
    from dataclasses import replace
    from smokespotter.skyline import fit_shared_lens
    dem = synthetic_dem()
    lat, lon, z, _ = dem.snap_to_summit(CAM_LAT, CAM_LON)
    az = np.arange(0, 360, 0.1)
    el, dist = skyline(dem, lat, lon, z + 10, az, d_max=100_000)
    sky = SkylineSamples(az, el, dist)
    W, H = 1536, 1024
    f_true = (W / 2) / np.tan(np.radians(88 / 2))
    items, truths = [], []
    for heading in (20.0, 95.0, 160.0, 250.0, 330.0):
        t = CameraModel(lat, lon, z + 10, heading + 2.5, 0.8, -0.6, f_true, -0.20, W, H, k2=0.05)
        rows = predicted_rows(t, sky, np.arange(W, dtype=float))
        yy = np.arange(H)[:, None]
        img = np.where(yy < rows[None, :], 1.0, 0.0)[..., None] * np.array([120, 170, 230]) \
            + np.where(yy >= rows[None, :], 1.0, 0.0)[..., None] * np.array([95, 90, 70])
        img += np.random.default_rng(int(heading)).normal(0, 5, img.shape)
        emap, _ = edge_map([Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))])
        f_guess = (W / 2) / np.tan(np.radians(87.5 / 2))
        start = CameraModel(lat, lon, z + 10, heading + 2.0, 0.6, -0.4, f_guess, -0.146, W, H)
        items.append((emap, sky, start))
        truths.append(t)
    lens, cams = fit_shared_lens(items)
    assert abs(lens["hfov_deg"] - 88.0) < 0.5
    for t, c in zip(truths, cams):
        xs = np.array([30.0, 400.0, 768.0, 1150.0, 1500.0])
        a_t, _ = t.pixel_to_azel(xs, np.full_like(xs, 450.0))
        a_c, _ = c.pixel_to_azel(xs, np.full_like(xs, 450.0))
        assert np.max(np.abs((a_c - a_t + 180) % 360 - 180)) < 0.3

import math

from smokespotter.calibrate import CalObs, NOMINAL, camera_table, fit, split_camera


def make_obs(cam, true_bearing, offset, hfov, width=3072):
    """Inverse model: which column would show a target at true_bearing."""
    _, d = split_camera(cam)
    rel = math.radians(true_bearing - NOMINAL[d] - offset)
    f = (width / 2) / math.tan(math.radians(hfov) / 2)
    return CalObs(cam, width / 2 + f * math.tan(rel), width, true_bearing)


def test_recovers_hfov_and_offsets_from_paired_views():
    hfov = 112.0
    offs = {"hp": 3.0, "vo": -2.5, "bm": 1.2}
    obs = [make_obs("hp-e", 146.0, offs["hp"], hfov), make_obs("hp-s", 146.0, offs["hp"], hfov),
           make_obs("vo-n", 306.0, offs["vo"], hfov), make_obs("vo-w", 306.0, offs["vo"], hfov),
           make_obs("bm-e", 71.0, offs["bm"], hfov)]
    h, o, rms, identified = fit(obs)
    assert identified
    assert abs(h - hfov) < 0.3
    for s in offs:
        assert abs(o[s] - offs[s]) < 0.3
    assert rms < 0.1
    table = {c.camera_id: c for c in camera_table(obs, h, o, [x.camera_id for x in obs])}
    assert abs(table["hp-s"].heading_deg - 183.0) < 0.3


def test_single_view_per_station_uses_default_hfov():
    obs = [make_obs("mg-e", 90.0, 4.0, 100.0)]
    h, o, _, identified = fit(obs, default_hfov=100.0)
    assert not identified and h == 100.0
    assert abs(o["mg"] - 4.0) < 1e-6


def test_hyphenated_station_names():
    assert split_camera("smer-tcs3-n") == ("smer-tcs3", "n")


def test_per_camera_fit_handles_unequal_mounts():
    hfov = 104.0
    # hp-e and hp-s rotated differently: station model can't fit, camera model can
    obs = [make_obs("hp-e", 60.0, 4.0, hfov), make_obs("hp-e", 130.0, 4.0, hfov),
           make_obs("hp-s", 150.0, -3.0, hfov), make_obs("hp-s", 210.0, -3.0, hfov)]
    h, o, rms, ok = fit(obs, group="camera")
    assert ok and abs(h - hfov) < 0.3 and rms < 0.1
    assert abs(o["hp-e"] - 4.0) < 0.3 and abs(o["hp-s"] + 3.0) < 0.3
    _, _, rms_station, _ = fit(obs, group="station")
    assert rms_station > 1.0
    table = {c.camera_id: c for c in camera_table(obs, h, o, ["hp-e", "hp-s", "hp-n"], "camera")}
    assert abs(table["hp-n"].heading_deg - 0.5) < 0.5     # station mean of +4 and -3

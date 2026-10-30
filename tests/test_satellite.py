"""Satellite cross-check: time-honest, matched only by camera evidence, never raises a level."""
from pathlib import Path

from smokespotter.agent import events
from smokespotter.desk import FireDesk, MuseBackend
from smokespotter.frames import Frame
from smokespotter.geo import bearing_deg
from smokespotter.observations import CameraTrack, FireObservations
from smokespotter.sam import Detection
from smokespotter.satellite import Hotspot

FIRE = (33.19, -116.70)
CAMS = {"bm": (33.16, -116.81), "hp": (33.36, -116.84)}


def track(st, start):
    frames = [Frame(Path(f"/x/{st}/{i}.jpg"), start + 60 * i, 60 * i) for i in range(12)]
    dets = [Detection(i, "0", 100, 100, 200, 200, 0.8, 5000, 150) for i in range(10)]
    la, lo = CAMS[st]
    return CameraTrack(f"{st}-e/0", f"{st}-e", st, f"seq_{st}", "0", dets, frames, la, lo,
                       bearing_deg(la, lo, *FIRE), 1.0, False, start, start + 120, start + 240,
                       {"is_smoke": True, "confidence": 0.9, "reason": "test"})


def hot(lat, lon, t):
    return Hotspot(lat, lon, t, t + 60, "N20 VIIRS", "n", 5.0)


def desk_with(hotspots, tracks):
    obs = FireObservations("t", tracks, ["bm-e", "hp-e"], 1000, 5000, 1000, hotspots=hotspots)
    return FireDesk(obs, MuseBackend(lambda t: t.verdict, lambda t, c: None))


def test_hotspot_invisible_before_it_is_published():
    d = desk_with([hot(*FIRE, 2000)], [track("bm", 1000)])
    d.now = 2030                                  # observed, but not yet published (+60 s)
    assert d.check_satellite()["hotspots_reported"] == 0
    d.now = 2060
    assert d.check_satellite()["hotspots_reported"] == 1


def test_hotspot_matches_a_verified_bearing_but_not_an_unrelated_one():
    far = (33.40, -116.30)                       # well off camera bm's bearing
    d = desk_with([hot(*FIRE, 1500), hot(*far, 1500)], [track("bm", 1000)])
    d.now = 1600
    d.verify_track("bm-e/0")
    r = d.check_satellite()
    assert r["hotspots_reported"] == 2 and len(r["matches"]) == 1
    assert r["matches"][0]["on_bearing_of_camera"] == "bm-e"


def test_satellite_never_creates_or_raises_a_level():
    d = desk_with([hot(*FIRE, 1500)], [track("bm", 1000)])
    d.now = 1600
    assert d.issue_alert("spotted", "x", "y")["error"]            # no verified camera smoke yet
    d.verify_track("bm-e/0")
    d.check_satellite()
    assert d.issue_alert("provisional", "x", "y")["error"].startswith("evidence supports at most 'spotted'")
    assert d.issue_alert("spotted", "Smoke spotted", "Smoke seen; satellite agrees.")["issued"]
    assert d.alerts[-1]["satellite"]


def test_new_satellite_match_allows_an_update_once():
    d = desk_with([hot(*FIRE, 2500)], [track("bm", 1000)])
    d.now = 1600
    d.verify_track("bm-e/0")
    d.issue_alert("spotted", "Smoke spotted", "Smoke seen; direction only.")
    assert d.issue_alert("update", "u", "v")["error"]               # nothing new yet
    d.now = 2600
    d.check_satellite()
    assert d.issue_alert("update", "Satellite agrees", "A satellite hotspot agrees.")["issued"]
    assert d.issue_alert("update", "again", "again")["error"]       # already announced


def test_satellite_passes_wake_the_agent():
    obs = FireObservations("t", [track("bm", 1000)], ["bm-e"], 1000, 5000, 1000,
                           hotspots=[hot(*FIRE, 3000), hot(*FIRE, 3000), hot(*FIRE, 9000)])
    ev = [e for e in events(obs) if any("satellite" in w for w in e.what)]
    assert [e.epoch for e in ev] == [3060]                         # one pass inside the window


def test_goes_pixel_wakes_once_and_persistent_pixels_never():
    goes = lambda t, persistent=False: Hotspot(*FIRE, t, t + 90, "GOES-18 ABI", "processed", 20.0, 3000, persistent)
    solar = Hotspot(33.30, -116.50, 900, 990, "GOES-18 ABI", "processed", 50.0, 3000, True)
    obs = FireObservations("t", [track("bm", 1000)], ["bm-e"], 1000, 5000, 1000,
                           hotspots=[solar, goes(2000), goes(2300), goes(2600), solar])
    ev = [e for e in events(obs) if any("satellite" in w for w in e.what)]
    assert [e.epoch for e in ev] == [2090]


def test_coarse_goes_pixel_still_matches_a_bearing():
    off = (FIRE[0] + 0.02, FIRE[1])                          # ~2 km off the fire
    d = desk_with([Hotspot(*off, 1500, 1590, "GOES-18 ABI", "processed", 20.0, 3000)], [track("bm", 1000)])
    d.now = 1600
    d.verify_track("bm-e/0")
    assert len(d.check_satellite()["matches"]) == 1

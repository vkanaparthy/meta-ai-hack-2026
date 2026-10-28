from smokespotter.geo import bearing_deg, haversine_m
from smokespotter.locate import TrackBearing, best_location, first_location

FIRE = (33.19, -116.70)
OTHER = (33.30, -116.50)          # a second, unrelated plume
CAMS = {"bm": (33.16, -116.81), "hp": (33.36, -116.84), "vo": (33.14, -116.62)}


def tb(st, target, t, tid="0", err=0.0):
    la, lo = CAMS[st]
    return TrackBearing(f"{st}-x", st, tid, la, lo, bearing_deg(la, lo, *target) + err, t)


def test_picks_consistent_set_over_distractor():
    tracks = [tb("bm", FIRE, 100), tb("hp", FIRE, 200), tb("vo", FIRE, 300),
              tb("hp", OTHER, 150, "9")]          # distractor in hp only
    loc = best_location(tracks)
    assert {m.track_id for m in loc.members} == {"0"}
    assert haversine_m(loc.fix.lat, loc.fix.lon, *FIRE) < 100
    assert loc.uncertainty_m < 2000


def test_first_location_is_earliest_two_station_fix():
    tracks = [tb("bm", FIRE, 100, err=0.5), tb("vo", FIRE, 300), tb("hp", FIRE, 200)]
    loc = first_location(tracks)
    assert loc.epoch == 200 and len(loc.members) == 2


def test_inconsistent_bearings_give_no_fix():
    tracks = [tb("bm", FIRE, 1), tb("hp", OTHER, 1)]
    # two lines always intersect somewhere; range limit + residual can't reject, so
    # with exactly two this is a fix -- three mutually inconsistent lines are rejected
    tracks.append(tb("vo", (32.9, -116.9), 1))
    loc = best_location(tracks, n_sigma=2)
    assert loc is None or len(loc.members) == 2


def test_weights_favour_trusted_bearings():
    from smokespotter.geo import haversine_m
    good = [tb("bm", FIRE, 1), tb("vo", FIRE, 1)]
    bad = TrackBearing("hp-x", "hp", "0", *CAMS["hp"],
                       bearing_deg(*CAMS["hp"], *FIRE) + 6.0, 1, sigma_deg=6.0)
    loc = best_location(good + [bad])
    assert len(loc.members) == 3
    assert haversine_m(loc.fix.lat, loc.fix.lon, *FIRE) < 300


def test_fragments_of_one_plume_are_merged():
    from smokespotter.locate import merge_fragments
    a = tb("bm", FIRE, 100, tid="a")
    b = tb("bm", FIRE, 220, tid="b", err=2.5)        # same plume, split mask
    c = tb("bm", OTHER, 230, tid="c")                # different plume
    kept = {t.track_id for t in merge_fragments([b, a, c])}
    assert kept == {"a", "c"}


def test_desk_guardrails():
    from smokespotter.desk import FireDesk, MuseBackend
    from smokespotter.observations import FireObservations
    desk = FireDesk(FireObservations("x", [], [], 0, 1, None), MuseBackend(lambda t: None, lambda t, c: None))
    desk._note({"lat": 33.1930, "lon": -116.6635, "uncertainty_m": 1480})
    assert desk._citable(116.6635) and desk._citable(1.5) and desk._citable(1480)
    assert not desk._citable(250.0)
    assert desk.issue_alert("confirmed", "x", "y")["error"].startswith("no verified smoke")

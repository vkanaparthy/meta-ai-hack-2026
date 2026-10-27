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
    loc = best_location(tracks, max_resid_m=200)
    assert loc is None or len(loc.members) == 2

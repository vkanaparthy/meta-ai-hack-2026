"""Agent loop mechanics with a scripted stand-in for Muse Spark (no API)."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

from smokespotter.agent import MuseAgent, events, run_replay
from smokespotter.desk import FireDesk, MuseBackend
from smokespotter.frames import Frame
from smokespotter.geo import bearing_deg
from smokespotter.observations import CameraTrack, FireObservations
from smokespotter.sam import Detection

FIRE = (33.19, -116.70)
CAMS = {"bm": (33.16, -116.81), "hp": (33.36, -116.84), "mg": (33.19, -116.76)}


def track(st, start, smoke=True):
    frames = [Frame(Path(f"/x/{st}/{i}.jpg"), start + 60 * i, 60 * i) for i in range(12)]
    dets = [Detection(i, "0", 100, 100, 200, 200, 0.8, 5000, 150) for i in range(10)]
    la, lo = CAMS[st]
    return CameraTrack(f"{st}-e/0", f"{st}-e", st, f"seq_{st}", "0", dets, frames, la, lo,
                       bearing_deg(la, lo, *FIRE), 1.0, False, start, start + 120, start + 240,
                       {"is_smoke": smoke, "confidence": 0.9, "reason": "test"})


class ScriptedMuse:
    """Plays the agent: situation -> verify unverified -> locate -> alert at the eligible level."""
    def __init__(self):
        self.responses = SimpleCalls()

class SimpleCalls:
    def __init__(self):
        self.n = 0
        self.pending = {}

    def create(self, input, previous_response_id=None, **kw):
        assert previous_response_id is None
        self.n += 1
        rid = f"r{self.n}"
        last = input[-1]
        if last.get("role") == "user":               # new event: look around
            return self._calls(rid, [("get_situation", {})])
        tail = []
        for o in reversed(input):                    # trailing tool results
            if o.get("type") != "function_call_output":
                break
            tail.insert(0, o)
        res = {o["call_id"]: json.loads(o["output"]) for o in tail}
        assert all(any(i.get("type") == "function_call" and i["call_id"] == c for i in input) for c in res)
        name = self.pending.pop(next(iter(res)))
        out = next(iter(res.values()))
        if name == "get_situation":
            todo = [t["track_id"] for t in out["plume_tracks"] if t["status"] == "unverified"]
            return self._calls(rid, [("verify_track", {"track_id": t}) for t in todo] or [("locate", {})])
        if name == "verify_track":
            return self._calls(rid, [("locate", {})])
        if name == "locate" and out.get("eligible_level"):
            level = out["eligible_level"]
            msg = (f"Located at {out['lat']}, {out['lon']}, about {out['uncertainty_m']} m uncertainty."
                   if out.get("located") else "Smoke seen; direction only.")
            return self._calls(rid, [("issue_alert", {"level": level, "headline": f"{level} fire",
                                                        "message": msg})])
        return NS(id=rid, output=[], output_text="done")

    def _calls(self, rid, calls):
        items = []
        for i, (name, args) in enumerate(calls):
            cid = f"{rid}c{i}"
            self.pending[cid] = name
            items.append(NS(type="function_call", name=name, arguments=json.dumps(args), call_id=cid))
        return NS(id=rid, output=items, output_text="")


def test_agent_loop_escalates_through_levels():
    tracks = [track("mg", 1000), track("bm", 1300), track("hp", 1600)]
    obs = FireObservations("t", tracks, ["bm-e", "hp-e", "mg-e"], 1000, 2400, 1000)
    desk = FireDesk(obs, MuseBackend(lambda t: t.verdict, lambda t, c: None))
    agent = MuseAgent(client=ScriptedMuse(), model="test")
    result = run_replay(obs, desk, agent)
    levels = [a["level"] for a in result["alerts"]]
    assert levels[0] == "spotted" and levels[-1] == "confirmed"
    assert all(c["tool"] for step in result["timeline"] for c in step["calls"])
    assert len(events(obs)) >= 3

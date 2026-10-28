"""Replay a fire as if live and let an agent run the dispatch desk.

The clock steps through the archive; the agent is woken whenever something new
becomes observable (a plume's bearing is usable, or it has enough history to be
verified). Two drivers share the same tools and guardrails:
  * MuseAgent  - Muse Spark decides which tools to call and writes the alerts
  * Autopilot  - a fixed policy, for offline runs and as a fallback
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .desk import TOOLS, FireDesk, hhmm
from .geo import haversine_m
from .observations import FireObservations

SYSTEM = """You are the duty officer of an automated wildfire watch for a network of fixed mountaintop cameras.
You are woken whenever something new is observable. Work like a careful dispatcher:
1. Call get_situation to see the plume tracks and what has already been done.
2. Verify any track that is 'unverified'. Ignore tracks marked 'too new to verify' for now.
3. If at least one track is verified as smoke, call locate. Then, if any track was judged 'not smoke'
   (especially from a camera looking from a different direction, or when locate says the fix is too
   uncertain), call recheck_track on it: views at the image edge or far away are often misjudged.
   If a recheck corroborates a track, call locate again.
4. Issue an alert only when the evidence supports a level that has not been issued yet
   (spotted -> provisional -> confirmed), or as an 'update' when the location moved or tightened.
   Headline under 90 characters. Message 2-4 sentences: what was seen, where (use the nearby place phrase),
   how certain (uncertainty and number of cameras), and what is still unknown.
5. Only cite numbers that tools returned. If issue_alert returns an error, fix the problem and try again.
If nothing warrants an alert, end with one short sentence saying why. Be concise."""


@dataclass
class Event:
    epoch: int
    what: list[str]


def events(obs: FireObservations) -> list[Event]:
    ev: dict[int, list[str]] = {}
    for t in obs.tracks:
        ev.setdefault(t.bearing_epoch, []).append(f"new plume track {t.track_id} on camera {t.camera_id}")
        if t.verify_epoch != t.bearing_epoch:
            ev.setdefault(t.verify_epoch, []).append(f"track {t.track_id} now has enough history to verify")
    # merge events less than 90 s apart (cameras are not synchronised)
    merged: list[Event] = []
    for e in sorted(ev):
        if merged and e - merged[-1].epoch < 90:
            merged[-1] = Event(e, merged[-1].what + ev[e])
        else:
            merged.append(Event(e, ev[e]))
    return merged


class Autopilot:
    """Deterministic policy using the same tools."""
    name = "autopilot"

    def handle(self, desk: FireDesk, event: Event) -> list[dict]:
        calls = []

        def call(name, **args):
            res = desk.call(name, args)
            calls.append({"tool": name, "args": args, "result": res})
            return res

        sit = call("get_situation")
        for t in sit["plume_tracks"]:
            if t["status"] == "unverified":
                call("verify_track", track_id=t["track_id"])
        loc = call("locate")
        if loc.get("located") or loc.get("bearing_lines"):
            for t in call("get_situation")["plume_tracks"]:
                if t["status"] == "not smoke":
                    r = call("recheck_track", track_id=t["track_id"])
                    if "corroborated" in str(r.get("result", "")):
                        loc = call("locate")
        level = loc.get("eligible_level")
        issued = [a["level"] for a in desk.alerts]
        order = ["spotted", "provisional", "confirmed"]
        if level and (not issued or order.index(level) > max(order.index(l) for l in issued if l in order)):
            call("issue_alert", level=level, **self._text(level, loc))
        elif loc.get("located") and any(l in ("provisional", "confirmed") for l in issued):
            r = desk.call("issue_alert", {"level": "update", **self._text("update", loc)})
            if r.get("issued"):
                calls.append({"tool": "issue_alert", "args": {"level": "update"}, "result": r})
        return calls

    @staticmethod
    def _text(level: str, loc: dict) -> dict:
        if not loc.get("located"):
            lines = ", ".join(f"{b['camera']} {b['bearing_deg']} deg" for b in loc.get("bearing_lines", []))
            if level == "update":
                return {"headline": "", "message": ""}
            return {"headline": "Smoke spotted; location not yet fixed",
                    "message": f"Verified smoke on bearing {lines}. Waiting for a second station to locate it."}
        place = loc["nearby"][0]["phrase"] if loc.get("nearby") else f"{loc['lat']}, {loc['lon']}"
        unc = loc["uncertainty_m"] / 1000
        return {"headline": f"{level.capitalize()}: smoke {place}",
                "message": f"Located at {loc['lat']}, {loc['lon']} by {', '.join(loc['cameras'])} "
                           f"with about {unc:.1f} km uncertainty."}


class MuseAgent:
    """Muse Spark drives the desk through tool calls (Responses API, function tools).
    The conversation history is kept client-side and sent in full each call, so
    nothing depends on server-side response storage."""
    name = "muse"

    def __init__(self, client=None, model: str | None = None, max_rounds: int = 12):
        from .models import client as default_client, reasoning_model

        self.client = client or default_client()
        self.model = model or reasoning_model()
        self.max_rounds = max_rounds
        self.history: list[dict] = []

    @staticmethod
    def _as_item(o) -> dict | None:
        """Model output item -> input item for the next request (reasoning items are dropped)."""
        kind = getattr(o, "type", None)
        if kind == "function_call":
            return {"type": "function_call", "call_id": o.call_id, "name": o.name, "arguments": o.arguments or "{}"}
        if kind == "message":
            text = "".join(getattr(c, "text", "") or "" for c in (getattr(o, "content", None) or []))
            return {"role": "assistant", "content": text} if text else None
        return None

    def handle(self, desk: FireDesk, event: Event) -> list[dict]:
        calls: list[dict] = []
        self.history.append({"role": "user", "content": f"Woken at {hhmm(event.epoch)}: " + "; ".join(event.what)})
        kw = dict(model=self.model, instructions=SYSTEM, tools=TOOLS)
        resp = None
        for _ in range(self.max_rounds):
            resp = self.client.responses.create(input=self.history, **kw)
            for o in resp.output:
                item = self._as_item(o)
                if item:
                    self.history.append(item)
            fcalls = [o for o in resp.output if getattr(o, "type", None) == "function_call"]
            if not fcalls:
                break
            for fc in fcalls:
                try:
                    args = json.loads(fc.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                res = desk.call(fc.name, args if isinstance(args, dict) else {})
                calls.append({"tool": fc.name, "args": args, "result": res})
                self.history.append({"type": "function_call_output", "call_id": fc.call_id,
                                     "output": json.dumps(res, default=str)})
        text = (getattr(resp, "output_text", "") or "") if resp else ""
        if text:
            calls.append({"tool": "_agent_says", "args": {}, "result": text.strip()})
        return calls


def run_replay(obs: FireObservations, desk: FireDesk, driver) -> dict:
    timeline = []
    for ev in events(obs):
        desk.now = ev.epoch
        calls = driver.handle(desk, ev)
        timeline.append({"time": hhmm(ev.epoch), "epoch": ev.epoch,
                         "elapsed_min": round((ev.epoch - obs.t_start) / 60, 1),
                         "trigger": ev.what, "calls": calls})
    return {"fire": obs.fire, "driver": driver.name, "muse": desk.muse.name,
            "t_start": obs.t_start, "t_end": obs.t_end, "timeline": timeline,
            "alerts": desk.alerts, "score": score(obs, desk)}


def score(obs: FireObservations, desk: FireDesk) -> dict:
    """Timing vs labelled first visible smoke, and final error vs the official point (if known)."""
    out = {}
    if obs.t_first_visible:
        for lvl in ("spotted", "provisional", "confirmed"):
            a = next((a for a in desk.alerts if a["level"] == lvl), None)
            out[f"{lvl}_min_after_first_smoke"] = round((a["epoch"] - obs.t_first_visible) / 60, 1) if a else None
    located = [a for a in desk.alerts if a.get("location")]
    if obs.ignition and located:
        last = located[-1]["location"]
        out["final_error_m"] = round(haversine_m(last["lat"], last["lon"], *obs.ignition))
        out["final_uncertainty_m"] = last["uncertainty_m"]
    return out

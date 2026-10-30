"""The agent's desk: state plus the tools it can call. All numbers come from
here; the agent decides what to do and writes the words. Guardrails stop it
from over-stating the evidence or citing numbers no tool produced."""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from typing import Callable

from .geo import bearing_deg, haversine_m
from .locate import Location, best_location
from .observations import CameraTrack, FireObservations
from .places import compass as _compass, nearby

LEVELS = ["spotted", "provisional", "confirmed"]
from zoneinfo import ZoneInfo

LOCAL = ZoneInfo("America/Los_Angeles")


def _rel(p) -> str:
    from .config import ROOT
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def hhmm(epoch: int) -> str:
    """Local wall-clock time (PST or PDT as appropriate)."""
    return dt.datetime.fromtimestamp(epoch, LOCAL).strftime("%H:%M %Z")


@dataclass
class MuseBackend:
    """Smoke checks. ``verify``/``recheck`` take a CameraTrack (+ context) and return
    a dict {is_smoke, confidence, reason} or None if unavailable."""
    verify: Callable[[CameraTrack], dict | None]
    recheck: Callable[[CameraTrack, str], dict | None]
    name: str = "muse"


def cached_backend() -> MuseBackend:
    """Offline: use verdicts cached by run_sam.py --verify; no rechecks."""
    return MuseBackend(verify=lambda t: t.verdict, recheck=lambda t, c: None, name="cached")


def live_backend() -> MuseBackend:
    import json as _json

    from . import muse

    def verify(t: CameraTrack):
        if t.verdict is not None:
            return t.verdict
        v = muse.verify_track(t.verify_frames(), t.verify_box(), at_edge=t.edge_clipped)
        d = {"is_smoke": v.is_smoke, "confidence": v.confidence, "reason": v.reason, "at_edge": t.edge_clipped}
        from .config import ROOT
        from .models import verdicts_name
        path = ROOT / "outputs" / t.sequence / verdicts_name()
        cache = _json.loads(path.read_text()) if path.exists() else {}
        cache[t.oid] = d
        path.write_text(_json.dumps(cache, indent=2))
        t.verdict = d
        return d

    def recheck(t: CameraTrack, context: str):
        ds = t.dets[:7]
        frames = [t.frames[d.frame].path for d in (ds[0], ds[len(ds) // 2], ds[-1])]
        x0, y0 = min(d.x0 for d in ds), min(d.y0 for d in ds)
        x1, y1 = max(d.x1 for d in ds), max(d.y1 for d in ds)
        v = muse.recheck_track(frames, (x0, y0, x1, y1), context)
        return {"is_smoke": v.is_smoke, "confidence": v.confidence, "reason": v.reason}

    return MuseBackend(verify=verify, recheck=recheck, name="live")


@dataclass
class FireDesk:
    obs: FireObservations
    muse: MuseBackend
    now: int = 0
    status: dict[str, dict] = field(default_factory=dict)   # track_id -> verification result
    alerts: list[dict] = field(default_factory=list)
    last_location: dict | None = None
    candidate: Location | None = None          # rough fix too uncertain to publish
    facts: list[float] = field(default_factory=list)         # numbers the agent may cite
    log: list[dict] = field(default_factory=list)
    sat_matches: dict = field(default_factory=dict)          # hotspot key -> match details
    sat_announced: set = field(default_factory=set)          # satellite systems already cited in an alert

    # ---------- helpers ----------
    def _track(self, track_id: str) -> CameraTrack | None:
        return next((t for t in self.obs.tracks if t.track_id == track_id), None)

    def _visible(self) -> list[CameraTrack]:
        return [t for t in self.obs.tracks if t.bearing_epoch <= self.now]

    def _accepted(self) -> list[CameraTrack]:
        return [t for t in self._visible()
                if self.status.get(t.track_id, {}).get("state") in ("smoke", "corroborated")]

    def _note(self, obj) -> None:
        """Record every number in a tool result as citable."""
        for m in re.findall(r"-?\d+(?:\.\d+)?", json.dumps(obj)):
            self.facts.append(float(m))

    def _elapsed(self) -> float:
        return round((self.now - self.obs.t_start) / 60, 1)

    MAX_PROVISIONAL_M = 10_000     # a pin vaguer than this is not a location
    MAX_CONFIRMED_M = 3_000

    def _eligible(self, loc: Location | None) -> str | None:
        if loc is not None and loc.uncertainty_m <= self.MAX_PROVISIONAL_M:
            k = len({m.station for m in loc.members})
            if loc.uncertainty_m <= self.MAX_CONFIRMED_M and (k >= 3 or loc.uncertainty_m <= 1500):
                return "confirmed"
            return "provisional"
        return "spotted" if self._accepted() else None

    # ---------- tools ----------
    def get_situation(self) -> dict:
        tracks = []
        for t in self._visible():
            st = self.status.get(t.track_id)
            if st:
                state = st["state"]
            elif t.verify_epoch > self.now:
                state = f"too new to verify (ready in {round((t.verify_epoch - self.now) / 60)} min)"
            else:
                state = "unverified"
            tracks.append({"track_id": t.track_id, "camera": t.camera_id,
                           "first_seen_min_ago": round((self.now - t.first_epoch) / 60),
                           "bearing_deg": round(t.bearing_deg, 1), "bearing_uncertainty_deg": t.sigma_deg,
                           "at_image_edge": t.edge_clipped, "status": state,
                           "reason": (st or {}).get("reason")})
        out = {"time": hhmm(self.now), "minutes_since_watch_started": self._elapsed(),
               "cameras_watching": self.obs.cameras, "plume_tracks": tracks,
               "alerts_issued": [{"level": a["level"], "time": a["time"], "headline": a["headline"]}
                                 for a in self.alerts],
               "last_location": self.last_location,
               "satellite_hotspots_reported": len(self._sat_available())}
        self._note(out)
        return out

    def verify_track(self, track_id: str) -> dict:
        t = self._track(track_id)
        if t is None or t.bearing_epoch > self.now:
            return {"error": f"unknown track {track_id}"}
        if t.verify_epoch > self.now:
            return {"error": f"not enough history yet; ready in {round((t.verify_epoch - self.now) / 60)} min"}
        if track_id in self.status:
            return {"track_id": track_id, **self.status[track_id], "note": "already checked"}
        v = self.muse.verify(t)
        if v is None:
            return {"track_id": track_id, "error": "smoke check unavailable"}
        self.status[track_id] = {"state": "smoke" if v["is_smoke"] else "not smoke",
                                 "confidence": v["confidence"], "reason": v["reason"]}
        res = {"track_id": track_id, **self.status[track_id]}
        self._note(res)
        return res

    def recheck_track(self, track_id: str) -> dict:
        t = self._track(track_id)
        if t is None or self.status.get(track_id, {}).get("state") != "not smoke":
            return {"error": "only tracks previously judged 'not smoke' can be rechecked"}
        acc = self._accepted()
        if not acc:
            return {"error": "no verified plume to compare against yet"}
        if self.last_location:
            ref_lat, ref_lon, ref_cams = self.last_location["lat"], self.last_location["lon"], self.last_location["cameras"]
        elif self.candidate is not None:
            ref_lat, ref_lon = self.candidate.fix.lat, self.candidate.fix.lon
            ref_cams = [m.camera_id for m in self.candidate.members]
        else:
            return {"error": "no located or roughly located fire yet; call locate first"}
        # would this plume fit a coherent location together with the verified ones?
        hyp = best_location([x.bearing() for x in acc] + [t.bearing()])
        fits = hyp is not None and any(m.track_id == track_id for m in hyp.members)
        to_fix = bearing_deg(t.lat, t.lon, ref_lat, ref_lon)
        miss = (t.bearing_deg - to_fix + 180) % 360 - 180
        res = {"track_id": track_id, "bearing_deg": round(t.bearing_deg, 1),
               "direction_to_fire_estimate_deg": round(to_fix, 1), "difference_deg": round(miss, 1),
               "bearing_uncertainty_deg": t.sigma_deg, "consistent": fits}
        if not fits:
            res["result"] = "does not line up with the verified plumes; not rechecked"
            self._note(res)
            return res
        context = (f"cameras {', '.join(ref_cams)} saw a fire in direction {to_fix:.0f} degrees from this "
                   f"camera, roughly {haversine_m(t.lat, t.lon, ref_lat, ref_lon) / 1000:.0f} km away")
        v = self.muse.recheck(t, context)
        if v is None:
            res["result"] = "recheck unavailable"
        elif v["is_smoke"]:
            self.status[track_id] = {"state": "corroborated", "confidence": v["confidence"], "reason": v["reason"]}
            res.update(result="corroborated: now used as supporting evidence; call locate again", reason=v["reason"])
        else:
            res.update(result="still not smoke", reason=v["reason"])
        self._note(res)
        return res

    def locate(self) -> dict:
        acc = self._accepted()
        if not acc:
            return {"error": "no verified plume yet"}
        loc = best_location([t.bearing() for t in acc])
        if loc is None:
            lines = [{"camera": t.camera_id, "bearing_deg": round(t.bearing_deg, 1),
                      "uncertainty_deg": t.sigma_deg} for t in acc]
            res = {"located": False, "reason": "plumes from fewer than two stations, or bearings that do not "
                   "intersect consistently", "bearing_lines": lines, "eligible_level": self._eligible(None)}
            self._note(res)
            return res
        stations = sorted({m.station for m in loc.members})
        self.candidate = loc
        if loc.uncertainty_m > self.MAX_PROVISIONAL_M:
            lines = [{"camera": t.camera_id, "bearing_deg": round(t.bearing_deg, 1),
                      "uncertainty_deg": t.sigma_deg} for t in acc]
            res = {"located": False, "reason": f"bearing lines cross, but too uncertain to place "
                   f"(about {round(loc.uncertainty_m / 1000)} km); need a camera from another direction",
                   "bearing_lines": lines, "eligible_level": self._eligible(loc)}
            self._note(res)
            return res
        self.last_location = {
            "lat": round(loc.fix.lat, 4), "lon": round(loc.fix.lon, 4),
            "uncertainty_m": int(round(loc.uncertainty_m, -1)), "cameras": [m.camera_id for m in loc.members],
            "track_ids": [m.track_id for m in loc.members], "stations": len(stations)}
        unused = [t.track_id for t in acc if t.track_id not in self.last_location["track_ids"]]
        res = {"located": True, **self.last_location, "unused_tracks": unused,
               "eligible_level": self._eligible(loc), "nearby": nearby(loc.fix.lat, loc.fix.lon)}
        self._note(res)
        return res

    # ---------- satellite cross-check ----------
    SAT_MAX_RANGE_M = 60_000       # a camera can't see smoke much farther than this

    def _sat_available(self) -> list:
        return [h for h in self.obs.hotspots if h.available_epoch <= self.now]

    def _new_sat_sources(self) -> set:
        """Satellite systems (GOES, polar) whose agreement hasn't been cited in an alert yet.
        One announcement per system: more pixels from a growing fire are not news."""
        return {k.split(":", 1)[0] for k in self.sat_matches} - self.sat_announced

    def check_satellite(self) -> dict:
        """NASA FIRMS hotspots reported so far that agree with the camera evidence. Never changes
        the alert level: it can only strengthen what an alert says."""
        avail = self._sat_available()
        note = ("Satellites pass a few times a day and often miss small or new fires, so no hotspot is "
                "not evidence against a fire. A matching hotspot supports the camera evidence; it does not "
                "change the alert level.")
        if not avail:
            res = {"hotspots_reported": 0, "matches": [], "note": note}
            self._note(res)
            return res
        matches = []
        fix = self.last_location
        import math
        seen = set()
        for h in reversed(avail):                       # newest first; one entry per sensor pixel
            src = "GOES" if h.satellite.startswith("GOES") else "polar"
            key = f"{src}:{h.lat:.3f}:{h.lon:.3f}" if src == "GOES" else f"{src}:{h.acq_epoch}:{h.lat:.4f}:{h.lon:.4f}"
            if key in seen:
                continue
            seen.add(key)
            m = None
            if fix:
                d = haversine_m(fix["lat"], fix["lon"], h.lat, h.lon)
                if d <= max(3000, 2 * fix["uncertainty_m"]) + h.pixel_m:
                    m = {"distance_from_fix_m": int(round(d, -1)),
                         "direction_from_fix": _compass(bearing_deg(fix["lat"], fix["lon"], h.lat, h.lon))}
            else:
                for t in self._accepted():
                    b = bearing_deg(t.lat, t.lon, h.lat, h.lon)
                    off = abs((b - t.bearing_deg + 180) % 360 - 180)
                    d = haversine_m(t.lat, t.lon, h.lat, h.lon)
                    slack = math.degrees(math.atan2(h.pixel_m, max(d, 1.0)))   # coarse pixels
                    if off <= max(2 * t.sigma_deg, 2.0) + slack and d <= self.SAT_MAX_RANGE_M:
                        m = {"on_bearing_of_camera": t.camera_id, "bearing_from_camera_deg": round(b, 1),
                             "offset_from_plume_bearing_deg": round(off, 1),
                             "distance_from_camera_km": round(d / 1000, 1)}
                        break
            if m:
                m.update({"observed": hhmm(h.acq_epoch), "satellite": h.satellite, "lat": round(h.lat, 4),
                          "lon": round(h.lon, 4), "confidence": h.confidence,
                          "pixel_size_m": int(h.pixel_m)})
                if h.persistent:
                    m["note"] = "this GOES pixel was already hot before the cameras saw smoke"
                near = nearby(h.lat, h.lon, k=1)
                if near:
                    m["near"] = near[0]["phrase"]
                matches.append(m)
                self.sat_matches[key] = m
        matches.sort(key=lambda m: ("note" in m, 0))        # fresh detections first
        res = {"hotspots_reported": len(seen), "matches": matches[:5],
               "unmatched_hotspots": len(seen) - len(matches),
               "new_since_last_alert": len(self._new_sat_sources()),
               "note": note}
        self._note(res)
        return res

    def nearby_places(self, lat: float, lon: float) -> dict:
        res = {"places": nearby(lat, lon)}
        self._note(res)
        return res

    def issue_alert(self, level: str, headline: str, message: str) -> dict:
        if level not in LEVELS + ["update"]:
            return {"error": f"level must be one of {LEVELS + ['update']}"}
        acc = self._accepted()
        loc = best_location([t.bearing() for t in acc]) if acc else None
        eligible = self._eligible(loc)
        if eligible is None:
            return {"error": "no verified smoke: nothing to alert on"}
        issued = [a["level"] for a in self.alerts]
        new_sat = self._new_sat_sources()
        if level == "update":
            if not self.alerts:
                return {"error": "update needs an earlier alert"}
            if new_sat:
                pass                          # new satellite corroboration is a real change
            elif self.last_location is None:
                return {"error": "update needs a location, or a new satellite hotspot matching the cameras"}
            prev = next((a for a in reversed(self.alerts) if a.get("location")), None)
            if prev and not new_sat:
                moved = haversine_m(prev["location"]["lat"], prev["location"]["lon"],
                                    self.last_location["lat"], self.last_location["lon"])
                shrank = self.last_location["uncertainty_m"] <= 0.7 * prev["location"]["uncertainty_m"]
                if moved < 300 and not shrank:
                    return {"error": "location has not changed enough for an update"}
        else:
            if LEVELS.index(level) > LEVELS.index(eligible):
                return {"error": f"evidence supports at most '{eligible}', not '{level}'"}
            if level in issued:
                return {"error": f"a '{level}' alert was already issued; use 'update' for changes"}
        text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", headline + " " + message)   # 1,010 -> 1010
        bad = [n for n in re.findall(r"\d+(?:\.\d+)?", text) if not self._citable(float(n))]
        if bad:
            return {"error": f"numbers not produced by any tool: {bad}. Only cite tool outputs."}
        alert = {"level": level, "time": hhmm(self.now), "epoch": self.now, "elapsed_min": self._elapsed(),
                 "headline": headline, "message": message,
                 "location": self.last_location if level != "spotted" else None,
                 "bearing_lines": [{"camera": t.camera_id, "lat": t.lat, "lon": t.lon,
                                    "bearing_deg": round(t.bearing_deg, 2), "sigma_deg": t.sigma_deg,
                                    "track_id": t.track_id} for t in acc],
                 "evidence": [{"track_id": t.track_id, "frames": [_rel(p) for p in t.verify_frames()],
                               "box": [round(v) for v in t.verify_box()],
                               "status": self.status[t.track_id]} for t in acc]}
        if self.sat_matches:
            alert["satellite"] = list(self.sat_matches.values())[-5:]
            self.sat_announced |= {k.split(":", 1)[0] for k in self.sat_matches}
        self.alerts.append(alert)
        return {"issued": True, "level": level, "alert_number": len(self.alerts)}

    def _citable(self, n: float) -> bool:
        if n <= 4 and float(n).is_integer():
            return True                      # small counts ("2 cameras", "3 stations")
        n = abs(n)                           # the minus sign is not part of the matched number
        for f in self.facts:
            f = abs(f)
            for g in (f, f / 1000):          # metres may be quoted in km
                if abs(n - g) <= max(0.06 * abs(g), 0.06) or round(g) == n or round(g, 1) == n:
                    return True
        return False

    # ---------- tool registry for the LLM ----------
    def call(self, name: str, args: dict) -> dict:
        fn = {"get_situation": self.get_situation, "verify_track": self.verify_track,
              "recheck_track": self.recheck_track, "locate": self.locate,
              "nearby_places": self.nearby_places, "issue_alert": self.issue_alert,
              "check_satellite": self.check_satellite}.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}


TOOLS = [
    {"type": "function", "name": "get_situation",
     "description": "Current time, cameras, every plume track seen so far with bearing, uncertainty and "
                    "verification status, alerts already issued, and the last computed location.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "verify_track",
     "description": "Ask the vision model whether a plume track is wildfire smoke (vs cloud, haze, glare), "
                    "using its first few minutes of images.",
     "parameters": {"type": "object", "properties": {"track_id": {"type": "string"}}, "required": ["track_id"]}},
    {"type": "function", "name": "recheck_track",
     "description": "Re-examine a track judged 'not smoke' that may belong to the fire (located or roughly "
                    "located). Only re-examined if it fits a coherent location with the verified plumes; can turn "
                    "it into supporting evidence, e.g. a view from another direction.",
     "parameters": {"type": "object", "properties": {"track_id": {"type": "string"}}, "required": ["track_id"]}},
    {"type": "function", "name": "locate",
     "description": "Triangulate the fire from all verified plume tracks (robust to a bad camera). Returns "
                    "position, uncertainty in metres, cameras used, nearby places and the highest alert level "
                    "the evidence supports.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "nearby_places",
     "description": "Nearest towns (and peak) to a point, with distance and direction.",
     "parameters": {"type": "object", "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}},
                    "required": ["lat", "lon"]}},
    {"type": "function", "name": "check_satellite",
     "description": "NASA FIRMS satellite hotspots reported so far near the cameras, and which agree with the "
                    "camera evidence (near the fix, or on a verified plume's bearing). Supports an alert's "
                    "message; never raises the alert level. No hotspot is not evidence against a fire.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "issue_alert",
     "description": "Publish an alert to dispatchers. level: 'spotted' (smoke seen, direction only), "
                    "'provisional' (located, uncertain), 'confirmed' (located by 3+ stations or tight "
                    "uncertainty), 'update' (location changed or tightened). Only cite numbers from tool outputs.",
     "parameters": {"type": "object", "properties": {
         "level": {"type": "string", "enum": ["spotted", "provisional", "confirmed", "update"]},
         "headline": {"type": "string", "description": "One line, under 90 characters."},
         "message": {"type": "string", "description": "2-4 sentences for a fire dispatcher."}},
         "required": ["level", "headline", "message"]}},
]

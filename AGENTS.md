# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository. Read it before changing anything.
`README.md` is for users and judges; this file is for whoever is modifying the code.

## Project in one paragraph

FireFinder detects wildfire smoke in fixed mountaintop camera footage (HPWREN), works out where each
camera points by matching the terrain skyline, triangulates the fire from several cameras, and runs a
Muse Spark agent that verifies plumes and issues escalating alerts. Every Muse call can also run on the
open-weight **Muse Spark Open** (Baseten) with `MUSE_PROVIDER=open` or `--provider open`, and smoke checks
on **Muse Glimmer 30B** (local Ollama) with `--provider glimmer`. The Baseten key works only with
`meta-models/muse-spark-open-1026`; don't try to deploy other models with it. It is a Meta Global AI Developer
Hackathon entry (Indie track; submission due **Nov 4, 2026, 9 PM ET**). Fires are replayed from archives;
there is no live feed yet.

## Pipeline and where it lives

| Stage | Code | Output |
|---|---|---|
| Download footage | `scripts/download_figlib.py` | `data/raw/<sequence>/` |
| SAM 3.1 segmentation (+ Muse check) | `scripts/run_sam.py`, `smokespotter/sam.py`, `tracks.py`, `muse.py` | `outputs/<seq>/sam_*.txt`, `dets_*.json`, `results.json`, `verdicts.json` |
| Elevation data | `scripts/download_dem.py`, `smokespotter/dem.py` | `data/dem/region.npz` |
| Skyline calibration | `scripts/calibrate_skyline.py`, `smokespotter/skyline.py`, `camera.py` | `outputs/<seq>/camera_model.json`, `data/lens.json` |
| Station position refinement | `scripts/refine_stations.py` | updates `camera_model.json`, `data/stations_refined.csv` |
| Triangulation | `smokespotter/locate.py`, `scripts/locate.py` | `outputs/<fire>_location.geojson` |
| Agent replay | `smokespotter/observations.py`, `desk.py`, `agent.py`, `scripts/run_agent.py` | `outputs/<fire>_timeline.json` (live), `_timeline_autopilot.json` (offline) |
| Open-weights runs | `smokespotter/models.py` (provider switch, cost meter), `agent.OpenMuseAgent`, `scripts/check_open_model.py`, `eval_verifier.py`, `compare_models.py` | `outputs/<seq>/verdicts_open.json`, `<fire>_timeline_open_r*.json`, `verifier_eval_open.json`, `model_comparison.json`, `usage_open.jsonl` |
| Satellite cross-check | `scripts/download_firms.py`, `scripts/download_goes.py`, `smokespotter/satellite.py`, `goes.py`, `desk.check_satellite`, `scripts/compare_firms.py` | `data/firms/<fire>.csv`, `outputs/firms_comparison.json` |
| Latency and cost | every model call → `outputs/usage_<provider>.jsonl`; `scripts/benchmark_latency.py`, `scripts/usage_report.py` | `outputs/latency_benchmark.json`, `outputs/usage_report.json` |
| Replay page | `scripts/export_ui.py`, `ui/index.html` | `ui/data/` |
| Demo slides | `ui/slides.html` (numbers typed in by hand from `model_comparison.json`) | |

Registry files: `data/fires.csv` (fire → camera sequences, role), `data/stations.csv` (station
coordinates, approximate), `data/ignitions.csv` (official origins with sources; **scoring only**).

## Setup and everyday commands

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                                   # must pass before you finish (currently 27 tests)
python scripts/run_agent.py --fire Kitchenfire --autopilot   # offline smoke test, no API calls
python -m http.server -d ui 8000                      # replay page
```

The offline check above must still end with a confirmed fix about **119 m** from the official point. If a
change moves that number, say so explicitly and explain why.

## Guardrails (do not break these)

### 1. Evaluation integrity
- **Official ignition points and FIgLib smoke-onset offsets are for scoring only.** Never use
  `data/ignitions.csv` or `Frame.offset` in detection, calibration, track selection, verification or agent
  decisions. (`calibrate.py` is the legacy fire-based calibration and is the one deliberate exception;
  results calibrated that way are not independent tests.)
- **Do not tune on the fire you report.** Thresholds, prompts and parameters are chosen on other fires or
  on calibration quality, never by checking which value makes a reported fire's error smaller.
- **Report failures honestly.** Misses (Bridge Fire) and refusals to place (Grove, Airport) are results, not
  bugs to hide. Never loosen a threshold just to make a fire "pass".
- Keep the README results table in sync with what the code actually produces.

### 2. The agent's guardrails live in code, not prompts
- In `smokespotter/desk.py`, code computes every number (bearings, coordinates, distances, uncertainty);
  the model only chooses tools and writes text.
- `issue_alert` must keep rejecting: a level above what the evidence supports, a repeated level, an
  "update" without a real change, and any number not produced by a tool (`_citable`).
- Alert levels: *spotted* needs a verified plume; *provisional* needs a fix within 10 km uncertainty
  (`MAX_PROVISIONAL_M`); *confirmed* needs ≤ 3 km and either 3+ stations or ≤ 1.5 km. Changing these is a
  policy change: flag it, don't slip it in.
- Smoke verification judges only a track's **first few frames** (indices 0, 2, 4). A live system cannot see
  the future; don't feed later frames to verification.
- Rechecks (`recheck_track`) only run for tracks that fit a coherent location with already verified plumes.
- **Satellite hotspots (NASA FIRMS) support, never decide.** `check_satellite` only reports hotspots published
  by the replay clock (observation time + `FIRMS_LATENCY_S`, default 60 s per NASA's US ultra-real-time feed),
  matches them only to camera evidence (near the published fix, or on a verified plume's bearing), and never
  raises an alert level. A new match may justify one `update` per satellite system (GOES, polar). GOES pixels
  count from their file's measured S3 publication time (`scripts/download_goes.py`); pixels already hot before
  the replay are `persistent` and never wake the agent. Don't change latencies to make a fire work out.
- FIRMS downloads cover a box around the fire's **cameras**, never the official origin, so the agent's
  satellite view can't leak the answer. The origin is used only in `compare_firms.py` (scoring).

### 3. Cost and external services
- **SAM 3.1 and Muse Spark calls cost money and time.** Results are cached in `outputs/`. Don't pass
  `--refresh` or `--reverify`, and don't run live agent runs, unless the user asks. Use `--autopilot` and
  cached verdicts to test logic.
- **The Baseten credit is limited ($100).** Every open-weights call is metered in `outputs/usage_open.jsonl`
  and refused once the total reaches `OPEN_MODEL_BUDGET_USD` (default 90). Never raise the cap, delete the log,
  or start multi-run evaluations without the user asking. A full five-fire, five-run evaluation costs about $7.
- **Keep the two models' results apart.** Muse Spark 1.3 verdicts live in `verdicts.json`, Muse Spark Open's in
  `verdicts_open.json` (`models.verdicts_name()`). Never copy one into the other or compare runs that mixed them.
- **Muse Spark Open reasons before answering.** Keep `OPEN_MODEL_MAX_TOKENS` generous (default 8192) and keep the
  retry on empty replies in `muse._ask_verdict`; an empty reply is a transport failure, never a "not smoke" verdict.
- **HPWREN is a research server.** Download only through `scripts/download_figlib.py` (one request at a
  time, delays, backoff, stops on 403). No parallel or ad-hoc scraping.
- Never print, log or commit API keys. `.env` is git-ignored; `.env.example` holds names only.

### 4. Data and caches
- Don't delete or regenerate `data/raw/`, `data/dem/` or `outputs/` caches without being asked; they take
  hours or API spend to rebuild.
- `.gitignore` deliberately commits a small whitelist from `outputs/` (SAM text, decoded detections,
  camera models, verdicts, results, timelines) so others can reproduce without an API key. Keep new large
  files (videos, overlays, diagnostics) out of git.
- **Rerunning `calibrate_skyline.py` overwrites station refinements** in `camera_model.json`. After
  recalibrating, rerun `refine_stations.py` for the affected stations.
- FIgLib folder names are not always right: `20240909_BridgeFire_*` actually shows the **Airport Fire**
  (registered as `AirportFire`). Trust timestamps and geometry over folder names.
- If you change results, update `ui/slides.html` and the README tables by hand; they are not generated.
- Keep attribution intact (HPWREN, Copernicus, GeoNames, CAL FIRE, USGS) in the README and on the page.

### 5. Git
- Commit or push only when the user asks. Work on the default branch only if the user does.

## Conventions

- Python 3.10+, standard library + the packages in `requirements.txt`. Ask before adding a dependency.
- Units: angles in **degrees**, bearings **clockwise from true north**; distances in **metres**; times as
  **Unix epoch seconds** internally, shown in `America/Los_Angeles` (PST/PDT via `desk.hhmm`).
- Image coordinates are **original-resolution pixels** (3072 × 2048 for these cameras) unless a variable
  says otherwise; SAM runs on downscaled tiles and results are mapped back.
- Camera IDs are `<station>-<direction>` (e.g. `hp-e`); track IDs are `<camera>/<tile>:<sam id>`.
- Bearing uncertainty (`observations.sigma_for`): 1° for good calibrations, 2.5° near image edges, 6° for
  weak skyline fits. Weak = residual > 2 px or edge coverage < 75%.
- Add or update a unit test with any change to geometry, calibration, triangulation or desk logic. Tests
  must not need network, API keys or downloaded data.
- `ui/index.html` is a single self-contained page with no build step; external libraries are pinned
  (`maplibre-gl@4.7.1`). Check UI changes through `python -m http.server`, at desktop and phone widths.

## Known limitations (context, not bugs to "fix" quietly)

- Fisheye cameras (Volcan, High Point south/north) and views blocked by towers or domes calibrate poorly;
  they are kept at low trust (6°).
- Distant early plumes are only a few pixels wide, so detection can take 10–20 minutes per camera.
- Two cameras looking along nearly the same line cannot place a fire; the system declines instead of
  guessing.
- Station coordinates from HPWREN are approximate (±0.005°); refinement helps but doesn't fully fix them.

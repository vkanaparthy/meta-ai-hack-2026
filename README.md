# FireFinder

> 🔥 Turn any fixed mountaintop camera into a calibrated fire finder: detect wildfire smoke with Meta's
> models, work out where every camera is pointing from the shape of the mountains, and triangulate the
> fire, with an agent that verifies, locates and writes the alert.

* * *

## Quick Start: Run the App Locally

The app is a replay page that runs in your browser. It needs only Python (no packages, no API key, no
data downloads), because the replay data for all six fires is included in the repo.

```
git clone https://github.com/vkanaparthy/meta-ai-hack-2026.git
cd meta-ai-hack-2026
python3 -m http.server -d ui 8000
```

Then open **http://localhost:8000** in your browser, pick a fire at the top, and press **Space** to play.
Stop the server with **Ctrl+C**.

To rerun the analysis itself rather than just watch it, see [Installation](#installation) and
[Running Locally](#running-locally).

* * *

## What It Does

Fire lookouts have located fires the same way for a century: two towers take a bearing on the same smoke,
and where the lines cross is the fire. FireFinder automates that with existing camera networks,
with no training data and no surveying.

1. **Watches** each camera's image stream (one frame a minute, replayed from the HPWREN archive).
2. **Segments and tracks smoke** with **SAM 3.1**, as video, on full-resolution horizon crops.
3. **Verifies** each plume with **Muse Spark** (smoke vs. cloud, haze, glare), judging only from the first
   few minutes, as a live system would.
4. **Calibrates every camera from the skyline**: a terrain skyline rendered from a 30 m elevation model is
   aligned with the sky/ridge edge in the image, giving each camera's heading, tilt, roll and lens. No fire
   data is used, so every fire is an independent test.
5. **Triangulates** plume bearings from different stations, robust to a bad camera, with an honest error circle.
6. **Runs the dispatch desk with an agent**: Muse Spark calls tools to verify, recheck doubted cameras that line
   up with the fire, locate, and issue alerts that escalate *spotted → provisional → confirmed*. Guardrails in
   code stop it from overstating the evidence or citing any number a tool didn't produce.
7. **Replays** each fire on a map: fire-finder rings at each station, bearing lines, the pin and error circle,
   live camera frames with detection boxes, and every step the agent took.

**Results on six archived fires** (live Muse Spark agent; error vs. the official CAL FIRE ignition point):

| Fire | First alert | Location | Notes |
|---|---|---|---|
| Kitchen (Jul 2024) | 7 min | **119 m** off, confirmed | spotted → provisional → confirmed |
| Junction (Jun 2026) | 19 min | 1.9 km off, confirmed | located only after the agent rechecked a doubted camera |
| Resort (Nov 2024) | 20 min | 1.0 km off, confirmed | |
| Grove (Jul 2024) | 7 min | declined to place | rejected 3 pre-ignition cloud tracks; the two cameras are nearly in line with the fire |
| Airport (Sep 2024) | 6 min after official start | declined to place | camera geometry exposed a mislabeled archive sequence |
| Bridge (Sep 2024) | missed | | a faint wisp under thunderstorms, 28 km away |

No false alerts on any fire. Times are measured from the first frame where smoke is visible in any camera.

**Same results on Meta's open weights.** With every Muse call moved to **Muse Spark Open** on Baseten, five
replays of each fire gave the same alerts and the same locations as the hosted Muse Spark 1.3. On 32 hard
negatives (the same crop from the same camera before smoke appeared) it raised 0 false alarms in 96 checks.
The whole evaluation cost $6.64. Details and commands: [path D](#d-run-on-muse-spark-open-open-weights).

* * *

## Technology Stack

| Layer | Technology |
|---|---|
| Smoke segmentation and tracking | **SAM 3.1** (Meta Model API), parsed with `meta-sam-parser` |
| Smoke verification, rechecks, alert writing | **Muse Spark** (`muse-spark-1.3`, Meta Model API) |
| Agent | Muse Spark function calling (Responses API) over a tool "desk" with guardrails |
| Camera calibration | Skyline matching against Copernicus GLO-30 elevation (NumPy, SciPy, rasterio) |
| Triangulation | Robust least squares with inlier/outlier selection and Monte Carlo uncertainty |
| Replay page | Plain HTML/JS, MapLibre GL, USGS National Map basemap |
| Camera data | HPWREN Fire Ignition images Library (FIgLib) |
| Open weights | **Muse Spark Open** (`meta-models/muse-spark-open-1026`) on Baseten, Chat Completions: smoke checks, rechecks and the agent |

* * *

## Prerequisites

- **Python 3.10 or newer**: `python3 --version`
- **git**: `git --version`
- **About 1 GB of free disk** for footage (path B), about 2 GB for the full pipeline (path C)
- **Meta Model API key**: only for path C (live SAM 3.1 and Muse Spark calls)
- **Baseten API key**: only for path D (Muse Spark Open)

ffmpeg is installed automatically through the `imageio-ffmpeg` package.

* * *

## Installation

### 1. Clone the repository

```
git clone https://github.com/vkanaparthy/meta-ai-hack-2026.git
cd meta-ai-hack-2026
```

### 2. Create a virtual environment and install dependencies

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download data (paths B and C only)

Camera footage for the six fires (public, from HPWREN; one archive per camera, downloaded politely):

```
python scripts/download_figlib.py --fire Kitchenfire --fire JunctionFire --fire ResortFire \
    --fire GroveFire --fire AirportFire --fire BridgeFire
```

Only for path C (calibrating cameras yourself), elevation tiles (~500 MB) and place names:

```
python scripts/download_dem.py
python scripts/download_places.py
```

* * *

## Environment Variables

### Meta Model API (`.env`)

```
cp .env.example .env
```

Set `MODEL_API_KEY`. The other values have working defaults (`https://api.meta.ai/v1`, `muse-spark-1.3`,
`sam-3.1`). **Never commit `.env`**; it is git-ignored.

Paths A and B need no key.

### Open weights on Baseten (`.env`, path D only)

```
OPEN_MODEL_API_KEY=<your Baseten key>
OPEN_MODEL_NAME=meta-models/muse-spark-open-1026
OPEN_MODEL_BASE_URL=https://inference.baseten.co/v1
OPEN_MODEL_BUDGET_USD=90
```

Every call to the open-weights endpoint is logged with its tokens and cost in `outputs/usage_open.jsonl`, and
calls stop once the total reaches `OPEN_MODEL_BUDGET_USD`. Set `MUSE_PROVIDER=open` to make it the default,
or pass `--provider open` per command.

* * *

## Running Locally

### A. See it: the replay page (no key, no downloads)

From the repository folder:

```
python3 -m http.server -d ui 8000
```

Open **http://localhost:8000**. (On Windows, use `python` instead of `python3`.)

| Control | What it does |
|---|---|
| Fire buttons (top) | switch between the six fires |
| **Space** or the play button | play or pause the replay |
| Speed menu | 15× to 120× (at 30×, an 80-minute fire plays in under 3 minutes) |
| Timeline | drag to scrub; colored notches mark alerts, the brown tick marks when smoke first became visible |
| Camera tiles (bottom) | click for a large live view with detection boxes; that camera's bearing and field of view are highlighted |
| **Satellite** | switch the base map from USGS topo to imagery |
| **Official origin** | show CAL FIRE's ignition point (diamond) to compare with the pin |

On the map, each station is a fire-finder ring with its cameras' fields of view as brown arcs. Bearing lines
appear once Muse Spark verifies a plume (dark: verified, blue: corroborated on a recheck, dashed gray:
doubted). The pin and error circle appear with provisional and confirmed alerts. The right panel shows the
scorecard, the alerts, and every step the agent took.

Good moments to watch: **Junction** from 17:26 (a doubted camera is rechecked and turns blue, then the pin
drops) and **Kitchen** from 10:35 (a clean spotted → provisional → confirmed sequence).

**Troubleshooting**
- **Open it through the server, not by double-clicking `index.html`.** Browsers block a page opened as a
  file from loading its data.
- **Port 8000 already in use:** pick another, e.g. `python3 -m http.server -d ui 8080`, and open that port.
- **Blank map background or plain fonts:** the base map, fonts and map library load from the internet, so
  check your connection. The replay itself still works.
- **"replay data is missing":** you're running from a checkout without `ui/data`; pull the latest, or
  regenerate it with `python scripts/export_ui.py` (needs the downloads from step 3).
- **After rerunning the agent**, refresh the data with `python scripts/export_ui.py --fire <name>` and reload
  the page (Cmd+Shift+R or Ctrl+Shift+R to bypass the cache).

### B. Reproduce the results offline (no key)

The repo includes the SAM 3.1 outputs, Muse Spark verdicts and skyline calibrations, so after step 3 you can
rerun triangulation and the agent's decision logic without any API calls:

```
python scripts/locate.py --fire Kitchenfire
python scripts/run_agent.py --fire Kitchenfire --autopilot
```

`--autopilot` drives the same tools and guardrails with a fixed policy instead of Muse Spark, using the
cached verdicts. Fires: `Kitchenfire`, `JunctionFire`, `ResortFire`, `GroveFire`, `AirportFire`, `BridgeFire`.

Offline runs use only the first-pass smoke checks. On the Junction Fire, the live agent also rechecked a
camera that Muse had first doubted, and that corroborated view is what tightens the fix from about 3.8 km
to 1.9 km; every step of that live run is recorded in `outputs/JunctionFire_timeline.json`.

### C. Run the full pipeline (Meta API key)

```
python scripts/run_sam.py --verify --tiles 2 --band 0.3:0.7 data/raw/20240701_Kitchenfire_*   # SAM 3.1 + Muse checks
python scripts/calibrate_skyline.py --fire Kitchenfire                                        # camera calibration
python scripts/run_agent.py --fire Kitchenfire --verbose                                      # live Muse Spark agent
python scripts/export_ui.py --fire Kitchenfire                                                # refresh the replay page
```

SAM output and Muse verdicts are cached in `outputs/`, so reruns are free unless you pass `--refresh` or `--reverify`.

### D. Run on Muse Spark Open (open weights)

The same desk, tools and guardrails, with every smoke check, recheck and agent turn on Muse Spark Open.
SAM 3.1 stays on the Meta API, but its outputs are already cached, so this path needs only a Baseten key.

```
python scripts/check_open_model.py                                   # text, image and tool call (< $0.01)
python scripts/eval_verifier.py --provider open --repeats 3          # smoke checks incl. hard negatives
python scripts/run_agent.py --provider open --runs 5 --fire Kitchenfire --fire ResortFire \
    --fire JunctionFire --fire GroveFire --fire AirportFire          # five replays per fire
python scripts/compare_models.py                                     # scorecard vs Muse Spark 1.3
```

Open-model verdicts are cached separately (`verdicts_open.json`), so they never mix with Muse Spark 1.3's.
Our results are committed: `outputs/model_comparison.json`, `outputs/verifier_eval_open.json` and
`outputs/<fire>_timeline_open_r1..r5.json`.

| Fire | Muse Spark 1.3 (1 run) | Muse Spark Open (5 runs) |
|---|---|---|
| Kitchen | confirmed, 119 m | confirmed 5/5, 119 m |
| Resort | confirmed, 1.0 km | confirmed 5/5, 1.0 km |
| Junction | confirmed, 1.9 km | confirmed 5/5, 1.9 km |
| Grove | declined to place | declined 5/5 |
| Airport | spotted only | spotted only 5/5 |

Smoke checks agreed with Muse Spark 1.3 on 21 of 22 real plumes; the one difference is Junction's edge-clipped
`hp-e` view, which the open model called smoke and which was the fire. One lesson: Muse Spark Open reasons before
it answers, so it needs a generous output cap (`OPEN_MODEL_MAX_TOKENS`, default 8192); with a small one, 27% of
replies came back empty.

* * *

## Verify the Setup

1. Run the unit tests (no data or key needed):

```
python -m pytest -q
```

Expected:

```
27 passed
```

2. After step 3, run the offline agent on the Kitchen Fire:

```
python scripts/run_agent.py --fire Kitchenfire --autopilot
```

Expected (abridged):

```
10:37 PDT  SPOTTED     Smoke spotted; location not yet fixed
10:49 PDT  PROVISIONAL Provisional: smoke 11.8 km SSE of Pine Valley @ 32.7227, -116.484 +-1640 m
11:07 PDT  CONFIRMED   Confirmed: smoke 12.4 km N of Campo @ 32.7181, -116.4709 +-1480 m
score: {"spotted_min_after_first_smoke": 7.2, ..., "final_error_m": 119, "final_uncertainty_m": 1480}
```

✅ A confirmed fix about 119 m from CAL FIRE's ignition point means everything is wired up.

* * *

## Project Structure

```
smokespotter/          core package
  models.py              model clients: Meta Model API, Muse Spark Open endpoint, cost meter
  sam.py, tracks.py      SAM 3.1 calls, mask parsing, plume tracks
  muse.py                Muse Spark smoke checks and rechecks
  camera.py, dem.py      camera model (heading, tilt, roll, lens) and elevation/skyline rendering
  skyline.py             skyline calibration, shared-lens fit
  geo.py, locate.py      bearings, robust triangulation, uncertainty
  observations.py        everything the agent can observe about a fire
  desk.py, agent.py      agent tools + guardrails, replay clock, Muse agents (1.3 and Open) and autopilot
  places.py              nearest towns for plain-language alerts
scripts/               command-line steps (download, run_sam, calibrate_skyline, locate, run_agent, export_ui,
                       check_open_model, eval_verifier, compare_models, ...)
data/                  fires.csv, stations.csv, ignitions.csv (official origins with sources), lens.json
outputs/               cached SAM outputs, Muse verdicts, camera models, agent timelines
ui/                    replay page (index.html), its exported data, and slides.html for the demo video
tests/                 unit tests, including a synthetic-terrain calibration round trip
```

* * *

## Planned Features (Future Phases)

- **Muse Glimmer at the edge**: smoke checks on the 30B open model next to the camera, scored with
  `eval_verifier.py` against Muse Spark Open
- **Live mode**: poll current HPWREN images every minute (e.g. a scheduled Modal job) instead of replaying archives
- **Faster detection of distant plumes**: tighter crops and tracking SAM backwards from a confirmed plume
- **Calibration for fisheye cameras** whose skylines are too featureless for the current lens model
- **Satellite cross-check**: confirm or flag detections against NASA FIRMS hotspots
- **A fine-tuned smoke checker**: Glimmer trained on FIgLib, using ignition timestamps as free labels

* * *

## Hackathon

Built for the **Meta Global AI Developer Hackathon** (October 26 to November 4, 2026), **Indie Developer**
track, theme **Push the Frontier**. Bonus categories in view: **Best Agentic Use-Case** and
**Best Build on Meta Open Weights**.

* * *

## Data Credits

- Fire imagery: **HPWREN** Fire Ignition images Library, http://hpwren.ucsd.edu/
- Elevation: **Copernicus DEM GLO-30**, © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018,
  provided under COPERNICUS by the European Union and ESA
- Place names: **GeoNames** (CC BY 4.0)
- Ignition points: **CAL FIRE** and InciWeb incident records (sources listed in `data/ignitions.csv`)
- Base map: **USGS The National Map**

* * *

## License

Code is released under the [MIT License](LICENSE), © 2026 Venu Kanaparthy. Camera imagery, terrain, ignition and place data keep their own terms; see Data Credits.

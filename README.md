# Smoke Spotter

Meta Global AI Developer Hackathon 2026 (Indie developer track, theme: Push the Frontier).

An agent that watches fixed mountaintop fire cameras, detects new wildfire smoke,
tracks it over time, and triangulates where the fire started from two or more
cameras, then drafts a verified alert with evidence.

## How it works

1. **Frames:** each camera publishes a still image about once a minute. Archive
   replay (FIgLib) and live mode share one pipeline.
2. **Candidates:** cheap change detection flags regions that changed (`smokespotter/detect.py`).
3. **Segment and track:** SAM 3.1 outlines the plume and follows it across frames.
4. **Verify:** Muse Spark rejects fog, haze and glare and matches plumes across cameras.
5. **Locate:** pixel column to compass bearing, then least-squares intersection (`smokespotter/geo.py`).
6. **Alert:** the agent writes a short alert with location, error radius and evidence.

| Feature | Meta tech |
|---|---|
| Plume segmentation and tracking | SAM 3.1 |
| Smoke vs. look-alike check, cross-camera matching, alert writing, agent loop | Muse Spark |
| One pipeline step on open weights | Muse Spark Open or Muse Glimmer (Modal/Baseten) |
| Optional illustrative visuals | Muse Image / Muse Wonder |

## Layout

```
data/fires.csv        multi-station fires from FIgLib (showcase / dev / test split)
data/stations.csv     station coordinates (to fill in)
data/cameras.csv      per-camera heading and field of view (to calibrate)
data/raw/             downloaded sequences (git-ignored)
scripts/download_figlib.py   fetch sequences
scripts/baseline_detect.py   day-one baseline: time-to-detect with change detection
scripts/run_sam.py           SAM 3.1 tracking (+ --verify with Muse Spark), metrics, overlays
smokespotter/         core package (frames, detect, geo, models)
tests/                unit tests
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/download_figlib.py --fire JunctionFire --every 2
python scripts/baseline_detect.py data/raw/20260629_JunctionFire_*   # change-detection baseline
python scripts/run_sam.py data/raw/20260629_JunctionFire_*            # SAM 3.1
python scripts/run_sam.py --verify data/raw/20260629_JunctionFire_*   # SAM 3.1 + Muse Spark check
```

Copy `.env.example` to `.env` and set `MODEL_API_KEY`. Results go to `outputs/<sequence>/`
(`results.json`, `first_detection.jpg`); SAM output and Muse verdicts are cached there,
so reruns are free unless you pass `--refresh`.

## Evaluation

- **Time to detect:** seconds after the labelled first-visible-smoke frame (FIgLib file names carry this offset).
- **False alarms:** detections in the 40 minutes before ignition.
- **Location error:** metres between the triangulated fix and the official ignition point.
- Fires are split into `dev` (tune on) and `test` (report on) in `data/fires.csv`.

## Data credit

Fire imagery: HPWREN Fire Ignition images Library, http://hpwren.ucsd.edu/

"""Download GOES fire detections (ABI L2 FDCC, every 5 minutes) for each replay window, from
NOAA's public AWS bucket (no key), and keep only fire pixels around the fire's cameras.

  python scripts/download_goes.py
  python scripts/download_goes.py --fire AirportFire

Each scan's "available" time is the file's publication time in the bucket (S3 LastModified):
measured, not assumed. Scans from 30 minutes before the replay starts are included so pixels that
were already hot (solar farms and other recurring false alarms) can be recognized.
Raw files go to data/goes/raw/ (git-ignored); fire pixels to data/goes/<fire>.csv. Needs h5py.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smokespotter.config import ROOT  # noqa: E402
from smokespotter.goes import read_fires  # noqa: E402
from smokespotter.observations import load_fire  # noqa: E402

# GOES-West first (best view of California); GOES-East also scans all of CONUS, as a fallback
BUCKETS = [("GOES-18", "https://noaa-goes18.s3.amazonaws.com"), ("GOES-19", "https://noaa-goes19.s3.amazonaws.com")]
PRODUCT = "ABI-L2-FDCC"
FIRES = ["Kitchenfire", "ResortFire", "JunctionFire", "GroveFire", "AirportFire", "BridgeFire"]
BASELINE_S = 30 * 60
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "firefinder-hackathon/0.1 (research)"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    return b""


def scan_start(key: str) -> int:
    s = key.rsplit("_s", 1)[1][:13]                       # YYYYDDDHHMMSS
    t = dt.datetime.strptime(s, "%Y%j%H%M%S").replace(tzinfo=dt.timezone.utc)
    return int(t.timestamp())


def list_hour(bucket: str, t: dt.datetime) -> list[tuple[str, int]]:
    prefix = f"{PRODUCT}/{t:%Y}/{t:%j}/{t:%H}/"
    root = ET.fromstring(get(f"{bucket}/?list-type=2&prefix={prefix}"))
    out = []
    for c in root.findall("s3:Contents", NS):
        key = c.find("s3:Key", NS).text
        mod = dt.datetime.strptime(c.find("s3:LastModified", NS).text[:19], "%Y-%m-%dT%H:%M:%S")
        out.append((key, int(mod.replace(tzinfo=dt.timezone.utc).timestamp())))
    return out


def main() -> int:
    from download_firms import camera_box   # same camera-centred search box as FIRMS

    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="append")
    args = ap.parse_args()
    for fire in args.fire or FIRES:
        obs = load_fire(fire)
        w, s, e, n = (float(v) for v in camera_box(fire).split(","))
        t0, t1 = obs.t_start - BASELINE_S, obs.t_end
        hours, h = [], dt.datetime.fromtimestamp(t0, dt.timezone.utc).replace(minute=0, second=0)
        while h.timestamp() <= t1:
            hours.append(h); h += dt.timedelta(hours=1)
        files, sat, bucket = [], None, None
        for sat, bucket in BUCKETS:
            files = [(k, m) for hr in hours for k, m in list_hour(bucket, hr) if t0 <= scan_start(k) <= t1]
            if files:
                break
        raw_dir = ROOT / "data" / "goes" / "raw" / fire
        raw_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for key, published in files:
            local = raw_dir / key.rsplit("/", 1)[1]
            if not local.exists():
                local.write_bytes(get(f"{bucket}/{key}"))
                time.sleep(0.3)
            for p in read_fires(local, (w, s, e, n)):
                rows.append({"scan_start": scan_start(key), "available": published, **p, "sat": sat,
                             "file": local.name})
        out = ROOT / "data" / "goes" / f"{fire}.csv"
        with open(out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["scan_start", "available", "lat", "lon", "mask", "frp_mw", "sat", "file"])
            wr.writeheader()
            wr.writerows(rows)
        lag = sorted(m - scan_start(k) for k, m in files)
        print(f"{fire}: {sat if files else ''} {len(files)} scans, {len(rows)} fire pixels near the cameras; "
              f"publication delay median {lag[len(lag) // 2] / 60:.1f} min" if lag else f"{fire}: no GOES files found")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())

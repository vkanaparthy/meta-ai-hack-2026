"""Download the Cal OES California Webcams camera layer (includes ALERTCalifornia fire cameras)
for the network what-if study. Public ArcGIS feature service; no key.

  python scripts/download_caloes_cameras.py

Writes data/cameras/caloes_webcams.geojson and prints the layers and fields it found.
Camera imagery terms: ALERTCalifornia requires the credit "ALERTCalifornia | UC San Diego".
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SVC = "https://services.arcgis.com/BLN4oKB0N1YSgvY8/arcgis/rest/services/CalOES_California_Webcams/FeatureServer"


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "firefinder-hackathon/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main() -> int:
    info = get(f"{SVC}?f=json")
    layers = info.get("layers", [])
    print("layers:", [(l["id"], l["name"]) for l in layers])
    point = next((l for l in layers if "loc" in l["name"].lower() or "camera" in l["name"].lower()), layers[0])
    feats, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson",
                                    "resultOffset": offset, "resultRecordCount": 1000})
        page = get(f"{SVC}/{point['id']}/query?{q}")
        got = page.get("features", [])
        feats += got
        if len(got) < 1000:
            break
        offset += 1000
        time.sleep(1)
    out = ROOT / "data" / "cameras" / "caloes_webcams.geojson"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    fields = sorted(feats[0]["properties"]) if feats else []
    print(f"layer {point['id']} ({point['name']}): {len(feats)} cameras")
    print("fields:", fields)
    if feats:
        print("example:", {k: feats[0]["properties"][k] for k in fields[:12]})
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Skyline fit residual by image region (vertical px), for checking lens/edge fit.

  python scripts/check_skyline_residuals.py 20241108_ResortFire_bm-n-mobo-c
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from PIL import Image
from dataclasses import replace
from smokespotter.camera import CameraModel
from smokespotter.frames import load_sequence
from smokespotter.skyline import SkylineSamples, edge_map, predicted_rows, observed_rows
for seq in sys.argv[1:]:
    cam = CameraModel.load(Path("outputs") / seq / "camera_model.json")
    st = seq.split("_")[-1].split("-")[0]
    d = np.load(f"outputs/skyline/{st}.npz"); sky = SkylineSamples(d["az"], d["el"], d["dist"])
    frames = load_sequence(Path("data/raw") / seq)
    emap, s = edge_map([Image.open(f.path) for f in frames[:5]])
    wc = replace(cam, f_px=cam.f_px / s, width=emap.shape[1], height=emap.shape[0])
    xs = np.arange(0, wc.width, 2.0)
    yp = predicted_rows(wc, sky, xs); obs = observed_rows(emap, xs, yp, 15)
    r = yp - obs
    print(seq.split("_")[-1])
    for a, b in [(0, .15), (.15, .35), (.35, .65), (.65, .85), (.85, 1)]:
        m = (xs >= a * wc.width) & (xs < b * wc.width) & np.isfinite(r)
        print(f"   x {a:.0%}-{b:.0%}: median resid {np.median(r[m]) if m.any() else float('nan'):+5.2f} px, |r| {np.median(np.abs(r[m])) if m.any() else float('nan'):4.2f}, n {m.sum()}")

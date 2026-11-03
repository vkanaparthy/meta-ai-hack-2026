"""Render the 'skyline snap' clip for the demo video: a camera frame, our first guess of
where the camera points (its skyline drawn from the terrain model), the fit snapping that
line onto the real ridge, then the plume's compass bearing.

  python scripts/make_skyline_clip.py            # Kitchen Fire, camera pi-e
Writes outputs/media/skyline_snap.mp4 (1920x1080, 24 fps, about 12 s). Needs only cached files.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smokespotter.camera import CameraModel  # noqa: E402
from smokespotter.frames import load_sequence  # noqa: E402
from smokespotter.observations import load_fire  # noqa: E402
from smokespotter.skyline import SkylineSamples, predicted_rows  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEQ, STATION, TRACK, FRAME = "20240701_Kitchenfire_pi-e-mobo-c", "pi", "pi-e/1:0", 52
OUT_W, OUT_H, FPS = 1920, 1080, 24
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"
AMBER, PINK, ORANGE, WHITE = (255, 196, 40), (255, 60, 200), (255, 120, 20), (255, 255, 255)


def ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-captions", action="store_true", help="for use under a voiceover")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--out", default="outputs/media/skyline_snap.mp4")
    args = ap.parse_args()
    fps = args.fps
    fitted = CameraModel.load(ROOT / "outputs" / SEQ / "camera_model.json")
    guess = replace(fitted, heading_deg=90.0, pitch_deg=0.0, roll_deg=0.0)    # "camera faces east, level"
    d = np.load(ROOT / "outputs" / "skyline" / f"{STATION}.npz")
    sky = SkylineSamples(d["az"], d["el"], d["dist"])
    frames = load_sequence(ROOT / "data" / "raw" / SEQ)
    img = Image.open(frames[FRAME].path).convert("RGB")
    W, H = img.size
    track = next(t for t in load_fire("Kitchenfire").tracks if t.track_id == TRACK)
    first = track.dets[:3]              # the bearing is measured at the plume's base in its first frames
    box = (min(q.x0 for q in first), min(q.y0 for q in first), max(q.x1 for q in first), max(q.y1 for q in first))

    xs = np.arange(0, W, 4.0)
    rows_fit = predicted_rows(fitted, sky, xs)
    mid = float(np.nanmedian(rows_fit))
    full = (0.0, min(max(mid - W * 9 / 32, 0), H - W * 9 / 16), float(W))       # x, y, width (16:9)
    zw = W / 2.4
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    zoom = (min(max(cx - zw / 2, 0), W - zw), min(max(cy - zw * 9 / 32, 0), H - zw * 9 / 16), zw)
    bx, _ = fitted.project(np.array([track.bearing_deg]), np.array([0.0]))
    bx = float(bx[0])

    font = ImageFont.truetype(FONT, 46)
    small = ImageFont.truetype(FONT, 38)

    def cam_at(p: float) -> CameraModel:
        return replace(fitted, heading_deg=guess.heading_deg + (fitted.heading_deg - guess.heading_deg) * p,
                       pitch_deg=guess.pitch_deg + (fitted.pitch_deg - guess.pitch_deg) * p,
                       roll_deg=guess.roll_deg + (fitted.roll_deg - guess.roll_deg) * p)

    beats = [  # (start s, end s, caption)
        (0.0, 1.8, "A fire camera sees smoke. It can't tell you where."),
        (1.8, 3.8, "Our first guess at where it points: its skyline, drawn from a terrain model"),
        (3.8, 6.4, "FireFinder fits heading, tilt and lens until the line sits on the real ridge"),
        (6.4, 7.8, "Calibrated from the mountains. No survey, no training data."),
        (7.8, 12.0, f"Now every pixel is a compass bearing: smoke at {track.bearing_deg:.1f}°"),
    ]
    total = beats[-1][1]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{OUT_W}x{OUT_H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                           "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                           str(out)], stdin=subprocess.PIPE)
    cache: dict[float, np.ndarray] = {}
    for i in range(int(total * fps)):
        s = i / fps
        p = ease((s - 3.8) / 2.2)
        z = ease((s - 7.8) / 1.6)
        win = tuple(a + (b - a) * z for a, b in zip(full, zoom))
        wx, wy, ww = win
        wh = ww * 9 / 16
        frame = img.crop((int(wx), int(wy), int(wx + ww), int(wy + wh))).resize((OUT_W, OUT_H), Image.BILINEAR)
        dr = ImageDraw.Draw(frame, "RGBA")
        k = OUT_W / ww

        def T(x, y):
            return ((x - wx) * k, (y - wy) * k)

        if s >= 1.8:
            key = round(p, 3)
            if key not in cache:
                cache[key] = predicted_rows(cam_at(p), sky, xs)
            rows = cache[key]
            col = tuple(int(a + (b - a) * p) for a, b in zip(AMBER, PINK))
            alpha = int(255 * min(1.0, (s - 1.8) / 0.5))
            seg = []
            for x, y in zip(xs, rows):
                if np.isfinite(y):
                    seg.append(T(x, y))
                elif len(seg) > 1:
                    dr.line(seg, fill=col + (alpha,), width=5); seg = []
                else:
                    seg = []
            if len(seg) > 1:
                dr.line(seg, fill=col + (alpha,), width=5)
        if s >= 7.8:
            a = int(255 * min(1.0, (s - 7.8) / 0.6))
            x0, y0 = T(box[0], box[1]); x1, y1 = T(box[2], box[3])
            pad = 10
            dr.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), outline=WHITE + (a,), width=7)
            dr.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), outline=ORANGE + (a,), width=4)
            if s >= 8.6:
                a2 = int(255 * min(1.0, (s - 8.6) / 0.5))
                X = T(bx, 0)[0]
                dr.line([(X, 0), (X, OUT_H)], fill=WHITE + (a2,), width=3)
                label = f"{track.bearing_deg:.1f}°"
                tw = dr.textlength(label, font=font)
                dr.rounded_rectangle((X + 16, 40, X + 40 + tw, 104), 8, fill=(20, 26, 22, int(0.8 * a2)))
                dr.text((X + 28, 46), label, font=font, fill=WHITE + (a2,))
        if not args.no_captions:
            cap = next(c for b0, b1, c in beats if b0 <= s < b1)
            tw = dr.textlength(cap, font=small)
            dr.rounded_rectangle(((OUT_W - tw) / 2 - 28, OUT_H - 128, (OUT_W + tw) / 2 + 28, OUT_H - 56), 10,
                                 fill=(20, 26, 22, 200))
            dr.text(((OUT_W - tw) / 2, OUT_H - 118), cap, font=small, fill=WHITE)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {out.relative_to(ROOT)} ({total:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Assemble the demo video on the voiceover's timing.

  python scripts/make_demo_video.py --scene 1      # ... --scene 9, one at a time
  python scripts/make_demo_video.py --final        # join scenes, add cleaned-up voiceover

Scene boundaries and the words key moments land on come from a forced alignment of
outputs/media/voiceover.m4a against the script (PocketSphinx): e.g. "ten thirty-seven" at 48.96 s,
"one hundred nineteen" at 65.84 s, "fix" at 78.83 s, "kilometers" at 81.01 s.
Inputs in outputs/media/: kitchen/junction/grove.mov (screen recordings of ?demo=...),
video/skyline_raw.mp4 (make_skyline_clip.py --no-captions --fps 30), slide1/2.png.
Output: outputs/media/firefinder_demo.mp4 (1920x1080, 30 fps).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "outputs" / "media"
V = M / "video"
W, H, FPS = 1920, 1080, 30
PAPER = "0xEEF0E6"
ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an"]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"

# scene start/end on the voiceover clock (cuts sit in the pauses between paragraphs)
CUTS = [0.0, 6.9, 23.7, 37.5, 53.9, 67.4, 83.2, 94.9, 108.3, 118.21]
FIT = f"scale={W}:-2:flags=lanczos,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={PAPER}"


def dur(n: int) -> float:
    return round(CUTS[n] - CUTS[n - 1], 3)


def ff(args: list[str], out: Path) -> None:
    tmp = out.with_name(out.stem + ".part.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args, *ENC, str(tmp)], check=True)
    os.replace(tmp, out)


def pieces(src: Path, segs: list[tuple[float, float, float]], total: float, out: Path) -> None:
    """Play [a, b] of src so it lasts d seconds, for each (a, b, d); join; fit to total."""
    parts, labels = [], []
    for i, (a, b, d) in enumerate(segs):
        k = d / (b - a)
        parts.append(f"[0:v]trim={a}:{b},setpts=(PTS-STARTPTS)*{k:.5f},fps={FPS},{FIT},setsar=1[p{i}]")
        labels.append(f"[p{i}]")
    graph = ";".join(parts) + f";{''.join(labels)}concat=n={len(segs)}:v=1:a=0," \
            f"tpad=stop_mode=clone:stop_duration=2,trim=duration={total}[v]"
    ff(["-i", str(src), "-filter_complex", graph, "-map", "[v]"], out)


def frames_to(out: Path, frames) -> None:
    tmp = out.with_name(out.stem + ".part.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", *ENC, str(tmp)], stdin=subprocess.PIPE)
    for im in frames:
        p.stdin.write(im.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit("ffmpeg failed")
    os.replace(tmp, out)


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def scene1(out: Path) -> None:
    """A real camera frame with the first smoke, slowly pushing in; brand and credit."""
    from smokespotter.desk import hhmm
    from smokespotter.frames import load_sequence
    from smokespotter.observations import load_fire

    seq = "20240701_Kitchenfire_pi-e-mobo-c"
    frames = load_sequence(ROOT / "data" / "raw" / seq)
    tr = next(t for t in load_fire("Kitchenfire").tracks if t.track_id == "pi-e/1:0")
    fr = frames[tr.dets[4].frame]
    img = Image.open(fr.path).convert("RGB")
    iw, ih = img.size
    d = tr.dets[4]
    cx, cy = (d.x0 + d.x1) / 2, (d.y0 + d.y1) / 2
    big = ImageFont.truetype(FONT, 64)
    small = ImageFont.truetype(FONT, 30)
    n = int(dur(1) * FPS)

    def gen():
        for i in range(n):
            u = ease(i / (n - 1))
            ww = iw * (1 - 0.55 * u)                     # push in towards the plume
            x0 = (iw - iw) / 2 * (1 - u) + (cx - ww / 2) * u
            y0 = ((ih - iw * 9 / 16) / 2) * (1 - u) + (cy - ww * 9 / 32) * u
            x0 = min(max(x0, 0), iw - ww); y0 = min(max(y0, 0), ih - ww * 9 / 16)
            f = img.crop((int(x0), int(y0), int(x0 + ww), int(y0 + ww * 9 / 16))).resize((W, H), Image.BILINEAR)
            dr = ImageDraw.Draw(f, "RGBA")
            a = int(255 * min(1.0, i / FPS / 0.8))
            dr.text((64, 52), "FireFinder", font=big, fill=(255, 255, 255, a))
            cap = f"HPWREN camera, Los Pinos (east view)  ·  Kitchen Fire  ·  July 1, 2024, {hhmm(fr.epoch)}"
            tw = dr.textlength(cap, font=small)
            dr.rounded_rectangle((52, H - 92, 52 + tw + 28, H - 42), 8, fill=(20, 26, 22, 170))
            dr.text((66, H - 86), cap, font=small, fill=(255, 255, 255, 235))
            if i < 15:                                   # fade in from black
                f = Image.blend(Image.new("RGB", (W, H)), f, i / 15)
            yield f
    frames_to(out, gen())


def scene2(out: Path) -> None:
    """The Kitchen replay before any smoke: cameras watching. Slow push in."""
    still = V / "kitchen_start.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "0.5", "-i", str(M / "kitchen.mov"),
                    "-frames:v", "1", "-vf", FIT, str(still)], check=True)
    img = Image.open(still).convert("RGB")
    n = int(dur(2) * FPS)

    def gen():
        for i in range(n):
            z = 1 + 0.10 * ease(i / (n - 1))
            ww, hh = W / z, H / z
            x0, y0 = (W - ww) * 0.38, (H - hh) * 0.45
            yield img.crop((int(x0), int(y0), int(x0 + ww), int(y0 + hh))).resize((W, H), Image.BILINEAR)
    frames_to(out, gen())


def scene_still(src: Path, out: Path, total: float, fade_out: bool = False) -> None:
    vf = f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={PAPER}," \
         f"format=yuv420p,loop=-1:1,trim=duration={total}"
    if fade_out:
        vf += f",fade=t=out:st={total - 1.2}:d=1.2"
    ff(["-framerate", str(FPS), "-i", str(src), "-vf", vf], out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, action="append")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    V.mkdir(parents=True, exist_ok=True)
    for n in args.scene or []:
        out = V / f"scene{n}.mp4"
        if n == 1:
            scene1(out)
        elif n == 2:
            scene2(out)
        elif n == 3:      # skyline snap, 12 s stretched to the paragraph
            pieces(V / "skyline_raw.mp4", [(0, 12.0, dur(3))], dur(3), out)
        elif n == 4:      # Kitchen: SPOTTED (recording 4.1 s) lands on "ten thirty-seven" (48.96)
            pieces(M / "kitchen.mov", [(1.8, 4.1, 48.96 - CUTS[3]), (4.1, 9.1, dur(4) - (48.96 - CUTS[3]))],
                   dur(4), out)
        elif n == 5:      # provisional 13.4 -> 60.3, confirmed 29.3 -> ~62.5, origin 33.1 -> 65.8
            pieces(M / "kitchen.mov", [(9.1, 13.4, 60.29 - CUTS[4]), (13.4, 29.3, 2.2),
                                       (29.3, 33.1, 65.84 - 60.29 - 2.2), (33.1, 36.0, CUTS[5] - 65.84)],
                   dur(5), out)
        elif n == 6:      # Junction: pin drop 19.7 -> "fix" 78.83, origin 24.7 -> 81.2
            pieces(M / "junction.mov", [(3.0, 19.7, 78.83 - CUTS[5]), (19.7, 24.7, 81.23 - 78.83),
                                        (24.7, 28.0, CUTS[6] - 81.23)], dur(6), out)
        elif n == 7:      # Grove: no pin
            pieces(M / "grove.mov", [(3.0, 3.0 + dur(7), dur(7))], dur(7), out)
        elif n == 8:
            scene_still(M / "slide1.png", out, dur(8))
        elif n == 9:
            scene_still(M / "slide2.png", out, dur(9), fade_out=True)
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True)
        print(f"scene {n}: {float(r.stdout):.2f} s (planned {dur(n):.2f})")
    if args.final:
        lst = V / "scenes.txt"
        lst.write_text("".join(f"file 'scene{n}.mp4'\n" for n in range(1, 10)))
        silent = V / "joined.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", str(silent)], check=True)
        out = M / "firefinder_demo.mp4"
        tmp = out.with_name("firefinder_demo.part.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(M / "voiceover.m4a"),
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                        "-af", "highpass=f=80,afftdn=nf=-30,loudnorm=I=-16:TP=-1.5:LRA=11",
                        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-shortest", "-movflags", "+faststart",
                        str(tmp)], check=True)
        os.replace(tmp, out)
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

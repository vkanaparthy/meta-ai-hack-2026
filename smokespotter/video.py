"""Turn a camera's still-image sequence into an MP4 that SAM 3.1 accepts
(H.264, one video frame per still, in capture order)."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .frames import Frame


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("ffmpeg not found; pip install imageio-ffmpeg")
        return exe


def frames_to_mp4(frames: list[Frame], out: Path, max_width: int = 1536, fps: int = 4,
                  crop: tuple[int, int, int, int] | None = None) -> Path:
    """Encode frames to ``out``. Width is capped to keep uploads small;
    SAM boxes come back in this video's pixels, so callers rescale by
    original_width / video_width. ``crop`` = (x, y, w, h) in original pixels."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{f.path.resolve()}'\nduration {1 / fps}\n" for f in frames)
                       + f"file '{frames[-1].path.resolve()}'\n")
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
               "-i", str(lst), "-vf", (f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}," if crop else "")
               + f"scale='min({max_width},iw)':-2,fps={fps}",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)]
        subprocess.run(cmd, check=True)
    return out

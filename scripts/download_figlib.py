"""Download FIgLib sequences listed in data/fires.csv, politely.

Strategy (fewest requests first):
  1. One .tgz archive per sequence from the Tar/ directory, if it exists.
  2. Otherwise, the individual JPGs listed on the sequence page.

Politeness:
  * one connection at a time, --delay seconds (+ jitter) between requests
  * honours Retry-After on 429/503, otherwise exponential backoff
  * stops a sequence on 403/404, aborts the run after repeated failures
  * --max-requests caps the total requests per run
  * resumable: files already on disk are never re-downloaded

Usage:
  python scripts/download_figlib.py --fire JunctionFire
  python scripts/download_figlib.py --role dev --every 2
  python scripts/download_figlib.py --role test --delay 3

Data credit: HPWREN, http://hpwren.ucsd.edu/
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data"
ROOT = Path(__file__).resolve().parent.parent
USER_AGENT = ("smokespotter-hackathon/0.1 (research use, credits HPWREN; "
              "+https://github.com/vkanaparthy/meta-ai-hack-2026)")
HREF = re.compile(r"""href\s*=\s*["']?([^"'\s>?#]+\.(?:jpe?g|mp4))""", re.IGNORECASE)
FRAME = re.compile(r"^\d{9,11}_[+-]?\d+\.jpe?g$", re.IGNORECASE)
MP4 = re.compile(r"\.mp4$", re.IGNORECASE)


class Abort(Exception):
    """Stop the whole run (server unhappy or request budget used up)."""


class Polite:
    def __init__(self, delay: float, max_requests: int, retries: int = 4):
        self.delay, self.max_requests, self.retries = delay, max_requests, retries
        self.requests = 0
        self.consecutive_failures = 0
        self._last = 0.0

    def _wait(self) -> None:
        gap = self.delay * random.uniform(0.8, 1.4) - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)

    def get(self, url: str) -> bytes | None:
        """Return the body, None for 404, or raise Abort."""
        for attempt in range(self.retries + 1):
            if self.requests >= self.max_requests:
                raise Abort(f"request budget of {self.max_requests} used; rerun later to resume")
            self._wait()
            self.requests += 1
            self._last = time.monotonic()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = r.read()
                self.consecutive_failures = 0
                return body
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code == 403:
                    raise Abort("server returned 403 Forbidden; stopping. Wait a while "
                                "or contact HPWREN before retrying.")
                if e.code in (429, 500, 502, 503, 504):
                    wait = _retry_after(e) or min(300, 10 * 2 ** attempt)
                    print(f"  server busy ({e.code}); waiting {wait:.0f}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                wait = min(300, 10 * 2 ** attempt)
                print(f"  network error ({e}); waiting {wait:.0f}s", file=sys.stderr)
                time.sleep(wait)
        self.consecutive_failures += 1
        if self.consecutive_failures >= 3:
            raise Abort("three requests failed in a row; stopping to be safe")
        return None


def _retry_after(e: urllib.error.HTTPError) -> float | None:
    try:
        return float(e.headers.get("Retry-After", ""))
    except ValueError:
        return None


def _write(dest: Path, data: bytes) -> None:
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)


def _have_frames(out: Path) -> int:
    return sum(1 for p in out.glob("*.jp*g") if FRAME.match(p.name)) if out.exists() else 0


def from_tar(http: Polite, seq: str, out: Path, every: int, with_mp4: bool) -> int | None:
    data = http.get(f"{BASE}/Tar/{seq}.tgz")
    if data is None:
        return None
    out.mkdir(parents=True, exist_ok=True)
    n, frames = 0, []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            name = Path(m.name).name           # flatten; never trust archive paths
            if not m.isfile():
                continue
            if FRAME.match(name):
                frames.append((name, m))
            elif with_mp4 and MP4.search(name):
                _write(out / name, tf.extractfile(m).read())
                n += 1
        for name, m in sorted(frames)[::every]:
            if not (out / name).exists():
                _write(out / name, tf.extractfile(m).read())
                n += 1
    return n


def from_page(http: Polite, seq: str, out: Path, every: int, with_mp4: bool) -> int:
    page = http.get(f"{BASE}/{seq}/index.html")
    if page is None:
        raise RuntimeError("sequence page not found")
    hrefs = sorted(set(HREF.findall(page.decode("utf-8", "replace"))))
    names = [urllib.parse.unquote(h.split("/")[-1]) for h in hrefs]
    if not names:
        raise RuntimeError("no image links found on the sequence page")
    wanted = sorted(n for n in names if FRAME.match(n))[::every]
    if with_mp4:
        wanted += [n for n in names if MP4.search(n)]
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in wanted:
        dest = out / name
        if dest.exists():
            continue
        data = http.get(f"{BASE}/{seq}/{urllib.parse.quote(name)}")
        if data is not None:
            _write(dest, data)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", action="append", help="showcase/dev/test (repeatable)")
    ap.add_argument("--fire", action="append", help="fire name from fires.csv (repeatable)")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--mp4", action="store_true", help="also keep the time-lapse MP4")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between requests (min 1)")
    ap.add_argument("--max-requests", type=int, default=400, help="request budget per run")
    ap.add_argument("--no-tar", action="store_true", help="skip the .tgz archives")
    ap.add_argument("--redo", action="store_true",
                    help="re-check sequences already on disk (e.g. to add frames skipped by --every)")
    args = ap.parse_args()
    http = Polite(max(1.0, args.delay), args.max_requests)
    roles = set(args.role or ["showcase", "dev"])

    rows = list(csv.DictReader(open(ROOT / "data" / "fires.csv")))
    rows = [r for r in rows if (args.fire and r["fire"] in args.fire) or
            (not args.fire and r["role"] in roles)]
    try:
        for r in rows:
            seq = r["sequence"]
            out = ROOT / "data" / "raw" / seq
            if _have_frames(out) >= 20 and not args.redo:
                print(f"{seq}: already downloaded, skipping")
                continue
            try:
                n = None if args.no_tar else from_tar(http, seq, out, args.every, args.mp4)
                how = "archive"
                if n is None:
                    n, how = from_page(http, seq, out, args.every, args.mp4), "images"
                print(f"{seq}: {n} new files ({how})")
            except Abort:
                raise
            except Exception as e:
                print(f"{seq}: FAILED ({e})", file=sys.stderr)
    except Abort as e:
        print(f"Stopped: {e}", file=sys.stderr)
        return 1
    finally:
        print(f"{http.requests} requests made", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

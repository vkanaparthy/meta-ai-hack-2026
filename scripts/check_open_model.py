"""Check the open-weights endpoint before spending real budget (costs well under a cent).

  python scripts/check_open_model.py                     # Muse Spark Open on Baseten
  python scripts/check_open_model.py --provider glimmer  # Muse Glimmer on a local Ollama server

Tests, in order: the key works, the model answers text, it reads an image, it calls a tool.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402

from smokespotter import models  # noqa: E402
from smokespotter.models import spent_usd  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["open", "glimmer"], default="open")
    args = ap.parse_args()
    if args.provider == "glimmer":
        open_client, open_model, open_chat = models.glimmer_client, models.glimmer_model, models.glimmer_chat
    else:
        open_client, open_model, open_chat = models.open_client, models.open_model, models.open_chat
    print(f"model: {open_model()}")
    try:
        ids = [m.id for m in open_client().models.list()]
        hits = [i for i in ids if "muse" in i.lower() or "glimmer" in i.lower()]
        print(f"1. key ok; Muse models listed: {hits or 'none (listing may be partial)'}")
    except Exception as e:      # some endpoints do not list models; carry on
        print(f"1. could not list models ({e}); trying a call anyway")

    r = open_chat([{"role": "user", "content": "Reply with the single word: ready"}], "check", max_tokens=20000)
    print(f"2. text: {r.choices[0].message.content!r}")

    from PIL import Image
    import base64, io
    img = Image.new("RGB", (64, 64), (200, 40, 40))
    buf = io.BytesIO(); img.save(buf, format="JPEG")
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    r = open_chat([{"role": "user", "content": [
        {"type": "text", "text": "What colour is this square? One word."},
        {"type": "image_url", "image_url": {"url": url}}]}], "check", max_tokens=20000)
    print(f"3. image: {r.choices[0].message.content!r}")

    tool = {"type": "function", "function": {
        "name": "locate", "description": "Locate the fire.",
        "parameters": {"type": "object", "properties": {}}}}
    r = open_chat([{"role": "user", "content": "Call the locate tool."}], "check", tools=[tool], max_tokens=2000)
    tc = r.choices[0].message.tool_calls
    print(f"4. tool call: {[(c.function.name, c.function.arguments) for c in tc] if tc else 'NONE - ' + repr(r.choices[0].message.content)}")
    print(f"spend so far: ${spent_usd():.4f}")
    return 0 if tc else 1


if __name__ == "__main__":
    raise SystemExit(main())

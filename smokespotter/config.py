"""Settings from the environment, with a tiny .env loader (no extra dependency)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path = ROOT / ".env") -> None:
    """Set variables from .env without overriding ones already in the environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip().strip('"').strip("'")
        if key.strip() and val:
            os.environ.setdefault(key.strip(), val)


def setting(name: str, default: str | None = None) -> str:
    load_env()
    val = os.getenv(name, default)
    if not val:
        raise RuntimeError(f"{name} is not set; add it to .env (see .env.example)")
    return val

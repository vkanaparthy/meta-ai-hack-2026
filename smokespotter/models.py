"""Model clients.

Two providers serve the Muse reasoning/vision model, picked by MUSE_PROVIDER:
  * meta (default) - Meta Model API, Muse Spark 1.3, Responses API
  * open           - Muse Spark Open (open weights) on a partner endpoint such as
                     Baseten, OpenAI-compatible Chat Completions
SAM 3.1 always comes from the Meta Model API.

Every open-weights call is metered (outputs/usage_open.jsonl) and refused once the
running cost passes OPEN_MODEL_BUDGET_USD.
"""
from __future__ import annotations

import json
import os
import time
from functools import lru_cache

from .config import ROOT, load_env, setting


@lru_cache(maxsize=1)
def client():
    """Meta Model API client (SAM and Muse Spark 1.3)."""
    from openai import OpenAI

    return OpenAI(
        base_url=setting("META_API_BASE_URL", "https://api.meta.ai/v1"),
        api_key=setting("MODEL_API_KEY"),
        max_retries=4,          # SDK backs off on 429/5xx
        timeout=600,
    )


@lru_cache(maxsize=1)
def open_client():
    """Open-weights endpoint (Baseten by default)."""
    from openai import OpenAI

    return OpenAI(
        base_url=setting("OPEN_MODEL_BASE_URL", "https://inference.baseten.co/v1"),
        api_key=setting("OPEN_MODEL_API_KEY"),
        max_retries=4,
        timeout=600,
    )


def provider() -> str:
    load_env()
    p = os.getenv("MUSE_PROVIDER", "meta").strip().lower() or "meta"
    if p not in ("meta", "open"):
        raise RuntimeError(f"MUSE_PROVIDER must be 'meta' or 'open', got {p!r}")
    return p


def set_provider(p: str) -> None:
    os.environ["MUSE_PROVIDER"] = p


def sam_model() -> str:
    return setting("SAM_MODEL", "sam-3.1")


def reasoning_model() -> str:
    if provider() == "open":
        return open_model()
    return setting("MUSE_REASONING_MODEL", "muse-spark-1.3")


def open_model() -> str:
    return setting("OPEN_MODEL_NAME", "meta-models/muse-spark-open-1026")


def open_max_tokens() -> int:
    """Muse Spark Open reasons before it answers; too small a cap returns an empty reply."""
    load_env()
    return int(os.getenv("OPEN_MODEL_MAX_TOKENS", "8192"))


def verdicts_name() -> str:
    """Cached smoke verdicts are kept per provider so the two never mix."""
    return "verdicts.json" if provider() == "meta" else f"verdicts_{provider()}.json"


# ---- cost meter for the open-weights endpoint ---------------------------------

USAGE_LOG = ROOT / "outputs" / "usage_open.jsonl"


def _price(name: str, default: float) -> float:
    load_env()
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def spent_usd() -> float:
    if not USAGE_LOG.exists():
        return 0.0
    return sum(json.loads(l).get("usd", 0.0) for l in USAGE_LOG.read_text().splitlines() if l.strip())


class BudgetExceeded(RuntimeError):
    pass


def check_budget() -> None:
    budget = _price("OPEN_MODEL_BUDGET_USD", 90.0)
    spent = spent_usd()
    if spent >= budget:
        raise BudgetExceeded(f"open-weights spend ${spent:.2f} reached the ${budget:.2f} cap "
                             "(OPEN_MODEL_BUDGET_USD)")


def record_usage(usage, purpose: str) -> float:
    """Log one call's tokens and cost; returns the cost in USD."""
    if usage is None:
        return 0.0
    tin = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
    tout = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0
    usd = (tin * _price("OPEN_MODEL_USD_PER_M_IN", 1.25) + tout * _price("OPEN_MODEL_USD_PER_M_OUT", 4.25)) / 1e6
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(USAGE_LOG, "a") as fh:
        fh.write(json.dumps({"t": round(time.time()), "purpose": purpose, "model": open_model(),
                             "in": tin, "out": tout, "usd": round(usd, 6)}) + "\n")
    return usd


def open_chat(messages: list[dict], purpose: str, **kw):
    """One metered Chat Completions call to the open-weights model."""
    check_budget()
    resp = open_client().chat.completions.create(model=open_model(), messages=messages, **kw)
    record_usage(getattr(resp, "usage", None), purpose)
    return resp


def ask_vision(prompt: str, image_urls: list[str], purpose: str = "verify") -> str:
    """Text + images in, text out, on whichever provider is selected."""
    if provider() == "open":
        content = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
        resp = open_chat([{"role": "user", "content": content}], purpose, max_tokens=open_max_tokens())
        return resp.choices[0].message.content or ""
    content = [{"type": "input_text", "text": prompt}]
    content += [{"type": "input_image", "image_url": u} for u in image_urls]
    resp = client().responses.create(model=reasoning_model(),
                                     input=[{"type": "message", "role": "user", "content": content}])
    return resp.output_text

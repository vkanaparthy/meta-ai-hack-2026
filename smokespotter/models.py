"""Model clients.

Three providers serve the Muse reasoning/vision model, picked by MUSE_PROVIDER:
  * meta (default) - Meta Model API, Muse Spark 1.3, Responses API
  * open           - Muse Spark Open (open weights) on a partner endpoint such as
                     Baseten, OpenAI-compatible Chat Completions
  * glimmer        - Muse Glimmer 30B (Apache 2.0) on any OpenAI-compatible endpoint:
                     a local Ollama server or a dedicated Baseten deployment (billed
                     by GPU time, so not metered per call)
SAM 3.1 always comes from the Meta Model API.

Every model call is logged with its latency, tokens and cost in outputs/usage_<provider>.jsonl
(scripts/usage_report.py summarizes them). Open-weights calls are refused once their running
cost passes OPEN_MODEL_BUDGET_USD.
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


PROVIDERS = ("meta", "open", "glimmer")


@lru_cache(maxsize=1)
def glimmer_client():
    """Muse Glimmer on a local OpenAI-compatible server (Ollama by default)."""
    from openai import OpenAI

    load_env()
    return OpenAI(base_url=os.getenv("GLIMMER_BASE_URL", "http://localhost:11434/v1"),
                  api_key=os.getenv("GLIMMER_API_KEY", "ollama"),   # Ollama ignores it
                  max_retries=2, timeout=900)


def glimmer_model() -> str:
    load_env()
    return os.getenv("GLIMMER_MODEL", "muse-glimmer:30b-mlx")


def provider() -> str:
    load_env()
    p = os.getenv("MUSE_PROVIDER", "meta").strip().lower() or "meta"
    if p not in PROVIDERS:
        raise RuntimeError(f"MUSE_PROVIDER must be one of {PROVIDERS}, got {p!r}")
    return p


def set_provider(p: str) -> None:
    os.environ["MUSE_PROVIDER"] = p


def sam_model() -> str:
    return setting("SAM_MODEL", "sam-3.1")


def reasoning_model() -> str:
    if provider() == "open":
        return open_model()
    if provider() == "glimmer":
        return glimmer_model()
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


# ---- latency and cost meter ---------------------------------------------------

USAGE_LOG = ROOT / "outputs" / "usage_open.jsonl"       # the open-weights budget is summed from this


def usage_log(p: str) -> "Path":
    return ROOT / "outputs" / f"usage_{p}.jsonl"


def _prices(p: str, purpose: str) -> tuple[float, float] | None:
    """USD per million input/output tokens, or None if unpriced."""
    if p == "open":
        return _price("OPEN_MODEL_USD_PER_M_IN", 1.25), _price("OPEN_MODEL_USD_PER_M_OUT", 4.25)
    if p == "meta":
        if purpose.startswith("sam"):
            return None                          # SAM 3.1 has no published per-token price
        return _price("META_USD_PER_M_IN", 1.25), _price("META_USD_PER_M_OUT", 4.25)
    return 0.0, 0.0                              # glimmer: runs on your own hardware


def record_call(p: str, purpose: str, model: str, seconds: float, usage=None) -> float:
    """Log one model call: provider, purpose, latency, tokens, cost. Returns the cost in USD."""
    tin = (getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0) if usage else 0
    tout = (getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0) if usage else 0
    pr = _prices(p, purpose)
    usd = None if pr is None else round((tin * pr[0] + tout * pr[1]) / 1e6, 6)
    path = usage_log(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps({"t": round(time.time()), "purpose": purpose, "model": model, "seconds": round(seconds, 2),
                             "in": tin, "out": tout, "usd": usd}) + "\n")
    return usd or 0.0


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


def open_chat(messages: list[dict], purpose: str, **kw):
    """One metered Chat Completions call to the open-weights model."""
    check_budget()
    t0 = time.time()
    resp = open_client().chat.completions.create(model=open_model(), messages=messages, **kw)
    record_call("open", purpose, open_model(), time.time() - t0, getattr(resp, "usage", None))
    return resp


def glimmer_is_local() -> bool:
    load_env()
    url = os.getenv("GLIMMER_BASE_URL", "http://localhost:11434/v1")
    return "localhost" in url or "127.0.0.1" in url


def glimmer_chat(messages: list[dict], purpose: str, **kw):
    """One call to Muse Glimmer: a local server (Ollama) or a dedicated deployment such as
    Baseten's (GLIMMER_BASE_URL = its .../environments/production/sync/v1 URL). A dedicated
    deployment is billed by GPU time, not tokens, so these calls are not metered here."""
    load_env()
    effort = os.getenv("GLIMMER_REASONING", "medium")      # low | medium | high | xhigh
    if effort and not (messages and messages[0].get("role") == "system"):
        messages = [{"role": "system", "content": f"Reasoning strength: {effort}"}] + messages
    t0 = time.time()
    resp = glimmer_client().chat.completions.create(model=glimmer_model(), messages=messages, **kw)
    record_call("glimmer", purpose, glimmer_model(), time.time() - t0, getattr(resp, "usage", None))
    return resp


def ask_vision(prompt: str, image_urls: list[str], purpose: str = "verify") -> str:
    """Text + images in, text out, on whichever provider is selected."""
    if provider() in ("open", "glimmer"):
        content = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
        chat = open_chat if provider() == "open" else glimmer_chat
        resp = chat([{"role": "user", "content": content}], purpose, max_tokens=open_max_tokens())
        return resp.choices[0].message.content or ""
    content = [{"type": "input_text", "text": prompt}]
    content += [{"type": "input_image", "image_url": u} for u in image_urls]
    t0 = time.time()
    resp = client().responses.create(model=reasoning_model(),
                                     input=[{"type": "message", "role": "user", "content": content}])
    record_call("meta", purpose, reasoning_model(), time.time() - t0, getattr(resp, "usage", None))
    return resp.output_text

"""Meta Model API client (OpenAI-compatible, https://dev.meta.ai/docs)."""
from __future__ import annotations

from functools import lru_cache

from .config import setting


@lru_cache(maxsize=1)
def client():
    from openai import OpenAI

    return OpenAI(
        base_url=setting("META_API_BASE_URL", "https://api.meta.ai/v1"),
        api_key=setting("MODEL_API_KEY"),
        max_retries=4,          # SDK backs off on 429/5xx
        timeout=600,
    )


def sam_model() -> str:
    return setting("SAM_MODEL", "sam-3.1")


def reasoning_model() -> str:
    return setting("MUSE_REASONING_MODEL", "muse-spark-1.3")

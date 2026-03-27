"""LLM model loading utilities for AI-TV-Studio.

Provides factory functions for constructing LangChain-compatible LLM clients
used throughout the pipeline.  The API key is read from the ``DEEPSEEK_API_KEY``
environment variable so that credentials are never hard-coded in source.
"""

import os
from pathlib import Path

from langchain_deepseek import ChatDeepSeek


def _get_api_key() -> str:
    """Resolve the DeepSeek API key.

    Checks in order:
    1. ``DEEPSEEK_API_KEY`` environment variable.
    2. ``deepseek.api_key`` in ``config/video_api_keys.yaml`` (workspace root).

    Returns:
        The API key string, or an empty string if not found.
    """
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key

    config_path = Path(__file__).parent.parent / "config" / "video_api_keys.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            key = (data or {}).get("deepseek", {}).get("api_key", "")
        except Exception:
            pass

    return key or ""


def load() -> ChatDeepSeek:
    """Load the default DeepSeek chat model.

    Returns:
        A ``ChatDeepSeek`` instance configured for the ``deepseek-chat`` model.
    """
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=_get_api_key(),
    )
    return llm


def load_reasoning() -> ChatDeepSeek:
    """Load the DeepSeek reasoning model (deepseek-r1).

    Returns:
        A ``ChatDeepSeek`` instance configured for the ``deepseek-r1`` model.
    """
    llm = ChatDeepSeek(
        model="deepseek-r1",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=_get_api_key(),
    )
    return llm


if __name__ == "__main__":
    llm = load()
    out = llm.invoke("hello")
    print(out)
"""LLM model loading utilities for AI-TV-Studio.

Provides factory functions for constructing LangChain-compatible LLM clients
used throughout the pipeline.  The API key is read from the ``DEEPSEEK_API_KEY``
environment variable so that credentials are never hard-coded in source.
"""

import os

from langchain_deepseek import ChatDeepSeek


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
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
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
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    )
    return llm


if __name__ == "__main__":
    llm = load()
    out = llm.invoke("hello")
    print(out)
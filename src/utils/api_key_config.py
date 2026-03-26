"""API key configuration helpers for external model providers."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_VIDEO_KEY_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "video_api_keys.yaml"
)


def load_video_api_keys(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load provider API keys from the dedicated video key config file.

    Args:
        config_path: Optional explicit config path.

    Returns:
        Parsed YAML dictionary. Empty dict if file does not exist.
    """
    path = Path(config_path) if config_path else DEFAULT_VIDEO_KEY_CONFIG
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if not isinstance(data, dict):
        return {}
    return data


def get_provider_api_key(
    provider: str,
    config_path: Optional[str] = None,
    env_var: Optional[str] = None,
) -> Optional[str]:
    """Resolve provider API key from config file, then environment fallback.

    Resolution order:
    1. ``config/video_api_keys.yaml`` value for the provider.
    2. ``env_var`` if provided.
    3. ``{PROVIDER}_API_KEY`` environment variable.

    Args:
        provider: Provider key in config, for example ``"ark"``.
        config_path: Optional explicit config path.
        env_var: Optional custom environment variable name.

    Returns:
        API key string or None if not found.
    """
    provider = provider.lower()
    keys = load_video_api_keys(config_path=config_path)

    provider_cfg = keys.get(provider, {})
    if isinstance(provider_cfg, dict):
        key = provider_cfg.get("api_key")
        if isinstance(key, str) and key.strip():
            return key.strip()

    if env_var:
        env_key = os.environ.get(env_var)
        if env_key:
            return env_key

    auto_env = f"{provider.upper()}_API_KEY"
    return os.environ.get(auto_env)

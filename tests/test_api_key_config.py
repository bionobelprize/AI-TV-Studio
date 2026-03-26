"""Tests for API key config helpers."""

from pathlib import Path

from src.utils.api_key_config import get_provider_api_key, load_video_api_keys


def test_load_video_api_keys_missing_file(tmp_path):
    missing = tmp_path / "missing.yaml"
    assert load_video_api_keys(str(missing)) == {}


def test_load_video_api_keys_and_get_provider_key(tmp_path):
    cfg = tmp_path / "video_api_keys.yaml"
    cfg.write_text(
        """
ark:
  api_key: test-key-001
""".strip(),
        encoding="utf-8",
    )

    loaded = load_video_api_keys(str(cfg))
    assert loaded["ark"]["api_key"] == "test-key-001"
    assert get_provider_api_key("ark", config_path=str(cfg)) == "test-key-001"


def test_get_provider_api_key_from_environment(tmp_path, monkeypatch):
    cfg = tmp_path / "video_api_keys.yaml"
    cfg.write_text("ark:\n  api_key: ''\n", encoding="utf-8")

    monkeypatch.setenv("ARK_API_KEY", "env-key-123")
    assert get_provider_api_key("ark", config_path=str(cfg)) == "env-key-123"

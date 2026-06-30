from __future__ import annotations

import pytest

from tldw_Server_API.app.core.TTS.tts_config import TTSConfigManager


@pytest.mark.unit
def test_fish_audio_api_key_env_overrides_provider_config(tmp_path, monkeypatch):
    yaml_path = tmp_path / "tts.yaml"
    yaml_path.write_text(
        """
providers:
  fish_s2:
    enabled: true
    backend: commercial_api
    api_key: null
""".strip(),
        encoding="utf-8",
    )
    config_txt_path = tmp_path / "config.txt"
    config_txt_path.write_text("", encoding="utf-8")

    monkeypatch.setenv("FISH_AUDIO_API_KEY", "fish-audio-secret")
    monkeypatch.delenv("FISH_API_KEY", raising=False)

    manager = TTSConfigManager(yaml_path=yaml_path, config_txt_path=config_txt_path)

    provider_config = manager.get_provider_config("fish_s2")
    assert provider_config is not None
    assert provider_config.backend == "commercial_api"
    assert provider_config.api_key == "fish-audio-secret"


@pytest.mark.unit
def test_fish_api_key_env_alias_is_supported(tmp_path, monkeypatch):
    yaml_path = tmp_path / "tts.yaml"
    yaml_path.write_text(
        """
providers:
  fish_s2:
    enabled: true
    backend: commercial_api
    api_key: null
""".strip(),
        encoding="utf-8",
    )
    config_txt_path = tmp_path / "config.txt"
    config_txt_path.write_text("", encoding="utf-8")

    monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)
    monkeypatch.setenv("FISH_API_KEY", "fish-secret")

    manager = TTSConfigManager(yaml_path=yaml_path, config_txt_path=config_txt_path)

    provider_config = manager.get_provider_config("fish_s2")
    assert provider_config is not None
    assert provider_config.backend == "commercial_api"
    assert provider_config.api_key == "fish-secret"

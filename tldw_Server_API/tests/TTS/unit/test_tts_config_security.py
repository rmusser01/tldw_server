"""Security regression tests for TTS config serialization."""

import pytest
import yaml

from tldw_Server_API.app.core.TTS.tts_config import ProviderConfig, TTSConfig, TTSConfigManager

pytestmark = pytest.mark.unit


def _manager_with_config(config: TTSConfig) -> TTSConfigManager:
    manager = TTSConfigManager.__new__(TTSConfigManager)
    manager.yaml_path = None
    manager.config_txt_path = None
    manager._config = config
    manager._env_overrides = {}
    manager._sources = {}
    return manager


def test_to_dict_redacts_provider_api_keys_by_default():
    manager = _manager_with_config(
        TTSConfig(
            providers={
                "openai": ProviderConfig(enabled=True, api_key="sk-secret-openai"),
                "elevenlabs": ProviderConfig(enabled=True, api_key="xi-secret"),
            }
        )
    )

    redacted = manager.to_dict()
    with_secrets = manager.to_dict(include_secrets=True)

    assert redacted["providers"]["openai"]["api_key"] == "********"
    assert redacted["providers"]["elevenlabs"]["api_key"] == "********"
    assert "sk-secret-openai" not in repr(redacted)
    assert with_secrets["providers"]["openai"]["api_key"] == "sk-secret-openai"


def test_save_yaml_redacts_provider_api_keys_by_default(tmp_path):
    manager = _manager_with_config(
        TTSConfig(
            providers={
                "openai": ProviderConfig(enabled=True, api_key="sk-secret-openai"),
            }
        )
    )
    path = tmp_path / "tts.yaml"

    manager.save_yaml(path)

    data = yaml.safe_load(path.read_text())
    assert data["providers"]["openai"]["api_key"] == "********"
    assert "sk-secret-openai" not in path.read_text()


def test_save_yaml_refuses_redacted_canonical_config(tmp_path):
    manager = _manager_with_config(
        TTSConfig(
            providers={
                "openai": ProviderConfig(enabled=True, api_key="sk-secret-openai"),
            }
        )
    )
    manager.yaml_path = tmp_path / "tts.yaml"

    with pytest.raises(ValueError, match="redacted provider secrets"):
        manager.save_yaml()


def test_save_yaml_can_persist_canonical_config_with_explicit_secrets(tmp_path):
    manager = _manager_with_config(
        TTSConfig(
            providers={
                "openai": ProviderConfig(enabled=True, api_key="sk-secret-openai"),
            }
        )
    )
    manager.yaml_path = tmp_path / "tts.yaml"

    manager.save_yaml(include_secrets=True)

    data = yaml.safe_load(manager.yaml_path.read_text())
    assert data["providers"]["openai"]["api_key"] == "sk-secret-openai"


def test_env_overrides_ignore_non_tts_anthropic_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    manager = TTSConfigManager.__new__(TTSConfigManager)

    env_config = manager._load_env_overrides()

    assert "anthropic" not in env_config.get("providers", {})

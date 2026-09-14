"""Security regression tests for TTS config serialization."""

import pytest
import yaml

from tldw_Server_API.app.core.TTS.gateway_config import GatewayConfig
from tldw_Server_API.app.core.TTS.tts_config import ProviderConfig, TTSConfig, TTSConfigManager

pytestmark = pytest.mark.unit


def _manager_with_config(config: TTSConfig) -> TTSConfigManager:
    """Build a TTSConfigManager around an in-memory config for tests."""
    manager = TTSConfigManager.__new__(TTSConfigManager)
    manager.yaml_path = None
    manager.config_txt_path = None
    manager._config = config
    manager._env_overrides = {}
    manager._sources = {}
    manager._gateway_specs = None
    return manager


def _enabled_gateway(**overrides):
    config = {
        "enabled": True,
        "display_name": "Round Trip Gateway",
        "base_url": "https://speech.example.com/v1/",
        "speech_path": "audio/speech",
        "api_key": "gateway-round-trip-secret",
        "default_model": "Vendor/Model",
        "default_voice": "narrator",
        "capability_defaults": {"formats": ["mp3", "pcm"]},
        "headers": {"X-Tenant": "configured"},
        "discovery": {
            "enabled": True,
            "models_path": "models",
            "query": {"output_modalities": "speech"},
        },
        "allowed_request_options": ["/provider/options/style"],
    }
    config.update(overrides)
    return config


def test_to_dict_redacts_provider_api_keys_by_default():
    """Verify serialized TTS config redacts provider API keys by default."""
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
    """Verify ad hoc YAML export redacts provider API keys by default."""
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
    """Verify canonical config saves cannot accidentally persist redacted secrets."""
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
    """Verify canonical config saves can include secrets only when explicit."""
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


def test_gateway_yaml_include_secrets_round_trips_through_safe_load(tmp_path):
    path = tmp_path / "tts-gateway.yaml"
    manager = _manager_with_config(
        TTSConfig(gateways={"round-trip": _enabled_gateway()})
    )
    manager.yaml_path = path

    manager.save_yaml(include_secrets=True)

    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    reloaded = TTSConfigManager(
        yaml_path=path,
        config_txt_path=tmp_path / "missing-config.txt",
    )
    gateway = reloaded.get_config().gateways["round-trip"]
    assert "!!python/tuple" not in text
    assert loaded["gateways"]["round-trip"]["api_key"] == "gateway-round-trip-secret"
    assert gateway.capability_defaults.formats == ("mp3", "pcm")
    assert gateway.headers == (("X-Tenant", "configured"),)
    assert gateway.discovery.query == (("output_modalities", "speech"),)
    assert gateway.allowed_request_options == ("/provider/options/style",)


def test_failed_reload_keeps_prior_config_specs_and_sources_atomic(tmp_path):
    path = tmp_path / "tts-atomic.yaml"
    path.write_text(
        yaml.safe_dump({"gateways": {"stable": _enabled_gateway()}}),
        encoding="utf-8",
    )
    manager = TTSConfigManager(
        yaml_path=path,
        config_txt_path=tmp_path / "missing-config.txt",
    )
    old_config = manager.get_config()
    old_specs = manager.get_gateway_specs()
    old_generation = old_specs["gateway:stable"].config_generation
    old_sources = manager.get_sources()
    path.write_text(
        yaml.safe_dump(
            {
                "gateways": {
                    "stable": _enabled_gateway(
                        models_path="models-a",
                        discovery={"models_path": "models-b"},
                    )
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="models_path"):
        manager.reload()

    assert manager.get_config() is old_config
    assert manager.get_gateway_specs() is old_specs
    assert manager.get_gateway_spec("stable").config_generation == old_generation
    assert manager.get_sources() == old_sources


def test_gateway_secret_is_hidden_from_repr_and_validation_errors():
    secret = "gateway-ultra-private-secret"
    config = GatewayConfig(enabled=False, api_key=secret)

    with pytest.raises(ValueError) as exc_info:
        TTSConfig(
            gateways={
                "broken": _enabled_gateway(
                    api_key=secret,
                    capability_defaults={
                        "formats": ["mp3"],
                        "max_response_bytes": "not-an-integer",
                    },
                )
            }
        )

    assert secret not in repr(config)
    assert secret not in str(exc_info.value)


def test_env_overrides_ignore_non_tts_anthropic_api_key(monkeypatch):
    """Verify generic Anthropic API keys are ignored by TTS env overrides."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    manager = TTSConfigManager.__new__(TTSConfigManager)

    env_config = manager._load_env_overrides()

    assert "anthropic" not in env_config.get("providers", {})

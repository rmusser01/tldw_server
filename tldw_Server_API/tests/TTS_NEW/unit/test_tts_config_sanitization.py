import pytest

from tldw_Server_API.app.core.TTS import tts_config


@pytest.mark.unit
def test_load_config_txt_sanitizes_config_loader_failure_log(monkeypatch):
    sensitive_error = "config backend exploded /tmp/private-config"

    def fail_load_config():
        raise RuntimeError(sensitive_error)

    monkeypatch.setattr(tts_config, "load_comprehensive_config", fail_load_config)
    manager = tts_config.TTSConfigManager.__new__(tts_config.TTSConfigManager)
    manager.config_txt_path = None

    messages: list[str] = []
    sink_id = tts_config.logger.add(
        lambda message: messages.append(str(message)),
        level="ERROR",
    )
    try:
        result = manager._load_config_txt()
    finally:
        tts_config.logger.remove(sink_id)

    rendered_logs = "\n".join(messages)
    assert result == {}
    assert "Error loading config.txt" in rendered_logs
    assert "config backend exploded" not in rendered_logs
    assert "/tmp/private-config" not in rendered_logs


@pytest.mark.unit
def test_gateway_secrets_are_redacted_and_detected():
    manager = tts_config.TTSConfigManager.__new__(tts_config.TTSConfigManager)
    gateway = {
        "enabled": False,
        "api_key": "gateway-secret",
    }
    manager._config = tts_config.TTSConfig(
        providers={"openrouter": {**gateway, "api_key": "openrouter-secret"}},
        gateways={"company": gateway},
    )

    exported = manager.to_dict()

    assert exported["providers"]["openrouter"]["api_key"] == tts_config.REDACTED_SECRET
    assert exported["gateways"]["company"]["api_key"] == tts_config.REDACTED_SECRET
    assert manager._has_provider_secrets() is True

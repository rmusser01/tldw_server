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

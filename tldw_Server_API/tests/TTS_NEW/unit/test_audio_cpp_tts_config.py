from pathlib import Path

import pytest

from tldw_Server_API.app.core.TTS.tts_config import TTSConfigManager

EMPTY_CONFIG_TXT_PATH = Path("tldw_Server_API/tests/TTS_NEW/fixtures/empty_config.txt")


@pytest.mark.unit
def test_audio_cpp_yaml_config_is_disabled_and_preserves_runtime_settings():
    manager = TTSConfigManager(
        yaml_path=Path("tldw_Server_API/Config_Files/tts_providers_config.yaml"),
        config_txt_path=EMPTY_CONFIG_TXT_PATH,
    )

    provider_config = manager.get_provider_config("audio_cpp")

    assert provider_config is not None
    assert provider_config.enabled is False
    assert provider_config.backend == "cuda"
    assert provider_config.base_url == "http://127.0.0.1:8080"
    assert provider_config.model == "audio-cpp/pocket-tts"
    assert provider_config.model_path == "models/audio_cpp/pocket-tts"
    assert provider_config.binary_path is None
    assert provider_config.device == "cuda"
    assert provider_config.timeout == 300
    assert provider_config.sample_rate == 24000
    assert provider_config.max_concurrent_generations == 1
    assert provider_config.auto_download is False

    extra = provider_config.extra_params
    assert extra["managed"] is False
    assert extra["allow_remote_base_url"] is False
    assert extra["external_voice_reference_mode"] == "disabled"
    assert extra["retain_request_artifacts"] is False
    assert extra["request_option_allowlist"] == ["max_tokens", "seed"]
    assert extra["server"]["host"] == "127.0.0.1"
    assert extra["server"]["autoselect_port"] is True
    assert extra["server"]["model"]["id"] == "pocket-tts"
    assert extra["voices"]["alba"]["request_field"] is None


@pytest.mark.unit
def test_audio_cpp_format_preferences_are_limited_to_verified_conversion_targets():
    manager = TTSConfigManager(
        yaml_path=Path("tldw_Server_API/Config_Files/tts_providers_config.yaml"),
        config_txt_path=EMPTY_CONFIG_TXT_PATH,
    )

    formats = manager.get_config().format_preferences["audio_cpp"]

    assert formats == ["wav", "mp3", "opus", "flac", "aac", "pcm"]
    assert "ogg" not in formats
    assert "webm" not in formats
    assert "ulaw" not in formats

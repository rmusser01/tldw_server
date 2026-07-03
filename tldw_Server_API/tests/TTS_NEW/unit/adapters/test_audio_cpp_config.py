from pathlib import Path

import pytest

from tldw_Server_API.app.core.TTS.adapters.audio_cpp_config import (
    AudioCppConfig,
    filter_request_options,
    validate_base_url,
    validate_managed_host,
)
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSValidationError


def _provider_config(**overrides):
    config = {
        "base_url": "http://127.0.0.1:8080",
        "model": "audio-cpp/pocket-tts",
        "model_path": "models/audio_cpp/pocket-tts",
        "timeout": 300,
        "extra_params": {
            "managed": False,
            "allow_remote_base_url": False,
            "external_voice_reference_mode": "disabled",
            "request_option_allowlist": ["max_tokens", "seed"],
            "server": {
                "host": "127.0.0.1",
                "port": 8080,
                "autoselect_port": True,
                "models_root": "models/audio_cpp",
                "shared_scratch_dir": "models/audio_cpp/runtime/scratch",
                "lazy_load": True,
                "device": 0,
                "threads": 1,
                "model": {
                    "id": "pocket-tts",
                    "family": "pocket_tts",
                    "path": "models/audio_cpp/pocket-tts",
                    "task": "tts",
                    "mode": "offline",
                    "load_options": {"language": "english"},
                    "session_options": {"language": "english"},
                },
            },
        },
    }
    config.update(overrides)
    return config


@pytest.mark.unit
def test_validate_base_url_requires_loopback_by_default():
    assert validate_base_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert validate_base_url("http://localhost:8080/") == "http://localhost:8080"

    with pytest.raises(TTSValidationError):
        validate_base_url("http://example.com:8080")

    assert (
        validate_base_url("http://example.com:8080", allow_remote_base_url=True)
        == "http://example.com:8080"
    )


@pytest.mark.unit
def test_validate_managed_host_allows_loopback_only():
    assert validate_managed_host("127.0.0.1") == "127.0.0.1"
    assert validate_managed_host("localhost") == "127.0.0.1"

    with pytest.raises(TTSValidationError):
        validate_managed_host("0.0.0.0")


@pytest.mark.unit
def test_filter_request_options_passes_only_allowlisted_scalars():
    filtered, ignored = filter_request_options(
        {
            "max_tokens": 128,
            "seed": 42,
            "temperature": 0.7,
            "nested": {"bad": True},
            "items": [1, 2],
            "empty": None,
        },
        allowlist=("max_tokens", "seed", "nested", "items", "empty"),
    )

    assert filtered == {"max_tokens": 128, "seed": 42}
    assert ignored == {
        "temperature": "not_allowlisted",
        "nested": "non_scalar",
        "items": "non_scalar",
        "empty": "none_value",
    }


@pytest.mark.unit
def test_audio_cpp_config_renders_single_model_server_config():
    cfg = AudioCppConfig.from_provider_config(_provider_config(), repo_root=Path.cwd())

    server_config = cfg.render_server_config()

    assert server_config["host"] == "127.0.0.1"
    assert server_config["port"] == 8080
    assert server_config["lazy_load"] is True
    assert server_config["device"] == 0
    assert server_config["threads"] == 1
    assert len(server_config["models"]) == 1
    model = server_config["models"][0]
    assert model["id"] == "pocket-tts"
    assert model["family"] == "pocket_tts"
    assert Path(model["path"]).is_absolute()
    assert model["task"] == "tts"
    assert model["mode"] == "offline"
    assert model["load_options"] == {"language": "english"}
    assert model["session_options"] == {"language": "english"}


@pytest.mark.unit
def test_audio_cpp_config_rejects_model_paths_outside_models_root():
    config = _provider_config()
    config["extra_params"]["server"]["model"]["path"] = "../outside-model"

    with pytest.raises(TTSValidationError):
        AudioCppConfig.from_provider_config(config, repo_root=Path.cwd()).render_server_config()


@pytest.mark.unit
def test_audio_cpp_config_builds_reference_paths_without_user_filename():
    cfg = AudioCppConfig.from_provider_config(_provider_config(), repo_root=Path.cwd())

    path = cfg.build_reference_scratch_path("my private voice.wav")

    assert path.parent == cfg.shared_scratch_dir
    assert path.suffix == ".wav"
    assert "my private voice" not in path.name
    assert path.name.startswith("voice_ref_")

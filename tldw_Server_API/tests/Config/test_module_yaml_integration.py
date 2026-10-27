import json
from unittest.mock import patch

import pytest

from tldw_Server_API.app.core import config as core_config
from tldw_Server_API.app.core.Embeddings import simplified_config
from tldw_Server_API.app.core.TTS.tts_config import TTSConfigManager


@pytest.fixture(autouse=True)
def _reset_config_cache(monkeypatch):
    core_config.clear_config_cache()
    for key in (
        "TLDW_CONFIG_FILE",
        "TLDW_CONFIG_PATH",
        "TLDW_CONFIG_DIR",
        "EMBEDDINGS_CONFIG_PATH",
        "EMBEDDINGS_DEFAULT_PROVIDER",
        "EMBEDDINGS_PROVIDER",
        "EMBEDDINGS_DEFAULT_MODEL",
        "EMBEDDINGS_MODEL",
        "EMBEDDINGS_CHUNK_SIZE",
        "EMBEDDINGS_CHUNK_OVERLAP",
        "EMBEDDINGS_API_URL",
        "EMBEDDINGS_LOCAL_API_URL",
        "TTS_DEFAULT_PROVIDER",
        "EVALUATIONS_CONFIG_OVERRIDES",
    ):
        monkeypatch.delenv(key, raising=False)
    yield
    core_config.clear_config_cache()


def test_merge_layers_sources():

    from tldw_Server_API.app.core.config_utils import merge_config_layers

    merged, sources = merge_config_layers(
        [
            ("yaml", {"a": 1, "nested": {"x": 1}}),
            ("config", {"nested": {"x": 2, "y": 3}}),
            ("env", {"a": 9}),
        ]
    )

    assert merged["a"] == 9
    assert merged["nested"]["x"] == 2
    assert sources["a"] == "env"
    assert sources["nested"]["x"] == "config"


def test_tts_precedence_env_over_config_and_yaml(tmp_path, monkeypatch):

    yaml_path = tmp_path / "tts_providers_config.yaml"
    yaml_path.write_text("default_provider: yaml_provider\nproviders: {}\n", encoding="utf-8")

    config_path = tmp_path / "config.txt"
    config_path.write_text(
        "[TTS-Settings]\ndefault_tts_provider = config_provider\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("TTS_DEFAULT_PROVIDER", "env_provider")

    manager = TTSConfigManager(yaml_path=yaml_path, config_txt_path=config_path)
    cfg = manager.get_config()

    assert cfg.default_provider == "env_provider"


def test_tts_config_txt_legacy_keys_map_to_canonical_fields(tmp_path):

    yaml_path = tmp_path / "tts_providers_config.yaml"
    yaml_path.write_text("providers: {}\n", encoding="utf-8")

    config_path = tmp_path / "config.txt"
    config_path.write_text(
        "\n".join(
            [
                "[TTS-Settings]",
                "default_tts_provider = legacy_provider",
                "default_tts_voice = legacy_voice",
                "default_tts_speed = 1.25",
                "local_tts_device = cuda",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manager = TTSConfigManager(yaml_path=yaml_path, config_txt_path=config_path)
    cfg = manager.get_config()

    assert cfg.default_provider == "legacy_provider"
    assert cfg.default_voice == "legacy_voice"
    assert cfg.default_speed == 1.25
    assert cfg.local_device == "cuda"


def test_tts_config_txt_canonical_keys_override_legacy_aliases(tmp_path):

    yaml_path = tmp_path / "tts_providers_config.yaml"
    yaml_path.write_text("providers: {}\n", encoding="utf-8")

    config_path = tmp_path / "config.txt"
    config_path.write_text(
        "\n".join(
            [
                "[TTS-Settings]",
                "default_provider = canonical_provider",
                "default_tts_provider = legacy_provider",
                "default_voice = canonical_voice",
                "default_tts_voice = legacy_voice",
                "default_speed = 1.1",
                "default_tts_speed = 1.9",
                "local_device = cpu",
                "local_tts_device = cuda",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manager = TTSConfigManager(yaml_path=yaml_path, config_txt_path=config_path)
    cfg = manager.get_config()

    assert cfg.default_provider == "canonical_provider"
    assert cfg.default_voice == "canonical_voice"
    assert cfg.default_speed == 1.1
    assert cfg.local_device == "cpu"


def test_tts_config_txt_legacy_keys_emit_deprecation_warnings(tmp_path):

    yaml_path = tmp_path / "tts_providers_config.yaml"
    yaml_path.write_text("providers: {}\n", encoding="utf-8")

    config_path = tmp_path / "config.txt"
    config_path.write_text(
        "\n".join(
            [
                "[TTS-Settings]",
                "default_tts_provider = legacy_provider",
                "local_tts_device = cuda",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with patch("tldw_Server_API.app.core.TTS.tts_config.logger.warning") as mock_warning:
        manager = TTSConfigManager(yaml_path=yaml_path, config_txt_path=config_path)
        _ = manager.get_config()

    warning_messages = [str(call.args[0]) for call in mock_warning.call_args_list]
    assert any("default_tts_provider" in msg for msg in warning_messages)
    assert any("local_tts_device" in msg for msg in warning_messages)
    assert any("2026-06-30" in msg for msg in warning_messages)


def test_tts_config_txt_legacy_key_ignored_when_canonical_present(tmp_path):

    yaml_path = tmp_path / "tts_providers_config.yaml"
    yaml_path.write_text("providers: {}\n", encoding="utf-8")

    config_path = tmp_path / "config.txt"
    config_path.write_text(
        "\n".join(
            [
                "[TTS-Settings]",
                "default_provider = canonical_provider",
                "default_tts_provider = legacy_provider",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with patch("tldw_Server_API.app.core.TTS.tts_config.logger.warning") as mock_warning:
        manager = TTSConfigManager(yaml_path=yaml_path, config_txt_path=config_path)
        cfg = manager.get_config()

    assert cfg.default_provider == "canonical_provider"
    warning_messages = [str(call.args[0]) for call in mock_warning.call_args_list]
    assert any("ignored when 'default_provider' is also set" in msg for msg in warning_messages)


def test_tts_config_txt_local_settings_fan_out_to_kitten_tts(tmp_path):

    yaml_path = tmp_path / "tts_providers_config.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "providers:",
                "  kitten_tts:",
                "    enabled: false",
                "    auto_download: false",
                "  kokoro:",
                "    enabled: false",
                "    auto_download: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "config.txt"
    config_path.write_text(
        "\n".join(
            [
                "[TTS-Settings]",
                "local_device = cuda",
                "auto_download_local_models = true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manager = TTSConfigManager(yaml_path=yaml_path, config_txt_path=config_path)
    cfg = manager.get_config()

    assert cfg.providers["kitten_tts"].device == "cuda"
    assert cfg.providers["kitten_tts"].auto_download is True


def test_default_tts_yaml_includes_kitten_tts_provider():

    manager = TTSConfigManager()
    cfg = manager.get_config()

    assert "kitten_tts" in cfg.providers
    assert cfg.providers["kitten_tts"].model == "KittenML/kitten-tts-nano-0.8"
    assert cfg.providers["kitten_tts"].model_revision == "8d6d5a1851ffd13c894c40227c888302c2a86ef7"
    assert cfg.providers["kitten_tts"].auto_download is True


def test_embeddings_precedence_env_over_config_over_yaml(tmp_path, monkeypatch):

    config_dir = tmp_path
    yaml_path = config_dir / "embeddings_config.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "providers:",
                "  - name: local",
                "    api_url: http://yaml.local",
                "default_provider: yaml_provider",
                "default_model: yaml_model",
                "chunk_size: 111",
                "chunk_overlap: 222",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config_path = config_dir / "config.txt"
    config_path.write_text(
        "\n".join(
            [
                "[Embeddings]",
                "embedding_provider = config_provider",
                "chunk_size = 333",
                "overlap = 444",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("TLDW_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "env_provider")
    monkeypatch.setenv("EMBEDDINGS_CHUNK_SIZE", "555")

    cfg = simplified_config.load_config()

    assert cfg.default_provider == "env_provider"
    assert cfg.chunk_size == 555
    assert cfg.chunk_overlap == 444


def test_evaluations_precedence_env_over_config_over_yaml(tmp_path, monkeypatch):

    yaml_path = tmp_path / "evaluations_config.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "rate_limiting:",
                "  global:",
                "    default_tier: free",
                "  tiers:",
                "    free:",
                "      evaluations_per_minute: 1",
                "      batch_evaluations_per_minute: 1",
                "      evaluations_per_day: 1",
                "      total_tokens_per_day: 1",
                "      burst_size: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "config.txt"
    config_path.write_text(
        "[Evaluations]\nrate_limiting.global.default_tier = premium\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("TLDW_CONFIG_FILE", str(config_path))
    monkeypatch.setenv(
        "EVALUATIONS_CONFIG_OVERRIDES",
        json.dumps({"rate_limiting": {"global": {"default_tier": "env"}}}),
    )

    from tldw_Server_API.app.core.Evaluations.config_manager import EvaluationsConfigManager

    manager = EvaluationsConfigManager(config_path=yaml_path, enable_hot_reload=False)

    assert manager.get_config("rate_limiting.global.default_tier") == "env"

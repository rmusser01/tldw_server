from __future__ import annotations

import pytest

from tldw_Server_API.app.core import config


@pytest.fixture(autouse=True)
def _reset_config_cache(monkeypatch):
    config.clear_config_cache()
    for key in ("TLDW_CONFIG_FILE", "TLDW_CONFIG_PATH", "TLDW_CONFIG_DIR"):
        monkeypatch.delenv(key, raising=False)
    yield
    config.clear_config_cache()
    for key in ("TLDW_CONFIG_FILE", "TLDW_CONFIG_PATH", "TLDW_CONFIG_DIR"):
        monkeypatch.delenv(key, raising=False)


def load_settings_for_test() -> dict:
    config.clear_config_cache()
    return dict(config.load_settings())


def load_runtime_config_for_test() -> dict:
    config.clear_config_cache()
    return dict(config.load_and_log_configs())


def _contains_placeholder_literal(value: object) -> bool:
    placeholders = {
        "FIXME",
        "TODO",
        "TBD",
        "CHANGE_ME",
        "CHANGE-ME",
        "PLACEHOLDER",
        "NONE",
        "NULL",
        "N/A",
        "NA",
    }
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in placeholders:
            return True
        if "..." in value and value.lower().startswith(("sk-", "el-")):
            return True
        return False
    if isinstance(value, dict):
        return any(_contains_placeholder_literal(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_placeholder_literal(v) for v in value)
    return False


def test_env_overrides_config_file_for_redis_host(tmp_path, monkeypatch):
    cfg = tmp_path / "config.txt"
    cfg.write_text("[Redis]\nredis_host=config-file-host\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_FILE", str(cfg))
    monkeypatch.setenv("REDIS_HOST", "env-host")

    settings = load_settings_for_test()

    assert settings["REDIS_HOST"] == "env-host"  # nosec B101


def test_env_overrides_custom_openai_endpoint_config_values(monkeypatch):
    class FakeConfig:
        def __init__(self, values):
            self._values = values

        def get(self, section, key, fallback=None):
            return self._values.get((section, key), fallback)

        def getboolean(self, section, key, fallback=False):  # noqa: ARG002
            value = self._values.get((section, key))
            if value is None:
                return fallback
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

        def getint(self, section, key, fallback=0):  # noqa: ARG002
            value = self._values.get((section, key))
            if value is None:
                return fallback
            return int(value)

        def getfloat(self, section, key, fallback=0.0):  # noqa: ARG002
            value = self._values.get((section, key))
            if value is None:
                return fallback
            return float(value)

        def has_section(self, section):  # noqa: ARG002
            return False

        def __contains__(self, section):  # noqa: ARG002
            return False

        def __getitem__(self, section):  # noqa: ARG002
            return {}

    fake = FakeConfig(
        {
            ("API", "custom_openai_api_ip"): "https://api.openai.com/v1",
            ("API", "custom_openai2_api_ip"): "https://api.openai.com/v1",
        }
    )

    def _fake_loader():
        return fake

    _fake_loader.cache_clear = lambda: None
    monkeypatch.setattr(config, "load_comprehensive_config", _fake_loader)
    monkeypatch.setenv("CUSTOM_OPENAI_API_IP", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("CUSTOM_OPENAI2_API_IP", "http://127.0.0.1:8002/v1")

    cfg = config.load_and_log_configs()

    assert cfg["custom_openai_api"]["api_ip"] == "http://127.0.0.1:8000/v1"  # nosec B101
    assert cfg["custom_openai_api_2"]["api_ip"] == "http://127.0.0.1:8002/v1"  # nosec B101


@pytest.mark.parametrize("alias_env", ["CUSTOM_OPENAI_API_BASE", "CUSTOM_OPENAI_API_URL"])
def test_custom_openai_endpoint_alias_overrides_config_value(monkeypatch, alias_env):
    class FakeConfig:
        def get(self, section, key, fallback=None):
            if (section, key) == ("API", "custom_openai_api_ip"):
                return "https://api.openai.com/v1"
            return fallback

        def getboolean(self, section, key, fallback=False):  # noqa: ARG002
            return fallback

        def getint(self, section, key, fallback=0):  # noqa: ARG002
            return fallback

        def getfloat(self, section, key, fallback=0.0):  # noqa: ARG002
            return fallback

        def has_section(self, section):  # noqa: ARG002
            return False

        def __contains__(self, section):  # noqa: ARG002
            return False

        def __getitem__(self, section):  # noqa: ARG002
            return {}

    def _fake_loader():
        return FakeConfig()

    _fake_loader.cache_clear = lambda: None
    monkeypatch.setattr(config, "load_comprehensive_config", _fake_loader)
    monkeypatch.setenv(alias_env, "http://127.0.0.1:9000/v1")

    cfg = config.load_and_log_configs()

    assert cfg["custom_openai_api"]["api_ip"] == "http://127.0.0.1:9000/v1"  # nosec B101


def test_custom_openai2_config_is_not_overridden_by_provider1_env(monkeypatch):
    class FakeConfig:
        def get(self, section, key, fallback=None):
            values = {
                ("API", "custom_openai_api_ip"): "https://provider-one.example/v1",
                ("API", "custom_openai2_api_ip"): "https://provider-two.example/v1",
            }
            return values.get((section, key), fallback)

        def getboolean(self, section, key, fallback=False):  # noqa: ARG002
            return fallback

        def getint(self, section, key, fallback=0):  # noqa: ARG002
            return fallback

        def getfloat(self, section, key, fallback=0.0):  # noqa: ARG002
            return fallback

        def has_section(self, section):  # noqa: ARG002
            return False

        def __contains__(self, section):  # noqa: ARG002
            return False

        def __getitem__(self, section):  # noqa: ARG002
            return {}

    def _fake_loader():
        return FakeConfig()

    _fake_loader.cache_clear = lambda: None
    monkeypatch.setattr(config, "load_comprehensive_config", _fake_loader)
    monkeypatch.setenv("CUSTOM_OPENAI_API_IP", "http://127.0.0.1:9001/v1")

    cfg = config.load_and_log_configs()

    assert cfg["custom_openai_api"]["api_ip"] == "http://127.0.0.1:9001/v1"  # nosec B101
    assert cfg["custom_openai_api_2"]["api_ip"] == "https://provider-two.example/v1"  # nosec B101


def test_numbered_custom_openai_endpoint_env_creates_config_section(monkeypatch):
    class FakeConfig:
        def get(self, section, key, fallback=None):  # noqa: ARG002
            return fallback

        def getboolean(self, section, key, fallback=False):  # noqa: ARG002
            return fallback

        def getint(self, section, key, fallback=0):  # noqa: ARG002
            return fallback

        def getfloat(self, section, key, fallback=0.0):  # noqa: ARG002
            return fallback

        def has_section(self, section):  # noqa: ARG002
            return False

        def has_option(self, section, option):  # noqa: ARG002
            return False

        def __contains__(self, section):  # noqa: ARG002
            return False

        def __getitem__(self, section):  # noqa: ARG002
            return {}

    def _fake_loader():
        return FakeConfig()

    _fake_loader.cache_clear = lambda: None
    monkeypatch.setattr(config, "load_comprehensive_config", _fake_loader)
    monkeypatch.setenv("CUSTOM_OPENAI_API_IP_37", "http://127.0.0.1:8037/v1")
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY_37", "key-37")
    monkeypatch.setenv("CUSTOM_OPENAI_API_MODEL_37", "model-37")

    cfg = config.load_and_log_configs()

    assert cfg.get("custom_openai_api_37", {}).get("api_ip") == "http://127.0.0.1:8037/v1"  # nosec B101
    assert cfg.get("custom_openai_api_37", {}).get("api_key") == "key-37"  # nosec B101
    assert cfg.get("custom_openai_api_37", {}).get("model") == "model-37"  # nosec B101


def test_numbered_custom_openai_endpoint_config_fallback_through_99(monkeypatch):
    class FakeConfig:
        def get(self, section, key, fallback=None):
            values = {
                ("API", "custom_openai99_api_ip"): "http://127.0.0.1:8099/v1",
                ("API", "custom_openai99_api_key"): "key-99",
                ("API", "custom_openai99_api_model"): "model-99",
            }
            return values.get((section, key), fallback)

        def getboolean(self, section, key, fallback=False):  # noqa: ARG002
            return fallback

        def getint(self, section, key, fallback=0):  # noqa: ARG002
            return fallback

        def getfloat(self, section, key, fallback=0.0):  # noqa: ARG002
            return fallback

        def has_section(self, section):  # noqa: ARG002
            return section == "API"

        def has_option(self, section, option):
            return (section, option) in {
                ("API", "custom_openai99_api_ip"),
                ("API", "custom_openai99_api_key"),
                ("API", "custom_openai99_api_model"),
            }

        def __contains__(self, section):  # noqa: ARG002
            return False

        def __getitem__(self, section):  # noqa: ARG002
            return {}

    def _fake_loader():
        return FakeConfig()

    _fake_loader.cache_clear = lambda: None
    monkeypatch.setattr(config, "load_comprehensive_config", _fake_loader)

    cfg = config.load_and_log_configs()

    assert cfg.get("custom_openai_api_99", {}).get("api_ip") == "http://127.0.0.1:8099/v1"  # nosec B101
    assert cfg.get("custom_openai_api_99", {}).get("api_key") == "key-99"  # nosec B101
    assert cfg.get("custom_openai_api_99", {}).get("model") == "model-99"  # nosec B101


def test_missing_tts_defaults_never_emit_fixme_literal():
    settings = load_settings_for_test()
    assert "FIXME" not in str(settings.get("TTS_CONFIG", {}))  # nosec B101


def test_section_loaders_return_typed_models():
    sections = config.load_all_sections_for_test()
    assert hasattr(sections, "auth")  # nosec B101
    assert hasattr(sections, "rag")  # nosec B101
    assert hasattr(sections, "audio")  # nosec B101
    assert hasattr(sections, "providers")  # nosec B101


def test_tts_defaults_are_valid_values_not_placeholders(monkeypatch):
    class FakeConfig:
        def __init__(self, values):
            self._values = values

        def get(self, section, key, fallback=None):
            return self._values.get((section, key), fallback)

        def getboolean(self, section, key, fallback=False):  # noqa: ARG002
            value = self._values.get((section, key))
            if value is None:
                return fallback
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

        def getint(self, section, key, fallback=0):  # noqa: ARG002
            value = self._values.get((section, key))
            if value is None:
                return fallback
            return int(value)

        def getfloat(self, section, key, fallback=0.0):  # noqa: ARG002
            value = self._values.get((section, key))
            if value is None:
                return fallback
            return float(value)

        def has_section(self, section):  # noqa: ARG002
            return False

        def __contains__(self, section):  # noqa: ARG002
            return False

        def __getitem__(self, section):  # noqa: ARG002
            return {}

    def _fake_loader():
        return FakeConfig({})

    _fake_loader.cache_clear = lambda: None
    monkeypatch.setattr(config, "load_comprehensive_config", _fake_loader)

    cfg = config.load_and_log_configs()
    tts = cfg["tts_settings"]

    assert tts["default_eleven_tts_model"] != "FIXME"  # nosec B101
    assert tts["default_eleven_tts_voice"] != "FIXME"  # nosec B101
    assert tts["default_google_tts_model"] != "FIXME"  # nosec B101
    assert tts["default_google_tts_voice"] != "FIXME"  # nosec B101


def test_runtime_config_never_returns_placeholder_literals():
    runtime_cfg = load_runtime_config_for_test()
    assert "FIXME" not in str(runtime_cfg)  # nosec B101
    assert not _contains_placeholder_literal(runtime_cfg.get("tts_settings", {}))  # nosec B101
    assert not _contains_placeholder_literal(config.APP_CONFIG)  # nosec B101

def test_local_api_and_custom_openai2_config_keys(monkeypatch):
    from tldw_Server_API.app.core import config as cfg

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
            ("Local-API", "aphrodite_api_timeout"): "123",
            ("Local-API", "llama_model"): "../../Language_Models/local-fixture.gguf",
            ("API", "custom_openai2_api_top_p"): "0.42",
        }
    )
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: fake)

    data = cfg.load_and_log_configs()
    assert data["aphrodite_api"]["api_timeout"] == "123"
    assert data["llama_api"]["model"] == "../../Language_Models/local-fixture.gguf"
    assert data["custom_openai_api_2"]["top_p"] == "0.42"

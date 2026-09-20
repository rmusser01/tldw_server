def test_web_scraper_router_config_keys(monkeypatch, tmp_path):
    from tldw_Server_API.app.core import config as cfg

    monkeypatch.delenv("CUSTOM_SCRAPERS_YAML_PATH", raising=False)
    monkeypatch.delenv("WEB_SCRAPER_DEFAULT_BACKEND", raising=False)
    monkeypatch.delenv("WEB_SCRAPER_UA_MODE", raising=False)

    class FakeConfig:
        def __init__(self, values):
            self._values = values

        def get(self, section, key, fallback=None):
            return self._values.get((section, key), fallback)

        def getboolean(self, section, key, fallback=False):  # noqa: ARG002
            return fallback

        def getint(self, section, key, fallback=0):  # noqa: ARG002
            return fallback

        def has_section(self, section):  # noqa: ARG002
            return False

        def __contains__(self, section):  # noqa: ARG002
            return False

        def __getitem__(self, section):  # noqa: ARG002
            return {}

    custom_scrapers_yaml_path = str(tmp_path / "custom_scrapers.yaml")

    fake = FakeConfig(
        {
            ("Web-Scraper", "custom_scrapers_yaml_path"): custom_scrapers_yaml_path,
            ("Web-Scraper", "web_scraper_default_backend"): "curl",
            ("Web-Scraper", "web_scraper_ua_mode"): "rotate",
        }
    )

    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: fake)

    data = cfg.load_and_log_configs()
    ws_cfg = data["web_scraper"]

    assert ws_cfg["custom_scrapers_yaml_path"] == custom_scrapers_yaml_path
    assert ws_cfg["web_scraper_default_backend"] == "curl"
    assert ws_cfg["web_scraper_ua_mode"] == "rotate"


def test_web_scraper_transport_uses_supplied_environment(monkeypatch, tmp_path):
    """Keep browser transport resolution inside the caller's environment snapshot."""
    from tldw_Server_API.app.core import config as cfg

    monkeypatch.setenv("WEB_BROWSER_TRANSPORT_MODE", "disabled")

    class FakeConfig:
        """Return fallbacks for unrelated configuration fields."""

        def get(self, _section, _key, fallback=None):
            """Return the supplied fallback."""
            return fallback

        def getboolean(self, _section, _key, fallback=False):
            """Return the supplied boolean fallback."""
            return fallback

        def getint(self, _section, _key, fallback=0):
            """Return the supplied integer fallback."""
            return fallback

        def has_section(self, _section):
            """Report no optional sections."""
            return False

        def __contains__(self, _section):
            """Report no mapped sections."""
            return False

        def __getitem__(self, _section):
            """Return an empty section mapping."""
            return {}

    monkeypatch.setattr(cfg, "load_comprehensive_config", FakeConfig)

    data = cfg.load_and_log_configs(
        environment={"WEB_BROWSER_TRANSPORT_MODE": "auto"}
    )

    assert data is not None
    assert data["web_scraper"]["web_browser_transport_mode"] == "auto"

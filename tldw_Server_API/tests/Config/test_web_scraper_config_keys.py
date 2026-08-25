import configparser


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


def test_web_scraper_preflight_config_keys_are_retained(monkeypatch):
    from tldw_Server_API.app.core import config as cfg

    configured = {
        "web_scraper_preflight_analyzers": "true",
        "web_scraper_preflight_timeout_s": "12.5",
        "web_scraper_preflight_scan_depth": "deep",
        "web_scraper_preflight_find_all_waf": "true",
        "web_scraper_preflight_impersonate": "true",
        "web_scraper_preflight_include_results": "true",
        "web_scraper_playwright_no_sandbox": "true",
        "web_scraper_preflight_enable_external_tools": "false",
    }
    parser = configparser.ConfigParser()
    parser["Web-Scraper"] = configured
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: parser)

    web_scraper = cfg.load_and_log_configs(environment={})["web_scraper"]

    assert {key: web_scraper[key] for key in configured} == configured


def test_web_scraper_absent_external_tools_config_remains_absent(monkeypatch):
    from tldw_Server_API.app.core import config as cfg

    parser = configparser.ConfigParser()
    parser["Web-Scraper"] = {"web_scraper_preflight_analyzers": "true"}
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: parser)

    web_scraper = cfg.load_and_log_configs(environment={})["web_scraper"]

    assert "web_scraper_preflight_enable_external_tools" not in web_scraper


def test_web_scraper_raw_section_values_do_not_break_preflight_config(monkeypatch):
    from tldw_Server_API.app.core import config as cfg

    parser = configparser.ConfigParser()
    parser.read_string(
        "[Web-Scraper]\n" "web_scraper_preflight_analyzers = true\n" "web_scraper_future_secret = 100%private\n"
    )
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: parser)

    web_scraper = cfg.load_and_log_configs(environment={})["web_scraper"]

    assert web_scraper["web_scraper_preflight_analyzers"] == "true"
    assert web_scraper["web_scraper_future_secret"] == "100%private"


def test_web_scraper_config_excludes_default_only_options(monkeypatch):
    from tldw_Server_API.app.core import config as cfg

    parser = configparser.ConfigParser(defaults={"unrelated_default_secret": "do-not-export"})
    parser["Web-Scraper"] = {"web_scraper_preflight_analyzers": "true"}
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: parser)

    web_scraper = cfg.load_and_log_configs(environment={})["web_scraper"]

    assert "unrelated_default_secret" not in web_scraper


def test_web_scraper_config_retains_explicit_override_of_default(monkeypatch):
    from tldw_Server_API.app.core import config as cfg

    parser = configparser.ConfigParser(defaults={"web_scraper_preflight_scan_depth": "normal"})
    parser["Web-Scraper"] = {"web_scraper_preflight_scan_depth": "deep"}
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: parser)

    web_scraper = cfg.load_and_log_configs(environment={})["web_scraper"]

    assert web_scraper["web_scraper_preflight_scan_depth"] == "deep"

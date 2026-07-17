import configparser

from tldw_Server_API.app.core import config


def test_claims_verification_env_overrides_config_defaults(monkeypatch):
    monkeypatch.setenv("CLAIMS_VERIFICATION_PROVIDER", "openrouter")
    monkeypatch.setenv("CLAIMS_VERIFICATION_MODEL", "claims-env-model")

    parser = configparser.ConfigParser()
    parser.add_section("Claims")
    parser.set("Claims", "CLAIMS_VERIFICATION_PROVIDER", "llamacpp")
    parser.set("Claims", "CLAIMS_VERIFICATION_MODEL", "claims-config-model")

    monkeypatch.setattr(config, "_load_env_files_early", lambda: None, raising=True)
    monkeypatch.setattr(config, "load_and_log_configs", lambda: {}, raising=True)
    monkeypatch.setattr(config, "load_comprehensive_config", lambda: parser, raising=True)

    settings = config.load_settings()

    assert settings["CLAIMS_VERIFICATION_PROVIDER"] == "openrouter"
    assert settings["CLAIMS_VERIFICATION_MODEL"] == "claims-env-model"

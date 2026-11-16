import pytest

from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as ccs


def _clear_cached_provider() -> None:
    try:
        ccs._config_default_llm_provider.cache_clear()
    except AttributeError:
        pass


@pytest.mark.unit
def test_character_default_provider_prefers_config_over_env(monkeypatch):
    _clear_cached_provider()
    monkeypatch.delenv("DEFAULT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.load_and_log_configs",
        lambda: {"llm_api_settings": {"default_api": "config-provider"}},
        raising=False,
    )
    assert ccs._get_default_provider() == "config-provider"
    _clear_cached_provider()


@pytest.mark.unit
def test_character_default_provider_falls_back_to_env(monkeypatch):
    _clear_cached_provider()
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_and_log_configs", lambda: {}, raising=False)
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "env-provider")
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
    assert ccs._get_default_provider() == "env-provider"
    _clear_cached_provider()


@pytest.mark.unit
def test_character_default_provider_uses_local_llm_in_test_mode(monkeypatch):
    _clear_cached_provider()
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_and_log_configs", lambda: {}, raising=False)
    monkeypatch.delenv("DEFAULT_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
    assert ccs._get_default_provider() == "local-llm"
    _clear_cached_provider()

import pytest

from tldw_Server_API.app.core import config_llm_providers
from tldw_Server_API.app.core.Chat import chat_target_resolution

pytestmark = pytest.mark.unit


def _set_env(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize(("config", "env", "test_mode", "expected"), [
    ({"llm_api_settings": {"default_api": " cfg-a "}, "API": {"default_api": "cfg-b"}}, "env-x", "true", "cfg-a"),
    ({"API": {"default_api": "cfg-b"}}, "env-x", "true", "cfg-b"),
    ({"llm_api_settings": {"default_api": "  "}}, " env-x ", "true", "env-x"),
    ({}, None, "true", "local-llm"),
    (None, None, None, "openai"),
])
def test_default_provider_precedence(monkeypatch, config, env, test_mode, expected):
    monkeypatch.setattr(config_llm_providers, "load_and_log_configs", lambda: config)
    _set_env(monkeypatch, "DEFAULT_LLM_PROVIDER", env)
    _set_env(monkeypatch, "TEST_MODE", test_mode)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
    assert config_llm_providers.get_default_provider() == expected


def test_default_provider_is_read_at_call_time(monkeypatch):
    monkeypatch.setattr(config_llm_providers, "load_and_log_configs", lambda: {})
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "first")
    assert config_llm_providers.get_default_provider() == "first"
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "second")
    assert config_llm_providers.get_default_provider() == "second"


def test_chat_target_resolution_reexports_the_core_resolvers():
    assert chat_target_resolution.get_default_provider is config_llm_providers.get_default_provider
    assert chat_target_resolution.config_default_llm_provider is config_llm_providers.config_default_llm_provider


def test_core_importer_reads_default_at_call_time(monkeypatch):
    from tldw_Server_API.app.core.Chunking import auto_boundary_assistant

    monkeypatch.setattr(config_llm_providers, "load_and_log_configs", lambda: {})
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "late-env")
    assert auto_boundary_assistant._configured_default_provider({}) == "late-env"


def test_endpoint_reads_default_at_call_time(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import slides

    monkeypatch.setattr(config_llm_providers, "load_and_log_configs", lambda: {})
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "late-env")
    assert slides._resolve_provider(None) == "late-env"

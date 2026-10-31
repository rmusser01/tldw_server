import pytest

from tldw_Server_API.app.api.v1.schemas import chat_request_schemas
from tldw_Server_API.app.core import config_llm_providers
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    LLMProviderOverride,
    set_llm_provider_overrides_cache_for_tests,
)
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.tests.helpers.provider_keys import api_keys_source


@pytest.mark.parametrize("prefer_module_keys_in_tests", [True, False])
def test_resolver_reads_what_production_reads_under_pytest_and_test_mode(monkeypatch, prefer_module_keys_in_tests):
    """TASK-13950.13: the resolver reads config_llm_providers.get_api_keys, whatever the test flags."""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "chat/test")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config_llm_providers, "get_api_keys", lambda: {"openai": "env-key"})

    key, debug = gateway.resolve_provider_api_key("openai", prefer_module_keys_in_tests=prefer_module_keys_in_tests)

    assert (key, debug["selected_source"]) == ("env-key", "config")
    assert "module_sources" not in debug and "test_flags" not in debug


def test_the_old_paths_are_the_moved_objects():
    assert chat_service.resolve_provider_api_key is gateway.resolve_provider_api_key
    assert chat_request_schemas.ALL_SUPPORTED_PROVIDER_NAMES_LIST is config_llm_providers.ALL_SUPPORTED_PROVIDER_NAMES_LIST
    assert not hasattr(chat_service, "resolve_static_provider_fallback")


def test_api_keys_source_layers_test_keys_over_the_real_source(monkeypatch):
    monkeypatch.setattr(config_llm_providers, "get_api_keys", lambda: {"openai": "env-key", "groq": "groq-key"})
    monkeypatch.setattr(config_llm_providers, "get_api_keys", api_keys_source({"openai": "test-key", "anthropic": None}))

    assert gateway.resolve_provider_api_key("openai")[0] == "test-key"
    assert gateway.resolve_provider_api_key("groq")[0] == "groq-key"
    assert gateway.resolve_provider_api_key("anthropic")[0] is None


def test_get_api_keys_supports_hyphenated_provider_env_vars(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "local-key")
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "custom-key")
    monkeypatch.setattr(config_llm_providers, "load_and_log_configs", lambda: {})

    keys = config_llm_providers.get_api_keys()

    assert keys.get("local-llm") == "local-key"
    assert keys.get("custom-openai-api") == "custom-key"


def test_get_api_keys_supports_numbered_custom_openai_env_vars(monkeypatch):
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY_99", "custom-key-99")
    monkeypatch.setattr(config_llm_providers, "load_and_log_configs", lambda: {})

    keys = config_llm_providers.get_api_keys()

    assert keys.get("custom-openai-api-99") == "custom-key-99"


def test_resolver_prefers_provider_override(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config_llm_providers, "get_api_keys", lambda: {"openai": "env-key"})

    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                api_key="override-key",
            )
        }
    )

    try:
        resolved_key, debug_info = gateway.resolve_provider_api_key("openai")
        assert resolved_key == "override-key"
        assert debug_info["selected_source"] == "override"
    finally:
        set_llm_provider_overrides_cache_for_tests({})

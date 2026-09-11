import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.api.v1.schemas import chat_request_schemas
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    LLMProviderOverride,
    set_llm_provider_overrides_cache_for_tests,
)
from tldw_Server_API.app.core.Chat.chat_service import resolve_provider_api_key


def test_resolver_prefers_module_keys_in_tests(monkeypatch):


    monkeypatch.setenv("PYTEST_CURRENT_TEST", "chat/test")
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    monkeypatch.setattr(chat_request_schemas, "API_KEYS", {"openai": "module-test-key"}, raising=False)
    monkeypatch.setattr(chat_endpoint, "API_KEYS", {"openai": "module-test-key"}, raising=False)
    monkeypatch.setattr(chat_request_schemas, "get_api_keys", lambda: {"openai": "env-key"})

    assert chat_request_schemas.API_KEYS.get("openai") == "module-test-key"
    assert chat_request_schemas.get_api_keys().get("openai") == "env-key"

    resolved_key, debug_info = resolve_provider_api_key("openai")

    assert debug_info["selected_source"] == "module_override"
    assert resolved_key == "module-test-key", debug_info
    assert debug_info["test_flags"]["pytest"] is True


def test_resolver_test_module_key_skips_provider_override_lookup(monkeypatch):
    """An exact test-module key must not depend on process-global override state."""
    from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "chat/test")
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        chat_request_schemas,
        "API_KEYS",
        {"openai": "module-test-key"},
        raising=False,
    )
    monkeypatch.setattr(
        chat_endpoint,
        "API_KEYS",
        {"openai": "module-test-key"},
        raising=False,
    )
    monkeypatch.setattr(
        chat_request_schemas,
        "get_api_keys",
        lambda: {"openai": "env-key"},
    )
    monkeypatch.setattr(
        llm_provider_overrides,
        "get_llm_provider_override",
        lambda _provider: pytest.fail("test module key must skip override lookup"),
    )

    resolved_key, debug_info = resolve_provider_api_key("openai")

    assert resolved_key == "module-test-key"
    assert debug_info["selected_source"] == "module_override"


@pytest.mark.parametrize(
    ("module_value", "expected_key"),
    [
        ("module-test-key", "module-test-key"),
        ("", None),
    ],
)
def test_static_fallback_preserves_test_module_override_without_dynamic_config_read(
    monkeypatch,
    module_value,
    expected_key,
):
    """Legacy test module keys remain authoritative without a config reload."""
    from tldw_Server_API.app.core.Chat import chat_service

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "chat/test")
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setattr(
        chat_request_schemas,
        "API_KEYS",
        {"openai": module_value},
        raising=False,
    )
    monkeypatch.setattr(
        chat_endpoint,
        "API_KEYS",
        {"openai": module_value},
        raising=False,
    )
    monkeypatch.setattr(
        chat_request_schemas,
        "get_api_keys",
        lambda: pytest.fail("dynamic API key config must not be read"),
    )
    monkeypatch.setattr(
        chat_service,
        "resolve_static_server_fallback",
        lambda _provider: pytest.fail("module override must short-circuit static config"),
        raising=False,
    )

    fallback = chat_service.resolve_static_provider_fallback("openai")

    assert fallback.api_key == expected_key
    assert fallback.credential_fields == {}
    assert fallback.app_config == {}


def test_resolver_ignores_module_keys_outside_tests(monkeypatch):


    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    monkeypatch.setattr(chat_request_schemas, "API_KEYS", {"openai": "module-test-key"}, raising=False)
    monkeypatch.setattr(chat_endpoint, "API_KEYS", {"openai": "module-test-key"}, raising=False)
    monkeypatch.setattr(chat_request_schemas, "get_api_keys", lambda: {"openai": "env-key"})

    set_llm_provider_overrides_cache_for_tests({})
    try:
        resolved_key, debug_info = resolve_provider_api_key("openai")

        assert resolved_key == "env-key"
        assert debug_info["selected_source"] != "module_override"
    finally:
        set_llm_provider_overrides_cache_for_tests({})


def test_resolver_can_skip_module_preference(monkeypatch):


    monkeypatch.setenv("PYTEST_CURRENT_TEST", "chat/test")
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    monkeypatch.setattr(chat_request_schemas, "API_KEYS", {"openai": "module-test-key"}, raising=False)
    monkeypatch.setattr(chat_endpoint, "API_KEYS", {"openai": "module-test-key"}, raising=False)
    monkeypatch.setattr(chat_request_schemas, "get_api_keys", lambda: {"openai": "env-key"})

    set_llm_provider_overrides_cache_for_tests({})
    try:
        resolved_key, debug_info = resolve_provider_api_key(
            "openai",
            prefer_module_keys_in_tests=False,
        )

        assert resolved_key == "env-key"
        assert debug_info["selected_source"] != "module_override"
    finally:
        set_llm_provider_overrides_cache_for_tests({})


def test_resolver_tldw_test_mode_y_prefers_module_keys(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "y")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    monkeypatch.setattr(chat_request_schemas, "API_KEYS", {"openai": "module-test-key"}, raising=False)
    monkeypatch.setattr(chat_endpoint, "API_KEYS", {"openai": "module-test-key"}, raising=False)
    monkeypatch.setattr(chat_request_schemas, "get_api_keys", lambda: {"openai": "env-key"})

    resolved_key, debug_info = resolve_provider_api_key("openai")

    assert resolved_key == "module-test-key"
    assert debug_info["selected_source"] == "module_override"
    assert debug_info["test_flags"]["test_mode"] is True


def test_get_api_keys_supports_hyphenated_provider_env_vars(monkeypatch):


    monkeypatch.setenv("LOCAL_LLM_API_KEY", "local-key")
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "custom-key")
    monkeypatch.setattr(chat_request_schemas, "load_and_log_configs", lambda: {})

    keys = chat_request_schemas.get_api_keys()

    assert keys.get("local-llm") == "local-key"
    assert keys.get("custom-openai-api") == "custom-key"


def test_get_api_keys_supports_numbered_custom_openai_env_vars(monkeypatch):
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY_99", "custom-key-99")
    monkeypatch.setattr(chat_request_schemas, "load_and_log_configs", lambda: {})

    keys = chat_request_schemas.get_api_keys()

    assert keys.get("custom-openai-api-99") == "custom-key-99"


def test_resolver_prefers_provider_override(monkeypatch):


    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(chat_request_schemas, "get_api_keys", lambda: {"openai": "env-key"})

    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                api_key="override-key",
            )
        }
    )

    try:
        resolved_key, debug_info = resolve_provider_api_key("openai")
        assert resolved_key == "override-key"
        assert debug_info["selected_source"] == "override"
    finally:
        set_llm_provider_overrides_cache_for_tests({})

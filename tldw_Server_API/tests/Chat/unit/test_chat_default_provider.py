import pytest

from tldw_Server_API.app.api.v1.endpoints import chat
from tldw_Server_API.app.core.LLM_Calls.routing.models import RoutingDecision


def _clear_cached_provider() -> None:


    try:
        chat._config_default_llm_provider.cache_clear()
    except AttributeError:
        pass


@pytest.mark.unit
def test_default_provider_prefers_config_over_env(monkeypatch):
    _clear_cached_provider()
    monkeypatch.delenv("DEFAULT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.load_and_log_configs",
        lambda: {"llm_api_settings": {"default_api": "config-provider"}},
        raising=False,
    )
    assert chat._get_default_provider() == "config-provider"
    _clear_cached_provider()


@pytest.mark.unit
def test_default_provider_falls_back_to_env_when_no_config(monkeypatch):
    _clear_cached_provider()
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_and_log_configs", lambda: {}, raising=False)
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "env-provider")
    monkeypatch.delenv("TEST_MODE", raising=False)
    assert chat._get_default_provider() == "env-provider"
    _clear_cached_provider()


@pytest.mark.unit
def test_default_provider_uses_test_mode_local_llm(monkeypatch):
    _clear_cached_provider()
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_and_log_configs", lambda: {}, raising=False)
    monkeypatch.delenv("DEFAULT_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("TEST_MODE", "true")
    assert chat._get_default_provider() == "local-llm"
    _clear_cached_provider()


@pytest.mark.unit
def test_default_provider_uses_tldw_test_mode_single_letter_y(monkeypatch):
    _clear_cached_provider()
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_and_log_configs", lambda: {}, raising=False)
    monkeypatch.delenv("DEFAULT_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "y")
    assert chat._get_default_provider() == "local-llm"
    _clear_cached_provider()


@pytest.mark.unit
def test_resolve_provider_and_model_respects_canonical_routing_decision():
    class RequestData:
        api_provider = None
        model = "auto"

    (
        metrics_provider,
        metrics_model,
        selected_provider,
        selected_model,
        debug_info,
    ) = chat.resolve_provider_and_model(
        request_data=RequestData(),
        metrics_default_provider="openai",
        normalize_default_provider="openai",
        routing_decision=RoutingDecision(
            provider="openrouter",
            model="anthropic/claude-4.5-sonnet",
            canonical=True,
            decision_source="rules_router",
        ),
    )

    assert (metrics_provider, metrics_model) == (
        "openrouter",
        "anthropic/claude-4.5-sonnet",
    )
    assert (selected_provider, selected_model) == (
        "openrouter",
        "anthropic/claude-4.5-sonnet",
    )
    assert debug_info["routing"]["canonical"] is True

"""Server-wide provider overrides must participate in catalog readiness."""

from configparser import ConfigParser
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import llm_providers
from tldw_Server_API.app.core.AuthNZ import byok_runtime, llm_provider_overrides
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import LLMProviderOverride
from tldw_Server_API.app.core.Security.egress import URLPolicyResult


@pytest.fixture
def catalog(monkeypatch):
    """Exercise real readiness and override merging without discovery or secrets."""
    parser = ConfigParser()
    parser.read_dict({"API": {"deepseek_model": "deepseek-chat"}, "Local-API": {}})
    with llm_provider_overrides._OVERRIDE_LOCK:
        original = dict(llm_provider_overrides._OVERRIDE_CACHE)
        healthy = llm_provider_overrides._OVERRIDE_CACHE_HEALTHY
        ttl_enabled = not llm_provider_overrides._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})
    monkeypatch.setattr(llm_providers, "load_comprehensive_config", lambda: parser)
    monkeypatch.setattr(llm_providers, "get_api_keys", lambda: {})
    monkeypatch.setattr(llm_providers, "get_provider_manager", lambda: None)
    monkeypatch.setattr(llm_providers, "_llm_registry_capability_envelopes", lambda: {})
    monkeypatch.setattr(llm_providers, "list_provider_models", lambda _provider: [])
    monkeypatch.setattr(llm_providers, "list_image_models_for_catalog", lambda: [])

    def no_external_calls(*_args, **_kwargs):
        pytest.fail("Catalog regression attempted external discovery")

    monkeypatch.setattr(llm_providers, "_http_fetch", no_external_calls)
    monkeypatch.setattr(llm_providers, "discover_models_from_endpoint", no_external_calls)
    monkeypatch.setattr(llm_providers, "discover_openrouter_models", no_external_calls)
    monkeypatch.setattr(byok_runtime, "resolve_byok_credentials", no_external_calls)
    monkeypatch.setattr(llm_provider_overrides, "_schedule_override_recovery", no_external_calls)
    monkeypatch.setattr(llm_providers, "_resolve_model_tokenizer_support", lambda *_args: {})
    monkeypatch.setenv("LLM_PROVIDER_READINESS_PROBE_ENDPOINTS", "0")
    app = FastAPI()
    app.include_router(llm_providers.router, prefix="/api/v1")
    app.state.llm_manager = SimpleNamespace(llamacpp_supervisor=None)
    try:
        with TestClient(app) as client:
            yield client, parser
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
            original,
            healthy=healthy,
            ttl_enabled=ttl_enabled,
        )


def _deepseek_override(**changes):
    values = {
        "provider": "deepseek",
        "api_key": "catalog-private-test-key",
        "is_enabled": True,
        "allowed_models": ["deepseek-chat"],
        "config": {"model": "deepseek-chat"},
    }
    values.update(changes)
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {"deepseek": LLMProviderOverride(**values)},
    )


@pytest.mark.parametrize("route", ["providers", "providers/deepseek", "models/metadata"])
def test_encrypted_server_override_configures_catalog_without_static_key(catalog, route):
    """An override-only key must reach the same readiness contract on every route."""
    client, _parser = catalog
    _deepseek_override()
    response = client.get(f"/api/v1/llm/{route}")
    assert response.status_code == 200
    data = response.json()
    if route == "models/metadata":
        entry = next(item for item in data["models"] if item["provider"] == "deepseek")
        assert entry["name"] == "deepseek-chat"
        assert entry["catalog_only"] is False
    elif route == "providers":
        entry = next(item for item in data["providers"] if item["name"] == "deepseek")
        assert entry["models"] == ["deepseek-chat"]
    else:
        entry = data
    assert entry["is_configured"] is True
    assert entry["provider_enabled"] is True
    assert entry["availability"] == "enabled"
    assert entry.get("readiness_reason_code") is None
    assert "catalog-private-test-key" not in response.text


@pytest.mark.parametrize(
    "barrier,reason",
    [
        ("unsupported", "unsupported_chat_provider"),
        ("health", "provider_health_unavailable"),
        ("adapter", "provider_unavailable"),
    ],
)
def test_override_key_does_not_bypass_existing_readiness(catalog, monkeypatch, barrier, reason):
    client, _parser = catalog
    _deepseek_override()
    if barrier == "unsupported":
        monkeypatch.setattr(llm_providers, "ALL_SUPPORTED_PROVIDER_NAMES_LIST", [])
    elif barrier == "health":
        monkeypatch.setattr(
            llm_providers,
            "get_provider_manager",
            lambda: SimpleNamespace(
                get_health_report=lambda: {"deepseek": {"status": "unhealthy"}},
            ),
        )
    else:
        monkeypatch.setattr(
            llm_providers,
            "_llm_registry_capability_envelopes",
            lambda: {
                "deepseek": {"availability": "unavailable"},
            },
        )
    response = client.get("/api/v1/llm/providers")
    entry = next(item for item in response.json()["providers"] if item["name"] == "deepseek")
    assert entry["is_configured"] is True
    assert entry["provider_enabled"] is False
    assert entry["availability"] == "unavailable"
    assert entry["readiness_reason_code"] == reason


def test_disabled_override_cannot_advertise_ready_provider(catalog):
    client, _parser = catalog
    _deepseek_override(is_enabled=False)
    response = client.get("/api/v1/llm/providers")
    entry = next(item for item in response.json()["providers"] if item["name"] == "deepseek")
    assert entry["enabled"] is False
    assert entry["provider_enabled"] is False
    assert entry["availability"] == "disabled"


@pytest.mark.parametrize("key", [None, "", "CHANGE_ME"])
def test_policy_only_or_placeholder_override_does_not_configure_provider(catalog, key):
    client, _parser = catalog
    _deepseek_override(api_key=key)
    response = client.get("/api/v1/llm/providers")
    assert not any(
        entry["name"] == "deepseek" and entry["provider_enabled"] for entry in response.json().get("providers", [])
    )


@pytest.mark.parametrize("peer_key_source", ["static", "override"])
def test_invalid_override_does_not_fall_back_to_static_key_or_hide_healthy_peer(catalog, peer_key_source):
    client, parser = catalog
    parser.set("API", "deepseek_api_key", "static-test-key")
    _deepseek_override(credentials_invalid=True)
    if peer_key_source == "static":
        parser.set("API", "openai_api_key", "healthy-peer-test-key")
    else:
        overrides = llm_provider_overrides.get_llm_provider_overrides_snapshot()
        overrides["openai"] = LLMProviderOverride(
            provider="openai",
            api_key="healthy-peer-test-key",
            is_enabled=True,
        )
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(overrides)
    response = client.get("/api/v1/llm/providers")
    assert response.status_code == 200
    providers = {entry["name"]: entry for entry in response.json()["providers"]}
    assert providers["deepseek"]["is_configured"] is False
    assert providers["deepseek"]["provider_enabled"] is False
    assert providers["deepseek"]["availability"] == "not-configured"
    assert providers["openai"]["is_configured"] is True
    assert providers["openai"]["provider_enabled"] is True
    assert providers["openai"]["availability"] == "enabled"
    for private_value in ("static-test-key", "catalog-private-test-key", "healthy-peer-test-key"):
        assert private_value not in response.text


def test_override_key_does_not_configure_local_provider_without_endpoint(catalog):
    client, _parser = catalog
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {
            "ollama": LLMProviderOverride(
                provider="ollama",
                api_key="local-test-key",
                is_enabled=True,
                allowed_models=["local-model"],
            ),
        }
    )
    response = client.get("/api/v1/llm/providers")
    entry = next(item for item in response.json()["providers"] if item["name"] == "ollama")
    assert entry["is_configured"] is False
    assert entry["provider_enabled"] is False
    assert entry["availability"] == "not-configured"


def test_policy_only_override_preserves_static_key_readiness(catalog):
    client, parser = catalog
    parser.set("API", "deepseek_api_key", "static-test-key")
    _deepseek_override(api_key=None)
    response = client.get("/api/v1/llm/providers")
    entry = next(item for item in response.json()["providers"] if item["name"] == "deepseek")
    assert entry["is_configured"] is True
    assert entry["provider_enabled"] is True
    assert "static-test-key" not in response.text


def test_unhealthy_override_store_cannot_advertise_static_credentials(catalog, monkeypatch):
    _client, parser = catalog
    parser.set("API", "deepseek_api_key", "static-test-key")
    _deepseek_override()
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        llm_provider_overrides.get_llm_provider_overrides_snapshot(),
        healthy=False,
    )
    monkeypatch.setattr(llm_provider_overrides, "_schedule_override_recovery", lambda: None)
    result = llm_providers.get_configured_providers()
    assert result["providers"] == []
    assert result["total_configured"] == 0
    assert "catalog-private-test-key" not in str(result)


def test_override_store_health_loss_during_fallback_still_fails_catalog_closed(catalog, monkeypatch):
    _client, parser = catalog
    parser.set("API", "openai_api_key", "healthy-peer-test-key")
    _deepseek_override()
    snapshot_type = llm_provider_overrides.ProviderOverrideCallSnapshot
    original_fallback = snapshot_type.server_fallback

    def unhealthy_fallback(snapshot, base_fallback=None):
        with llm_provider_overrides._OVERRIDE_LOCK:
            llm_provider_overrides._OVERRIDE_CACHE_HEALTHY = False
        return original_fallback(snapshot, base_fallback)

    monkeypatch.setattr(snapshot_type, "server_fallback", unhealthy_fallback)
    result = llm_providers.get_configured_providers()
    assert result["providers"] == []
    assert result["total_configured"] == 0


@pytest.mark.parametrize(
    "allowed,enabled,reason",
    [
        (True, True, None),
        (False, False, "egress_blocked"),
    ],
)
def test_local_endpoint_readiness_is_not_replaced_by_override_key(
    catalog,
    monkeypatch,
    allowed,
    enabled,
    reason,
):
    client, parser = catalog
    parser.set("Local-API", "ollama_api_IP", "http://127.0.0.1:11434")
    parser.set("Local-API", "ollama_model", "local-model")
    monkeypatch.setattr(
        llm_providers,
        "evaluate_url_policy",
        lambda *_args, **_kwargs: URLPolicyResult(allowed, "test endpoint policy", reason_code="private_address"),
    )
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {
            "ollama": LLMProviderOverride(
                provider="ollama",
                api_key="local-test-key",
                is_enabled=True,
                allowed_models=["local-model"],
            ),
        }
    )
    response = client.get("/api/v1/llm/providers")
    entry = next(item for item in response.json()["providers"] if item["name"] == "ollama")
    assert entry["is_configured"] is True
    assert entry["provider_enabled"] is enabled
    assert entry["readiness_reason_code"] == reason

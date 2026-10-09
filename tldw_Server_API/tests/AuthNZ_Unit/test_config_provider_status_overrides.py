"""Public setup status must honor server-wide overrides without exposing keys."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import config_info
from tldw_Server_API.app.core.AuthNZ import byok_runtime, llm_provider_overrides
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import LLMProviderOverride


@pytest.fixture
def provider_status(monkeypatch):
    with llm_provider_overrides._OVERRIDE_LOCK:
        original = dict(llm_provider_overrides._OVERRIDE_CACHE)
        healthy = llm_provider_overrides._OVERRIDE_CACHE_HEALTHY
        ttl_enabled = not llm_provider_overrides._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})
    static_keys = {}
    monkeypatch.setattr(config_info, "_resolve_provider_key", static_keys.get)

    def no_external_calls(*_args, **_kwargs):
        pytest.fail("Public provider status attempted discovery or principal resolution")

    monkeypatch.setattr(byok_runtime, "resolve_byok_credentials", no_external_calls)
    monkeypatch.setattr(llm_provider_overrides, "_schedule_override_recovery", no_external_calls)
    app = FastAPI()
    app.include_router(config_info.router, prefix="/api/v1")
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, static_keys
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
            original,
            healthy=healthy,
            ttl_enabled=ttl_enabled,
        )


def _set_override(**changes):
    values = {
        "provider": "deepseek",
        "api_key": "encrypted-status-private-value",
        "is_enabled": True,
    }
    values.update(changes)
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {"deepseek": LLMProviderOverride(**values)},
    )


def _status(client):
    response = client.get("/api/v1/config/providers")
    assert response.status_code == 200
    payload = response.json()
    return response, payload, {entry["name"]: entry for entry in payload["providers"]}


def test_enabled_encrypted_override_configures_public_status_without_hint(provider_status):
    client, _keys = provider_status
    _set_override()
    response, payload, providers = _status(client)
    assert providers["deepseek"]["configured"] is True
    assert payload["any_configured"] is True
    assert providers["deepseek"]["key_hint"] is None
    assert providers["deepseek"]["key_source"] is None
    for private_value in ("encrypted-status-private-value", "enc...alue"):
        assert private_value not in response.text


@pytest.mark.parametrize("with_static_key", [False, True])
def test_disabled_override_does_not_advertise_cloud_readiness(provider_status, with_static_key):
    client, keys = provider_status
    if with_static_key:
        keys["deepseek"] = "static-disabled-private-value"
    _set_override(is_enabled=False)
    response, payload, providers = _status(client)
    assert providers["deepseek"]["configured"] is False
    assert payload["any_configured"] is False
    assert providers["deepseek"]["key_hint"] is None
    assert providers["deepseek"]["key_source"] is None
    assert "static-disabled-private-value" not in response.text


@pytest.mark.parametrize("peer_source", ["static", "override"])
def test_invalid_override_isolated_without_static_fallback_or_key_hint(provider_status, peer_source):
    client, keys = provider_status
    keys["deepseek"] = "static-invalid-private-value"
    _set_override(credentials_invalid=True)
    if peer_source == "static":
        keys["openai"] = "healthy-peer-private-value"
    else:
        overrides = llm_provider_overrides.get_llm_provider_overrides_snapshot()
        overrides["openai"] = LLMProviderOverride(
            provider="openai",
            api_key="healthy-peer-private-value",
            is_enabled=True,
        )
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(overrides)
    response, payload, providers = _status(client)
    assert providers["deepseek"]["configured"] is False
    assert providers["deepseek"]["key_hint"] is None
    assert providers["deepseek"]["key_source"] is None
    assert providers["openai"]["configured"] is True
    assert payload["any_configured"] is True
    if peer_source == "override":
        assert providers["openai"]["key_hint"] is None
        assert providers["openai"]["key_source"] is None
    for private_value in (
        "static-invalid-private-value",
        "encrypted-status-private-value",
        "healthy-peer-private-value",
    ):
        assert private_value not in response.text


def test_static_and_local_status_contract_unchanged(provider_status, monkeypatch):
    client, keys = provider_status
    keys["openai"] = "sk-static-test-12345678"
    monkeypatch.setenv("OPENAI_API_KEY", keys["openai"])
    _response, payload, providers = _status(client)
    assert providers["openai"]["configured"] is True
    assert providers["openai"]["key_hint"] == "sk-...5678"
    assert providers["openai"]["key_source"] == "env"
    assert payload["any_configured"] is True
    for name in ("ollama", "custom-openai-api", "custom-openai-api-99"):
        assert providers[name]["configured"] is True
        assert providers[name]["requires_api_key"] is False
        assert providers[name]["key_hint"] is None


def test_policy_only_override_preserves_static_readiness(provider_status):
    client, keys = provider_status
    keys["deepseek"] = "static-policy-test-key"
    _set_override(api_key=None)
    _response, _payload, providers = _status(client)
    assert providers["deepseek"]["configured"] is True
    assert providers["deepseek"]["key_hint"] is None
    assert providers["deepseek"]["key_source"] is None


def test_unhealthy_override_cache_cannot_advertise_static_readiness(provider_status, monkeypatch):
    client, keys = provider_status
    keys["deepseek"] = "static-unhealthy-private-value"
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({}, healthy=False)
    monkeypatch.setattr(llm_provider_overrides, "_schedule_override_recovery", lambda: None)
    response = client.get("/api/v1/config/providers")
    assert response.status_code == 503
    assert "static-unhealthy-private-value" not in response.text


def test_store_health_loss_after_capture_fails_closed(provider_status, monkeypatch):
    client, keys = provider_status
    keys["deepseek"] = "static-unhealthy-private-value"
    capture = config_info.capture_provider_override_call_snapshot

    def lose_health(provider):
        snapshot = capture(provider)
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({}, healthy=False)
        return snapshot

    monkeypatch.setattr(config_info, "capture_provider_override_call_snapshot", lose_health)
    response = client.get("/api/v1/config/providers")
    assert response.status_code == 503
    assert "static-unhealthy-private-value" not in response.text


@pytest.mark.parametrize("key", ["CHANGE_ME", ""])
def test_placeholder_or_empty_override_cannot_fall_back_to_static_key(provider_status, key):
    client, keys = provider_status
    keys["deepseek"] = "static-invalid-private-value"
    _set_override(api_key=key)
    _response, payload, providers = _status(client)
    assert providers["deepseek"]["configured"] is False
    assert payload["any_configured"] is False
    assert providers["deepseek"]["key_hint"] is None
    assert providers["deepseek"]["key_source"] is None

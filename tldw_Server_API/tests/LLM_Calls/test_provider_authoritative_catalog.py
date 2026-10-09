"""Current cloud IDs cannot be resurrected by historical/configured inventories."""

from configparser import ConfigParser
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_Server_API.app.api.v1.endpoints import llm_providers as catalog
from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as overrides
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.LLM_Calls.provider_readiness import ModelDiscoveryResult


@pytest.fixture
def inventory(monkeypatch, healthy_no_override_tts_credential_snapshot):
    parser = ConfigParser()
    parser.read_dict({"API": {
        "anthropic_api_key": "synthetic-key",
        "anthropic_model": "claude-3-retired",
    }})
    monkeypatch.setattr(catalog, "load_comprehensive_config", lambda: parser)
    monkeypatch.setattr(catalog, "get_api_keys", lambda: {})
    monkeypatch.setattr(catalog, "get_provider_manager", lambda: None)
    monkeypatch.setattr(catalog, "list_provider_models", lambda _: ["claude-3-retired"], raising=False)
    monkeypatch.setattr(catalog, "list_image_models_for_catalog", lambda: [])
    monkeypatch.setattr(catalog, "_llm_registry_capability_envelopes", lambda: {})
    monkeypatch.setattr(catalog, "_resolve_model_tokenizer_support", lambda *args: {})
    monkeypatch.setattr(catalog, "get_llm_provider_overrides_snapshot", lambda: {}, raising=False)
    monkeypatch.setattr(catalog, "resolve_static_server_fallback", lambda _: SimpleNamespace(app_config={}), raising=False)
    probe = Mock(return_value=ModelDiscoveryResult("ready", ("claude-current",)))
    monkeypatch.setattr(catalog, "discover_provider_models", probe, raising=False)
    return probe


def _anthropic():
    return next(p for p in catalog.get_configured_providers()["providers"]
                if p["name"] == "anthropic")


@pytest.mark.unit
def test_catalog_uses_live_ids_not_prices_or_defaults(inventory):
    provider = _anthropic()
    assert provider["models"] == ["claude-current"]
    assert provider["default_model"] == "claude-current"
    assert provider["model_inventory_source"] == "provider"
    assert [m["name"] for m in provider["models_info"]] == ["claude-current"]
    inventory.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("status", ["unreachable", "auth_failed", "unsupported", "server_error"])
def test_failed_inventory_never_falls_back(inventory, status):
    inventory.return_value = ModelDiscoveryResult(status)
    provider = _anthropic()
    assert provider["models"] == []
    assert provider["default_model"] is None
    assert provider["provider_enabled"] is False
    assert provider["availability"] == "unavailable"
    assert provider["readiness_reason_code"]


@pytest.mark.unit
@pytest.mark.parametrize("live", [[], ["claude-current"]])
def test_override_lists_cannot_resurrect_retired_ids(live):
    override = overrides.LLMProviderOverride(
        provider="anthropic", config={
            "models": ["claude-3-retired", "claude-current"],
            "default_model": "claude-3-retired",
        }, allowed_models=["claude-3-retired", "claude-current"],
    )
    payload = {"providers": [{
        "name": "anthropic", "type": "commercial",
        "model_inventory_source": "provider", "models": live,
        "default_model": live[0] if live else None,
        "models_info": [{"name": model} for model in live],
    }]}
    provider = overrides.apply_llm_provider_overrides_to_listing(
        payload, overrides={"anthropic": override},
    )["providers"][0]
    assert provider["models"] == live
    assert provider["default_model"] == (live[0] if live else None)


@pytest.mark.unit
def test_catalog_uses_encrypted_override_key(inventory, monkeypatch):
    override = overrides.LLMProviderOverride(provider="anthropic", api_key="override-key")
    monkeypatch.setattr(catalog, "get_llm_provider_overrides_snapshot",
                        lambda: {"anthropic": override}, raising=False)
    assert _anthropic()["models"] == ["claude-current"]
    assert inventory.call_args.args[1] == "override-key"


@pytest.mark.unit
def test_inventory_honors_runtime_key_precedence(inventory, monkeypatch):
    monkeypatch.setattr(catalog, "get_api_keys", lambda: {"anthropic": "runtime-key"})
    _anthropic()
    assert inventory.call_args.args[1] == "runtime-key"


@pytest.mark.unit
@pytest.mark.parametrize("disabled,invalid", [(True, False), (False, True)])
def test_disabled_or_invalid_override_never_discovers(inventory, monkeypatch, disabled, invalid):
    override = overrides.LLMProviderOverride(provider="anthropic", api_key="override-key",
                                             is_enabled=not disabled, credentials_invalid=invalid)
    monkeypatch.setattr(catalog, "get_llm_provider_overrides_snapshot",
                        lambda: {"anthropic": override}, raising=False)
    provider = _anthropic()
    assert provider["models"] == []
    assert provider["provider_enabled"] is False
    if invalid:
        assert provider["is_configured"] is False
    if disabled:
        assert provider["availability"] == "disabled"
    inventory.assert_not_called()


@pytest.mark.unit
def test_invalid_resolved_endpoint_never_reverts_to_global_default(inventory, monkeypatch):
    monkeypatch.setattr(catalog, "resolve_provider_models_base_url", lambda *_args, **_kwargs: None)
    assert _anthropic()["models"] == []
    inventory.assert_not_called()


@pytest.mark.unit
def test_empty_authoritative_inventory_is_unavailable(inventory):
    inventory.return_value = ModelDiscoveryResult("ready")
    provider = _anthropic()
    assert provider["models"] == []
    assert provider["provider_enabled"] is False
    assert provider["readiness_reason_code"] == "no_models_reported"


@pytest.mark.unit
def test_unhealthy_credential_store_never_reverts_to_plaintext(inventory, monkeypatch):
    def unavailable():
        raise ByokResolutionError("credential_store_unavailable", "provider-overrides")

    monkeypatch.setattr(catalog, "get_llm_provider_overrides_snapshot", unavailable)
    result = catalog.get_configured_providers()
    assert result["providers"] == []
    assert result["total_configured"] == 0
    assert result["error"] == "Provider credential store is unavailable."
    inventory.assert_not_called()


@pytest.mark.unit
def test_credential_store_health_loss_during_resolution_fails_entire_catalog(inventory, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise ByokResolutionError("credential_store_unavailable", "provider-overrides")

    monkeypatch.setattr(overrides.ProviderOverrideCallSnapshot, "server_fallback", unavailable)
    result = catalog.get_configured_providers()
    assert result["providers"] == []
    assert result["total_configured"] == 0
    inventory.assert_not_called()

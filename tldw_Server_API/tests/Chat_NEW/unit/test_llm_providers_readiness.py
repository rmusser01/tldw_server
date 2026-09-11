"""Tests for LLM provider readiness metadata surfaced to clients."""

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import llm_providers
from tldw_Server_API.app.core import config
from tldw_Server_API.app.core.config_paths import resolve_config_file, resolve_config_root
from tldw_Server_API.app.core.LLM_Calls.provider_readiness import (
    ModelDiscoveryResult,
    provider_readiness,
)
from tldw_Server_API.app.core.Security.egress import URLPolicyResult


class _EmptyProviderManager:
    """Provider manager stub that reports no provider health state."""

    def get_health_report(self) -> dict[str, object]:
        """Return an empty health report for readiness tests."""
        return {}


def _config(sections: dict[str, dict[str, str]]) -> ConfigParser:
    """Build a minimal ConfigParser from section dictionaries."""
    parser = ConfigParser()
    for section, values in sections.items():
        parser.add_section(section)
        for key, value in values.items():
            parser.set(section, key, value)
    return parser


def _client_for_config(monkeypatch: pytest.MonkeyPatch, parser: ConfigParser) -> TestClient:
    """Create a FastAPI test client with provider dependencies patched."""
    async def _configured_providers(*args, **kwargs):
        return await llm_providers.get_configured_providers_async(*args, **kwargs)

    monkeypatch.setattr(llm_providers, "load_comprehensive_config", lambda: parser)
    monkeypatch.setattr(llm_providers, "get_api_keys", lambda: {})
    monkeypatch.setattr(llm_providers, "get_provider_manager", lambda: _EmptyProviderManager())
    monkeypatch.setattr(llm_providers, "list_provider_models", lambda _provider: [])
    monkeypatch.setattr(llm_providers, "list_image_models_for_catalog", lambda: [])
    monkeypatch.setattr(llm_providers, "discover_models_from_endpoint", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(llm_providers, "apply_llm_provider_overrides_to_listing", lambda result: result)

    app = FastAPI()
    app.include_router(llm_providers.router, prefix="/api/v1")
    app.state.llm_manager = SimpleNamespace(llamacpp_supervisor=None)
    return TestClient(app)


def _provider(data: dict[str, object], name: str) -> dict[str, object]:
    """Return one provider entry from a providers response payload."""
    providers = data.get("providers")
    assert isinstance(providers, list)
    found = next((item for item in providers if item.get("name") == name), None)
    assert found is not None
    return found


def _model(data: dict[str, object], provider: str, name: str) -> dict[str, object]:
    """Return one model entry from a model metadata response payload."""
    models = data.get("models")
    assert isinstance(models, list)
    found = next(
        (
            item
            for item in models
            if item.get("provider") == provider
            and (item.get("name") == name or item.get("id") == name)
        ),
        None,
    )
    assert found is not None
    return found


@pytest.mark.unit
@pytest.mark.parametrize(
    ("result", "has_explicit_models", "probe", "enabled", "reason"),
    [
        (ModelDiscoveryResult("ready", ("discovered",)), False, False, True, None),
        (ModelDiscoveryResult("ready", ()), False, False, True, "no_models_reported"),
        (ModelDiscoveryResult("auth_failed", ()), False, False, False, "auth_failed"),
        (ModelDiscoveryResult("server_error", ()), False, False, False, "endpoint_error"),
        (ModelDiscoveryResult("unsupported", ()), False, False, True, "model_discovery_unavailable"),
        (ModelDiscoveryResult("unreachable", ()), False, False, False, "endpoint_unreachable"),
        (None, True, False, True, None),
        (ModelDiscoveryResult("ready", ()), True, True, True, None),
        (ModelDiscoveryResult("auth_failed", ()), True, True, False, "auth_failed"),
        (ModelDiscoveryResult("server_error", ()), True, True, False, "endpoint_error"),
        (ModelDiscoveryResult("unsupported", ()), True, True, True, "model_discovery_unavailable"),
        (ModelDiscoveryResult("unreachable", ()), True, True, False, "endpoint_unreachable"),
    ],
)
def test_provider_readiness_reduces_precomputed_discovery_without_io(
    result,
    has_explicit_models,
    probe,
    enabled,
    reason,
):
    readiness = provider_readiness(
        provider_name="llama",
        provider_info={"display_name": "Llama.cpp", "type": "local"},
        is_configured=True,
        endpoint_url="http://10.0.0.5:18080/v1",
        api_key_value=None,
        current_availability=None,
        health_entry=None,
        supported_chat_providers={"llama.cpp"},
        endpoint_policy=URLPolicyResult(True),
        discovery_result=result,
        has_explicit_models=has_explicit_models,
        endpoint_probe_enabled=probe,
    )

    assert readiness["provider_enabled"] is enabled
    assert readiness["readiness_reason_code"] == reason


@pytest.mark.unit
def test_provider_readiness_policy_and_health_failures_override_models():
    common = {
        "provider_name": "llama",
        "provider_info": {"display_name": "Llama.cpp", "type": "local"},
        "is_configured": True,
        "endpoint_url": "http://10.0.0.5:18080/v1",
        "api_key_value": None,
        "current_availability": None,
        "supported_chat_providers": {"llama.cpp"},
        "discovery_result": ModelDiscoveryResult("ready", ("manual",)),
        "has_explicit_models": True,
        "endpoint_probe_enabled": False,
    }

    policy_blocked = provider_readiness(
        **common,
        endpoint_policy=URLPolicyResult(False, "blocked", reason_code="address_forbidden"),
        health_entry=None,
    )
    unhealthy = provider_readiness(
        **common,
        endpoint_policy=URLPolicyResult(True),
        health_entry={"status": "unhealthy"},
    )

    assert policy_blocked["readiness_reason_code"] == "egress_blocked"
    assert policy_blocked["provider_enabled"] is False
    assert unhealthy["readiness_reason_code"] == "provider_health_unavailable"
    assert unhealthy["provider_enabled"] is False


@pytest.mark.unit
def test_llama_manual_model_on_scoped_lan_nonstandard_port_is_selectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trusted LAN endpoints bypass global private/port defaults only for their exact origin."""
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWED_PORTS", "80,443")
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    parser = _config(
        {
            "Local-API": {
                "llama_api_IP": "http://192.168.2.216:18080/v1",
                "llama_model": "manual-llama.gguf",
            }
        }
    )

    with _client_for_config(monkeypatch, parser) as client:
        providers_response = client.get("/api/v1/llm/providers")
        models_response = client.get("/api/v1/llm/models/metadata")

    assert providers_response.status_code == 200, providers_response.text
    llama = _provider(providers_response.json(), "llama")
    assert llama["availability"] == "enabled"
    assert llama["provider_enabled"] is True
    assert llama["readiness_reason_code"] is None
    assert llama["chat_provider"] == "llama.cpp"

    assert models_response.status_code == 200, models_response.text
    model = _model(models_response.json(), "llama", "manual-llama.gguf")
    assert model["availability"] == "enabled"
    assert model["provider_enabled"] is True
    assert model.get("readiness_reason_code") is None
    assert model["catalog_only"] is False


@pytest.mark.unit
def test_llama_manual_model_on_metadata_target_is_egress_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _config(
        {
            "Local-API": {
                "llama_api_IP": "http://169.254.169.254:18080/v1",
                "llama_model": "manual-llama.gguf",
            }
        }
    )

    with _client_for_config(monkeypatch, parser) as client:
        models_response = client.get("/api/v1/llm/models/metadata")

    assert models_response.status_code == 200, models_response.text
    model = _model(models_response.json(), "llama", "manual-llama.gguf")
    assert model["availability"] == "unavailable"
    assert model["provider_enabled"] is False
    assert model["readiness_reason_code"] == "egress_blocked"


@pytest.mark.unit
def test_catalog_computes_discovery_once_and_passes_same_result_to_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _config(
        {"Local-API": {"llama_api_IP": "http://10.0.0.5:18080/v1"}}
    )
    result = ModelDiscoveryResult("ready", ("discovered-llama",))
    discovery_calls = []
    readiness_results = []
    original_readiness = provider_readiness

    def discover(*args, **kwargs):
        discovery_calls.append((args, kwargs))
        return result

    def reduce_readiness(**kwargs):
        if kwargs["provider_name"] == "llama":
            readiness_results.append(kwargs["discovery_result"])
        return original_readiness(**kwargs)

    client = _client_for_config(monkeypatch, parser)
    monkeypatch.setattr(llm_providers, "discover_models_from_endpoint", discover)
    monkeypatch.setattr(llm_providers, "_provider_readiness", reduce_readiness)
    with client:
        response = client.get("/api/v1/llm/providers")

    assert response.status_code == 200, response.text
    assert len(discovery_calls) == 1
    assert readiness_results == [result]
    assert readiness_results[0] is result
    assert _provider(response.json(), "llama")["models"] == ["discovered-llama"]


@pytest.mark.unit
def test_catalog_maps_dns_unresolved_to_endpoint_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _config(
        {
            "Local-API": {
                "llama_api_IP": "http://llama.internal:18080/v1",
                "llama_model": "manual-llama.gguf",
            }
        }
    )
    client = _client_for_config(monkeypatch, parser)
    monkeypatch.setattr(
        llm_providers,
        "evaluate_url_policy",
        lambda *_args, **_kwargs: URLPolicyResult(
            False,
            "Host could not be resolved",
            reason_code="dns_unresolved",
        ),
    )

    with client:
        response = client.get("/api/v1/llm/providers")

    assert response.status_code == 200, response.text
    llama = _provider(response.json(), "llama")
    assert llama["provider_enabled"] is False
    assert llama["availability"] == "unavailable"
    assert llama["readiness_reason_code"] == "endpoint_unreachable"
    assert "llama.internal" not in llama["readiness_message"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("result", "reason_code"),
    [
        (ModelDiscoveryResult("ready", ()), "no_models_reported"),
        (ModelDiscoveryResult("unsupported", ()), "model_discovery_unavailable"),
    ],
)
def test_catalog_keeps_empty_discovery_diagnostics_enabled(
    monkeypatch: pytest.MonkeyPatch,
    result: ModelDiscoveryResult,
    reason_code: str,
) -> None:
    parser = _config(
        {"Local-API": {"llama_api_IP": "http://10.0.0.5:18080/v1"}}
    )
    client = _client_for_config(monkeypatch, parser)
    monkeypatch.setattr(
        llm_providers,
        "discover_models_from_endpoint",
        lambda *_args, **_kwargs: result,
    )

    with client:
        response = client.get("/api/v1/llm/providers")

    assert response.status_code == 200, response.text
    llama = _provider(response.json(), "llama")
    assert llama["provider_enabled"] is True
    assert llama["availability"] == "enabled"
    assert llama["models"] == []
    assert llama["endpoint_only"] is True
    assert llama["readiness_reason_code"] == reason_code


@pytest.mark.unit
def test_catalog_requested_discovery_merges_after_explicit_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER_READINESS_PROBE_ENDPOINTS", "1")
    parser = _config(
        {
            "Local-API": {
                "llama_api_IP": "http://10.0.0.5:18080/v1",
                "llama_model": "manual-llama,shared-model",
            }
        }
    )
    discovery_calls = []
    client = _client_for_config(monkeypatch, parser)
    monkeypatch.setattr(
        llm_providers,
        "_resolve_model_tokenizer_support",
        lambda *_args, **_kwargs: {
            "available": False,
            "strict_mode_effective": False,
        },
    )

    def discover(*args, **kwargs):
        discovery_calls.append((args, kwargs))
        return ModelDiscoveryResult(
            "ready",
            ("shared-model", "discovered-model", "manual-llama"),
        )

    monkeypatch.setattr(llm_providers, "discover_models_from_endpoint", discover)

    with client:
        response = client.get("/api/v1/llm/providers")

    assert response.status_code == 200, response.text
    llama = _provider(response.json(), "llama")
    assert len(discovery_calls) == 1
    assert llama["models"] == ["manual-llama", "shared-model", "discovered-model"]
    assert llama["provider_enabled"] is True


@pytest.mark.unit
def test_llm_provider_readiness_marks_unreachable_local_endpoint_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in endpoint probes mark unreachable local endpoints unavailable."""
    monkeypatch.setenv("LLM_PROVIDER_READINESS_PROBE_ENDPOINTS", "1")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWED_PORTS", "*")
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "false")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWLIST", "127.0.0.1,localhost")
    parser = _config(
        {
            "Local-API": {
                "vllm_api_IP": "http://127.0.0.1:18080/v1",
                "vllm_model": "local-model",
            }
        }
    )
    monkeypatch.setattr(
        llm_providers,
        "discover_models_from_endpoint",
        lambda *_args, **_kwargs: [],
    )

    with _client_for_config(monkeypatch, parser) as client:
        response = client.get("/api/v1/llm/providers")

    assert response.status_code == 200, response.text
    vllm = _provider(response.json(), "vllm")
    assert vllm["availability"] == "unavailable"
    assert vllm["provider_enabled"] is False
    assert vllm["readiness_reason_code"] == "endpoint_unreachable"
    assert "could not be reached" in vllm["readiness_message"]


@pytest.mark.unit
def test_llm_provider_readiness_marks_external_custom_openai_without_key_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External custom OpenAI-compatible endpoints require credentials."""
    parser = _config(
        {
            "API": {
                "custom_openai_api_ip": "https://api.openai.com/v1",
                "custom_openai_api_model": "gpt-4.1-2025-04-14",
            }
        }
    )

    with _client_for_config(monkeypatch, parser) as client:
        response = client.get("/api/v1/llm/providers")

    assert response.status_code == 200, response.text
    provider = _provider(response.json(), "custom_openai_api")
    assert provider["availability"] == "not-configured"
    assert provider["provider_enabled"] is False
    assert provider["readiness_reason_code"] == "missing_credentials"
    assert "requires credentials" in provider["readiness_message"]
    assert provider["chat_provider"] == "custom-openai-api"


@pytest.mark.unit
def test_custom_openai_catalog_uses_env_endpoint_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env-configured local custom OpenAI endpoints are selectable in model metadata."""
    monkeypatch.setenv("CUSTOM_OPENAI_API_URL", "http://127.0.0.1:9099/v1")
    monkeypatch.setenv("CUSTOM_OPENAI_API_MODEL", "local-gemma.gguf")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWED_PORTS", "*")
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "false")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWLIST", "127.0.0.1,localhost")
    parser = _config(
        {
            "API": {
                "custom_openai_api_ip": "https://api.openai.com/v1",
                "custom_openai_api_model": "gpt-4.1-2025-04-14",
            }
        }
    )

    with _client_for_config(monkeypatch, parser) as client:
        providers_response = client.get("/api/v1/llm/providers")
        models_response = client.get("/api/v1/llm/models/metadata")

    assert providers_response.status_code == 200, providers_response.text
    provider = _provider(providers_response.json(), "custom_openai_api")
    assert provider["endpoint"] == "http://127.0.0.1:9099/v1"
    assert provider["models"] == ["local-gemma.gguf"]
    assert provider["is_configured"] is True
    assert provider["provider_enabled"] is True
    assert provider["availability"] == "enabled"
    assert provider["readiness_reason_code"] is None
    assert provider["chat_provider"] == "custom-openai-api"
    assert provider["requires_api_key"] is False

    assert models_response.status_code == 200, models_response.text
    model = _model(models_response.json(), "custom_openai_api", "local-gemma.gguf")
    assert model["is_configured"] is True
    assert model["provider_is_configured"] is True
    assert model["provider_enabled"] is True
    assert model["catalog_only"] is False
    assert model["availability"] == "enabled"


@pytest.mark.unit
def test_custom_openai_catalog_uses_env_endpoint_and_model_without_api_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env-only custom OpenAI catalog config is usable without an [API] section."""
    monkeypatch.setenv("CUSTOM_OPENAI_API_URL", "http://127.0.0.1:9099/v1")
    monkeypatch.setenv("CUSTOM_OPENAI_API_MODEL", "local-gemma.gguf")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWED_PORTS", "*")
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "false")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWLIST", "127.0.0.1,localhost")
    parser = _config({})

    with _client_for_config(monkeypatch, parser) as client:
        providers_response = client.get("/api/v1/llm/providers")
        models_response = client.get("/api/v1/llm/models/metadata")

    assert providers_response.status_code == 200, providers_response.text
    provider = _provider(providers_response.json(), "custom_openai_api")
    assert provider["endpoint"] == "http://127.0.0.1:9099/v1"
    assert provider["models"] == ["local-gemma.gguf"]
    assert provider["is_configured"] is True
    assert provider["provider_enabled"] is True
    assert provider["availability"] == "enabled"
    assert provider["requires_api_key"] is False

    assert models_response.status_code == 200, models_response.text
    model = _model(models_response.json(), "custom_openai_api", "local-gemma.gguf")
    assert model["is_configured"] is True
    assert model["provider_enabled"] is True
    assert model["availability"] == "enabled"


@pytest.mark.unit
def test_critical_e2e_fixture_discovers_only_the_env_custom_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The critical fixture keeps legacy local-provider probes out of E2E startup."""
    fixture_path = (
        Path(__file__).resolve().parents[3]
        / "Config_Files"
        / "e2e-critical-config.txt"
    )
    mock_endpoint = "http://127.0.0.1:18091/v1"
    monkeypatch.setenv("TLDW_CONFIG_FILE", str(fixture_path))
    monkeypatch.setenv("CUSTOM_OPENAI_API_IP", mock_endpoint)
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "sk-uat-mock-openai")
    monkeypatch.setenv("CUSTOM_OPENAI_API_MODEL", "local-uat-chat")
    monkeypatch.setenv("LLM_PROVIDER_READINESS_PROBE_ENDPOINTS", "1")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWED_PORTS", "*")
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "false")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWLIST", "127.0.0.1,localhost")
    config.clear_config_cache()

    assert fixture_path.is_file()
    assert resolve_config_file() == fixture_path
    assert resolve_config_root() == fixture_path.parent
    parser = config.load_comprehensive_config()
    discovery_calls: list[tuple[str, str]] = []

    def discover(provider_name: str, endpoint_url: str, *_args, **_kwargs):
        discovery_calls.append((provider_name, endpoint_url))
        return ModelDiscoveryResult("ready", ("local-uat-chat",))

    try:
        with _client_for_config(monkeypatch, parser) as client:
            monkeypatch.setattr(llm_providers, "discover_models_from_endpoint", discover)
            response = client.get("/api/v1/llm/models/metadata?type=chat&output_modality=text")
    finally:
        config.clear_config_cache()

    assert response.status_code == 200, response.text
    assert discovery_calls == [("custom_openai_api", mock_endpoint)]
    model = _model(response.json(), "custom_openai_api", "local-uat-chat")
    assert model["provider_enabled"] is True
    assert model["catalog_only"] is False
    assert model["availability"] == "enabled"


@pytest.mark.unit
def test_external_custom_openai_env_placeholder_key_is_missing_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known placeholder API keys do not satisfy external custom OpenAI readiness."""
    monkeypatch.setenv("CUSTOM_OPENAI_API_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("CUSTOM_OPENAI_API_MODEL", "gpt-4.1-2025-04-14")
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "CHANGE_ME_TO_SECURE_API_KEY")
    parser = _config({})

    with _client_for_config(monkeypatch, parser) as client:
        providers_response = client.get("/api/v1/llm/providers")
        models_response = client.get("/api/v1/llm/models/metadata")

    assert providers_response.status_code == 200, providers_response.text
    provider = _provider(providers_response.json(), "custom_openai_api")
    assert provider["is_configured"] is False
    assert provider["provider_enabled"] is False
    assert provider["availability"] == "not-configured"
    assert provider["readiness_reason_code"] == "missing_credentials"
    assert "requires credentials" in provider["readiness_message"]

    assert models_response.status_code == 200, models_response.text
    model = _model(models_response.json(), "custom_openai_api", "gpt-4.1-2025-04-14")
    assert model["is_configured"] is False
    assert model["provider_enabled"] is False
    assert model["availability"] == "not-configured"


@pytest.mark.unit
def test_llm_provider_readiness_marks_unsupported_catalog_alias_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catalog providers unsupported by chat completions are unavailable."""
    parser = _config(
        {
            "API": {},
            "Local-API": {},
            "MLX": {
                "mlx_model_path": "Qwen/Qwen3-0.6B-MLX-4bit",
            }
        }
    )

    with _client_for_config(monkeypatch, parser) as client:
        response = client.get("/api/v1/llm/providers")

    assert response.status_code == 200, response.text
    mlx = _provider(response.json(), "mlx")
    assert mlx["availability"] == "unavailable"
    assert mlx["provider_enabled"] is False
    assert mlx["readiness_reason_code"] == "unsupported_chat_provider"
    assert "is not supported by chat completions" in mlx["readiness_message"]

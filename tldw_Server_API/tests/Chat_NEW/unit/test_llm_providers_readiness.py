"""Tests for LLM provider readiness metadata surfaced to clients."""

from __future__ import annotations

from configparser import ConfigParser
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import llm_providers


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
def test_llm_provider_readiness_marks_egress_blocked_ollama_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Egress-blocked local endpoints are reported as unavailable."""
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWED_PORTS", "80,443")
    parser = _config(
        {
            "Local-API": {
                "ollama_api_IP": "http://192.168.2.216:11434/v1",
                "ollama_model": "gemma3:1b",
            }
        }
    )

    with _client_for_config(monkeypatch, parser) as client:
        providers_response = client.get("/api/v1/llm/providers")
        models_response = client.get("/api/v1/llm/models/metadata")

    assert providers_response.status_code == 200, providers_response.text
    ollama = _provider(providers_response.json(), "ollama")
    assert ollama["availability"] == "unavailable"
    assert ollama["provider_enabled"] is False
    assert ollama["readiness_reason_code"] == "egress_blocked"
    assert "Port not allowed: 11434" in ollama["readiness_message"]
    assert ollama["chat_provider"] == "ollama"

    assert models_response.status_code == 200, models_response.text
    model = _model(models_response.json(), "ollama", "gemma3:1b")
    assert model["availability"] == "unavailable"
    assert model["provider_enabled"] is False
    assert model["readiness_reason_code"] == "egress_blocked"


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

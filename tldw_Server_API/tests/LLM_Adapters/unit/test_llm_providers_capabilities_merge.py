import configparser
import pytest


@pytest.fixture(autouse=True)
def _healthy_provider_override_snapshot():
    """Keep this lightweight router test independent of app-startup refresh."""
    from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides

    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})
    yield
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})


def _fake_config():
    cfg = configparser.ConfigParser()
    cfg.add_section("API")
    cfg.set("API", "openai_api_key", "sk-test")
    cfg.set("API", "openai_model", "gpt-4o-mini")
    cfg.set("API", "default_api", "openai")
    # Provide minimal Local-API section to satisfy callers that probe both
    cfg.add_section("Local-API")
    return cfg


def _fake_config_with_llama():
    cfg = _fake_config()
    cfg.set("Local-API", "llama_api_IP", "http://localhost:8001/v1/chat/completions")
    return cfg


def _provider_by_display_name(data: dict, display_name: str) -> dict:
    return next(provider for provider in data.get("providers", []) if provider.get("display_name") == display_name)


@pytest.fixture
def llm_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from tldw_Server_API.app.api.v1.endpoints.llm_providers import router as llm_router

    app = FastAPI()
    app.include_router(llm_router, prefix="/api/v1")
    with TestClient(app) as client:
        yield client


def test_llm_providers_merges_adapter_capabilities_from_envelope(monkeypatch, llm_client):
    # Force adapters available even though this is not strictly required for the endpoint

    # Stub configuration loader to return a controlled config
    import tldw_Server_API.app.core.config as core_config
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints

    monkeypatch.setattr(core_config, "load_comprehensive_config", _fake_config)
    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", _fake_config)

    # Stub registry capability discovery
    import tldw_Server_API.app.core.LLM_Calls.adapter_registry as reg_mod

    class _DummyReg:
        def list_capabilities(self, include_disabled=True):
            assert include_disabled is True
            return [
                {
                    "provider": "openai",
                    "availability": "enabled",
                    "capabilities": {"json_mode": True, "supports_tools": True, "extra_cap": "adapter"},
                }
            ]

    monkeypatch.setattr(reg_mod, "get_registry", lambda: _DummyReg())

    r = llm_client.get("/api/v1/llm/providers")
    assert r.status_code == 200
    data = r.json()
    providers = {p["name"]: p for p in data.get("providers", [])}
    assert "openai" in providers
    caps = providers["openai"].get("capabilities", {})
    # Adapter-provided fields should be present
    assert caps.get("json_mode") is True
    assert caps.get("supports_tools") is True
    assert caps.get("extra_cap") == "adapter"
    assert providers["openai"].get("availability") == "enabled"
    assert providers["openai"].get("capability_envelope") == {
        "provider": "openai",
        "availability": "enabled",
        "capabilities": {"json_mode": True, "supports_tools": True, "extra_cap": "adapter"},
    }


def test_public_llm_providers_never_exposes_private_override_envelope(
    monkeypatch,
    llm_client,
):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints
    import tldw_Server_API.app.core.config as core_config
    import tldw_Server_API.app.core.LLM_Calls.adapter_registry as reg_mod
    from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides

    class _DummyReg:
        def list_capabilities(self, include_disabled=True):
            assert include_disabled is True
            return []

    original = llm_provider_overrides.get_llm_provider_overrides_snapshot()
    monkeypatch.setattr(core_config, "load_comprehensive_config", _fake_config)
    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", _fake_config)
    monkeypatch.setattr(reg_mod, "get_registry", lambda: _DummyReg())
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {
            "openai": llm_provider_overrides.LLMProviderOverride(
                provider="openai",
                is_enabled=False,
                allowed_models=["gpt-4o-mini"],
                config={"private_canary": "public-hidden-config"},
                api_key="public-hidden-key",
                api_key_hint="public-hidden-hint",
                credential_fields={"org_id": "public-hidden-org"},
            )
        }
    )

    try:
        response = llm_client.get("/api/v1/llm/providers")
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(original)

    assert response.status_code == 200, response.text
    openai = next(
        provider
        for provider in response.json()["providers"]
        if provider["name"] == "openai"
    )
    assert openai["enabled"] is False
    assert openai["models"] == ["gpt-4o-mini"]
    assert "override" not in openai
    for hidden in (
        "public-hidden-config",
        "public-hidden-key",
        "public-hidden-hint",
        "public-hidden-org",
    ):
        assert hidden not in response.text


def test_llm_providers_legacy_capabilities_fallback(monkeypatch, llm_client):
    import tldw_Server_API.app.core.config as core_config
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints

    monkeypatch.setattr(core_config, "load_comprehensive_config", _fake_config)
    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", _fake_config)

    import tldw_Server_API.app.core.LLM_Calls.adapter_registry as reg_mod

    class _DummyLegacyReg:
        def get_all_capabilities(self):
            return {"openai": {"json_mode": True, "supports_tools": True}}

    monkeypatch.setattr(reg_mod, "get_registry", lambda: _DummyLegacyReg())

    r = llm_client.get("/api/v1/llm/providers")
    assert r.status_code == 200
    data = r.json()
    providers = {p["name"]: p for p in data.get("providers", [])}
    assert "openai" in providers
    caps = providers["openai"].get("capabilities", {})
    assert caps.get("json_mode") is True
    assert caps.get("supports_tools") is True
    assert providers["openai"].get("availability") == "enabled"


def test_llm_providers_includes_model_level_extra_body_compat(monkeypatch, llm_client):
    import tldw_Server_API.app.core.config as core_config
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints

    monkeypatch.setattr(core_config, "load_comprehensive_config", _fake_config)
    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", _fake_config)

    import tldw_Server_API.app.core.LLM_Calls.adapter_registry as reg_mod

    class _DummyReg:
        def list_capabilities(self, include_disabled=True):
            return []

    monkeypatch.setattr(reg_mod, "get_registry", lambda: _DummyReg())

    r = llm_client.get("/api/v1/llm/providers")
    assert r.status_code == 200
    data = r.json()
    providers = {p["name"]: p for p in data.get("providers", [])}
    assert "openai" in providers
    assert "extra_body_compat" in providers["openai"]
    assert isinstance(providers["openai"]["extra_body_compat"].get("known_params"), list)
    models_info = providers["openai"].get("models_info") or []
    assert models_info
    assert "extra_body_compat" in models_info[0]


def test_llm_providers_extra_body_compat_reflects_strict_runtime(monkeypatch, llm_client):
    import tldw_Server_API.app.core.config as core_config
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints

    monkeypatch.setattr(core_config, "load_comprehensive_config", _fake_config)
    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", _fake_config)
    monkeypatch.setenv("LOCAL_LLM_STRICT_OPENAI_COMPAT", "true")

    import tldw_Server_API.app.core.LLM_Calls.adapter_registry as reg_mod

    class _DummyReg:
        def list_capabilities(self, include_disabled=True):
            return []

    monkeypatch.setattr(reg_mod, "get_registry", lambda: _DummyReg())

    r = llm_client.get("/api/v1/llm/providers")
    assert r.status_code == 200
    data = r.json()
    providers = {p["name"]: p for p in data.get("providers", [])}
    assert providers["openai"]["extra_body_compat"]["supported"] is False
    assert "strict_openai_compat" in str(providers["openai"]["extra_body_compat"]["effective_reason"])
    first_model = providers["openai"].get("models_info", [])[0]
    assert first_model["extra_body_compat"]["supported"] is False
    assert "strict_openai_compat" in str(first_model["extra_body_compat"]["effective_reason"])


def test_llm_providers_includes_model_level_tokenizer_metadata(monkeypatch, llm_client):
    import tldw_Server_API.app.core.config as core_config
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints

    monkeypatch.setattr(core_config, "load_comprehensive_config", _fake_config)
    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", _fake_config)

    import tldw_Server_API.app.core.LLM_Calls.adapter_registry as reg_mod

    class _DummyReg:
        def list_capabilities(self, include_disabled=True):
            return []

    monkeypatch.setattr(reg_mod, "get_registry", lambda: _DummyReg())

    r = llm_client.get("/api/v1/llm/providers")
    assert r.status_code == 200
    data = r.json()
    providers = {p["name"]: p for p in data.get("providers", [])}
    openai = providers["openai"]
    models_info = openai.get("models_info") or []
    assert models_info

    selected = None
    for model_info in models_info:
        if model_info.get("name") == "gpt-4o-mini":
            selected = model_info
            break
    assert selected is not None

    assert "tokenizer_available" in selected
    assert "tokenizer" in selected
    assert "tokenizer_kind" in selected
    assert "tokenizer_source" in selected
    assert "detokenize_available" in selected

    tokenizers = openai.get("tokenizers")
    assert isinstance(tokenizers, dict)
    assert "gpt-4o-mini" in tokenizers
    assert isinstance(tokenizers["gpt-4o-mini"], dict)
    assert "available" in tokenizers["gpt-4o-mini"]
    assert "tokenizer" in tokenizers["gpt-4o-mini"]
    assert "kind" in tokenizers["gpt-4o-mini"]
    assert "source" in tokenizers["gpt-4o-mini"]
    assert "detokenize" in tokenizers["gpt-4o-mini"]


def test_llm_providers_exposes_llama_cpp_controls_block(monkeypatch, llm_client):
    import tldw_Server_API.app.core.config as core_config
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints
    import tldw_Server_API.app.core.LLM_Calls.adapter_registry as reg_mod

    monkeypatch.setattr(core_config, "load_comprehensive_config", _fake_config_with_llama)
    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", _fake_config_with_llama)

    class _DummyReg:
        def list_capabilities(self, include_disabled=True):
            return []

    monkeypatch.setattr(reg_mod, "get_registry", lambda: _DummyReg())

    response = llm_client.get("/api/v1/llm/providers")
    assert response.status_code == 200
    llama = _provider_by_display_name(response.json(), "Llama.cpp")
    controls = llama["llama_cpp_controls"]
    assert controls["grammar"]["supported"] is True
    assert "thinking_budget" in controls
    assert controls["reserved_extra_body_keys"] == ["grammar"]


def test_llm_providers_disables_thinking_budget_without_verified_mapping(monkeypatch, llm_client):
    import tldw_Server_API.app.core.config as core_config
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints
    import tldw_Server_API.app.core.LLM_Calls.adapter_registry as reg_mod

    monkeypatch.setattr(core_config, "load_comprehensive_config", _fake_config_with_llama)
    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", _fake_config_with_llama)
    monkeypatch.delenv("LLAMA_CPP_THINKING_BUDGET_PARAM", raising=False)

    class _DummyReg:
        def list_capabilities(self, include_disabled=True):
            return []

    monkeypatch.setattr(reg_mod, "get_registry", lambda: _DummyReg())

    response = llm_client.get("/api/v1/llm/providers")
    assert response.status_code == 200
    controls = _provider_by_display_name(response.json(), "Llama.cpp")["llama_cpp_controls"]
    assert controls["thinking_budget"]["supported"] is False
    assert controls["thinking_budget"]["request_key"] is None


def test_llm_providers_exposes_reserved_key_when_mapping_configured(monkeypatch, llm_client):
    import tldw_Server_API.app.core.config as core_config
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints
    import tldw_Server_API.app.core.LLM_Calls.adapter_registry as reg_mod

    monkeypatch.setattr(core_config, "load_comprehensive_config", _fake_config_with_llama)
    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", _fake_config_with_llama)
    monkeypatch.setenv("LLAMA_CPP_THINKING_BUDGET_PARAM", "reasoning_budget")

    class _DummyReg:
        def list_capabilities(self, include_disabled=True):
            return []

    monkeypatch.setattr(reg_mod, "get_registry", lambda: _DummyReg())

    response = llm_client.get("/api/v1/llm/providers")
    assert response.status_code == 200
    controls = _provider_by_display_name(response.json(), "Llama.cpp")["llama_cpp_controls"]
    assert controls["thinking_budget"]["request_key"] == "reasoning_budget"
    assert "reasoning_budget" in controls["reserved_extra_body_keys"]


def test_public_llm_providers_applies_override_merge_once(monkeypatch, llm_client):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints

    merge_calls = 0
    payload = {
        "providers": [{"name": "openai", "models": ["gpt-4o-mini"]}],
        "default_provider": "openai",
        "total_configured": 1,
    }

    async def configured_providers(**_kwargs):
        return payload

    def merge_once(value):
        nonlocal merge_calls
        merge_calls += 1
        return value

    monkeypatch.setattr(llm_endpoints, "get_configured_providers_async", configured_providers)
    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", configparser.ConfigParser)
    monkeypatch.setattr(llm_endpoints, "apply_llm_provider_overrides_to_listing", merge_once)

    response = llm_client.get("/api/v1/llm/providers")

    assert response.status_code == 200, response.text
    assert merge_calls == 1

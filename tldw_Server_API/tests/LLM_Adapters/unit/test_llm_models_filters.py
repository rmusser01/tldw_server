import configparser

from tldw_Server_API.app.core.exceptions import EgressPolicyError


def _fake_config():
    cfg = configparser.ConfigParser()
    cfg.add_section("API")
    cfg.set("API", "openai_api_key", "sk-test")
    cfg.set("API", "openai_model", "gpt-4o-mini, text-embedding-3-small")
    cfg.set("API", "default_api", "openai")
    # Provide minimal Local-API section to satisfy callers that probe both
    cfg.add_section("Local-API")
    return cfg


def _fake_config_with_local_discovery():
    cfg = _fake_config()
    cfg.set("Local-API", "vllm_api_IP", "http://127.0.0.1:8080/v1")
    return cfg


def _fake_config_openrouter_mixed():
    cfg = configparser.ConfigParser()
    cfg.add_section("API")
    cfg.set("API", "openrouter_api_key", "sk-or-test")
    cfg.set("API", "openrouter_model", "openai/gpt-4o-mini")
    cfg.set("API", "default_api", "openrouter")
    cfg.add_section("Local-API")
    return cfg


def _patch_llm_providers(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_providers

    monkeypatch.setattr(llm_providers, "load_comprehensive_config", _fake_config)
    monkeypatch.setattr(llm_providers, "list_provider_models", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(llm_providers, "apply_llm_provider_overrides_to_listing", lambda result: result)
    monkeypatch.setattr(llm_providers, "get_api_keys", lambda: {})
    monkeypatch.setattr(llm_providers, "list_image_models_for_catalog", _fake_image_models)


def _fake_image_models():
    return [
        {
            "provider": "image",
            "id": "image/stable_diffusion_cpp",
            "name": "stable_diffusion_cpp",
            "type": "image",
            "capabilities": {"image_generation": True},
            "modalities": {"input": ["text"], "output": ["image"]},
            "is_configured": False,
            "supported_formats": ["png", "jpg", "webp"],
        }
    ]


def test_llm_models_filter_type(monkeypatch, client_user_only):
    _patch_llm_providers(monkeypatch)

    client = client_user_only
    r = client.get("/api/v1/llm/models?type=chat")
    assert r.status_code == 200
    data = r.json()
    assert "openai/gpt-4o-mini" in data
    assert "openai/text-embedding-3-small" not in data

    r = client.get("/api/v1/llm/models?type=embedding")
    assert r.status_code == 200
    data = r.json()
    assert "openai/text-embedding-3-small" in data
    assert "openai/gpt-4o-mini" not in data
    assert "image/stable_diffusion_cpp" not in data

    r = client.get("/api/v1/llm/models?type=image")
    assert r.status_code == 200
    data = r.json()
    assert "image/stable_diffusion_cpp" in data


def test_llm_models_metadata_filter_modalities(monkeypatch, client_user_only):
    _patch_llm_providers(monkeypatch)

    client = client_user_only
    r = client.get("/api/v1/llm/models/metadata?input_modality=image")
    assert r.status_code == 200
    payload = r.json()
    names = {m.get("name") for m in payload.get("models", [])}
    assert "gpt-4o-mini" in names
    assert "text-embedding-3-small" not in names
    assert "stable_diffusion_cpp" not in names

    r = client.get("/api/v1/llm/models/metadata?output_modality=image")
    assert r.status_code == 200
    payload = r.json()
    names = {m.get("name") for m in payload.get("models", [])}
    assert "stable_diffusion_cpp" in names
    assert "gpt-4o-mini" not in names


def test_llm_models_metadata_handles_local_discovery_policy_errors(monkeypatch, client_user_only):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_providers

    calls = {"count": 0}

    def _raise_egress_policy(*_args, **_kwargs):
        calls["count"] += 1
        raise EgressPolicyError("Port not allowed: 8080")

    monkeypatch.setattr(llm_providers, "load_comprehensive_config", _fake_config_with_local_discovery)
    monkeypatch.setattr(llm_providers, "list_provider_models", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(llm_providers, "apply_llm_provider_overrides_to_listing", lambda result: result)
    monkeypatch.setattr(llm_providers, "get_api_keys", lambda: {})
    monkeypatch.setattr(llm_providers, "list_image_models_for_catalog", _fake_image_models)
    monkeypatch.setattr(llm_providers, "_http_fetch", _raise_egress_policy)

    client = client_user_only
    response = client.get("/api/v1/llm/models/metadata")
    assert response.status_code == 200

    payload = response.json()
    names = {m.get("name") for m in payload.get("models", [])}
    assert "gpt-4o-mini" in names
    assert calls["count"] > 0


def test_llm_models_metadata_includes_curated_qwen_models(monkeypatch, client_user_only):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_providers

    _patch_llm_providers(monkeypatch)
    monkeypatch.setattr(
        llm_providers,
        "list_provider_models",
        lambda provider: ["qwen-max", "qwen-plus", "qwen-turbo"] if provider == "qwen" else [],
    )

    client = client_user_only
    response = client.get("/api/v1/llm/models/metadata")
    assert response.status_code == 200
    payload = response.json()
    provider_and_name = {
        (str(model.get("provider")), str(model.get("name")))
        for model in payload.get("models", [])
    }
    assert ("qwen", "qwen-max") in provider_and_name
    assert ("qwen", "qwen-plus") in provider_and_name
    assert ("qwen", "qwen-turbo") in provider_and_name


def test_llm_models_metadata_marks_unconfigured_provider_catalog_models(
    monkeypatch, client_user_only
):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_providers

    _patch_llm_providers(monkeypatch)
    monkeypatch.setattr(
        llm_providers,
        "list_provider_models",
        lambda provider: (
            ["qwen-max"]
            if provider == "qwen"
            else ["gpt-4o-mini"]
            if provider == "openai"
            else []
        ),
    )

    client = client_user_only
    response = client.get("/api/v1/llm/models/metadata")

    assert response.status_code == 200
    payload = response.json()
    models = {
        (str(model.get("provider")), str(model.get("name"))): model
        for model in payload.get("models", [])
    }

    assert models[("openai", "gpt-4o-mini")]["is_configured"] is True
    assert models[("openai", "gpt-4o-mini")]["provider_is_configured"] is True
    assert models[("openai", "gpt-4o-mini")]["catalog_only"] is False
    assert models[("qwen", "qwen-max")]["is_configured"] is False
    assert models[("qwen", "qwen-max")]["provider_is_configured"] is False
    assert models[("qwen", "qwen-max")]["catalog_only"] is True


def test_llm_models_filter_type_openrouter_image_hints(monkeypatch, client_user_only):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_providers

    monkeypatch.setattr(llm_providers, "load_comprehensive_config", _fake_config_openrouter_mixed)
    monkeypatch.setattr(
        llm_providers,
        "list_provider_models",
        lambda provider: (
            ["black-forest-labs/flux.1-schnell", "openai/gpt-4o-mini"]
            if provider == "openrouter"
            else []
        ),
    )
    monkeypatch.setattr(llm_providers, "apply_llm_provider_overrides_to_listing", lambda result: result)
    monkeypatch.setattr(llm_providers, "get_api_keys", lambda: {"openrouter": "sk-or-test"})
    monkeypatch.setattr(llm_providers, "discover_openrouter_models", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(llm_providers, "list_image_models_for_catalog", lambda: [])
    monkeypatch.setattr(
        llm_providers,
        "_resolve_model_tokenizer_support",
        lambda *_args, **_kwargs: {
            "available": False,
            "tokenizer": None,
            "kind": None,
            "source": None,
            "detokenize": False,
            "count_accuracy": "unavailable",
            "strict_mode_effective": False,
        },
    )

    client = client_user_only

    chat_resp = client.get("/api/v1/llm/models?type=chat")
    assert chat_resp.status_code == 200
    chat_models = set(chat_resp.json())
    assert "openrouter/openai/gpt-4o-mini" in chat_models
    assert "openrouter/black-forest-labs/flux.1-schnell" not in chat_models

    image_resp = client.get("/api/v1/llm/models?type=image")
    assert image_resp.status_code == 200
    image_models = set(image_resp.json())
    assert "openrouter/black-forest-labs/flux.1-schnell" in image_models
    assert "openrouter/openai/gpt-4o-mini" not in image_models

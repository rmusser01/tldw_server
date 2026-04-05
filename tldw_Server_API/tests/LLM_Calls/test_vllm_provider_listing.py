import configparser

import pytest

from tldw_Server_API.app.core.VLLM_Management.models import VLLMInstanceCreate
from tldw_Server_API.app.core.VLLM_Management.sqlite_repo import SqliteVLLMInstanceRepository


def _fake_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.add_section("API")
    cfg.set("API", "default_api", "openai")
    cfg.add_section("Local-API")
    return cfg


def _fake_config_with_legacy_vllm(endpoint: str | None = None, model: str | None = None) -> configparser.ConfigParser:
    cfg = _fake_config()
    if endpoint is not None:
        cfg.set("Local-API", "vllm_api_IP", endpoint)
    if model is not None:
        cfg.set("Local-API", "vllm_model", model)
    return cfg


def _seed_repo(tmp_path):
    repo = SqliteVLLMInstanceRepository(tmp_path / "vllm_instances.db")
    vision = repo.create_instance(
        VLLMInstanceCreate(
            name="vision-a100",
            execution_mode="local",
            transport_config={},
            launch_spec={
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "served_model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
                "port": 8001,
            },
            routing_policy={},
            declared_capabilities={"chat": True, "vision": True},
        )
    )
    repo.update_instance_runtime(
        vision.instance_id,
        {
            "desired_state": "running",
            "observed_state": "healthy",
            "effective_capabilities": {"chat": True, "vision": True},
            "last_known_base_url": "http://127.0.0.1:8001/v1",
        },
    )

    embeddings = repo.create_instance(
        VLLMInstanceCreate(
            name="embed-l4",
            execution_mode="local",
            transport_config={},
            launch_spec={
                "model": "BAAI/bge-m3",
                "served_model_name": "BAAI/bge-m3",
                "port": 8010,
            },
            routing_policy={},
            declared_capabilities={"chat": True, "embeddings": True},
        )
    )
    repo.update_instance_runtime(
        embeddings.instance_id,
        {
            "desired_state": "running",
            "observed_state": "healthy",
            "effective_capabilities": {"chat": True, "embeddings": True},
            "last_known_base_url": "http://127.0.0.1:8010/v1",
        },
    )
    repo.set_default_instance(vision.instance_id)
    return repo, vision, embeddings


@pytest.mark.unit
def test_provider_listing_includes_managed_vllm_default_and_capabilities(monkeypatch, tmp_path):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints
    import tldw_Server_API.app.core.LLM_Calls.provider_metadata as provider_metadata
    import tldw_Server_API.app.core.VLLM_Management.resolver as resolver_module
    import tldw_Server_API.app.core.LLM_Calls.tokenizer_resolver as tokenizer_resolver

    repo, vision, embeddings = _seed_repo(tmp_path)
    managed_metadata = provider_metadata.get_managed_vllm_provider_metadata(repository=repo)
    captured = {}

    class _FakeTokenizerAdapter(tokenizer_resolver.ProviderNativeTokenizerHTTPAdapter):
        def __init__(self, *, base_url, model, api_key, timeout_seconds=10.0) -> None:
            super().__init__(
                base_url=base_url,
                model=model,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
            captured["base_url"] = base_url
            captured["model"] = model
            captured["api_key"] = api_key

        def encode(self, text: str) -> list[int]:
            return [1, 2, 3]

        def decode(self, token_ids: list[int]) -> str:
            return "ok"

    def _resolve_with_fake_adapter(provider, model, **kwargs):
        return tokenizer_resolver.resolve_tokenizer_metadata(
            provider,
            model,
            adapter_cls=_FakeTokenizerAdapter,
            **kwargs,
        )

    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", _fake_config)
    monkeypatch.setattr(llm_endpoints, "list_provider_models", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(llm_endpoints, "apply_llm_provider_overrides_to_listing", lambda result: result)
    monkeypatch.setattr(llm_endpoints, "get_api_keys", lambda: {})
    monkeypatch.setattr(llm_endpoints, "list_image_models_for_catalog", lambda: [])
    monkeypatch.setattr(llm_endpoints, "_llm_registry_capability_envelopes", lambda: {})
    monkeypatch.setattr(llm_endpoints, "get_managed_vllm_provider_metadata", lambda: managed_metadata)
    monkeypatch.setattr(llm_endpoints, "resolve_tokenizer_metadata", _resolve_with_fake_adapter)
    monkeypatch.setattr(resolver_module, "get_default_vllm_instance_repository", lambda: repo)

    payload = llm_endpoints.get_configured_providers(include_deprecated=False)
    providers = {provider["name"]: provider for provider in payload["providers"]}
    vllm = providers["vllm"]

    assert vllm["managed_instances"]["count"] == 2
    assert vllm["managed_instances"]["default_instance_id"] == vision.instance_id
    assert vllm["managed_instances"]["capabilities"]["vision"] is True
    assert vllm["managed_instances"]["capabilities"]["embeddings"] is True
    assert vllm["endpoint"] == "http://127.0.0.1:8001/v1"
    assert vllm["default_model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert vllm["tokenizers"]["Qwen/Qwen2.5-VL-7B-Instruct"]["available"] is True
    assert set(vllm["models"]) == {"Qwen/Qwen2.5-VL-7B-Instruct", "BAAI/bge-m3"}
    assert captured["base_url"] == "http://127.0.0.1:8001/v1"
    assert captured["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert embeddings.instance_id in {
        instance["instance_id"]
        for instance in vllm["managed_instances"]["instances"]
    }


@pytest.mark.unit
def test_provider_listing_uses_legacy_vllm_fallback_when_managed_default_is_unset(monkeypatch, tmp_path):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints
    import tldw_Server_API.app.core.LLM_Calls.provider_metadata as provider_metadata

    repo, vision, _embeddings = _seed_repo(tmp_path)
    repo.set_default_instance(None)
    managed_metadata = provider_metadata.get_managed_vllm_provider_metadata(repository=repo)

    monkeypatch.setattr(
        llm_endpoints,
        "load_comprehensive_config",
        lambda: _fake_config_with_legacy_vllm(
            endpoint="http://127.0.0.1:9000/v1",
            model="legacy-vllm-model",
        ),
    )
    monkeypatch.setattr(llm_endpoints, "list_provider_models", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(llm_endpoints, "apply_llm_provider_overrides_to_listing", lambda result: result)
    monkeypatch.setattr(llm_endpoints, "get_api_keys", lambda: {})
    monkeypatch.setattr(llm_endpoints, "list_image_models_for_catalog", lambda: [])
    monkeypatch.setattr(llm_endpoints, "_llm_registry_capability_envelopes", lambda: {})
    monkeypatch.setattr(llm_endpoints, "get_managed_vllm_provider_metadata", lambda: managed_metadata)
    monkeypatch.setattr(
        llm_endpoints,
        "resolve_tokenizer_metadata",
        lambda *_args, **_kwargs: {
            "available": False,
            "tokenizer": None,
            "kind": None,
            "source": None,
            "detokenize": False,
            "count_accuracy": "unavailable",
            "strict_mode_effective": False,
            "error": "Tokenizer not available",
        },
    )

    payload = llm_endpoints.get_configured_providers(include_deprecated=False)
    providers = {provider["name"]: provider for provider in payload["providers"]}
    vllm = providers["vllm"]

    assert vllm["managed_instances"]["default_instance_id"] is None
    assert vllm["managed_instances"]["default_model"] is None
    assert vllm["managed_instances"]["default_base_url"] is None
    assert vllm["default_model"] == "legacy-vllm-model"
    assert vllm["endpoint"] == "http://127.0.0.1:9000/v1"
    assert "Qwen/Qwen2.5-VL-7B-Instruct" in vllm["models"]
    assert "BAAI/bge-m3" in vllm["models"]
    assert vision.instance_id in {
        instance["instance_id"]
        for instance in vllm["managed_instances"]["instances"]
    }


@pytest.mark.unit
def test_provider_listing_leaves_vllm_default_unset_without_managed_or_legacy_default(monkeypatch, tmp_path):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints
    import tldw_Server_API.app.core.LLM_Calls.provider_metadata as provider_metadata

    repo, _vision, _embeddings = _seed_repo(tmp_path)
    repo.set_default_instance(None)
    managed_metadata = provider_metadata.get_managed_vllm_provider_metadata(repository=repo)

    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", _fake_config)
    monkeypatch.setattr(llm_endpoints, "list_provider_models", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(llm_endpoints, "apply_llm_provider_overrides_to_listing", lambda result: result)
    monkeypatch.setattr(llm_endpoints, "get_api_keys", lambda: {})
    monkeypatch.setattr(llm_endpoints, "list_image_models_for_catalog", lambda: [])
    monkeypatch.setattr(llm_endpoints, "_llm_registry_capability_envelopes", lambda: {})
    monkeypatch.setattr(llm_endpoints, "get_managed_vllm_provider_metadata", lambda: managed_metadata)
    monkeypatch.setattr(
        llm_endpoints,
        "resolve_tokenizer_metadata",
        lambda *_args, **_kwargs: {
            "available": False,
            "tokenizer": None,
            "kind": None,
            "source": None,
            "detokenize": False,
            "count_accuracy": "unavailable",
            "strict_mode_effective": False,
            "error": "Tokenizer not available",
        },
    )

    payload = llm_endpoints.get_configured_providers(include_deprecated=False)
    providers = {provider["name"]: provider for provider in payload["providers"]}
    vllm = providers["vllm"]

    assert vllm["managed_instances"]["default_instance_id"] is None
    assert vllm["managed_instances"]["default_model"] is None
    assert vllm["managed_instances"]["default_base_url"] is None
    assert vllm["default_model"] is None
    assert "Qwen/Qwen2.5-VL-7B-Instruct" in vllm["models"]
    assert "BAAI/bge-m3" in vllm["models"]

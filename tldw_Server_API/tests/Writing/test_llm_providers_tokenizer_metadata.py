import configparser


def _fake_config(mlx_model_path: str = "test-mlx-model") -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.add_section("API")
    cfg.set("API", "openai_api_key", "sk-test")
    cfg.set("API", "openai_model", "gpt-4o-mini")
    cfg.set("API", "anthropic_api_key", "anthropic-test-key")
    cfg.set("API", "anthropic_model", "claude-opus-4-20250514")
    cfg.set("API", "google_api_key", "google-test-key")
    cfg.set("API", "google_model", "gemini-2.5-flash")
    cfg.set("API", "groq_api_key", "groq-test-key")
    cfg.set("API", "groq_model", "openai/gpt-4o-mini")
    cfg.set("API", "cohere_api_key", "cohere-test-key")
    cfg.set("API", "cohere_model", "command-a-03-2025")
    cfg.set("API", "deepseek_api_key", "deepseek-test-key")
    cfg.set("API", "deepseek_model", "deepseek-chat")
    cfg.set("API", "mistral_api_key", "mistral-test-key")
    cfg.set("API", "mistral_model", "mistral-large-latest")
    cfg.set("API", "bedrock_api_key", "bedrock-test-key")
    cfg.set("API", "bedrock_model", "anthropic.claude-3-5-sonnet-20240620-v1:0")
    cfg.set("API", "bedrock_region", "us-west-2")
    cfg.set("API", "default_api", "openai")

    cfg.add_section("Local-API")
    cfg.set("Local-API", "ollama_api_IP", "http://127.0.0.1:11434")
    cfg.set("Local-API", "ollama_model", "llama3.2")

    cfg.add_section("MLX")
    cfg.set("MLX", "mlx_model_path", mlx_model_path)
    return cfg


def _fake_openai_only_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.add_section("API")
    cfg.set("API", "openai_api_key", "sk-test")
    cfg.set("API", "openai_model", "gpt-4o-mini")
    cfg.set("API", "default_api", "openai")
    cfg.add_section("Local-API")
    return cfg


def _ready_provider_readiness(*, provider_name: str, **_kwargs) -> dict[str, object]:
    chat_provider_aliases = {
        "custom_openai_api": "custom-openai-api",
        "llama": "llama.cpp",
    }
    chat_provider = chat_provider_aliases.get(provider_name, provider_name)
    return {
        "availability": "enabled",
        "provider_enabled": True,
        "readiness_reason_code": None,
        "readiness_message": None,
        "chat_provider": chat_provider,
    }


def test_llm_providers_tokenizer_metadata_mirrors_strict_fields(monkeypatch, tmp_path):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints

    mlx_model_dir = tmp_path / "mlx-model"
    mlx_model_dir.mkdir()
    monkeypatch.setattr(
        llm_endpoints,
        "load_comprehensive_config",
        lambda: _fake_config(str(mlx_model_dir)),
    )
    monkeypatch.setattr(llm_endpoints, "list_provider_models", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(llm_endpoints, "apply_llm_provider_overrides_to_listing", lambda result: result)
    monkeypatch.setattr(llm_endpoints, "get_api_keys", lambda: {})
    monkeypatch.setattr(llm_endpoints, "list_image_models_for_catalog", lambda: [])
    monkeypatch.setattr(llm_endpoints, "_llm_registry_capability_envelopes", lambda: {})
    monkeypatch.setattr(llm_endpoints, "_provider_readiness", _ready_provider_readiness)
    monkeypatch.setattr(
        llm_endpoints,
        "ALL_SUPPORTED_PROVIDER_NAMES_LIST",
        [*llm_endpoints.ALL_SUPPORTED_PROVIDER_NAMES_LIST, "mlx"],
    )

    def _fake_tokenizer_metadata(provider, model, **_kwargs):
        key = (provider.strip().lower(), model.strip())
        if key[0] in {"ollama", "mlx"}:
            return {
                "available": True,
                "tokenizer": f"{key[0]}:remote",
                "kind": "provider-native",
                "source": f"{key[0]}.http.tokenize",
                "detokenize": True,
                "count_accuracy": "exact",
                "strict_mode_effective": False,
            }
        return {
            "available": False,
            "tokenizer": None,
            "kind": None,
            "source": None,
            "detokenize": False,
            "count_accuracy": "unavailable",
            "strict_mode_effective": False,
            "error": "Tokenizer not available",
        }

    monkeypatch.setattr(llm_endpoints, "resolve_tokenizer_metadata", _fake_tokenizer_metadata)
    monkeypatch.setattr(
        llm_endpoints,
        "_resolve_model_tokenizer_support",
        lambda provider, model, _config: _fake_tokenizer_metadata(provider, model),
    )

    payload = llm_endpoints.get_configured_providers(include_deprecated=False)
    providers = {p["name"]: p for p in payload["providers"]}

    assert "ollama" in providers
    assert "mlx" in providers

    ollama_model = providers["ollama"]["models"][0]
    ollama_tokenizer = providers["ollama"]["tokenizers"][ollama_model]
    assert ollama_tokenizer["count_accuracy"] == "exact"
    assert ollama_tokenizer["strict_mode_effective"] is False

    mlx_model = providers["mlx"]["models"][0]
    mlx_tokenizer = providers["mlx"]["tokenizers"][mlx_model]
    assert mlx_tokenizer["count_accuracy"] == "exact"
    assert mlx_tokenizer["strict_mode_effective"] is False

    for model_info in providers["openai"]["models_info"]:
        assert "count_accuracy" in model_info
        assert "strict_mode_effective" in model_info


def test_llm_providers_tokenizer_metadata_reflects_strict_runtime_env(monkeypatch, tmp_path):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints

    monkeypatch.setenv("STRICT_TOKEN_COUNTING", "true")
    mlx_model_dir = tmp_path / "mlx-model"
    mlx_model_dir.mkdir()
    monkeypatch.setattr(
        llm_endpoints,
        "load_comprehensive_config",
        lambda: _fake_config(str(mlx_model_dir)),
    )
    monkeypatch.setattr(llm_endpoints, "list_provider_models", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(llm_endpoints, "apply_llm_provider_overrides_to_listing", lambda result: result)
    monkeypatch.setattr(llm_endpoints, "get_api_keys", lambda: {})
    monkeypatch.setattr(llm_endpoints, "list_image_models_for_catalog", lambda: [])
    monkeypatch.setattr(llm_endpoints, "_llm_registry_capability_envelopes", lambda: {})
    monkeypatch.setattr(llm_endpoints, "_provider_readiness", _ready_provider_readiness)
    monkeypatch.setattr(
        llm_endpoints,
        "ALL_SUPPORTED_PROVIDER_NAMES_LIST",
        [*llm_endpoints.ALL_SUPPORTED_PROVIDER_NAMES_LIST, "mlx"],
    )

    def _fake_tokenizer_metadata(provider, model, **kwargs):  # noqa: ARG001
        strict_effective = bool(kwargs.get("strict_mode_effective"))
        return {
            "available": True,
            "tokenizer": "fake:strict",
            "kind": "provider-native",
            "source": "fake.strict",
            "detokenize": True,
            "count_accuracy": "exact",
            "strict_mode_effective": strict_effective,
        }

    monkeypatch.setattr(llm_endpoints, "resolve_tokenizer_metadata", _fake_tokenizer_metadata)

    def _fake_model_tokenizer_support(provider, model, _config):
        return _fake_tokenizer_metadata(
            provider,
            model,
            strict_mode_effective=llm_endpoints._strict_token_counting_enabled_shared(default=False),
        )

    monkeypatch.setattr(
        llm_endpoints,
        "_resolve_model_tokenizer_support",
        _fake_model_tokenizer_support,
    )

    payload = llm_endpoints.get_configured_providers(include_deprecated=False)
    providers = {p["name"]: p for p in payload["providers"]}

    ollama_model = providers["ollama"]["models"][0]
    mlx_model = providers["mlx"]["models"][0]
    openai_model = providers["openai"]["models"][0]

    assert providers["ollama"]["tokenizers"][ollama_model]["strict_mode_effective"] is True
    assert providers["mlx"]["tokenizers"][mlx_model]["strict_mode_effective"] is True
    assert providers["openai"]["tokenizers"][openai_model]["strict_mode_effective"] is True

    for model_info in providers["openai"]["models_info"]:
        assert model_info["strict_mode_effective"] is True


def test_llm_providers_real_resolver_exact_for_anthropic_google_cohere_bedrock_groq(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints
    import tldw_Server_API.app.core.LLM_Calls.tokenizer_resolver as resolver_module
    from tldw_Server_API.app.core.LLM_Calls.tokenizer_resolver import (
        resolve_tokenizer_metadata as resolve_tokenizer_metadata_shared,
    )

    monkeypatch.delenv("STRICT_TOKEN_COUNTING", raising=False)
    monkeypatch.delenv("BEDROCK_RUNTIME_ENDPOINT", raising=False)
    monkeypatch.delenv("BEDROCK_API_BASE_URL", raising=False)
    monkeypatch.delenv("BEDROCK_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("BEDROCK_REGION", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE123")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-example-key")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", _fake_config)
    monkeypatch.setattr(llm_endpoints, "list_provider_models", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(llm_endpoints, "apply_llm_provider_overrides_to_listing", lambda result: result)
    monkeypatch.setattr(llm_endpoints, "get_api_keys", lambda: {})
    monkeypatch.setattr(llm_endpoints, "list_image_models_for_catalog", lambda: [])
    monkeypatch.setattr(llm_endpoints, "_llm_registry_capability_envelopes", lambda: {})

    class _FakeTokenizer:
        name = "o200k_base"

        def encode(self, text: str, disallowed_special=()):  # noqa: ARG002
            return [ord(ch) for ch in text]

        def decode(self, token_ids):
            return "".join(chr(int(token_id)) for token_id in token_ids)

    monkeypatch.setattr(resolver_module, "resolve_tiktoken_encoding", lambda _model: _FakeTokenizer())

    class _FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self) -> dict[str, object]:
            return self._payload

    def _fake_post(*, url: str, payload, headers, timeout):  # noqa: ANN001, ARG001
        lowered = str(url).lower()
        if "api.anthropic.com" in lowered and "count_tokens" in lowered:
            return _FakeResponse(200, {"input_tokens": 11})
        if "generativelanguage.googleapis.com" in lowered and "counttokens" in lowered:
            return _FakeResponse(200, {"totalTokens": 9})
        if "api.cohere.ai" in lowered and "tokenize" in lowered:
            return _FakeResponse(200, {"tokens": [1, 2, 3]})
        if "api.cohere.ai" in lowered and "detokenize" in lowered:
            return _FakeResponse(200, {"text": "ok"})
        if "bedrock-runtime.us-west-2.amazonaws.com" in lowered and "count-tokens" in lowered:
            return _FakeResponse(200, {"inputTokens": 8})
        return _FakeResponse(404, {})

    monkeypatch.setattr(resolver_module, "_http_post", _fake_post)

    # Avoid local env crashes from MLX artifact fallback importing transformers/torch in this test process.
    def _safe_resolve_metadata(provider, model, **kwargs):
        if str(provider).strip().lower() == "mlx":
            return {
                "available": False,
                "tokenizer": None,
                "kind": None,
                "source": None,
                "detokenize": False,
                "count_accuracy": "unavailable",
                "strict_mode_effective": bool(kwargs.get("strict_mode_effective", False)),
                "error": "MLX tokenizer unavailable in test process",
            }
        return resolve_tokenizer_metadata_shared(provider, model, **kwargs)

    monkeypatch.setattr(llm_endpoints, "resolve_tokenizer_metadata", _safe_resolve_metadata)

    payload = llm_endpoints.get_configured_providers(include_deprecated=False)
    providers = {p["name"]: p for p in payload["providers"]}

    anthropic_model = providers["anthropic"]["models"][0]
    anthropic_tok = providers["anthropic"]["tokenizers"][anthropic_model]
    assert anthropic_tok["count_accuracy"] == "exact"
    assert anthropic_tok["kind"] == "provider-native-count"
    assert anthropic_tok["detokenize"] is False

    google_model = providers["google"]["models"][0]
    google_tok = providers["google"]["tokenizers"][google_model]
    assert google_tok["count_accuracy"] == "exact"
    assert google_tok["kind"] == "provider-native-count"
    assert google_tok["detokenize"] is False

    cohere_model = providers["cohere"]["models"][0]
    cohere_tok = providers["cohere"]["tokenizers"][cohere_model]
    assert cohere_tok["count_accuracy"] == "exact"
    assert cohere_tok["kind"] == "provider-native"
    assert cohere_tok["detokenize"] is True

    bedrock_model = providers["bedrock"]["models"][0]
    bedrock_tok = providers["bedrock"]["tokenizers"][bedrock_model]
    assert bedrock_tok["count_accuracy"] == "exact"
    assert bedrock_tok["kind"] == "provider-native-count"
    assert bedrock_tok["detokenize"] is False

    groq_model = providers["groq"]["models"][0]
    groq_tok = providers["groq"]["tokenizers"][groq_model]
    assert groq_tok["count_accuracy"] == "exact"
    assert groq_tok["kind"] == "tiktoken"
    assert groq_tok["detokenize"] is True

    deepseek_model = providers["deepseek"]["models"][0]
    deepseek_tok = providers["deepseek"]["tokenizers"][deepseek_model]
    assert deepseek_tok["count_accuracy"] == "unavailable"
    assert deepseek_tok["kind"] == "tiktoken"

    mistral_model = providers["mistral"]["models"][0]
    mistral_tok = providers["mistral"]["tokenizers"][mistral_model]
    assert mistral_tok["count_accuracy"] == "unavailable"
    assert mistral_tok["kind"] == "tiktoken"


def test_llm_providers_skips_tokenizer_probe_for_non_text_models(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints

    probe_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", _fake_config)
    monkeypatch.setattr(llm_endpoints, "apply_llm_provider_overrides_to_listing", lambda result: result)
    monkeypatch.setattr(llm_endpoints, "get_api_keys", lambda: {})
    monkeypatch.setattr(llm_endpoints, "list_image_models_for_catalog", lambda: [])
    monkeypatch.setattr(llm_endpoints, "_llm_registry_capability_envelopes", lambda: {})

    def _fake_list_provider_models(provider: str) -> list[str]:
        if provider == "google":
            return ["gemini-2.5-flash", "imagen-4.0-generate-001"]
        return []

    monkeypatch.setattr(llm_endpoints, "list_provider_models", _fake_list_provider_models)

    def _fake_tokenizer_metadata(provider: str, model: str, **_kwargs):
        probe_calls.append((provider, model))
        return {
            "available": True,
            "tokenizer": "fake:remote-count",
            "kind": "provider-native-count",
            "source": "fake.google",
            "detokenize": False,
            "count_accuracy": "exact",
            "strict_mode_effective": False,
        }

    monkeypatch.setattr(llm_endpoints, "resolve_tokenizer_metadata", _fake_tokenizer_metadata)

    payload = llm_endpoints.get_configured_providers(include_deprecated=False)
    providers = {p["name"]: p for p in payload["providers"]}

    google = providers["google"]
    assert "gemini-2.5-flash" in google["models"]
    assert "imagen-4.0-generate-001" in google["models"]

    assert ("google", "gemini-2.5-flash") in probe_calls
    assert ("google", "imagen-4.0-generate-001") not in probe_calls

    image_tok = google["tokenizers"]["imagen-4.0-generate-001"]
    assert image_tok["available"] is False
    assert "skipped" in str(image_tok.get("error") or "").lower()


def test_llm_providers_skips_runtime_tokenizer_probe_for_inprocess_test_mode(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints

    probe_calls: list[tuple[str, str]] = []

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("E2E_INPROCESS", "1")
    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", _fake_openai_only_config)
    monkeypatch.setattr(llm_endpoints, "list_provider_models", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(llm_endpoints, "apply_llm_provider_overrides_to_listing", lambda result: result)
    monkeypatch.setattr(llm_endpoints, "get_api_keys", lambda: {})
    monkeypatch.setattr(llm_endpoints, "list_image_models_for_catalog", lambda: [])
    monkeypatch.setattr(llm_endpoints, "_llm_registry_capability_envelopes", lambda: {})

    def _fake_tokenizer_metadata(provider: str, model: str, **_kwargs):
        probe_calls.append((provider, model))
        return {
            "available": True,
            "tokenizer": "fake:runtime",
            "kind": "provider-native",
            "source": "fake.runtime",
            "detokenize": True,
            "count_accuracy": "exact",
            "strict_mode_effective": False,
        }

    monkeypatch.setattr(llm_endpoints, "resolve_tokenizer_metadata", _fake_tokenizer_metadata)

    payload = llm_endpoints.get_configured_providers(include_deprecated=False)
    providers = {p["name"]: p for p in payload["providers"]}
    openai_model = providers["openai"]["models"][0]
    openai_tokenizer = providers["openai"]["tokenizers"][openai_model]

    assert probe_calls == []
    assert openai_tokenizer["available"] is False
    assert "in-process test mode" in str(openai_tokenizer["error"]).lower()


def test_llm_providers_probes_only_configured_commercial_models(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_endpoints

    probe_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", _fake_openai_only_config)
    monkeypatch.setattr(llm_endpoints, "apply_llm_provider_overrides_to_listing", lambda result: result)
    monkeypatch.setattr(llm_endpoints, "get_api_keys", lambda: {})
    monkeypatch.setattr(llm_endpoints, "list_image_models_for_catalog", lambda: [])
    monkeypatch.setattr(llm_endpoints, "_llm_registry_capability_envelopes", lambda: {})
    monkeypatch.setattr(
        llm_endpoints,
        "list_provider_models",
        lambda provider: ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"] if provider == "openai" else [],
    )

    def _fake_tokenizer_metadata(provider: str, model: str, **_kwargs):
        probe_calls.append((provider, model))
        return {
            "available": True,
            "tokenizer": "fake:tiktoken",
            "kind": "tiktoken",
            "source": "tiktoken.fake",
            "detokenize": True,
            "count_accuracy": "exact",
            "strict_mode_effective": False,
        }

    monkeypatch.setattr(llm_endpoints, "resolve_tokenizer_metadata", _fake_tokenizer_metadata)

    payload = llm_endpoints.get_configured_providers(include_deprecated=False)
    providers = {p["name"]: p for p in payload["providers"]}
    openai = providers["openai"]

    assert openai["models"] == ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    openai_probe_calls = [model for provider, model in probe_calls if provider == "openai"]
    assert openai_probe_calls == ["gpt-4o-mini"]

    skipped_tok = openai["tokenizers"]["gpt-4o"]
    assert skipped_tok["available"] is False
    assert "configured" in str(skipped_tok.get("error") or "").lower()

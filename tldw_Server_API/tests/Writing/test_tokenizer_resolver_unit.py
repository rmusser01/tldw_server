import configparser
from pathlib import Path

import pytest


class _FakeTokenizer:
    def __init__(self, name: str = "o200k_base") -> None:
        self.name = name

    def encode(self, text: str, disallowed_special=()):  # noqa: ARG002 - compat signature
        return [ord(ch) for ch in text]

    def decode(self, token_ids):
        return "".join(chr(int(token_id)) for token_id in token_ids)


def test_resolve_tokenizer_openrouter_openai_canonical_exact(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    monkeypatch.setattr(resolver, "resolve_tiktoken_encoding", lambda model: _FakeTokenizer("o200k_base"))

    resolution = resolver.resolve_tokenizer(
        "openrouter",
        "openai/gpt-4o-mini",
        strict_mode_effective=True,
    )

    assert resolution.available is True
    assert resolution.count_accuracy == "exact"
    assert resolution.tokenizer == "tiktoken:o200k_base"
    assert resolution.kind == "tiktoken"
    assert resolution.strict_mode_effective is True


def test_resolve_tokenizer_groq_openai_canonical_exact(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    monkeypatch.setattr(resolver, "resolve_tiktoken_encoding", lambda model: _FakeTokenizer("o200k_base"))

    resolution = resolver.resolve_tokenizer(
        "groq",
        "openai/gpt-4o-mini",
        strict_mode_effective=True,
    )

    assert resolution.available is True
    assert resolution.count_accuracy == "exact"
    assert resolution.tokenizer == "tiktoken:o200k_base"
    assert resolution.kind == "tiktoken"
    assert resolution.strict_mode_effective is True


def test_resolve_tokenizer_non_exact_best_effort_classified_unavailable(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    monkeypatch.setattr(resolver, "resolve_tiktoken_encoding", lambda model: _FakeTokenizer("cl100k_base"))

    resolution = resolver.resolve_tokenizer(
        "deepseek",
        "gpt-3.5-turbo",
        strict_mode_effective=True,
    )

    assert resolution.available is True
    assert resolution.count_accuracy == "unavailable"
    assert resolution.tokenizer == "tiktoken:cl100k_base"
    assert resolution.kind == "tiktoken"


def test_resolve_tokenizer_mistral_best_effort_classified_unavailable(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    monkeypatch.setattr(resolver, "resolve_tiktoken_encoding", lambda model: _FakeTokenizer("cl100k_base"))

    resolution = resolver.resolve_tokenizer(
        "mistral",
        "mistral-large-latest",
        strict_mode_effective=True,
    )

    assert resolution.available is True
    assert resolution.count_accuracy == "unavailable"
    assert resolution.tokenizer == "tiktoken:cl100k_base"
    assert resolution.kind == "tiktoken"


def test_resolve_tokenizer_runtime_probe_downgrades_failed_provider_native(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    monkeypatch.setattr(resolver, "resolve_tiktoken_encoding", lambda _model: _FakeTokenizer("cl100k_base"))

    config = configparser.ConfigParser()
    config.add_section("Local-API")
    config.set("Local-API", "ollama_api_IP", "http://127.0.0.1:11434/v1")

    def _raise_unavailable(*, url: str, payload, headers, timeout):  # noqa: ANN001, ARG001
        raise RuntimeError("connection refused")

    monkeypatch.setattr(resolver, "_http_post", _raise_unavailable)

    resolution = resolver.resolve_tokenizer(
        "ollama",
        "llama3.2",
        strict_mode_effective=True,
        config_parser=config,
        runtime_probe_exact=True,
    )

    assert resolution.available is True
    assert resolution.count_accuracy == "unavailable"
    assert resolution.kind == "tiktoken"
    assert resolution.tokenizer == "tiktoken:cl100k_base"


def test_resolve_tokenizer_openai_unavailable_error_not_masked_by_native_config(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    def _raise_unavailable(_model: str):
        raise resolver.TokenizerUnavailable("Tokenizer not available for provider/model")

    monkeypatch.setattr(resolver, "resolve_tiktoken_encoding", _raise_unavailable)

    resolution = resolver.resolve_tokenizer(
        "openai",
        "definitely-not-a-real-model",
        strict_mode_effective=False,
    )

    assert resolution.available is False
    assert "not available" in str(resolution.error or "").lower()
    assert "provider-native tokenizer is not configured for provider" not in str(resolution.error or "").lower()


def test_resolve_tokenizer_custom_openai_api_openai_host_guard(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    monkeypatch.setattr(resolver, "resolve_tiktoken_encoding", lambda _model: _FakeTokenizer("cl100k_base"))

    config = configparser.ConfigParser()
    config.add_section("API")
    config.set("API", "custom_openai_api_ip", "https://api.openai.com/v1")
    config.set("API", "custom_openai_api_model", "gpt-4.1-2025-04-14")

    resolution = resolver.resolve_tokenizer(
        "custom_openai_api",
        "gpt-4.1-2025-04-14",
        strict_mode_effective=True,
        config_parser=config,
    )

    assert resolution.available is True
    assert resolution.count_accuracy == "unavailable"
    assert resolution.kind == "tiktoken"
    assert resolution.tokenizer == "tiktoken:cl100k_base"


def test_resolve_tokenizer_numbered_custom_openai_openai_host_guard(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    monkeypatch.setattr(resolver, "resolve_tiktoken_encoding", lambda _model: _FakeTokenizer("cl100k_base"))

    config = configparser.ConfigParser()
    config.add_section("API")
    config.set("API", "custom_openai37_api_ip", "https://api.openai.com/v1")
    config.set("API", "custom_openai37_api_model", "gpt-4.1-2025-04-14")

    resolution = resolver.resolve_tokenizer(
        "custom-openai-api-37",
        "gpt-4.1-2025-04-14",
        strict_mode_effective=True,
        config_parser=config,
    )

    assert resolution.available is True
    assert resolution.count_accuracy == "unavailable"
    assert resolution.kind == "tiktoken"
    assert resolution.tokenizer == "tiktoken:cl100k_base"


def test_resolve_tokenizer_anthropic_count_only_exact_from_config():
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    config = configparser.ConfigParser()
    config.add_section("API")
    config.set("API", "anthropic_api_key", "anthropic-test-key")

    resolution = resolver.resolve_tokenizer(
        "anthropic",
        "claude-opus-4-20250514",
        strict_mode_effective=True,
        config_parser=config,
    )

    assert resolution.available is True
    assert resolution.count_accuracy == "exact"
    assert resolution.kind == "provider-native-count"
    assert resolution.source == "anthropic.http.count_tokens"
    assert resolution.tokenizer == "anthropic:remote-count"
    assert resolution.detokenize_available is False
    assert callable(getattr(resolution.encoding, "count_tokens", None))


def test_resolve_tokenizer_bedrock_openai_host_runtime_guard(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    monkeypatch.setattr(resolver, "resolve_tiktoken_encoding", lambda _model: _FakeTokenizer("cl100k_base"))

    config = configparser.ConfigParser()
    config.add_section("API")
    config.set("API", "bedrock_api_key", "bedrock-test-key")
    config.set("API", "bedrock_api_base_url", "https://api.openai.com/v1")

    resolution = resolver.resolve_tokenizer(
        "bedrock",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        strict_mode_effective=True,
        config_parser=config,
    )

    assert resolution.available is True
    assert resolution.count_accuracy == "unavailable"
    assert resolution.kind == "tiktoken"
    assert resolution.tokenizer == "tiktoken:cl100k_base"


def test_resolve_tokenizer_google_count_only_exact_from_config():
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    config = configparser.ConfigParser()
    config.add_section("API")
    config.set("API", "google_api_key", "google-test-key")

    resolution = resolver.resolve_tokenizer(
        "google",
        "gemini-2.5-flash",
        strict_mode_effective=True,
        config_parser=config,
    )

    assert resolution.available is True
    assert resolution.count_accuracy == "exact"
    assert resolution.kind == "provider-native-count"
    assert resolution.source == "google.http.count_tokens"
    assert resolution.tokenizer == "google:remote-count"
    assert resolution.detokenize_available is False
    assert callable(getattr(resolution.encoding, "count_tokens", None))


def test_resolve_tokenizer_cohere_tokenizer_exact_from_config():
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    config = configparser.ConfigParser()
    config.add_section("API")
    config.set("API", "cohere_api_key", "cohere-test-key")

    resolution = resolver.resolve_tokenizer(
        "cohere",
        "command-a-03-2025",
        strict_mode_effective=True,
        config_parser=config,
    )

    assert resolution.available is True
    assert resolution.count_accuracy == "exact"
    assert resolution.kind == "provider-native"
    assert resolution.source == "cohere.http.tokenize"
    assert resolution.tokenizer == "cohere:remote"
    assert resolution.detokenize_available is True
    assert callable(getattr(resolution.encoding, "encode", None))
    assert callable(getattr(resolution.encoding, "decode", None))


def test_resolve_tokenizer_bedrock_anthropic_count_only_exact_from_config(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    monkeypatch.delenv("BEDROCK_RUNTIME_ENDPOINT", raising=False)
    monkeypatch.delenv("BEDROCK_API_BASE_URL", raising=False)
    monkeypatch.delenv("BEDROCK_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("BEDROCK_REGION", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE123")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-example-key")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    config = configparser.ConfigParser()
    config.add_section("API")
    config.set("API", "bedrock_api_key", "bedrock-test-key")
    config.set("API", "bedrock_region", "us-west-2")
    config.set("API", "bedrock_model", "anthropic.claude-3-5-sonnet-20240620-v1:0")

    resolution = resolver.resolve_tokenizer(
        "bedrock",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        strict_mode_effective=True,
        config_parser=config,
    )

    assert resolution.available is True
    assert resolution.count_accuracy == "exact"
    assert resolution.kind == "provider-native-count"
    assert resolution.source == "bedrock.http.count_tokens"
    assert resolution.tokenizer == "bedrock:remote-count"
    assert resolution.detokenize_available is False
    assert callable(getattr(resolution.encoding, "count_tokens", None))


def test_resolve_tokenizer_bedrock_non_anthropic_not_exact(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    monkeypatch.setattr(resolver, "resolve_tiktoken_encoding", lambda _model: _FakeTokenizer("cl100k_base"))

    config = configparser.ConfigParser()
    config.add_section("API")
    config.set("API", "bedrock_api_key", "bedrock-test-key")

    resolution = resolver.resolve_tokenizer(
        "bedrock",
        "openai.gpt-oss-20b-1:0",
        strict_mode_effective=True,
        config_parser=config,
    )

    assert resolution.available is True
    assert resolution.count_accuracy == "unavailable"
    assert resolution.kind == "tiktoken"
    assert resolution.tokenizer == "tiktoken:cl100k_base"


def test_bedrock_count_only_adapter_calls_runtime_count_tokens(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    calls: list[tuple[str, dict[str, str], dict[str, object]]] = []

    class _FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self) -> dict[str, object]:
            return self._payload

    def _fake_post(*, url: str, payload, headers, timeout):  # noqa: ANN001, ARG001
        calls.append((url, dict(headers), dict(payload)))
        return _FakeResponse(200, {"inputTokens": 9})

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE123")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-example-key")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.setattr(resolver, "_http_post", _fake_post)

    adapter = resolver.BedrockCountOnlyHTTPAdapter(
        base_url="https://bedrock-runtime.us-west-2.amazonaws.com",
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        api_key=None,
        region="us-west-2",
        aws_credentials=None,
    )

    count = adapter.count_tokens("hello")
    assert count == 9
    assert calls
    assert "/model/anthropic.claude-3-5-sonnet-20240620-v1%3A0/count-tokens" in calls[0][0]
    assert str(calls[0][1].get("Authorization", "")).startswith("AWS4-HMAC-SHA256 ")
    assert calls[0][1].get("X-Amz-Date")
    assert calls[0][1].get("X-Amz-Content-Sha256")


def test_bedrock_commercial_exact_requires_sigv4_for_aws_host(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    monkeypatch.delenv("BEDROCK_RUNTIME_ENDPOINT", raising=False)
    monkeypatch.delenv("BEDROCK_API_BASE_URL", raising=False)
    monkeypatch.delenv("BEDROCK_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("BEDROCK_REGION", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.setattr(resolver, "_resolve_bedrock_sigv4_credentials", lambda parser: None)

    config = configparser.ConfigParser()
    config.add_section("API")
    config.set("API", "bedrock_region", "us-west-2")
    config.set("API", "bedrock_api_key", "legacy-bearer-key")

    resolution = resolver.resolve_commercial_exact_tokenizer(
        "bedrock",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        strict_mode_effective=True,
        config_parser=config,
    )

    assert resolution.available is False
    assert "sigv4" in str(resolution.error or "").lower()


def test_bedrock_count_only_adapter_local_proxy_uses_bearer_without_sigv4(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    calls: list[tuple[str, dict[str, str], dict[str, object]]] = []

    class _FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self) -> dict[str, object]:
            return self._payload

    def _fake_post(*, url: str, payload, headers, timeout):  # noqa: ANN001, ARG001
        calls.append((url, dict(headers), dict(payload)))
        return _FakeResponse(200, {"inputTokens": 4})

    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.setattr(resolver, "_http_post", _fake_post)

    adapter = resolver.BedrockCountOnlyHTTPAdapter(
        base_url="http://127.0.0.1:9000",
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        api_key="proxy-key",
        region="us-west-2",
        aws_credentials=None,
    )

    count = adapter.count_tokens("hello")
    assert count == 4
    assert calls
    assert calls[0][1].get("Authorization") == "Bearer proxy-key"
    assert "X-Amz-Date" not in calls[0][1]


def test_resolve_tokenizer_ollama_native_exact_from_config():
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    config = configparser.ConfigParser()
    config.add_section("Local-API")
    config.set("Local-API", "ollama_api_IP", "http://127.0.0.1:11434/api/chat")

    resolution = resolver.resolve_tokenizer(
        "ollama",
        "llama3.2",
        strict_mode_effective=True,
        config_parser=config,
    )

    assert resolution.available is True
    assert resolution.count_accuracy == "exact"
    assert resolution.kind == "provider-native"
    assert resolution.source == "ollama.http.tokenize"
    assert resolution.tokenizer == "ollama:remote"


def test_resolve_tokenizer_mlx_prefers_active_registry(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    active = _FakeTokenizer("mlx-active")

    monkeypatch.setattr(resolver, "_get_active_mlx_tokenizer", lambda _model: (active, "active-model"))
    monkeypatch.setattr(
        resolver,
        "_load_mlx_artifact_tokenizer",
        lambda _model: pytest.fail("artifact fallback should not run when registry tokenizer is active"),
    )

    resolution = resolver.resolve_tokenizer("mlx", "active-model", strict_mode_effective=True)

    assert resolution.available is True
    assert resolution.count_accuracy == "exact"
    assert resolution.kind == "provider-native"
    assert resolution.source == "mlx.registry.active"
    assert resolution.tokenizer == "mlx:active:active-model"


def test_resolve_tokenizer_mlx_uses_artifact_fallback(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    monkeypatch.setattr(resolver, "_get_active_mlx_tokenizer", lambda _model: None)
    monkeypatch.setattr(resolver, "_load_mlx_artifact_tokenizer", lambda _model: _FakeTokenizer("mlx-artifact"))

    resolution = resolver.resolve_tokenizer("mlx", "/tmp/fake-mlx-model", strict_mode_effective=True)

    assert resolution.available is True
    assert resolution.count_accuracy == "exact"
    assert resolution.kind == "provider-native"
    assert resolution.source == "mlx.artifact.tokenizer"
    assert resolution.tokenizer == "mlx:artifact"


def test_resolve_tokenizer_metadata_contains_strict_fields(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    monkeypatch.setattr(resolver, "resolve_tiktoken_encoding", lambda model: _FakeTokenizer("cl100k_base"))

    metadata = resolver.resolve_tokenizer_metadata(
        "openai",
        "gpt-4o-mini",
        strict_mode_effective=False,
    )

    assert metadata["available"] is True
    assert metadata["count_accuracy"] == "exact"
    assert metadata["strict_mode_effective"] is False
    assert metadata["tokenizer"] == "tiktoken:cl100k_base"


def test_google_count_only_adapter_falls_back_to_query_key_auth(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    calls: list[tuple[str, dict[str, str]]] = []

    class _FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self) -> dict[str, object]:
            return self._payload

    def _fake_post(*, url: str, payload, headers, timeout):  # noqa: ANN001, ARG001
        calls.append((url, dict(headers)))
        if "?key=test-google-key" in url:
            return _FakeResponse(200, {"totalTokens": 7})
        return _FakeResponse(401, {"error": {"message": "invalid key transport"}})

    monkeypatch.setattr(resolver, "_http_post", _fake_post)
    monkeypatch.setenv("GOOGLE_COUNTTOKENS_ALLOW_QUERY_KEY_FALLBACK", "true")

    adapter = resolver.GoogleCountOnlyHTTPAdapter(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        model="gemini-2.5-flash",
        api_key="test-google-key",
    )

    count = adapter.count_tokens("hello world")
    assert count == 7
    assert any("?key=test-google-key" in url for url, _headers in calls)
    assert any(headers.get("x-goog-api-key") == "test-google-key" for _url, headers in calls)


def test_google_count_only_adapter_does_not_use_query_key_fallback_by_default(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    calls: list[str] = []

    class _FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self) -> dict[str, object]:
            return self._payload

    def _fake_post(*, url: str, payload, headers, timeout):  # noqa: ANN001, ARG001
        calls.append(url)
        return _FakeResponse(401, {"error": {"message": "invalid key transport"}})

    monkeypatch.setattr(resolver, "_http_post", _fake_post)
    monkeypatch.delenv("GOOGLE_COUNTTOKENS_ALLOW_QUERY_KEY_FALLBACK", raising=False)

    adapter = resolver.GoogleCountOnlyHTTPAdapter(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        model="gemini-2.5-flash",
        api_key="test-google-key",
    )

    with pytest.raises(resolver.TokenizerUnavailable, match="401"):
        adapter.count_tokens("hello world")

    assert all("?key=" not in url for url in calls)


def test_coerce_int_rejects_non_integral_float():
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    assert resolver._coerce_int(12.5) is None
    assert resolver._coerce_int(12.0) == 12


def test_mlx_candidate_paths_blocks_parent_traversal(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    monkeypatch.setenv("MLX_MODEL_DIR", "/tmp/mlx-models")

    assert resolver._mlx_candidate_paths("../outside") == []


def test_mlx_candidate_paths_resolves_relative_model_within_root(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    model_root = Path("/tmp/mlx-models")
    monkeypatch.setenv("MLX_MODEL_DIR", str(model_root))

    candidates = resolver._mlx_candidate_paths("family/model")

    expected = [(model_root.resolve(strict=False) / "family/model").resolve(strict=False)]
    assert candidates == expected


def test_mlx_candidate_paths_rejects_absolute_path_outside_root(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import tokenizer_resolver as resolver

    monkeypatch.setenv("MLX_MODEL_DIR", "/tmp/mlx-models")

    assert resolver._mlx_candidate_paths("/etc/passwd") == []

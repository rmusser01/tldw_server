from __future__ import annotations

import uuid
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.schemas.embeddings_models import (
    CreateEmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Embeddings.orchestrator import EmbeddingExecutorOutput
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingExecutionError,
    EmbeddingExecutionResult,
    EmbeddingInputError,
    EmbeddingPolicyError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
)
from tldw_Server_API.app.main import app

pytestmark = pytest.mark.unit


@pytest.fixture
def client(monkeypatch):
    original_overrides = dict(app.dependency_overrides)
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("AUTO_DOWNLOAD_MODELS", "false")
    monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)

    async def override_user():
        return User(
            id=1,
            username="embedding-user",
            email="embedding-user@example.test",
            is_active=True,
            is_admin=False,
        )

    app.dependency_overrides[get_request_user] = override_user

    with TestClient(app) as test_client:
        csrf_token = f"test-csrf-{uuid.uuid4().hex}"
        test_client.cookies.set("csrf_token", csrf_token)
        test_client.headers["X-CSRF-Token"] = csrf_token
        test_client.headers["Authorization"] = "Bearer test-api-key"
        yield test_client

    app.dependency_overrides.clear()
    app.dependency_overrides.update(original_overrides)


def _ok_response(model: str = "patched-model") -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        data=[EmbeddingData(embedding=[0.1, 0.2], index=0)],
        model=model,
        usage=EmbeddingUsage(prompt_tokens=2, total_tokens=2),
    )


class FakePrepared:
    def __init__(self, total_tokens: int = 2) -> None:
        self.normalized_input = SimpleNamespace(total_tokens=total_tokens)


class FakeOrchestrator:
    def __init__(self, *, result=None, prepare_error=None, execute_error=None) -> None:
        self.result = result
        self.prepare_error = prepare_error
        self.execute_error = execute_error
        self.prepare_calls = []
        self.execute_calls = []

    def prepare(self, raw_input, context):
        self.prepare_calls.append((raw_input, context))
        if self.prepare_error is not None:
            raise self.prepare_error
        return FakePrepared(total_tokens=3)

    async def execute(self, prepared):
        self.execute_calls.append(prepared)
        if self.execute_error is not None:
            raise self.execute_error
        return self.result or EmbeddingExecutionResult(
            vectors=[[0.25, 0.75]],
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
            prompt_tokens=3,
            total_tokens=3,
            cache_hits=0,
            cache_misses=1,
        )


def _user() -> User:
    return User(
        id=1,
        username="embedding-user",
        email="embedding-user@example.test",
        is_active=True,
        is_admin=False,
    )


def _request():
    return SimpleNamespace()


class FakeCredentials:
    def __init__(
        self,
        *,
        api_key: str | None,
        source: str = "server",
        auth_source: str | None = None,
        credential_fields: dict[str, object] | None = None,
        app_config: dict[str, object] | None = None,
    ) -> None:
        self.api_key = api_key
        self.source = source
        self.auth_source = auth_source
        self.credential_fields = credential_fields or {}
        self.app_config = app_config
        self.touch_last_used = AsyncMock()


class _NoopMetric:
    def labels(self, **_kwargs):
        return self

    def inc(self, *_args, **_kwargs):
        return None

    def dec(self, *_args, **_kwargs):
        return None

    def observe(self, *_args, **_kwargs):
        return None


class _RecordingCounter:
    def __init__(self) -> None:
        self.label_calls: list[dict[str, object]] = []
        self.inc_calls: list[object] = []

    def labels(self, **kwargs):
        self.label_calls.append(dict(kwargs))
        return self

    def inc(self, amount=1):
        self.inc_calls.append(amount)
        return None


class _ParityCache:
    def __init__(self, cached_vectors):
        self.cached_vectors = {
            _friendly_cache_key(*key): [float(value) for value in vector]
            for key, vector in (cached_vectors or {}).items()
        }
        self.gets: list[str] = []
        self.sets: list[tuple[str, list[float]]] = []

    async def get(self, key: str):
        self.gets.append(key)
        value = self.cached_vectors.get(key)
        return list(value) if value is not None else None

    async def set(self, key: str, value):
        vector = [float(item) for item in value]
        self.sets.append((key, vector))
        self.cached_vectors[key] = vector


class _ParityRGGovernor:
    def __init__(self) -> None:
        self.reserves = []
        self.commits = []

    async def reserve(self, rg_request, *, op_id=None):
        self.reserves.append((rg_request, op_id))
        return SimpleNamespace(allowed=True, retry_after=None), "parity-rg-handle"

    async def commit(self, handle_id, *, actuals=None, op_id=None):
        self.commits.append((handle_id, actuals, op_id))


class _ParityRGPolicyLoader:
    def get_policy(self, _policy_id):
        return {}


def _friendly_cache_key(text, provider, model, dimensions=None, backend_identity=None):
    _ = backend_identity
    return f"{provider}|{model}|{dimensions if dimensions is not None else ''}|{text}"


def _decoded_token_text(token_array):
    return "tokens:" + "-".join(str(token) for token in token_array)


def _parity_tokens_to_texts(tokens_input, _model):
    if tokens_input and isinstance(tokens_input[0], int):
        lengths = [len(tokens_input)]
        texts = [_decoded_token_text(tokens_input)]
    else:
        lengths = [len(item) for item in tokens_input]
        texts = [_decoded_token_text(item) for item in tokens_input]
    return texts, sum(lengths), lengths


def _assert_cache_writes_are_float_vectors(cache_sets):
    for key, vector in cache_sets:
        assert isinstance(key, str)
        assert isinstance(vector, list)
        assert vector
        assert all(isinstance(item, float) for item in vector)


def _assert_response_parity(result):
    compared_headers = [
        "X-Embeddings-Provider",
        "X-Embeddings-Fallback-From",
        "X-Embeddings-Dimensions-Policy",
        "Retry-After",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "X-RateLimit-PerMinute-Limit",
        "X-RateLimit-PerMinute-Remaining",
        "X-RateLimit-Tokens-Remaining",
    ]
    legacy = result["legacy"]
    orchestrator = result["orchestrator"]

    assert legacy["status"] == orchestrator["status"]
    assert legacy["json"] == orchestrator["json"]
    assert legacy["usage"] == orchestrator["usage"]
    for header in compared_headers:
        assert legacy["headers"].get(header) == orchestrator["headers"].get(header)
    _assert_cache_writes_are_float_vectors(legacy["cache_sets"])
    _assert_cache_writes_are_float_vectors(orchestrator["cache_sets"])


def _run_dual_path_embedding_request(
    client,
    monkeypatch,
    *,
    mod,
    payload,
    headers=None,
    provider_vectors=None,
    cached_vectors=None,
    failing_providers=None,
    mismatch_providers=None,
    dimensions_policy="reduce",
    allow_fallback_with_header=False,
):
    headers = headers or {}
    provider_vectors = provider_vectors or {}
    cached_vectors = cached_vectors or {}
    failing_providers = set(failing_providers or set())
    mismatch_providers = set(mismatch_providers or set())
    fallback_chain = {"openai": ["openai", "huggingface"], "huggingface": ["huggingface"]}
    fallback_model_map = {
        "openai:text-embedding-3-small": {
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        },
    }

    async def fake_backpressure(*_args, **_kwargs):
        return None

    async def fake_log_usage(*_args, **_kwargs):
        return None

    async def fake_backfill(*_args, **_kwargs):
        return None

    async def fake_resolve(provider, *_args, **_kwargs):
        provider = (provider or "").strip().lower()
        return FakeCredentials(
            api_key="test-provider-key" if provider in {"openai", "cohere", "google"} else None,
            source="user" if provider in {"openai", "cohere", "google"} else "none",
        )

    def fake_policy_setting(name, default):
        if name == "EMBEDDINGS_FALLBACK_CHAIN":
            return fallback_chain
        if name == "EMBEDDINGS_FALLBACK_MODEL_MAP":
            return fallback_model_map
        return default

    def fake_map_model(src_provider, dst_provider, model_id):
        return fallback_model_map.get(f"{src_provider}:{model_id}", {}).get(dst_provider, model_id)

    def fake_count_tokens(text, _model):
        return max(1, len(str(text).split()))

    async def fake_provider(texts, provider, model, config, metadata=None, dimensions=None):
        _ = (config, metadata, dimensions)
        provider = provider.lower()
        if provider in failing_providers:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"{provider} down")
        vectors = []
        for index, text in enumerate(texts):
            key = (provider, model, text)
            vector = provider_vectors.get(key)
            if vector is None:
                vector = [float(index + 1), float(len(text))]
            vectors.append([float(item) for item in vector])
        if provider in mismatch_providers:
            return vectors[:-1]
        return vectors

    def fake_build_provider_config(provider_enum, model_id, api_key=None, api_url=None, dimensions=None):
        return {
            "provider": getattr(provider_enum, "value", str(provider_enum)),
            "model": model_id,
            "api_key": api_key,
            "api_url": api_url,
            "dimensions": dimensions,
        }

    for metric_name in (
        "active_embedding_requests",
        "embedding_request_duration",
        "embedding_requests_total",
        "embedding_cache_hits",
        "embedding_cache_misses",
        "embedding_fallbacks_total",
        "embedding_provider_failures",
        "embedding_provider_failures_total",
        "embedding_dimension_adjustments_total",
        "embedding_token_inputs_total",
        "embedding_policy_denied_total",
    ):
        if hasattr(mod, metric_name):
            monkeypatch.setattr(mod, metric_name, _NoopMetric(), raising=False)

    monkeypatch.setattr(mod, "_check_backpressure_and_quotas", fake_backpressure)
    monkeypatch.setattr(mod, "log_llm_usage", fake_log_usage)
    monkeypatch.setattr(mod, "backfill_legacy_tokens_to_ledger", fake_backfill)
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_cache_key", _friendly_cache_key)
    monkeypatch.setattr(mod, "_orchestrator_backend_identity", lambda _provider, _model: None)
    monkeypatch.setattr(mod, "build_provider_config", fake_build_provider_config)
    monkeypatch.setattr(mod, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(mod, "tokens_to_texts", _parity_tokens_to_texts)
    monkeypatch.setattr(mod, "_embedding_policy_setting", fake_policy_setting)
    monkeypatch.setattr(mod, "map_model_for_provider", fake_map_model)
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "true")
    monkeypatch.setenv("EMBEDDINGS_DIMENSION_POLICY", dimensions_policy)
    if allow_fallback_with_header:
        monkeypatch.setenv("EMBEDDINGS_ALLOW_FALLBACK_WITH_HEADER", "true")
    else:
        monkeypatch.delenv("EMBEDDINGS_ALLOW_FALLBACK_WITH_HEADER", raising=False)

    results = {}
    for label, flag_enabled in (("legacy", False), ("orchestrator", True)):
        if flag_enabled:
            monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
        else:
            monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)

        cache = _ParityCache(cached_vectors)
        provider_call = AsyncMock(side_effect=fake_provider)
        monkeypatch.setattr(mod.embedding_cache, "get", cache.get)
        monkeypatch.setattr(mod.embedding_cache, "set", cache.set)
        monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", provider_call)

        rg_governor = _ParityRGGovernor()
        monkeypatch.setattr(app.state, "rg_governor", rg_governor, raising=False)
        monkeypatch.setattr(app.state, "rg_policy_loader", _ParityRGPolicyLoader(), raising=False)

        response = client.post(
            "/api/v1/embeddings",
            headers=headers,
            json=payload,
        )
        body = response.json()
        results[label] = {
            "status": response.status_code,
            "json": body,
            "usage": body.get("usage") if isinstance(body, dict) else None,
            "headers": response.headers,
            "cache_gets": list(cache.gets),
            "cache_sets": list(cache.sets),
            "provider_calls": [call.args for call in provider_call.await_args_list],
            "rg_reserves": list(rg_governor.reserves),
            "rg_commits": list(rg_governor.commits),
        }

    _assert_response_parity(results)
    return results


def test_dual_path_single_string_numeric_embedding_response(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    result = _run_dual_path_embedding_request(
        client,
        monkeypatch,
        mod=mod,
        payload={
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "input": "red seed",
        },
        headers={"x-provider": "huggingface"},
        provider_vectors={
            ("huggingface", "sentence-transformers/all-MiniLM-L6-v2", "red seed"): [0.2, 0.4],
        },
    )

    assert result["legacy"]["status"] == status.HTTP_200_OK
    assert result["legacy"]["json"]["data"][0]["embedding"] == pytest.approx([0.4472136, 0.8944272])
    assert result["legacy"]["json"]["usage"] == {"prompt_tokens": 2, "total_tokens": 2}


def test_dual_path_batch_string_response_preserves_indexes(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    result = _run_dual_path_embedding_request(
        client,
        monkeypatch,
        mod=mod,
        payload={
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "input": ["first", "second"],
        },
        headers={"x-provider": "huggingface"},
        provider_vectors={
            ("huggingface", "sentence-transformers/all-MiniLM-L6-v2", "first"): [1.0, 0.0],
            ("huggingface", "sentence-transformers/all-MiniLM-L6-v2", "second"): [0.0, 1.0],
        },
    )

    assert [item["index"] for item in result["legacy"]["json"]["data"]] == [0, 1]
    assert [item["embedding"] for item in result["legacy"]["json"]["data"]] == [[1.0, 0.0], [0.0, 1.0]]


def test_dual_path_single_token_array_response(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    result = _run_dual_path_embedding_request(
        client,
        monkeypatch,
        mod=mod,
        payload={
            "model": "text-embedding-3-small",
            "input": [101, 102, 103],
        },
        provider_vectors={
            ("openai", "text-embedding-3-small", "tokens:101-102-103"): [0.1, 0.3],
        },
    )

    assert result["legacy"]["status"] == status.HTTP_200_OK
    assert result["legacy"]["json"]["usage"] == {"prompt_tokens": 3, "total_tokens": 3}
    assert result["legacy"]["json"]["data"][0]["embedding"] == pytest.approx([0.3162278, 0.9486833])


def test_dual_path_batch_token_array_base64_with_dimensions(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    result = _run_dual_path_embedding_request(
        client,
        monkeypatch,
        mod=mod,
        payload={
            "model": "text-embedding-3-small",
            "input": [[101, 102], [103, 104, 105]],
            "encoding_format": "base64",
            "dimensions": 2,
        },
        provider_vectors={
            ("openai", "text-embedding-3-small", "tokens:101-102"): [0.1, 0.2, 0.3],
            ("openai", "text-embedding-3-small", "tokens:103-104-105"): [0.4, 0.5, 0.6],
        },
    )

    assert result["legacy"]["headers"]["X-Embeddings-Dimensions-Policy"] == "reduce"
    for item, expected in zip(result["legacy"]["json"]["data"], ([0.1, 0.2], [0.4, 0.5])):
        raw = np.frombuffer(base64.b64decode(item["embedding"]), dtype=np.float32)
        assert raw.tolist() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("policy", "dimensions", "provider_vector", "expected_length"),
    [
        ("reduce", 2, [0.1, 0.2, 0.3, 0.4], 2),
        ("pad", 4, [0.1, 0.2], 4),
        ("ignore", 5, [0.1, 0.2, 0.3], 3),
    ],
)
def test_dual_path_huggingface_dimensions_policies(
    client,
    monkeypatch,
    policy,
    dimensions,
    provider_vector,
    expected_length,
):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    result = _run_dual_path_embedding_request(
        client,
        monkeypatch,
        mod=mod,
        payload={
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "input": "dimension policy",
            "dimensions": dimensions,
        },
        headers={"x-provider": "huggingface"},
        provider_vectors={
            ("huggingface", "sentence-transformers/all-MiniLM-L6-v2", "dimension policy"): provider_vector,
        },
        dimensions_policy=policy,
    )

    assert result["legacy"]["headers"]["X-Embeddings-Dimensions-Policy"] == policy
    assert len(result["legacy"]["json"]["data"][0]["embedding"]) == expected_length


def test_dual_path_full_cache_hit_skips_provider_execution(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    result = _run_dual_path_embedding_request(
        client,
        monkeypatch,
        mod=mod,
        payload={
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "input": ["cached one", "cached two"],
        },
        headers={"x-provider": "huggingface"},
        cached_vectors={
            ("cached one", "huggingface", "sentence-transformers/all-MiniLM-L6-v2", None, None): [1.0, 0.0],
            ("cached two", "huggingface", "sentence-transformers/all-MiniLM-L6-v2", None, None): [0.0, 1.0],
        },
    )

    assert result["legacy"]["provider_calls"] == []
    assert result["orchestrator"]["provider_calls"] == []
    assert result["legacy"]["cache_sets"] == []
    assert result["orchestrator"]["cache_sets"] == []
    assert result["legacy"]["json"]["data"][0]["embedding"] == [1.0, 0.0]


def test_dual_path_partial_cache_hit_calls_provider_for_misses_only(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    result = _run_dual_path_embedding_request(
        client,
        monkeypatch,
        mod=mod,
        payload={
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "input": ["hit", "miss"],
        },
        headers={"x-provider": "huggingface"},
        cached_vectors={
            ("hit", "huggingface", "sentence-transformers/all-MiniLM-L6-v2", None, None): [0.8, 0.2],
        },
        provider_vectors={
            ("huggingface", "sentence-transformers/all-MiniLM-L6-v2", "miss"): [0.2, 0.8],
        },
    )

    assert [call[0] for call in result["legacy"]["provider_calls"]] == [["miss"]]
    assert [call[0] for call in result["orchestrator"]["provider_calls"]] == [["miss"]]
    assert result["legacy"]["cache_sets"] == [
        ("huggingface|sentence-transformers/all-MiniLM-L6-v2||miss", [0.2, 0.8])
    ]
    assert result["orchestrator"]["cache_sets"] == result["legacy"]["cache_sets"]


def test_dual_path_openai_primary_falls_back_to_huggingface_with_headers(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    result = _run_dual_path_embedding_request(
        client,
        monkeypatch,
        mod=mod,
        payload={
            "model": "text-embedding-3-small",
            "input": "fallback me",
        },
        failing_providers={"openai"},
        provider_vectors={
            ("huggingface", "sentence-transformers/all-MiniLM-L6-v2", "fallback me"): [0.6, 0.4],
        },
    )

    assert result["legacy"]["status"] == status.HTTP_200_OK
    assert result["legacy"]["headers"]["X-Embeddings-Provider"] == "huggingface"
    assert result["legacy"]["headers"]["X-Embeddings-Fallback-From"] == "openai"
    assert result["legacy"]["json"]["model"] == "huggingface:sentence-transformers/all-MiniLM-L6-v2"


def test_dual_path_explicit_provider_suppresses_fallback(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    result = _run_dual_path_embedding_request(
        client,
        monkeypatch,
        mod=mod,
        payload={
            "model": "text-embedding-3-small",
            "input": "no fallback",
        },
        headers={"x-provider": "openai"},
        failing_providers={"openai"},
        provider_vectors={
            ("huggingface", "sentence-transformers/all-MiniLM-L6-v2", "no fallback"): [0.6, 0.4],
        },
    )

    assert result["legacy"]["status"] == status.HTTP_503_SERVICE_UNAVAILABLE
    assert result["legacy"]["headers"].get("X-Embeddings-Fallback-From") is None


def test_dual_path_provider_vector_count_mismatch_maps_to_502(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    result = _run_dual_path_embedding_request(
        client,
        monkeypatch,
        mod=mod,
        payload={
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "input": ["one", "two"],
        },
        headers={"x-provider": "huggingface"},
        provider_vectors={
            ("huggingface", "sentence-transformers/all-MiniLM-L6-v2", "one"): [1.0, 0.0],
            ("huggingface", "sentence-transformers/all-MiniLM-L6-v2", "two"): [0.0, 1.0],
        },
        mismatch_providers={"huggingface"},
    )

    assert result["legacy"]["status"] == status.HTTP_502_BAD_GATEWAY


def test_flag_unset_calls_legacy_path(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    legacy = AsyncMock(return_value=_ok_response("legacy-model"))
    orchestrator = AsyncMock(side_effect=AssertionError("orchestrator should not be called"))
    monkeypatch.setattr(mod, "_create_embedding_legacy", legacy, raising=False)
    monkeypatch.setattr(mod, "_create_embedding_with_orchestrator", orchestrator, raising=False)

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "hello world"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["model"] == "legacy-model"
    assert legacy.await_count == 1
    assert orchestrator.await_count == 0


def test_flag_true_calls_orchestrator_path(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    legacy = AsyncMock(side_effect=AssertionError("legacy should not be called"))
    orchestrator = AsyncMock(return_value=_ok_response("orchestrator-model"))
    monkeypatch.setattr(mod, "_create_embedding_legacy", legacy, raising=False)
    monkeypatch.setattr(mod, "_create_embedding_with_orchestrator", orchestrator, raising=False)

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "hello world"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["model"] == "orchestrator-model"
    assert legacy.await_count == 0
    assert orchestrator.await_count == 1


def test_orchestrator_input_error_maps_to_current_400_shape(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    fake_orchestrator = FakeOrchestrator(
        prepare_error=EmbeddingInputError("empty_input", "Input cannot be empty")
    )
    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
        raising=False,
    )

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "   "},
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Input cannot be empty"


def test_orchestrator_token_limit_error_preserves_top_level_shape(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    fake_orchestrator = FakeOrchestrator(
        prepare_error=EmbeddingInputError(
            "input_too_long",
            "One or more inputs exceed max tokens 2 for model text-embedding-3-small",
            details=[{"index": 0, "tokens": 3}],
        )
    )
    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
        raising=False,
    )

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "three token input"},
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["error"] == "input_too_long"
    assert response.json()["details"] == [{"index": 0, "tokens": 3}]


def test_orchestrator_missing_credentials_preserves_503_detail_dict(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    fake_orchestrator = FakeOrchestrator(
        execute_error=EmbeddingProviderError(
            "missing_provider_credentials",
            "Embeddings provider 'cohere' requires an API key.",
            provider="cohere",
            model="embed-english-v3.0",
        )
    )
    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
        raising=False,
    )

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "cohere"},
        json={"model": "embed-english-v3.0", "input": "hello world"},
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"]["error_code"] == "missing_provider_credentials"


def test_orchestrator_full_cache_hit_still_requires_provider_credentials(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")

    async def fake_resolve(*_args, **_kwargs):
        return FakeCredentials(api_key=None, source="server")

    cache_get = AsyncMock(return_value=[0.0, 1.0])
    cache_set = AsyncMock()
    provider_call = AsyncMock(side_effect=AssertionError("provider path should not be called"))
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod.embedding_cache, "get", cache_get)
    monkeypatch.setattr(mod.embedding_cache, "set", cache_set)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", provider_call)

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "cohere"},
        json={"model": "embed-english-v3.0", "input": "cached but missing key"},
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"]["error_code"] == "missing_provider_credentials"
    assert cache_get.await_count == 0
    assert cache_set.await_count == 0
    assert provider_call.await_count == 0


def test_orchestrator_full_cache_hit_touches_resolved_provider_credentials(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    credentials = FakeCredentials(api_key="cohere-key", source="user")

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    cache_get = AsyncMock(return_value=[0.0, 1.0])
    cache_set = AsyncMock()
    provider_call = AsyncMock(side_effect=AssertionError("provider path should not be called"))
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod.embedding_cache, "get", cache_get)
    monkeypatch.setattr(mod.embedding_cache, "set", cache_set)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", provider_call)

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "cohere"},
        json={"model": "embed-english-v3.0", "input": "cached with key"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["data"][0]["embedding"] == [0.0, 1.0]
    assert cache_get.await_count == 1
    assert cache_set.await_count == 0
    assert provider_call.await_count == 0
    credentials.touch_last_used.assert_awaited_once()


def test_orchestrator_rate_limit_error_includes_retry_after(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    fake_orchestrator = FakeOrchestrator(
        execute_error=EmbeddingRateLimitError(
            "provider_rate_limited",
            "Rate limit exceeded",
            retry_after=7,
            provider="openai",
            model="text-embedding-3-small",
        )
    )
    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
        raising=False,
    )

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "hello world"},
    )

    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert response.headers["Retry-After"] == "7"
    assert response.json()["detail"] == "Rate limit exceeded"


def test_orchestrator_response_headers_are_applied(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    fake_orchestrator = FakeOrchestrator(
        result=EmbeddingExecutionResult(
            vectors=[[0.25, 0.75]],
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
            prompt_tokens=3,
            total_tokens=3,
            cache_hits=0,
            cache_misses=1,
            response_headers={"X-Embeddings-Provider": "huggingface"},
        )
    )
    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
        raising=False,
    )

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "huggingface"},
        json={"model": "sentence-transformers/all-MiniLM-L6-v2", "input": "hello world"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["X-Embeddings-Provider"] == "huggingface"
    assert response.json()["model"] == "huggingface:sentence-transformers/all-MiniLM-L6-v2"


def test_orchestrator_cache_hits_increment_legacy_metric(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    cache_hits = _RecordingCounter()
    fake_orchestrator = FakeOrchestrator(
        result=EmbeddingExecutionResult(
            vectors=[[0.25, 0.75]],
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
            prompt_tokens=3,
            total_tokens=3,
            cache_hits=2,
            cache_misses=0,
        )
    )
    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
        raising=False,
    )
    monkeypatch.setattr(mod, "embedding_cache_hits", cache_hits, raising=False)

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "huggingface"},
        json={"model": "sentence-transformers/all-MiniLM-L6-v2", "input": "cached"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert cache_hits.label_calls == [
        {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"}
    ]
    assert cache_hits.inc_calls == [2]


def test_orchestrator_missing_model_policy_error_maps_to_400():
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    mapped = mod._embedding_domain_error_to_http(
        EmbeddingPolicyError("model_required", "Model is required")
    )

    assert isinstance(mapped, HTTPException)
    assert mapped.status_code == status.HTTP_400_BAD_REQUEST
    assert mapped.detail == "Model is required"


@pytest.mark.asyncio
async def test_executor_retries_openai_oauth_401_with_forced_refresh_and_touches_final_credentials(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "true")
    initial_credentials = FakeCredentials(
        api_key="oauth-initial-key",
        source="user",
        auth_source="oauth",
    )
    refreshed_credentials = FakeCredentials(
        api_key="oauth-refreshed-key",
        source="user",
        auth_source="oauth",
    )
    refresh_flags: list[bool] = []

    async def fake_resolve(provider, current_user, request, *, force_oauth_refresh=False):
        assert provider == "openai"
        assert current_user.id == 1
        assert request is not None
        refresh_flags.append(force_oauth_refresh)
        return refreshed_credentials if force_oauth_refresh else initial_credentials

    provider_calls: list[dict[str, object]] = []

    async def fake_provider(texts, provider, model, config, metadata=None, dimensions=None):
        provider_calls.append(
            {
                "texts": list(texts),
                "provider": provider,
                "model": model,
                "api_key": config.get("api_key"),
                "metadata": metadata,
                "dimensions": dimensions,
            }
        )
        if len(provider_calls) == 1:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="expired token")
        return [[0.9, 0.1]]

    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", fake_provider)
    monkeypatch.setattr(mod, "_record_oauth_401_retry", lambda *_args, **_kwargs: None)

    executor = mod._EndpointEmbeddingExecutor(
        request=_request(),
        current_user=_user(),
        user_metadata={"user_id": 1},
    )

    vectors = await executor.create(
        ["refresh me"],
        provider="openai",
        model="text-embedding-3-small",
        dimensions=None,
    )

    assert vectors == [[0.9, 0.1]]
    assert refresh_flags == [False, True]
    assert [call["api_key"] for call in provider_calls] == [
        "oauth-initial-key",
        "oauth-refreshed-key",
    ]
    refreshed_credentials.touch_last_used.assert_awaited_once()
    initial_credentials.touch_last_used.assert_not_called()


def test_orchestrator_openai_oauth_second_401_propagates_original_auth_error(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "true")
    initial_credentials = FakeCredentials(
        api_key="oauth-initial-key",
        source="user",
        auth_source="oauth",
    )
    refreshed_credentials = FakeCredentials(
        api_key="oauth-refreshed-key",
        source="user",
        auth_source="oauth",
    )
    refresh_flags: list[bool] = []
    provider_call_count = 0

    async def fake_resolve(provider, current_user, request, *, force_oauth_refresh=False):
        assert provider == "openai"
        assert current_user.id == 1
        assert request is not None
        refresh_flags.append(force_oauth_refresh)
        return refreshed_credentials if force_oauth_refresh else initial_credentials

    async def fake_provider(texts, provider, model, config, metadata=None, dimensions=None):
        nonlocal provider_call_count
        _ = (texts, provider, model, config, metadata, dimensions)
        provider_call_count += 1
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="oauth auth failure")

    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", fake_provider)
    monkeypatch.setattr(mod, "_record_oauth_401_retry", lambda *_args, **_kwargs: None)

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "openai"},
        json={"model": "text-embedding-3-small", "input": "refresh fails"},
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "oauth auth failure"
    assert refresh_flags == [False, True]
    assert provider_call_count == 2
    initial_credentials.touch_last_used.assert_not_called()
    refreshed_credentials.touch_last_used.assert_not_called()


@pytest.mark.asyncio
async def test_executor_missing_key_raises_missing_credentials_domain_error(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    async def fake_resolve(*_args, **_kwargs):
        return FakeCredentials(api_key=None, source="server")

    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)

    executor = mod._EndpointEmbeddingExecutor(
        request=_request(),
        current_user=_user(),
        user_metadata=None,
    )

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await executor.create(
            ["needs key"],
            provider="cohere",
            model="embed-english-v3.0",
            dimensions=None,
        )

    assert exc_info.value.code == "missing_provider_credentials"


@pytest.mark.asyncio
async def test_executor_batches_provider_calls_and_concatenates_vectors(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    async def fake_resolve(*_args, **_kwargs):
        return FakeCredentials(api_key=None, source="none")

    provider_batches: list[list[str]] = []

    async def fake_provider(texts, provider, model, config, metadata=None, dimensions=None):
        _ = (provider, model, config, metadata, dimensions)
        provider_batches.append(list(texts))
        return [[float(len(provider_batches)), float(index)] for index, _ in enumerate(texts)]

    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", fake_provider)
    monkeypatch.setattr(mod, "MAX_BATCH_SIZE", 2)

    executor = mod._EndpointEmbeddingExecutor(
        request=_request(),
        current_user=_user(),
        user_metadata=None,
    )

    vectors = await executor.create(
        ["one", "two", "three", "four", "five"],
        provider="huggingface",
        model="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=None,
    )

    assert provider_batches == [["one", "two"], ["three", "four"], ["five"]]
    assert vectors == [[1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0], [3.0, 0.0]]


@pytest.mark.asyncio
async def test_executor_forwards_byok_endpoint_config_to_provider_and_cache_identity(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    credentials = FakeCredentials(
        api_key="local-key",
        source="user",
        credential_fields={
            "base_url": "http://127.0.0.1:8081/v1/embeddings?api_key=secret&tenant=alpha"
        },
    )

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    captured_configs: list[dict[str, object]] = []

    def fake_build_provider_config(provider_enum, model_id, api_key=None, api_url=None, dimensions=None):
        config = {
            "provider": getattr(provider_enum, "value", str(provider_enum)),
            "model_name_or_path": model_id,
            "api_key": api_key,
            "api_url": api_url,
            "dimensions": dimensions,
        }
        captured_configs.append(config)
        return config

    async def fake_provider(texts, provider, model, config, metadata=None, dimensions=None):
        _ = (texts, provider, model, metadata, dimensions)
        assert config["api_url"] == "http://127.0.0.1:8081/v1/embeddings?api_key=secret&tenant=alpha"
        return [[0.1, 0.2]]

    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "build_provider_config", fake_build_provider_config)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", fake_provider)

    executor = mod._EndpointEmbeddingExecutor(
        request=_request(),
        current_user=_user(),
        user_metadata=None,
    )

    vectors = await executor.create(
        ["one"],
        provider="local_api",
        model="test-local-model",
        dimensions=None,
    )

    assert vectors == [[0.1, 0.2]]
    assert captured_configs[0]["api_key"] == "local-key"
    assert executor.backend_identity("local_api", "test-local-model") == (
        "http://127.0.0.1:8081/v1/embeddings?tenant=alpha"
    )


@pytest.mark.asyncio
async def test_executor_uses_adapter_registry_before_provider_execution(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "true")
    credentials = FakeCredentials(api_key="adapter-key", source="user")

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class FakeAdapter:
        def embed(self, request):
            assert request["input"] == ["one", "two"]
            assert request["model"] == "text-embedding-3-small"
            assert request["api_key"] == "adapter-key"
            return {
                "data": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]},
                ]
            }

    class FakeRegistry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return FakeAdapter()

    provider_call = AsyncMock(side_effect=AssertionError("provider path should not be called"))
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", provider_call)

    executor = mod._EndpointEmbeddingExecutor(
        request=_request(),
        current_user=_user(),
        user_metadata=None,
    )

    vectors = await executor.create_adapter(
        ["one", "two"],
        provider="openai",
        model="text-embedding-3-small",
        dimensions=None,
    )

    assert isinstance(vectors, EmbeddingExecutorOutput)
    assert vectors.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert vectors.embeddings_from_adapter is True
    assert provider_call.await_count == 0
    credentials.touch_last_used.assert_awaited_once()


@pytest.mark.asyncio
async def test_executor_adapter_runs_sync_embed_in_thread_and_logs_sanitized_failure(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "true")
    credentials = FakeCredentials(api_key="sk-test-secret", source="user")

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class FakeAdapter:
        def embed(self, request):
            assert request["api_key"] == "sk-test-secret"
            raise RuntimeError("adapter failed with sk-test-secret")

    class FakeRegistry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return FakeAdapter()

    to_thread_calls = []

    async def fake_to_thread(func, *args, **kwargs):
        to_thread_calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    log_messages: list[str] = []

    def fake_debug(message, *args, **kwargs):
        _ = (args, kwargs)
        log_messages.append(str(message))

    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())
    monkeypatch.setattr(mod.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(mod.logger, "debug", fake_debug)

    executor = mod._EndpointEmbeddingExecutor(
        request=_request(),
        current_user=_user(),
        user_metadata=None,
    )

    result = await executor.create_adapter(
        ["one"],
        provider="openai",
        model="text-embedding-3-small",
        dimensions=None,
    )

    assert result is None
    assert len(to_thread_calls) == 1
    assert log_messages
    assert all("sk-test-secret" not in message for message in log_messages)


def test_orchestrator_adapter_path_preserves_adapter_vector_scale(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "true")
    credentials = FakeCredentials(api_key="adapter-key", source="user")

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class FakeAdapter:
        def embed(self, request):
            assert request["input"] == "adapter scale"
            assert request["api_key"] == "adapter-key"
            return {"data": [{"embedding": [3.0, 4.0]}]}

    class FakeRegistry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return FakeAdapter()

    provider_call = AsyncMock(side_effect=AssertionError("provider path should not be called"))
    cache_get = AsyncMock(return_value=[0.6, 0.8])
    cache_set = AsyncMock()
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", provider_call)
    monkeypatch.setattr(mod.embedding_cache, "get", cache_get)
    monkeypatch.setattr(mod.embedding_cache, "set", cache_set)

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "adapter scale"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["data"][0]["embedding"] == [3.0, 4.0]
    assert provider_call.await_count == 0
    assert cache_get.await_count == 0
    assert cache_set.await_count == 0
    credentials.touch_last_used.assert_awaited_once()


def test_orchestrator_adapter_enabled_uses_provider_cache_when_no_adapter_serves(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "true")
    credentials = FakeCredentials(api_key="provider-key", source="user")

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class FakeRegistry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return None

    provider_call = AsyncMock(side_effect=AssertionError("provider path should not be called"))
    cache_get = AsyncMock(return_value=[1.0, 0.0])
    cache_set = AsyncMock()
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", provider_call)
    monkeypatch.setattr(mod.embedding_cache, "get", cache_get)
    monkeypatch.setattr(mod.embedding_cache, "set", cache_set)

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "cached adapter miss"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["data"][0]["embedding"] == [1.0, 0.0]
    assert cache_get.await_count == 1
    assert cache_set.await_count == 0
    assert provider_call.await_count == 0
    credentials.touch_last_used.assert_awaited_once()


def test_endpoint_real_orchestrator_applies_fallback_response_headers(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    monkeypatch.setenv("EMBEDDINGS_ALLOW_FALLBACK_WITH_HEADER", "true")
    provider_calls: list[str] = []

    async def fake_create(self, texts, *, provider, model, dimensions):
        _ = (self, texts, model, dimensions)
        provider_calls.append(provider)
        if provider == "openai":
            raise EmbeddingProviderError(
                "provider_unavailable",
                "openai unavailable",
                retryable=True,
                provider=provider,
                model=model,
            )
        return [[0.25, 0.75]]

    monkeypatch.setattr(mod._EndpointEmbeddingExecutor, "create", fake_create)

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "fallback headers"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert provider_calls == ["openai", "huggingface"]
    assert response.headers["X-Embeddings-Provider"] == "huggingface"
    assert response.headers["X-Embeddings-Fallback-From"] == "openai"
    assert response.json()["model"] == "huggingface:sentence-transformers/all-MiniLM-L6-v2"


def test_endpoint_created_error_codes_are_part_of_domain_contract():
    for code in ("circuit_breaker_open", "internal_execution_failure"):
        exc = EmbeddingExecutionError(code, "execution failed")
        assert exc.code == code

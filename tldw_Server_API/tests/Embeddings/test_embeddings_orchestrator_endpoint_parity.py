from __future__ import annotations

import asyncio
import base64
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
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
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
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
        test_client.headers["Authorization"] = (
            f"Bearer {get_settings().SINGLE_USER_API_KEY}"
        )
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
        self.normalized_input = SimpleNamespace(
            texts=["prepared input"],
            total_tokens=total_tokens,
        )
        self.policy_decision = SimpleNamespace(
            fallback_allowed=True,
            fallback_chain=["huggingface"],
        )
        self.execution_plan = SimpleNamespace(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=None,
            fallback_chain=["huggingface"],
            execution_path="legacy",
            cache_namespace=None,
        )
        self.prompt_tokens = total_tokens
        self.total_tokens = total_tokens


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


async def _wait_for_thread_event(
    event: threading.Event,
    *,
    timeout: float = 1.0,
) -> None:
    """Wait for a thread event without using the default executor under test."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set():
        if loop.time() >= deadline:
            raise AssertionError("thread event was not signalled before timeout")
        await asyncio.sleep(0.001)


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
    assert [request.categories for request, _op_id in legacy["rg_reserves"]] == [
        request.categories for request, _op_id in orchestrator["rg_reserves"]
    ]
    assert [actuals for _handle_id, actuals, _op_id in legacy["rg_commits"]] == [
        actuals for _handle_id, actuals, _op_id in orchestrator["rg_commits"]
    ]


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


def test_orchestrator_path_uses_inline_workflow_runner_and_preserves_rg_reservation(client, monkeypatch):
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
        )
    )
    runner_calls: list[tuple[str, object]] = []
    rg_governor = SimpleNamespace(commit=AsyncMock())

    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
        raising=False,
    )

    async def fake_reserve_embedding_rg_tokens(*, request, current_user, token_total):
        del request, current_user
        runner_calls.append(("reserved", token_total))
        return rg_governor, "rg-handle", "rg-op", token_total

    monkeypatch.setattr(
        mod,
        "_reserve_embedding_rg_tokens",
        fake_reserve_embedding_rg_tokens,
        raising=False,
    )

    class RunnerProbe:
        def __init__(self, orchestrator, *, trace_collector=None, pre_execute=None):
            assert trace_collector is None
            self.orchestrator = orchestrator
            self.pre_execute = pre_execute

        async def run(self, raw_input, context):
            runner_calls.append(("runner_started", raw_input))
            prepared = self.orchestrator.prepare(raw_input, context)
            runner_calls.append(("prepared", prepared.normalized_input.total_tokens))
            assert self.pre_execute is not None
            await self.pre_execute(prepared)
            result = await self.orchestrator.execute(prepared)
            runner_calls.append(("executed", result.provider))
            return result

    monkeypatch.setattr(mod, "EmbeddingInlineWorkflowRunner", RunnerProbe, raising=False)

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "huggingface"},
        json={"model": "sentence-transformers/all-MiniLM-L6-v2", "input": "workflow facade"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["data"][0]["index"] == 0
    assert runner_calls == [
        ("runner_started", "workflow facade"),
        ("prepared", 3),
        ("reserved", 3),
        ("executed", "huggingface"),
    ]
    rg_governor.commit.assert_awaited_once_with(
        "rg-handle",
        actuals={"tokens": 3},
        op_id="rg-op",
    )


def test_orchestrator_path_commits_reserved_units_after_execute_failure(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    fake_orchestrator = FakeOrchestrator(
        execute_error=EmbeddingProviderError(
            "provider_unavailable",
            "provider unavailable",
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
    )
    rg_governor = SimpleNamespace(commit=AsyncMock())
    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
        raising=False,
    )

    async def fake_reserve_embedding_rg_tokens(*, request, current_user, token_total):
        del request, current_user
        return rg_governor, "rg-handle", "rg-op", token_total

    monkeypatch.setattr(
        mod,
        "_reserve_embedding_rg_tokens",
        fake_reserve_embedding_rg_tokens,
        raising=False,
    )

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "huggingface"},
        json={"model": "sentence-transformers/all-MiniLM-L6-v2", "input": "workflow failure"},
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    rg_governor.commit.assert_awaited_once_with(
        "rg-handle",
        actuals={"tokens": 3},
        op_id="rg-op",
    )


def test_orchestrator_path_does_not_execute_or_commit_after_rg_denial(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    fake_orchestrator = FakeOrchestrator()
    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
        raising=False,
    )

    async def deny_reservation(*, request, current_user, token_total):
        del request, current_user, token_total
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )

    monkeypatch.setattr(
        mod,
        "_reserve_embedding_rg_tokens",
        deny_reservation,
        raising=False,
    )

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "huggingface"},
        json={"model": "sentence-transformers/all-MiniLM-L6-v2", "input": "workflow denied"},
    )

    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert fake_orchestrator.execute_calls == []


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


def test_orchestrator_resolved_credentials_bypass_shared_provider_cache(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    credentials = FakeCredentials(api_key="cohere-key", source="user")

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    cache_get = AsyncMock(return_value=[0.0, 1.0])
    cache_set = AsyncMock()
    provider_call = AsyncMock(return_value=[[0.0, 1.0]])
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
    assert cache_get.await_count == 0
    assert cache_set.await_count == 0
    assert provider_call.await_count == 1
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

    async def fake_resolve(
        provider,
        current_user,
        request,
        *,
        model=None,
        force_oauth_refresh=False,
        rejected_credentials=None,
    ):
        assert provider == "openai"
        assert model == "text-embedding-3-small"
        assert rejected_credentials is (
            initial_credentials if force_oauth_refresh else None
        )
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


def test_orchestrator_openai_oauth_second_401_maps_to_upstream_502(client, monkeypatch):
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

    async def fake_resolve(
        provider,
        current_user,
        request,
        *,
        model=None,
        force_oauth_refresh=False,
        rejected_credentials=None,
    ):
        assert provider == "openai"
        assert model == "text-embedding-3-small"
        assert rejected_credentials is (
            initial_credentials if force_oauth_refresh else None
        )
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

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"] == "Embedding provider authentication failed"
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
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"index": 1, "embedding": [0.3, 0.4]},
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
async def test_executor_adapter_runs_in_bounded_worker_and_fails_closed_safely(monkeypatch):
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

    bounded_calls = []

    async def fake_bounded_call(
        call,
        *,
        pool,
        exhaustion_message,
        on_cancel_result=None,
    ):
        bounded_calls.append((call, pool, exhaustion_message, on_cancel_result))
        return call()

    log_messages: list[str] = []

    def fake_debug(message, *args, **kwargs):
        _ = (args, kwargs)
        log_messages.append(str(message))

    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())
    monkeypatch.setattr(mod, "await_bounded_sync_call", fake_bounded_call)
    monkeypatch.setattr(mod.logger, "debug", fake_debug)

    executor = mod._EndpointEmbeddingExecutor(
        request=_request(),
        current_user=_user(),
        user_metadata=None,
    )

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await executor.create_adapter(
            ["one"],
            provider="openai",
            model="text-embedding-3-small",
            dimensions=None,
        )

    assert exc_info.value.message == "Embedding provider request failed"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert len(bounded_calls) == 1
    assert bounded_calls[0][1] is mod.SYNC_ADAPTER_CALL_POOL
    assert bounded_calls[0][2] == "Embeddings adapter capacity is exhausted"
    assert callable(bounded_calls[0][3])
    assert log_messages
    assert all("sk-test-secret" not in message for message in log_messages)


@pytest.mark.asyncio
async def test_adapter_cancel_starts_outside_saturated_default_executor_and_touches_after_release(
    monkeypatch,
):
    """A valid late adapter result is owned through exit and touches its exact snapshot."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool

    lifecycle: list[str] = []

    class ReleaseTrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            lifecycle.append("capacity-release")
            super()._release_capacity()

    loop = asyncio.get_running_loop()
    previous_default_executor = getattr(loop, "_default_executor", None)
    default_executor = ThreadPoolExecutor(max_workers=1)
    default_entered = threading.Event()
    default_release = threading.Event()
    adapter_entered = threading.Event()
    adapter_release = threading.Event()
    credentials = FakeCredentials(api_key="late-valid-secret", source="user")
    credentials.touch_last_used.side_effect = lambda: lifecycle.append("touch")
    pool = ReleaseTrackingPool(capacity=1)
    task: asyncio.Task | None = None

    def block_default_executor() -> None:
        default_entered.set()
        assert default_release.wait(timeout=2.0)

    class BlockingAdapter:
        def embed(self, request):
            assert request["api_key"] == "late-valid-secret"
            lifecycle.append("adapter-start")
            adapter_entered.set()
            assert adapter_release.wait(timeout=2.0)
            lifecycle.append("adapter-exit")
            return {"data": [{"index": 0, "embedding": [0.2, 0.8]}]}

    loop.set_default_executor(default_executor)
    default_blocker = loop.run_in_executor(None, block_default_executor)
    monkeypatch.setattr(mod, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    try:
        await _wait_for_thread_event(default_entered)
        task = asyncio.create_task(
            mod._execute_embeddings_adapter_with_oauth_retry(
                BlockingAdapter(),
                ["late result"],
                provider="openai",
                model="text-embedding-3-small",
                dimensions=None,
                credentials=credentials,
                refresh_credentials=None,
            )
        )

        await _wait_for_thread_event(adapter_entered)
        assert not default_release.is_set()
        assert pool.active_count == 1

        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        assert credentials.touch_last_used.await_count == 0

        adapter_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

        assert lifecycle == [
            "adapter-start",
            "adapter-exit",
            "capacity-release",
            "touch",
        ]
        assert credentials.touch_last_used.await_count == 1
        assert pool.active_count == 0
    finally:
        adapter_release.set()
        default_release.set()
        await asyncio.gather(default_blocker, return_exceptions=True)
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        replacement_executor = previous_default_executor or ThreadPoolExecutor()
        loop.set_default_executor(replacement_executor)
        default_executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.asyncio
async def test_adapter_cancel_does_not_touch_malformed_late_result(monkeypatch):
    """Cancellation never records malformed adapter output as credential use."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool

    entered = threading.Event()
    release = threading.Event()
    credentials = FakeCredentials(api_key="malformed-late-secret", source="user")

    class MalformedAdapter:
        def embed(self, _request):
            entered.set()
            assert release.wait(timeout=2.0)
            return {"data": [{"index": 0, "embedding": []}]}

    monkeypatch.setattr(
        mod,
        "SYNC_ADAPTER_CALL_POOL",
        BoundedDaemonPool(capacity=1),
        raising=False,
    )
    task = asyncio.create_task(
        mod._execute_embeddings_adapter_with_oauth_retry(
            MalformedAdapter(),
            ["malformed"],
            provider="openai",
            model="text-embedding-3-small",
            dimensions=None,
            credentials=credentials,
            refresh_credentials=None,
        )
    )
    try:
        await _wait_for_thread_event(entered)
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    credentials.touch_last_used.assert_not_called()


@pytest.mark.asyncio
async def test_adapter_cancel_during_normal_touch_drains_exactly_once(monkeypatch):
    """A valid normal result keeps its single usage touch owned through cancellation."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool

    touch_entered = asyncio.Event()
    touch_release = asyncio.Event()
    credentials = FakeCredentials(api_key="normal-touch-secret", source="user")

    async def blocking_touch() -> None:
        touch_entered.set()
        await touch_release.wait()

    credentials.touch_last_used.side_effect = blocking_touch

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class Adapter:
        def embed(self, _request):
            return {"data": [{"index": 0, "embedding": [0.3, 0.7]}]}

    class Registry:
        def get_adapter(self, _provider):
            return Adapter()

    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: Registry())
    monkeypatch.setattr(
        mod,
        "SYNC_ADAPTER_CALL_POOL",
        BoundedDaemonPool(capacity=1),
    )
    executor = mod._EndpointEmbeddingExecutor(
        request=_request(),
        current_user=_user(),
        user_metadata=None,
    )
    task = asyncio.create_task(
        executor.create_adapter(
            ["normal result"],
            provider="openai",
            model="text-embedding-3-small",
            dimensions=None,
        )
    )
    try:
        await asyncio.wait_for(touch_entered.wait(), timeout=1.0)
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        assert credentials.touch_last_used.await_count == 1

        touch_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        touch_release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert credentials.touch_last_used.await_count == 1


@pytest.mark.asyncio
async def test_adapter_capacity_rejection_is_predispatch_and_never_touches(monkeypatch):
    """Exhausted adapter capacity fails closed without queueing secret-bearing work."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool

    entered = threading.Event()
    release = threading.Event()
    starts: list[str] = []
    pool = BoundedDaemonPool(capacity=1)
    admitted_credentials = FakeCredentials(api_key="admitted-secret", source="user")
    rejected_credentials = FakeCredentials(api_key="rejected-secret-sentinel", source="user")

    class Adapter:
        def embed(self, request):
            starts.append(request["api_key"])
            if request["api_key"] == "admitted-secret":
                entered.set()
                assert release.wait(timeout=2.0)
            return {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

    monkeypatch.setattr(mod, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    admitted = asyncio.create_task(
        mod._execute_embeddings_adapter_with_oauth_retry(
            Adapter(),
            ["admitted"],
            provider="openai",
            model="text-embedding-3-small",
            dimensions=None,
            credentials=admitted_credentials,
            refresh_credentials=None,
        )
    )
    try:
        await _wait_for_thread_event(entered)
        result, failure, _used = await mod._execute_embeddings_adapter_with_oauth_retry(
            Adapter(),
            ["rejected"],
            provider="openai",
            model="text-embedding-3-small",
            dimensions=None,
            credentials=rejected_credentials,
            refresh_credentials=None,
        )
        assert result is None
        assert failure is not None
        assert failure.kind == "execution"
        assert "rejected-secret-sentinel" not in repr(failure)
        assert starts == ["admitted-secret"]
        rejected_credentials.touch_last_used.assert_not_called()

        release.set()
        await asyncio.wait_for(admitted, timeout=1.0)
        await asyncio.sleep(0)
        assert starts == ["admitted-secret"]
        assert pool.active_count == 0
    finally:
        release.set()
        if not admitted.done():
            admitted.cancel()
        await asyncio.gather(admitted, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_oauth_retry_touches_only_valid_refreshed_snapshot(monkeypatch):
    """A cancelled late retry cannot charge the stale OAuth credential snapshot."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError

    retry_entered = threading.Event()
    retry_release = threading.Event()
    initial = FakeCredentials(
        api_key="oauth-stale-secret",
        source="user",
        auth_source="oauth",
    )
    refreshed = FakeCredentials(
        api_key="oauth-refreshed-secret",
        source="user",
        auth_source="oauth",
    )
    calls: list[str] = []

    class OAuthAdapter:
        def embed(self, request):
            calls.append(request["api_key"])
            if request["api_key"] == "oauth-stale-secret":
                raise ChatAuthenticationError(
                    provider="openai-embeddings",
                    message="expired",
                    status_code=401,
                )
            retry_entered.set()
            assert retry_release.wait(timeout=2.0)
            return {"data": [{"index": 0, "embedding": [0.4, 0.6]}]}

    async def refresh_credentials():
        return refreshed

    monkeypatch.setattr(
        mod,
        "SYNC_ADAPTER_CALL_POOL",
        BoundedDaemonPool(capacity=1),
        raising=False,
    )
    task = asyncio.create_task(
        mod._execute_embeddings_adapter_with_oauth_retry(
            OAuthAdapter(),
            ["retry"],
            provider="openai",
            model="text-embedding-3-small",
            dimensions=None,
            credentials=initial,
            refresh_credentials=refresh_credentials,
        )
    )
    try:
        await _wait_for_thread_event(retry_entered)
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        initial.touch_last_used.assert_not_called()
        refreshed.touch_last_used.assert_not_called()

        retry_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        retry_release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert calls == ["oauth-stale-secret", "oauth-refreshed-secret"]
    initial.touch_last_used.assert_not_called()
    refreshed.touch_last_used.assert_awaited_once()


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
            return {"data": [{"index": 0, "embedding": [3.0, 4.0]}]}

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


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
def test_public_adapter_failure_severs_raw_response_log_and_exception_chain(
    client,
    monkeypatch,
    orchestrator_enabled,
):
    from loguru import logger

    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError

    sentinel = f"public-{orchestrator_enabled}-raw-key-endpoint-body-sentinel"
    credentials = FakeCredentials(
        api_key="public-runtime-key",
        source="user",
        credential_fields={"base_url": "https://public-runtime.example/v1"},
        app_config={"openai_api": {"api_base_url": "https://public-runtime.example/v1"}},
    )
    adapter_calls = []
    captured_http_errors = []
    log_records = []

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class FakeAdapter:
        def embed(self, request):
            adapter_calls.append(request)
            try:
                raise RuntimeError(sentinel)
            except RuntimeError as exc:
                raise ChatProviderError(provider="openai", message=sentinel) from exc

    class FakeRegistry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return FakeAdapter()

    original_mapper = mod._embedding_domain_error_to_http

    def capture_mapper(exc):
        mapped = original_mapper(exc)
        if isinstance(mapped, HTTPException):
            captured_http_errors.append(mapped)
        return mapped

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())
    monkeypatch.setattr(mod, "_embedding_domain_error_to_http", capture_mapper)
    sink_id = logger.add(log_records.append, format="{message} {extra}")
    try:
        response = client.post(
            "/api/v1/embeddings",
            json={"model": "text-embedding-3-small", "input": "public failure"},
        )
    finally:
        logger.remove(sink_id)

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json() == {"detail": "Embedding provider request failed"}
    assert len(adapter_calls) == 1
    assert sentinel not in response.text
    assert sentinel not in "".join(map(str, log_records))
    assert captured_http_errors
    for error in captured_http_errors:
        assert error.__cause__ is None
        assert error.__context__ is None


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
def test_public_google_unsafe_model_is_bounded_before_dispatch_or_fallback(
    client,
    monkeypatch,
    orchestrator_enabled,
):
    import tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter as google_module
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    unsafe_model = "../credential-admin-sentinel"
    credentials = FakeCredentials(
        api_key="google-runtime-key",
        source="user",
        credential_fields={"base_url": "https://google-runtime.example/v1"},
        app_config={"google_api": {"api_base_url": "https://google-runtime.example/v1"}},
    )

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class FakeRegistry:
        def get_adapter(self, provider):
            assert provider == "google"
            return GoogleEmbeddingsAdapter()

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())
    monkeypatch.setattr(
        google_module,
        "create_client",
        lambda **_kwargs: pytest.fail("unsafe model must not dispatch HTTP"),
    )
    legacy_fallback = AsyncMock(side_effect=AssertionError("unsafe model must not fall back"))
    monkeypatch.setattr(mod, "create_embeddings_batch_async", legacy_fallback)

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "google"},
        json={"model": unsafe_model, "input": "public unsafe model"},
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json() == {"detail": "Invalid embedding provider request"}
    assert unsafe_model not in response.text
    assert legacy_fallback.await_count == 0


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
def test_public_adapter_unavailable_falls_back_once_with_same_key_and_endpoint(
    client,
    monkeypatch,
    orchestrator_enabled,
):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.LLM_Calls.providers.base import (
        EmbeddingsAdapterUnavailableError,
    )

    endpoint = "https://fallback-runtime.example/v1"
    credentials = FakeCredentials(
        api_key="fallback-runtime-key",
        source="user",
        credential_fields={"base_url": endpoint},
        app_config={"openai_api": {"api_base_url": endpoint}},
    )
    adapter_calls = []
    fallback_calls = []

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class FakeAdapter:
        def embed(self, request):
            adapter_calls.append((request["api_key"], request["base_url"]))
            raise EmbeddingsAdapterUnavailableError("native path disabled before dispatch")

    class FakeRegistry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return FakeAdapter()

    async def fake_provider(
        texts,
        provider,
        model_id,
        config,
        metadata=None,
        dimensions=None,
    ):
        del metadata, dimensions
        fallback_calls.append(
            (
                texts,
                provider,
                model_id,
                config["api_key"],
                config["api_url"],
                config.get("_runtime_credentials_private"),
            )
        )
        return [[0.1, 0.2]]

    async def fake_legacy_batch(
        texts,
        provider,
        model_id=None,
        dimensions=None,
        api_key=None,
        api_url=None,
        metadata=None,
        cache_scope_sensitive=False,
    ):
        del dimensions, metadata
        fallback_calls.append(
            (texts, provider, model_id, api_key, api_url, cache_scope_sensitive)
        )
        return [[0.1, 0.2]]

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", fake_provider)
    monkeypatch.setattr(mod, "create_embeddings_batch_async", fake_legacy_batch)
    monkeypatch.setattr(mod.embedding_cache, "get", AsyncMock(return_value=None))
    monkeypatch.setattr(mod.embedding_cache, "set", AsyncMock())

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "fallback exactly once"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert adapter_calls == [("fallback-runtime-key", endpoint)]
    assert len(fallback_calls) == 1
    expected_tail = ("fallback-runtime-key", endpoint, True)
    assert fallback_calls[0][3:] == expected_tail


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
@pytest.mark.parametrize(
    ("provider", "environment", "default_base"),
    [
        (
            "google",
            "GOOGLE_GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1",
        ),
        (
            "huggingface",
            "HUGGINGFACE_INFERENCE_BASE_URL",
            "https://api-inference.huggingface.co/models",
        ),
    ],
)
def test_public_key_only_native_unavailable_fallback_keeps_adapter_default_at_http_boundary(
    client,
    monkeypatch,
    orchestrator_enabled,
    provider,
    environment,
    default_base,
):
    """Both endpoint engines preserve the exact default chosen before fallback."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter import (
        HuggingFaceEmbeddingsAdapter,
    )

    credentials = FakeCredentials(
        api_key=f"key-{provider}",
        source="server_default",
        credential_fields={},
        app_config=None,
    )
    calls = []

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class FakeRegistry:
        def get_adapter(self, selected):
            assert selected == provider
            return {
                "google": GoogleEmbeddingsAdapter(),
                "huggingface": HuggingFaceEmbeddingsAdapter(),
            }[selected]

    class FakeResponse:
        status_code = 200

        def json(self):
            if provider == "google":
                return {"embeddings": [{"values": [0.6, 0.8]}]}
            return [[0.6, 0.8]]

        async def aclose(self):
            return None

    async def fake_afetch(**kwargs):
        calls.append(kwargs)
        return FakeResponse()

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", raising=False)
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_HUGGINGFACE", raising=False)
    monkeypatch.setenv(environment, f"https://late-{provider}-env.invalid")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())
    monkeypatch.setattr(mod.connection_manager, "get_session", AsyncMock(return_value=object()))
    monkeypatch.setattr(mod, "_http_afetch", fake_afetch)
    monkeypatch.setattr(mod.embedding_cache, "get", AsyncMock(return_value=None))
    monkeypatch.setattr(mod.embedding_cache, "set", AsyncMock())

    model = "text-embedding-004" if provider == "google" else "org/runtime-model"
    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": provider},
        json={"model": model, "input": f"text-{provider}"},
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    assert len(calls) == 1
    if provider == "google":
        assert calls[0]["url"] == (
            f"{default_base}/models/text-embedding-004:batchEmbedContents"
        )
        assert calls[0]["headers"]["x-goog-api-key"] == "key-google"
    else:
        assert calls[0]["url"] == f"{default_base}/org/runtime-model"
        assert calls[0]["headers"]["Authorization"] == "Bearer key-huggingface"
    credentials.touch_last_used.assert_awaited_once()


@pytest.mark.concurrent
@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
def test_concurrent_public_keyless_local_and_keyed_remote_hf_stay_isolated(
    client,
    monkeypatch,
    orchestrator_enabled,
):
    """The public adapter boundary keeps local and remote HF execution modes paired."""
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter as hf_mod
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter import (
        HuggingFaceEmbeddingsAdapter,
    )

    credentials = {
        "local": FakeCredentials(api_key=None, source="none"),
        "remote": FakeCredentials(api_key="key-remote", source="server_default"),
    }
    local_calls = []
    remote_calls = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class FakeBreaker:
        async def call_async(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

    async def fake_resolve(_provider, _current_user, request, **_kwargs):
        return credentials[request.headers["x-test-execution-mode"]]

    class FakeRegistry:
        def get_adapter(self, selected):
            assert selected == "huggingface"
            return HuggingFaceEmbeddingsAdapter()

    def mark_arrived() -> None:
        with lock:
            if local_calls and remote_calls:
                both_arrived.set()

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [[0.0, 1.0]]

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, headers=None, json=None, **_kwargs):
            remote_calls.append((url, headers["Authorization"], json["inputs"]))
            mark_arrived()
            if not release.wait(10):
                raise TimeoutError("remote HF boundary was not released")
            return FakeResponse()

    async def fake_local_batcher(**kwargs):
        local_calls.append(kwargs)
        mark_arrived()
        if not await asyncio.to_thread(release.wait, 10):
            raise TimeoutError("local HF boundary was not released")
        return [[1.0, 0.0]]

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_HUGGINGFACE", "1")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.delenv("HUGGINGFACE_INFERENCE_BASE_URL", raising=False)
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())
    monkeypatch.setattr(mod, "get_or_create_circuit_breaker", lambda _provider: FakeBreaker())
    monkeypatch.setattr(mod, "batching_create_embeddings_batch_async", fake_local_batcher)
    monkeypatch.setattr(hf_mod, "create_client", lambda **_kwargs: FakeClient())
    monkeypatch.setattr(mod.embedding_cache, "get", AsyncMock(return_value=None))
    monkeypatch.setattr(mod.embedding_cache, "set", AsyncMock())

    def post(label):
        return client.post(
            "/api/v1/embeddings",
            headers={
                "x-provider": "huggingface",
                "x-test-execution-mode": label,
            },
            json={"model": f"org/{label}-model", "input": label},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        local = executor.submit(post, "local")
        remote = executor.submit(post, "remote")
        try:
            assert both_arrived.wait(10)
        finally:
            release.set()
        responses = (local.result(timeout=10), remote.result(timeout=10))

    assert [response.status_code for response in responses] == [200, 200]
    assert len(local_calls) == 1
    assert local_calls[0]["model_id_override"] == "huggingface:org/local-model"
    assert remote_calls == [
        (
            "https://api-inference.huggingface.co/models/org/remote-model",
            "Bearer key-remote",
            "remote",
        )
    ]


@pytest.mark.concurrent
@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
def test_concurrent_public_endpoint_modes_keep_resolved_adapter_snapshots_paired(
    client,
    monkeypatch,
    orchestrator_enabled,
):
    """Both endpoint engines must pass one immutable credential snapshot per call."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    credentials = {
        label: FakeCredentials(
            api_key=f"key-{label}",
            source="user",
            credential_fields={"base_url": f"https://endpoint-{label}.example/v1"},
            app_config={
                "openai_api": {
                    "api_base_url": f"https://endpoint-{label}.example/v1",
                    "organization": f"org-{label}",
                }
            },
        )
        for label in ("alpha", "beta")
    }
    calls: list[tuple[str, str, str, str, str]] = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class FakeBreaker:
        async def call_async(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

    async def fake_resolve(_provider, _current_user, request, **_kwargs):
        return credentials[request.headers["x-test-credential-label"]]

    class _Adapter:
        def embed(self, request):
            app_config = request["app_config"]["openai_api"]
            call = (
                request["api_key"],
                request["base_url"],
                app_config["organization"],
                request["model"],
                request["input"],
            )
            with lock:
                calls.append(call)
                if len(calls) == 2:
                    both_arrived.set()
            if not release.wait(10):
                raise TimeoutError("concurrent public embedding calls were not released")
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    class _Registry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return _Adapter()

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "https://ambient-attacker.example/v1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: _Registry())

    def _post(label: str):
        return client.post(
            "/api/v1/embeddings",
            headers={"x-test-credential-label": label},
            json={"model": f"model-{label}", "input": label},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        alpha = executor.submit(_post, "alpha")
        beta = executor.submit(_post, "beta")
        try:
            assert both_arrived.wait(10)
        finally:
            release.set()
        responses = (alpha.result(timeout=10), beta.result(timeout=10))

    assert [response.status_code for response in responses] == [200, 200]
    assert set(calls) == {
        (
            "key-alpha",
            "https://endpoint-alpha.example/v1",
            "org-alpha",
            "model-alpha",
            "alpha",
        ),
        (
            "key-beta",
            "https://endpoint-beta.example/v1",
            "org-beta",
            "model-beta",
            "beta",
        ),
    }
    assert "ambient-attacker" not in repr(calls)
    credentials["alpha"].touch_last_used.assert_awaited_once()
    credentials["beta"].touch_last_used.assert_awaited_once()


@pytest.mark.concurrent
@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
def test_concurrent_server_overrides_reach_provider_as_atomic_key_endpoint_snapshots(
    client,
    monkeypatch,
    orchestrator_enabled,
):
    """The real fallback resolver keeps consolidated server credentials atomic."""
    import tldw_Server_API.app.core.AuthNZ.byok_helpers as byok_helpers
    import tldw_Server_API.app.core.AuthNZ.byok_runtime as byok_runtime
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
        LLMProviderOverride,
        get_llm_provider_overrides_snapshot,
        set_llm_provider_overrides_cache_for_tests,
    )

    overrides = {
        provider: LLMProviderOverride(
            provider=provider,
            api_key=f"override-key-{provider}",
            credential_fields={"base_url": f"https://example.com/{provider}/custom"},
        )
        for provider in ("openai", "google")
    }
    calls = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class FakeBreaker:
        async def call_async(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

    async def fake_provider(texts, provider, model_id, config, **_kwargs):
        call = (
            provider,
            config["api_key"],
            config["api_url"],
            model_id,
            texts[0],
        )
        with lock:
            calls.append(call)
            if len(calls) == 2:
                both_arrived.set()
        if not await asyncio.to_thread(release.wait, 10):
            raise TimeoutError("concurrent server override calls were not released")
        return [[1.0, 0.0]]

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "0")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(
        byok_helpers,
        "resolve_byok_base_url_allowlist",
        lambda: {"openai", "google"},
    )
    monkeypatch.setattr(mod, "get_or_create_circuit_breaker", lambda _provider: FakeBreaker())
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", fake_provider)
    monkeypatch.setattr(mod.embedding_cache, "get", AsyncMock(return_value=None))
    monkeypatch.setattr(mod.embedding_cache, "set", AsyncMock())
    original_overrides = get_llm_provider_overrides_snapshot()
    set_llm_provider_overrides_cache_for_tests(overrides)

    def post(provider):
        model = {
            "openai": "text-embedding-3-small",
            "google": "text-embedding-004",
        }[provider]
        return client.post(
            "/api/v1/embeddings",
            headers={"x-provider": provider},
            json={"model": model, "input": f"text-{provider}"},
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            openai = executor.submit(post, "openai")
            google = executor.submit(post, "google")
            try:
                assert both_arrived.wait(10)
            finally:
                release.set()
            responses = (openai.result(timeout=10), google.result(timeout=10))
    finally:
        release.set()
        set_llm_provider_overrides_cache_for_tests(original_overrides)

    assert [response.status_code for response in responses] == [200, 200]
    assert set(calls) == {
        (
            "openai",
            "override-key-openai",
            "https://example.com/openai/custom",
            "text-embedding-3-small",
            "text-openai",
        ),
        (
            "google",
            "override-key-google",
            "https://example.com/google/custom",
            "text-embedding-004",
            "text-google",
        ),
    }


@pytest.mark.concurrent
@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
def test_resolved_keys_never_share_cached_vectors_sequentially_or_concurrently(
    client,
    monkeypatch,
    orchestrator_enabled,
):
    """Same endpoint/model/text with different keys must always cross the provider boundary."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    credentials = {
        label: FakeCredentials(
            api_key=f"key-{label}",
            source="user",
            credential_fields={"base_url": "https://shared-tenant.example/v1"},
            app_config={"openai_api": {"api_base_url": "https://shared-tenant.example/v1"}},
        )
        for label in ("alpha", "beta", "gamma", "delta")
    }
    vectors = {
        "key-alpha": [1.0, 0.0],
        "key-beta": [0.0, 1.0],
        "key-gamma": [0.8, 0.6],
        "key-delta": [0.6, 0.8],
    }
    provider_calls = []
    cache_values = {}
    cache_gets = []
    cache_sets = []
    lock = threading.Lock()
    concurrent_arrived = threading.Event()
    release = threading.Event()

    class FakeBreaker:
        async def call_async(self, func, *args, **kwargs):
            return await func(*args, **kwargs)

    async def fake_resolve(_provider, _current_user, request, **_kwargs):
        return credentials[request.headers["x-test-credential-label"]]

    async def cache_get(key):
        cache_gets.append(key)
        return cache_values.get(key)

    async def cache_set(key, value):
        cache_sets.append(key)
        cache_values[key] = value

    async def fake_provider(texts, provider, model_id, config, **_kwargs):
        key = config["api_key"]
        call = (key, config["api_url"], model_id, texts[0])
        with lock:
            provider_calls.append(call)
            concurrent_count = sum(item[0] in {"key-gamma", "key-delta"} for item in provider_calls)
            if concurrent_count == 2:
                concurrent_arrived.set()
        if key in {"key-gamma", "key-delta"} and not await asyncio.to_thread(
            release.wait, 10
        ):
            raise TimeoutError("concurrent cache-isolation calls were not released")
        return [vectors[key]]

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "0")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_or_create_circuit_breaker", lambda _provider: FakeBreaker())
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", fake_provider)
    monkeypatch.setattr(mod.embedding_cache, "get", cache_get)
    monkeypatch.setattr(mod.embedding_cache, "set", cache_set)

    def post(label):
        return client.post(
            "/api/v1/embeddings",
            headers={"x-test-credential-label": label},
            json={"model": "shared-model", "input": "same text"},
        )

    alpha_response = post("alpha")
    beta_response = post("beta")
    with ThreadPoolExecutor(max_workers=2) as executor:
        gamma = executor.submit(post, "gamma")
        delta = executor.submit(post, "delta")
        try:
            assert concurrent_arrived.wait(10)
        finally:
            release.set()
        gamma_response = gamma.result(timeout=10)
        delta_response = delta.result(timeout=10)

    assert [
        alpha_response.status_code,
        beta_response.status_code,
        gamma_response.status_code,
        delta_response.status_code,
    ] == [200, 200, 200, 200]
    assert len(provider_calls) == 4
    assert {call[0] for call in provider_calls} == {
        "key-alpha",
        "key-beta",
        "key-gamma",
        "key-delta",
    }
    assert cache_gets == []
    assert cache_sets == []


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
def test_public_malformed_cached_vector_is_replaced_from_provider_boundary(
    client,
    monkeypatch,
    orchestrator_enabled,
):
    """Malformed legacy cache entries are misses and never reach normalization."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    credentials = FakeCredentials(api_key=None, source="none")
    provider_calls = []

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    async def fake_provider(texts, provider, model_id, config, **_kwargs):
        provider_calls.append((texts, provider, model_id, config))
        return [[1.0, 0.0]]

    cache_get = AsyncMock(return_value=[float("nan"), 0.0])
    cache_set = AsyncMock()
    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "0")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", fake_provider)
    monkeypatch.setattr(mod.embedding_cache, "get", cache_get)
    monkeypatch.setattr(mod.embedding_cache, "set", cache_set)

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "huggingface"},
        json={"model": "org/local-model", "input": "same text"},
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["data"][0]["embedding"] == [1.0, 0.0]
    assert len(provider_calls) == 1
    cache_get.assert_awaited_once()
    cache_set.assert_awaited_once()


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
@pytest.mark.parametrize("provider", ["google", "huggingface"])
def test_public_direct_provider_boundary_rejects_malformed_vectors(
    client,
    monkeypatch,
    orchestrator_enabled,
    provider,
):
    """Non-adapter Google and HF responses cross the same strict shape boundary."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    credentials = FakeCredentials(api_key=f"key-{provider}", source="server_default")

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    async def malformed_provider(*_args, **_kwargs):
        return [[]]

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "0")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", malformed_provider)

    model = "text-embedding-004" if provider == "google" else "org/remote-model"
    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": provider},
        json={"model": model, "input": "malformed"},
    )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY, response.text
    credentials.touch_last_used.assert_not_awaited()


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
@pytest.mark.parametrize("malformed", [False, True], ids=("out-of-order-valid", "duplicate-invalid"))
def test_public_openai_fallback_validates_and_reorders_explicit_wire_rows(
    client,
    monkeypatch,
    orchestrator_enabled,
    malformed,
):
    """Adapter-unavailable public calls preserve indexed OpenAI fallback ordering."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.LLM_Calls.providers.base import (
        EmbeddingsAdapterUnavailableError,
    )

    endpoint = "https://explicit-openai.example/v1"
    credentials = FakeCredentials(
        api_key="explicit-runtime-key",
        source="user",
        credential_fields={"base_url": endpoint},
        app_config={"openai_api": {"api_base_url": endpoint}},
    )
    calls = []
    adapter_calls = []

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class UnavailableAdapter:
        def embed(self, request):
            raw_input = request["input"]
            adapter_calls.append(
                (
                    request["api_key"],
                    request["base_url"],
                    tuple(raw_input) if isinstance(raw_input, list) else (raw_input,),
                )
            )
            raise EmbeddingsAdapterUnavailableError("native path unavailable before dispatch")

    class Registry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return UnavailableAdapter()

    class Response:
        status_code = 200

        def json(self):
            first_index, second_index = (0, 0) if malformed else (1, 0)
            return {
                "data": [
                    {"index": first_index, "embedding": [0.0, 1.0]},
                    {"index": second_index, "embedding": [1.0, 0.0]},
                ]
            }

        def close(self):
            return None

    def fake_fetch(**kwargs):
        calls.append(kwargs)
        return Response()

    async def fail_global_batcher(**_kwargs):
        raise AssertionError("resolved credentials must bypass the global request batcher")

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: Registry())
    monkeypatch.setattr(
        mod,
        "batching_create_embeddings_batch_async",
        fail_global_batcher,
    )
    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", fake_fetch)

    nonce = uuid.uuid4().hex
    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "openai"},
        json={
            "model": "text-embedding-3-small",
            "input": [f"one-{nonce}", f"two-{nonce}"],
        },
    )

    expected_status = status.HTTP_502_BAD_GATEWAY if malformed else status.HTTP_200_OK
    assert response.status_code == expected_status, response.text
    assert len(calls) == 1
    assert adapter_calls == [
        (
            "explicit-runtime-key",
            endpoint,
            (f"one-{nonce}", f"two-{nonce}"),
        )
    ]
    assert calls[0]["url"] == f"{endpoint}/embeddings"
    assert calls[0]["headers"]["Authorization"] == "Bearer explicit-runtime-key"
    if malformed:
        credentials.touch_last_used.assert_not_awaited()
    else:
        assert [item["embedding"] for item in response.json()["data"]] == [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        credentials.touch_last_used.assert_awaited_once()


@pytest.mark.concurrent
@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
def test_concurrent_public_openai_fallback_keeps_malformed_and_valid_calls_isolated(
    client,
    monkeypatch,
    orchestrator_enabled,
):
    """Unavailable-adapter fallbacks keep concurrent OpenAI responses isolated."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.LLM_Calls.providers.base import (
        EmbeddingsAdapterUnavailableError,
    )

    credentials = {
        label: FakeCredentials(
            api_key=f"key-{label}",
            source="user",
            credential_fields={"base_url": f"https://{label}.example/v1"},
            app_config={"openai_api": {"api_base_url": f"https://{label}.example/v1"}},
        )
        for label in ("malformed", "valid")
    }
    calls = []
    adapter_calls = []
    calls_lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    async def fake_resolve(_provider, _current_user, request, **_kwargs):
        return credentials[request.headers["x-test-credential-label"]]

    class UnavailableAdapter:
        def embed(self, request):
            raw_input = request["input"]
            with calls_lock:
                adapter_calls.append(
                    (
                        request["api_key"],
                        request["base_url"],
                        tuple(raw_input) if isinstance(raw_input, list) else (raw_input,),
                    )
                )
            raise EmbeddingsAdapterUnavailableError("native path unavailable before dispatch")

    class Registry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return UnavailableAdapter()

    class Response:
        status_code = 200

        def __init__(self, label):
            self.label = label

        def json(self):
            indices = (0, 0) if self.label == "malformed" else (1, 0)
            return {
                "data": [
                    {"index": indices[0], "embedding": [0.0, 1.0]},
                    {"index": indices[1], "embedding": [1.0, 0.0]},
                ]
            }

        def close(self):
            return None

    def fake_fetch(**kwargs):
        authorization = kwargs["headers"]["Authorization"]
        label = authorization.removeprefix("Bearer key-")
        with calls_lock:
            calls.append((label, authorization, kwargs["url"], tuple(kwargs["json"]["input"])))
            if len(calls) == 2:
                both_arrived.set()
        if not release.wait(10):
            raise TimeoutError("concurrent explicit OpenAI calls were not released")
        return Response(label)

    async def fail_global_batcher(**_kwargs):
        raise AssertionError("resolved credentials must bypass the global request batcher")

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: Registry())
    monkeypatch.setattr(mod, "batching_create_embeddings_batch_async", fail_global_batcher)
    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", fake_fetch)

    def post(label):
        return client.post(
            "/api/v1/embeddings",
            headers={"x-provider": "openai", "x-test-credential-label": label},
            json={
                "model": "text-embedding-3-small",
                "input": [f"{label}-one", f"{label}-two"],
            },
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        malformed_future = executor.submit(post, "malformed")
        valid_future = executor.submit(post, "valid")
        try:
            assert both_arrived.wait(10)
        finally:
            release.set()
        malformed_response = malformed_future.result(timeout=10)
        valid_response = valid_future.result(timeout=10)

    assert malformed_response.status_code == status.HTTP_502_BAD_GATEWAY, malformed_response.text
    assert valid_response.status_code == status.HTTP_200_OK, valid_response.text
    assert [item["embedding"] for item in valid_response.json()["data"]] == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    assert set(calls) == {
        (
            "malformed",
            "Bearer key-malformed",
            "https://malformed.example/v1/embeddings",
            ("malformed-one", "malformed-two"),
        ),
        (
            "valid",
            "Bearer key-valid",
            "https://valid.example/v1/embeddings",
            ("valid-one", "valid-two"),
        ),
    }
    assert set(adapter_calls) == {
        (
            "key-malformed",
            "https://malformed.example/v1",
            ("malformed-one", "malformed-two"),
        ),
        (
            "key-valid",
            "https://valid.example/v1",
            ("valid-one", "valid-two"),
        ),
    }
    credentials["malformed"].touch_last_used.assert_not_awaited()
    credentials["valid"].touch_last_used.assert_awaited_once()


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
@pytest.mark.parametrize("malformed", [False, True], ids=("valid", "malformed"))
def test_public_local_api_fallback_binds_runtime_key_and_endpoint(
    client,
    monkeypatch,
    orchestrator_enabled,
    malformed,
):
    """An unavailable adapter preserves one protected local credential snapshot."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.LLM_Calls.providers.base import (
        EmbeddingsAdapterUnavailableError,
    )

    endpoint = "https://local-runtime.example/embeddings"
    credentials = FakeCredentials(
        api_key="local-runtime-key",
        source="user",
        credential_fields={"base_url": endpoint},
        app_config={"local_api": {"api_url": endpoint}},
    )
    calls = []
    adapter_calls = []

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class UnavailableAdapter:
        def embed(self, request):
            adapter_calls.append(
                (request["api_key"], request["base_url"], request["input"])
            )
            raise EmbeddingsAdapterUnavailableError("native path unavailable before dispatch")

    class Registry:
        def get_adapter(self, provider):
            assert provider == "local_api"
            return UnavailableAdapter()

    class Response:
        status_code = 200

        def json(self):
            return {"embeddings": [[] if malformed else [0.25, 0.75]]}

        def close(self):
            return None

    def fake_fetch(**kwargs):
        calls.append(kwargs)
        return Response()

    async def fail_global_batcher(**_kwargs):
        raise AssertionError("resolved credentials must bypass the global request batcher")

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: Registry())
    monkeypatch.setattr(mod, "batching_create_embeddings_batch_async", fail_global_batcher)
    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", fake_fetch)

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "local_api"},
        json={"model": "local-model", "input": "protected local"},
    )

    expected_status = status.HTTP_502_BAD_GATEWAY if malformed else status.HTTP_200_OK
    assert response.status_code == expected_status, response.text
    assert len(calls) == 1
    assert adapter_calls == [
        ("local-runtime-key", endpoint, "protected local")
    ]
    assert calls[0]["url"] == endpoint
    assert calls[0]["headers"]["Authorization"] == "Bearer local-runtime-key"
    assert calls[0]["json"] == {"texts": ["protected local"], "model": "local-model"}
    if malformed:
        credentials.touch_last_used.assert_not_awaited()
    else:
        assert response.json()["data"][0]["embedding"] == pytest.approx(
            [0.31622777, 0.9486833]
        )
        credentials.touch_last_used.assert_awaited_once()


@pytest.mark.concurrent
@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
def test_concurrent_public_local_api_fallback_keeps_key_endpoint_pairs_isolated(
    client,
    monkeypatch,
    orchestrator_enabled,
):
    """Unavailable-adapter local fallbacks cannot exchange keys or endpoints."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.LLM_Calls.providers.base import (
        EmbeddingsAdapterUnavailableError,
    )

    credentials = {
        label: FakeCredentials(
            api_key=f"local-key-{label}",
            source="user",
            credential_fields={"base_url": f"https://local-{label}.example/embeddings"},
            app_config={
                "local_api": {"api_url": f"https://local-{label}.example/embeddings"}
            },
        )
        for label in ("alpha", "beta")
    }
    calls = []
    adapter_calls = []
    calls_lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    async def fake_resolve(_provider, _current_user, request, **_kwargs):
        return credentials[request.headers["x-test-credential-label"]]

    class UnavailableAdapter:
        def embed(self, request):
            with calls_lock:
                adapter_calls.append(
                    (request["api_key"], request["base_url"], request["input"])
                )
            raise EmbeddingsAdapterUnavailableError("native path unavailable before dispatch")

    class Registry:
        def get_adapter(self, provider):
            assert provider == "local_api"
            return UnavailableAdapter()

    class Response:
        status_code = 200

        def __init__(self, label):
            self.label = label

        def json(self):
            return {"embeddings": [[1.0, 0.0] if self.label == "alpha" else [0.0, 1.0]]}

        def close(self):
            return None

    def fake_fetch(**kwargs):
        authorization = kwargs["headers"]["Authorization"]
        label = authorization.removeprefix("Bearer local-key-")
        with calls_lock:
            calls.append((label, authorization, kwargs["url"], kwargs["json"]["texts"][0]))
            if len(calls) == 2:
                both_arrived.set()
        if not release.wait(10):
            raise TimeoutError("concurrent protected local calls were not released")
        return Response(label)

    async def fail_global_batcher(**_kwargs):
        raise AssertionError("resolved credentials must bypass the global request batcher")

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: Registry())
    monkeypatch.setattr(mod, "batching_create_embeddings_batch_async", fail_global_batcher)
    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch", fake_fetch)

    def post(label):
        return client.post(
            "/api/v1/embeddings",
            headers={
                "x-provider": "local_api",
                "x-test-credential-label": label,
            },
            json={"model": "local-model", "input": label},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        alpha_future = executor.submit(post, "alpha")
        beta_future = executor.submit(post, "beta")
        try:
            assert both_arrived.wait(10)
        finally:
            release.set()
        alpha_response = alpha_future.result(timeout=10)
        beta_response = beta_future.result(timeout=10)

    assert alpha_response.status_code == status.HTTP_200_OK, alpha_response.text
    assert beta_response.status_code == status.HTTP_200_OK, beta_response.text
    assert alpha_response.json()["data"][0]["embedding"] == [1.0, 0.0]
    assert beta_response.json()["data"][0]["embedding"] == [0.0, 1.0]
    assert set(calls) == {
        (
            "alpha",
            "Bearer local-key-alpha",
            "https://local-alpha.example/embeddings",
            "alpha",
        ),
        (
            "beta",
            "Bearer local-key-beta",
            "https://local-beta.example/embeddings",
            "beta",
        ),
    }
    assert set(adapter_calls) == {
        (
            "local-key-alpha",
            "https://local-alpha.example/embeddings",
            "alpha",
        ),
        (
            "local-key-beta",
            "https://local-beta.example/embeddings",
            "beta",
        ),
    }
    credentials["alpha"].touch_last_used.assert_awaited_once()
    credentials["beta"].touch_last_used.assert_awaited_once()


@pytest.mark.concurrent
@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
def test_concurrent_none_adapter_result_cannot_borrow_unavailable_fallback_permission(
    client,
    monkeypatch,
    orchestrator_enabled,
):
    """Only the call that raised the pre-dispatch sentinel may enter fallback."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.LLM_Calls.providers.base import (
        EmbeddingsAdapterUnavailableError,
    )

    credentials = FakeCredentials(
        api_key="runtime-key",
        source="user",
        credential_fields={"base_url": "https://runtime.example/v1"},
        app_config={"openai_api": {"api_base_url": "https://runtime.example/v1"}},
    )
    malformed_arrived = threading.Event()
    unavailable_arrived = threading.Event()
    release = threading.Event()
    fallback_calls = []

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class FakeAdapter:
        def embed(self, request):
            if request["input"] == "malformed":
                malformed_arrived.set()
                if not unavailable_arrived.wait(10):
                    raise TimeoutError("unavailable adapter call did not arrive")
                if not release.wait(10):
                    raise TimeoutError("malformed adapter call was not released")
                return None
            unavailable_arrived.set()
            raise EmbeddingsAdapterUnavailableError("disabled before dispatch")

    class FakeRegistry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return FakeAdapter()

    async def fallback(*args, **kwargs):
        texts = kwargs.get("texts") or args[0]
        fallback_calls.append(texts[0])
        return [[0.2, 0.8]]

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", fallback)
    monkeypatch.setattr(mod, "create_embeddings_batch_async", fallback)

    def post(text):
        return client.post(
            "/api/v1/embeddings",
            json={"model": "text-embedding-3-small", "input": text},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        malformed = executor.submit(post, "malformed")
        assert malformed_arrived.wait(10)
        unavailable = executor.submit(post, "unavailable")
        try:
            assert unavailable_arrived.wait(10)
        finally:
            release.set()
        malformed_response = malformed.result(timeout=10)
        unavailable_response = unavailable.result(timeout=10)

    assert malformed_response.status_code == status.HTTP_502_BAD_GATEWAY
    assert unavailable_response.status_code == status.HTTP_200_OK
    assert fallback_calls == ["unavailable"]


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
@pytest.mark.parametrize(
    ("inputs", "vectors"),
    [
        pytest.param(["one"], [[]], id="empty"),
        pytest.param(["one"], [[1.0, "not-a-number"]], id="nonnumeric"),
        pytest.param(["one"], [[1.0, float("nan")]], id="nonfinite"),
        pytest.param(["one"], [[1.0, [2.0]]], id="nested"),
        pytest.param(["one"], [[True, 1.0]], id="boolean"),
        pytest.param(["one", "two"], [[1.0, 2.0], [3.0]], id="mixed-width"),
    ],
)
def test_public_adapter_boundary_rejects_malformed_vector_shapes(
    client,
    monkeypatch,
    orchestrator_enabled,
    inputs,
    vectors,
):
    """Both endpoint engines reject malformed vectors before normalization or caching."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    credentials = FakeCredentials(api_key="runtime-key", source="user")

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class FakeAdapter:
        def embed(self, _request):
            return {
                "data": [
                    {"index": index, "embedding": vector}
                    for index, vector in enumerate(vectors)
                ]
            }

    class FakeRegistry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return FakeAdapter()

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": inputs},
    )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY, response.text
    credentials.touch_last_used.assert_not_awaited()


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
@pytest.mark.parametrize(
    "adapter_data",
    [
        pytest.param(
            [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 0, "embedding": [0.0, 1.0]},
            ],
            id="duplicate-index",
        ),
        pytest.param(
            [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 2, "embedding": [0.0, 1.0]},
            ],
            id="out-of-range-index",
        ),
        pytest.param(
            [
                {"embedding": [1.0, 0.0]},
                {"index": 1, "embedding": [0.0, 1.0]},
            ],
            id="missing-index",
        ),
    ],
)
def test_public_adapter_boundary_rejects_invalid_vector_indices(
    client,
    monkeypatch,
    orchestrator_enabled,
    adapter_data,
):
    """Adapter indices must identify every input exactly once."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    credentials = FakeCredentials(api_key="runtime-key", source="user")

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class FakeAdapter:
        def embed(self, _request):
            return {"data": adapter_data}

    class FakeRegistry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return FakeAdapter()

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": ["one", "two"]},
    )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY, response.text
    credentials.touch_last_used.assert_not_awaited()


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
def test_public_adapter_boundary_reconstructs_vectors_by_index(
    client,
    monkeypatch,
    orchestrator_enabled,
):
    """Out-of-order adapter data is paired back to the corresponding input."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    credentials = FakeCredentials(api_key="runtime-key", source="user")

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class FakeAdapter:
        def embed(self, _request):
            return {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }

    class FakeRegistry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return FakeAdapter()

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": ["one", "two"]},
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    assert [item["embedding"] for item in response.json()["data"]] == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]


@pytest.mark.concurrent
@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
def test_concurrent_malformed_and_valid_adapter_vectors_remain_isolated(
    client,
    monkeypatch,
    orchestrator_enabled,
):
    """A malformed response cannot contaminate a concurrent valid adapter result."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    credentials = FakeCredentials(api_key="runtime-key", source="user")
    calls = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class FakeAdapter:
        def embed(self, request):
            text = request["input"]
            with lock:
                calls.append(text)
                if len(calls) == 2:
                    both_arrived.set()
            if not release.wait(10):
                raise TimeoutError("concurrent vector-validation calls were not released")
            vector = [float("inf"), 0.0] if text == "malformed" else [0.0, 1.0]
            return {"data": [{"index": 0, "embedding": vector}]}

    class FakeRegistry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return FakeAdapter()

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())

    def post(text):
        return client.post(
            "/api/v1/embeddings",
            json={"model": "text-embedding-3-small", "input": text},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        malformed = executor.submit(post, "malformed")
        valid = executor.submit(post, "valid")
        try:
            assert both_arrived.wait(10)
        finally:
            release.set()
        malformed_response = malformed.result(timeout=10)
        valid_response = valid.result(timeout=10)

    assert malformed_response.status_code == status.HTTP_502_BAD_GATEWAY
    assert valid_response.status_code == status.HTTP_200_OK, valid_response.text
    assert valid_response.json()["data"][0]["embedding"] == [0.0, 1.0]
    assert set(calls) == {"malformed", "valid"}
    assert credentials.touch_last_used.await_count == 1


@pytest.mark.concurrent
@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
def test_concurrent_public_endpoint_modes_isolate_real_legacy_oauth_refresh_policy(
    client,
    monkeypatch,
    orchestrator_enabled,
):
    """A 401 refresh and a concurrent 403 must not exchange keys, endpoints, or policy."""
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_embeddings_adapter import (
        OpenAIEmbeddingsAdapter,
    )

    snapshots = {
        ("alpha", False): FakeCredentials(
            api_key="key-alpha-old",
            source="user",
            auth_source="oauth",
            credential_fields={"base_url": "https://alpha-old.example/v1"},
            app_config={"openai_api": {"api_base_url": "https://alpha-old.example/v1"}},
        ),
        ("alpha", True): FakeCredentials(
            api_key="key-alpha-new",
            source="user",
            auth_source="oauth",
            credential_fields={"base_url": "https://alpha-new.example/v1"},
            app_config={"openai_api": {"api_base_url": "https://alpha-new.example/v1"}},
        ),
        ("beta", False): FakeCredentials(
            api_key="key-beta",
            source="user",
            auth_source="oauth",
            credential_fields={"base_url": "https://beta.example/v1"},
            app_config={"openai_api": {"api_base_url": "https://beta.example/v1"}},
        ),
    }
    resolve_calls: list[tuple[str, bool]] = []
    transport_calls: list[tuple[str, str, str, str]] = []
    lock = threading.Lock()
    initial_calls_arrived = threading.Event()
    release = threading.Event()

    async def fake_resolve(
        _provider,
        _current_user,
        request,
        *,
        model=None,
        force_oauth_refresh=False,
        rejected_credentials=None,
    ):
        label = request.headers["x-test-credential-label"]
        assert model == f"model-{label}"
        assert rejected_credentials is (
            snapshots[(label, False)] if force_oauth_refresh else None
        )
        with lock:
            resolve_calls.append((label, force_oauth_refresh))
        return snapshots[(label, force_oauth_refresh)]

    class _RawLegacyAuthError(RuntimeError):
        def __init__(self, status_code: int, sentinel: str) -> None:
            super().__init__(sentinel)
            self.status_code = status_code

    def _legacy_single(text, model, app_config=None, dimensions=None):
        del dimensions
        config = (app_config or {})["openai_api"]
        call = (config["api_key"], config["api_base_url"], model, text)
        with lock:
            transport_calls.append(call)
            initial_count = sum(
                item[0] in {"key-alpha-old", "key-beta"}
                for item in transport_calls
            )
            if initial_count == 2:
                initial_calls_arrived.set()
        if call[0] in {"key-alpha-old", "key-beta"} and not release.wait(10):
            raise TimeoutError("concurrent legacy OAuth calls were not released")
        if call[0] == "key-alpha-old":
            raise _RawLegacyAuthError(401, "raw-alpha-oauth-secret")
        if call[0] == "key-beta":
            raise _RawLegacyAuthError(403, "raw-beta-oauth-secret")
        return [0.1, 0.2]

    class _Registry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return OpenAIEmbeddingsAdapter()

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.delenv("LLM_EMBEDDINGS_NATIVE_HTTP_OPENAI", raising=False)
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: _Registry())
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.get_openai_embeddings",
        _legacy_single,
    )

    def _post(label: str):
        return client.post(
            "/api/v1/embeddings",
            headers={"x-test-credential-label": label},
            json={"model": f"model-{label}", "input": label},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        alpha = executor.submit(_post, "alpha")
        beta = executor.submit(_post, "beta")
        try:
            assert initial_calls_arrived.wait(10)
        finally:
            release.set()
        alpha_response = alpha.result(timeout=10)
        beta_response = beta.result(timeout=10)

    assert alpha_response.status_code == 200
    assert beta_response.status_code == status.HTTP_502_BAD_GATEWAY
    assert beta_response.json() == {"detail": "Embedding provider authentication failed"}
    assert "raw-" not in alpha_response.text + beta_response.text
    assert set(transport_calls) == {
        ("key-alpha-old", "https://alpha-old.example/v1", "model-alpha", "alpha"),
        ("key-alpha-new", "https://alpha-new.example/v1", "model-alpha", "alpha"),
        ("key-beta", "https://beta.example/v1", "model-beta", "beta"),
    }
    assert set(resolve_calls) == {
        ("alpha", False),
        ("alpha", True),
        ("beta", False),
    }
    snapshots[("alpha", True)].touch_last_used.assert_awaited_once()
    snapshots[("alpha", False)].touch_last_used.assert_not_awaited()
    snapshots[("beta", False)].touch_last_used.assert_not_awaited()


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
@pytest.mark.parametrize(
    ("first_status", "second_status", "expected_status", "expected_calls"),
    [
        (401, 200, 200, 2),
        (401, 401, 502, 2),
        (403, 200, 502, 1),
    ],
)
def test_public_openai_oauth_adapter_refresh_policy_is_identical_across_endpoint_modes(
    client,
    monkeypatch,
    orchestrator_enabled,
    first_status,
    second_status,
    expected_status,
    expected_calls,
):
    from loguru import logger

    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError

    sentinel = "public-oauth-raw-provider-body-sentinel"
    initial = FakeCredentials(
        api_key="oauth-old-key",
        source="user",
        auth_source="oauth",
        credential_fields={"base_url": "https://oauth-old.example/v1"},
        app_config={"openai_api": {"api_base_url": "https://oauth-old.example/v1"}},
    )
    refreshed = FakeCredentials(
        api_key="oauth-new-key",
        source="user",
        auth_source="oauth",
        credential_fields={"base_url": "https://oauth-new.example/v1"},
        app_config={"openai_api": {"api_base_url": "https://oauth-new.example/v1"}},
    )
    resolve_calls = []
    adapter_calls = []
    log_records = []

    async def fake_resolve(*_args, force_oauth_refresh=False, **_kwargs):
        resolve_calls.append(force_oauth_refresh)
        return refreshed if force_oauth_refresh else initial

    class FakeAdapter:
        def embed(self, request):
            adapter_calls.append((request["api_key"], request["base_url"]))
            status_code = first_status if len(adapter_calls) == 1 else second_status
            if status_code in {401, 403}:
                raise ChatAuthenticationError(
                    provider="openai-embeddings",
                    message=sentinel,
                    status_code=status_code,
                )
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    class FakeRegistry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return FakeAdapter()

    async def fail_fallback(*_args, **_kwargs):
        raise AssertionError("OAuth adapter auth handling must not use legacy fallback")

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", fail_fallback)
    monkeypatch.setattr(mod, "create_embeddings_batch_async", fail_fallback)
    sink_id = logger.add(log_records.append, format="{message} {extra}")
    try:
        response = client.post(
            "/api/v1/embeddings",
            json={"model": "text-embedding-3-small", "input": "oauth public"},
        )
    finally:
        logger.remove(sink_id)

    assert response.status_code == expected_status
    assert len(adapter_calls) == expected_calls
    assert adapter_calls[0] == ("oauth-old-key", "https://oauth-old.example/v1")
    if expected_calls == 2:
        assert adapter_calls[1] == ("oauth-new-key", "https://oauth-new.example/v1")
        assert resolve_calls == [False, True]
    else:
        assert resolve_calls == [False]
    if expected_status != 200:
        assert response.json() == {"detail": "Embedding provider authentication failed"}
    assert sentinel not in response.text
    assert sentinel not in "".join(map(str, log_records))


def test_orchestrator_adapter_miss_with_resolved_key_bypasses_shared_cache(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "true")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    credentials = FakeCredentials(api_key="provider-key", source="user")

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class FakeRegistry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return None

    provider_call = AsyncMock(return_value=[[0.0, 1.0]])
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
    assert response.json()["data"][0]["embedding"] == [0.0, 1.0]
    assert cache_get.await_count == 0
    assert cache_set.await_count == 0
    assert provider_call.await_count == 1
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


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["cancel", "timeout"])
@pytest.mark.parametrize("outcome", ["valid", "malformed", "error"])
async def test_executor_explicit_fallback_owns_late_result_before_cancellation_ends(
    monkeypatch,
    termination,
    outcome,
):
    """The fallback worker exits and only strict-valid vectors charge credentials."""

    from tldw_Server_API.app.api.v1.endpoints import (
        embeddings_v5_production_enhanced as mod,
    )

    entered = asyncio.Event()
    release = asyncio.Event()
    lifecycle: list[str] = []
    credentials = FakeCredentials(
        api_key="explicit-fallback-secret",
        source="user",
    )
    credentials.touch_last_used.side_effect = lambda: lifecycle.append("touch")

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    async def blocking_provider(*_args, **_kwargs):
        lifecycle.append("provider-start")
        entered.set()
        await release.wait()
        lifecycle.append("provider-exit")
        if outcome == "error":
            raise RuntimeError("late provider failure with explicit-fallback-secret")
        if outcome == "malformed":
            return [[]]
        return [[1.0, 0.0]]

    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", blocking_provider)
    executor = mod._EndpointEmbeddingExecutor(
        request=_request(),
        current_user=_user(),
        user_metadata=None,
    )
    if termination == "timeout":
        task = asyncio.create_task(
            asyncio.wait_for(
                executor.create(
                    ["late result"],
                    provider="openai",
                    model="text-embedding-3-small",
                    dimensions=None,
                ),
                timeout=0.01,
            )
        )
    else:
        task = asyncio.create_task(
            executor.create(
                ["late result"],
                provider="openai",
                model="text-embedding-3-small",
                dimensions=None,
            )
        )

    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        if termination == "cancel":
            task.cancel()
        await asyncio.sleep(0.03)
        assert task.done() is False
        assert credentials.touch_last_used.await_count == 0

        release.set()
        expected_error = asyncio.TimeoutError if termination == "timeout" else asyncio.CancelledError
        with pytest.raises(expected_error):
            await asyncio.wait_for(task, timeout=1.0)
        lifecycle.append("ownership-end")
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    expected_lifecycle = ["provider-start", "provider-exit"]
    if outcome == "valid":
        expected_lifecycle.append("touch")
        credentials.touch_last_used.assert_awaited_once()
    else:
        credentials.touch_last_used.assert_not_awaited()
    expected_lifecycle.append("ownership-end")
    assert lifecycle == expected_lifecycle


@pytest.mark.asyncio
async def test_executor_explicit_fallback_capacity_is_canonical_and_predispatch(
    monkeypatch,
    tmp_path,
):
    """Saturated fallback capacity becomes the existing sanitized service error."""

    from tldw_Server_API.app.api.v1.endpoints import (
        embeddings_v5_production_enhanced as mod,
    )
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import (
        Embeddings_Create as embeddings_create,
    )

    pool = BoundedDaemonPool(capacity=1)
    holder_entered = threading.Event()
    holder_release = threading.Event()
    holder_released = threading.Event()
    starts: list[str] = []
    credentials = FakeCredentials(
        api_key="capacity-rejected-secret",
        source="user",
    )

    def hold_capacity() -> None:
        holder_entered.set()
        assert holder_release.wait(timeout=2.0)

    def provider_call(*_args, **_kwargs):
        starts.append("provider-entered-with-capacity-rejected-secret")
        return [[1.0, 0.0]]

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    pool.start(
        hold_capacity,
        name="embeddings-fallback-holder",
        released_event=holder_released,
    )
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(
        embeddings_create,
        "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT",
        tmp_path.resolve(),
    )
    monkeypatch.setattr(
        embeddings_create,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    monkeypatch.setattr(embeddings_create, "create_embeddings_batch", provider_call)
    executor = mod._EndpointEmbeddingExecutor(
        request=_request(),
        current_user=_user(),
        user_metadata=None,
    )
    try:
        await _wait_for_thread_event(holder_entered)
        with pytest.raises(EmbeddingProviderError) as exc_info:
            await executor.create(
                ["must-not-dispatch"],
                provider="openai",
                model="text-embedding-3-small",
                dimensions=None,
            )

        assert exc_info.value.code == "provider_unavailable"
        assert exc_info.value.message == mod.EMBEDDING_SERVICE_FAILED_DETAIL
        assert "capacity-rejected-secret" not in repr(exc_info.value)
        assert starts == []
        credentials.touch_last_used.assert_not_awaited()
        assert pool.active_count == 1
    finally:
        holder_release.set()
        assert holder_released.wait(timeout=2.0)

    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_executor_explicit_fallback_cancelled_oauth_retry_touches_only_refresh(
    monkeypatch,
):
    """A late valid OAuth retry cannot charge the rejected credential snapshot."""

    from tldw_Server_API.app.api.v1.endpoints import (
        embeddings_v5_production_enhanced as mod,
    )

    retry_entered = asyncio.Event()
    retry_release = asyncio.Event()
    calls: list[str] = []
    initial = FakeCredentials(
        api_key="oauth-fallback-stale-secret",
        source="user",
        auth_source="oauth",
    )
    refreshed = FakeCredentials(
        api_key="oauth-fallback-refreshed-secret",
        source="user",
        auth_source="oauth",
    )

    async def fake_resolve(*_args, force_oauth_refresh=False, **_kwargs):
        return refreshed if force_oauth_refresh else initial

    async def oauth_provider(_texts, _provider, _model, config, **_kwargs):
        api_key = config["api_key"]
        calls.append(api_key)
        if api_key == "oauth-fallback-stale-secret":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="expired oauth-fallback-stale-secret",
            )
        retry_entered.set()
        await retry_release.wait()
        return [[0.4, 0.6]]

    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", oauth_provider)
    executor = mod._EndpointEmbeddingExecutor(
        request=_request(),
        current_user=_user(),
        user_metadata=None,
    )
    task = asyncio.create_task(
        executor.create(
            ["oauth retry"],
            provider="openai",
            model="text-embedding-3-small",
            dimensions=None,
        )
    )
    try:
        await asyncio.wait_for(retry_entered.wait(), timeout=1.0)
        task.cancel()
        await asyncio.sleep(0.03)
        assert task.done() is False
        initial.touch_last_used.assert_not_awaited()
        refreshed.touch_last_used.assert_not_awaited()

        retry_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        retry_release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert calls == [
        "oauth-fallback-stale-secret",
        "oauth-fallback-refreshed-secret",
    ]
    initial.touch_last_used.assert_not_awaited()
    refreshed.touch_last_used.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["legacy-single", "batch"])
async def test_public_explicit_fallback_cancel_drains_valid_result_and_touch(
    monkeypatch,
    surface,
):
    """Both public fallback handlers keep the exact credential snapshot owned."""

    from tldw_Server_API.app.api.v1.endpoints import (
        embeddings_v5_production_enhanced as mod,
    )

    entered = asyncio.Event()
    release = asyncio.Event()
    credentials = FakeCredentials(
        api_key=f"public-{surface}-secret",
        source="user",
    )

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    async def blocking_fallback(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return [[1.0, 0.0]]

    async def no_backpressure(*_args, **_kwargs):
        return None

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(rg_governor=None, rg_policy_loader=None)
        ),
        state=SimpleNamespace(),
        headers={},
        method="POST",
        url=SimpleNamespace(path="/api/v1/embeddings"),
    )
    response = SimpleNamespace(headers={})
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.delenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", raising=False)
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "create_embeddings_batch_async", blocking_fallback)
    monkeypatch.setattr(mod, "_check_backpressure_and_quotas", no_backpressure)
    monkeypatch.setattr(mod, "count_tokens", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(mod, "_get_model_max_tokens", lambda *_args, **_kwargs: 100)
    monkeypatch.setattr(
        mod,
        "_enforce_embedding_policy_decision",
        lambda **_kwargs: SimpleNamespace(fallback_chain=["openai"]),
    )

    if surface == "batch":
        operation = mod.create_embeddings_batch_endpoint(
            payload=mod.EmbeddingsBatchRequest(
                texts=["public batch"],
                model="text-embedding-3-small",
                provider="openai",
            ),
            request=request,
            current_user=_user(),
            response=response,
        )
    else:
        operation = mod._create_embedding_legacy(
            request=request,
            embedding_request=mod.CreateEmbeddingRequest(
                input="public single",
                model="text-embedding-3-small",
            ),
            current_user=_user(),
            background_tasks=SimpleNamespace(),
            x_provider="openai",
            response=response,
        )

    task = asyncio.create_task(operation)
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        task.cancel()
        await asyncio.sleep(0.03)
        assert task.done() is False
        credentials.touch_last_used.assert_not_awaited()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    credentials.touch_last_used.assert_awaited_once()


@pytest.mark.parametrize(
    "surface",
    ["orchestrator-single", "legacy-single", "batch"],
)
def test_public_explicit_fallback_capacity_response_is_sanitized_and_predispatch(
    client,
    monkeypatch,
    tmp_path,
    surface,
):
    """Every public explicit fallback surface reuses the canonical 503 contract."""

    from tldw_Server_API.app.api.v1.endpoints import (
        embeddings_v5_production_enhanced as mod,
    )
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import (
        Embeddings_Create as embeddings_create,
    )

    pool = BoundedDaemonPool(capacity=1)
    holder_entered = threading.Event()
    holder_release = threading.Event()
    holder_released = threading.Event()
    starts: list[str] = []
    credentials = FakeCredentials(
        api_key=f"public-capacity-{surface}-secret",
        source="user",
    )

    def hold_capacity() -> None:
        holder_entered.set()
        assert holder_release.wait(timeout=10.0)

    def provider_call(*_args, **_kwargs):
        starts.append(f"entered-{surface}-with-secret")
        return [[1.0, 0.0]]

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    original_policy_setting = mod._embedding_policy_setting

    def single_provider_policy(name, default):
        if name == "EMBEDDINGS_FALLBACK_CHAIN":
            return {"openai": ["openai"]}
        return original_policy_setting(name, default)

    pool.start(
        hold_capacity,
        name="public-embeddings-fallback-holder",
        released_event=holder_released,
    )
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.delenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", raising=False)
    if surface == "orchestrator-single":
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "_embedding_policy_setting", single_provider_policy)
    monkeypatch.setattr(
        embeddings_create,
        "_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT",
        tmp_path.resolve(),
    )
    monkeypatch.setattr(
        embeddings_create,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    monkeypatch.setattr(embeddings_create, "create_embeddings_batch", provider_call)
    try:
        assert holder_entered.wait(timeout=2.0)
        if surface == "batch":
            response = client.post(
                "/api/v1/embeddings/batch",
                json={
                    "texts": [f"capacity-{uuid.uuid4().hex}"],
                    "model": "text-embedding-3-small",
                    "provider": "openai",
                },
            )
        else:
            response = client.post(
                "/api/v1/embeddings",
                json={
                    "model": "text-embedding-3-small",
                    "input": f"capacity-{uuid.uuid4().hex}",
                },
            )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert response.json() == {"detail": mod.EMBEDDING_SERVICE_FAILED_DETAIL}
        assert f"public-capacity-{surface}-secret" not in response.text
        assert starts == []
        credentials.touch_last_used.assert_not_awaited()
        assert pool.active_count == 1
    finally:
        holder_release.set()
        assert holder_released.wait(timeout=2.0)

    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_executor_multibatch_marks_first_valid_batch_before_second_error(
    monkeypatch,
):
    """A later batch failure cannot erase an earlier strict-valid provider use."""

    from tldw_Server_API.app.api.v1.endpoints import (
        embeddings_v5_production_enhanced as mod,
    )

    second_entered = asyncio.Event()
    second_release = asyncio.Event()
    calls = 0
    credentials = FakeCredentials(
        api_key="multibatch-executor-secret",
        source="user",
    )

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    async def provider_call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [[1.0, 0.0]]
        second_entered.set()
        await second_release.wait()
        raise RuntimeError("second provider batch failed")

    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "MAX_BATCH_SIZE", 1)
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", provider_call)
    executor = mod._EndpointEmbeddingExecutor(
        request=_request(),
        current_user=_user(),
        user_metadata=None,
    )
    task = asyncio.create_task(
        executor.create(
            ["first", "second"],
            provider="openai",
            model="text-embedding-3-small",
            dimensions=None,
        )
    )
    try:
        await asyncio.wait_for(second_entered.wait(), timeout=1.0)
        credentials.touch_last_used.assert_awaited_once()
    finally:
        second_release.set()
        result = await asyncio.gather(task, return_exceptions=True)

    assert isinstance(result[0], EmbeddingExecutionError)
    assert calls == 2
    credentials.touch_last_used.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["legacy-single", "batch"])
async def test_public_multibatch_marks_first_valid_batch_before_second_cancel(
    monkeypatch,
    surface,
):
    """Public fallbacks retain prior provider use when a later batch is cancelled."""

    from tldw_Server_API.app.api.v1.endpoints import (
        embeddings_v5_production_enhanced as mod,
    )

    second_entered = asyncio.Event()
    second_release = asyncio.Event()
    calls = 0
    credentials = FakeCredentials(
        api_key=f"multibatch-{surface}-secret",
        source="user",
    )

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    async def provider_call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [[1.0, 0.0]]
        second_entered.set()
        await second_release.wait()
        return [[]]

    async def no_backpressure(*_args, **_kwargs):
        return None

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(rg_governor=None, rg_policy_loader=None)
        ),
        state=SimpleNamespace(),
        headers={},
        method="POST",
        url=SimpleNamespace(path="/api/v1/embeddings"),
    )
    response = SimpleNamespace(headers={})
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.delenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", raising=False)
    monkeypatch.setattr(mod, "MAX_BATCH_SIZE", 1)
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", provider_call)
    monkeypatch.setattr(mod, "_check_backpressure_and_quotas", no_backpressure)
    monkeypatch.setattr(mod, "count_tokens", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(mod, "_get_model_max_tokens", lambda *_args, **_kwargs: 100)
    monkeypatch.setattr(
        mod,
        "_enforce_embedding_policy_decision",
        lambda **_kwargs: SimpleNamespace(fallback_chain=["openai"]),
    )

    if surface == "batch":
        operation = mod.create_embeddings_batch_endpoint(
            payload=mod.EmbeddingsBatchRequest(
                texts=["first", "second"],
                model="text-embedding-3-small",
                provider="openai",
            ),
            request=request,
            current_user=_user(),
            response=response,
        )
    else:
        operation = mod._create_embedding_legacy(
            request=request,
            embedding_request=mod.CreateEmbeddingRequest(
                input=["first", "second"],
                model="text-embedding-3-small",
            ),
            current_user=_user(),
            background_tasks=SimpleNamespace(),
            x_provider="openai",
            response=response,
        )

    task = asyncio.create_task(operation)
    try:
        await asyncio.wait_for(second_entered.wait(), timeout=1.0)
        credentials.touch_last_used.assert_awaited_once()
        task.cancel()
        await asyncio.sleep(0.03)
        assert task.done() is False
        second_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        second_release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert calls == 2
    credentials.touch_last_used.assert_awaited_once()

import base64
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.main import app


@pytest.fixture
def client():
    original_testing = os.environ.get("TESTING")
    original_orchestrator = os.environ.get("EMBEDDINGS_ORCHESTRATOR_ENABLED")
    original_overrides = dict(app.dependency_overrides)

    try:
        os.environ["TESTING"] = "true"
        os.environ.pop("EMBEDDINGS_ORCHESTRATOR_ENABLED", None)

        async def override_user():
            return User(
                id=1,
                username="embedding-user",
                email="embedding-user@example.com",
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
    finally:
        if original_testing is None:
            os.environ.pop("TESTING", None)
        else:
            os.environ["TESTING"] = original_testing
        if original_orchestrator is None:
            os.environ.pop("EMBEDDINGS_ORCHESTRATOR_ENABLED", None)
        else:
            os.environ["EMBEDDINGS_ORCHESTRATOR_ENABLED"] = original_orchestrator

        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_full_cache_hit_skips_provider_and_preserves_order(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    async def provider_should_not_be_called(*args, **kwargs):
        raise AssertionError("provider should not be called on a full cache hit")

    cache_set = AsyncMock()
    monkeypatch.setattr(
        mod.embedding_cache,
        "get",
        AsyncMock(side_effect=[[1.0, 0.0], [0.0, 1.0]]),
    )
    monkeypatch.setattr(mod.embedding_cache, "set", cache_set)
    monkeypatch.setattr(
        mod,
        "create_embeddings_with_circuit_breaker",
        provider_should_not_be_called,
        raising=True,
    )

    result = await mod.create_embeddings_batch_async(
        ["a", "b"],
        provider="huggingface",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
    )

    assert result == [[1.0, 0.0], [0.0, 1.0]]  # nosec B101
    assert cache_set.await_count == 0  # nosec B101


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_partial_cache_hit_executes_only_misses_and_writes_float_vectors(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    async def fake_provider(texts, provider, model_id, config, metadata=None, dimensions=None):
        _ = (provider, model_id, config, metadata, dimensions)
        assert texts == ["miss"]  # nosec B101
        return [[0.25, 0.75]]

    cache_set = AsyncMock()
    monkeypatch.setattr(
        mod.embedding_cache,
        "get",
        AsyncMock(side_effect=[[1.0, 0.0], None]),
    )
    monkeypatch.setattr(mod.embedding_cache, "set", cache_set)
    monkeypatch.setattr(
        mod,
        "create_embeddings_with_circuit_breaker",
        fake_provider,
        raising=True,
    )

    result = await mod.create_embeddings_batch_async(
        ["hit", "miss"],
        provider="huggingface",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
    )

    assert result == [[1.0, 0.0], [0.25, 0.75]]  # nosec B101
    cached_value = cache_set.await_args.args[1]
    assert cached_value == [0.25, 0.75]  # nosec B101
    assert all(isinstance(item, float) for item in cached_value)  # nosec B101


@pytest.mark.unit
def test_endpoint_base64_response_does_not_write_base64_to_cache(client, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    async def fake_provider(texts, provider, model_id, config, metadata=None, dimensions=None):
        _ = (provider, model_id, config, metadata, dimensions)
        assert texts == ["cache me"]  # nosec B101
        return [[0.25, 0.75]]

    cache_set = AsyncMock()
    monkeypatch.setattr(mod.embedding_cache, "get", AsyncMock(return_value=None))
    monkeypatch.setattr(mod.embedding_cache, "set", cache_set)
    monkeypatch.setattr(
        mod,
        "create_embeddings_with_circuit_breaker",
        fake_provider,
        raising=True,
    )

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "huggingface"},
        json={
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "input": "cache me",
            "encoding_format": "base64",
            "dimensions": 2,
        },
    )

    assert response.status_code == 200  # nosec B101
    encoded = response.json()["data"][0]["embedding"]
    decoded = np.frombuffer(base64.b64decode(encoded), dtype=np.float32)
    assert decoded.tolist() == pytest.approx([0.25, 0.75])  # nosec B101
    cached_value = cache_set.await_args.args[1]
    assert cached_value == [0.25, 0.75]  # nosec B101
    assert all(isinstance(item, float) for item in cached_value)  # nosec B101


@pytest.mark.unit
def test_endpoint_dimension_adjustment_cache_write_order_is_characterized(client, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    async def fake_provider(texts, provider, model_id, config, metadata=None, dimensions=None):
        _ = (provider, model_id, config, metadata, dimensions)
        assert texts == ["cache me"]  # nosec B101
        return [[0.1, 0.2, 0.3, 0.4]]

    cache_set = AsyncMock()
    monkeypatch.setattr(mod.embedding_cache, "get", AsyncMock(return_value=None))
    monkeypatch.setattr(mod.embedding_cache, "set", cache_set)
    monkeypatch.setattr(
        mod,
        "create_embeddings_with_circuit_breaker",
        fake_provider,
        raising=True,
    )

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "huggingface"},
        json={
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "input": "cache me",
            "dimensions": 2,
        },
    )

    assert response.status_code == 200  # nosec B101
    assert len(response.json()["data"][0]["embedding"]) == 2  # nosec B101
    cached_value = cache_set.await_args.args[1]
    # Compatibility behavior: the legacy path writes the pre-adjustment provider
    # vector before endpoint-level dimension postprocessing.
    assert len(cached_value) == 4  # nosec B101


@pytest.mark.unit
def test_endpoint_full_cache_hit_still_reserves_and_commits_rg_tokens(client, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    async def provider_should_not_be_called(*args, **kwargs):
        raise AssertionError("provider should not be called on a full cache hit")

    class FakePolicyLoader:
        def get_policy(self, policy_id):
            _ = policy_id
            return {"tokens": {"daily_cap": 1000, "per_min": 100}}

    class FakeGovernor:
        def __init__(self):
            self.reserve = AsyncMock(
                return_value=(SimpleNamespace(allowed=True, retry_after=None), "handle-1")
            )
            self.commit = AsyncMock()

    missing = object()
    governor = FakeGovernor()
    old_governor = app.state.rg_governor if hasattr(app.state, "rg_governor") else missing
    old_loader = app.state.rg_policy_loader if hasattr(app.state, "rg_policy_loader") else missing
    app.state.rg_governor = governor
    app.state.rg_policy_loader = FakePolicyLoader()

    try:
        monkeypatch.setattr(mod.embedding_cache, "get", AsyncMock(return_value=[0.25, 0.75]))
        monkeypatch.setattr(mod.embedding_cache, "set", AsyncMock())
        monkeypatch.setattr(
            mod,
            "create_embeddings_with_circuit_breaker",
            provider_should_not_be_called,
            raising=True,
        )

        response = client.post(
            "/api/v1/embeddings",
            headers={"x-provider": "huggingface"},
            json={
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "input": "cache me",
            },
        )
    finally:
        if old_governor is missing:
            delattr(app.state, "rg_governor")
        else:
            app.state.rg_governor = old_governor
        if old_loader is missing:
            delattr(app.state, "rg_policy_loader")
        else:
            app.state.rg_policy_loader = old_loader

    assert response.status_code == 200  # nosec B101
    governor.reserve.assert_awaited_once()
    governor.commit.assert_awaited_once()
    reserved_units = governor.reserve.await_args.args[0].categories["tokens"]["units"]
    assert governor.commit.await_args.args[0] == "handle-1"  # nosec B101
    committed_units = governor.commit.await_args.kwargs["actuals"]["tokens"]
    assert reserved_units > 0  # nosec B101
    assert committed_units > 0  # nosec B101


@pytest.mark.unit
def test_endpoint_vector_count_mismatch_maps_to_502(client, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    async def fake_provider(texts, provider, model_id, config, metadata=None, dimensions=None):
        _ = (texts, provider, model_id, config, metadata, dimensions)
        return [[0.25, 0.75]]

    cache_set = AsyncMock()
    monkeypatch.setattr(mod.embedding_cache, "get", AsyncMock(return_value=None))
    monkeypatch.setattr(mod.embedding_cache, "set", cache_set)
    monkeypatch.setattr(
        mod,
        "create_embeddings_with_circuit_breaker",
        fake_provider,
        raising=True,
    )

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "huggingface"},
        json={
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "input": ["one", "two"],
        },
    )

    assert response.status_code == 502  # nosec B101
    cache_set.assert_not_awaited()
    assert "returned 1 embeddings" in response.json()["detail"]  # nosec B101
    assert "expected 2" in response.json()["detail"]  # nosec B101

import os
import base64
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


@pytest.fixture(autouse=True)
def _testing_env():
    keys = (
        "TESTING",
        "EMBEDDINGS_DIMENSION_POLICY",
        "USE_REAL_OPENAI_IN_TESTS",
    )
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["TESTING"] = "true"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture
def client():
    with TestClient(app) as c:
        c.cookies.set("csrf_token", "x")
        c.headers["X-CSRF-Token"] = "x"
        c.headers["Authorization"] = "Bearer key"
        yield c


def _override_user(admin=False):


    async def _f():
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        return User(id=1, username="u", email="u@x", is_active=True, is_admin=admin)
    return _f


class _MC:
    def labels(self, **kwargs):
        return self
    def inc(self, *args, **kwargs):
        return None


@pytest.mark.unit
def test_dimensions_reduce_policy(client, monkeypatch):
     # Force reduce
    os.environ["EMBEDDINGS_DIMENSION_POLICY"] = "reduce"
    os.environ["USE_REAL_OPENAI_IN_TESTS"] = "true"  # force async path

    # Patch async batch to return HF-like 384 dim
    async def fake_batch_async(texts, provider, model_id=None, dimensions=None, api_key=None, api_url=None, metadata=None):
        return [[0.1] * 384 for _ in texts]

    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod
    monkeypatch.setattr(mod, "create_embeddings_batch_async", fake_batch_async, raising=True)
    monkeypatch.setattr(mod, "embedding_dimension_adjustments_total", _MC(), raising=False)

    app.dependency_overrides[get_request_user] = _override_user()
    r = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "huggingface"},
        json={"input": "txt", "model": "sentence-transformers/all-MiniLM-L6-v2", "dimensions": 128}
    )
    assert r.status_code == 200
    body = r.json()
    vec = body["data"][0]["embedding"]
    assert len(vec) == 128
    # header present
    assert r.headers.get("X-Embeddings-Dimensions-Policy") == "reduce"


@pytest.mark.unit
def test_dimensions_pad_policy(client, monkeypatch):
    os.environ["EMBEDDINGS_DIMENSION_POLICY"] = "pad"
    os.environ["USE_REAL_OPENAI_IN_TESTS"] = "true"

    async def fake_batch_async(texts, provider, model_id=None, dimensions=None, api_key=None, api_url=None, metadata=None):
        return [[0.2] * 384 for _ in texts]

    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod
    monkeypatch.setattr(mod, "create_embeddings_batch_async", fake_batch_async, raising=True)
    monkeypatch.setattr(mod, "embedding_dimension_adjustments_total", _MC(), raising=False)

    app.dependency_overrides[get_request_user] = _override_user()
    r = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "huggingface"},
        json={"input": "txt", "model": "sentence-transformers/all-MiniLM-L6-v2", "dimensions": 512}
    )
    assert r.status_code == 200
    vec = r.json()["data"][0]["embedding"]
    assert len(vec) == 512
    assert r.headers.get("X-Embeddings-Dimensions-Policy") == "pad"


@pytest.mark.unit
def test_dimensions_ignore_policy(client, monkeypatch):
    os.environ["EMBEDDINGS_DIMENSION_POLICY"] = "ignore"
    os.environ["USE_REAL_OPENAI_IN_TESTS"] = "true"

    async def fake_batch_async(texts, provider, model_id=None, dimensions=None, api_key=None, api_url=None, metadata=None):
        return [[0.3] * 384 for _ in texts]

    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod
    monkeypatch.setattr(mod, "create_embeddings_batch_async", fake_batch_async, raising=True)
    monkeypatch.setattr(mod, "embedding_dimension_adjustments_total", _MC(), raising=False)

    app.dependency_overrides[get_request_user] = _override_user()
    r = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "huggingface"},
        json={"input": "txt", "model": "sentence-transformers/all-MiniLM-L6-v2", "dimensions": 1024}
    )
    assert r.status_code == 200
    vec = r.json()["data"][0]["embedding"]
    # Should remain native size
    assert len(vec) == 384
    assert r.headers.get("X-Embeddings-Dimensions-Policy") == "ignore"

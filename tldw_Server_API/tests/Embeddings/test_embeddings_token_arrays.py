import os
import base64
import numpy as np
import pytest
from unittest.mock import AsyncMock, Mock
from fastapi.testclient import TestClient
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


@pytest.fixture(autouse=True)
def _enable_testing_env():
    os.environ["TESTING"] = "true"
    yield
    os.environ.pop("TESTING", None)


@pytest.fixture
def client():
    with TestClient(app) as c:
        c.cookies.set("csrf_token", "test-csrf")
        c.headers["X-CSRF-Token"] = "test-csrf"
        c.headers["Authorization"] = "Bearer test-api-key"
        yield c


@pytest.mark.unit
def test_single_token_array_input(client):
    async def override_user():
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        return User(id=1, username="u", email="u@x", is_active=True, is_admin=False)

    app.dependency_overrides[get_request_user] = override_user

    payload = {
        "model": "text-embedding-3-small",
        "input": [101, 102, 103, 104]
    }
    r = client.post("/api/v1/embeddings", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert len(data["data"]) == 1
    emb = data["data"][0]["embedding"]
    assert isinstance(emb, list) and len(emb) > 0


@pytest.mark.unit
def test_batch_token_arrays_input_base64(client):
    async def override_user():
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        return User(id=1, username="u", email="u@x", is_active=True, is_admin=False)

    app.dependency_overrides[get_request_user] = override_user

    payload = {
        "model": "text-embedding-3-small",
        "input": [[101, 102], [103, 104, 105]],
        "encoding_format": "base64",
        "dimensions": 128
    }
    r = client.post("/api/v1/embeddings", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert len(data["data"]) == 2
    for item in data["data"]:
        b64 = item["embedding"]
        raw = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
        # Should match requested dimensions
        assert len(raw) == 128


@pytest.mark.unit
def test_token_array_uses_raw_token_length_for_limits(client, monkeypatch):
    async def override_user():
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        return User(id=1, username="u", email="u@x", is_active=True, is_admin=False)

    app.dependency_overrides[get_request_user] = override_user

    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_mod
    monkeypatch.setattr(emb_mod, "_get_model_max_tokens", lambda _provider, _model: 2)

    payload = {
        "model": "text-embedding-3-small",
        "input": [101, 102, 103],
    }
    r = client.post("/api/v1/embeddings", json=payload)
    assert r.status_code == 400
    data = r.json()
    assert data.get("error") == "input_too_long"
    assert data.get("details")[0]["tokens"] == 3


@pytest.mark.unit
def test_tokens_to_texts_decode_failure_raises_value_error(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_mod

    class _BadEncoder:
        def decode(self, _tokens):
            raise ValueError("boom")

    monkeypatch.setattr(emb_mod, "get_tokenizer", lambda _model: _BadEncoder())
    warn = Mock()
    monkeypatch.setattr(emb_mod.logger, "warning", warn)

    with pytest.raises(ValueError, match="Invalid token array input"):
        emb_mod.tokens_to_texts([1, 2], "text-embedding-3-small")
    assert warn.called


@pytest.mark.unit
def test_embeddings_endpoint_decode_failure_short_circuits_downstream_creation(client, monkeypatch):
    async def override_user():
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        return User(id=1, username="u", email="u@x", is_active=True, is_admin=False)

    app.dependency_overrides[get_request_user] = override_user

    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_mod

    class _BadEncoder:
        def decode(self, _tokens):
            raise ValueError("boom")

    monkeypatch.setattr(emb_mod, "get_tokenizer", lambda _model: _BadEncoder())
    create_embeddings_mock = AsyncMock()
    monkeypatch.setattr(emb_mod, "create_embeddings_batch_async", create_embeddings_mock)

    payload = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "input": [101, 102, 103, 104],
    }
    r = client.post("/api/v1/embeddings", json=payload, headers={"x-provider": "huggingface"})

    assert r.status_code == 400
    assert r.json()["detail"] == "Invalid token array input"
    create_embeddings_mock.assert_not_awaited()


@pytest.mark.unit
def test_list_input_rejects_empty_strings(client):
    async def override_user():
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        return User(id=1, username="u", email="u@x", is_active=True, is_admin=False)

    app.dependency_overrides[get_request_user] = override_user

    payload = {
        "model": "text-embedding-3-small",
        "input": ["ok", "  "],
    }
    r = client.post("/api/v1/embeddings", json=payload)
    assert r.status_code == 400
    assert "empty strings" in r.json().get("detail", "").lower()


@pytest.mark.unit
def test_single_token_array_allows_large_length(client, monkeypatch):
    async def override_user():
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        return User(id=1, username="u", email="u@x", is_active=True, is_admin=False)

    app.dependency_overrides[get_request_user] = override_user

    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_mod
    monkeypatch.setattr(emb_mod, "_get_model_max_tokens", lambda _provider, _model: 4096)

    payload = {
        "model": "text-embedding-3-small",
        "input": list(range(2050)),
    }
    r = client.post("/api/v1/embeddings", json=payload)
    assert r.status_code == 200


@pytest.mark.unit
def test_provider_header_case_insensitive(client):
    async def override_user():
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        return User(id=1, username="u", email="u@x", is_active=True, is_admin=False)

    app.dependency_overrides[get_request_user] = override_user

    client.headers["x-provider"] = "OpenAI"
    payload = {
        "model": "text-embedding-3-small",
        "input": "hello",
    }
    r = client.post("/api/v1/embeddings", json=payload)
    assert r.status_code == 200

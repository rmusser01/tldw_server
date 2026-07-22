import pytest

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.main import app


@pytest.fixture(autouse=True)
def _override_user():
    async def _user():
        return User(id=1, username="u", email="u@x", is_active=True, is_admin=False)

    app.dependency_overrides[get_request_user] = _user
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_request_user, None)


@pytest.mark.unit
def test_batch_dimensions_allowed_for_non_openai(test_client, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_mod

    monkeypatch.setattr(emb_mod, "_should_enforce_policy", lambda _user=None: False)
    cache_scope_values: list[bool] = []

    async def fake_batch_async(
        texts,
        provider,
        model_id=None,
        dimensions=None,
        api_key=None,
        api_url=None,
        metadata=None,
        cache_scope_sensitive=False,
    ):
        cache_scope_values.append(cache_scope_sensitive)
        return [[float(i) for i in range(256)] for _ in texts]

    monkeypatch.setattr(emb_mod, "create_embeddings_batch_async", fake_batch_async, raising=True)

    resp = test_client.post(
        "/api/v1/embeddings/batch",
        json={
            "texts": ["hello"],
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "provider": "huggingface",
            "dimensions": 128,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["embeddings"][0]) == 128
    assert cache_scope_values == [False]


@pytest.mark.unit
def test_batch_dimensions_rejected_for_non_openai_over_max(test_client, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_mod

    monkeypatch.setattr(emb_mod, "_should_enforce_policy", lambda _user=None: False)

    resp = test_client.post(
        "/api/v1/embeddings/batch",
        json={
            "texts": ["hello"],
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "provider": "huggingface",
            "dimensions": 5000,
        },
    )
    assert resp.status_code == 400
    assert "dimensions" in resp.json().get("detail", "").lower()


@pytest.mark.unit
def test_batch_dimensions_rejected_for_openai_non_3_models(test_client, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_mod

    monkeypatch.setattr(emb_mod, "_should_enforce_policy", lambda _user=None: False)

    resp = test_client.post(
        "/api/v1/embeddings/batch",
        json={
            "texts": ["hello"],
            "model": "text-embedding-ada-002",
            "provider": "openai",
            "dimensions": 128,
        },
    )
    assert resp.status_code == 400
    assert "dimensions" in resp.json().get("detail", "").lower()

import pytest
from fastapi.testclient import TestClient
from tldw_Server_API.app.main import app


@pytest.mark.unit
def test_batch_endpoint_adds_rate_headers(disable_heavy_startup, admin_user, monkeypatch, test_client):
     # Patch backpressure/quotas checker to inject rate limit state
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_mod

    async def _stub_check(request, user):
        try:
            request.state.rate_limit_limit = 10
            request.state.rate_limit_remaining = 7
        except Exception:
            _ = None
        return None

    monkeypatch.setattr(emb_mod, "_check_backpressure_and_quotas", _stub_check)

    # Patch the batch async creator to return deterministic vectors quickly
    async def _stub_batch(
        texts,
        provider,
        model_id=None,
        dimensions=None,
        api_key=None,
        api_url=None,
        metadata=None,
        cache_scope_sensitive=False,
    ):
        _ = cache_scope_sensitive
        return [[0.0, 1.0] for _ in texts]

    monkeypatch.setattr(emb_mod, "create_embeddings_batch_async", _stub_batch)

    resp = test_client.post("/api/v1/embeddings/batch", json={
        "texts": ["a", "b"],
        "model": "text-embedding-3-small",
        "provider": "openai"
    })
    assert resp.status_code == 200, resp.text
    # Headers should be present per parity with single endpoint
    assert resp.headers.get("X-RateLimit-Limit") == "10"
    assert resp.headers.get("X-RateLimit-Remaining") == "7"

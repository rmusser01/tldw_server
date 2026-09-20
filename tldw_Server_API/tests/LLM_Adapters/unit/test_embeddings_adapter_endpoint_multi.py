import math
from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import router as embeddings_router
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User


@pytest.mark.unit
def test_embeddings_endpoint_adapter_multi_with_l2(monkeypatch):
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("LLM_EMBEDDINGS_L2_NORMALIZE", "1")
    monkeypatch.setenv("TEST_MODE", "1")

    async def _no_backpressure(*_args, **_kwargs):  # noqa: D401, ANN001
        return None

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced._check_backpressure_and_quotas",
        _no_backpressure,
        raising=False,
    )

    # Build a lightweight app to avoid full main.py startup
    app = FastAPI()
    app.include_router(embeddings_router, prefix="/api/v1")

    # Provide a dummy user via dependency override
    test_user = User(id=1, username="tester", email="t@example.com", is_active=True)

    async def _mock_user(_request: Request):  # noqa: D401, ARG001
        return test_user

    app.dependency_overrides[get_request_user] = _mock_user

    class _StubAdapter:
        def capabilities(self):  # noqa: D401
            return {"dimensions_default": None, "max_batch_size": 2048}

        def embed(self, request, *, timeout=None):  # noqa: ANN001
            # Return two embeddings with obvious norms
            return {
                "data": [
                    {"index": 0, "embedding": [3.0, 4.0]},
                    {"index": 1, "embedding": [0.0, 5.0]},
                ],
                "model": request.get("model"),
            }

    class _StubRegistry:
        def get_adapter(self, name):  # noqa: ANN001
            return _StubAdapter()

    async def _resolved_credentials(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return ResolvedByokCredentials(
            provider="huggingface",
            api_key="test-huggingface-key",
            app_config=None,
            credential_fields={},
            source="server",
            allowlisted=True,
        )

    with (
        patch(
            "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.get_embeddings_registry",
            return_value=_StubRegistry(),
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced._resolve_embeddings_byok",
            side_effect=_resolved_credentials,
        ),
        TestClient(app) as client,
    ):
        payload = {"model": "huggingface:stub-model", "input": ["a", "b"]}
        r = client.post("/api/v1/embeddings", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        embs = [d["embedding"] for d in body.get("data", [])]
        assert len(embs) == 2
        # Check unit length after L2 normalization
        for v in embs:
            n = math.sqrt(sum(x * x for x in v))
            assert abs(n - 1.0) < 1e-5

    app.dependency_overrides.clear()

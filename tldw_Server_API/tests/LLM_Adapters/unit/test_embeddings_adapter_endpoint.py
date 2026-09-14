from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from fastapi import Request
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User


@pytest.mark.unit
def test_embeddings_endpoint_uses_adapter_when_enabled(monkeypatch):
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("TEST_MODE", "1")

    # Provide a dummy user via dependency override
    test_user = User(id=1, username="tester", email="t@example.com", is_active=True)

    async def _mock_user(_request: Request):
        return test_user

    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_request_user] = _mock_user

    class _StubAdapter:
        def capabilities(self):
            return {"dimensions_default": None, "max_batch_size": 2048}

        def embed(self, request, *, timeout=None):  # noqa: ANN001
            # Return OpenAI-like shape
            return {
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                ],
                "model": request.get("model"),
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            }

    class _StubRegistry:
        def get_adapter(self, name):  # noqa: ANN001
            return _StubAdapter()

    async def _resolved_credentials(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return ResolvedByokCredentials(
            provider="openai",
            api_key="test-openai-key",
            app_config=None,
            credential_fields={},
            source="server",
            allowlisted=True,
        )

    try:
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
            # CSRF token (optional in TEST_MODE)
            resp = client.get("/api/v1/health")
            csrf_token = resp.cookies.get("csrf_token", "")
            headers = {"X-CSRF-Token": csrf_token} if csrf_token else {}
            payload = {
                "model": "text-embedding-3-small",
                "input": "hello world",
            }
            r = client.post("/api/v1/embeddings", json=payload, headers=headers)
            assert r.status_code == 200, r.text
            body = r.json()
            assert isinstance(body, dict)
            assert "data" in body and isinstance(body["data"], list)
            assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3]
            assert body.get("model") in ("text-embedding-3-small", "openai:text-embedding-3-small")
    finally:
        app.dependency_overrides = original_overrides

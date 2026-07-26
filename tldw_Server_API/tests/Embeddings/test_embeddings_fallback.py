import asyncio
from typing import List, Optional
import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


@pytest.fixture(autouse=True)
def _enable_testing_env(monkeypatch):
    monkeypatch.setenv("TESTING", "true")


@pytest.fixture
def client():
    with TestClient(app) as c:
        c.cookies.set("csrf_token", "test-csrf")
        c.headers["X-CSRF-Token"] = "test-csrf"
        c.headers["Authorization"] = "Bearer test-api-key"
        yield c


@pytest.mark.unit
def test_provider_fallback_to_hf(client, monkeypatch):
    async def override_user():
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        return User(id=1, username="u", email="u@x", is_active=True, is_admin=False)

    monkeypatch.setitem(app.dependency_overrides, get_request_user, override_user)

    # Patch the async batch creator to fail for openai and succeed for huggingface
    async def fake_batch_async(
        texts: List[str],
        provider: str,
        model_id: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        metadata: Optional[dict] = None,
        cache_scope_sensitive: bool = False,
    ):
        _ = cache_scope_sensitive
        if provider == "openai":
            raise HTTPException(status_code=503, detail="openai down")
        elif provider == "huggingface":
            # Return simple 384-dim zero vector to simulate HF
            return [[0.0] * 384 for _ in texts]
        raise HTTPException(status_code=400, detail="unknown provider")

    # Patch metrics to avoid registry issues
    class _MC:
        def labels(self, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            return None

    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod
    monkeypatch.setattr(mod, "create_embeddings_batch_async", fake_batch_async, raising=True)
    monkeypatch.setattr(mod, "embedding_fallbacks_total", _MC(), raising=False)
    monkeypatch.setattr(mod, "embedding_provider_failures_total", _MC(), raising=False)

    # Disable synthetic OpenAI so the endpoint uses the async path we patched
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "true")

    # Request declares openai; should fallback to huggingface and still succeed
    payload = {
        "model": "text-embedding-3-small",
        "input": "fallback test"
    }
    r = client.post("/api/v1/embeddings", json=payload)
    assert r.status_code == 200
    data = r.json()
    # Expect model string to indicate the actual fallback model
    assert data["model"] == "huggingface:sentence-transformers/all-MiniLM-L6-v2"
    # Headers reflect fallback
    assert r.headers.get("X-Embeddings-Provider") == "huggingface"
    assert r.headers.get("X-Embeddings-Fallback-From") == "openai"
    emb = data["data"][0]["embedding"]
    assert isinstance(emb, list)
    assert len(emb) == 384


@pytest.mark.unit
def test_no_fallback_when_header_specified(client, monkeypatch):
    async def override_user():
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        return User(id=1, username="u", email="u@x", is_active=True, is_admin=False)

    monkeypatch.setitem(app.dependency_overrides, get_request_user, override_user)

    # Fail for openai, succeed for huggingface; header will disable fallback and keep failure
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
        _ = cache_scope_sensitive
        from fastapi import HTTPException
        if provider == "openai":
            raise HTTPException(status_code=503, detail="openai down")
        return [[0.0] * 384 for _ in texts]

    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod
    monkeypatch.setattr(mod, "create_embeddings_batch_async", fake_batch_async, raising=True)
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "true")

    payload = {"model": "text-embedding-3-small", "input": "no-fallback"}
    r = client.post("/api/v1/embeddings", json=payload, headers={"x-provider": "openai"})
    assert r.status_code == 503
    # No fallback headers expected
    assert r.headers.get("X-Embeddings-Provider") is None or r.headers.get("X-Embeddings-Provider") == "openai"
    assert r.headers.get("X-Embeddings-Fallback-From") is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("error_code", "expected_status"),
    [
        ("invalid_provider_credentials", 503),
        ("credential_store_unavailable", 503),
        ("credential_scope_revoked", 503),
        ("provider_disabled", 403),
        ("model_not_allowed", 403),
    ],
)
def test_legacy_credential_policy_failure_never_uses_provider_fallback(
    client,
    monkeypatch,
    error_code,
    expected_status,
):
    """Credential-policy failures are request terminal, not provider outages."""
    from tldw_Server_API.app.api.v1.endpoints import (
        embeddings_v5_production_enhanced as mod,
    )
    from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
        ProviderOverridePolicyError,
    )
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def override_user():
        return User(id=1, username="u", email="u@x", is_active=True, is_admin=False)

    resolution_calls = []
    provider_calls = []

    async def resolve_credentials(provider, *_args, **_kwargs):
        resolution_calls.append(provider)
        if provider == "openai":
            error = (
                ProviderOverridePolicyError(error_code, provider)
                if error_code in {"provider_disabled", "model_not_allowed"}
                else mod.ByokResolutionError(error_code, provider)
            )
            raise mod._embeddings_credential_http_exception(
                error
            )
        return mod.ResolvedByokCredentials(
            provider=provider,
            api_key="fallback-key-must-not-be-used",
            app_config={},
            credential_fields={},
            source="server",
            allowlisted=True,
        )

    async def provider_dispatch(*, texts, provider, **_kwargs):
        provider_calls.append(provider)
        return [[0.0] * 384 for _ in texts]

    monkeypatch.setitem(app.dependency_overrides, get_request_user, override_user)
    monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.delenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", raising=False)
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", resolve_credentials)
    monkeypatch.setattr(mod, "create_embeddings_batch_async", provider_dispatch)

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "fail closed"},
    )

    assert resolution_calls == ["openai"]
    assert provider_calls == []
    assert response.status_code == expected_status
    assert response.json()["detail"]["error_code"] == error_code
    assert resolution_calls == ["openai"]
    assert provider_calls == []
    assert "fallback-key-must-not-be-used" not in response.text

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ import byok_runtime
from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as overrides_module
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import LLMProviderOverride
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.main import app


@pytest.fixture(autouse=True)
def _testing_env():
    os.environ["TESTING"] = "true"
    os.environ["USE_REAL_OPENAI_IN_TESTS"] = "true"  # force async path
    # Allow fallback even when x-provider header is present for this mapping test suite
    os.environ["EMBEDDINGS_ALLOW_FALLBACK_WITH_HEADER"] = "true"
    yield
    os.environ.pop("TESTING", None)
    os.environ.pop("USE_REAL_OPENAI_IN_TESTS", None)
    os.environ.pop("EMBEDDINGS_ALLOW_FALLBACK_WITH_HEADER", None)


@pytest.fixture(autouse=True)
def _restore_provider_override_state():
    original = overrides_module.get_llm_provider_overrides_snapshot()
    try:
        yield
    finally:
        overrides_module.set_llm_provider_overrides_cache_for_tests(original)
        app.dependency_overrides.pop(get_request_user, None)


@pytest.fixture
def client():
    with TestClient(app) as c:
        c.cookies.set("csrf_token", "x")
        c.headers["X-CSRF-Token"] = "x"
        c.headers["Authorization"] = "Bearer key"
        yield c


def _override_user():


    async def _f():
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        return User(id=1, username="u", email="u@x", is_active=True, is_admin=False)
    return _f


@pytest.mark.unit
def test_fallback_model_mapping_openai_to_hf(client, monkeypatch):
     # Capture the model_id used for fallback call
    calls = {"args": None}

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
        calls["args"] = {"provider": provider, "model_id": model_id}
        if provider == "openai":
            # simulate failure for openai to force fallback
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail="openai down")
        # for HF, return simple vectors
        return [[0.0] * 384 for _ in texts]

    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod
    monkeypatch.setattr(mod, "create_embeddings_batch_async", fake_batch_async, raising=True)

    app.dependency_overrides[get_request_user] = _override_user()
    r = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "openai"},
        json={"input": "fallback mapping", "model": "text-embedding-3-small"}
    )
    assert r.status_code == 200
    assert calls["args"]["provider"] == "huggingface"
    # Must map openai small to HF all-MiniLM-L6-v2 by default
    assert calls["args"]["model_id"] == "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.unit
def test_fallback_and_oauth_refresh_resolve_the_exact_dispatched_model(
    client,
    monkeypatch,
):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    source_model = "source-embedding-model"
    fallback_model = "text-embedding-3-small"
    hf_credentials = ResolvedByokCredentials(
        provider="huggingface",
        api_key="hf-key",
        app_config={},
        credential_fields={},
        source="user",
        allowlisted=True,
        auth_source="api_key",
    )
    initial_oauth = ResolvedByokCredentials(
        provider="openai",
        api_key="oauth-old-key",
        app_config={},
        credential_fields={},
        source="user",
        allowlisted=True,
        auth_source="oauth",
    )
    refreshed_oauth = ResolvedByokCredentials(
        provider="openai",
        api_key="oauth-new-key",
        app_config={},
        credential_fields={},
        source="user",
        allowlisted=True,
        auth_source="oauth",
    )
    resolve_calls = []
    dispatch_calls = []

    async def resolve_credentials(
        provider,
        _current_user,
        _request,
        *,
        model=None,
        force_oauth_refresh=False,
        rejected_credentials=None,
    ):
        resolve_calls.append(
            (provider, model, force_oauth_refresh, rejected_credentials)
        )
        if provider == "huggingface":
            return hf_credentials
        return refreshed_oauth if force_oauth_refresh else initial_oauth

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
        del dimensions, api_url, metadata, cache_scope_sensitive
        dispatch_calls.append((provider, model_id, api_key, tuple(texts)))
        if provider == "huggingface":
            raise HTTPException(status_code=503, detail="huggingface unavailable")
        if api_key == "oauth-old-key":
            raise HTTPException(status_code=401, detail="expired oauth token")
        return [[0.2, 0.8] for _ in texts]

    monkeypatch.delenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", raising=False)
    monkeypatch.setattr(
        mod,
        "_enforce_embedding_policy_decision",
        lambda **_kwargs: SimpleNamespace(
            fallback_chain=["huggingface", "openai"]
        ),
    )
    monkeypatch.setattr(
        mod,
        "map_model_for_provider",
        lambda _src, dst, model: fallback_model if dst == "openai" else model,
    )
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", resolve_credentials)
    monkeypatch.setattr(mod, "create_embeddings_batch_async", fake_batch_async)
    app.dependency_overrides[get_request_user] = _override_user()

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "huggingface"},
        json={"input": "mapped oauth fallback", "model": source_model},
    )

    assert response.status_code == 200
    assert [(provider, model, forced) for provider, model, forced, _ in resolve_calls] == [
        ("huggingface", source_model, False),
        ("openai", fallback_model, False),
        ("openai", fallback_model, True),
    ]
    assert resolve_calls[-1][3] is initial_oauth
    assert dispatch_calls == [
        ("huggingface", source_model, "hf-key", ("mapped oauth fallback",)),
        ("openai", fallback_model, "oauth-old-key", ("mapped oauth fallback",)),
        ("openai", fallback_model, "oauth-new-key", ("mapped oauth fallback",)),
    ]


@pytest.mark.unit
@pytest.mark.concurrent
def test_concurrent_fallback_mapping_policy_change_fails_closed_before_dispatch(
    client,
    monkeypatch,
):
    """A mapped model is checked against the policy current at resolution."""
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    source_model = "source-embedding-model"
    mapped_model = "sentence-transformers/all-MiniLM-L6-v2"
    mapped = threading.Event()
    release_mapping = threading.Event()
    dispatch_calls = []

    def gated_map(_src, destination, model):
        if destination != "huggingface":
            return model
        mapped.set()
        if not release_mapping.wait(10):
            raise TimeoutError("fallback mapping policy gate was not released")
        return mapped_model

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
        del dimensions, api_url, metadata, cache_scope_sensitive
        dispatch_calls.append((provider, model_id, api_key, tuple(texts)))
        if provider == "openai":
            raise HTTPException(status_code=503, detail="openai unavailable")
        return [[0.4, 0.6] for _ in texts]

    monkeypatch.delenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", raising=False)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(
        mod,
        "_enforce_embedding_policy_decision",
        lambda **_kwargs: SimpleNamespace(
            fallback_chain=["openai", "huggingface"]
        ),
    )
    monkeypatch.setattr(mod, "map_model_for_provider", gated_map)
    monkeypatch.setattr(mod, "create_embeddings_batch_async", fake_batch_async)
    app.dependency_overrides[get_request_user] = _override_user()
    openai_override = LLMProviderOverride(
        provider="openai",
        is_enabled=True,
        allowed_models=[source_model],
        api_key="openai-key",
    )
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {
            "openai": openai_override,
            "huggingface": LLMProviderOverride(
                provider="huggingface",
                is_enabled=True,
                allowed_models=[mapped_model],
                api_key="huggingface-old-policy-key",
            ),
        }
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            client.post,
            "/api/v1/embeddings",
            headers={"x-provider": "openai"},
            json={"input": "policy race", "model": source_model},
        )
        try:
            assert mapped.wait(10)
            overrides_module.set_llm_provider_overrides_cache_for_tests(
                {
                    "openai": openai_override,
                    "huggingface": LLMProviderOverride(
                        provider="huggingface",
                        is_enabled=True,
                        allowed_models=[source_model],
                        api_key="huggingface-new-policy-key",
                    ),
                }
            )
        finally:
            release_mapping.set()
        response = future.result(timeout=10)

    assert response.status_code == 403
    assert response.json()["detail"]["error_code"] == "model_not_allowed"
    assert dispatch_calls == [
        ("openai", source_model, "openai-key", ("policy race",)),
    ]

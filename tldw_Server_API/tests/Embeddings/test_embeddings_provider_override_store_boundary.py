"""Public embedding regressions for provider-override store outages."""

from __future__ import annotations

import asyncio
import copy
import threading
from typing import Any

import pytest
from fastapi import BackgroundTasks, HTTPException, Response
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints import (
    embeddings_v5_production_enhanced as embeddings_endpoint,
)
from tldw_Server_API.app.api.v1.schemas.embeddings_models import CreateEmbeddingRequest
from tldw_Server_API.app.core.AuthNZ import byok_runtime
from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as overrides_module
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ResolvedByokCredentials,
    ServerFallbackCredentials,
)
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import LLMProviderOverride
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError


@pytest.fixture(autouse=True)
def healthy_override_cache_between_tests():
    original = overrides_module.get_llm_provider_overrides_snapshot()
    overrides_module.set_llm_provider_overrides_cache_for_tests(original)
    try:
        yield
    finally:
        overrides_module.set_llm_provider_overrides_cache_for_tests(original)


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
@pytest.mark.asyncio
async def test_public_embeddings_store_outage_is_503_before_static_or_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    orchestrator_enabled: bool,
) -> None:
    static_calls: list[str] = []
    dispatch_calls: list[str] = []

    def static_snapshot() -> dict[str, Any]:
        static_calls.append("openai")
        return {"openai_api": {"api_key": "must-not-be-used"}}

    async def engine_boundary(**kwargs):
        await embeddings_endpoint._resolve_embeddings_byok(
            "openai",
            kwargs["current_user"],
            kwargs["request"],
        )
        dispatch_calls.append("provider")
        raise AssertionError("embedding provider must not run while credential storage is unhealthy")

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(
        embeddings_endpoint,
        "load_server_config_snapshot",
        static_snapshot,
    )
    monkeypatch.setattr(embeddings_endpoint, "_create_embedding_with_orchestrator", engine_boundary)
    monkeypatch.setattr(embeddings_endpoint, "_create_embedding_legacy", engine_boundary)
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="last-good-key")},
        healthy=False,
    )

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/embeddings",
            "headers": [],
            "query_string": b"",
        }
    )
    payload = CreateEmbeddingRequest(model="text-embedding-3-small", input="hello")
    user = User(
        id=1,
        username="embedding-user",
        email="embedding-user@example.test",
        is_active=True,
        is_admin=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        await embeddings_endpoint.create_embedding_endpoint(
            request=request,
            embedding_request=payload,
            current_user=user,
            background_tasks=BackgroundTasks(),
            x_provider="openai",
            response=Response(),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {
        "error_code": "credential_store_unavailable",
        "message": "Provider credential storage is temporarily unavailable.",
    }
    assert static_calls == []
    assert dispatch_calls == []


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
@pytest.mark.asyncio
async def test_concurrent_oauth_401_refresh_store_outage_is_503_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    orchestrator_enabled: bool,
) -> None:
    """Concurrent adapter refreshes fail closed without a second provider call."""
    real_resolve = embeddings_endpoint._resolve_embeddings_byok
    initial_credentials = ResolvedByokCredentials(
        provider="openai",
        api_key="oauth-old-key",
        app_config={},
        credential_fields={},
        source="user",
        allowlisted=True,
        auth_source="oauth",
    )
    resolve_calls: list[bool] = []
    static_calls: list[str] = []
    adapter_calls: list[str | None] = []
    simultaneous_dispatch = threading.Barrier(2)

    async def resolve_credentials(
        provider: str,
        current_user: User | None,
        request: Request | None,
        *,
        model: str | None = None,
        force_oauth_refresh: bool = False,
        rejected_credentials: ResolvedByokCredentials | None = None,
    ) -> ResolvedByokCredentials:
        assert model == "text-embedding-3-small"
        assert rejected_credentials is (
            initial_credentials if force_oauth_refresh else None
        )
        resolve_calls.append(force_oauth_refresh)
        if not force_oauth_refresh:
            return initial_credentials
        return await real_resolve(
            provider,
            current_user,
            request,
            model=model,
            force_oauth_refresh=True,
            rejected_credentials=rejected_credentials,
        )

    def static_snapshot() -> dict[str, Any]:
        static_calls.append("openai")
        return {"openai_api": {"api_key": "must-not-be-used"}}

    class ExpiredOAuthAdapter:
        def embed(self, adapter_request):
            adapter_calls.append(adapter_request.get("api_key"))
            simultaneous_dispatch.wait(timeout=5)
            raise ChatAuthenticationError(
                provider="openai-embeddings",
                message="expired OAuth token",
                status_code=401,
            )

    class AdapterRegistry:
        def get_adapter(self, provider: str):
            assert provider == "openai"
            return ExpiredOAuthAdapter()

    async def engine_boundary(**kwargs):
        executor = embeddings_endpoint._EndpointEmbeddingExecutor(
            request=kwargs["request"],
            current_user=kwargs["current_user"],
            user_metadata=None,
        )
        await executor.create_adapter(
            ["hello"],
            provider="openai",
            model="text-embedding-3-small",
            dimensions=None,
        )
        raise AssertionError("an unhealthy credential store must stop adapter execution")

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(
        embeddings_endpoint,
        "load_server_config_snapshot",
        static_snapshot,
    )
    monkeypatch.setattr(embeddings_endpoint, "_resolve_embeddings_byok", resolve_credentials)
    monkeypatch.setattr(embeddings_endpoint, "get_embeddings_registry", lambda: AdapterRegistry())
    monkeypatch.setattr(embeddings_endpoint, "_create_embedding_with_orchestrator", engine_boundary)
    monkeypatch.setattr(embeddings_endpoint, "_create_embedding_legacy", engine_boundary)
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="last-good-key")},
        healthy=False,
    )

    payload = CreateEmbeddingRequest(model="text-embedding-3-small", input="hello")
    user = User(
        id=1,
        username="embedding-user",
        email="embedding-user@example.test",
        is_active=True,
        is_admin=False,
    )

    async def make_request() -> object:
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/v1/embeddings",
                "headers": [],
                "query_string": b"",
            }
        )
        try:
            return await embeddings_endpoint.create_embedding_endpoint(
                request=request,
                embedding_request=payload,
                current_user=user,
                background_tasks=BackgroundTasks(),
                x_provider="openai",
                response=Response(),
            )
        except HTTPException as exc:
            return exc

    results = await asyncio.gather(make_request(), make_request())

    assert len(results) == 2
    for result in results:
        assert isinstance(result, HTTPException)
        assert result.status_code == 503
        assert result.detail == {
            "error_code": "credential_store_unavailable",
            "message": "Provider credential storage is temporarily unavailable.",
        }
    assert resolve_calls.count(False) == 2
    assert resolve_calls.count(True) == 2
    assert adapter_calls == ["oauth-old-key", "oauth-old-key"]
    assert static_calls == []


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_late_override_after_structured_absence_cannot_mix_embedding_adapter_credentials(
    monkeypatch: pytest.MonkeyPatch,
    orchestrator_enabled: bool,
) -> None:
    """Embedding adapters receive the static snapshot selected after atomic absence."""
    structured_absent = threading.Event()
    static_lookup_started = threading.Event()
    release_static_lookup = threading.Event()
    static_calls: list[str] = []
    adapter_calls: list[dict[str, Any]] = []

    def capture_snapshot(provider: str):
        snapshot = overrides_module.capture_provider_override_call_snapshot(provider)
        structured_absent.set()
        return snapshot

    def gated_static_lookup() -> dict[str, Any]:
        assert structured_absent.is_set()
        static_calls.append("openai")
        static_lookup_started.set()
        if not release_static_lookup.wait(10):
            raise TimeoutError("embedding static-key race gate was not released")
        return {"openai_api": {"api_key": "configured-static-key"}}

    class CapturingAdapter:
        def embed(self, adapter_request: dict[str, Any]) -> dict[str, Any]:
            adapter_calls.append(adapter_request)
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    class AdapterRegistry:
        def get_adapter(self, provider: str):
            assert provider == "openai"
            return CapturingAdapter()

    async def engine_adapter_boundary(**kwargs: Any):
        executor = embeddings_endpoint._EndpointEmbeddingExecutor(
            request=kwargs["request"],
            current_user=kwargs["current_user"],
            user_metadata=None,
        )
        return await executor.create_adapter(
            ["hello"],
            provider="openai",
            model="text-embedding-3-small",
            dimensions=None,
        )

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(
        embeddings_endpoint,
        "capture_provider_override_call_snapshot",
        capture_snapshot,
    )
    monkeypatch.setattr(
        embeddings_endpoint,
        "load_server_config_snapshot",
        gated_static_lookup,
    )
    monkeypatch.setattr(embeddings_endpoint, "get_embeddings_registry", lambda: AdapterRegistry())
    monkeypatch.setattr(
        embeddings_endpoint,
        "_create_embedding_with_orchestrator",
        engine_adapter_boundary,
    )
    monkeypatch.setattr(embeddings_endpoint, "_create_embedding_legacy", engine_adapter_boundary)
    overrides_module.set_llm_provider_overrides_cache_for_tests({})

    user = User(
        id=1,
        username="embedding-user",
        email="embedding-user@example.test",
        is_active=True,
        is_admin=False,
    )
    payload = CreateEmbeddingRequest(model="text-embedding-3-small", input="hello")

    def invoke_endpoint() -> object:
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/v1/embeddings",
                "headers": [],
                "query_string": b"",
            }
        )
        return asyncio.run(
            embeddings_endpoint.create_embedding_endpoint(
                request=request,
                embedding_request=payload,
                current_user=user,
                background_tasks=BackgroundTasks(),
                x_provider="openai",
                response=Response(),
            )
        )

    endpoint_task = asyncio.create_task(asyncio.to_thread(invoke_endpoint))
    structured_seen = await asyncio.to_thread(structured_absent.wait, 10)
    static_seen = await asyncio.to_thread(static_lookup_started.wait, 10)
    if structured_seen and static_seen:
        overrides_module.set_llm_provider_overrides_cache_for_tests(
            {
                "openai": LLMProviderOverride(
                    provider="openai",
                    api_key="late-override-key",
                    credential_fields={
                        "base_url": "https://late-override.example/v1",
                    },
                )
            }
        )
    release_static_lookup.set()
    await asyncio.wait_for(endpoint_task, timeout=10)

    assert structured_seen
    assert static_seen
    assert static_calls == ["openai"]
    assert len(adapter_calls) == 1
    assert adapter_calls[0]["api_key"] == "configured-static-key"
    assert adapter_calls[0].get("base_url") != "https://late-override.example/v1"
    assert "late-override.example" not in repr(adapter_calls[0]["app_config"])


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
@pytest.mark.parametrize(
    (
        "provider",
        "model",
        "initial_config",
        "rotated_config",
        "expected_key",
        "expected_base_url",
        "expected_app_config",
    ),
    (
        (
            "openai",
            "text-embedding-3-small",
            {
                "openai_api": {
                    "api_key": "static-key-a",
                    "api_base_url": "https://static-a.example/v1",
                }
            },
            {
                "openai_api": {
                    "api_key": "static-key-b",
                    "api_base_url": "https://static-b.example/v1",
                }
            },
            "static-key-a",
            "https://static-a.example/v1",
            {
                "openai_api": {
                    "api_base_url": "https://static-a.example/v1",
                }
            },
        ),
        (
            "huggingface",
            "sentence-transformers/all-MiniLM-L6-v2",
            {},
            {
                "huggingface_api": {
                    "api_base_url": "https://huggingface-static-b.example/models",
                }
            },
            None,
            None,
            {},
        ),
    ),
    ids=("a-to-b", "absent-to-b"),
)
@pytest.mark.asyncio
async def test_embeddings_static_fallback_freezes_one_config_generation_at_adapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
    orchestrator_enabled: bool,
    provider: str,
    model: str,
    initial_config: dict[str, Any],
    rotated_config: dict[str, Any],
    expected_key: str | None,
    expected_base_url: str | None,
    expected_app_config: dict[str, Any],
) -> None:
    """Embedding adapters cannot receive endpoint config from a later generation."""
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    adapter_calls: list[dict[str, Any]] = []

    def load_static_snapshot() -> dict[str, Any]:
        monkeypatch.setattr(byok_runtime, "loaded_config_data", rotated_config)
        return copy.deepcopy(initial_config)

    class CapturingAdapter:
        def embed(self, adapter_request: dict[str, Any]) -> dict[str, Any]:
            adapter_calls.append(adapter_request)
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    class AdapterRegistry:
        def get_adapter(self, requested_provider: str):
            assert requested_provider == provider
            return CapturingAdapter()

    async def engine_adapter_boundary(**kwargs: Any):
        executor = embeddings_endpoint._EndpointEmbeddingExecutor(
            request=kwargs["request"],
            current_user=kwargs["current_user"],
            user_metadata=None,
        )
        return await executor.create_adapter(
            ["hello"],
            provider=provider,
            model=model,
            dimensions=None,
        )

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    for environment_name in (
        "OPENAI_API_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "MOCK_OPENAI_BASE_URL",
        "HUGGINGFACE_INFERENCE_BASE_URL",
    ):
        monkeypatch.delenv(environment_name, raising=False)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "load_server_config_snapshot", load_static_snapshot)
    monkeypatch.setattr(byok_helpers, "load_server_config_snapshot", load_static_snapshot)
    monkeypatch.setattr(
        embeddings_endpoint,
        "load_server_config_snapshot",
        load_static_snapshot,
    )
    monkeypatch.setattr(byok_runtime, "loaded_config_data", initial_config)
    monkeypatch.setattr(embeddings_endpoint, "get_embeddings_registry", lambda: AdapterRegistry())
    monkeypatch.setattr(
        embeddings_endpoint,
        "_create_embedding_with_orchestrator",
        engine_adapter_boundary,
    )
    monkeypatch.setattr(
        embeddings_endpoint,
        "_create_embedding_legacy",
        engine_adapter_boundary,
    )
    overrides_module.set_llm_provider_overrides_cache_for_tests({})

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/embeddings",
            "headers": [],
            "query_string": b"",
        }
    )
    payload = CreateEmbeddingRequest(model=model, input="hello")
    user = User(
        id=1,
        username="embedding-user",
        email="embedding-user@example.test",
        is_active=True,
        is_admin=False,
    )

    await embeddings_endpoint.create_embedding_endpoint(
        request=request,
        embedding_request=payload,
        current_user=user,
        background_tasks=BackgroundTasks(),
        x_provider=provider,
        response=Response(),
    )

    assert len(adapter_calls) == 1
    assert adapter_calls[0]["api_key"] == expected_key
    assert adapter_calls[0].get("base_url") == expected_base_url
    assert adapter_calls[0]["app_config"] == expected_app_config


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_embedding_override_merges_one_frozen_static_config_at_adapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent override/config rotation cannot split one adapter snapshot."""
    fallback_started = threading.Event()
    release_fallback = threading.Event()
    static_calls: list[str] = []
    base_fallbacks: list[ServerFallbackCredentials | None] = []
    adapter_calls: list[dict[str, Any]] = []
    static_state = {
        "fallback": ServerFallbackCredentials(
            api_key="static-a-key",
            credential_fields={},
            app_config={
                "openai_api": {
                    "api_base_url": "https://static-a.example/v1",
                    "timeout": 17,
                },
                "HTTP": {"connect_timeout": 3},
            },
        )
    }
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                api_key="override-a-key",
                config={"api_base_url": "https://override-a.example/v1"},
            )
        }
    )
    captured = overrides_module.capture_provider_override_call_snapshot("openai")

    class GatedSnapshot:
        provider = captured.provider

        def enforce(self, model: str | None) -> None:
            captured.enforce(model)

        def ensure_healthy(self) -> None:
            captured.ensure_healthy()

        def server_fallback(
            self,
            base_fallback: ServerFallbackCredentials | None = None,
        ) -> ServerFallbackCredentials | None:
            base_fallbacks.append(base_fallback)
            fallback_started.set()
            if not release_fallback.wait(10):
                raise TimeoutError("embedding override/static merge gate was not released")
            return captured.server_fallback(base_fallback)

    def load_static_snapshot() -> dict[str, Any]:
        static_calls.append("openai")
        fallback = copy.deepcopy(static_state["fallback"])
        snapshot = copy.deepcopy(dict(fallback.app_config or {}))
        provider_config = dict(snapshot.get("openai_api") or {})
        provider_config["api_key"] = fallback.api_key
        snapshot["openai_api"] = provider_config
        return snapshot

    class CapturingAdapter:
        def embed(self, adapter_request: dict[str, Any]) -> dict[str, Any]:
            adapter_calls.append(adapter_request)
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    class AdapterRegistry:
        def get_adapter(self, provider: str):
            assert provider == "openai"
            return CapturingAdapter()

    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(
        embeddings_endpoint,
        "capture_provider_override_call_snapshot",
        lambda _provider: GatedSnapshot(),
    )
    monkeypatch.setattr(
        embeddings_endpoint,
        "load_server_config_snapshot",
        load_static_snapshot,
    )
    monkeypatch.setattr(
        embeddings_endpoint,
        "get_embeddings_registry",
        lambda: AdapterRegistry(),
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/embeddings",
            "headers": [],
            "query_string": b"",
        }
    )
    user = User(
        id=1,
        username="embedding-user",
        email="embedding-user@example.test",
        is_active=True,
        is_admin=False,
    )

    def invoke_adapter() -> object:
        executor = embeddings_endpoint._EndpointEmbeddingExecutor(
            request=request,
            current_user=user,
            user_metadata=None,
        )
        return asyncio.run(
            executor.create_adapter(
                ["hello"],
                provider="openai",
                model="text-embedding-3-small",
                dimensions=None,
            )
        )

    adapter_task = asyncio.create_task(asyncio.to_thread(invoke_adapter))
    fallback_seen = await asyncio.to_thread(fallback_started.wait, 10)
    if fallback_seen:
        static_state["fallback"] = ServerFallbackCredentials(
            api_key="static-b-key",
            credential_fields={},
            app_config={
                "openai_api": {
                    "api_base_url": "https://static-b.example/v1",
                    "timeout": 99,
                },
                "HTTP": {"connect_timeout": 8},
            },
        )
        overrides_module.set_llm_provider_overrides_cache_for_tests(
            {
                "openai": LLMProviderOverride(
                    provider="openai",
                    api_key="override-b-key",
                    config={"api_base_url": "https://override-b.example/v1"},
                )
            }
        )
    release_fallback.set()
    await asyncio.wait_for(adapter_task, timeout=10)

    assert fallback_seen
    assert static_calls == ["openai"]
    assert len(base_fallbacks) == 1
    assert base_fallbacks[0] is not None
    assert len(adapter_calls) == 1
    assert adapter_calls[0]["api_key"] == "override-a-key"
    assert adapter_calls[0]["base_url"] == "https://override-a.example/v1"
    assert adapter_calls[0]["app_config"] == {
        "openai_api": {
            "api_base_url": "https://override-a.example/v1",
            "timeout": 17,
        },
        "HTTP": {"connect_timeout": 3},
    }


@pytest.mark.parametrize("orchestrator_enabled", [True, False], ids=("orchestrator", "legacy"))
@pytest.mark.parametrize(
    (
        "provider",
        "model",
        "key_environment",
        "endpoint_environment",
        "initial_endpoint",
        "expected_key",
        "expected_base_url",
    ),
    (
        (
            "openai",
            "text-embedding-3-small",
            "OPENAI_API_KEY",
            "OPENAI_API_BASE_URL",
            "https://openai-env-a.example/v1",
            "openai-env-key",
            "https://openai-env-a.example/v1",
        ),
        (
            "openai",
            "text-embedding-3-small",
            "OPENAI_API_KEY",
            "OPENAI_API_BASE_URL",
            None,
            "openai-env-key",
            "https://api.openai.com/v1",
        ),
        (
            "google",
            "text-embedding-004",
            "GOOGLE_API_KEY",
            "GOOGLE_GEMINI_BASE_URL",
            None,
            "google-env-key",
            "https://generativelanguage.googleapis.com/v1",
        ),
        (
            "huggingface",
            "sentence-transformers/all-MiniLM-L6-v2",
            None,
            "HUGGINGFACE_INFERENCE_BASE_URL",
            None,
            None,
            None,
        ),
        (
            "huggingface",
            "sentence-transformers/all-MiniLM-L6-v2",
            "HUGGINGFACE_API_KEY",
            "HUGGINGFACE_INFERENCE_BASE_URL",
            "https://huggingface-env-a.example/models",
            "huggingface-env-key",
            "https://huggingface-env-a.example/models",
        ),
    ),
    ids=(
        "openai-captured-env-a-to-b",
        "openai-absent-env-to-b",
        "google-absent-env-to-b-official-default",
        "huggingface-keyless-absent-env-to-b-local",
        "huggingface-captured-env-a-to-b",
    ),
)
@pytest.mark.asyncio
async def test_embeddings_adapter_uses_captured_environment_or_immutable_default(
    monkeypatch: pytest.MonkeyPatch,
    orchestrator_enabled: bool,
    provider: str,
    model: str,
    key_environment: str | None,
    endpoint_environment: str,
    initial_endpoint: str | None,
    expected_key: str | None,
    expected_base_url: str | None,
) -> None:
    """Resolved credentials never reread a provider endpoint environment variable."""
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    adapter_calls: list[dict[str, Any]] = []
    original_resolver = embeddings_endpoint._resolve_embeddings_byok

    class CapturingAdapter:
        def embed(self, adapter_request: dict[str, Any]) -> dict[str, Any]:
            adapter_calls.append(adapter_request)
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    class AdapterRegistry:
        def get_adapter(self, requested_provider: str):
            assert requested_provider == provider
            return CapturingAdapter()

    async def resolve_then_rotate_environment(*args: Any, **kwargs: Any):
        credentials = await original_resolver(*args, **kwargs)
        monkeypatch.setenv(endpoint_environment, "https://late-env-b.example/v1")
        return credentials

    async def engine_adapter_boundary(**kwargs: Any):
        executor = embeddings_endpoint._EndpointEmbeddingExecutor(
            request=kwargs["request"],
            current_user=kwargs["current_user"],
            user_metadata=None,
        )
        return await executor.create_adapter(
            ["hello"],
            provider=provider,
            model=model,
            dimensions=None,
        )

    if orchestrator_enabled:
        monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "1")
    else:
        monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "1")
    for environment_name in (
        "OPENAI_API_KEY",
        "OPENAI_API_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "MOCK_OPENAI_BASE_URL",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_GEMINI_BASE_URL",
        "HUGGINGFACE_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACE_INFERENCE_BASE_URL",
    ):
        monkeypatch.delenv(environment_name, raising=False)
    if key_environment is not None:
        monkeypatch.setenv(key_environment, expected_key or "")
    if initial_endpoint is not None:
        monkeypatch.setenv(endpoint_environment, initial_endpoint)

    monkeypatch.setattr(byok_helpers, "load_and_log_configs", lambda **_kwargs: {})
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(
        embeddings_endpoint,
        "_resolve_embeddings_byok",
        resolve_then_rotate_environment,
    )
    monkeypatch.setattr(embeddings_endpoint, "get_embeddings_registry", lambda: AdapterRegistry())
    monkeypatch.setattr(
        embeddings_endpoint,
        "_create_embedding_with_orchestrator",
        engine_adapter_boundary,
    )
    monkeypatch.setattr(
        embeddings_endpoint,
        "_create_embedding_legacy",
        engine_adapter_boundary,
    )
    overrides_module.set_llm_provider_overrides_cache_for_tests({})

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/embeddings",
            "headers": [],
            "query_string": b"",
        }
    )
    user = User(
        id=1,
        username="embedding-user",
        email="embedding-user@example.test",
        is_active=True,
        is_admin=False,
    )

    await embeddings_endpoint.create_embedding_endpoint(
        request=request,
        embedding_request=CreateEmbeddingRequest(model=model, input="hello"),
        current_user=user,
        background_tasks=BackgroundTasks(),
        x_provider=provider,
        response=Response(),
    )

    assert len(adapter_calls) == 1
    assert adapter_calls[0]["api_key"] == expected_key
    assert adapter_calls[0].get("base_url") == expected_base_url


@pytest.mark.parametrize(
    ("late_policy", "expected_code", "pass_model"),
    (
        (
            {"is_enabled": False, "allowed_models": ["text-embedding-3-small"]},
            "provider_disabled",
            False,
        ),
        (
            {"is_enabled": True, "allowed_models": ["text-embedding-3-large"]},
            "model_not_allowed",
            True,
        ),
    ),
)
@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_late_embedding_override_policy_change_blocks_adapter_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    late_policy: dict[str, Any],
    expected_code: str,
    pass_model: bool,
) -> None:
    """Embedding resolution enforces the policy paired with its credentials."""
    adapter_ready = asyncio.Event()
    release_adapter = asyncio.Event()
    outbound_calls: list[str] = []
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/embeddings",
            "headers": [],
            "query_string": b"",
        }
    )
    user = User(
        id=1,
        username="embedding-user",
        email="embedding-user@example.test",
        is_active=True,
        is_admin=False,
    )

    async def adapter_boundary() -> None:
        adapter_ready.set()
        await release_adapter.wait()
        resolve_kwargs = {"model": "text-embedding-3-small"} if pass_model else {}
        await embeddings_endpoint._resolve_embeddings_byok(
            "openai",
            user,
            request,
            **resolve_kwargs,
        )
        outbound_calls.append("openai")

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                is_enabled=True,
                allowed_models=["text-embedding-3-small"],
                api_key="enabled-key",
            )
        }
    )

    adapter_task = asyncio.create_task(adapter_boundary())
    await asyncio.wait_for(adapter_ready.wait(), timeout=10)
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                api_key="late-policy-key",
                **late_policy,
            )
        }
    )
    release_adapter.set()

    with pytest.raises(HTTPException) as exc_info:
        await asyncio.wait_for(adapter_task, timeout=10)

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error_code"] == expected_code
    assert outbound_calls == []


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_sequential_embedding_executors_coalesce_oauth_refresh_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale sibling request adopts a completed OAuth refresh without rotating again."""
    current = {"api_key": "oauth-stale-key", "generation": "generation-stale"}
    token_calls = 0
    resolver_calls: list[tuple[bool, str | None]] = []

    async def resolve_credentials(
        provider: str,
        *,
        force_oauth_refresh: bool = False,
        rejected_credential_generation: str | None = None,
        **_kwargs: Any,
    ) -> ResolvedByokCredentials:
        nonlocal token_calls
        resolver_calls.append(
            (force_oauth_refresh, rejected_credential_generation)
        )
        if force_oauth_refresh and (
            rejected_credential_generation is None
            or rejected_credential_generation == current["generation"]
        ):
            token_calls += 1
            current.update(
                api_key=f"oauth-refreshed-key-{token_calls}",
                generation=f"generation-refreshed-{token_calls}",
            )
        return ResolvedByokCredentials(
            provider=provider,
            api_key=current["api_key"],
            app_config={},
            credential_fields={},
            source="user",
            allowlisted=True,
            auth_source="oauth",
            _credential_generation=current["generation"],
        )

    monkeypatch.setattr(
        embeddings_endpoint,
        "resolve_byok_credentials",
        resolve_credentials,
    )
    overrides_module.set_llm_provider_overrides_cache_for_tests({})
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/embeddings",
            "headers": [],
            "query_string": b"",
        }
    )
    user = User(
        id=1,
        username="embedding-user",
        email="embedding-user@example.test",
        is_active=True,
        is_admin=False,
    )
    executor_a = embeddings_endpoint._EndpointEmbeddingExecutor(
        request=request,
        current_user=user,
        user_metadata=None,
    )
    executor_b = embeddings_endpoint._EndpointEmbeddingExecutor(
        request=request,
        current_user=user,
        user_metadata=None,
    )

    initial_a = await executor_a._resolve_provider_credentials(
        "openai",
        "text-embedding-3-small",
    )
    initial_b = await executor_b._resolve_provider_credentials(
        "openai",
        "text-embedding-3-small",
    )
    refreshed_a = await executor_a._resolve_provider_credentials(
        "openai",
        "text-embedding-3-small",
        force_oauth_refresh=True,
    )
    refreshed_b = await executor_b._resolve_provider_credentials(
        "openai",
        "text-embedding-3-small",
        force_oauth_refresh=True,
    )

    assert initial_a.api_key == initial_b.api_key == "oauth-stale-key"
    assert refreshed_a.api_key == refreshed_b.api_key == "oauth-refreshed-key-1"
    assert token_calls == 1
    assert resolver_calls == [
        (False, None),
        (False, None),
        (True, "generation-stale"),
        (True, "generation-stale"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocked_model",
    (
        "text-embedding-3-large",
        "text-embedding-3-small ",
    ),
    ids=("different-model", "distinct-wire-model"),
)
async def test_embedding_executor_rechecks_policy_for_each_adapter_model(
    monkeypatch: pytest.MonkeyPatch,
    blocked_model: str,
) -> None:
    """A provider cache entry for one model cannot authorize another model."""
    allowed_model = "text-embedding-3-small"
    resolve_calls: list[tuple[str, str | None]] = []
    adapter_calls: list[str] = []
    credentials = ResolvedByokCredentials(
        provider="openai",
        api_key="model-scoped-key",
        app_config={},
        credential_fields={},
        source="user",
        allowlisted=True,
        auth_source="api_key",
    )

    async def resolve_credentials(
        provider: str,
        _current_user: User | None,
        _request: Request | None,
        *,
        model: str | None = None,
        **_kwargs: Any,
    ) -> ResolvedByokCredentials:
        resolve_calls.append((provider, model))
        if model == blocked_model:
            raise HTTPException(
                status_code=403,
                detail={
                    "error_code": "model_not_allowed",
                    "message": "The selected model is not allowed for this provider.",
                },
            )
        return credentials

    class CapturingAdapter:
        def embed(self, adapter_request: dict[str, Any]) -> dict[str, Any]:
            adapter_calls.append(adapter_request["model"])
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    class AdapterRegistry:
        def get_adapter(self, provider: str):
            assert provider == "openai"
            return CapturingAdapter()

    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "1")
    monkeypatch.setattr(
        embeddings_endpoint,
        "_resolve_embeddings_byok",
        resolve_credentials,
    )
    monkeypatch.setattr(
        embeddings_endpoint,
        "get_embeddings_registry",
        lambda: AdapterRegistry(),
    )
    executor = embeddings_endpoint._EndpointEmbeddingExecutor(
        request=Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/v1/embeddings",
                "headers": [],
                "query_string": b"",
            }
        ),
        current_user=User(
            id=1,
            username="embedding-user",
            email="embedding-user@example.test",
            is_active=True,
            is_admin=False,
        ),
        user_metadata=None,
    )

    allowed = await executor.create_adapter(
        ["allowed"],
        provider="openai",
        model=allowed_model,
        dimensions=None,
    )
    with pytest.raises(HTTPException) as exc_info:
        await executor.create_adapter(
            ["blocked"],
            provider="openai",
            model=blocked_model,
            dimensions=None,
        )

    assert allowed is not None
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error_code"] == "model_not_allowed"
    assert resolve_calls == [
        ("openai", allowed_model),
        ("openai", blocked_model),
    ]
    assert adapter_calls == [allowed_model]

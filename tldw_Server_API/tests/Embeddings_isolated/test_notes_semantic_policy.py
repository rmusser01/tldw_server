from __future__ import annotations

from types import SimpleNamespace

import pytest

import tldw_Server_API.app.core.Notes_Graph.semantic_embeddings as semantic_embeddings
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_embeddings import (
    NotesEmbeddingExecutor,
    PendingSemanticConfig,
    RunMemoryEmbeddingCache,
    SemanticEmbeddingSystemError,
    build_notes_semantic_orchestrator,
)

pytestmark = pytest.mark.unit


class RecordingAdapter:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def capabilities(self) -> dict[str, object]:
        return {"model_revision": "capability-revision"}

    def embed(self, request: dict[str, object], *, timeout: float | None = None) -> dict[str, object]:
        del timeout
        self.requests.append(request)
        return self.response


class FailingAdapter(RecordingAdapter):
    def embed(self, request: dict[str, object], *, timeout: float | None = None) -> dict[str, object]:
        del request, timeout
        raise RuntimeError("credential-shaped-provider-detail")


class Registry:
    def __init__(self, adapter: RecordingAdapter | None) -> None:
        self.adapter = adapter

    def get_adapter(self, provider: str) -> RecordingAdapter | None:
        assert provider == "openai"
        return self.adapter


def _credentials(
    *,
    source: str = "server_default",
    base_url: str = "https://api.openai.com/v1",
) -> ResolvedByokCredentials:
    return ResolvedByokCredentials(
        provider="openai",
        api_key="not-logged",
        app_config={"openai_api": {"api_base_url": base_url}},
        credential_fields={"base_url": base_url},
        source=source,
        allowlisted=True,
        status=ByokResolutionStatus.RESOLVED,
    )


def _config(**overrides: object) -> PendingSemanticConfig:
    values = {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "model_revision": None,
        "endpoint_origin": "https://api.openai.com",
        "credential_source": "server_default",
        "consented": True,
        "dimensions": 2,
    }
    values.update(overrides)
    return PendingSemanticConfig(**values)


def test_notes_orchestrator_disables_fallback_and_uses_only_run_memory_cache(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    class RecordingOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(semantic_embeddings, "EmbeddingRequestOrchestrator", RecordingOrchestrator)

    runtime = build_notes_semantic_orchestrator(
        _config(),
        user_id="7",
        executor=sentinel,
    )

    assert isinstance(captured["cache"], RunMemoryEmbeddingCache)
    assert captured["executor"] is sentinel
    assert captured["allow_fallback_with_header"] is False
    assert captured["allowed_providers"] == {"openai"}
    assert captured["allowed_models"] == {"text-embedding-3-small"}
    assert captured["settings_fallback_chain"] == {}
    assert captured["settings_fallback_model_map"] == {}
    assert captured["dimension_policy"] == "ignore"
    assert runtime.orchestrator.__class__ is RecordingOrchestrator


@pytest.mark.asyncio
async def test_executor_resolves_only_explicit_durable_credentials_without_request() -> None:
    resolver_calls: list[dict[str, object]] = []
    adapter = RecordingAdapter(
        {
            "data": [{"index": 0, "embedding": [1.0, 2.0]}],
            "model": "text-embedding-3-small",
            "model_revision": "response-revision",
        }
    )

    async def resolver(provider: str, **kwargs: object) -> ResolvedByokCredentials:
        resolver_calls.append({"provider": provider, **kwargs})
        return _credentials()

    executor = NotesEmbeddingExecutor(
        config=_config(),
        user_id="7",
        credential_resolver=resolver,
        adapter_registry=Registry(adapter),
    )

    vectors = await executor.create(
        ["public input"],
        provider="openai",
        model="text-embedding-3-small",
        dimensions=2,
    )

    assert vectors == [[1.0, 2.0]]
    assert resolver_calls == [
        {
            "provider": "openai",
            "user_id": 7,
            "request": None,
            "required_source": "server_default",
        }
    ]
    assert executor.execution_identity().model_revision == "response-revision"
    app_config = adapter.requests[0]["app_config"]
    assert isinstance(app_config, dict)
    assert app_config["HTTP"]["allow_redirects"] is False
    assert app_config["HTTP"]["allow_cross_host_redirects"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["request", "user"])
async def test_executor_rejects_request_only_or_wrong_durable_source(source: str) -> None:
    async def resolver(provider: str, **kwargs: object) -> ResolvedByokCredentials:
        del provider, kwargs
        return _credentials(source=source)

    executor = NotesEmbeddingExecutor(
        config=_config(),
        user_id="7",
        credential_resolver=resolver,
        adapter_registry=Registry(RecordingAdapter({})),
    )

    with pytest.raises(SemanticEmbeddingSystemError, match="durable_credentials_unavailable"):
        await executor.create(
            ["input"],
            provider="openai",
            model="text-embedding-3-small",
            dimensions=2,
        )


@pytest.mark.asyncio
async def test_executor_rejects_endpoint_model_and_cross_origin_redirect_drift() -> None:
    async def wrong_origin(provider: str, **kwargs: object) -> ResolvedByokCredentials:
        del provider, kwargs
        return _credentials(base_url="https://proxy.example/v1")

    endpoint_executor = NotesEmbeddingExecutor(
        config=_config(),
        user_id="7",
        credential_resolver=wrong_origin,
        adapter_registry=Registry(RecordingAdapter({})),
    )
    with pytest.raises(SemanticEmbeddingSystemError, match="endpoint_origin_mismatch"):
        await endpoint_executor.create(
            ["input"], provider="openai", model="text-embedding-3-small", dimensions=2
        )

    async def resolver(provider: str, **kwargs: object) -> ResolvedByokCredentials:
        del provider, kwargs
        return _credentials()

    model_executor = NotesEmbeddingExecutor(
        config=_config(),
        user_id="7",
        credential_resolver=resolver,
        adapter_registry=Registry(
            RecordingAdapter(
                {
                    "data": [{"index": 0, "embedding": [1.0, 2.0]}],
                    "model": "text-embedding-3-large",
                }
            )
        ),
    )
    with pytest.raises(SemanticEmbeddingSystemError, match="provider_model_drift"):
        await model_executor.create(
            ["input"], provider="openai", model="text-embedding-3-small", dimensions=2
        )

    redirect_executor = NotesEmbeddingExecutor(
        config=_config(),
        user_id="7",
        credential_resolver=resolver,
        adapter_registry=Registry(
            RecordingAdapter(
                {
                    "data": [{"index": 0, "embedding": [1.0, 2.0]}],
                    "model": "text-embedding-3-small",
                    "redirect_origins": ["https://api.openai.com", "https://other.example"],
                }
            )
        ),
    )
    with pytest.raises(SemanticEmbeddingSystemError, match="cross_origin_redirect"):
        await redirect_executor.create(
            ["input"], provider="openai", model="text-embedding-3-small", dimensions=2
        )


@pytest.mark.asyncio
async def test_executor_rejects_unavailable_pinned_provider() -> None:
    executor = NotesEmbeddingExecutor(
        config=_config(),
        user_id="7",
        credential_resolver=SimpleNamespace(),
        adapter_registry=Registry(None),
    )

    with pytest.raises(SemanticEmbeddingSystemError, match="provider_unavailable"):
        await executor.create(
            ["input"], provider="openai", model="text-embedding-3-small", dimensions=2
        )


@pytest.mark.asyncio
async def test_executor_maps_credential_and_provider_failures_to_content_free_codes() -> None:
    async def failed_resolver(provider: str, **kwargs: object) -> ResolvedByokCredentials:
        del provider, kwargs
        raise RuntimeError("credential-shaped-resolution-detail")

    credential_executor = NotesEmbeddingExecutor(
        config=_config(),
        user_id="7",
        credential_resolver=failed_resolver,
        adapter_registry=Registry(RecordingAdapter({})),
    )
    with pytest.raises(SemanticEmbeddingSystemError, match="durable_credentials_unavailable") as exc_info:
        await credential_executor.create(
            ["input"], provider="openai", model="text-embedding-3-small", dimensions=2
        )
    assert "credential-shaped" not in str(exc_info.value)

    async def resolver(provider: str, **kwargs: object) -> ResolvedByokCredentials:
        del provider, kwargs
        return _credentials()

    provider_executor = NotesEmbeddingExecutor(
        config=_config(),
        user_id="7",
        credential_resolver=resolver,
        adapter_registry=Registry(FailingAdapter({})),
    )
    with pytest.raises(SemanticEmbeddingSystemError, match="provider_execution_failed") as exc_info:
        await provider_executor.create(
            ["input"], provider="openai", model="text-embedding-3-small", dimensions=2
        )
    assert "credential-shaped" not in str(exc_info.value)

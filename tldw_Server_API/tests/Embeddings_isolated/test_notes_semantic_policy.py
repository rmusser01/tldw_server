from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from httpx import URL

import tldw_Server_API.app.core.Notes_Graph.semantic_embeddings as semantic_embeddings
from tldw_Server_API.app.core.AuthNZ.byok_config import is_runtime_base_url_override
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_content import build_semantic_chunks
from tldw_Server_API.app.core.Notes_Graph.semantic_embeddings import (
    NotesEmbeddingExecutor,
    NotesSemanticEmbedder,
    PendingSemanticConfig,
    ResolvedSemanticConfig,
    RunMemoryEmbeddingCache,
    SemanticEmbeddingSystemError,
    build_notes_semantic_orchestrator,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_settings import SemanticIndexSettings

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


class ProviderRegistry:
    def __init__(self, provider: str, adapter: object) -> None:
        self.provider = provider
        self.adapter = adapter

    def get_adapter(self, provider: str) -> object | None:
        assert provider == self.provider
        return self.adapter


class AnyProviderRegistry:
    def __init__(self, adapter: object) -> None:
        self.adapter = adapter

    def get_adapter(self, _provider: str) -> object:
        return self.adapter


def _credentials(
    *,
    source: str = "server_default",
    base_url: str = "https://api.openai.com/v1",
    provider: str = "openai",
    api_key: str = "not-logged",
) -> ResolvedByokCredentials:
    return ResolvedByokCredentials(
        provider=provider,
        api_key=api_key,
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
    assert adapter.requests[0]["base_url"] == "https://api.openai.com/v1"
    assert is_runtime_base_url_override(adapter.requests[0]["_runtime_base_url_override"])


@pytest.mark.asyncio
async def test_executor_origin_matches_httpx_idna_runtime_target() -> None:
    runtime_base_url = "https://faß.de:443/v1"
    expected_origin = "https://xn--fa-hia.de:443"
    adapter = RecordingAdapter(
        {
            "data": [{"index": 0, "embedding": [1.0, 2.0]}],
            "model": "text-embedding-3-small",
        }
    )

    async def resolver(provider: str, **kwargs: object) -> ResolvedByokCredentials:
        del provider, kwargs
        return _credentials(base_url=runtime_base_url)

    executor = NotesEmbeddingExecutor(
        config=_config(endpoint_origin=expected_origin),
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

    dispatched_base_url = adapter.requests[0]["base_url"]
    assert vectors == [[1.0, 2.0]]
    assert dispatched_base_url == runtime_base_url
    assert URL(str(dispatched_base_url)).raw_host == b"xn--fa-hia.de"
    assert executor.execution_identity().endpoint_origin == expected_origin


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "google", "huggingface"])
async def test_executor_request_is_accepted_by_real_provider_adapter(
    provider: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = "https://embeddings.example/v1"
    captured: list[dict[str, object]] = []

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            if provider == "openai":
                return {"data": [{"index": 0, "embedding": [1.0, 2.0]}]}
            if provider == "google":
                return {"embeddings": [{"values": [1.0, 2.0]}]}
            return [[1.0, 2.0]]

    class Client:
        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def post(self, url: str, **kwargs: object) -> Response:
            captured.append({"url": url, **kwargs})
            return Response()

    if provider == "openai":
        from tldw_Server_API.app.core import http_client
        from tldw_Server_API.app.core.LLM_Calls.providers.openai_embeddings_adapter import (
            OpenAIEmbeddingsAdapter,
        )

        monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_OPENAI", "1")

        def fetch(**kwargs: object) -> Response:
            captured.append(kwargs)
            return Response()

        monkeypatch.setattr(http_client, "fetch", fetch)
        adapter = OpenAIEmbeddingsAdapter()
        model = "text-embedding-3-small"
    elif provider == "google":
        import tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter as module
        from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
            GoogleEmbeddingsAdapter,
        )

        monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE", "1")
        monkeypatch.setattr(module, "create_client", lambda **_kwargs: Client())
        adapter = GoogleEmbeddingsAdapter()
        model = "text-embedding-004"
    else:
        import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter as module
        from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_embeddings_adapter import (
            HuggingFaceEmbeddingsAdapter,
        )

        monkeypatch.setenv("LLM_EMBEDDINGS_NATIVE_HTTP_HUGGINGFACE", "1")
        monkeypatch.setattr(module, "create_client", lambda **_kwargs: Client())
        adapter = HuggingFaceEmbeddingsAdapter()
        model = "sentence-transformers/all-MiniLM-L6-v2"

    async def resolver(name: str, **kwargs: object) -> ResolvedByokCredentials:
        del name, kwargs
        return _credentials(base_url=base_url, provider=provider)

    executor = NotesEmbeddingExecutor(
        config=_config(
            provider=provider,
            model=model,
            endpoint_origin="https://embeddings.example",
        ),
        user_id="7",
        credential_resolver=resolver,
        adapter_registry=ProviderRegistry(provider, adapter),
    )

    vectors = await executor.create(
        ["provider contract input"],
        provider=provider,
        model=model,
        dimensions=2,
    )

    assert vectors == [[1.0, 2.0]]
    assert captured
    assert "embeddings.example" in str(captured[0]["url"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    [
        "anthropic",
        "cohere",
        "mistral",
        "openrouter",
        "voyage",
        "mlx",
        "unknown-provider",
    ],
)
async def test_executor_rejects_every_provider_outside_the_semantic_catalog(
    provider: str,
) -> None:
    resolver_called = False

    async def resolver(_provider: str, **_kwargs: object) -> ResolvedByokCredentials:
        nonlocal resolver_called
        resolver_called = True
        return _credentials(provider=provider)

    executor = NotesEmbeddingExecutor(
        config=_config(
            provider=provider,
            model="embedding-model",
            endpoint_origin="https://embeddings.example",
        ),
        user_id="7",
        credential_resolver=resolver,
        adapter_registry=AnyProviderRegistry(
            RecordingAdapter(
                {
                    "data": [{"index": 0, "embedding": [1.0, 2.0]}],
                    "model": "embedding-model",
                }
            )
        ),
    )

    with pytest.raises(SemanticEmbeddingSystemError, match="provider_unavailable"):
        await executor.create(
            ["input"],
            provider=provider,
            model="embedding-model",
            dimensions=2,
        )

    assert resolver_called is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "base_url", "endpoint_origin"),
    [
        ("openai", "https://api.openai.com/v1", "https://api.openai.com"),
        (
            "google",
            "https://generativelanguage.googleapis.com/v1",
            "https://generativelanguage.googleapis.com",
        ),
        (
            "huggingface",
            "https://api-inference.huggingface.co/models",
            "https://api-inference.huggingface.co",
        ),
    ],
)
async def test_executor_admits_every_provider_in_the_semantic_catalog(
    provider: str,
    base_url: str,
    endpoint_origin: str,
) -> None:
    adapter = RecordingAdapter(
        {
            "data": [{"index": 0, "embedding": [1.0, 2.0]}],
            "model": "embedding-model",
        }
    )

    async def resolver(_provider: str, **_kwargs: object) -> ResolvedByokCredentials:
        return _credentials(provider=provider, base_url=base_url)

    executor = NotesEmbeddingExecutor(
        config=_config(
            provider=provider,
            model="embedding-model",
            endpoint_origin=endpoint_origin,
        ),
        user_id="7",
        credential_resolver=resolver,
        adapter_registry=ProviderRegistry(provider, adapter),
    )

    vectors = await executor.create(
        ["input"],
        provider=provider,
        model="embedding-model",
        dimensions=2,
    )

    assert vectors == [[1.0, 2.0]]
    assert adapter.requests[0]["base_url"] == base_url


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
async def test_executor_rejects_endpoint_and_model_drift() -> None:
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
        await endpoint_executor.create(["input"], provider="openai", model="text-embedding-3-small", dimensions=2)

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
        await model_executor.create(["input"], provider="openai", model="text-embedding-3-small", dimensions=2)


@pytest.mark.asyncio
async def test_executor_pins_normalized_full_endpoint_before_second_dispatch() -> None:
    adapter = RecordingAdapter(
        {
            "data": [{"index": 0, "embedding": [1.0, 2.0]}],
            "model": "text-embedding-3-small",
        }
    )
    credentials = iter(
        [
            _credentials(base_url="  https://api.openai.com/v1/  "),
            _credentials(base_url="https://api.openai.com/proxy/v1/"),
        ]
    )

    async def resolver(provider: str, **kwargs: object) -> ResolvedByokCredentials:
        del provider, kwargs
        return next(credentials)

    executor = NotesEmbeddingExecutor(
        config=_config(),
        user_id="7",
        credential_resolver=resolver,
        adapter_registry=Registry(adapter),
    )

    assert await executor.create(
        ["first payload"],
        provider="openai",
        model="text-embedding-3-small",
        dimensions=2,
    ) == [[1.0, 2.0]]
    with pytest.raises(SemanticEmbeddingSystemError, match="endpoint_identity_mismatch"):
        await executor.create(
            ["second payload must not dispatch"],
            provider="openai",
            model="text-embedding-3-small",
            dimensions=2,
        )

    assert len(adapter.requests) == 1
    assert adapter.requests[0]["base_url"] == "https://api.openai.com/v1"


@pytest.mark.asyncio
async def test_executor_allows_key_rotation_at_exact_normalized_endpoint() -> None:
    adapter = RecordingAdapter(
        {
            "data": [{"index": 0, "embedding": [1.0, 2.0]}],
            "model": "text-embedding-3-small",
        }
    )
    credentials = iter(
        [
            _credentials(base_url="https://api.openai.com/v1/", api_key="key-one"),
            _credentials(base_url=" https://api.openai.com/v1 ", api_key="key-two"),
        ]
    )

    async def resolver(provider: str, **kwargs: object) -> ResolvedByokCredentials:
        del provider, kwargs
        return next(credentials)

    executor = NotesEmbeddingExecutor(
        config=_config(),
        user_id="7",
        credential_resolver=resolver,
        adapter_registry=Registry(adapter),
    )

    for payload in ("first payload", "second payload"):
        assert await executor.create(
            [payload],
            provider="openai",
            model="text-embedding-3-small",
            dimensions=2,
        ) == [[1.0, 2.0]]

    assert [request["api_key"] for request in adapter.requests] == ["key-one", "key-two"]
    assert [request["base_url"] for request in adapter.requests] == [
        "https://api.openai.com/v1",
        "https://api.openai.com/v1",
    ]


@pytest.mark.asyncio
async def test_executor_rejects_unavailable_pinned_provider() -> None:
    executor = NotesEmbeddingExecutor(
        config=_config(),
        user_id="7",
        credential_resolver=SimpleNamespace(),
        adapter_registry=Registry(None),
    )

    with pytest.raises(SemanticEmbeddingSystemError, match="provider_unavailable"):
        await executor.create(["input"], provider="openai", model="text-embedding-3-small", dimensions=2)


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
        await credential_executor.create(["input"], provider="openai", model="text-embedding-3-small", dimensions=2)
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
        await provider_executor.create(["input"], provider="openai", model="text-embedding-3-small", dimensions=2)
    assert "credential-shaped" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_real_google_batch_result_count_is_validated_by_notes_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter as module
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    calls: list[str] = []

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"embeddings": [{"values": [1.0, 2.0]}]}

    class Client:
        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def post(self, url: str, **_kwargs: object) -> Response:
            calls.append(url)
            return Response()

    monkeypatch.setattr(module, "create_client", lambda **_kwargs: Client())

    async def resolver(provider: str, **kwargs: object) -> ResolvedByokCredentials:
        del provider, kwargs
        return _credentials(
            provider="google",
            base_url="https://embeddings.example/v1",
        )

    executor = NotesEmbeddingExecutor(
        config=_config(
            provider="google",
            model="text-embedding-004",
            endpoint_origin="https://embeddings.example",
        ),
        user_id="7",
        credential_resolver=resolver,
        adapter_registry=ProviderRegistry("google", GoogleEmbeddingsAdapter()),
    )

    with pytest.raises(SemanticEmbeddingSystemError, match="invalid_vectors"):
        await executor.create(
            ["first note", "second note"],
            provider="google",
            model="text-embedding-004",
            dimensions=2,
        )

    assert calls == ["https://embeddings.example/v1/models/text-embedding-004:batchEmbedContents"]


@pytest.mark.asyncio
async def test_pinned_google_dimension_is_transmitted_and_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter as module
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    payloads: list[dict[str, object]] = []

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"embeddings": [{"values": [1.0, 2.0, 3.0]}]}

    class Client:
        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def post(self, _url: str, **kwargs: object) -> Response:
            payloads.append(kwargs["json"])
            return Response()

    monkeypatch.setattr(module, "create_client", lambda **_kwargs: Client())

    async def resolver(provider: str, **kwargs: object) -> ResolvedByokCredentials:
        del provider, kwargs
        return _credentials(
            provider="google",
            base_url="https://embeddings.example/v1",
        )

    async def record_usage(**kwargs: object) -> None:
        del kwargs

    settings = SemanticIndexSettings(max_chunk_code_points=4)
    config = ResolvedSemanticConfig(
        provider="google",
        model="text-embedding-004",
        model_revision=None,
        endpoint_origin="https://embeddings.example",
        credential_source="server_default",
        dimensions=2,
    )
    runtime = build_notes_semantic_orchestrator(
        config,
        user_id="7",
        settings=settings,
        credential_resolver=resolver,
        adapter_registry=ProviderRegistry("google", GoogleEmbeddingsAdapter()),
    )
    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda run_config, user_id: runtime,
        usage_logger=record_usage,
        settings=settings,
    )
    chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title="",
        content="abcd",
        content_version=1,
        settings=settings,
    )

    with pytest.raises(SemanticEmbeddingSystemError, match="dimension_mismatch"):
        await embedder.embed_chunks(chunks, config, user_id="7")

    assert payloads == [
        {
            "requests": [
                {
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": "abcd"}]},
                    "embedContentConfig": {"outputDimensionality": 2},
                }
            ]
        }
    ]
    assert "outputDimensionality" not in payloads[0]["requests"][0]


@pytest.mark.asyncio
async def test_cancelled_real_google_batch_records_failure_without_later_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter as module
    from tldw_Server_API.app.core.LLM_Calls.providers.google_embeddings_adapter import (
        GoogleEmbeddingsAdapter,
    )

    request_started = threading.Event()
    release_request = threading.Event()
    client_closed = threading.Event()
    calls: list[str] = []
    usage_calls: list[dict[str, object]] = []

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "embeddings": [
                    {"values": [1.0, 0.0]},
                    {"values": [0.0, 1.0]},
                ]
            }

    class Client:
        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: object) -> bool:
            client_closed.set()
            return False

        def post(self, url: str, **_kwargs: object) -> Response:
            calls.append(url)
            request_started.set()
            assert release_request.wait(10)
            return Response()

    monkeypatch.setattr(module, "create_client", lambda **_kwargs: Client())

    async def resolver(provider: str, **kwargs: object) -> ResolvedByokCredentials:
        del provider, kwargs
        return _credentials(
            provider="google",
            base_url="https://embeddings.example/v1",
        )

    async def record_usage(**kwargs: object) -> None:
        usage_calls.append(kwargs)

    settings = SemanticIndexSettings(max_chunk_code_points=4)
    config = ResolvedSemanticConfig(
        provider="google",
        model="text-embedding-004",
        model_revision=None,
        endpoint_origin="https://embeddings.example",
        credential_source="server_default",
        dimensions=2,
    )
    runtime = build_notes_semantic_orchestrator(
        config,
        user_id="7",
        settings=settings,
        credential_resolver=resolver,
        adapter_registry=ProviderRegistry("google", GoogleEmbeddingsAdapter()),
    )
    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda run_config, user_id: runtime,
        usage_logger=record_usage,
        settings=settings,
    )
    chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title="",
        content="abcdefgh",
        content_version=1,
        settings=settings,
    )

    task = asyncio.create_task(embedder.embed_chunks(chunks, config, user_id="7"))
    assert await asyncio.to_thread(request_started.wait, 10)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release_request.set()

    assert await asyncio.to_thread(client_closed.wait, 10)
    assert calls == ["https://embeddings.example/v1/models/text-embedding-004:batchEmbedContents"]
    assert len(usage_calls) == 1
    assert usage_calls[0]["status"] == 502
    assert usage_calls[0]["usage_metadata"] == {
        "attempt_status": "failed",
        "cache_hit_count": 0,
        "cache_miss_count": 2,
        "provider_input_count": 2,
        "provider_request_count": 1,
    }

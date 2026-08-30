"""Credential-runtime propagation tests for authenticated unified RAG endpoints."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator
from contextvars import ContextVar
from functools import wraps
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import BackgroundTasks, HTTPException, Response
from loguru import logger
from starlette.requests import Request

import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_endpoint
import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as unified_pipeline_module
from tldw_Server_API.app.core.AuthNZ import byok_runtime, provider_credential_runtime
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
)
from tldw_Server_API.app.core.RAG.rag_service.agentic_chunker import (
    agentic_rag_pipeline as production_agentic_rag_pipeline,
)
from tldw_Server_API.app.core.RAG.rag_service.generation import AnswerGenerator
from tldw_Server_API.app.core.RAG.rag_service.request_bundle import ResolvedRequestBundle
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.streaming_executor import stream_rag_events
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    UnifiedSearchResult,
    unified_batch_pipeline,
    unified_rag_pipeline,
)

pytestmark = pytest.mark.unit
_SENTINEL_SECRET = "rag-runtime-secret-must-not-leak"
_BROWSER_PROVIDER_SECRET = "browser-provider-secret-must-not-reach-adapter"


class _RecordingRuntime:
    created: list[_RecordingRuntime] = []

    def __init__(self, **scope: Any) -> None:
        self.scope = scope
        self.close_calls = 0
        self.secret = _SENTINEL_SECRET
        type(self).created.append(self)

    async def close(self) -> None:
        self.close_calls += 1


def _request(path: str = "/api/v1/rag/search") -> Request:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "query_string": b"",
        }
    )
    principal = AuthPrincipal(
        kind="user",
        user_id=42,
        username="rag-user",
        team_ids=[7, 8],
        org_ids=[11, 12],
        active_team_id=8,
        active_org_id=11,
    )
    request.state.auth = AuthContext(principal=principal)
    return request


def _user() -> Any:
    return SimpleNamespace(id=42, id_int=42, username="rag-user")


def test_trusted_scope_does_not_infer_singleton_team_ahead_of_active_org() -> None:
    request = _request()
    principal = request.state.auth.principal
    principal.team_ids = [7]
    principal.org_ids = [11]
    principal.active_team_id = None
    principal.active_org_id = 11

    _user_id, team_ids, org_ids, _trusted = rag_endpoint._trusted_credential_runtime_scope(
        request,
        _user(),
    )

    assert team_ids == []
    assert org_ids == [11]


@pytest.mark.parametrize("kind", ["team", "org"])
@pytest.mark.parametrize("active_id", ["malformed", 99])
def test_rag_trusted_scope_rejects_invalid_active_id(kind: str, active_id: Any) -> None:
    request = _request()
    setattr(request.state.auth.principal, f"active_{kind}_id", active_id)

    with pytest.raises(ByokResolutionError) as exc_info:
        rag_endpoint._trusted_credential_runtime_scope(request, _user())

    assert exc_info.value.code == "credential_scope_revoked"


def _db(path: str) -> Any:
    return SimpleNamespace(db_path=path, db_path_str=path)


def _bundle(strategy: str = "standard") -> ResolvedRequestBundle:
    payload = {
        "query": "credential runtime",
        "strategy": strategy,
        "sources": ["media_db"],
        "search_mode": "fts",
        "top_k": 3,
        "min_score": 0.0,
        "enable_generation": True,
        "generation_provider": "openai",
    }
    resolved_request = ResolvedRAGRequest(
        query="credential runtime",
        strategy=strategy,
        payload=payload,
        index_namespace=None,
        rag_profile=None,
        user_id="42",
        feedback_user_id="42",
    )
    retrieval_plan = RetrievalPlan(
        query="credential runtime",
        sources=("media_db",),
        search_mode="fts",
        top_k=3,
        min_score=0.0,
        index_namespace=None,
    )
    return ResolvedRequestBundle(
        resolved_request=resolved_request,
        retrieval_plan=retrieval_plan,
        pipeline_kwargs={
            "query": "credential runtime",
            "resolved_request": resolved_request,
            "retrieval_plan": retrieval_plan,
        },
    )


def _result() -> UnifiedSearchResult:
    return UnifiedSearchResult(
        documents=[],
        query="credential runtime",
        generated_answer="safe answer",
    )


def _install_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingRuntime.created.clear()
    monkeypatch.setattr(
        rag_endpoint,
        "ProviderCredentialRuntime",
        _RecordingRuntime,
        raising=False,
    )


def _install_common_endpoint_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bundle: ResolvedRequestBundle,
) -> None:
    monkeypatch.setattr(
        rag_endpoint,
        "_build_standard_request_bundle",
        lambda *args, **kwargs: bundle,
    )

    async def no_usage_log(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(rag_endpoint, "_log_rag_queries_for_org", no_usage_log)
    monkeypatch.setattr(rag_endpoint, "rag_result_from_unified_search_result", lambda result: result)
    monkeypatch.setattr(
        rag_endpoint,
        "rag_result_to_response",
        lambda result: {"query": result.query, "answer": result.generated_answer},
    )


def _assert_trusted_scope(runtime: _RecordingRuntime) -> None:
    assert runtime.scope["user_id"] == 42  # nosec B101
    assert runtime.scope["team_ids"] == [8]  # nosec B101
    assert runtime.scope["org_ids"] == [11]  # nosec B101
    assert runtime.scope["trusted_base_url_override"] is False  # nosec B101
    assert "fallback_resolver" not in runtime.scope  # nosec B101
    assert callable(runtime.scope["override_snapshot_resolver"])  # nosec B101


@pytest.mark.asyncio
async def test_standard_search_passes_ephemeral_runtime_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle("standard")
    original_payload = dict(bundle.resolved_request.payload)
    captured: dict[str, Any] = {}
    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    _install_runtime(monkeypatch)
    _install_common_endpoint_fakes(monkeypatch, bundle=bundle)

    async def fake_pipeline(**kwargs: Any) -> UnifiedSearchResult:
        captured.update(kwargs)
        return _result()

    monkeypatch.setattr(rag_endpoint, "unified_rag_pipeline", fake_pipeline)
    try:
        response = await rag_endpoint.unified_search_endpoint(
            request_raw=_request(),
            request=rag_endpoint.UnifiedRAGRequest(query="credential runtime"),
            background_tasks=BackgroundTasks(),
            current_user=_user(),
            media_db=_db("media.db"),
            chacha_db=_db("notes.db"),
            prompts_db=_db("prompts.db"),
            collections_db=SimpleNamespace(),
        )
    finally:
        logger.remove(sink_id)

    runtime = _RecordingRuntime.created[0]
    _assert_trusted_scope(runtime)
    assert captured["credential_runtime"] is runtime  # nosec B101
    assert bundle.resolved_request.payload == original_payload  # nosec B101
    assert "credential_runtime" not in bundle.resolved_request.payload  # nosec B101
    assert _SENTINEL_SECRET not in json.dumps(response)  # nosec B101
    rendered_logs = "".join(logs)
    assert _SENTINEL_SECRET not in rendered_logs  # nosec B101
    assert "credential runtime" not in rendered_logs  # nosec B101
    assert "query_len=18" in rendered_logs  # nosec B101
    assert runtime.close_calls == 1  # nosec B101


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize("streaming", [False, True], ids=["non-stream", "stream"])
async def test_knowledge_qa_routes_isolate_configured_server_credentials_at_adapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
) -> None:
    """Knowledge QA keeps each configured server key/model pair request-owned."""
    from tldw_Server_API.app.core.Chat import chat_service

    request_label: ContextVar[str] = ContextVar("rag_acceptance_request_label")
    snapshots = {
        "alpha": {
            "openai_api": {
                "api_key": "configured-server-key-alpha",
                "model": "configured-model-alpha",
            }
        },
        "beta": {
            "openai_api": {
                "api_key": "configured-server-key-beta",
                "model": "configured-model-beta",
            }
        },
    }
    models = {
        "alpha": "configured-model-alpha",
        "beta": "configured-model-beta",
    }
    adapter_calls: list[dict[str, Any]] = []
    adapter_providers: list[str] = []
    both_adapter_calls_entered = asyncio.Event()
    release_adapter_calls = asyncio.Event()

    class HealthyAbsentOverrideSnapshot:
        def enforce(self, _model: str | None) -> None:
            return None

        def ensure_healthy(self) -> None:
            return None

        def server_fallback(self, base_fallback: Any = None) -> Any:
            return base_fallback

    class GatedOpenAIAdapter:
        async_chat_is_native = True

        async def _capture(self, request: dict[str, Any]) -> None:
            adapter_calls.append(
                {
                    "api_key": request.get("api_key"),
                    "model": request.get("model"),
                    "stream": request.get("stream") is True,
                    "credentials_resolved": request.get("credentials_resolved"),
                }
            )
            if len(adapter_calls) == 2:
                both_adapter_calls_entered.set()
            await release_adapter_calls.wait()

        async def achat(self, request: dict[str, Any]) -> dict[str, Any]:
            await self._capture(request)
            return {
                "choices": [
                    {
                        "message": {
                            "content": f"answer from {request.get('model')}",
                        }
                    }
                ]
            }

        async def astream(
            self,
            request: dict[str, Any],
        ) -> AsyncIterator[dict[str, Any]]:
            await self._capture(request)
            yield {
                "choices": [
                    {
                        "delta": {
                            "content": f"answer from {request.get('model')}",
                        }
                    }
                ]
            }

    adapter = GatedOpenAIAdapter()

    class Registry:
        def get_adapter(self, provider: str) -> GatedOpenAIAdapter:
            adapter_providers.append(provider)
            return adapter

    def load_server_snapshot() -> dict[str, Any]:
        return snapshots[request_label.get()]

    async def no_usage_log(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def acceptance_pipeline(**kwargs: Any) -> UnifiedSearchResult:
        query = str(kwargs.get("query") or "")
        documents = [
            {
                "id": f"document-{query}",
                "content": "Configured server credentials stay request-owned.",
                "metadata": {"title": "Credential isolation evidence"},
                "score": 1.0,
            }
        ]
        if kwargs.get("enable_generation") is False:
            return UnifiedSearchResult(documents=documents, query=query)

        generated = await AnswerGenerator(
            provider=kwargs.get("generation_provider"),
            model=kwargs.get("generation_model"),
            credential_runtime=kwargs["credential_runtime"],
        ).generate(
            query=query,
            context="Configured server credentials stay request-owned.",
        )
        return UnifiedSearchResult(
            documents=documents,
            query=query,
            generated_answer=str(generated["answer"]),
        )

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(
        provider_credential_runtime,
        "load_server_config_snapshot",
        load_server_snapshot,
    )
    monkeypatch.setattr(
        rag_endpoint,
        "capture_provider_override_call_snapshot",
        lambda _provider: HealthyAbsentOverrideSnapshot(),
    )
    monkeypatch.setattr(rag_endpoint, "_resolve_kanban_db_path", lambda *_args: "kanban.db")
    monkeypatch.setattr(rag_endpoint, "_log_rag_queries_for_org", no_usage_log)
    monkeypatch.setattr(rag_endpoint, "unified_rag_pipeline", acceptance_pipeline)
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: Registry())

    async def invoke(label: str) -> Any:
        token = request_label.set(label)
        try:
            request = rag_endpoint.UnifiedRAGRequest(
                query=f"credential isolation {label}",
                sources=["media_db"],
                enable_generation=True,
                enable_cache=False,
                enable_reranking=False,
                enable_pre_retrieval_clarification=False,
                generation_provider="openai",
                generation_model=models[label],
                api_key=_BROWSER_PROVIDER_SECRET,
            )
            if not streaming:
                return await rag_endpoint.unified_search_endpoint(
                    request_raw=_request(),
                    request=request,
                    background_tasks=BackgroundTasks(),
                    current_user=_user(),
                    media_db=_db(f"media-{label}.db"),
                    chacha_db=_db(f"notes-{label}.db"),
                    prompts_db=_db(f"prompts-{label}.db"),
                    collections_db=SimpleNamespace(),
                )

            response = await rag_endpoint.unified_search_stream_endpoint(
                request_raw=_request("/api/v1/rag/search/stream"),
                request=request,
                current_user=_user(),
                media_db=_db(f"media-{label}.db"),
                chacha_db=_db(f"notes-{label}.db"),
                prompts_db=_db(f"prompts-{label}.db"),
                collections_db=SimpleNamespace(),
            )
            return [chunk async for chunk in response.body_iterator]
        finally:
            request_label.reset(token)

    tasks = [
        asyncio.create_task(invoke("alpha")),
        asyncio.create_task(invoke("beta")),
    ]
    results: list[Any] = []
    try:
        await asyncio.wait_for(both_adapter_calls_entered.wait(), timeout=5)
        release_adapter_calls.set()
        results = list(await asyncio.gather(*tasks))
    finally:
        release_adapter_calls.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert adapter_providers == ["openai", "openai"]  # nosec B101
    assert {
        (
            call["api_key"],
            call["model"],
            call["stream"],
            call["credentials_resolved"],
        )
        for call in adapter_calls
    } == {
        (
            "configured-server-key-alpha",
            "configured-model-alpha",
            streaming,
            True,
        ),
        (
            "configured-server-key-beta",
            "configured-model-beta",
            streaming,
            True,
        ),
    }
    serialized_results = repr(results)
    assert _BROWSER_PROVIDER_SECRET not in serialized_results  # nosec B101
    assert "configured-server-key-alpha" not in serialized_results  # nosec B101
    assert "configured-server-key-beta" not in serialized_results  # nosec B101


@pytest.mark.asyncio
async def test_agentic_search_passes_same_runtime_without_putting_it_in_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle("agentic")
    captured: dict[str, Any] = {}
    _install_runtime(monkeypatch)
    _install_common_endpoint_fakes(monkeypatch, bundle=bundle)

    @wraps(production_agentic_rag_pipeline)
    async def fake_agentic_pipeline(**kwargs: Any) -> UnifiedSearchResult:
        captured.update(kwargs)
        return _result()

    monkeypatch.setattr(rag_endpoint, "agentic_rag_pipeline", fake_agentic_pipeline)
    await rag_endpoint.unified_search_endpoint(
        request_raw=_request(),
        request=rag_endpoint.UnifiedRAGRequest(query="credential runtime", strategy="agentic"),
        background_tasks=BackgroundTasks(),
        current_user=_user(),
        media_db=_db("media.db"),
        chacha_db=_db("notes.db"),
        prompts_db=_db("prompts.db"),
        collections_db=SimpleNamespace(),
    )

    runtime = _RecordingRuntime.created[0]
    assert captured["credential_runtime"] is runtime  # nosec B101
    assert "credential_runtime" not in bundle.resolved_request.payload  # nosec B101
    assert runtime.close_calls == 1  # nosec B101


@pytest.mark.asyncio
async def test_stream_runtime_lives_until_body_iterator_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle("standard")
    captured: dict[str, Any] = {}
    _install_runtime(monkeypatch)
    _install_common_endpoint_fakes(monkeypatch, bundle=bundle)

    async def fake_stream_rag_events(**kwargs: Any):
        captured.update(kwargs)
        yield {"type": "delta", "text": "safe"}

    monkeypatch.setattr(rag_endpoint, "stream_rag_events", fake_stream_rag_events)
    response = await rag_endpoint.unified_search_stream_endpoint(
        request_raw=_request("/api/v1/rag/search/stream"),
        request=rag_endpoint.UnifiedRAGRequest(query="credential runtime", enable_generation=True),
        current_user=_user(),
        media_db=_db("media.db"),
        chacha_db=_db("notes.db"),
        prompts_db=_db("prompts.db"),
        collections_db=SimpleNamespace(),
    )

    assert _RecordingRuntime.created == []  # nosec B101
    chunks = [chunk async for chunk in response.body_iterator]

    runtime = _RecordingRuntime.created[0]
    assert captured["extra_context"]["credential_runtime"] is runtime  # nosec B101
    assert runtime.close_calls == 1  # nosec B101
    assert (
        _SENTINEL_SECRET
        not in b"".join(chunk if isinstance(chunk, bytes) else chunk.encode() for chunk in chunks).decode()
    )  # nosec B101


@pytest.mark.asyncio
async def test_stream_does_not_create_runtime_when_body_is_never_iterated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle("standard")
    _install_runtime(monkeypatch)
    _install_common_endpoint_fakes(monkeypatch, bundle=bundle)

    response = await rag_endpoint.unified_search_stream_endpoint(
        request_raw=_request("/api/v1/rag/search/stream"),
        request=rag_endpoint.UnifiedRAGRequest(query="credential runtime", enable_generation=True),
        current_user=_user(),
        media_db=_db("media.db"),
        chacha_db=_db("notes.db"),
        prompts_db=_db("prompts.db"),
        collections_db=SimpleNamespace(),
    )

    assert response.body_iterator is not None  # nosec B101
    assert _RecordingRuntime.created == []  # nosec B101


@pytest.mark.asyncio
async def test_stream_aclose_before_first_iteration_does_not_create_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle("standard")
    _install_runtime(monkeypatch)
    _install_common_endpoint_fakes(monkeypatch, bundle=bundle)

    response = await rag_endpoint.unified_search_stream_endpoint(
        request_raw=_request("/api/v1/rag/search/stream"),
        request=rag_endpoint.UnifiedRAGRequest(query="credential runtime", enable_generation=True),
        current_user=_user(),
        media_db=_db("media.db"),
        chacha_db=_db("notes.db"),
        prompts_db=_db("prompts.db"),
        collections_db=SimpleNamespace(),
    )

    await response.body_iterator.aclose()

    assert _RecordingRuntime.created == []  # nosec B101


@pytest.mark.asyncio
async def test_batch_passes_one_runtime_and_closes_after_all_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle("standard")
    captured: dict[str, Any] = {}
    _install_runtime(monkeypatch)

    async def fake_limit_checker(**kwargs: Any) -> None:
        return None

    async def fake_batch_pipeline(**kwargs: Any) -> list[UnifiedSearchResult]:
        captured.update(kwargs)
        return [_result(), _result()]

    async def no_usage_log(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(rag_endpoint, "require_within_limit", lambda *args, **kwargs: fake_limit_checker)
    monkeypatch.setattr(rag_endpoint, "_build_batch_request_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(rag_endpoint, "unified_batch_pipeline", fake_batch_pipeline)
    monkeypatch.setattr(rag_endpoint, "_log_rag_queries_for_org", no_usage_log)

    await rag_endpoint.unified_batch_endpoint(
        request_raw=_request("/api/v1/rag/batch"),
        response=Response(),
        request=rag_endpoint.UnifiedBatchRequest(queries=["one", "two"]),
        background_tasks=BackgroundTasks(),
        current_user=_user(),
        principal=AuthPrincipal(kind="user", user_id=42),
        media_db=_db("media.db"),
        chacha_db=_db("notes.db"),
        prompts_db=_db("prompts.db"),
    )

    runtime = _RecordingRuntime.created[0]
    assert captured["credential_runtime"] is runtime  # nosec B101
    assert len(_RecordingRuntime.created) == 1  # nosec B101
    assert runtime.close_calls == 1  # nosec B101


@pytest.mark.asyncio
async def test_checkpoint_batch_cancellation_waits_for_started_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release_normally = asyncio.Event()
    release_cancellation_cleanup = asyncio.Event()
    child_tasks: set[asyncio.Task[Any]] = set()
    cancelled_tasks: set[asyncio.Task[Any]] = set()

    async def blocking_pipeline(**kwargs: Any) -> UnifiedSearchResult:
        task = asyncio.current_task()
        assert task is not None  # nosec B101
        child_tasks.add(task)
        if len(child_tasks) == 2:
            started.set()
        try:
            await release_normally.wait()
        except asyncio.CancelledError:
            cancelled_tasks.add(task)
            await release_cancellation_cleanup.wait()
            raise
        return _result()

    monkeypatch.setenv("RAG_BATCH_DISABLE_CLUSTERING", "1")
    monkeypatch.setattr(unified_pipeline_module, "unified_rag_pipeline", blocking_pipeline)
    batch_task = asyncio.create_task(
        unified_batch_pipeline(
            queries=["one", "two"],
            max_concurrent=2,
            on_query_done=lambda *args: None,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    batch_task.cancel()
    for _ in range(20):
        if len(cancelled_tasks) == 2:
            break
        await asyncio.sleep(0)

    try:
        assert len(cancelled_tasks) == 2  # nosec B101
        assert not batch_task.done()  # nosec B101
    finally:
        release_cancellation_cleanup.set()
        release_normally.set()
        try:
            await batch_task
        except asyncio.CancelledError:
            pass
        await asyncio.gather(*child_tasks, return_exceptions=True)

    assert all(task.done() for task in child_tasks)  # nosec B101


@pytest.mark.asyncio
async def test_resume_batch_passes_ephemeral_runtime_without_checkpoint_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.RAG.rag_service import checkpoint as checkpoint_module

    bundle = _bundle("standard")
    captured: dict[str, Any] = {}
    checkpoint = SimpleNamespace(
        is_complete=False,
        total_items=1,
        completed_items=0,
        config={"queries": ["remaining"], "max_concurrent": 1},
        results=[],
    )
    _install_runtime(monkeypatch)

    class FakeCheckpointManager:
        def load_by_id(self, checkpoint_id: str) -> Any:
            assert checkpoint_id == "checkpoint-1"  # nosec B101
            return checkpoint

        def save_progress(self, state: Any, payload: dict[str, Any]) -> Any:
            return state

        def save_batch_progress(self, state: Any, payload: list[dict[str, Any]]) -> Any:
            return state

    async def fake_batch_pipeline(**kwargs: Any) -> list[UnifiedSearchResult]:
        captured.update(kwargs)
        return [_result()]

    monkeypatch.setattr(checkpoint_module, "CheckpointManager", FakeCheckpointManager)
    monkeypatch.setattr(rag_endpoint, "_build_resume_batch_request", lambda *args, **kwargs: object())
    monkeypatch.setattr(rag_endpoint, "_build_batch_request_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(rag_endpoint, "unified_batch_pipeline", fake_batch_pipeline)

    await rag_endpoint.resume_batch_endpoint(
        checkpoint_id="checkpoint-1",
        request_raw=_request("/api/v1/rag/batch/resume/checkpoint-1"),
        response=Response(),
        background_tasks=BackgroundTasks(),
        current_user=_user(),
        principal=AuthPrincipal(
            kind="user",
            user_id=42,
            permissions=["media.read", "system.configure"],
        ),
        media_db=_db("media.db"),
        chacha_db=_db("notes.db"),
        prompts_db=_db("prompts.db"),
    )

    runtime = _RecordingRuntime.created[0]
    assert captured["credential_runtime"] is runtime  # nosec B101
    assert "credential_runtime" not in checkpoint.config  # nosec B101
    assert runtime.scope["user_id"] is None  # nosec B101
    assert runtime.scope["team_ids"] == []  # nosec B101
    assert runtime.scope["org_ids"] == []  # nosec B101
    assert runtime.scope["trusted_base_url_override"] is False  # nosec B101
    assert runtime.close_calls == 1  # nosec B101


@pytest.mark.asyncio
async def test_stream_runtime_closes_when_body_iterator_is_closed_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle("standard")
    keep_open = asyncio.Event()
    _install_runtime(monkeypatch)
    _install_common_endpoint_fakes(monkeypatch, bundle=bundle)

    async def fake_stream_rag_events(**kwargs: Any):
        yield {"type": "delta", "text": "first"}
        await keep_open.wait()

    monkeypatch.setattr(rag_endpoint, "stream_rag_events", fake_stream_rag_events)
    response = await rag_endpoint.unified_search_stream_endpoint(
        request_raw=_request("/api/v1/rag/search/stream"),
        request=rag_endpoint.UnifiedRAGRequest(query="credential runtime", enable_generation=True),
        current_user=_user(),
        media_db=_db("media.db"),
        chacha_db=_db("notes.db"),
        prompts_db=_db("prompts.db"),
        collections_db=SimpleNamespace(),
    )

    iterator = response.body_iterator
    await iterator.__anext__()
    await iterator.aclose()

    assert _RecordingRuntime.created[0].close_calls == 1  # nosec B101


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_stream_endpoint_closes_each_rag_stream_before_its_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle("standard")
    lifecycle: dict[str, list[str]] = {"a": [], "b": []}

    class OrderedRuntime:
        created = 0

        def __init__(self, **_scope: Any) -> None:
            self.label = ("a", "b")[type(self).created]
            type(self).created += 1

        async def close(self) -> None:
            lifecycle[self.label].append("runtime_close")

    async def fake_stream_rag_events(**kwargs: Any):
        runtime = kwargs["extra_context"]["credential_runtime"]
        try:
            yield {"type": "delta", "text": runtime.label}
            await asyncio.Event().wait()
        finally:
            lifecycle[runtime.label].append("rag_stream_close")

    monkeypatch.setattr(rag_endpoint, "ProviderCredentialRuntime", OrderedRuntime)
    _install_common_endpoint_fakes(monkeypatch, bundle=bundle)
    monkeypatch.setattr(rag_endpoint, "stream_rag_events", fake_stream_rag_events)

    async def build_and_close() -> None:
        response = await rag_endpoint.unified_search_stream_endpoint(
            request_raw=_request("/api/v1/rag/search/stream"),
            request=rag_endpoint.UnifiedRAGRequest(
                query="credential runtime",
                enable_generation=True,
            ),
            current_user=_user(),
            media_db=_db("media.db"),
            chacha_db=_db("notes.db"),
            prompts_db=_db("prompts.db"),
            collections_db=SimpleNamespace(),
        )
        iterator = response.body_iterator
        await iterator.__anext__()
        await iterator.aclose()

    await asyncio.gather(build_and_close(), build_and_close())

    assert lifecycle == {
        "a": ["rag_stream_close", "runtime_close"],
        "b": ["rag_stream_close", "runtime_close"],
    }


def test_checkpoint_sanitizer_excludes_runtime() -> None:
    runtime = _RecordingRuntime(user_id=42)

    sanitized = rag_endpoint._sanitize_checkpoint_config_for_persistence(
        {"query": "safe", "credential_runtime": runtime}
    )

    assert sanitized == {"query": "safe"}  # nosec B101
    assert _SENTINEL_SECRET not in json.dumps(sanitized)  # nosec B101


@pytest.mark.parametrize(
    ("error", "status_code", "error_code"),
    [
        (ChatBadRequestError("raw bad request", provider="openai"), 400, "provider_request_invalid"),
        (
            ChatAuthenticationError(_SENTINEL_SECRET, provider="openai"),
            502,
            "provider_authentication_failed",
        ),
        (
            ChatConfigurationError(_SENTINEL_SECRET, provider="openai"),
            503,
            "provider_configuration_invalid",
        ),
        (
            ChatConfigurationError(
                _SENTINEL_SECRET,
                provider="openai",
                error_code="missing_provider_credentials",
            ),
            503,
            "missing_provider_credentials",
        ),
        (
            ByokResolutionError("credential_store_unavailable", "openai"),
            503,
            "credential_store_unavailable",
        ),
    ],
)
def test_typed_provider_errors_map_to_bounded_http_responses(
    error: BaseException,
    status_code: int,
    error_code: str,
) -> None:
    mapped = rag_endpoint._rag_provider_http_exception(error)

    assert isinstance(mapped, HTTPException)  # nosec B101
    assert mapped.status_code == status_code  # nosec B101
    assert mapped.detail["error_code"] == error_code  # nosec B101
    assert _SENTINEL_SECRET not in json.dumps(mapped.detail)  # nosec B101


@pytest.mark.asyncio
async def test_streaming_credential_failure_is_terminal_and_sanitized() -> None:
    bundle = _bundle("standard")

    async def failing_pipeline(**kwargs: Any) -> UnifiedSearchResult:
        raise ByokResolutionError("credential_store_unavailable", "openai")

    events = [
        event
        async for event in stream_rag_events(
            resolved_request=bundle.resolved_request,
            retrieval_plan=bundle.retrieval_plan,
            standard_pipeline=failing_pipeline,
            extra_context={"credential_runtime": object()},
        )
    ]

    assert events == [  # nosec B101
        {
            "schema_version": 1,
            "type": "error",
            "code": "credential_store_unavailable",
            "status_code": 503,
            "upstream_dispatched": True,
            "output_emitted": False,
            "allow_non_stream_fallback": False,
            "message": "Provider credential storage is temporarily unavailable.",
        }
    ]


@pytest.mark.asyncio
async def test_streaming_generation_context_retains_runtime_identity() -> None:
    bundle = _bundle("standard")
    runtime = object()
    captured: dict[str, Any] = {}

    async def empty_pipeline(**kwargs: Any) -> UnifiedSearchResult:
        return UnifiedSearchResult(documents=[], query=str(kwargs.get("query") or ""))

    async def capture_generation(context: Any, **kwargs: Any) -> Any:
        captured["credential_runtime"] = context.credential_runtime

        async def empty_stream():
            if False:
                yield ""

        context.stream_generator = empty_stream()
        return context

    events = [
        event
        async for event in stream_rag_events(
            resolved_request=bundle.resolved_request,
            retrieval_plan=bundle.retrieval_plan,
            standard_pipeline=empty_pipeline,
            extra_context={
                "credential_runtime": runtime,
                "generate_streaming_response": capture_generation,
            },
        )
    ]

    assert captured["credential_runtime"] is runtime  # nosec B101
    assert [event["type"] for event in events] == [  # nosec B101
        "contexts",
        "reasoning",
        "complete",
    ]
    assert events[-1]["output_emitted"] is False  # nosec B101


@pytest.mark.asyncio
async def test_agentic_typed_failure_bypasses_raw_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle("agentic")
    _install_runtime(monkeypatch)
    _install_common_endpoint_fakes(monkeypatch, bundle=bundle)

    async def failing_agentic_pipeline(**kwargs: Any) -> UnifiedSearchResult:
        raise ChatAuthenticationError(_SENTINEL_SECRET, provider="openai")

    monkeypatch.setattr(rag_endpoint, "agentic_rag_pipeline", failing_agentic_pipeline)
    with pytest.raises(HTTPException) as exc_info:
        await rag_endpoint.unified_search_endpoint(
            request_raw=_request(),
            request=rag_endpoint.UnifiedRAGRequest(query="credential runtime", strategy="agentic"),
            background_tasks=BackgroundTasks(),
            current_user=_user(),
            media_db=_db("media.db"),
            chacha_db=_db("notes.db"),
            prompts_db=_db("prompts.db"),
            collections_db=SimpleNamespace(),
        )

    assert exc_info.value.status_code == 502  # nosec B101
    assert exc_info.value.detail["error_code"] == "provider_authentication_failed"  # nosec B101
    assert _SENTINEL_SECRET not in json.dumps(exc_info.value.detail)  # nosec B101
    assert exc_info.value.__cause__ is None  # nosec B101
    assert exc_info.value.__context__ is None  # nosec B101
    assert _RecordingRuntime.created[0].close_calls == 1  # nosec B101


@pytest.mark.asyncio
async def test_runtime_construction_typed_failure_is_detached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credential-runtime construction never retains private provider details."""

    def failing_runtime(*_args: Any, **_kwargs: Any) -> _RecordingRuntime:
        raise ChatAuthenticationError(_SENTINEL_SECRET, provider="openai")

    monkeypatch.setattr(rag_endpoint, "_build_credential_runtime", failing_runtime)

    with pytest.raises(HTTPException) as exc_info:
        await rag_endpoint.unified_search_endpoint(
            request_raw=_request(),
            request=rag_endpoint.UnifiedRAGRequest(query="credential runtime"),
            background_tasks=BackgroundTasks(),
            current_user=_user(),
            media_db=_db("media.db"),
            chacha_db=_db("notes.db"),
            prompts_db=_db("prompts.db"),
            collections_db=SimpleNamespace(),
        )

    assert exc_info.value.status_code == 502  # nosec B101
    assert exc_info.value.detail["error_code"] == "provider_authentication_failed"  # nosec B101
    assert _SENTINEL_SECRET not in json.dumps(exc_info.value.detail)  # nosec B101
    assert exc_info.value.__cause__ is None  # nosec B101
    assert exc_info.value.__context__ is None  # nosec B101


@pytest.mark.asyncio
async def test_agentic_untyped_failure_fails_closed_without_fabricated_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle("agentic")
    _install_runtime(monkeypatch)
    _install_common_endpoint_fakes(monkeypatch, bundle=bundle)

    raw_failure = (
        "provider request failed at https://upstream.invalid/v1/chat "
        "body=credential-runtime-secret-response"
    )

    async def failing_agentic_pipeline(**_kwargs: Any) -> UnifiedSearchResult:
        raise RuntimeError(raw_failure)

    log_records: list[Any] = []
    sink_id = logger.add(log_records.append)
    monkeypatch.setattr(rag_endpoint, "agentic_rag_pipeline", failing_agentic_pipeline)
    try:
        with pytest.raises(HTTPException) as exc_info:
            await rag_endpoint.unified_search_endpoint(
                request_raw=_request(),
                request=rag_endpoint.UnifiedRAGRequest(
                    query="credential runtime",
                    strategy="agentic",
                    debug_mode=True,
                ),
                background_tasks=BackgroundTasks(),
                current_user=_user(),
                media_db=_db("media.db"),
                chacha_db=_db("notes.db"),
                prompts_db=_db("prompts.db"),
                collections_db=SimpleNamespace(),
            )
    finally:
        logger.remove(sink_id)

    serialized = json.dumps(exc_info.value.detail)
    rendered_logs = "\n".join(str(message) for message in log_records)
    for fragment in (
        raw_failure,
        "upstream.invalid",
        "credential-runtime-secret-response",
    ):
        assert fragment not in serialized  # nosec B101
        assert fragment not in rendered_logs  # nosec B101
    assert exc_info.value.status_code == 500  # nosec B101
    assert exc_info.value.detail == "Search failed due to an internal error."  # nosec B101
    assert exc_info.value.__cause__ is None  # nosec B101
    assert exc_info.value.__context__ is None  # nosec B101

    fallback_logs = [
        message.record
        for message in log_records
        if message.record["message"]
        == "Agentic RAG pipeline failed"
    ]
    assert len(fallback_logs) == 1  # nosec B101
    assert fallback_logs[0]["exception"] is None  # nosec B101
    assert _RecordingRuntime.created[0].close_calls == 1  # nosec B101


@pytest.mark.asyncio
@pytest.mark.parametrize("debug_mode", [False, True], ids=["normal", "debug"])
async def test_standard_untyped_failure_has_bounded_http_error_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    debug_mode: bool,
) -> None:
    bundle = _bundle("standard")
    _install_runtime(monkeypatch)
    _install_common_endpoint_fakes(monkeypatch, bundle=bundle)
    raw_failure = (
        "provider request failed at https://upstream.invalid/v1/chat "
        "body=credential-runtime-secret-response"
    )

    async def failing_standard_pipeline(**_kwargs: Any) -> UnifiedSearchResult:
        raise RuntimeError(raw_failure)

    monkeypatch.setattr(rag_endpoint, "unified_rag_pipeline", failing_standard_pipeline)
    log_records: list[Any] = []
    sink_id = logger.add(log_records.append)
    try:
        with pytest.raises(HTTPException) as exc_info:
            await rag_endpoint.unified_search_endpoint(
                request_raw=_request(),
                request=rag_endpoint.UnifiedRAGRequest(
                    query="credential runtime",
                    strategy="standard",
                    debug_mode=debug_mode,
                ),
                background_tasks=BackgroundTasks(),
                current_user=_user(),
                media_db=_db("media.db"),
                chacha_db=_db("notes.db"),
                prompts_db=_db("prompts.db"),
                collections_db=SimpleNamespace(),
            )
    finally:
        logger.remove(sink_id)

    rendered = json.dumps(exc_info.value.detail) + "\n" + "\n".join(
        str(message) for message in log_records
    )
    for fragment in (
        raw_failure,
        "upstream.invalid",
        "credential-runtime-secret-response",
    ):
        assert fragment not in rendered  # nosec B101
    assert exc_info.value.status_code == 500  # nosec B101
    assert exc_info.value.detail == "Search failed due to an internal error."  # nosec B101
    assert exc_info.value.__cause__ is None  # nosec B101
    assert exc_info.value.__context__ is None  # nosec B101
    assert all(record.record["exception"] is None for record in log_records)  # nosec B101
    assert _RecordingRuntime.created[0].close_calls == 1  # nosec B101


@pytest.mark.asyncio
async def test_simple_wrapper_forwards_runtime_without_changing_legacy_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_pipeline(**kwargs: Any) -> UnifiedSearchResult:
        calls.append(kwargs)
        return UnifiedSearchResult(documents=[], query=kwargs["query"])

    monkeypatch.setattr(unified_pipeline_module, "unified_rag_pipeline", fake_pipeline)
    runtime = object()

    await unified_pipeline_module.simple_search("runtime", credential_runtime=runtime)
    await unified_pipeline_module.simple_search("legacy")

    assert calls[0]["credential_runtime"] is runtime  # nosec B101
    assert "credential_runtime" not in calls[1]  # nosec B101


def test_pipeline_entry_points_expose_optional_runtime_keyword() -> None:
    agentic_parameter = inspect.signature(production_agentic_rag_pipeline).parameters["credential_runtime"]
    rag_parameter = inspect.signature(unified_rag_pipeline).parameters["credential_runtime"]
    batch_parameter = inspect.signature(unified_batch_pipeline).parameters["credential_runtime"]

    assert agentic_parameter.default is None  # nosec B101
    assert rag_parameter.default is None  # nosec B101
    assert batch_parameter.default is None  # nosec B101

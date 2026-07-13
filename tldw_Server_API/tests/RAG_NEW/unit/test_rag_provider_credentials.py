"""Credential-runtime propagation tests for authenticated unified RAG endpoints."""

from __future__ import annotations

import asyncio
import inspect
import json
from functools import wraps
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import BackgroundTasks, HTTPException, Response
from loguru import logger
from starlette.requests import Request

import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_endpoint
import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as unified_pipeline_module
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
    assert callable(runtime.scope["fallback_resolver"])  # nosec B101


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
    assert _SENTINEL_SECRET not in "".join(logs)  # nosec B101
    assert runtime.close_calls == 1  # nosec B101


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
        principal=AuthPrincipal(kind="user", user_id=42),
        media_db=_db("media.db"),
        chacha_db=_db("notes.db"),
        prompts_db=_db("prompts.db"),
    )

    runtime = _RecordingRuntime.created[0]
    assert captured["credential_runtime"] is runtime  # nosec B101
    assert "credential_runtime" not in checkpoint.config  # nosec B101
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
            "type": "error",
            "code": "credential_store_unavailable",
            "status_code": 503,
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
    assert [event["type"] for event in events] == ["contexts", "reasoning"]  # nosec B101


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
    assert _RecordingRuntime.created[0].close_calls == 1  # nosec B101


def test_pipeline_entry_points_expose_optional_runtime_keyword() -> None:
    agentic_parameter = inspect.signature(production_agentic_rag_pipeline).parameters["credential_runtime"]
    rag_parameter = inspect.signature(unified_rag_pipeline).parameters["credential_runtime"]
    batch_parameter = inspect.signature(unified_batch_pipeline).parameters["credential_runtime"]

    assert agentic_parameter.default is None  # nosec B101
    assert rag_parameter.default is None  # nosec B101
    assert batch_parameter.default is None  # nosec B101

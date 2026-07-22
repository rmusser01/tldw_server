import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCredentialRuntime,
    reject_provider_call_credentials,
)
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import UnifiedEvaluationService


async def _wait_for_thread_event(
    event: threading.Event,
    *,
    timeout: float = 1.0,
) -> None:
    """Wait for a thread event without consuming the default executor under test."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set():
        if loop.time() >= deadline:
            raise AssertionError("Timed out waiting for thread event")
        await asyncio.sleep(0.001)


@pytest.mark.asyncio
@pytest.mark.parametrize("captured_key", ["eval-key-a", None], ids=["a-to-b", "absent-to-b"])
@pytest.mark.parametrize("evaluation_kind", ["geval", "rag", "response_quality"])
async def test_specialized_evaluation_keeps_one_snapshot_at_llm_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    evaluation_kind: str,
    captured_key: str | None,
) -> None:
    """Every specialized evaluation must preserve its captured adapter config."""
    from tldw_Server_API.app.core.Evaluations import ms_g_eval, rag_evaluator, response_quality_evaluator

    config_a = {"local_llm": {"model": "model-a", "api_ip": "http://a.invalid"}}

    async def resolver(
        normalized_provider: str,
        **_kwargs: Any,
    ) -> ResolvedByokCredentials:
        return ResolvedByokCredentials(
            provider=normalized_provider,
            api_key=captured_key,
            app_config=config_a,
            credential_fields={"base_url": "http://a.invalid"},
            source="user",
            allowlisted=True,
            status=(
                ByokResolutionStatus.RESOLVED
                if captured_key is not None
                else ByokResolutionStatus.ABSENT
            ),
            auth_source="api_key" if captured_key is not None else None,
        )

    issuer = ProviderCredentialRuntime(
        user_id=41,
        team_ids=(),
        org_ids=(),
        trusted_base_url_override=True,
        server_config_snapshot={},
        resolver=resolver,
    )
    try:
        handle = await issuer.resolve("local-llm", model="model-a")
    finally:
        await issuer.close()
    captured_requests: list[dict[str, Any]] = []
    durable_writes: list[dict[str, Any]] = []
    service = UnifiedEvaluationService(
        db_path=str(tmp_path / f"{evaluation_kind}.db"),
        enable_webhooks=False,
    )

    async def store_result(**kwargs: Any) -> str:
        reject_provider_call_credentials(kwargs)
        json.dumps(kwargs)
        durable_writes.append(kwargs)
        return "eval-1"

    monkeypatch.setattr(service, "_store_evaluation_result", store_result)

    class RecordingAdapter:
        def chat(self, request: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
            captured_requests.append({"boundary": "adapter", **request})
            return {"choices": [{"message": {"content": "4"}}]}

    monkeypatch.setattr(ms_g_eval, "get_adapter_or_raise", lambda _provider: RecordingAdapter())

    async def call_with_breaker(_provider: str, function, *args: Any, **kwargs: Any):
        result = function(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    monkeypatch.setattr(
        rag_evaluator.llm_circuit_breaker,
        "call_with_breaker",
        call_with_breaker,
    )
    monkeypatch.setattr(
        response_quality_evaluator.llm_circuit_breaker,
        "call_with_breaker",
        call_with_breaker,
    )

    def analyze_boundary(
        _api_name: str,
        _input_data: Any,
        _custom_prompt: str | None,
        api_key: str | None,
        _system_message: str | None,
        _temperature: float | None,
        **kwargs: Any,
    ) -> str:
        captured_requests.append({"boundary": "sgl", "api_key": api_key, **kwargs})
        return "4"

    monkeypatch.setattr(rag_evaluator, "analyze", analyze_boundary)
    monkeypatch.setattr(response_quality_evaluator, "analyze", analyze_boundary)

    common = {
        "api_name": "local-llm",
        "api_key": captured_key,
        "app_config": config_a,
        "credentials_resolved": True,
        "provider_credentials": handle,
        "model": "model-a",
        "user_id": "user-1",
    }
    if evaluation_kind == "geval":
        await service.evaluate_geval(
            source_text="source",
            summary="summary",
            metrics=["coherence"],
            **common,
        )
    elif evaluation_kind == "rag":
        await service.evaluate_rag(
            query="query",
            contexts=["context"],
            response="response",
            metrics=["relevance"],
            **common,
        )
    else:
        await service.evaluate_response_quality(
            prompt="prompt",
            response="response",
            **common,
        )

    assert captured_requests
    assert all(request["api_key"] == captured_key for request in captured_requests)
    assert all(request["app_config"] == config_a for request in captured_requests)
    assert all(request["credentials_resolved"] is True for request in captured_requests)
    assert all(
        (
            request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY]
            if request["boundary"] == "adapter"
            else request["provider_credentials"]
        )
        is handle
        for request in captured_requests
    )
    assert all(
        request.get("model") == "model-a"
        or request.get("model_override") == "model-a"
        for request in captured_requests
    )
    assert durable_writes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_name", "api_key", "app_config"),
    [
        (
            "anthropic",
            "anthropic-secret-must-not-cross-host",
            {"anthropic_api": {"model": "claude-test"}},
        ),
        (
            "openai",
            "openai-llm-secret-must-not-reach-embedding-host",
            {
                "openai_api": {
                    "model": "gpt-test",
                    "api_url": "https://llm-origin.invalid/v1",
                },
                "embedding_config": {
                    "provider": "openai",
                    "api_url": "https://different-embedding-origin.invalid/v1",
                },
            },
        ),
    ],
    ids=["cross-provider", "same-provider-different-origin"],
)
async def test_rag_evaluation_never_sends_resolved_llm_key_to_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    api_name: str,
    api_key: str,
    app_config: dict[str, Any],
) -> None:
    """A resolved LLM secret must stay out of independently configured embeddings."""
    from tldw_Server_API.app.core.Evaluations import rag_evaluator

    service = UnifiedEvaluationService(
        db_path=str(tmp_path / "rag-wrong-host.db"),
        enable_webhooks=False,
    )

    async def store_result(**_kwargs: Any) -> str:
        return "eval-1"

    monkeypatch.setattr(service, "_store_evaluation_result", store_result)

    embedding_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def forbidden_embedding(*args: Any, **kwargs: Any):
        embedding_calls.append((args, kwargs))
        raise AssertionError("resolved LLM key reached the embedding path")

    monkeypatch.setattr(rag_evaluator, "create_embedding", forbidden_embedding)

    llm_calls: list[dict[str, Any]] = []

    def analyze_boundary(
        api_name: str,
        _input_data: Any,
        _custom_prompt: str | None,
        api_key: str | None,
        _system_message: str | None,
        _temperature: float | None,
        **kwargs: Any,
    ) -> str:
        llm_calls.append({"api_name": api_name, "api_key": api_key, **kwargs})
        return "4"

    async def call_with_breaker(_provider: str, function, *args: Any, **kwargs: Any):
        result = function(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    monkeypatch.setattr(rag_evaluator, "analyze", analyze_boundary)
    monkeypatch.setattr(
        rag_evaluator.llm_circuit_breaker,
        "call_with_breaker",
        call_with_breaker,
    )

    await service.evaluate_rag(
        query="query",
        contexts=["context"],
        response="response",
        ground_truth="ground truth",
        metrics=["answer_similarity"],
        api_name=api_name,
        api_key=api_key,
        app_config=app_config,
        credentials_resolved=True,
        user_id="user-1",
    )

    assert embedding_calls == []
    assert llm_calls
    assert all(call["api_name"] == api_name for call in llm_calls)
    assert all(call["api_key"] == api_key for call in llm_calls)


@pytest.mark.asyncio
async def test_claim_faithfulness_binds_selected_evaluation_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ClaimsEngine globals must not replace the selected evaluation credentials."""
    from tldw_Server_API.app.core.Evaluations import rag_evaluator

    captured: list[dict[str, Any]] = []

    def analyze_boundary(
        api_name: str,
        input_data: Any,
        custom_prompt_arg: str | None,
        api_key: str | None,
        system_message: str | None,
        temp: float | None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        captured.append(
            {
                "api_name": api_name,
                "input_data": input_data,
                "custom_prompt": custom_prompt_arg,
                "api_key": api_key,
                "system_message": system_message,
                "temperature": temp,
                "args": args,
                **kwargs,
            }
        )
        return "{}"

    class HostileGlobalClaimsEngine:
        def __init__(self, analyze_fn):
            self.analyze_fn = analyze_fn

        async def run(self, **_kwargs: Any):
            self.analyze_fn(
                "openai",
                "claim",
                "judge",
                None,
                "system",
                0.2,
                model_override="global-model-must-not-win",
            )
            return {"summary": {"supported": 1, "refuted": 0, "nei": 0}}

    monkeypatch.setattr(rag_evaluator, "analyze", analyze_boundary)
    monkeypatch.setattr(rag_evaluator, "ClaimsEngine", HostileGlobalClaimsEngine)

    handle = object()
    evaluator = rag_evaluator.RAGEvaluator(
        embedding_provider=None,
        embedding_model=None,
        api_key="anthropic-eval-key",
        app_config={"anthropic_api": {"model": "claude-eval"}},
        credentials_resolved=True,
        provider_credentials=handle,
    )
    metric_name, metric = await evaluator._evaluate_claim_faithfulness(
        "answer",
        ["context"],
        "anthropic",
        "claude-eval",
    )

    assert metric_name == "claim_faithfulness"
    assert metric["score"] == 1.0
    assert captured == [
        {
            "api_name": "anthropic",
            "input_data": "claim",
            "custom_prompt": "judge",
            "api_key": "anthropic-eval-key",
            "system_message": "system",
            "temperature": 0.2,
            "args": (),
            "streaming": False,
            "recursive_summarization": False,
            "chunked_summarization": False,
            "chunk_options": None,
            "model_override": "claude-eval",
            "app_config": {"anthropic_api": {"model": "claude-eval"}},
            "credentials_resolved": True,
            "provider_credentials": handle,
        }
    ]


def test_rag_analyze_drops_unsupported_claims_response_format_at_sgl_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claims JSON hints must not cross the strict legacy SGL signature."""
    from tldw_Server_API.app.core.Evaluations import rag_evaluator

    captured: dict[str, Any] = {}

    def strict_sgl_analyze(
        api_name: str,
        input_data: Any,
        custom_prompt_arg: str | None,
        api_key: str | None = None,
        system_message: str | None = None,
        temp: float | None = None,
        streaming: bool = False,
        recursive_summarization: bool = False,
        chunked_summarization: bool = False,
        chunk_options: dict[str, Any] | None = None,
        model_override: str | None = None,
        *,
        app_config: dict[str, Any] | None = None,
        credentials_resolved: bool = False,
        raise_on_error: bool = False,
    ) -> str:
        captured.update(
            {
                "api_name": api_name,
                "input_data": input_data,
                "custom_prompt_arg": custom_prompt_arg,
                "api_key": api_key,
                "system_message": system_message,
                "temp": temp,
                "streaming": streaming,
                "recursive_summarization": recursive_summarization,
                "chunked_summarization": chunked_summarization,
                "chunk_options": chunk_options,
                "model_override": model_override,
                "app_config": app_config,
                "credentials_resolved": credentials_resolved,
                "raise_on_error": raise_on_error,
            }
        )
        return '{"claims": []}'

    monkeypatch.setattr(rag_evaluator.sgl, "analyze", strict_sgl_analyze)

    result = rag_evaluator.analyze(
        "anthropic",
        "claim source",
        "return claims JSON",
        "credential-a",
        "system",
        0.1,
        model_override="claude-eval",
        app_config={"anthropic_api": {"model": "claude-eval"}},
        credentials_resolved=True,
        response_format={"type": "json_schema", "json_schema": {"name": "claims"}},
    )

    assert result == '{"claims": []}'
    assert captured == {
        "api_name": "anthropic",
        "input_data": "claim source",
        "custom_prompt_arg": "return claims JSON",
        "api_key": "credential-a",
        "system_message": "system",
        "temp": 0.1,
        "streaming": False,
        "recursive_summarization": False,
        "chunked_summarization": False,
        "chunk_options": None,
        "model_override": "claude-eval",
        "app_config": {"anthropic_api": {"model": "claude-eval"}},
        "credentials_resolved": True,
        "raise_on_error": True,
    }


@pytest.mark.asyncio
async def test_geval_starts_direct_and_drains_cancel_before_mark_and_runtime_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """G-Eval bypasses the default executor and retains its credential runtime."""
    from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified as endpoint
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Evaluations import ms_g_eval
    from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as service_module

    lifecycle: list[str] = []

    class _ReleaseTrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            lifecycle.append("capacity-release")
            super()._release_capacity()

    class _TrackingMetrics:
        enabled = True

        @contextmanager
        def track_sli_request(self, path: str):
            assert path == "/evaluations/geval"
            lifecycle.append("metrics-enter")
            try:
                yield
            finally:
                lifecycle.append("metrics-exit")

    class _Runtime:
        async def mark_used(self, resolved_handle: object) -> None:
            assert resolved_handle is handle
            lifecycle.append("mark")

        async def close(self) -> None:
            lifecycle.append("close")

    loop = asyncio.get_running_loop()
    previous_executor = getattr(loop, "_default_executor", None)
    default_executor = ThreadPoolExecutor(max_workers=1)
    default_entered = threading.Event()
    default_release = threading.Event()
    adapter_entered = threading.Event()
    adapter_release = threading.Event()
    adapter_starts = 0
    pool = _ReleaseTrackingPool(capacity=1)
    handle = object()
    runtime = _Runtime()

    def _block_default_executor() -> None:
        default_entered.set()
        assert default_release.wait(timeout=2.0)

    def blocking_geval(**kwargs: Any) -> dict[str, Any]:
        nonlocal adapter_starts
        adapter_starts += 1
        assert kwargs["api_key"] == "geval-late-secret"
        lifecycle.append("adapter-start")
        adapter_entered.set()
        assert adapter_release.wait(timeout=2.0)
        lifecycle.append("adapter-exit")
        return {"metrics": {}, "average_score": 0.0, "assessment": "done"}

    monkeypatch.setattr(ms_g_eval, "run_geval", blocking_geval)
    monkeypatch.setattr(service_module, "advanced_metrics", _TrackingMetrics())
    monkeypatch.setattr(service_module, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    service = UnifiedEvaluationService(
        db_path=str(tmp_path / "geval-cancel.db"),
        enable_webhooks=False,
    )

    async def _store_result(**_kwargs: Any) -> str:
        lifecycle.append("semantic-store")
        return "eval-1"

    async def _endpoint_scope() -> dict[str, Any]:
        try:
            return await endpoint._await_evaluation_and_mark_used(
                service.evaluate_geval(
                    source_text="source",
                    summary="summary",
                    api_name="openai",
                    api_key="geval-late-secret",
                    model="model-a",
                    app_config={"openai_api": {"model": "model-a"}},
                    credentials_resolved=True,
                ),
                credential_runtime=runtime,  # type: ignore[arg-type]
                credential_handle=handle,  # type: ignore[arg-type]
            )
        finally:
            await runtime.close()

    monkeypatch.setattr(service, "_store_evaluation_result", _store_result)
    loop.set_default_executor(default_executor)
    default_blocker = loop.run_in_executor(None, _block_default_executor)
    task: asyncio.Task[dict[str, Any]] | None = None
    try:
        await _wait_for_thread_event(default_entered)
        task = asyncio.create_task(_endpoint_scope())
        await _wait_for_thread_event(adapter_entered)

        assert default_release.is_set() is False
        assert pool.active_count == 1
        task.cancel()
        await asyncio.sleep(0.03)
        assert task.done() is False
        assert lifecycle == ["metrics-enter", "adapter-start"]

        adapter_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

        assert lifecycle == [
            "metrics-enter",
            "adapter-start",
            "adapter-exit",
            "capacity-release",
            "metrics-exit",
            "semantic-store",
            "mark",
            "close",
        ]
        assert pool.active_count == 0
    finally:
        adapter_release.set()
        default_release.set()
        await asyncio.gather(default_blocker, return_exceptions=True)
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        loop.set_default_executor(previous_executor or ThreadPoolExecutor())
        default_executor.shutdown(wait=True, cancel_futures=True)

    await asyncio.sleep(0)
    assert adapter_starts == 1


@pytest.mark.asyncio
async def test_geval_pool_exhaustion_rejects_before_dispatch_without_mark_or_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """G-Eval capacity rejection is bounded and cannot queue credentialed work."""
    from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified as endpoint
    from tldw_Server_API.app.core.Chat.bounded_daemon import (
        BoundedDaemonPool,
        DaemonCapacityError,
    )
    from tldw_Server_API.app.core.Evaluations import ms_g_eval
    from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as service_module

    lifecycle: list[str] = []

    class _TrackingMetrics:
        enabled = True

        @contextmanager
        def track_sli_request(self, path: str):
            assert path == "/evaluations/geval"
            lifecycle.append("metrics-enter")
            try:
                yield
            finally:
                lifecycle.append("metrics-exit")

    class _Runtime:
        async def mark_used(self, _resolved_handle: object) -> None:
            lifecycle.append("mark")

        async def close(self) -> None:
            lifecycle.append("close")

    secret = "geval-rejected-secret-sentinel"
    pool = BoundedDaemonPool(capacity=1)
    holder_entered = threading.Event()
    holder_release = threading.Event()
    adapter_started = threading.Event()
    store_calls: list[str] = []
    logs: list[str] = []
    handle = object()
    runtime = _Runtime()

    def _hold_capacity() -> None:
        holder_entered.set()
        assert holder_release.wait(timeout=2.0)

    def _forbidden_geval(**_kwargs: Any) -> dict[str, Any]:
        adapter_started.set()
        return {"metrics": {}, "average_score": 0.0, "assessment": "must not run"}

    async def _store_result(**_kwargs: Any) -> str:
        store_calls.append("store")
        return "eval-1"

    service = UnifiedEvaluationService(
        db_path=str(tmp_path / "geval-capacity.db"),
        enable_webhooks=False,
    )
    monkeypatch.setattr(ms_g_eval, "run_geval", _forbidden_geval)
    monkeypatch.setattr(service_module, "advanced_metrics", _TrackingMetrics())
    monkeypatch.setattr(service_module, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    monkeypatch.setattr(service, "_store_evaluation_result", _store_result)

    async def _endpoint_scope() -> dict[str, Any]:
        try:
            return await endpoint._await_evaluation_and_mark_used(
                service.evaluate_geval(
                    source_text="source",
                    summary="summary",
                    api_name="openai",
                    api_key=secret,
                    model="model-a",
                    app_config={"openai_api": {"model": "model-a"}},
                    credentials_resolved=True,
                ),
                credential_runtime=runtime,  # type: ignore[arg-type]
                credential_handle=handle,  # type: ignore[arg-type]
            )
        finally:
            await runtime.close()

    holder = pool.start(
        _hold_capacity,
        name="geval-test-capacity-holder",
        exhaustion_message="test capacity exhausted",
    )
    sink_id = logger.add(logs.append, format="{message}")
    try:
        await _wait_for_thread_event(holder_entered)
        with pytest.raises(DaemonCapacityError) as exc_info:
            await _endpoint_scope()

        assert str(exc_info.value) == "G-Eval adapter capacity is exhausted"
        assert adapter_started.is_set() is False
        assert store_calls == []
        assert lifecycle == ["metrics-enter", "metrics-exit", "close"]
        assert pool.active_count == 1
        assert secret not in repr(exc_info.value)
        assert secret not in "".join(logs)
    finally:
        logger.remove(sink_id)
        holder_release.set()
        holder.join(timeout=1.0)

    await asyncio.sleep(0)
    assert adapter_started.is_set() is False
    assert pool.active_count == 0


def test_geval_provider_failure_never_exposes_exception_body_or_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider exception text must stay out of returned metrics and logs."""
    from tldw_Server_API.app.core.Evaluations import ms_g_eval

    secret = "sk-secret-in-body https://private-provider.invalid/v1"
    log_calls: list[tuple[Any, ...]] = []

    def fail_provider(*_args: Any, **_kwargs: Any):
        raise RuntimeError(secret)

    monkeypatch.setattr(ms_g_eval, "geval_summarization", fail_provider)
    monkeypatch.setattr(
        ms_g_eval.logger,
        "warning",
        lambda *args, **_kwargs: log_calls.append(args),
    )

    result = ms_g_eval.run_geval(
        transcript="source",
        summary="summary",
        api_key="safe-test-key",
        api_name="openai",
        model="model-a",
        app_config={"openai_api": {"model": "model-a"}},
        credentials_resolved=True,
    )

    assert result["assessment"] == "Provider evaluation failed."
    assert secret not in repr(result)
    assert secret not in repr(log_calls)


def test_schedule_service_shutdown_logs_noncritical_failure(monkeypatch):
    from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as svc_mod

    class _BrokenService:
        def shutdown(self):
            raise RuntimeError("shutdown unavailable")

    debug_calls: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        svc_mod.logger,
        "debug",
        lambda message, *args, **_kwargs: debug_calls.append((message, args)),
    )

    svc_mod._schedule_service_shutdown(_BrokenService())  # type: ignore[arg-type]

    assert debug_calls
    message, args = debug_calls[0]
    assert "shutdown scheduling skipped" in message
    assert args[0] == "_BrokenService"
    assert isinstance(args[1], RuntimeError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("eval_type", "expected_sub_type"),
    [
        ("geval", "summarization"),
        ("rag", "rag"),
        ("response_quality", "response_quality"),
    ],
)
async def test_create_evaluation_maps_to_model_graded(tmp_path, eval_type, expected_sub_type):
    svc = UnifiedEvaluationService(db_path=str(tmp_path / "evals.db"), enable_webhooks=False)

    evaluation = await svc.create_evaluation(
        name=f"test_{eval_type}",
        eval_type=eval_type,
        eval_spec={"metrics": ["relevance"]},
        created_by="tester",
    )

    assert evaluation["eval_type"] == "model_graded"
    assert evaluation["eval_spec"].get("sub_type") == expected_sub_type


def test_unified_service_uses_backend_adapter_for_postgres_webhooks(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
    from tldw_Server_API.app.core.Evaluations import db_adapter as db_adapter_mod
    from tldw_Server_API.app.core.Evaluations import eval_runner as eval_runner_mod
    from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as svc_mod
    from tldw_Server_API.app.core.Evaluations import webhook_manager as webhook_manager_mod

    class _FakeBackend:
        backend_type = BackendType.POSTGRESQL

    class _FakeDB:
        backend_type = BackendType.POSTGRESQL
        backend = _FakeBackend()

    class _DummyRunner:
        def __init__(self, _db_path):
            self.running_tasks = {}

    captured = {}
    sentinel_adapter = object()

    class _DummyWebhookManager:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(eval_runner_mod, "EvaluationRunner", _DummyRunner)
    monkeypatch.setattr(svc_mod, "_create_evals_db", lambda db_path: _FakeDB())
    monkeypatch.setattr(
        db_adapter_mod,
        "create_adapter_from_backend",
        lambda backend: sentinel_adapter,
    )
    monkeypatch.setattr(webhook_manager_mod, "WebhookManager", _DummyWebhookManager)

    UnifiedEvaluationService(db_path=str(tmp_path / "evals.db"), enable_webhooks=True)

    kwargs = captured["kwargs"]
    assert kwargs.get("adapter") is sentinel_adapter
    assert "db_path" not in kwargs


@pytest.mark.asyncio
async def test_run_evaluation_async_skips_cancelled_webhook_when_status_is_already_terminal(
    monkeypatch,
    tmp_path,
):
    service = UnifiedEvaluationService(
        db_path=str(tmp_path / "evals.db"),
        enable_webhooks=True,
    )
    service.webhook_manager = SimpleNamespace(send_webhook=AsyncMock())

    async def _cancelled_call(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(service.circuit_breaker, "call", _cancelled_call)
    monkeypatch.setattr(
        service.db,
        "get_run",
        lambda run_id, created_by=None: {"id": run_id, "status": "completed"},
    )

    status_updates: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _record_status(*args, **kwargs):
        status_updates.append((args, kwargs))

    monkeypatch.setattr(service.db, "update_run_status", _record_status)

    with pytest.raises(asyncio.CancelledError):
        await service._run_evaluation_async(
            run_id="run_terminal",
            eval_id="eval_1",
            eval_config={"webhook_url": "https://example.com/webhook"},
            created_by="tenant-user",
            webhook_user_id="user_tenant-user",
        )

    assert status_updates == []
    service.webhook_manager.send_webhook.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_run_cleans_up_terminal_task_without_rewriting_status_or_logging_cancel(
    monkeypatch,
    tmp_path,
):
    service = UnifiedEvaluationService(
        db_path=str(tmp_path / "evals.db"),
        enable_webhooks=False,
    )

    class _Task:
        def __init__(self) -> None:
            self.cancel_called = False

        def cancel(self) -> None:
            self.cancel_called = True

    task = _Task()
    service.runner.running_tasks["run_terminal"] = task

    terminal_run = {"id": "run_terminal", "status": "completed"}
    monkeypatch.setattr(service.db, "get_run", lambda run_id, created_by=None: terminal_run)
    monkeypatch.setattr(service.runner.db, "get_run", lambda run_id: terminal_run)

    status_updates: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _record_status(*args, **kwargs):
        status_updates.append((args, kwargs))
        return True

    monkeypatch.setattr(service.db, "update_run_status", _record_status)
    monkeypatch.setattr(service.runner.db, "update_run_status", _record_status)

    audit_log = AsyncMock()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.unified_evaluation_service.log_run_cancelled_async",
        audit_log,
    )

    assert await service.cancel_run(
        "run_terminal",
        cancelled_by="tester",
        created_by="tester",
    ) is True
    assert task.cancel_called is True
    assert "run_terminal" not in service.runner.running_tasks
    assert status_updates == []
    audit_log.assert_not_awaited()

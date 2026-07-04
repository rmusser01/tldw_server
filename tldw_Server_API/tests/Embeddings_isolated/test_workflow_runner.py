from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingDomainError,
    EmbeddingExecutionResult,
    EmbeddingProviderError,
    EmbeddingRequestContext,
)
from tldw_Server_API.app.core.Embeddings.workflow_runner import EmbeddingInlineWorkflowRunner
from tldw_Server_API.app.core.Embeddings.workflow_types import (
    EmbeddingInMemoryWorkflowTraceCollector,
    EmbeddingNoopWorkflowTraceCollector,
)


pytestmark = pytest.mark.unit


def _request_context() -> EmbeddingRequestContext:
    return EmbeddingRequestContext(
        user_id="user-1",
        model_field="sentence-transformers/all-MiniLM-L6-v2",
        provider_header="huggingface",
        dimensions=None,
        encoding_format="float",
        request_id="req-123",
        endpoint_path="/api/v1/embeddings",
    )


def _prepared_request() -> SimpleNamespace:
    return SimpleNamespace(
        normalized_input=SimpleNamespace(texts=["one", "two"], total_tokens=3),
        provider_intent=SimpleNamespace(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
        ),
        policy_decision=SimpleNamespace(
            fallback_allowed=True,
            fallback_chain=["huggingface"],
        ),
        execution_plan=SimpleNamespace(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=None,
            fallback_chain=["huggingface"],
            execution_path="legacy",
            cache_namespace="endpoint",
        ),
        prompt_tokens=3,
        total_tokens=3,
    )


def _execution_result() -> EmbeddingExecutionResult:
    return EmbeddingExecutionResult(
        vectors=[[0.1, 0.2], [0.3, 0.4]],
        provider="huggingface",
        model="sentence-transformers/all-MiniLM-L6-v2",
        prompt_tokens=3,
        total_tokens=3,
        cache_hits=1,
        cache_misses=1,
        fallback_from=None,
        response_headers={
            "x-request-id": "req-provider-secret",
            "authorization": "Bearer provider-secret",
        },
        embeddings_from_adapter=False,
    )


class FakeOrchestrator:
    def __init__(
        self,
        *,
        prepared: SimpleNamespace | None = None,
        result: EmbeddingExecutionResult | None = None,
        prepare_error: BaseException | None = None,
        execute_error: BaseException | None = None,
    ) -> None:
        self.prepared = prepared or _prepared_request()
        self.result = result or _execution_result()
        self.prepare_error = prepare_error
        self.execute_error = execute_error
        self.prepare_calls: list[tuple[object, EmbeddingRequestContext]] = []
        self.execute_calls: list[object] = []

    def prepare(self, raw_input: object, context: EmbeddingRequestContext) -> SimpleNamespace:
        self.prepare_calls.append((raw_input, context))
        if self.prepare_error is not None:
            raise self.prepare_error
        return self.prepared

    async def execute(self, prepared: object) -> EmbeddingExecutionResult:
        self.execute_calls.append(prepared)
        if self.execute_error is not None:
            raise self.execute_error
        return self.result


@pytest.mark.asyncio
async def test_runner_returns_orchestrator_result_and_records_safe_success_events():
    raw_input = {"input": ["one", "two"], "api_key": "sk-secret"}
    context = _request_context()
    result = _execution_result()
    orchestrator = FakeOrchestrator(result=result)
    collector = EmbeddingInMemoryWorkflowTraceCollector()
    runner = EmbeddingInlineWorkflowRunner(orchestrator, trace_collector=collector)

    observed = await runner.run(raw_input, context)

    assert observed is result
    assert orchestrator.prepare_calls == [(raw_input, context)]
    assert orchestrator.execute_calls == [orchestrator.prepared]
    assert [event.event_type for event in collector.events] == [
        "workflow_started",
        "phase_changed",
        "prepare_completed",
        "phase_changed",
        "execute_completed",
        "workflow_completed",
    ]
    assert {event.workflow_id for event in collector.events} == {"req-123"}
    assert collector.events[0].metadata == {
        "endpoint_path": "/api/v1/embeddings",
        "runner_mode": "inline",
    }
    assert collector.events[2].metadata == {
        "item_count": 2,
        "total_tokens": 3,
        "prompt_tokens": 3,
        "provider": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": None,
        "fallback_allowed": True,
        "fallback_chain_length": 1,
        "execution_path": "legacy",
        "cache_namespace": "endpoint",
    }
    assert collector.events[4].metadata == {
        "vector_count": 2,
        "cache_hits": 1,
        "cache_misses": 1,
        "provider": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "fallback_source": None,
        "adapter_used": False,
        "response_header_names": ["authorization", "x-request-id"],
    }
    assert "sk-secret" not in repr(collector.events)
    assert "provider-secret" not in repr(collector.events)
    assert "Bearer provider-secret" not in repr(collector.events)


@pytest.mark.asyncio
async def test_runner_awaits_optional_pre_execute_hook_between_prepare_and_execute():
    order: list[str] = []
    prepared = _prepared_request()
    result = _execution_result()

    class OrderedOrchestrator(FakeOrchestrator):
        def prepare(self, raw_input: object, context: EmbeddingRequestContext) -> SimpleNamespace:
            order.append("prepare")
            return super().prepare(raw_input, context)

        async def execute(self, prepared: object) -> EmbeddingExecutionResult:
            order.append("execute")
            return await super().execute(prepared)

    async def pre_execute(observed_prepared: object) -> None:
        order.append("pre_execute")
        assert observed_prepared is prepared

    orchestrator = OrderedOrchestrator(prepared=prepared, result=result)
    runner = EmbeddingInlineWorkflowRunner(
        orchestrator,
        trace_collector=EmbeddingInMemoryWorkflowTraceCollector(),
        pre_execute=pre_execute,
    )

    observed = await runner.run(["one", "two"], _request_context())

    assert observed is result
    assert order == ["prepare", "pre_execute", "execute"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        EmbeddingDomainError(
            "provider_denied",
            "raw input sk-secret must not be traced",
            retryable=False,
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
            cause_class="PolicyCause",
            details=[{"raw_body": "sk-secret", "authorization": "Bearer secret"}],
        ),
        EmbeddingProviderError(
            "provider_unavailable",
            "provider body Bearer provider-secret must not be traced",
            retryable=True,
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
            cause_class="TimeoutError",
        ),
    ],
)
async def test_domain_failures_are_traced_with_safe_metadata_and_reraised_unchanged(error):
    collector = EmbeddingInMemoryWorkflowTraceCollector()
    orchestrator = FakeOrchestrator(execute_error=error)
    runner = EmbeddingInlineWorkflowRunner(orchestrator, trace_collector=collector)

    with pytest.raises(type(error)) as exc_info:
        await runner.run(["one", "two"], _request_context())

    assert exc_info.value is error
    failed = collector.events[-1]
    assert failed.event_type == "workflow_failed"
    assert failed.phase == "executing"
    assert failed.status == "failed"
    assert failed.metadata == {
        "error_code": error.code,
        "provider": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "retryable": error.retryable,
        "cause_class": error.cause_class,
        "phase": "executing",
    }
    assert "sk-secret" not in repr(collector.events)
    assert "Bearer" not in repr(collector.events)
    assert "raw input" not in repr(collector.events)
    assert "provider body" not in repr(collector.events)


@pytest.mark.asyncio
async def test_prepare_domain_failure_is_traced_in_prepare_phase_and_reraised_unchanged():
    error = EmbeddingDomainError(
        "invalid_input_type",
        "raw input must not be traced",
        retryable=False,
        provider=None,
        model=None,
        cause_class="ValueError",
    )
    collector = EmbeddingInMemoryWorkflowTraceCollector()
    runner = EmbeddingInlineWorkflowRunner(
        FakeOrchestrator(prepare_error=error),
        trace_collector=collector,
    )

    with pytest.raises(EmbeddingDomainError) as exc_info:
        await runner.run({"input": "secret text"}, _request_context())

    assert exc_info.value is error
    assert collector.events[-1].metadata == {
        "error_code": "invalid_input_type",
        "provider": None,
        "model": None,
        "retryable": False,
        "cause_class": "ValueError",
        "phase": "normalizing",
    }


@pytest.mark.asyncio
async def test_unexpected_exceptions_trace_cause_class_and_phase_only_then_reraise():
    error = RuntimeError("raw provider body with sk-secret")
    collector = EmbeddingInMemoryWorkflowTraceCollector()
    runner = EmbeddingInlineWorkflowRunner(
        FakeOrchestrator(execute_error=error),
        trace_collector=collector,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await runner.run(["one"], _request_context())

    assert exc_info.value is error
    failed = collector.events[-1]
    assert failed.event_type == "workflow_failed"
    assert failed.phase == "executing"
    assert failed.status == "failed"
    assert failed.metadata == {
        "cause_class": "RuntimeError",
        "phase": "executing",
    }
    assert "sk-secret" not in repr(collector.events)
    assert "raw provider body" not in repr(collector.events)


@pytest.mark.asyncio
async def test_default_noop_collector_is_disabled_and_retains_no_events():
    runner = EmbeddingInlineWorkflowRunner(FakeOrchestrator())

    await runner.run(["one", "two"], _request_context())

    assert isinstance(runner.trace_collector, EmbeddingNoopWorkflowTraceCollector)
    assert runner.trace_collector.enabled is False
    assert not hasattr(runner.trace_collector, "events")

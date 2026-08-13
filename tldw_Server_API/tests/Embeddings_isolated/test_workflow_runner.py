from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingDomainError,
    EmbeddingExecutionOutcome,
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


def _execution_outcome(
    *,
    vector_count: int = 2,
    attempt_count: int = 3,
    fallback_attempt_count: int = 1,
) -> EmbeddingExecutionOutcome:
    fallback_from = "huggingface" if fallback_attempt_count else None
    return EmbeddingExecutionOutcome(
        vectors=tuple((float(index), float(index + 1)) for index in range(vector_count)),
        provider="huggingface",
        model="sentence-transformers/all-MiniLM-L6-v2",
        prompt_tokens=3,
        total_tokens=3,
        cache_hits=min(1, vector_count),
        cache_misses=max(0, vector_count - 1),
        requested_dimensions=None,
        effective_dimension_policy="strict",
        attempt_count=attempt_count,
        fallback_attempt_count=fallback_attempt_count,
        fallback_from=fallback_from,
        embeddings_from_adapter=False,
    )


class FakePreparationPipeline:
    def __init__(
        self,
        *,
        prepared: SimpleNamespace | None = None,
        prepare_error: BaseException | None = None,
        order: list[str] | None = None,
    ) -> None:
        self.prepared = prepared or _prepared_request()
        self.prepare_error = prepare_error
        self.order = order
        self.prepare_calls: list[tuple[object, EmbeddingRequestContext]] = []

    def prepare(self, raw_input, context, phase_sink=None) -> SimpleNamespace:
        self.prepare_calls.append((raw_input, context))
        if self.order is not None:
            self.order.append("prepare")
        for phase in ("resolving_intent", "normalizing", "resolving_policy", "planning"):
            if phase_sink is not None:
                phase_sink(phase)
            if phase == "normalizing" and self.prepare_error is not None:
                raise self.prepare_error
        return self.prepared


class FakeExecutionCoordinator:
    def __init__(
        self,
        *,
        outcome: EmbeddingExecutionOutcome | None = None,
        execute_error: BaseException | None = None,
        order: list[str] | None = None,
    ) -> None:
        self.outcome = outcome or _execution_outcome()
        self.execute_error = execute_error
        self.order = order
        self.execute_calls: list[object] = []

    async def execute(self, prepared: object) -> EmbeddingExecutionOutcome:
        self.execute_calls.append(prepared)
        if self.order is not None:
            self.order.append("execute")
        if self.execute_error is not None:
            raise self.execute_error
        return self.outcome


class FailingFailureCollector(EmbeddingInMemoryWorkflowTraceCollector):
    def record(self, event):
        if event.event_type == "workflow_failed":
            raise RuntimeError("collector failure")
        super().record(event)


@pytest.mark.asyncio
async def test_runner_returns_canonical_outcome_and_records_exact_safe_success_events():
    raw_input = {"input": ["one", "two"], "api_key": "sk-secret"}
    context = _request_context()
    outcome = _execution_outcome()
    preparation_pipeline = FakePreparationPipeline()
    execution_coordinator = FakeExecutionCoordinator(outcome=outcome)
    collector = EmbeddingInMemoryWorkflowTraceCollector()
    runner = EmbeddingInlineWorkflowRunner(
        preparation_pipeline,
        execution_coordinator,
        trace_collector=collector,
    )

    observed = await runner.run(raw_input, context)

    assert observed is outcome
    assert preparation_pipeline.prepare_calls == [(raw_input, context)]
    assert execution_coordinator.execute_calls == [preparation_pipeline.prepared]
    assert [event.event_type for event in collector.events] == [
        "workflow_started",
        "phase_changed",
        "phase_changed",
        "phase_changed",
        "phase_changed",
        "prepare_completed",
        "phase_changed",
        "execute_completed",
        "phase_changed",
        "workflow_completed",
    ]
    assert [event.phase for event in collector.events] == [
        "created",
        "resolving_intent",
        "normalizing",
        "resolving_policy",
        "planning",
        "planning",
        "executing",
        "executing",
        "finalizing",
        "finalizing",
    ]
    assert [event.status for event in collector.events] == [
        "running",
        "running",
        "running",
        "running",
        "running",
        "running",
        "running",
        "running",
        "running",
        "completed",
    ]
    workflow_ids = {event.workflow_id for event in collector.events}
    assert len(workflow_ids) == 1
    assert next(iter(workflow_ids)).startswith("emb-wf-")
    assert "req-123" not in repr(collector.events)
    assert collector.events[0].metadata == {
        "endpoint_path": "/api/v1/embeddings",
        "runner_mode": "inline",
    }
    assert collector.events[5].metadata == {
        "item_count": 2,
        "total_tokens": 3,
        "prompt_tokens": 3,
        "dimensions": None,
        "fallback_allowed": True,
        "fallback_chain_length": 1,
        "execution_path": "legacy",
    }
    assert collector.events[7].metadata == {
        "attempt_count": 3,
        "fallback_attempt_count": 1,
        "vector_count": 2,
        "cache_hits": 1,
        "cache_misses": 1,
        "adapter_used": False,
    }
    assert all("response_header_count" not in event.metadata for event in collector.events)
    assert "sk-secret" not in repr(collector.events)
    assert "provider-secret" not in repr(collector.events)
    assert "Bearer provider-secret" not in repr(collector.events)


@pytest.mark.asyncio
async def test_runner_awaits_optional_pre_execute_hook_between_prepare_and_execute():
    order: list[str] = []
    prepared = _prepared_request()
    outcome = _execution_outcome()

    async def pre_execute(observed_prepared: object) -> None:
        order.append("pre_execute")
        assert observed_prepared is prepared

    preparation_pipeline = FakePreparationPipeline(prepared=prepared, order=order)
    execution_coordinator = FakeExecutionCoordinator(outcome=outcome, order=order)
    runner = EmbeddingInlineWorkflowRunner(
        preparation_pipeline,
        execution_coordinator,
        trace_collector=EmbeddingInMemoryWorkflowTraceCollector(),
        pre_execute=pre_execute,
    )

    observed = await runner.run(["one", "two"], _request_context())

    assert observed is outcome
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
    runner = EmbeddingInlineWorkflowRunner(
        FakePreparationPipeline(),
        FakeExecutionCoordinator(execute_error=error),
        trace_collector=collector,
    )

    with pytest.raises(type(error)) as exc_info:
        await runner.run(["one", "two"], _request_context())

    assert exc_info.value is error
    failed = collector.events[-1]
    assert failed.event_type == "workflow_failed"
    assert failed.phase == "executing"
    assert failed.status == "failed"
    assert failed.metadata == {
        "failure_kind": "domain",
        "retryable": error.retryable,
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
        FakePreparationPipeline(prepare_error=error),
        FakeExecutionCoordinator(),
        trace_collector=collector,
    )

    with pytest.raises(EmbeddingDomainError) as exc_info:
        await runner.run({"input": "secret text"}, _request_context())

    assert exc_info.value is error
    assert collector.events[-1].metadata == {
        "failure_kind": "domain",
        "retryable": False,
        "phase": "normalizing",
    }


@pytest.mark.asyncio
async def test_unexpected_exceptions_trace_failure_kind_and_phase_only_then_reraise():
    error = RuntimeError("raw provider body with sk-secret")
    collector = EmbeddingInMemoryWorkflowTraceCollector()
    runner = EmbeddingInlineWorkflowRunner(
        FakePreparationPipeline(),
        FakeExecutionCoordinator(execute_error=error),
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
        "failure_kind": "unexpected",
        "phase": "executing",
    }
    assert "sk-secret" not in repr(collector.events)
    assert "raw provider body" not in repr(collector.events)


@pytest.mark.asyncio
async def test_execute_cancellation_propagates_without_terminal_trace_event():
    cancellation = asyncio.CancelledError("cancelled")
    preparation_pipeline = FakePreparationPipeline()
    execution_coordinator = FakeExecutionCoordinator(execute_error=cancellation)
    collector = EmbeddingInMemoryWorkflowTraceCollector()
    runner = EmbeddingInlineWorkflowRunner(
        preparation_pipeline,
        execution_coordinator,
        trace_collector=collector,
    )

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await runner.run(["one", "two"], _request_context())

    assert exc_info.value is cancellation
    assert len(preparation_pipeline.prepare_calls) == 1
    assert len(execution_coordinator.execute_calls) == 1
    assert [event.event_type for event in collector.events] == [
        "workflow_started",
        "phase_changed",
        "phase_changed",
        "phase_changed",
        "phase_changed",
        "prepare_completed",
        "phase_changed",
    ]
    assert [event.phase for event in collector.events] == [
        "created",
        "resolving_intent",
        "normalizing",
        "resolving_policy",
        "planning",
        "planning",
        "executing",
    ]
    assert collector.events[-1].phase == "executing"
    assert all(event.event_type not in {"workflow_failed", "workflow_completed"} for event in collector.events)


@pytest.mark.asyncio
async def test_failure_collector_errors_do_not_replace_original_execute_exception():
    original = RuntimeError("original")
    collector = FailingFailureCollector()
    runner = EmbeddingInlineWorkflowRunner(
        FakePreparationPipeline(),
        FakeExecutionCoordinator(execute_error=original),
        trace_collector=collector,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await runner.run(["one"], _request_context())

    assert exc_info.value is original
    assert [event.event_type for event in collector.events] == [
        "workflow_started",
        "phase_changed",
        "phase_changed",
        "phase_changed",
        "phase_changed",
        "prepare_completed",
        "phase_changed",
    ]


@pytest.mark.asyncio
async def test_pre_execute_failure_is_traced_in_planning_phase_and_reraised_unchanged():
    original = RuntimeError("reservation failed")
    collector = EmbeddingInMemoryWorkflowTraceCollector()

    async def pre_execute(_prepared):
        raise original

    runner = EmbeddingInlineWorkflowRunner(
        FakePreparationPipeline(),
        FakeExecutionCoordinator(),
        trace_collector=collector,
        pre_execute=pre_execute,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await runner.run(["one"], _request_context())

    assert exc_info.value is original
    failed = collector.events[-1]
    assert failed.event_type == "workflow_failed"
    assert failed.phase == "planning"
    assert failed.metadata == {
        "failure_kind": "unexpected",
        "phase": "planning",
    }


@pytest.mark.asyncio
async def test_runner_never_traces_caller_controlled_provider_model_or_header_names():
    prepared = _prepared_request()
    prepared.execution_plan.provider = "AKIAIOSFODNN7EXAMPLE"
    prepared.execution_plan.model = "AIzaSyA123456789012345678901234567890123"
    prepared.execution_plan.cache_namespace = "eyJhbGciOiJIUzI1NiJ9.payload.signature"
    outcome = EmbeddingExecutionOutcome(
        vectors=((0.1,),),
        provider="github_pat_0123456789abcdef",
        model="hf_0123456789abcdef",
        prompt_tokens=1,
        total_tokens=1,
        cache_hits=0,
        cache_misses=1,
        requested_dimensions=None,
        effective_dimension_policy="strict",
        attempt_count=2,
        fallback_attempt_count=1,
        fallback_from="sk-proj-0123456789abcdef",
    )
    collector = EmbeddingInMemoryWorkflowTraceCollector()
    runner = EmbeddingInlineWorkflowRunner(
        FakePreparationPipeline(prepared=prepared),
        FakeExecutionCoordinator(outcome=outcome),
        trace_collector=collector,
    )

    observed = await runner.run(["one"], _request_context())

    assert observed is outcome
    trace = repr(collector.events)
    assert "AKIA" not in trace
    assert "AIza" not in trace
    assert "github_pat" not in trace
    assert "hf_" not in trace
    assert "sk-proj" not in trace
    assert "eyJ" not in trace


@pytest.mark.asyncio
async def test_default_noop_collector_is_disabled_and_retains_no_events():
    runner = EmbeddingInlineWorkflowRunner(FakePreparationPipeline(), FakeExecutionCoordinator())

    await runner.run(["one", "two"], _request_context())

    assert isinstance(runner.trace_collector, EmbeddingNoopWorkflowTraceCollector)
    assert runner.trace_collector.enabled is False
    assert not hasattr(runner.trace_collector, "events")


@pytest.mark.asyncio
@given(
    item_count=st.integers(min_value=1, max_value=25),
    fallback_attempt_count=st.integers(min_value=0, max_value=20),
)
async def test_success_trace_cardinality_is_independent_of_input_and_fallback_size(
    item_count: int,
    fallback_attempt_count: int,
):
    prepared = _prepared_request()
    prepared.normalized_input.texts = [str(index) for index in range(item_count)]
    prepared.execution_plan.fallback_chain = [f"fallback-{index}" for index in range(fallback_attempt_count)]
    outcome = _execution_outcome(
        vector_count=item_count,
        attempt_count=fallback_attempt_count + 1,
        fallback_attempt_count=fallback_attempt_count,
    )
    collector = EmbeddingInMemoryWorkflowTraceCollector()
    runner = EmbeddingInlineWorkflowRunner(
        FakePreparationPipeline(prepared=prepared),
        FakeExecutionCoordinator(outcome=outcome),
        trace_collector=collector,
    )

    await runner.run(prepared.normalized_input.texts, _request_context())

    assert len(collector.events) == 10
    assert sum(event.event_type == "execute_completed" for event in collector.events) == 1
    assert all(event.event_type != "item_state_changed" for event in collector.events)
    assert all("response_header_count" not in event.metadata for event in collector.events)

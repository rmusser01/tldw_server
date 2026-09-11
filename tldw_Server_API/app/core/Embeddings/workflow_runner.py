"""Inline workflow runner for embeddings prepare/execute orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingDomainError,
    EmbeddingExecutionOutcome,
    EmbeddingRequestContext,
    PreparedEmbeddingRequest,
)
from tldw_Server_API.app.core.Embeddings.workflow_types import (
    EmbeddingNoopWorkflowTraceCollector,
    EmbeddingWorkflowContext,
    EmbeddingWorkflowEvent,
    EmbeddingWorkflowEventType,
    EmbeddingWorkflowPhase,
    EmbeddingWorkflowStatus,
    EmbeddingWorkflowTraceCollector,
    SafeWorkflowMetadataValue,
)

PreExecuteHook = Callable[[PreparedEmbeddingRequest], Awaitable[None]]


class PreparationPipeline(Protocol):
    """Preparation contract consumed directly by the inline workflow runner."""

    def prepare(
        self,
        raw_input: Any,
        context: EmbeddingRequestContext,
        phase_sink: Callable[[EmbeddingWorkflowPhase], None] | None = None,
    ) -> PreparedEmbeddingRequest:
        """Normalize and plan one raw embedding request."""
        raise NotImplementedError


class ExecutionCoordinator(Protocol):
    """Canonical execution contract consumed directly by the inline workflow runner."""

    async def execute(self, prepared: PreparedEmbeddingRequest) -> EmbeddingExecutionOutcome:
        """Execute one prepared embedding request."""
        raise NotImplementedError


class EmbeddingInlineWorkflowRunner:
    """Trace a synchronous prepare/async execute embeddings workflow inline."""

    def __init__(
        self,
        preparation_pipeline: PreparationPipeline,
        execution_coordinator: ExecutionCoordinator,
        *,
        trace_collector: EmbeddingWorkflowTraceCollector | None = None,
        pre_execute: PreExecuteHook | None = None,
    ) -> None:
        """Configure concrete workflow components and optional boundary hooks."""
        self._preparation_pipeline = preparation_pipeline
        self._execution_coordinator = execution_coordinator
        self.trace_collector = trace_collector or EmbeddingNoopWorkflowTraceCollector()
        self._pre_execute = pre_execute

    async def run(self, raw_input: Any, context: EmbeddingRequestContext) -> EmbeddingExecutionOutcome:
        """Run preparation and execution inline while emitting safe lifecycle events."""
        workflow_context = EmbeddingWorkflowContext.create(
            endpoint_path=context.endpoint_path,
            runner_mode="inline",
        )
        phase: EmbeddingWorkflowPhase = "created"
        self._record(
            workflow_context,
            "workflow_started",
            phase=phase,
            status="running",
            metadata={
                "endpoint_path": workflow_context.endpoint_path,
                "runner_mode": workflow_context.runner_mode,
            },
        )

        def enter_phase(next_phase: EmbeddingWorkflowPhase) -> None:
            nonlocal phase
            phase = next_phase
            self._record(workflow_context, "phase_changed", phase=phase, status="running")

        try:
            prepared = self._preparation_pipeline.prepare(
                raw_input,
                context,
                phase_sink=enter_phase,
            )
            self._record(
                workflow_context,
                "prepare_completed",
                phase=phase,
                status="running",
                metadata=_prepare_metadata(prepared),
            )

            if self._pre_execute is not None:
                await self._pre_execute(prepared)

            enter_phase("executing")
            outcome = await self._execution_coordinator.execute(prepared)
            self._record(
                workflow_context,
                "execute_completed",
                phase=phase,
                status="running",
                metadata=_execute_metadata(outcome),
            )
            enter_phase("finalizing")
            self._record(workflow_context, "workflow_completed", phase=phase, status="completed")
            return outcome
        except EmbeddingDomainError as exc:
            self._record_failure(
                workflow_context,
                "workflow_failed",
                phase=phase,
                status="failed",
                metadata=_domain_error_metadata(exc, phase),
            )
            raise
        except Exception:
            self._record_failure(
                workflow_context,
                "workflow_failed",
                phase=phase,
                status="failed",
                metadata={
                    "failure_kind": "unexpected",
                    "phase": phase,
                },
            )
            raise

    def _record_failure(
        self,
        workflow_context: EmbeddingWorkflowContext,
        event_type: EmbeddingWorkflowEventType,
        *,
        phase: EmbeddingWorkflowPhase | None,
        status: EmbeddingWorkflowStatus,
        metadata: Mapping[str, SafeWorkflowMetadataValue],
    ) -> None:
        """Record failure metadata without replacing the original exception."""
        try:
            self._record(
                workflow_context,
                event_type,
                phase=phase,
                status=status,
                metadata=metadata,
            )
        except Exception as trace_error:  # noqa: BLE001 - tracing must not mask the request failure.
            del trace_error

    def _record(
        self,
        workflow_context: EmbeddingWorkflowContext,
        event_type: EmbeddingWorkflowEventType,
        *,
        phase: EmbeddingWorkflowPhase | None = None,
        status: EmbeddingWorkflowStatus | None = None,
        metadata: Mapping[str, SafeWorkflowMetadataValue] | None = None,
    ) -> None:
        """Record one event when trace collection is enabled."""
        if not self.trace_collector.enabled:
            return
        self.trace_collector.record(
            EmbeddingWorkflowEvent(
                event_type=event_type,
                workflow_id=workflow_context.workflow_id,
                phase=phase,
                status=status,
                metadata=metadata or {},
            )
        )


def _prepare_metadata(
    prepared: PreparedEmbeddingRequest,
) -> dict[str, SafeWorkflowMetadataValue]:
    """Derive aggregate allowlisted metadata from a prepared request."""
    normalized = prepared.normalized_input
    policy = prepared.policy_decision
    plan = prepared.execution_plan
    return {
        "item_count": len(normalized.texts),
        "total_tokens": prepared.total_tokens,
        "prompt_tokens": prepared.prompt_tokens,
        "dimensions": plan.dimensions,
        "fallback_allowed": policy.fallback_allowed,
        "fallback_chain_length": len(plan.fallback_chain),
        "execution_path": plan.execution_path,
    }


def _execute_metadata(outcome: EmbeddingExecutionOutcome) -> dict[str, SafeWorkflowMetadataValue]:
    """Derive aggregate allowlisted metadata from a canonical execution outcome."""
    return {
        "attempt_count": outcome.attempt_count,
        "fallback_attempt_count": outcome.fallback_attempt_count,
        "vector_count": len(outcome.vectors),
        "cache_hits": outcome.cache_hits,
        "cache_misses": outcome.cache_misses,
        "adapter_used": outcome.embeddings_from_adapter,
    }


def _domain_error_metadata(
    error: EmbeddingDomainError,
    phase: EmbeddingWorkflowPhase,
) -> dict[str, SafeWorkflowMetadataValue]:
    """Reduce a domain failure to fixed, non-sensitive trace metadata."""
    return {
        "failure_kind": "domain",
        "retryable": error.retryable,
        "phase": phase,
    }


__all__ = [
    "EmbeddingInlineWorkflowRunner",
    "ExecutionCoordinator",
    "PreExecuteHook",
    "PreparationPipeline",
]

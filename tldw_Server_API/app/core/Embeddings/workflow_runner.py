"""Inline workflow runner for embeddings prepare/execute orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingDomainError,
    EmbeddingExecutionResult,
    EmbeddingRequestContext,
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

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Embeddings.orchestrator import PreparedEmbeddingRequest


PreExecuteHook = Callable[["PreparedEmbeddingRequest"], Awaitable[None]]


class PrepareExecuteOrchestrator(Protocol):
    """Minimal orchestrator contract needed by the inline workflow runner."""

    def prepare(self, raw_input: Any, context: EmbeddingRequestContext) -> PreparedEmbeddingRequest:
        """Normalize and plan one raw embedding request."""
        raise NotImplementedError

    async def execute(self, prepared: PreparedEmbeddingRequest) -> EmbeddingExecutionResult:
        """Execute one prepared embedding request."""
        raise NotImplementedError


class EmbeddingInlineWorkflowRunner:
    """Trace a synchronous prepare/async execute embeddings workflow inline."""

    def __init__(
        self,
        orchestrator: PrepareExecuteOrchestrator,
        *,
        trace_collector: EmbeddingWorkflowTraceCollector | None = None,
        pre_execute: PreExecuteHook | None = None,
    ) -> None:
        """Configure the orchestrator, optional collector, and boundary hook."""
        self._orchestrator = orchestrator
        self.trace_collector = trace_collector or EmbeddingNoopWorkflowTraceCollector()
        self._pre_execute = pre_execute

    async def run(self, raw_input: Any, context: EmbeddingRequestContext) -> EmbeddingExecutionResult:
        """Run prepare and execute inline while emitting safe lifecycle events."""
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

        try:
            phase = "normalizing"
            self._record(workflow_context, "phase_changed", phase=phase, status="running")
            prepared = self._orchestrator.prepare(raw_input, context)
            phase = "planning"
            self._record(workflow_context, "phase_changed", phase=phase, status="running")
            self._record(
                workflow_context,
                "prepare_completed",
                phase=phase,
                status="running",
                metadata=_prepare_metadata(prepared),
            )

            if self._pre_execute is not None:
                await self._pre_execute(prepared)

            phase = "executing"
            self._record(workflow_context, "phase_changed", phase=phase, status="running")
            result = await self._orchestrator.execute(prepared)
            self._record(
                workflow_context,
                "execute_completed",
                phase=phase,
                status="running",
                metadata=_execute_metadata(result),
            )
            self._record(workflow_context, "workflow_completed", phase="finalizing", status="completed")
            return result
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


def _execute_metadata(result: EmbeddingExecutionResult) -> dict[str, SafeWorkflowMetadataValue]:
    """Derive aggregate allowlisted metadata from an execution result."""
    return {
        "vector_count": len(result.vectors),
        "cache_hits": result.cache_hits,
        "cache_misses": result.cache_misses,
        "adapter_used": result.embeddings_from_adapter,
        "response_header_count": len(result.response_headers),
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
    "PreExecuteHook",
    "PrepareExecuteOrchestrator",
]

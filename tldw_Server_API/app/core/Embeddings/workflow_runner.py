"""Inline workflow runner for embeddings prepare/execute orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
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
    EmbeddingWorkflowPhase,
    EmbeddingWorkflowTraceCollector,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Embeddings.orchestrator import PreparedEmbeddingRequest


PreExecuteHook = Callable[["PreparedEmbeddingRequest"], Awaitable[None]]


class PrepareExecuteOrchestrator(Protocol):
    """Minimal orchestrator contract needed by the inline workflow runner."""

    def prepare(self, raw_input: Any, context: EmbeddingRequestContext) -> "PreparedEmbeddingRequest":
        raise NotImplementedError

    async def execute(self, prepared: "PreparedEmbeddingRequest") -> EmbeddingExecutionResult:
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
        self._orchestrator = orchestrator
        self.trace_collector = trace_collector or EmbeddingNoopWorkflowTraceCollector()
        self._pre_execute = pre_execute

    async def run(self, raw_input: Any, context: EmbeddingRequestContext) -> EmbeddingExecutionResult:
        workflow_context = EmbeddingWorkflowContext.from_request(
            request_id=context.request_id,
            user_id=context.user_id,
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
            self._record(
                workflow_context,
                "prepare_completed",
                phase=phase,
                status="running",
                metadata=_prepare_metadata(prepared),
            )

            if self._pre_execute is not None:
                phase = "planning"
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
        except Exception as exc:
            self._record_failure(
                workflow_context,
                "workflow_failed",
                phase=phase,
                status="failed",
                metadata={
                    "cause_class": exc.__class__.__name__,
                    "phase": phase,
                },
            )
            raise

    def _record_failure(
        self,
        workflow_context: EmbeddingWorkflowContext,
        event_type: str,
        *,
        phase: EmbeddingWorkflowPhase | None,
        status: str,
        metadata: dict[str, object],
    ) -> None:
        try:
            self._record(
                workflow_context,
                event_type,
                phase=phase,
                status=status,
                metadata=metadata,
            )
        except Exception as trace_error:
            del trace_error

    def _record(
        self,
        workflow_context: EmbeddingWorkflowContext,
        event_type: str,
        *,
        phase: EmbeddingWorkflowPhase | None = None,
        status: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
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


def _prepare_metadata(prepared: "PreparedEmbeddingRequest") -> dict[str, object]:
    normalized = prepared.normalized_input
    policy = prepared.policy_decision
    plan = prepared.execution_plan
    return {
        "item_count": len(normalized.texts),
        "total_tokens": prepared.total_tokens,
        "prompt_tokens": prepared.prompt_tokens,
        "provider": plan.provider,
        "model": plan.model,
        "dimensions": plan.dimensions,
        "fallback_allowed": policy.fallback_allowed,
        "fallback_chain_length": len(plan.fallback_chain),
        "execution_path": plan.execution_path,
        "cache_namespace": plan.cache_namespace,
    }


def _execute_metadata(result: EmbeddingExecutionResult) -> dict[str, object]:
    return {
        "vector_count": len(result.vectors),
        "cache_hits": result.cache_hits,
        "cache_misses": result.cache_misses,
        "provider": result.provider,
        "model": result.model,
        "fallback_source": result.fallback_from,
        "adapter_used": result.embeddings_from_adapter,
        "response_header_names": sorted(result.response_headers),
    }


def _domain_error_metadata(error: EmbeddingDomainError, phase: EmbeddingWorkflowPhase) -> dict[str, object]:
    return {
        "error_code": error.code,
        "provider": error.provider,
        "model": error.model,
        "retryable": error.retryable,
        "cause_class": error.cause_class or _cause_class(error),
        "phase": phase,
    }


def _cause_class(error: BaseException) -> str | None:
    if error.__cause__ is None:
        return None
    return error.__cause__.__class__.__name__


__all__ = [
    "EmbeddingInlineWorkflowRunner",
    "PreExecuteHook",
    "PrepareExecuteOrchestrator",
]

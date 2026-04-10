"""Shared request bundle builder for transport-facing RAG endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from tldw_Server_API.app.core.RAG.rag_service.request_resolution import (
    ResolvedRAGRequest,
    resolve_rag_request,
)
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import (
    RetrievalPlan,
    build_retrieval_plan,
)

PipelineKwargsBuilder = Callable[..., dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ResolvedRequestBundle:
    """Canonical request context used by endpoint transport adapters."""

    resolved_request: ResolvedRAGRequest
    retrieval_plan: RetrievalPlan
    pipeline_kwargs: dict[str, Any]


def build_request_bundle(
    *,
    request: Any,
    current_user: Optional[Any],
    pipeline_kwargs_builder: PipelineKwargsBuilder,
    resolve_request_kwargs: Optional[dict[str, Any]] = None,
    resolve_request_fn: Callable[..., ResolvedRAGRequest] = resolve_rag_request,
    build_retrieval_plan_fn: Callable[[ResolvedRAGRequest], RetrievalPlan] = build_retrieval_plan,
) -> ResolvedRequestBundle:
    """Resolve a request once and expose a stable bundle for transport handlers."""

    request_kwargs = dict(resolve_request_kwargs or {})
    resolved_request = resolve_request_fn(
        request,
        current_user=current_user,
        **request_kwargs,
    )
    retrieval_plan = build_retrieval_plan_fn(resolved_request)
    pipeline_kwargs = dict(
        pipeline_kwargs_builder(
            resolved_request=resolved_request,
            retrieval_plan=retrieval_plan,
        )
    )
    pipeline_kwargs["resolved_request"] = resolved_request
    pipeline_kwargs["retrieval_plan"] = retrieval_plan
    return ResolvedRequestBundle(
        resolved_request=resolved_request,
        retrieval_plan=retrieval_plan,
        pipeline_kwargs=pipeline_kwargs,
    )

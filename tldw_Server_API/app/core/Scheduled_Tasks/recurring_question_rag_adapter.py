"""RAG adapter for Recurring Question scheduled tasks."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGRequest, UnifiedRAGResponse

_PRIVATE_KEY_PARTS = (
    "api_key",
    "apikey",
    "access_token",
    "token",
    "secret",
    "password",
    "raw_text",
    "rawtext",
    "full_text",
    "document_text",
    "content",
)


class RecurringQuestionRAGError(Exception):
    """Expected Recurring Question RAG execution failure."""

    def __init__(self, code: str, *, retryable: bool = False, details: dict[str, Any] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.details = details or {}


@dataclass(frozen=True)
class RecurringQuestionRAGResult:
    """Classified result from one Recurring Question RAG execution."""

    outcome: str
    title: str
    summary: str
    answer: Any | None = None
    answer_mode: str = "none"
    confidence: dict[str, Any] = field(default_factory=dict)
    source_refs: list[dict[str, Any]] = field(default_factory=list)
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    failure_reason: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = field(default_factory=dict)


def build_rag_request_from_definition(
    definition: Any,
    *,
    scope_snapshot: dict[str, Any],
    finding_policy: dict[str, Any],
    generation_mode: str = "optional",
) -> UnifiedRAGRequest:
    """Map a Recurring Question definition snapshot to a safe Unified RAG request."""
    sources = _sources_from_scope(scope_snapshot)
    if not sources:
        raise RecurringQuestionRAGError("scope_empty", retryable=False)
    question = str(getattr(definition, "input", {}).get("question") or "").strip()
    if not question:
        raise RecurringQuestionRAGError("question_empty", retryable=False)
    top_k = _coerce_int(finding_policy.get("top_k"), default=10, minimum=1, maximum=100)
    min_score = _coerce_float(finding_policy.get("min_score"), default=0.0, minimum=0.0, maximum=1.0)
    profile = _profile_for_policy(finding_policy)
    enable_generation = generation_mode != "disabled"
    request_payload: dict[str, Any] = {
        "query": question,
        "sources": sources,
        "rag_profile": profile,
        "top_k": top_k,
        "min_score": min_score,
        "enable_generation": enable_generation,
        "enable_chunk_citations": True,
    }
    if scope_snapshot.get("workspace_id"):
        request_payload["workspace_id"] = str(scope_snapshot["workspace_id"])
    collection_ids = scope_snapshot.get("collection_ids")
    if isinstance(collection_ids, list) and collection_ids:
        with_context = collection_ids[0]
        try:
            request_payload["collection_id"] = int(with_context)
        except (TypeError, ValueError):
            pass
    return UnifiedRAGRequest(**request_payload)


async def execute_recurring_question_rag(
    request: UnifiedRAGRequest,
    *,
    rag_executor: Callable[[UnifiedRAGRequest], Any] | None = None,
    generation_mode: str = "optional",
    finding_policy: dict[str, Any] | None = None,
) -> RecurringQuestionRAGResult:
    """Execute RAG with an injectable executor and classify the response."""
    executor = rag_executor or _default_rag_executor
    try:
        response = executor(request)
        if inspect.isawaitable(response):
            response = await response
    except RecurringQuestionRAGError:
        raise
    except Exception as exc:
        raise RecurringQuestionRAGError(
            "rag_unavailable",
            retryable=True,
            details={"error_type": type(exc).__name__},
        ) from exc
    if not isinstance(response, UnifiedRAGResponse):
        response = UnifiedRAGResponse.model_validate(response)
    return classify_rag_response(
        response,
        generation_mode=generation_mode,
        finding_policy=finding_policy or {},
    )


def classify_rag_response(
    response: UnifiedRAGResponse,
    *,
    generation_mode: str,
    finding_policy: dict[str, Any],
) -> RecurringQuestionRAGResult:
    """Classify a Unified RAG response into run/result semantics."""
    evidence_summary = summarize_rag_response(response)
    source_refs = _source_refs(response.documents)
    if response.errors:
        return RecurringQuestionRAGResult(
            outcome="degraded",
            title="Recurring Question RAG failed",
            summary="RAG returned errors while processing the question.",
            evidence_summary=evidence_summary,
            failure_reason={"code": "rag_response_error", "messages": list(response.errors)},
            response_metadata=response.metadata,
        )
    if not response.documents:
        return RecurringQuestionRAGResult(
            outcome="no_match",
            title="No matching sources found",
            summary="No searchable source matched this recurring question.",
            confidence={"label": "none"},
            evidence_summary=evidence_summary,
            response_metadata=response.metadata,
        )
    answer = response.generated_answer
    if isinstance(answer, dict):
        answer = answer.get("answer") or answer.get("text") or answer
    if generation_mode == "required" and not answer:
        return RecurringQuestionRAGResult(
            outcome="degraded",
            title="Answer generation unavailable",
            summary="Matching evidence was found, but required answer generation did not produce an answer.",
            confidence={"label": "low"},
            source_refs=source_refs,
            evidence_summary=evidence_summary,
            failure_reason={"code": "generation_required_unavailable"},
            response_metadata=response.metadata,
        )
    confidence = _confidence_from_documents(response.documents, finding_policy)
    if answer:
        return RecurringQuestionRAGResult(
            outcome="finding",
            title="Possible answer found",
            summary=_short_text(str(answer), limit=240),
            answer=answer,
            answer_mode="synthesized",
            confidence=confidence,
            source_refs=source_refs,
            evidence_summary=evidence_summary,
            response_metadata=response.metadata,
        )
    return RecurringQuestionRAGResult(
        outcome="finding",
        title="Relevant evidence found",
        summary="Matching evidence was found, but no generated answer was produced.",
        answer=None,
        answer_mode="evidence_only",
        confidence=confidence,
        source_refs=source_refs,
        evidence_summary=evidence_summary,
        response_metadata=response.metadata,
    )


def summarize_rag_response(response: UnifiedRAGResponse) -> dict[str, Any]:
    """Return a compact, storage-safe response summary."""
    return {
        "document_count": len(response.documents),
        "citation_count": len(response.citations or []) + len(response.chunk_citations or []),
        "answer_present": bool(response.generated_answer),
        "total_time": response.total_time,
        "cache_hit": response.cache_hit,
    }


def safe_rag_request_snapshot(
    request: UnifiedRAGRequest,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a storage-safe RAG request snapshot."""
    snapshot = {
        "query": request.query,
        "sources": list(request.sources or []),
        "rag_profile": request.rag_profile,
        "search_mode": request.search_mode,
        "top_k": request.top_k,
        "min_score": request.min_score,
        "enable_generation": request.enable_generation,
        "workspace_id": request.workspace_id,
        "collection_id": request.collection_id,
    }
    if extra:
        snapshot["extra"] = extra
    return _sanitize_snapshot(snapshot)


async def _default_rag_executor(request: UnifiedRAGRequest) -> UnifiedRAGResponse:
    from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline

    return await unified_rag_pipeline(**request.model_dump())


def _sources_from_scope(scope: dict[str, Any]) -> list[str]:
    if scope.get("mode") == "all_searchable_library":
        raw_sources = scope.get("resolved_sources")
    else:
        raw_sources = scope.get("sources")
    if not isinstance(raw_sources, list):
        return []
    return list(dict.fromkeys(source.strip() for source in raw_sources if isinstance(source, str) and source.strip()))


def _profile_for_policy(policy: dict[str, Any]) -> str:
    preset = str(policy.get("preset") or "balanced_findings")
    if preset == "high_confidence_only":
        return "accuracy"
    return "balanced"


def _confidence_from_documents(documents: list[dict[str, Any]], finding_policy: dict[str, Any]) -> dict[str, Any]:
    scores = [_coerce_float(doc.get("score"), default=0.0, minimum=0.0, maximum=1.0) for doc in documents]
    best = max(scores or [0.0])
    min_score = _coerce_float(finding_policy.get("min_score"), default=0.0, minimum=0.0, maximum=1.0)
    if best >= max(0.8, min_score):
        label = "high"
    elif best >= max(0.5, min_score):
        label = "medium"
    else:
        label = "low"
    return {"label": label, "best_score": best}


def _source_refs(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for index, document in enumerate(documents[:5]):
        refs.append(
            {
                "source_id": str(
                    document.get("source_id")
                    or document.get("media_id")
                    or document.get("id")
                    or f"source-{index + 1}"
                ),
                "title": _short_text(str(document.get("title") or document.get("source") or "Untitled"), limit=160),
                "score": document.get("score"),
                "snippet": _short_text(
                    str(document.get("snippet") or document.get("excerpt") or document.get("summary") or ""),
                    limit=300,
                ),
            }
        )
    return refs


def _sanitize_snapshot(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if _is_private_key(str(key)):
                continue
            sanitized[key] = _sanitize_snapshot(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_snapshot(item) for item in value]
    if isinstance(value, str):
        return _short_text(value, limit=500)
    return value


def _is_private_key(key: str) -> bool:
    normalized = "".join(ch for ch in key.lower() if ch.isalnum() or ch == "_")
    compact = normalized.replace("_", "")
    return any(part in normalized or part.replace("_", "") in compact for part in _PRIVATE_KEY_PARTS)


def _short_text(value: str, *, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1].rstrip()}..."


def _coerce_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _coerce_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))

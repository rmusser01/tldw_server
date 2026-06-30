"""Core mapping helpers from internal RAG results to API responses."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse

from .result_model import RAGResult
from .trust_contracts import EVIDENCE_TEXT_KEYS, classify_knowledge_answer_trust

EVIDENCE_METADATA_KEYS = (
    "source_id",
    "source_type",
    "chunk_id",
    "evidence_origin",
    "source_status",
    "unavailable_reason",
)

def _result_field(result: Any, key: str, default: Any = None) -> Any:
    """Read a top-level field from attr-based or dict-shaped result objects."""
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _extract_field(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from direct attrs, wrapped `.document`, or dict shapes."""
    if hasattr(obj, key):
        try:
            return getattr(obj, key)
        except Exception:  # noqa: BLE001 - preserve endpoint-wrapper behavior
            obj = obj

    if hasattr(obj, "document"):
        doc_obj = obj.document
        if isinstance(doc_obj, dict):
            if key in doc_obj:
                return doc_obj.get(key, default)
        elif hasattr(doc_obj, key):
            try:
                return getattr(doc_obj, key)
            except Exception:  # noqa: BLE001 - preserve endpoint-wrapper behavior
                doc_obj = doc_obj

    if isinstance(obj, dict):
        if key in obj:
            return obj.get(key, default)
        doc_dict = obj.get("document") if isinstance(obj.get("document"), dict) else None
        if doc_dict and key in doc_dict:
            return doc_dict.get(key, default)

    return default


def _normalize_documents(documents: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for doc in documents or []:
        doc_id = _extract_field(doc, "id")
        content = next(
            (
                value
                for value in (_extract_field(doc, key) for key in EVIDENCE_TEXT_KEYS)
                if isinstance(value, str) and value.strip()
            ),
            _extract_field(doc, "content"),
        )
        metadata = _extract_field(doc, "metadata", {}) or {}
        score = _extract_field(doc, "score", 0.0)

        if not isinstance(metadata, dict):
            try:
                metadata = dict(metadata)
            except (TypeError, ValueError):
                metadata = {"value": str(metadata)}
        else:
            metadata = dict(metadata)

        for key in EVIDENCE_METADATA_KEYS:
            value = _extract_field(doc, key)
            if value is not None and metadata.get(key) is None:
                metadata[key] = value

        normalized.append(
            {
                "id": doc_id if doc_id is not None else str(_extract_field(doc, "chunk_id", "unknown")),
                "content": content if isinstance(content, str) else (str(content) if content is not None else ""),
                "metadata": metadata,
                "score": float(score) if isinstance(score, (int, float)) else 0.0,
            }
        )
    return normalized


def _web_fallback_used(metadata: dict[str, Any], documents: list[dict[str, Any]]) -> bool:
    """Return whether metadata or normalized evidence indicates web fallback use."""
    web_fallback_metadata = metadata.get("web_fallback")
    if isinstance(web_fallback_metadata, dict) and any(
        bool(web_fallback_metadata.get(key)) for key in ("triggered", "used")
    ):
        return True
    if bool(metadata.get("web_fallback_used")):
        return True
    return any(
        document.get("metadata", {}).get("evidence_origin") == "web_fallback"
        or document.get("evidence_origin") == "web_fallback"
        for document in documents
    )


def rag_result_from_unified_search_result(result: Any) -> RAGResult:
    """Adapt unified pipeline results into the core result contract."""
    metadata = _result_field(result, "metadata", None) or {}

    verification_report = _result_field(result, "verification_report", None)
    if verification_report is None and isinstance(metadata, dict):
        verification_report = metadata.get("verification_report")

    chunk_citations = _result_field(result, "chunk_citations", None)
    if chunk_citations is None and isinstance(metadata, dict):
        chunk_citations = metadata.get("chunk_citations")

    generated_answer = _result_field(result, "generated_answer", None)
    if generated_answer is None:
        answer_payload = _result_field(result, "answer", None)
        if answer_payload is not None:
            generated_answer = answer_payload

    return RAGResult(
        documents=list(_result_field(result, "documents", None) or []),
        query=str(_result_field(result, "query", "")),
        expanded_queries=list(_result_field(result, "expanded_queries", None) or []),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
        timings=dict(_result_field(result, "timings", None) or {}),
        citations=list(_result_field(result, "citations", None) or []),
        academic_citations=list(metadata.get("academic_citations", []) or []),
        chunk_citations=list(chunk_citations or []),
        feedback_id=_result_field(result, "feedback_id", None),
        generated_answer=generated_answer,
        cache_hit=bool(_result_field(result, "cache_hit", False)),
        errors=list(_result_field(result, "errors", None) or []),
        security_report=_result_field(result, "security_report", None),
        total_time=float(_result_field(result, "total_time", 0.0) or 0.0),
        claims=_result_field(result, "claims", None),
        factuality=_result_field(result, "factuality", None),
        verification_report=verification_report,
        retrieval_metrics=metadata.get("retrieval_metrics"),
        faithfulness=metadata.get("faithfulness"),
        query_classification=metadata.get("query_classification"),
        reformulated_query=metadata.get("reformulated_query"),
        research_summary=metadata.get("research"),
        suggestions=metadata.get("suggestions"),
        images=metadata.get("images"),
        videos=metadata.get("videos"),
    )


def rag_result_to_response(result: RAGResult) -> UnifiedRAGResponse:
    """Convert the core result contract into the declared API response."""
    metadata = dict(result.metadata or {})
    documents = _normalize_documents(result.documents)
    trust = classify_knowledge_answer_trust(
        answer=result.generated_answer,
        documents=documents,
        citations=result.citations or result.chunk_citations,
        web_fallback_used=_web_fallback_used(metadata, documents),
    )
    metadata["knowledge_trust"] = {
        "state": trust["state"],
        "reason_codes": trust["reason_codes"],
        "evidence_origin": trust["evidence_origin"],
    }
    return UnifiedRAGResponse(
        documents=documents,
        query=result.query,
        expanded_queries=result.expanded_queries,
        metadata=metadata,
        timings=result.timings,
        citations=result.citations,
        academic_citations=result.academic_citations,
        chunk_citations=result.chunk_citations,
        feedback_id=result.feedback_id,
        generated_answer=result.generated_answer,
        cache_hit=result.cache_hit,
        errors=result.errors,
        security_report=result.security_report,
        total_time=result.total_time,
        claims=result.claims,
        factuality=result.factuality,
        verification_report=result.verification_report,
        retrieval_metrics=result.retrieval_metrics,
        faithfulness=result.faithfulness,
        query_classification=result.query_classification,
        reformulated_query=result.reformulated_query,
        research_summary=result.research_summary,
        suggestions=result.suggestions,
        images=result.images,
        videos=result.videos,
    )

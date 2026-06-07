"""Deterministic Knowledge QA trust classification helpers."""

from __future__ import annotations

import re
from typing import Any

KNOWLEDGE_TRUST_STATES = {
    "cited_answer",
    "uncited_degraded_answer",
    "no_answer_insufficient_evidence",
    "no_results",
    "failed_search",
    "unsynced_local_result",
    "unknown_trust",
}

EVIDENCE_ORIGINS = {"local_library", "web_fallback", "mixed", "unknown_origin"}
EVIDENCE_TEXT_KEYS = ("content", "excerpt", "text", "chunk")
UNINSPECTABLE_SOURCE_STATUSES = {
    "unavailable",
    "error",
    "deleted",
    "missing",
}


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _metadata(document: Any) -> dict[str, Any]:
    raw = _field(document, "metadata", {}) or {}
    if isinstance(raw, dict):
        return raw
    try:
        return dict(raw)
    except (TypeError, ValueError):
        return {}


def _identifier_values(document: Any) -> set[str]:
    metadata = _metadata(document)
    values = {
        _field(document, "id"),
        _field(document, "source_id"),
        _field(document, "chunk_id"),
        metadata.get("source_id"),
        metadata.get("chunk_id"),
        metadata.get("media_id"),
        metadata.get("id"),
    }
    return {str(value) for value in values if value not in (None, "")}


def _citation_document_id(citation: Any) -> str | None:
    for key in ("document_id", "documentId", "doc_id", "source_id", "sourceId", "id"):
        value = _field(citation, key)
        if value not in (None, ""):
            return str(value)
    return None


def _citation_index(citation: Any) -> int | None:
    value = _field(citation, "index")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _inline_citations_from_answer(answer: str | None) -> list[dict[str, int]]:
    if not isinstance(answer, str):
        return []
    seen: set[int] = set()
    citations: list[dict[str, int]] = []
    for match in re.finditer(r"\[(\d+)\]", answer):
        index = int(match.group(1))
        if index in seen:
            continue
        seen.add(index)
        citations.append({"index": index})
    return citations


def _document_for_citation(
    citation: Any, documents: list[Any], document_ids: list[set[str]]
) -> Any | None:
    citation_document_id = _citation_document_id(citation)
    if citation_document_id is not None:
        for index, identifiers in enumerate(document_ids):
            if citation_document_id in identifiers:
                return documents[index]
        return None

    citation_index = _citation_index(citation)
    if citation_index is not None and 1 <= citation_index <= len(documents):
        return documents[citation_index - 1]
    return None


def _has_inspectable_text(document: Any) -> bool:
    for key in EVIDENCE_TEXT_KEYS:
        value = _field(document, key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _is_inspectable(document: Any) -> bool:
    metadata = _metadata(document)
    status = str(
        _field(document, "source_status", metadata.get("source_status", "searched"))
    ).lower()
    unavailable_reason = _field(
        document, "unavailable_reason", metadata.get("unavailable_reason")
    )
    if status in UNINSPECTABLE_SOURCE_STATUSES:
        return False
    if unavailable_reason not in (None, ""):
        return False
    return _has_inspectable_text(document)


def _has_low_relevance_signal(documents: list[Any]) -> bool:
    for document in documents:
        metadata = _metadata(document)
        if bool(_field(document, "low_relevance", metadata.get("low_relevance"))):
            return True
        relevance_status = _field(
            document, "relevance_status", metadata.get("relevance_status")
        )
        if relevance_status == "low_relevance":
            return True
    return False


def _normalize_origin(raw_origin: Any) -> str | None:
    if not isinstance(raw_origin, str):
        return None
    origin = raw_origin.strip()
    return origin if origin in EVIDENCE_ORIGINS else None


def _evidence_origin(documents: list[Any], web_fallback_used: bool) -> str:
    origins: set[str] = set()
    for document in documents:
        metadata = _metadata(document)
        origin = _normalize_origin(
            _field(document, "evidence_origin", metadata.get("evidence_origin"))
        )
        if origin == "mixed":
            return "mixed"
        if origin:
            origins.add(origin)

    if "local_library" in origins and "web_fallback" in origins:
        return "mixed"
    if "web_fallback" in origins:
        return "web_fallback"
    if "local_library" in origins:
        return "local_library"
    if web_fallback_used:
        return "web_fallback"
    return "local_library"


def _trust(
    state: str, reason_codes: list[str], evidence_origin: str
) -> dict[str, Any]:
    return {
        "state": state if state in KNOWLEDGE_TRUST_STATES else "unknown_trust",
        "reason_codes": reason_codes,
        "evidence_origin": evidence_origin
        if evidence_origin in EVIDENCE_ORIGINS
        else "unknown_origin",
    }


def classify_knowledge_answer_trust(
    *,
    answer: str | None,
    documents: list[Any],
    citations: list[Any],
    web_fallback_used: bool,
) -> dict[str, Any]:
    """Classify Knowledge QA answer trust from structural evidence contracts.

    This classifier checks whether citations resolve to returned sources with
    inspectable evidence. It does not make semantic claim-support judgments.
    """
    normalized_documents = list(documents or [])
    normalized_citations = list(citations or []) or _inline_citations_from_answer(answer)
    evidence_origin = _evidence_origin(normalized_documents, web_fallback_used)
    has_answer = isinstance(answer, str) and bool(answer.strip())

    if not normalized_documents:
        return _trust("no_results", ["no_evidence"], evidence_origin)

    if not any(_is_inspectable(document) for document in normalized_documents):
        return _trust(
            "no_answer_insufficient_evidence",
            ["missing_inspectable_evidence"],
            evidence_origin,
        )

    if _has_low_relevance_signal(normalized_documents):
        return _trust(
            "no_answer_insufficient_evidence",
            ["low_relevance"],
            evidence_origin,
        )

    if has_answer and not normalized_citations:
        return _trust(
            "uncited_degraded_answer",
            ["missing_citations"],
            evidence_origin,
        )

    if has_answer and normalized_citations:
        document_ids = [_identifier_values(document) for document in normalized_documents]
        cited_documents = [
            _document_for_citation(citation, normalized_documents, document_ids)
            for citation in normalized_citations
        ]
        if any(document is None for document in cited_documents):
            return _trust(
                "uncited_degraded_answer",
                ["citation_source_not_returned"],
                evidence_origin,
            )
        if not all(_is_inspectable(document) for document in cited_documents):
            return _trust(
                "no_answer_insufficient_evidence",
                ["missing_inspectable_evidence"],
                evidence_origin,
            )
        return _trust("cited_answer", [], evidence_origin)

    return _trust("unknown_trust", ["unclassified"], evidence_origin)

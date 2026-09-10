from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
from tldw_Server_API.app.core.Scheduled_Tasks.recurring_question_rag_adapter import (
    RecurringQuestionRAGError,
    build_rag_request_from_definition,
    execute_recurring_question_rag,
    safe_rag_request_snapshot,
)


def _definition(question: str = "What changed?") -> SimpleNamespace:
    return SimpleNamespace(
        id="definition-1",
        input={"question": question},
        finding_policy={"preset": "balanced_findings", "min_score": 0.25, "top_k": 8},
    )


def test_build_rag_request_maps_scope_policy_and_generation_disabled():
    request = build_rag_request_from_definition(
        _definition(),
        scope_snapshot={"mode": "sources", "sources": ["media_db", "notes"]},
        finding_policy={"preset": "balanced_findings", "min_score": 0.25, "top_k": 8},
        generation_mode="disabled",
    )

    assert request.query == "What changed?"  # nosec B101
    assert request.sources == ["media_db", "notes"]  # nosec B101
    assert request.rag_profile == "balanced"  # nosec B101
    assert request.top_k == 8  # nosec B101
    assert request.min_score == 0.25  # nosec B101
    assert request.enable_generation is False  # nosec B101


def test_build_rag_request_rejects_empty_scope():
    with pytest.raises(RecurringQuestionRAGError, match="scope_empty"):
        build_rag_request_from_definition(
            _definition(),
            scope_snapshot={"mode": "sources", "sources": []},
            finding_policy={"preset": "balanced_findings"},
        )


def test_safe_rag_request_snapshot_strips_private_and_raw_fields():
    request = build_rag_request_from_definition(
        _definition(),
        scope_snapshot={"mode": "sources", "sources": ["media_db"]},
        finding_policy={
            "preset": "balanced_findings",
            "api_key": "secret",
            "raw_text": "RAW FULL SOURCE TEXT",
        },
    )

    snapshot = safe_rag_request_snapshot(request, extra={"access_token": "secret", "rawText": "RAW FULL SOURCE TEXT"})

    assert snapshot["query"] == "What changed?"  # nosec B101
    assert "api_key" not in str(snapshot)  # nosec B101
    assert "access_token" not in str(snapshot)  # nosec B101
    assert "RAW FULL SOURCE TEXT" not in str(snapshot)  # nosec B101


def test_safe_rag_request_snapshot_preserves_nonsecret_content_metadata():
    request = build_rag_request_from_definition(
        _definition(),
        scope_snapshot={"mode": "sources", "sources": ["media_db"]},
        finding_policy={"preset": "balanced_findings"},
    )

    snapshot = safe_rag_request_snapshot(
        request,
        extra={
            "content_type": "text/html",
            "content_length": 4096,
            "content": "RAW DOCUMENT BODY",
        },
    )

    assert snapshot["extra"]["content_type"] == "text/html"  # nosec B101
    assert snapshot["extra"]["content_length"] == 4096  # nosec B101
    assert "RAW DOCUMENT BODY" not in str(snapshot)  # nosec B101


@pytest.mark.asyncio
async def test_execute_recurring_question_rag_classifies_synthesized_and_evidence_only_findings():
    request = build_rag_request_from_definition(
        _definition(),
        scope_snapshot={"mode": "sources", "sources": ["media_db"]},
        finding_policy={"preset": "balanced_findings"},
        generation_mode="optional",
    )

    async def _synthesized(_request):
        return UnifiedRAGResponse(
            query=_request.query,
            documents=[{"id": "doc-1", "title": "Doc", "score": 0.91, "snippet": "short"}],
            generated_answer="A useful answer.",
        )

    synthesized = await execute_recurring_question_rag(request, rag_executor=_synthesized)

    assert synthesized.outcome == "finding"  # nosec B101
    assert synthesized.answer_mode == "synthesized"  # nosec B101
    assert synthesized.answer == "A useful answer."  # nosec B101

    async def _evidence_only(_request):
        return UnifiedRAGResponse(
            query=_request.query,
            documents=[{"id": "doc-1", "title": "Doc", "score": 0.91, "snippet": "short"}],
            generated_answer=None,
        )

    evidence_only = await execute_recurring_question_rag(request, rag_executor=_evidence_only)

    assert evidence_only.outcome == "finding"  # nosec B101
    assert evidence_only.answer_mode == "evidence_only"  # nosec B101
    assert evidence_only.answer is None  # nosec B101

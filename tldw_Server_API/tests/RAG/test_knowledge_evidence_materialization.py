from dataclasses import dataclass
from typing import Any

import pytest

from tldw_Server_API.app.core.RAG.rag_service.response_mapping import (
    _normalize_documents,
)


pytestmark = pytest.mark.unit


@dataclass
class WrappedDocument:
    document: Any
    score: float
    source_id: str
    chunk_id: str
    evidence_origin: str
    source_status: str


def test_normalize_documents_preserves_knowledge_evidence_fields() -> None:
    doc = {
        "id": "media:42:chunk:7",
        "content": "Visible matched excerpt.",
        "metadata": {
            "title": "Grounded QA checklist",
            "source_type": "media_db",
            "source_id": "42",
            "chunk_id": "7",
            "evidence_origin": "local_library",
            "source_status": "searched",
            "unavailable_reason": None,
        },
        "score": 0.91,
    }

    [normalized] = _normalize_documents([doc])

    assert normalized["id"] == "media:42:chunk:7"
    assert normalized["content"] == "Visible matched excerpt."
    assert normalized["score"] == pytest.approx(0.91)
    assert normalized["metadata"]["source_id"] == "42"
    assert normalized["metadata"]["source_type"] == "media_db"
    assert normalized["metadata"]["chunk_id"] == "7"
    assert normalized["metadata"]["evidence_origin"] == "local_library"
    assert normalized["metadata"]["source_status"] == "searched"
    assert normalized["metadata"]["unavailable_reason"] is None


def test_normalize_documents_uses_excerpt_and_merges_wrapped_identifiers() -> None:
    doc = WrappedDocument(
        document={
            "id": "note:abc:chunk:3",
            "excerpt": "Excerpt-only evidence should still be visible.",
            "metadata": {
                "title": "Meeting note",
                "source_type": "notes",
            },
        },
        score=0.72,
        source_id="abc",
        chunk_id="3",
        evidence_origin="local_library",
        source_status="searched",
    )

    [normalized] = _normalize_documents([doc])

    assert normalized["id"] == "note:abc:chunk:3"
    assert normalized["content"] == "Excerpt-only evidence should still be visible."
    assert normalized["score"] == pytest.approx(0.72)
    assert normalized["metadata"]["source_id"] == "abc"
    assert normalized["metadata"]["chunk_id"] == "3"
    assert normalized["metadata"]["evidence_origin"] == "local_library"
    assert normalized["metadata"]["source_status"] == "searched"


def test_normalize_documents_preserves_unavailable_reason_without_content() -> None:
    doc = {
        "id": "media:99:chunk:2",
        "content": "",
        "metadata": {
            "title": "Deleted source",
            "source_type": "media_db",
            "source_id": "99",
            "chunk_id": "2",
        },
        "source_status": "unavailable",
        "unavailable_reason": "deleted_or_unavailable",
        "evidence_origin": "local_library",
        "score": 0.0,
    }

    [normalized] = _normalize_documents([doc])

    assert normalized["id"] == "media:99:chunk:2"
    assert normalized["content"] == ""
    assert normalized["metadata"]["source_id"] == "99"
    assert normalized["metadata"]["chunk_id"] == "2"
    assert normalized["metadata"]["source_status"] == "unavailable"
    assert normalized["metadata"]["unavailable_reason"] == "deleted_or_unavailable"
    assert normalized["metadata"]["evidence_origin"] == "local_library"

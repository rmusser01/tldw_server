from __future__ import annotations

from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    _serialize_result_document,
)


def test_serialize_result_document_normalizes_dict_backed_documents() -> None:
    serialized = _serialize_result_document(
        {
            "id": "doc-1",
            "text": "Paris is the capital of France.",
            "score": 0.95,
            "source": "media_db",
            "media_id": "10",
            "metadata": {
                "note_id": "note-7",
            },
        }
    )

    assert serialized["id"] == "doc-1"
    assert serialized["content"] == "Paris is the capital of France."
    assert serialized["score"] == 0.95
    assert serialized["source"] == "media_db"
    assert serialized["metadata"]["source"] == "media_db"
    assert serialized["metadata"]["media_id"] == "10"
    assert serialized["metadata"]["note_id"] == "note-7"


def test_serialize_result_document_top_level_source_overrides_metadata_marker() -> None:
    serialized = _serialize_result_document(
        {
            "id": "doc-1",
            "content": "Authoritative media content",
            "source": DataSource.MEDIA_DB,
            "metadata": {"source": "notes", "media_id": 10},
        }
    )

    assert serialized["source"] == "media_db"
    assert serialized["metadata"]["source"] == "media_db"


def test_serialize_result_document_preserves_dict_compatibility_without_inventing_source() -> None:
    serialized = _serialize_result_document(
        {
            "id": "doc-1",
            "content": "Legacy dict content",
            "metadata": {"source": "media_db", "media_id": 10},
        }
    )

    assert "source" not in serialized
    assert serialized["metadata"]["source"] == "media_db"


def test_serialize_result_document_uses_document_source_and_authoritative_locators() -> None:
    serialized = _serialize_result_document(
        Document(
            id="note-chunk-1",
            content="A note must remain a note.",
            source=DataSource.NOTES,
            score=0.9,
            metadata={
                "source": "media_db",
                "media_id": 10,
                "chunk_index": 999,
                "start_char": 999,
                "end_char": 999,
            },
            chunk_index=2,
            start_char=10,
            end_char=35,
        )
    )

    assert serialized["source"] == "notes"
    assert serialized["metadata"]["source"] == "notes"
    assert serialized["metadata"]["chunk_id"] == "note-chunk-1"
    assert serialized["metadata"]["chunk_index"] == 2
    assert serialized["metadata"]["start_char"] == 10
    assert serialized["metadata"]["end_char"] == 35

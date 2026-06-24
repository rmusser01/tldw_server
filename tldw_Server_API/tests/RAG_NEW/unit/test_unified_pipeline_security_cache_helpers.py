"""Focused coverage for unified pipeline security and cache helpers."""

from typing import Any

import numpy as np
import pytest

from tldw_Server_API.app.core.RAG.rag_service.types import Document
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    _clone_cached_documents,
    _resolve_security_user_id,
)


pytestmark = pytest.mark.unit


def test_resolve_security_user_id_prefers_request_user_then_feedback_user() -> None:
    """Verify security user resolution prefers request, feedback, then anonymous."""
    scenarios = [
        ("request-user", "feedback-user", "request-user"),
        (None, "feedback-user", "feedback-user"),
        ("", "feedback-user", "feedback-user"),
        (None, None, "anonymous"),
        ("   ", "", "anonymous"),
    ]

    assert [
        _resolve_security_user_id(request_user, feedback_user)
        for request_user, feedback_user, _expected in scenarios
    ] == [expected for _request_user, _feedback_user, expected in scenarios]


def test_clone_cached_documents_returns_independent_document_instances() -> None:
    """Verify cached document clones protect mutable document fields."""
    original = Document(
        id="doc-1",
        content="original content",
        metadata={"nested": {"count": 1}},
        score=0.7,
    )

    first_clone = _clone_cached_documents([original])
    assert first_clone[0] is not original

    first_clone[0].content = "redacted content"
    first_clone[0].metadata["nested"]["count"] = 2

    assert original.content == "original content"
    assert original.metadata["nested"]["count"] == 1

    second_clone = _clone_cached_documents(first_clone)
    assert second_clone[0] is not first_clone[0]

    second_clone[0].metadata["nested"]["count"] = 3
    assert first_clone[0].metadata["nested"]["count"] == 2


def test_clone_cached_documents_preserves_embedding_reference_without_deepcopy() -> None:
    """Verify cache cloning avoids duplicating large embedding arrays."""
    embedding = np.array([0.1, 0.2, 0.3])
    original = Document(
        id="doc-embedding",
        content="content",
        metadata={"nested": {"count": 1}},
        score=0.5,
        embedding=embedding,
    )

    cloned = _clone_cached_documents([original])[0]

    assert isinstance(cloned, Document)
    assert cloned is not original
    assert cloned.embedding is embedding
    cloned.metadata["nested"]["count"] = 2
    assert original.metadata["nested"]["count"] == 1


def test_clone_cached_documents_copies_dict_metadata_without_embedding_copy() -> None:
    """Verify dict document clones isolate metadata but share embeddings."""
    embedding = np.array([0.1, 0.2, 0.3])
    original: dict[str, Any] = {
        "id": "dict-doc",
        "content": "content",
        "metadata": {"nested": {"count": 1}},
        "embedding": embedding,
    }

    cloned = _clone_cached_documents([original])[0]

    assert cloned is not original
    assert cloned["embedding"] is embedding
    cloned["metadata"]["nested"]["count"] = 2
    assert original["metadata"]["nested"]["count"] == 1

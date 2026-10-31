"""Focused coverage for unified pipeline security and cache helpers."""

from typing import Any

import numpy as np
import pytest

from tldw_Server_API.app.core.DB_Management.scope_context import (
    ScopeContext,
    content_authorization_cache_scope,
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    _clone_cached_documents,
    _resolve_security_user_id,
)

pytestmark = pytest.mark.unit


def test_content_authorization_cache_scope_is_sorted_and_copy_safe() -> None:
    """Authorization cache identity snapshots mutable membership collections."""
    scope = ScopeContext(
        user_id=7,
        org_ids=[30, 10, 30],
        team_ids=[40, 20, 40],
        active_org_id=10,
        active_team_id=20,
        is_admin=True,
        session_role="content_reader",
    )

    identity = content_authorization_cache_scope(scope)
    scope.org_ids.append(99)
    scope.team_ids.append(88)

    assert identity == {
        "user_id": 7,
        "org_ids": (10, 30),
        "team_ids": (20, 40),
        "active_org_id": 10,
        "active_team_id": 20,
        "is_admin": True,
        "session_role": "content_reader",
    }
    assert content_authorization_cache_scope(scope) is not identity


def test_content_authorization_cache_scope_hashes_full_oversized_role() -> None:
    """Oversized roles remain bounded without prefix-truncation collisions."""
    shared_prefix = "r" * 512
    first_scope = ScopeContext(
        user_id=7,
        org_ids=[],
        team_ids=[],
        active_org_id=None,
        active_team_id=None,
        session_role=f"{shared_prefix}-first",
    )
    second_scope = ScopeContext(
        user_id=7,
        org_ids=[],
        team_ids=[],
        active_org_id=None,
        active_team_id=None,
        session_role=f"{shared_prefix}-second",
    )

    first = content_authorization_cache_scope(first_scope)["session_role"]
    second = content_authorization_cache_scope(second_scope)["session_role"]

    assert isinstance(first, str)
    assert isinstance(second, str)
    assert first.startswith("sha256:")
    assert second.startswith("sha256:")
    assert len(first) <= 128
    assert first != second
    assert first == content_authorization_cache_scope(first_scope)["session_role"]


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

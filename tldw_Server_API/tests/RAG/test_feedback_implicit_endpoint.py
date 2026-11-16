from __future__ import annotations

import pytest
from fastapi import status

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit
from tldw_Server_API.app.api.v1.endpoints import rag_unified
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


@pytest.fixture()
def implicit_feedback_client(client_user_only):
    client_user_only.app.dependency_overrides[check_rate_limit] = lambda: None
    try:
        yield client_user_only
    finally:
        client_user_only.app.dependency_overrides.pop(check_rate_limit, None)


@pytest.mark.integration
def test_implicit_feedback_records_citation_used(implicit_feedback_client, monkeypatch):
    class _Collector:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def record_implicit_interaction(self, **kwargs) -> None:
            self.calls.append(kwargs)

    collector = _Collector()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.implicit_feedback_enabled",
        lambda: True,
    )
    monkeypatch.setattr("tldw_Server_API.app.core.RAG.rag_service.analytics_system.UnifiedFeedbackSystem", lambda: collector)

    resp = implicit_feedback_client.post(
        "/api/v1/rag/feedback/implicit",
        json={
            "event_type": "citation_used",
            "query": "reset auth",
            "doc_id": "doc-1",
            "chunk_ids": ["chunk-1"],
            "rank": 2,
            "impression_list": ["doc-1", "doc-2"],
            "session_id": "sess-1",
            "conversation_id": "C_1",
            "message_id": "M_1",
        },
    )

    assert resp.status_code == status.HTTP_200_OK
    assert collector.calls
    call = collector.calls[0]
    assert call["event_type"] == "citation_used"
    assert call["doc_id"] == "doc-1"
    assert call["chunk_ids"] == ["chunk-1"]
    assert call["rank"] == 2
    assert call["user_id"] == str(DatabasePaths.get_single_user_id())


@pytest.mark.integration
def test_implicit_feedback_dwell_requires_dwell_ms(implicit_feedback_client):
    resp = implicit_feedback_client.post(
        "/api/v1/rag/feedback/implicit",
        json={"event_type": "dwell_time"},
    )
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.integration
def test_implicit_feedback_records_dwell_time(implicit_feedback_client, monkeypatch):
    class _Collector:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def record_implicit_interaction(self, **kwargs) -> None:
            self.calls.append(kwargs)

    collector = _Collector()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.implicit_feedback_enabled",
        lambda: True,
    )
    monkeypatch.setattr("tldw_Server_API.app.core.RAG.rag_service.analytics_system.UnifiedFeedbackSystem", lambda: collector)

    resp = implicit_feedback_client.post(
        "/api/v1/rag/feedback/implicit",
        json={
            "event_type": "dwell_time",
            "dwell_ms": 3200,
            "message_id": "M_1",
            "conversation_id": "C_1",
            "session_id": "sess-1",
        },
    )
    assert resp.status_code == status.HTTP_200_OK
    assert collector.calls
    assert collector.calls[0]["event_type"] == "dwell_time"
    assert collector.calls[0]["dwell_ms"] == 3200
    assert collector.calls[0]["user_id"] == str(DatabasePaths.get_single_user_id())


@pytest.mark.integration
def test_implicit_feedback_normalizes_legacy_single_user_id(implicit_feedback_client, monkeypatch):
    class _Collector:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def record_implicit_interaction(self, **kwargs) -> None:
            self.calls.append(kwargs)

    collector = _Collector()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.implicit_feedback_enabled",
        lambda: True,
    )
    monkeypatch.setattr("tldw_Server_API.app.core.RAG.rag_service.analytics_system.UnifiedFeedbackSystem", lambda: collector)

    resp = implicit_feedback_client.post(
        "/api/v1/rag/feedback/implicit",
        json={
            "event_type": "click",
            "doc_id": "doc-legacy",
            "user_id": "single_user",
        },
    )

    assert resp.status_code == status.HTTP_200_OK
    assert collector.calls
    assert collector.calls[0]["user_id"] == str(DatabasePaths.get_single_user_id())

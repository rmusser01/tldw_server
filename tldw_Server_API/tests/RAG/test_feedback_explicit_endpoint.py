from __future__ import annotations

import json

import pytest
from fastapi import status

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit
from tldw_Server_API.app.api.v1.endpoints import feedback as feedback_endpoint
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.app.core.RAG.rag_service.analytics_system import UnifiedFeedbackSystem


def _row_to_dict(row, cursor):

    if isinstance(row, dict):
        return row
    try:
        return dict(row)
    except (TypeError, ValueError):
        columns = [col[0] for col in cursor.description] if cursor.description else []
        return {columns[idx]: row[idx] for idx in range(len(columns))}


@pytest.fixture()
def feedback_setup(tmp_path, client_user_only):
    db = CharactersRAGDB(
        db_path=str(tmp_path / "feedback_chacha.db"),
        client_id="1",
    )
    character_id = db.add_character_card(
        {
            "name": "Feedback Assistant",
            "description": "Test assistant",
            "personality": "Helpful",
            "system_prompt": "Be helpful.",
            "client_id": "1",
        }
    )
    conversation_id = db.add_conversation(
        {
            "character_id": character_id,
            "title": "Feedback Conversation",
            "client_id": "1",
        }
    )
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "Derived from message content",
            "client_id": "1",
        }
    )

    client_user_only.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    client_user_only.app.dependency_overrides[check_rate_limit] = lambda: None

    try:
        yield client_user_only, db, conversation_id, message_id
    finally:
        client_user_only.app.dependency_overrides.pop(get_chacha_db_for_user, None)
        client_user_only.app.dependency_overrides.pop(check_rate_limit, None)


@pytest.mark.integration
def test_explicit_feedback_derives_query_from_message(feedback_setup):
    client, db, conversation_id, message_id = feedback_setup

    payload = {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "feedback_type": "helpful",
        "helpful": True,
    }

    resp = client.post("/api/v1/feedback/explicit", json=payload)
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["ok"] is True
    assert data["feedback_id"]

    cursor = db.execute_query(
        "SELECT query FROM conversation_feedback WHERE id = ?",
        (data["feedback_id"],),
    )
    row = cursor.fetchone()
    assert row is not None
    record = _row_to_dict(row, cursor)
    assert record["query"] == "Derived from message content"


@pytest.mark.integration
def test_explicit_feedback_idempotent_merge_updates_issues_and_notes(feedback_setup):
    client, db, conversation_id, message_id = feedback_setup

    payload = {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "feedback_type": "helpful",
        "helpful": False,
        "issues": ["missing_details"],
        "user_notes": "Initial notes",
    }

    resp = client.post("/api/v1/feedback/explicit", json=payload)
    assert resp.status_code == status.HTTP_200_OK
    feedback_id = resp.json()["feedback_id"]

    payload_update = {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "feedback_type": "helpful",
        "helpful": False,
        "issues": ["incorrect_information"],
        "user_notes": "Updated notes",
    }

    resp_update = client.post("/api/v1/feedback/explicit", json=payload_update)
    assert resp_update.status_code == status.HTTP_200_OK
    assert resp_update.json()["feedback_id"] == feedback_id

    cursor = db.execute_query(
        "SELECT issues, user_notes FROM conversation_feedback WHERE id = ?",
        (feedback_id,),
    )
    row = cursor.fetchone()
    assert row is not None
    record = _row_to_dict(row, cursor)
    issues = json.loads(record["issues"]) if record.get("issues") else []
    assert set(issues) == {"missing_details", "incorrect_information"}
    assert record["user_notes"] == "Updated notes"

    cursor = db.execute_query(
        "SELECT COUNT(*) AS count FROM conversation_feedback WHERE conversation_id = ?",
        (conversation_id,),
    )
    count_row = cursor.fetchone()
    assert count_row is not None
    count_record = _row_to_dict(count_row, cursor)
    assert count_record["count"] == 1


@pytest.mark.integration
def test_explicit_feedback_persists_idempotency_state_in_db(feedback_setup):
    client, db, conversation_id, message_id = feedback_setup

    payload = {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "feedback_type": "helpful",
        "helpful": False,
        "issues": ["missing_details"],
        "user_notes": "Initial notes",
        "idempotency_key": "feedback-idem-key-1",
    }
    first = client.post("/api/v1/feedback/explicit", json=payload)
    assert first.status_code == status.HTTP_200_OK
    feedback_id = first.json()["feedback_id"]
    assert feedback_id

    payload_update = dict(payload)
    payload_update["issues"] = ["incorrect_information"]
    payload_update["user_notes"] = "Merged notes"

    second = client.post("/api/v1/feedback/explicit", json=payload_update)
    assert second.status_code == status.HTTP_200_OK
    assert second.json()["feedback_id"] == feedback_id

    cursor = db.execute_query(
        "SELECT feedback_id, pending, issues, user_notes FROM feedback_idempotency WHERE dedupe_key = ?",
        ("idem:1:feedback-idem-key-1",),
    )
    row = cursor.fetchone()
    assert row is not None
    record = _row_to_dict(row, cursor)
    issues = json.loads(record["issues"]) if record.get("issues") else []
    assert record["feedback_id"] == feedback_id
    assert set(issues) == {"missing_details", "incorrect_information"}
    assert record["user_notes"] == "Merged notes"
    assert int(record["pending"]) == 0


@pytest.mark.integration
def test_explicit_feedback_rag_only_accepts_query(feedback_setup):
    client, _db, _conversation_id, _message_id = feedback_setup

    payload = {
        "feedback_type": "relevance",
        "relevance_score": 4,
        "query": "How do I reset auth?",
    }

    resp = client.post("/api/v1/feedback/explicit", json=payload)
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["ok"] is True
    assert data["feedback_id"] is None


@pytest.mark.integration
def test_explicit_feedback_rejects_empty_query(feedback_setup):
    client, _db, _conversation_id, _message_id = feedback_setup

    payload = {
        "feedback_type": "helpful",
        "helpful": True,
        "query": "   ",
    }

    resp = client.post("/api/v1/feedback/explicit", json=payload)
    assert resp.status_code in (status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY)
    detail = resp.json().get("detail")
    expected = "query is required when message_id is not provided"
    if isinstance(detail, list):
        messages = " ".join(item.get("msg", "") for item in detail if isinstance(item, dict))
        assert expected in messages
    else:
        assert detail == expected


# ---------------------------------------------------------------------------
# GET  /api/v1/feedback  – list feedback
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_list_feedback_returns_entries(feedback_setup):
    client, db, conversation_id, message_id = feedback_setup

    # Submit feedback first
    payload = {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "feedback_type": "helpful",
        "helpful": True,
        "issues": ["not_relevant"],
        "user_notes": "Test note",
    }
    resp = client.post("/api/v1/feedback/explicit", json=payload)
    assert resp.status_code == status.HTTP_200_OK

    # List feedback
    resp = client.get("/api/v1/feedback", params={"conversation_id": conversation_id})
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["ok"] is True
    assert len(data["feedback"]) >= 1
    entry = data["feedback"][0]
    assert entry["conversation_id"] == conversation_id
    assert entry["user_notes"] == "Test note"


@pytest.mark.integration
def test_get_feedback_returns_single_entry(feedback_setup):
    client, _db, conversation_id, message_id = feedback_setup

    payload = {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "feedback_type": "helpful",
        "helpful": True,
        "issues": ["not_relevant"],
        "user_notes": "Detail note",
    }
    create_resp = client.post("/api/v1/feedback/explicit", json=payload)
    assert create_resp.status_code == status.HTTP_200_OK
    feedback_id = create_resp.json()["feedback_id"]

    resp = client.get(f"/api/v1/feedback/{feedback_id}")
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["id"] == feedback_id
    assert data["conversation_id"] == conversation_id
    assert data["message_id"] == message_id
    assert data["user_notes"] == "Detail note"


@pytest.mark.integration
def test_get_feedback_fails_closed_when_conversation_is_missing(feedback_setup, monkeypatch):
    client, _db, _conversation_id, _message_id = feedback_setup

    class _OrphanedFeedbackStore:
        async def get_feedback_by_id(self, _feedback_id: str):
            return {
                "id": "fb_orphaned",
                "conversation_id": "missing-conversation",
                "helpful": True,
            }

    class _OrphanedFeedbackSystem:
        def __init__(self, chacha_db):
            self.user_feedback = _OrphanedFeedbackStore()

    monkeypatch.setattr(feedback_endpoint, "UnifiedFeedbackSystem", _OrphanedFeedbackSystem)

    resp = client.get("/api/v1/feedback/fb_orphaned")

    assert resp.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.integration
def test_list_feedback_not_found_for_missing_conversation(feedback_setup):
    client, _db, _conversation_id, _message_id = feedback_setup

    resp = client.get("/api/v1/feedback", params={"conversation_id": "C_nonexistent"})
    assert resp.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.integration
def test_list_feedback_returns_500_for_db_error(feedback_setup, monkeypatch):
    client, _db, conversation_id, _message_id = feedback_setup

    class _FailingFeedbackStore:
        async def get_conversation_feedback(self, _conversation_id: str):
            raise CharactersRAGDBError("feedback query failed")

    class _FailingFeedbackSystem:
        def __init__(self, chacha_db):
            self.user_feedback = _FailingFeedbackStore()

    monkeypatch.setattr(feedback_endpoint, "UnifiedFeedbackSystem", _FailingFeedbackSystem)

    resp = client.get("/api/v1/feedback", params={"conversation_id": conversation_id})

    assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert resp.json()["detail"] == "Failed to list feedback"


# ---------------------------------------------------------------------------
# DELETE  /api/v1/feedback/{feedback_id}
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_delete_feedback_removes_record(feedback_setup):
    client, db, conversation_id, message_id = feedback_setup

    # Submit
    payload = {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "feedback_type": "helpful",
        "helpful": False,
    }
    resp = client.post("/api/v1/feedback/explicit", json=payload)
    assert resp.status_code == status.HTTP_200_OK
    feedback_id = resp.json()["feedback_id"]
    assert feedback_id

    # Delete
    resp = client.delete(f"/api/v1/feedback/{feedback_id}")
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["ok"] is True
    assert data["deleted"] is True

    # Verify gone
    cursor = db.execute_query(
        "SELECT COUNT(*) AS count FROM conversation_feedback WHERE id = ?",
        (feedback_id,),
    )
    row = cursor.fetchone()
    record = _row_to_dict(row, cursor)
    assert record["count"] == 0


@pytest.mark.integration
def test_delete_feedback_not_found(feedback_setup):
    client, _db, _conversation_id, _message_id = feedback_setup

    resp = client.delete("/api/v1/feedback/fb_nonexistent")
    assert resp.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.integration
def test_delete_feedback_returns_500_for_db_error(feedback_setup, monkeypatch):
    client, _db, conversation_id, _message_id = feedback_setup

    class _FailingFeedbackStore:
        async def get_feedback_by_id(self, _feedback_id: str):
            return {"conversation_id": conversation_id}

        async def delete_feedback(self, _feedback_id: str):
            raise CharactersRAGDBError("delete feedback failed")

    class _FailingFeedbackSystem:
        def __init__(self, chacha_db):
            self.user_feedback = _FailingFeedbackStore()

    monkeypatch.setattr(feedback_endpoint, "UnifiedFeedbackSystem", _FailingFeedbackSystem)

    resp = client.delete("/api/v1/feedback/fb_problem")

    assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert resp.json()["detail"] == "Failed to delete feedback"


# ---------------------------------------------------------------------------
# PATCH  /api/v1/feedback/{feedback_id}
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_patch_feedback_updates_notes(feedback_setup):
    client, db, conversation_id, message_id = feedback_setup

    # Submit
    payload = {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "feedback_type": "helpful",
        "helpful": True,
        "user_notes": "Original",
    }
    resp = client.post("/api/v1/feedback/explicit", json=payload)
    assert resp.status_code == status.HTTP_200_OK
    feedback_id = resp.json()["feedback_id"]

    # Patch
    resp = client.patch(
        f"/api/v1/feedback/{feedback_id}",
        json={"user_notes": "Updated via PATCH"},
    )
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["feedback_id"] == feedback_id

    # Verify
    cursor = db.execute_query(
        "SELECT user_notes FROM conversation_feedback WHERE id = ?",
        (feedback_id,),
    )
    row = cursor.fetchone()
    record = _row_to_dict(row, cursor)
    assert record["user_notes"] == "Updated via PATCH"


@pytest.mark.integration
def test_patch_feedback_not_found(feedback_setup):
    client, _db, _conversation_id, _message_id = feedback_setup

    resp = client.patch(
        "/api/v1/feedback/fb_nonexistent",
        json={"user_notes": "Won't work"},
    )
    assert resp.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.integration
def test_patch_feedback_returns_500_for_db_error(feedback_setup, monkeypatch):
    client, _db, conversation_id, _message_id = feedback_setup

    class _FailingFeedbackStore:
        async def get_feedback_by_id(self, _feedback_id: str):
            return {"conversation_id": conversation_id}

        async def merge_feedback_update(self, _feedback_id: str, *, issues=None, user_notes=None):
            raise CharactersRAGDBError("update feedback failed")

    class _FailingFeedbackSystem:
        def __init__(self, chacha_db):
            self.user_feedback = _FailingFeedbackStore()

    monkeypatch.setattr(feedback_endpoint, "UnifiedFeedbackSystem", _FailingFeedbackSystem)

    resp = client.patch(
        "/api/v1/feedback/fb_problem",
        json={"user_notes": "Won't work"},
    )

    assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert resp.json()["detail"] == "Failed to update feedback"


@pytest.mark.integration
def test_explicit_feedback_finalizes_pending_idempotency_merges(feedback_setup, monkeypatch):
    client, db, conversation_id, message_id = feedback_setup

    captured_dedupe_key: dict[str, str] = {}
    original_reserve = feedback_endpoint._reserve_idempotency_record

    async def _capture_reserve(db, key: str, issues: list[str], user_notes: str | None):
        reserved, record = await original_reserve(db, key, issues, user_notes)
        if reserved:
            captured_dedupe_key["value"] = key
        return reserved, record

    original_submit = UnifiedFeedbackSystem.submit_feedback

    async def _submit_with_pending_merge(self, **kwargs):
        dedupe_key = captured_dedupe_key.get("value")
        if dedupe_key:
            await feedback_endpoint._update_idempotency_record(
                db,
                dedupe_key,
                ["missing_details", "incorrect_information"],
                "Merged while pending",
            )
        return await original_submit(self, **kwargs)

    monkeypatch.setattr(feedback_endpoint, "_reserve_idempotency_record", _capture_reserve)
    monkeypatch.setattr(UnifiedFeedbackSystem, "submit_feedback", _submit_with_pending_merge)

    payload = {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "feedback_type": "helpful",
        "helpful": False,
        "issues": ["missing_details"],
        "user_notes": "Original notes",
    }

    resp = client.post("/api/v1/feedback/explicit", json=payload)
    assert resp.status_code == status.HTTP_200_OK
    feedback_id = resp.json()["feedback_id"]

    cursor = db.execute_query(
        "SELECT issues, user_notes FROM conversation_feedback WHERE id = ?",
        (feedback_id,),
    )
    row = cursor.fetchone()
    assert row is not None
    record = _row_to_dict(row, cursor)
    issues = json.loads(record["issues"]) if record.get("issues") else []
    assert set(issues) == {"missing_details", "incorrect_information"}
    assert record["user_notes"] == "Merged while pending"

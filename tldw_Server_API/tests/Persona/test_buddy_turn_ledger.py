"""Lease reacquisition must not revive work or hide committed Chat replies."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.buddy_turns import _service, router
from tldw_Server_API.app.core.Buddy.service import BuddyService
from tldw_Server_API.app.core.DB_Management.Buddy_Turns_DB import (
    BuddyPublicationRevokedError,
    BuddyTurnRepository,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def ledger(tmp_path):
    db = CharactersRAGDB(tmp_path / "turns.sqlite", "1")
    repository = BuddyTurnRepository(db, "1")
    conversation_id = db.add_conversation({"title": "Lease test", "client_id": "1"})
    repository.claim("owner")
    row = repository.create(
        {
            "id": "accepted-turn",
            "owner_id": "owner",
            "client_slot": "default",
            "client_request_id": "same-request",
            "request_digest": "input-digest",
            "conversation_id": conversation_id,
            "conversation_title": "Lease test",
            "conversation_version": 1,
            "workspace_id": None,
            "attachment_version": 1,
            "created_at": "2026-09-08T00:00:00Z",
        }
    )
    repository.transition(row["id"], "running")
    yield repository, row
    db.close_connection()


def _expire(repository):
    with repository.db.transaction() as conn:
        conn.execute("UPDATE buddy_turn_owners SET expires_at = 0 WHERE user_id = ?", (repository.user_id,))


@pytest.mark.parametrize("status", ["queued", "running"])
def test_same_expired_owner_cannot_revive_unpublished_work(ledger, status):
    repository, row = ledger
    repository.transition(row["id"], status)
    _expire(repository)

    repository.claim("owner")

    current = repository.get(row["id"])
    assert current["status"] == "failed"
    assert current["error_code"] == "interrupted_unknown"
    repository.transition(row["id"], "running")
    with repository.db.transaction() as conn, pytest.raises(BuddyPublicationRevokedError):
        repository.assert_publication(conn, row, row["conversation_id"])
    assert repository.by_key("same-request")["id"] == row["id"]


def test_live_owner_renewal_keeps_running_work_publishable(ledger):
    repository, row = ledger
    repository.claim("owner")
    assert repository.get(row["id"])["status"] == "running"
    with repository.db.transaction() as conn:
        repository.assert_publication(conn, row, row["conversation_id"])


@pytest.mark.parametrize("reconcile", ["read", "same_owner", "new_owner", "runtime_failure", "shutdown"])
def test_expiry_reconciles_committed_reply_as_completed(ledger, reconcile):
    repository, row = ledger
    # Use the same transaction ordering as the canonical Chat persistence seam.
    with repository.db.transaction() as conn:
        repository.assert_publication(conn, row, row["conversation_id"])
        message_id = repository.db.add_message(
            {"conversation_id": row["conversation_id"], "sender": "assistant", "content": "Already committed"}
        )
        repository.record_message(conn, row["id"], "assistant", message_id)
    _expire(repository)

    if reconcile == "read":
        repository.expire_interrupted()
    elif reconcile in {"runtime_failure", "shutdown"}:
        error_code = "interrupted_unknown" if reconcile == "shutdown" else "completion_unknown"
        repository.transition(row["id"], "failed", error_code=error_code)
    else:
        repository.claim("owner" if reconcile == "same_owner" else "replacement")

    current = repository.get(row["id"])
    assert current["status"] == "completed"
    assert current["error_code"] is None
    assert current["result_message_id"] == message_id
    assert repository.by_key("same-request")["status"] == "completed"
    assert repository.stop(row["id"])["status"] == "completed"
    assert len(repository.db.get_messages_for_conversation(row["conversation_id"])) == 1


def test_active_status_filter_finds_old_work_behind_a_full_recent_page(ledger):
    repository, row = ledger
    second = repository.create(
        {**row, "id": "second-active", "client_request_id": "second-active", "created_at": "2026-09-08T01:00:00Z"}
    )
    for index in range(100):
        completed = repository.create(
            {
                **row,
                "id": f"recent-{index:03d}",
                "client_request_id": f"recent-{index:03d}",
                "created_at": "2026-09-09T00:00:00Z",
            }
        )
        repository.transition(completed["id"], "completed")
    repository.create(
        {
            **row,
            "id": "another-slot",
            "client_request_id": "another-slot",
            "client_slot": "other",
            "created_at": "2026-09-10T00:00:00Z",
        }
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/buddies")
    app.dependency_overrides[_service] = lambda: BuddyService(repository.db, "1")

    with TestClient(app) as client:
        recent = client.get("/api/v1/buddies/turns?limit=100")
        assert recent.status_code == 200
        assert len(recent.json()["turns"]) == 100
        assert all(turn["status"] == "completed" for turn in recent.json()["turns"])
        active = client.get("/api/v1/buddies/turns?status=active")
        assert active.status_code == 200
        assert {turn["id"] for turn in active.json()["turns"]} == {row["id"], second["id"]}
        page = client.get("/api/v1/buddies/turns?status=active&limit=1&offset=1").json()
        assert page["limit"] == 1
        assert page["offset"] == 1
        assert [turn["id"] for turn in page["turns"]] == [row["id"]]
        assert client.get("/api/v1/buddies/turns?status=completed").status_code == 422
        assert client.get("/api/v1/buddies/turns?status=active&limit=101").status_code == 422
        assert client.get("/api/v1/buddies/turns?status=active&offset=-1").status_code == 422

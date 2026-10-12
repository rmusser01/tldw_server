"""Original retention preparation leases are not publication or byte admission."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.native_fork_store import NativeAttempt
from tldw_Server_API.tests.DB_Management.test_native_fork_transactions import native_fork_db as native_fork_db
from tldw_Server_API.tests.DB_Management.test_native_fork_transactions import owner
from tldw_Server_API.tests.DB_Management.test_native_retention_receipts import KIND, OPERATION, seeded
from tldw_Server_API.tests.DB_Management.test_native_retention_reservation import original_request, row

pytestmark = pytest.mark.integration


def grant(db, value):
    with db.transaction() as conn:
        return db.native_forks.claim_retention_attempt(owner(), value, conn=conn)


def test_typed_grant_assigns_one_lease_and_live_repeat_stays_pending(native_fork_db):
    db = native_fork_db
    value = original_request(db)
    before = datetime.now(timezone.utc)
    attempt = grant(db, value)
    after = datetime.now(timezone.utc)
    assert isinstance(attempt, NativeAttempt) and attempt.generation == 1
    assert (
        before + timedelta(seconds=300)
        <= datetime.fromisoformat(attempt.lease_expires_at)
        <= after + timedelta(seconds=300)
    )
    original = row(db)
    assert grant(db, value).state == "pending"
    assert row(db) == original


def test_stale_source_does_not_grant_or_rotate_lease(native_fork_db):
    db = native_fork_db
    value = original_request(db)
    grant(db, value)
    before = row(db)
    db.add_message({"conversation_id": value.source_conversation_id, "sender": "user", "content": "Changed"})
    with pytest.raises(ValueError, match="stale_retention_source"):
        grant(db, value)
    assert row(db) == before


def test_actual_fractional_stored_generation_never_creates_attempt(native_fork_db):
    db = native_fork_db
    if db.backend_type.value != "sqlite":
        pytest.skip("PostgreSQL INTEGER cannot store a fractional generation")
    value = original_request(db)
    with db.transaction() as conn:
        db.native_forks.reserve_retention_operation(owner(), value, conn=conn)
        conn.execute("UPDATE native_chat_operations SET attempt_generation = 0.5")
    before = row(db)
    with pytest.raises(ValueError):
        with db.transaction() as conn:
            db.native_forks.claim_attempt(
                owner(), KIND, OPERATION, value.request_digest, now=datetime.now(timezone.utc), conn=conn
            )
    assert row(db) == before


def test_post_grant_error_rolls_back_new_key_and_preserves_caller_work(native_fork_db, monkeypatch):
    db = native_fork_db
    value = original_request(db)
    original = db.native_forks.claim_attempt

    def fail_after(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("lease update interrupted")

    monkeypatch.setattr(db.native_forks, "claim_attempt", fail_after)
    with db.transaction() as conn:
        caller = db.add_conversation({"character_id": None, "title": "Caller survives"}, conn=conn)
        with pytest.raises(RuntimeError, match="lease update interrupted"):
            db.native_forks.claim_retention_attempt(owner(), value, conn=conn)
    assert row(db) is None and db.get_conversation_by_id(caller)["title"] == "Caller survives"


@pytest.mark.parametrize("terminal", [False, True])
def test_original_saved_result_never_calls_preparation_clock_or_claim(native_fork_db, monkeypatch, terminal):
    import tldw_Server_API.app.core.DB_Management.chacha.native_fork_store as module

    db = native_fork_db
    authorized, value, result = seeded(db)
    if terminal:
        assert db.hard_delete_conversation(value.source_conversation_id)
    before = row(db)

    class ForbiddenClock(datetime):
        @classmethod
        def now(cls, tz=None):
            pytest.fail("saved result read preparation clock")

    monkeypatch.setattr(module, "datetime", ForbiddenClock)
    monkeypatch.setattr(db.native_forks, "claim_attempt", lambda *a, **k: pytest.fail("committed result claimed lease"))
    monkeypatch.setattr(
        db.native_forks, "read_retention_source_revision", lambda *a, **k: pytest.fail("committed source rechecked")
    )
    with db.transaction() as conn:
        returned = db.native_forks.claim_retention_attempt(authorized, value, conn=conn)
    assert returned.state == ("gone" if terminal else "committed")
    if not terminal:
        assert returned.receipt == result
    assert row(db) == before


def test_preparation_clock_is_captured_after_real_reservation(native_fork_db, monkeypatch):
    import tldw_Server_API.app.core.DB_Management.chacha.native_fork_store as module

    db = native_fork_db
    value = original_request(db)
    events = []
    original = db.native_forks.reserve_retention_operation

    def admitted(*args, **kwargs):
        result = original(*args, **kwargs)
        events.append("admitted")
        return result

    point = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            assert events == ["admitted"]
            events.append("clock")
            return point

    monkeypatch.setattr(db.native_forks, "reserve_retention_operation", admitted)
    monkeypatch.setattr(module, "datetime", Clock)
    attempt = grant(db, value)
    assert events == ["admitted", "clock"]
    assert datetime.fromisoformat(attempt.lease_expires_at) == point + timedelta(seconds=300)


def test_concurrent_typed_grants_only_one_generation(native_fork_db):
    db = native_fork_db
    value = original_request(db)
    started = threading.Barrier(2)

    def claim():
        started.wait(timeout=8)
        try:
            return grant(db, value)
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = [future.result(timeout=20) for future in [workers.submit(claim) for _ in range(2)]]
    assert sum(isinstance(result, NativeAttempt) for result in results) == 1
    assert sum(getattr(result, "state", None) == "pending" for result in results) == 1
    assert row(db)["attempt_generation"] == 1

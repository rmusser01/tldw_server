"""Typed retention reserves metadata, not claims or actual publication."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeAssetRetentionRequestV1
from tldw_Server_API.tests.DB_Management.test_native_fork_transactions import native_fork_db as native_fork_db
from tldw_Server_API.tests.DB_Management.test_native_fork_transactions import owner
from tldw_Server_API.tests.DB_Management.test_native_retention_receipts import KIND, OPERATION, payload, request, seeded

pytestmark = pytest.mark.integration


def original_request(db):
    cid = db.add_conversation({"character_id": None, "title": "Original reservation source"})
    with db.transaction() as conn:
        revision = db.native_forks.read_retention_source_revision(owner(), cid, conn=conn)
    return request(payload(cid) | {"expected_asset_manifest_revision": revision})


def reserve(db, value, *, authorized=None):
    with db.transaction() as conn:
        return db.native_forks.reserve_retention_operation(authorized or owner(), value, conn=conn)


def row(db):
    record = db.execute_query(
        "SELECT * FROM native_chat_operations WHERE operation_kind = ? AND operation_id = ?", (KIND, OPERATION)
    ).fetchone()
    return dict(record) if record else None


def test_first_reservation_freezes_original_source_and_replays_without_new_work(native_fork_db, monkeypatch):
    db = native_fork_db
    value = original_request(db)
    result = reserve(db, value)
    assert result.state == "preparing"
    original = row(db)
    assert original["source_conversation_id"] == value.source_conversation_id
    assert json.loads(original["canonical_request_json"]) == value.model_dump(mode="json")
    assert original["attempt_generation"] == 0 and original["child_conversation_id"] is None

    assert reserve(db, value) == result
    assert row(db) == original
    for table in ("native_chat_asset_candidates", "native_chat_asset_claims", "native_chat_quota_intents"):
        assert db.execute_query("SELECT COUNT(*) AS n FROM " + table).fetchone()["n"] == 0


@pytest.mark.parametrize("fault", ["owner", "digest", "changed_request", "raw_request"])
def test_invalid_request_cannot_mutate_existing_key(native_fork_db, fault):
    db = native_fork_db
    value = original_request(db)
    reserve(db, value)
    before = row(db)
    authorized = owner()
    if fault == "owner":
        authorized = replace(owner(), owner_key="foreign")
    elif fault == "digest":
        value = NativeAssetRetentionRequestV1.model_validate(
            value.model_dump(mode="python") | {"request_digest": "f" * 64}
        )
    elif fault == "changed_request":
        value = request(value.model_dump(mode="json") | {"expected_asset_manifest_revision": "changed"})
    else:
        value = value.model_dump(mode="json")
    with pytest.raises(ValueError):
        reserve(db, value, authorized=authorized)
    assert row(db) == before


def test_stale_new_source_rolls_back_reservation_but_preserves_caller_work(native_fork_db):
    db = native_fork_db
    value = original_request(db)
    db.add_message({"conversation_id": value.source_conversation_id, "sender": "user", "content": "Source changed"})
    with db.transaction() as conn:
        caller = db.add_conversation({"character_id": None, "title": "Caller work"}, conn=conn)
        with pytest.raises(ValueError, match="stale_retention_source"):
            db.native_forks.reserve_retention_operation(owner(), value, conn=conn)
    assert row(db) is None
    assert db.get_conversation_by_id(caller)["title"] == "Caller work"


def test_existing_pending_lookup_survives_source_deletion_but_resume_refuses(native_fork_db, monkeypatch):
    db = native_fork_db
    value = original_request(db)
    before_result = reserve(db, value)
    assert db.hard_delete_conversation(value.source_conversation_id)
    before = row(db)
    with pytest.raises(ValueError):
        reserve(db, value)
    assert row(db) == before
    monkeypatch.setattr(
        db.native_forks, "read_retention_source_revision", lambda *a, **k: pytest.fail("source checked on lookup")
    )
    with db.transaction() as conn:
        result = db.native_forks.resolve_operation(owner(), KIND, OPERATION, value.request_digest, conn=conn)
    assert result == before_result and row(db) == before


@pytest.mark.parametrize("terminal", [False, True])
def test_original_seeded_committed_or_gone_resolves_before_source_reader(native_fork_db, monkeypatch, terminal):
    db = native_fork_db
    authorized, value, result = seeded(db)
    if terminal:
        assert db.hard_delete_conversation(value.source_conversation_id)
    before = row(db)
    monkeypatch.setattr(
        db.native_forks,
        "read_retention_source_revision",
        lambda *a, **k: pytest.fail("source rechecked before known result"),
    )
    resolved = reserve(db, value, authorized=authorized)
    assert resolved.state == ("gone" if terminal else "committed")
    if not terminal:
        assert resolved.receipt == result
    assert row(db) == before


def test_original_transaction_required(native_fork_db):
    value = original_request(native_fork_db)
    with pytest.raises(ValueError, match="active_transaction_required"):
        native_fork_db.native_forks.reserve_retention_operation(owner(), value, conn=native_fork_db.get_connection())


def test_two_concurrent_same_key_calls_keep_one_original_reservation(native_fork_db, monkeypatch):
    db = native_fork_db
    value = original_request(db)
    started = threading.Barrier(2)
    if db.backend_type.value == "postgresql":
        absent = threading.Barrier(2)
        original = db.native_forks._row

        def simultaneous_absence(*args, **kwargs):
            found = original(*args, **kwargs)
            if found is None:
                absent.wait(timeout=8)
            return found

        monkeypatch.setattr(db.native_forks, "_row", simultaneous_absence)

    def run():
        started.wait(timeout=8)
        try:
            return reserve(db, value)
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        results = [future.result(timeout=20) for future in futures]
    assert results[0] == results[1]
    assert db.execute_query("SELECT COUNT(*) AS n FROM native_chat_operations").fetchone()["n"] == 1
    assert row(db)["source_conversation_id"] == value.source_conversation_id


def test_committed_insert_conflict_winner_resolves_before_poisoned_source(native_fork_db, monkeypatch):
    """Inject only the stale absence observation; the committed row is real and seeded."""
    db = native_fork_db
    authorized, value, result = seeded(db)
    before = row(db)
    original = db.native_forks._row
    first = True

    def stale_absence(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            return None
        return original(*args, **kwargs)

    monkeypatch.setattr(db.native_forks, "_row", stale_absence)
    monkeypatch.setattr(
        db.native_forks,
        "read_retention_source_revision",
        lambda *a, **k: pytest.fail("winning result lost to source check"),
    )
    assert reserve(db, value, authorized=authorized).receipt == result
    assert row(db) == before


def test_unqualified_legacy_preparing_binding_is_never_adopted(native_fork_db, monkeypatch):
    db = native_fork_db
    value = original_request(db)
    with db.transaction() as conn:
        db.native_forks.reserve_operation(
            owner(), KIND, OPERATION, value.request_digest, value.model_dump(mode="json"), conn=conn
        )
    before = row(db)
    monkeypatch.setattr(
        db.native_forks,
        "read_retention_source_revision",
        lambda *a, **k: pytest.fail("legacy key reached source admission"),
    )
    with pytest.raises(ValueError, match="retention_source_binding_mismatch"):
        reserve(db, value)
    assert row(db) == before


def test_minimal_gone_tombstone_remains_resolvable_without_source_or_request_body(native_fork_db, monkeypatch):
    db = native_fork_db
    authorized, value, _ = seeded(db)
    assert db.hard_delete_conversation(value.source_conversation_id)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE native_chat_operations SET canonical_request_json = NULL,source_conversation_id = NULL,child_conversation_id = NULL"
        )
    before = row(db)
    monkeypatch.setattr(
        db.native_forks, "read_retention_source_revision", lambda *a, **k: pytest.fail("tombstone recaptured source")
    )
    assert reserve(db, value, authorized=authorized).state == "gone"
    assert row(db) == before


def test_stale_existing_preparation_keeps_original_key_and_generation(native_fork_db):
    db = native_fork_db
    value = original_request(db)
    reserve(db, value)
    before = row(db)
    db.add_message(
        {"conversation_id": value.source_conversation_id, "sender": "user", "content": "Changed while pending"}
    )
    with pytest.raises(ValueError, match="stale_retention_source"):
        reserve(db, value)
    assert row(db) == before and before["attempt_generation"] == 0


def test_existing_preparing_projection_mismatch_cannot_resume_or_mutate(native_fork_db):
    db = native_fork_db
    value = original_request(db)
    reserve(db, value)
    with db.transaction() as conn:
        conn.execute("UPDATE native_chat_operations SET projection_version = 'unsupported-v2'")
    before = row(db)
    with pytest.raises(ValueError):
        reserve(db, value)
    assert row(db) == before

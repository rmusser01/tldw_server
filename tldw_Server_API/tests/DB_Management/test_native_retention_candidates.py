"""Frozen-reference retention candidate metadata is not byte publication."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.tests.DB_Management.test_native_fork_transactions import native_fork_db as native_fork_db
from tldw_Server_API.tests.DB_Management.test_native_fork_transactions import owner
from tldw_Server_API.tests.DB_Management.test_native_retention_receipts import OPERATION, request
from tldw_Server_API.tests.DB_Management.test_native_retention_reservation import original_request

pytestmark = pytest.mark.integration


def setup(db, *, unavailable=False):
    value = original_request(db)
    if unavailable:
        data = value.model_dump(mode="json")
        data["asset_manifest"][0].update(representation="unavailable_v1", hash=None, fidelity_disposition="unavailable")
        value = request(data)
    with db.transaction() as conn:
        attempt = db.native_forks.claim_retention_attempt(owner(), value, conn=conn)
    return value, attempt.generation


def reserve(db, value, generation, *, conn=None, authorized=None, **changes):
    kwargs = {
        "reference_id": "ref-a",
        "candidate_id": "candidate-one",
        "namespace_id": "a" * 32,
        "upload_id": "upload-one",
        "size_bytes": 17,
        "mime_type": "image/png",
    } | changes
    if conn is None:
        with db.transaction() as current:
            return reserve(db, value, generation, conn=current, authorized=authorized, **kwargs)
    return db.native_assets.reserve_retention_candidate(authorized or owner(), value, generation, conn=conn, **kwargs)


def candidate(db):
    row = db.execute_query(
        "SELECT * FROM native_chat_asset_candidates WHERE candidate_id = ?", ("candidate-one",)
    ).fetchone()
    return dict(row) if row else None


def test_original_reference_derives_hash_representation_and_metadata_only(native_fork_db):
    db = native_fork_db
    value, generation = setup(db)
    result = reserve(db, value, generation)
    actual = candidate(db)
    assert result.state == "reserved" and actual["expected_hash"] == value.asset_manifest[0].hash
    assert actual["representation"] == value.asset_manifest[0].representation
    assert actual["operation_id"] == OPERATION and actual["attempt_generation"] == generation
    for table in ("native_chat_asset_claims", "native_chat_asset_references", "native_chat_quota_intents"):
        assert db.execute_query("SELECT COUNT(*) AS n FROM " + table).fetchone()["n"] == 0


def test_same_candidate_replay_and_compatible_alias_preserve_original(native_fork_db):
    db = native_fork_db
    value, generation = setup(db)
    original = reserve(db, value, generation)
    before = candidate(db)
    assert reserve(db, value, generation) == original
    assert reserve(db, value, generation, reference_id="ref-b") == original
    assert candidate(db) == before


@pytest.mark.parametrize("fault", ["reference", "unavailable", "source", "owner", "generation", "expiry", "request"])
def test_invalid_retention_candidate_writes_nothing(native_fork_db, fault):
    db = native_fork_db
    value, generation = setup(db, unavailable=fault == "unavailable")
    kwargs = {}
    authorized = owner()
    if fault == "reference":
        kwargs["reference_id"] = "foreign-reference"
    elif fault == "source":
        db.add_message({"conversation_id": value.source_conversation_id, "sender": "user", "content": "Changed"})
    elif fault == "owner":
        authorized = replace(owner(), owner_key="foreign")
    elif fault == "generation":
        generation += 1
    elif fault == "expiry":
        with db.transaction() as conn:
            conn.execute(
                "UPDATE native_chat_operations SET lease_expires_at = ?",
                ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),),
            )
    elif fault == "request":
        data = value.model_dump(mode="json")
        data["asset_manifest"][0]["hash"] = "f" * 64
        value = request(data)
    with pytest.raises(ValueError):
        reserve(db, value, generation, authorized=authorized, **kwargs)
    assert candidate(db) is None


@pytest.mark.parametrize(
    "field,value", [("size_bytes", 18), ("mime_type", "image/jpeg"), ("namespace_id", "b" * 32), ("upload_id", "other")]
)
def test_replay_cannot_replace_original_candidate_metadata(native_fork_db, field, value):
    db = native_fork_db
    original, generation = setup(db)
    reserve(db, original, generation)
    before = candidate(db)
    with pytest.raises(ValueError, match="candidate_id_conflict"):
        reserve(db, original, generation, **{field: value})
    assert candidate(db) == before


def test_post_insert_failure_caught_by_caller_preserves_caller_work(native_fork_db, monkeypatch):
    db = native_fork_db
    value, generation = setup(db)
    original = db.native_assets.reserve_candidate

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("candidate insertion interrupted")

    monkeypatch.setattr(db.native_assets, "reserve_candidate", interrupted)
    with db.transaction() as conn:
        caller = db.add_conversation({"id": "caller-work", "character_id": None, "title": "Caller"}, conn=conn)
        with pytest.raises(RuntimeError, match="candidate insertion interrupted"):
            reserve(db, value, generation, conn=conn)
    assert candidate(db) is None and db.get_conversation_by_id(caller)["title"] == "Caller"


def test_original_transaction_required(native_fork_db):
    db = native_fork_db
    value, generation = setup(db)
    with pytest.raises(ValueError, match="active_transaction_required"):
        reserve(db, value, generation, conn=db.get_connection())


@pytest.mark.parametrize("expired", [False, True])
def test_clock_advance_after_real_reservation_checks_original_lease(native_fork_db, monkeypatch, expired):
    import tldw_Server_API.app.core.DB_Management.chacha.native_chat_asset_store as module

    db = native_fork_db
    value, generation = setup(db)
    start = datetime.now(timezone.utc)
    clock = {"current": start}

    class ControlledClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["current"]

    monkeypatch.setattr(module, "datetime", ControlledClock)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE native_chat_operations SET lease_expires_at = ?", ((start + timedelta(seconds=1)).isoformat(),)
        )
    before = dict(db.execute_query("SELECT * FROM native_chat_operations").fetchone())
    original = db.native_assets.reserve_candidate

    def advance_after_insert(*args, **kwargs):
        result = original(*args, **kwargs)
        clock["current"] = start + timedelta(seconds=2 if expired else 0.5)
        return result

    monkeypatch.setattr(db.native_assets, "reserve_candidate", advance_after_insert)
    with db.transaction() as conn:
        caller = db.add_conversation({"id": "caller-work", "character_id": None, "title": "Caller"}, conn=conn)
        if expired:
            with pytest.raises(ValueError, match="stale_attempt"):
                reserve(db, value, generation, conn=conn)
        else:
            assert reserve(db, value, generation, conn=conn).state == "reserved"
    assert (candidate(db) is None) == expired
    assert dict(db.execute_query("SELECT * FROM native_chat_operations").fetchone()) == before
    assert db.get_conversation_by_id(caller)["title"] == "Caller"


@pytest.mark.parametrize("boundary", ["candidate", "reservation", "lease"])
def test_unchecked_scope_cannot_be_normalized_into_original_request(native_fork_db, boundary):
    from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeScopeV1

    db = native_fork_db
    value, generation = setup(db)
    malformed = value.model_copy(update={"scope": NativeScopeV1.model_construct(kind="global", workspace_id="foreign")})
    before = dict(db.execute_query("SELECT * FROM native_chat_operations").fetchone())
    with pytest.raises(ValueError):
        with db.transaction() as conn:
            if boundary == "candidate":
                reserve(db, malformed, generation, conn=conn)
            elif boundary == "reservation":
                db.native_forks.reserve_retention_operation(owner(), malformed, conn=conn)
            else:
                db.native_forks.claim_retention_attempt(owner(), malformed, conn=conn)
    assert candidate(db) is None and dict(db.execute_query("SELECT * FROM native_chat_operations").fetchone()) == before

"""Native fork operation keys survive replay, rollback, and child deletion."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeScopeV1
from tldw_Server_API.app.core.Chat.native_fork_projection import AuthorizedNativeOwner
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.native_fork_store import NativeAttempt
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture(params=["sqlite", "postgres"])
def native_fork_db(request, tmp_path):
    kwargs = {"db_path": str(tmp_path / "operations.sqlite"), "client_id": "alice"}
    if request.param == "postgres":
        kwargs["backend"] = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(**kwargs)
    yield db
    db.close_connection()
    if request.param == "postgres":
        kwargs["backend"].get_pool().close_all()


def owner(owner_key="native:alice"):
    return AuthorizedNativeOwner("alice", owner_key, NativeScopeV1(kind="global"))


def workspace_owner():
    return AuthorizedNativeOwner(
        "alice", "native:alice", NativeScopeV1(kind="workspace", workspace_id="workspace-one")
    )


def reserve(db, digest="sha256:one", operation_id="fork-one"):
    with db.transaction() as conn:
        return db.native_forks.reserve_operation(
            owner(), "native_fork_v1", operation_id, digest, {"title": "A"}, conn=conn
        )


def test_same_key_replays_original_receipt_and_changed_digest_conflicts(native_fork_db):
    db = native_fork_db
    original = reserve(db)
    assert original.state == "preparing"
    assert reserve(db) == original
    with db.transaction() as conn:
        with pytest.raises(ValueError, match="operation_id_conflict"):
            db.native_forks.reserve_operation(
                owner(), "native_fork_v1", "fork-one", "sha256:changed", {"title": "A"}, conn=conn
            )
        with pytest.raises(ValueError, match="operation_owner_mismatch"):
            db.native_forks.resolve_operation(owner("native:other"), "native_fork_v1", "fork-one", "sha256:one", conn=conn)
        assert db.native_forks.resolve_operation(owner(), "native_fork_v1", "fork-one", "sha256:one", conn=conn) == original


def test_reservation_rollback_leaves_key_unrecorded(native_fork_db):
    db = native_fork_db
    with pytest.raises(RuntimeError, match="rollback"):
        with db.transaction() as conn:
            db.native_forks.reserve_operation(owner(), "native_fork_v1", "rolled-back", "sha256:one", {}, conn=conn)
            raise RuntimeError("rollback")
    with db.transaction() as conn:
        result = db.native_forks.resolve_operation(owner(), "native_fork_v1", "rolled-back", "sha256:one", conn=conn)
    assert result.state == "not_recorded"


def test_deleted_child_operation_never_becomes_fresh(native_fork_db):
    db = native_fork_db
    child = db.add_conversation({"character_id": 1, "title": "Child"})
    reserve(db)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE native_chat_operations SET state = 'committed', child_conversation_id = ?, result_json = '{}' "
            "WHERE client_id = 'alice' AND operation_id = 'fork-one'",
            (child,),
        )
        conn.execute(
            "UPDATE conversations SET native_creation_operation_kind = 'native_fork_v1', "
            "native_creation_operation_id = 'fork-one', required_projection_version = 'native-fork-v1' WHERE id = ?",
            (child,),
        )
    assert db.hard_delete_conversation(child)
    with db.transaction() as conn:
        result = db.native_forks.resolve_operation(owner(), "native_fork_v1", "fork-one", "sha256:one", conn=conn)
    assert result.state == "gone"


def test_soft_delete_then_restore_does_not_revive_creation_key(native_fork_db):
    db = native_fork_db
    child = db.add_conversation({"character_id": 1, "title": "Restorable child"})
    reserve(db)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE native_chat_operations SET state = 'committed', child_conversation_id = ?, result_json = '{}' "
            "WHERE client_id = 'alice' AND operation_id = 'fork-one'",
            (child,),
        )
        version = conn.execute("SELECT version FROM conversations WHERE id = ?", (child,)).fetchone()["version"]
    assert db.soft_delete_conversation(child, version)
    assert db.restore_conversation(child, version + 1)
    with db.transaction() as conn:
        result = db.native_forks.resolve_operation(owner(), "native_fork_v1", "fork-one", "sha256:one", conn=conn)
    assert result.state == "gone"


def test_reserve_replay_reconciles_missing_child_before_disclosing_map(native_fork_db):
    db = native_fork_db
    child = db.add_conversation({"character_id": 1, "title": "Child"})
    reserve(db)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE native_chat_operations SET state = 'committed', child_conversation_id = ?, result_json = '{}' "
            "WHERE client_id = 'alice' AND operation_id = 'fork-one'",
            (child,),
        )
        # Model a legacy/admin deletion path that bypasses the native store.
        conn.execute("DELETE FROM conversations WHERE id = ?", (child,))
    assert reserve(db).state == "gone"


def test_committed_replay_requires_protected_child_binding(native_fork_db):
    db = native_fork_db
    child = db.add_conversation({"character_id": 1, "title": "Child"})
    reserve(db)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE native_chat_operations SET state = 'committed', child_conversation_id = ?, result_json = '{}' "
            "WHERE client_id = 'alice' AND operation_id = 'fork-one'",
            (child,),
        )
        conn.execute(
            "UPDATE conversations SET required_projection_version = 'native-fork-v1' WHERE id = ?",
            (child,),
        )
    with pytest.raises(ValueError, match="operation_child_binding_mismatch"):
        reserve(db)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET native_creation_operation_kind = 'native_fork_v1', "
            "native_creation_operation_id = 'fork-one' WHERE id = ?",
            (child,),
        )
    assert reserve(db).state == "committed"


def test_incomplete_committed_receipt_fails_closed(native_fork_db):
    db = native_fork_db
    reserve(db)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE native_chat_operations SET state = 'committed' "
            "WHERE client_id = 'alice' AND operation_id = 'fork-one'"
        )
    with pytest.raises(ValueError, match="operation_receipt_incomplete"):
        reserve(db)


def test_conflict_after_absent_observation_reconciles_winning_receipt(native_fork_db, monkeypatch):
    db = native_fork_db
    reserve(db)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE native_chat_operations SET state = 'committed', child_conversation_id = 'missing-child' "
            "WHERE client_id = 'alice' AND operation_id = 'fork-one'"
        )
    original = db.native_forks._row
    observations = 0

    def stale_first_read(*args, **kwargs):
        nonlocal observations
        observations += 1
        return None if observations == 1 else original(*args, **kwargs)

    monkeypatch.setattr(db.native_forks, "_row", stale_first_read)
    assert reserve(db).state == "gone"


def test_claim_attempt_leases_one_generation_and_rotates_after_expiry(native_fork_db):
    db = native_fork_db
    reserve(db)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with db.transaction() as conn:
        first = db.native_forks.claim_attempt(owner(), "native_fork_v1", "fork-one", "sha256:one", now=now, conn=conn)
    assert first.generation == 1
    with db.transaction() as conn:
        live = db.native_forks.claim_attempt(
            owner(), "native_fork_v1", "fork-one", "sha256:one", now=now + timedelta(seconds=1), conn=conn
        )
    assert live.state == "pending"
    with db.transaction() as conn:
        second = db.native_forks.claim_attempt(
            owner(), "native_fork_v1", "fork-one", "sha256:one", now=now + timedelta(seconds=301), conn=conn
        )
    assert second.generation == 2


def test_expired_attempt_with_candidate_requires_reconciliation(native_fork_db):
    db = native_fork_db
    reserve(db)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with db.transaction() as conn:
        first = db.native_forks.claim_attempt(owner(), "native_fork_v1", "fork-one", "sha256:one", now=now, conn=conn)
        conn.execute(
            "INSERT INTO native_chat_asset_candidates "
            "(client_id, candidate_id, operation_kind, operation_id, owner_key, attempt_generation, "
            "storage_namespace_id, upload_id, expected_hash, expected_size_bytes, mime_type, "
            "representation, state, created_at, updated_at) "
            "VALUES ('alice', 'candidate-1', 'native_fork_v1', 'fork-one', 'native:alice', ?, "
            "'namespace', 'upload', ?, 1, 'image/png', 'file_v1', 'prepared', ?, ?)",
            (first.generation, "a" * 64, now.isoformat(), now.isoformat()),
        )
    with db.transaction() as conn:
        result = db.native_forks.claim_attempt(
            owner(), "native_fork_v1", "fork-one", "sha256:one", now=now + timedelta(seconds=301), conn=conn
        )
    assert result.state == "reconciliation_required"


def test_postgres_concurrent_claim_has_one_active_generation(pg_database_config):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="alice", backend=backend)
    reserve(db)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    barrier = threading.Barrier(2)

    def claim():
        barrier.wait(timeout=10)
        with db.transaction() as conn:
            return db.native_forks.claim_attempt(
                owner(), "native_fork_v1", "fork-one", "sha256:one", now=now, conn=conn
            )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(claim) for _ in range(2)]
            results = [future.result(timeout=20) for future in futures]
        assert sum(isinstance(result, NativeAttempt) for result in results) == 1
        assert sum(getattr(result, "state", None) == "pending" for result in results) == 1
    finally:
        db.close_connection()
        backend.get_pool().close_all()


def test_candidate_reservation_replays_exact_inputs_and_rejects_changed_hash(native_fork_db):
    db = native_fork_db
    reserve(db)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with db.transaction() as conn:
        attempt = db.native_forks.claim_attempt(owner(), "native_fork_v1", "fork-one", "sha256:one", now=now, conn=conn)
    assert isinstance(attempt, NativeAttempt)

    def candidate(content_hash: str):
        with db.transaction() as conn:
            return db.native_assets.reserve_candidate(
                owner(), "native_fork_v1", "fork-one", "sha256:one", attempt.generation,
                "candidate-one", "namespace-one", "upload-one", content_hash, 3, "image/png", "file_v1",
                now=now + timedelta(seconds=1), conn=conn,
            )

    original = candidate("a" * 64)
    assert original.state == "reserved"
    assert candidate("a" * 64) == original
    with pytest.raises(ValueError, match="candidate_id_conflict"):
        candidate("b" * 64)


def test_expired_worker_cannot_reserve_candidate_after_new_generation(native_fork_db):
    db = native_fork_db
    reserve(db)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with db.transaction() as conn:
        first = db.native_forks.claim_attempt(owner(), "native_fork_v1", "fork-one", "sha256:one", now=now, conn=conn)
    with db.transaction() as conn:
        second = db.native_forks.claim_attempt(
            owner(), "native_fork_v1", "fork-one", "sha256:one", now=now + timedelta(seconds=301), conn=conn
        )
    assert isinstance(first, NativeAttempt) and isinstance(second, NativeAttempt)
    with db.transaction() as conn:
        with pytest.raises(ValueError, match="stale_attempt"):
            db.native_assets.reserve_candidate(
                owner(), "native_fork_v1", "fork-one", "sha256:one", first.generation,
                "stale", "namespace", "upload", "a" * 64, 3, "image/png", "file_v1",
                now=now + timedelta(seconds=302), conn=conn,
            )


def test_candidate_preparation_requires_matching_descriptor_and_live_generation(native_fork_db):
    db = native_fork_db
    reserve(db)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with db.transaction() as conn:
        attempt = db.native_forks.claim_attempt(owner(), "native_fork_v1", "fork-one", "sha256:one", now=now, conn=conn)
        db.native_assets.reserve_candidate(
            owner(), "native_fork_v1", "fork-one", "sha256:one", attempt.generation,
            "candidate", "namespace", "upload", "a" * 64, 3, "image/png", "file_v1",
            now=now, conn=conn,
        )
    with db.transaction() as conn:
        with pytest.raises(ValueError, match="invalid_candidate_identity"):
            db.native_assets.mark_candidate_prepared(
                owner(), "native_fork_v1", "fork-one", "sha256:one", True,
                "candidate", "a" * 64, 3, "image/png", "storage-key", now=now + timedelta(seconds=1), conn=conn,
            )
        with pytest.raises(ValueError, match="candidate_content_mismatch"):
            db.native_assets.mark_candidate_prepared(
                owner(), "native_fork_v1", "fork-one", "sha256:one", attempt.generation,
                "candidate", "b" * 64, 3, "image/png", "storage-key", now=now + timedelta(seconds=1), conn=conn,
            )
        ready = db.native_assets.mark_candidate_prepared(
            owner(), "native_fork_v1", "fork-one", "sha256:one", attempt.generation,
            "candidate", "a" * 64, 3, "image/png", "storage-key", now=now + timedelta(seconds=1), conn=conn,
        )
    assert ready.state == "prepared"
    with db.transaction() as conn:
        with pytest.raises(ValueError, match="stale_attempt"):
            db.native_assets.mark_candidate_prepared(
                owner(), "native_fork_v1", "fork-one", "sha256:one", attempt.generation,
                "candidate", "a" * 64, 3, "image/png", "storage-key", now=now + timedelta(seconds=301), conn=conn,
            )


def test_expired_unclaimed_candidate_reclaims_before_new_generation(native_fork_db):
    db = native_fork_db
    reserve(db)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with db.transaction() as conn:
        attempt = db.native_forks.claim_attempt(
            owner(), "native_fork_v1", "fork-one", "sha256:one", now=now, conn=conn
        )
        db.native_assets.reserve_candidate(
            owner(), "native_fork_v1", "fork-one", "sha256:one", attempt.generation,
            "candidate", "namespace", "upload", "a" * 64, 3, "image/png", "file_v1",
            now=now, conn=conn,
        )
    expired = now + timedelta(seconds=301)
    with db.transaction() as conn:
        reclaim = db.native_assets.claim_reclamation(
            owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate", now=expired, conn=conn
        )
        assert reclaim.state == "reclaiming"
        assert (reclaim.namespace_id, reclaim.upload_id, reclaim.expected_hash, reclaim.size_bytes) == (
            "namespace", "upload", "a" * 64, 3,
        )
    with db.transaction() as conn:
        assert db.native_assets.claim_reclamation(
            owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate", now=expired, conn=conn
        ) == reclaim
        assert db.native_forks.claim_attempt(
            owner(), "native_fork_v1", "fork-one", "sha256:one", now=expired, conn=conn
        ).state == "reconciliation_required"
        with pytest.raises(ValueError, match="stale_reclamation"):
            db.native_assets.finish_reclamation(
                owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate", attempt.generation + 1,
                conn=conn,
            )
        assert db.native_assets.finish_reclamation(
            owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate", attempt.generation,
            conn=conn,
        )
    with db.transaction() as conn:
        assert db.native_forks.claim_attempt(
            owner(), "native_fork_v1", "fork-one", "sha256:one", now=expired, conn=conn
        ).generation == 2


def test_active_or_claimed_candidate_cannot_enter_reclamation(native_fork_db):
    db = native_fork_db
    reserve(db)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with db.transaction() as conn:
        attempt = db.native_forks.claim_attempt(
            owner(), "native_fork_v1", "fork-one", "sha256:one", now=now, conn=conn
        )
        db.native_assets.reserve_candidate(
            owner(), "native_fork_v1", "fork-one", "sha256:one", attempt.generation,
            "candidate", "namespace", "upload", "a" * 64, 3, "image/png", "file_v1",
            now=now, conn=conn,
        )
        assert db.native_assets.claim_reclamation(
            owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate",
            now=now + timedelta(seconds=1), conn=conn,
        ) is None
        conn.execute(
            "INSERT INTO native_chat_asset_claims "
            "(client_id, claim_id, owner_key, conversation_id, candidate_id, asset_id, asset_revision, "
            "content_hash, size_bytes, mime_type, representation, state, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'live', ?, ?)",
            ("alice", "claim", "native:alice", "child", "candidate", "asset", 1,
             "a" * 64, 3, "image/png", "file_v1", now.isoformat(), now.isoformat()),
        )
    with db.transaction() as conn:
        with pytest.raises(ValueError, match="candidate_still_claimed"):
            db.native_assets.claim_reclamation(
                owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate",
                now=now + timedelta(seconds=301), conn=conn,
            )


def test_reclamation_waits_for_confirmed_quota_release(native_fork_db):
    db = native_fork_db
    reserve(db)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with db.transaction() as conn:
        attempt = db.native_forks.claim_attempt(
            owner(), "native_fork_v1", "fork-one", "sha256:one", now=now, conn=conn
        )
        db.native_assets.reserve_candidate(
            owner(), "native_fork_v1", "fork-one", "sha256:one", attempt.generation,
            "candidate", "namespace", "upload", "a" * 64, 3, "image/png", "file_v1",
            now=now, conn=conn,
        )
        conn.execute(
            "INSERT INTO native_chat_quota_intents "
            "(client_id, candidate_id, intent_kind, owner_key, size_bytes, state, created_at, updated_at) "
            "VALUES ('alice', 'candidate', 'reserve', 'native:alice', 3, 'pending', ?, ?)",
            (now.isoformat(), now.isoformat()),
        )
    with db.transaction() as conn:
        db.native_assets.claim_reclamation(
            owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate",
            now=now + timedelta(seconds=301), conn=conn,
        )
        with pytest.raises(ValueError, match="quota_release_unconfirmed"):
            db.native_assets.finish_reclamation(
                owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate", attempt.generation,
                conn=conn,
            )
        conn.execute(
            "INSERT INTO native_chat_quota_intents "
            "(client_id, candidate_id, intent_kind, owner_key, size_bytes, state, created_at, updated_at) "
            "VALUES ('alice', 'candidate', 'release', 'native:alice', 3, 'pending', ?, ?)",
            (now.isoformat(), now.isoformat()),
        )
        with pytest.raises(ValueError, match="quota_release_unconfirmed"):
            db.native_assets.finish_reclamation(
                owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate", attempt.generation,
                conn=conn,
            )
        conn.execute(
            "UPDATE native_chat_quota_intents SET state = 'confirmed' "
            "WHERE candidate_id = 'candidate' AND intent_kind = 'release'"
        )
        conn.execute(
            "UPDATE native_chat_quota_intents SET owner_key = 'other' "
            "WHERE candidate_id = 'candidate' AND intent_kind = 'release'"
        )
        with pytest.raises(ValueError, match="quota_intent_mismatch"):
            db.native_assets.finish_reclamation(
                owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate", attempt.generation,
                conn=conn,
            )
        conn.execute(
            "UPDATE native_chat_quota_intents SET owner_key = 'native:alice' "
            "WHERE candidate_id = 'candidate' AND intent_kind = 'release'"
        )
        assert db.native_assets.finish_reclamation(
            owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate", attempt.generation,
            conn=conn,
        )


def test_gone_adopted_candidate_waits_for_claim_release_even_after_workspace_closure(native_fork_db):
    db = native_fork_db
    db.upsert_workspace("workspace-one", "Workspace One")
    authorized = workspace_owner()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with db.transaction() as conn:
        db.native_forks.reserve_operation(
            authorized, "native_fork_v1", "fork-one", "sha256:one", {}, conn=conn
        )
        attempt = db.native_forks.claim_attempt(
            authorized, "native_fork_v1", "fork-one", "sha256:one", now=now, conn=conn
        )
        db.native_assets.reserve_candidate(
            authorized, "native_fork_v1", "fork-one", "sha256:one", attempt.generation,
            "candidate", "namespace", "upload", "a" * 64, 3, "image/png", "file_v1",
            now=now, conn=conn,
        )
        conn.execute(
            "UPDATE native_chat_asset_candidates SET state = 'adopted' WHERE candidate_id = 'candidate'"
        )
        conn.execute(
            "UPDATE native_chat_operations SET state = 'gone' WHERE operation_id = 'fork-one'"
        )
        conn.execute(
            "UPDATE workspaces SET native_chat_admission_closed = ? WHERE id = ?",
            (True, "workspace-one"),
        )
        conn.execute(
            "INSERT INTO native_chat_asset_claims "
            "(client_id, claim_id, owner_key, conversation_id, candidate_id, asset_id, asset_revision, "
            "content_hash, size_bytes, mime_type, representation, state, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'live', ?, ?)",
            ("alice", "claim", "native:alice", "child", "candidate", "asset", 1,
             "a" * 64, 3, "image/png", "file_v1", now.isoformat(), now.isoformat()),
        )
    with db.transaction() as conn:
        with pytest.raises(ValueError, match="candidate_still_claimed"):
            db.native_assets.claim_reclamation(
                authorized, "native_fork_v1", "fork-one", "sha256:one", "candidate", now=now, conn=conn
            )
        conn.execute("UPDATE native_chat_asset_claims SET state = 'released' WHERE claim_id = 'claim'")
        assert db.native_assets.claim_reclamation(
            authorized, "native_fork_v1", "fork-one", "sha256:one", "candidate", now=now, conn=conn
        ).state == "reclaiming"
        assert db.native_assets.finish_reclamation(
            authorized, "native_fork_v1", "fork-one", "sha256:one", "candidate", attempt.generation,
            conn=conn,
        )
        assert not db.native_assets.finish_reclamation(
            authorized, "native_fork_v1", "fork-one", "sha256:one", "candidate", attempt.generation,
            conn=conn,
        )


@pytest.mark.parametrize("workspace_state", ["open", "closed", "deleted", "staged"])
def test_workspace_candidate_admission_requires_open_owned_scope(native_fork_db, workspace_state):
    db = native_fork_db
    db.upsert_workspace("workspace-one", "Workspace One")
    authorized = workspace_owner()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with db.transaction() as conn:
        db.native_forks.reserve_operation(
            authorized, "native_fork_v1", "fork-one", "sha256:one", {}, conn=conn
        )
        attempt = db.native_forks.claim_attempt(
            authorized, "native_fork_v1", "fork-one", "sha256:one", now=now, conn=conn
        )
    assert isinstance(attempt, NativeAttempt)
    if workspace_state != "open":
        column, value = {
            "closed": ("native_chat_admission_closed", True),
            "deleted": ("deleted", True),
            "staged": ("system_operation_state", "staged"),
        }[workspace_state]
        with db.transaction() as conn:
            conn.execute(f"UPDATE workspaces SET {column} = ? WHERE id = ?", (value, "workspace-one"))  # nosec B608 - fixed test-only column set.
    with db.transaction() as conn:
        if workspace_state == "open":
            candidate = db.native_assets.reserve_candidate(
                authorized, "native_fork_v1", "fork-one", "sha256:one", attempt.generation,
                "candidate", "namespace", "upload", "a" * 64, 3, "image/png", "file_v1",
                now=now + timedelta(seconds=1), conn=conn,
            )
            assert candidate.state == "reserved"
        else:
            with pytest.raises(ValueError, match="workspace_native_unavailable"):
                db.native_assets.reserve_candidate(
                    authorized, "native_fork_v1", "fork-one", "sha256:one", attempt.generation,
                    "candidate", "namespace", "upload", "a" * 64, 3, "image/png", "file_v1",
                    now=now + timedelta(seconds=1), conn=conn,
                )
            assert conn.execute(
                "SELECT COUNT(*) AS n FROM native_chat_asset_candidates WHERE candidate_id = 'candidate'"
            ).fetchone()["n"] == 0


def test_workspace_closure_blocks_prepared_transition(native_fork_db):
    db = native_fork_db
    db.upsert_workspace("workspace-one", "Workspace One")
    authorized = workspace_owner()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with db.transaction() as conn:
        db.native_forks.reserve_operation(
            authorized, "native_fork_v1", "fork-one", "sha256:one", {}, conn=conn
        )
        attempt = db.native_forks.claim_attempt(
            authorized, "native_fork_v1", "fork-one", "sha256:one", now=now, conn=conn
        )
        db.native_assets.reserve_candidate(
            authorized, "native_fork_v1", "fork-one", "sha256:one", attempt.generation,
            "candidate", "namespace", "upload", "a" * 64, 3, "image/png", "file_v1",
            now=now + timedelta(seconds=1), conn=conn,
        )
    with db.transaction() as conn:
        conn.execute(
            "UPDATE workspaces SET native_chat_admission_closed = ? WHERE id = ?",
            (True, "workspace-one"),
        )
    with db.transaction() as conn:
        with pytest.raises(ValueError, match="workspace_native_unavailable"):
            db.native_assets.mark_candidate_prepared(
                authorized, "native_fork_v1", "fork-one", "sha256:one", attempt.generation,
                "candidate", "a" * 64, 3, "image/png", "storage-key",
                now=now + timedelta(seconds=2), conn=conn,
            )
        assert conn.execute(
            "SELECT state FROM native_chat_asset_candidates WHERE candidate_id = 'candidate'"
        ).fetchone()["state"] == "reserved"


@pytest.mark.parametrize("workspace_state", ["closed", "missing", "foreign"])
def test_unavailable_workspace_rejects_new_operation_without_reserving_key(native_fork_db, workspace_state):
    db = native_fork_db
    if workspace_state == "closed":
        db.upsert_workspace("workspace-one", "Workspace One")
        with db.transaction() as conn:
            conn.execute(
                "UPDATE workspaces SET native_chat_admission_closed = ? WHERE id = ?",
                (True, "workspace-one"),
            )
    elif workspace_state == "foreign":
        other = CharactersRAGDB(db_path=db.db_path_str, client_id="bob", backend=db.backend)
        try:
            other.upsert_workspace("workspace-one", "Bob's workspace")
        finally:
            other.close_connection()
    with db.transaction() as conn:
        with pytest.raises(ValueError, match="workspace_native_unavailable"):
            db.native_forks.reserve_operation(
                workspace_owner(), "native_fork_v1", "fork-one", "sha256:one", {}, conn=conn
            )
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM native_chat_operations WHERE operation_id = 'fork-one'"
        ).fetchone()["n"] == 0


def test_closed_workspace_cannot_claim_another_attempt(native_fork_db):
    db = native_fork_db
    db.upsert_workspace("workspace-one", "Workspace One")
    authorized = workspace_owner()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with db.transaction() as conn:
        db.native_forks.reserve_operation(
            authorized, "native_fork_v1", "fork-one", "sha256:one", {}, conn=conn
        )
    with db.transaction() as conn:
        conn.execute(
            "UPDATE workspaces SET native_chat_admission_closed = ? WHERE id = ?",
            (True, "workspace-one"),
        )
    with db.transaction() as conn:
        assert db.native_forks.reserve_operation(
            authorized, "native_fork_v1", "fork-one", "sha256:one", {}, conn=conn
        ).state == "preparing"
        with pytest.raises(ValueError, match="workspace_native_unavailable"):
            db.native_forks.claim_attempt(
                authorized, "native_fork_v1", "fork-one", "sha256:one", now=now, conn=conn
            )
        assert conn.execute(
            "SELECT attempt_generation FROM native_chat_operations WHERE operation_id = 'fork-one'"
        ).fetchone()["attempt_generation"] == 0


def test_postgres_candidate_admission_serializes_with_workspace_closure(pg_database_config, monkeypatch):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="alice", backend=backend)
    db.upsert_workspace("workspace-one", "Workspace One")
    authorized = workspace_owner()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with db.transaction() as conn:
        db.native_forks.reserve_operation(
            authorized, "native_fork_v1", "fork-one", "sha256:one", {}, conn=conn
        )
        attempt = db.native_forks.claim_attempt(
            authorized, "native_fork_v1", "fork-one", "sha256:one", now=now, conn=conn
        )
    admitted = threading.Event()
    release_admission = threading.Event()
    delete_attempted = threading.Event()
    original_admission = db.native_forks.lock_open_workspace
    original_delete_lock = db._lock_native_workspace_delete

    def pause_after_admission(owner, *, conn):
        original_admission(owner, conn=conn)
        admitted.set()
        assert release_admission.wait(timeout=10)

    def observe_delete_lock(conn, workspace_id):
        delete_attempted.set()
        return original_delete_lock(conn, workspace_id)

    monkeypatch.setattr(db.native_forks, "lock_open_workspace", pause_after_admission)
    monkeypatch.setattr(db, "_lock_native_workspace_delete", observe_delete_lock)

    def admit_candidate():
        with db.transaction() as conn:
            db.native_assets.reserve_candidate(
                authorized, "native_fork_v1", "fork-one", "sha256:one", attempt.generation,
                "candidate", "namespace", "upload", "a" * 64, 3, "image/png", "file_v1",
                now=now + timedelta(seconds=1), conn=conn,
            )
        db.close_connection()

    def delete_workspace():
        try:
            return db.delete_workspace("workspace-one", expected_version=1)
        finally:
            db.close_connection()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            admission = pool.submit(admit_candidate)
            assert admitted.wait(timeout=10)
            deletion = pool.submit(delete_workspace)
            assert delete_attempted.wait(timeout=10)
            assert not deletion.done()
            assert not db.get_workspace("workspace-one")["native_chat_admission_closed"]
            release_admission.set()
            admission.result(timeout=20)
            assert deletion.result(timeout=20)
        assert db.get_workspace("workspace-one") is None
    finally:
        release_admission.set()
        db.close_connection()
        backend.get_pool().close_all()


def test_postgres_concurrent_same_key_reserves_one_receipt(pg_database_config, monkeypatch):
    """Two transactions that both observe absence converge on the same key."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="alice", backend=backend)
    barrier = threading.Barrier(2)
    original = db.native_forks._row

    def observe_absence(*args, **kwargs):
        row = original(*args, **kwargs)
        if row is None:
            barrier.wait(timeout=10)
        return row

    monkeypatch.setattr(db.native_forks, "_row", observe_absence)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(reserve, db, "sha256:race", "same-key") for _ in range(2)]
            results = [future.result(timeout=20) for future in futures]
        assert results[0] == results[1]
        with db.transaction() as conn:
            assert conn.execute(
                "SELECT COUNT(*) AS n FROM native_chat_operations WHERE operation_id = 'same-key'"
            ).fetchone()["n"] == 1
    finally:
        db.close_connection()
        backend.get_pool().close_all()

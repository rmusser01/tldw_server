"""Restored native bytes survive permanently non-replayable creation keys."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeScopeV1
from tldw_Server_API.app.core.Chat.native_fork_projection import AuthorizedNativeOwner
from tldw_Server_API.app.core.Sync.v2.blob_store import SyncBlobStoreError
from tldw_Server_API.tests.DB_Management.test_native_chat_asset_io import owner
from tldw_Server_API.tests.DB_Management.test_native_chat_asset_preparation import row
from tldw_Server_API.tests.DB_Management.test_native_chat_asset_read import committed, read

pytest_plugins = ("tldw_Server_API.tests.DB_Management.test_native_chat_asset_io",)
pytestmark = pytest.mark.integration


def records(db):
    """Compare all durable asset authority; this reader must never revive a key."""
    return {
        table: [dict(row) for row in db.execute_query("SELECT * FROM " + table).fetchall()]
        for table in (
            "native_chat_operations",
            "native_chat_asset_candidates",
            "native_chat_asset_claims",
            "native_chat_quota_intents",
        )
    }


def restore(db, cid):
    version = db.get_conversation_by_id(cid)["version"]
    assert db.soft_delete_conversation(cid, version)
    assert db.restore_conversation(cid, version + 1)


@pytest.mark.asyncio
async def test_actual_restore_reads_exact_retained_bytes_without_reviving_creation_key(
    native_fork_db, tmp_path, monkeypatch
):
    db = native_fork_db
    adapter, root, cid, claim, payload = await committed(db, tmp_path, monkeypatch)
    with read(adapter, cid, claim) as (_, handle):
        assert handle.read() == payload
    version = db.get_conversation_by_id(cid)["version"]
    assert db.soft_delete_conversation(cid, version)
    with pytest.raises(ValueError):
        with read(adapter, cid, claim):
            pytest.fail("deleted conversation yielded a byte handle")
    with pytest.raises(ValueError, match="candidate_still_claimed"):
        adapter.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", claim["candidate_id"])
    assert db.restore_conversation(cid, version + 1)
    before = records(db)
    assert before["native_chat_operations"][0]["state"] == "gone"
    assert before["native_chat_operations"][0]["result_json"] is None
    with read(adapter, cid, claim) as (descriptor, handle):
        assert handle.read() == payload
        assert descriptor["asset_id"] == claim["asset_id"] and descriptor["asset_revision"] == 1
        assert (
            not db.get_connection().in_transaction
            if hasattr(db.get_connection(), "in_transaction")
            else db._connection_state().tx_depth == 0
        )
    assert handle.closed
    with db.transaction() as conn:
        result = db.native_forks.resolve_operation(owner(), "native_fork_v1", "fork-one", "sha256:one", conn=conn)
    assert result.state == "gone" and result.receipt is None
    assert records(db) == before


@pytest.mark.asyncio
async def test_original_operation_projection_mismatch_never_yields_bytes(native_fork_db, tmp_path, monkeypatch):
    db = native_fork_db
    adapter, _, cid, claim, _ = await committed(db, tmp_path, monkeypatch)
    with db.transaction() as conn:
        conn.execute("UPDATE native_chat_operations SET projection_version = 'unsupported-v2'")
    before = records(db)
    with pytest.raises(ValueError):
        with read(adapter, cid, claim):
            pytest.fail("unsupported projection yielded bytes")
    assert records(db) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "foreign_owner",
        "wrong_scope",
        "deleted_again",
        "purged",
        "expired",
        "rejected",
        "preparing",
        "released_claim",
        "candidate_generation",
        "quota_pending",
        "projection",
        "unknown_gone_reason",
        "release_started",
    ],
)
async def test_restored_bytes_still_require_current_exact_owned_claim(native_fork_db, tmp_path, monkeypatch, fault):
    db = native_fork_db
    adapter, _, cid, claim, _ = await committed(db, tmp_path, monkeypatch)
    restore(db, cid)
    authorized = owner()
    if fault == "foreign_owner":
        authorized = AuthorizedNativeOwner("1", "native:foreign", owner().scope)
    elif fault == "wrong_scope":
        authorized = AuthorizedNativeOwner(
            "1", owner().owner_key, NativeScopeV1(kind="workspace", workspace_id="foreign")
        )
    elif fault == "deleted_again":
        assert db.soft_delete_conversation(cid, db.get_conversation_by_id(cid)["version"])
    elif fault == "purged":
        assert db.hard_delete_conversation(cid)
    else:
        with db.transaction() as conn:
            if fault in {"expired", "rejected", "preparing"}:
                conn.execute("UPDATE native_chat_operations SET state = ?", (fault,))
            elif fault == "released_claim":
                conn.execute("UPDATE native_chat_asset_claims SET state = 'released'")
            elif fault == "candidate_generation":
                conn.execute("UPDATE native_chat_asset_candidates SET attempt_generation = attempt_generation + 1")
            elif fault == "quota_pending":
                conn.execute("UPDATE native_chat_quota_intents SET state = 'pending'")
            elif fault == "projection":
                conn.execute("UPDATE native_chat_operations SET projection_version = 'unsupported-v2'")
            elif fault == "unknown_gone_reason":
                conn.execute("UPDATE native_chat_operations SET terminal_reason = 'unverified'")
            elif fault == "release_started":
                conn.execute(
                    "INSERT INTO native_chat_quota_intents (client_id,candidate_id,intent_kind,owner_key,size_bytes,state,created_at,updated_at) SELECT client_id,candidate_id,'release',owner_key,size_bytes,'pending',created_at,updated_at FROM native_chat_quota_intents"
                )
    before = records(db)
    with pytest.raises(ValueError):
        with read(adapter, cid, claim, authorized=authorized):
            pytest.fail("retired or foreign restored authority yielded bytes")
    assert records(db) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["missing", "tampered", "truncated"])
async def test_restored_physical_bytes_still_verify_before_yield(native_fork_db, tmp_path, monkeypatch, fault):
    db = native_fork_db
    adapter, root, cid, claim, payload = await committed(db, tmp_path, monkeypatch)
    restore(db, cid)
    target = root / "chat_assets" / row(db)["storage_key"]
    if fault == "missing":
        target.unlink()
    elif fault == "tampered":
        target.write_bytes(b"x" * len(payload))
    else:
        target.write_bytes(payload[:-1])
    before = records(db)
    with pytest.raises((ValueError, SyncBlobStoreError, OSError)):
        with read(adapter, cid, claim):
            pytest.fail("corrupt restored bytes yielded handle")
    assert records(db) == before


@pytest.mark.asyncio
async def test_actual_deletion_during_restored_read_guard_refuses_before_yield(native_fork_db, tmp_path, monkeypatch):
    db = native_fork_db
    adapter, _, cid, claim, _ = await committed(db, tmp_path, monkeypatch)
    restore(db, cid)
    entered, release, yielded = threading.Event(), threading.Event(), threading.Event()
    original_guard = adapter._guard

    @contextmanager
    def held_guard(*args, **kwargs):
        with original_guard(*args, **kwargs) as guarded:
            entered.set()
            assert release.wait(8), "owned read guard not released"
            yield guarded

    monkeypatch.setattr(adapter, "_guard", held_guard)

    def consume():
        try:
            with read(adapter, cid, claim):
                yielded.set()
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(consume)
        try:
            assert entered.wait(5), "restored reader did not reach actual owned guard"
            assert db.soft_delete_conversation(cid, db.get_conversation_by_id(cid)["version"])
            release.set()
            with pytest.raises(ValueError):
                pending.result(timeout=8)
            assert not yielded.is_set()
        finally:
            release.set()
    assert records(db)["native_chat_operations"][0]["state"] == "gone"


@pytest.mark.asyncio
async def test_actual_workspace_closure_denies_restored_bytes(native_fork_db, tmp_path, monkeypatch):
    """Seed scope metadata for the actual closure negative, not workspace publication."""
    db = native_fork_db
    adapter, _, cid, claim, _ = await committed(db, tmp_path, monkeypatch)
    restore(db, cid)
    db.upsert_workspace("workspace", "Owned workspace")
    authorized = AuthorizedNativeOwner(
        "1", owner().owner_key, NativeScopeV1(kind="workspace", workspace_id="workspace")
    )
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET scope_type = 'workspace',workspace_id = 'workspace' WHERE id = ?", (cid,)
        )
        conn.execute("UPDATE native_chat_operations SET scope_type = 'workspace',workspace_id = 'workspace'")

    def held_cascade(*args, **kwargs):
        raise RuntimeError("held workspace cascade")

    monkeypatch.setattr(db, "soft_delete_conversation", held_cascade)
    with pytest.raises(Exception, match="workspace_delete_incomplete"):
        db.delete_workspace("workspace", expected_version=1)
    assert db.get_workspace("workspace")["native_chat_admission_closed"]
    before = records(db)
    with pytest.raises(ValueError, match="workspace_native_unavailable"):
        with read(adapter, cid, claim, authorized=authorized):
            pytest.fail("closed workspace yielded restored bytes")
    assert records(db) == before

"""Exact committed native claim reads use owned immutable bytes and anchored handles."""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeScopeV1
from tldw_Server_API.app.core.Chat.native_fork_projection import AuthorizedNativeOwner
from tldw_Server_API.app.core.Sync.v2.blob_store import SyncBlobStoreError
from tldw_Server_API.tests.DB_Management.test_native_chat_asset_adoption import adopt, child
from tldw_Server_API.tests.DB_Management.test_native_chat_asset_io import owner
from tldw_Server_API.tests.DB_Management.test_native_chat_asset_preparation import case, prepare, row

pytest_plugins = ("tldw_Server_API.tests.DB_Management.test_native_chat_asset_io",)
pytestmark = pytest.mark.integration


async def committed(db, tmp_path, monkeypatch):
    payload = "Résumé 😀 retained bytes".encode()
    adapter, generation, root = case(db, tmp_path, monkeypatch, payload, "text/plain", "document_text_v1")
    await prepare(adapter, generation, payload, "processed.txt")
    cid = child(db)
    with db.transaction() as conn:
        claim = adopt(db, generation, cid, conn=conn)
        conn.execute(
            "UPDATE native_chat_operations SET state = 'committed', child_conversation_id = ? WHERE operation_id = ?",
            (cid, "fork-one"),
        )
    return adapter, root, cid, claim, payload


def read(adapter, cid, claim, *, authorized=None, revision="1"):
    return adapter.read_owned_asset(authorized or owner(), cid, claim["asset_id"], revision)


@pytest.mark.asyncio
async def test_committed_exact_claim_reads_verified_owned_handle_after_upload_closes(
    native_fork_db, tmp_path, monkeypatch
):
    db = native_fork_db
    adapter, root, cid, claim, payload = await committed(db, tmp_path, monkeypatch)
    original_guard = adapter._guard

    @contextmanager
    def observed_guard(*args, **kwargs):
        connection = db.get_connection()
        if hasattr(connection, "in_transaction"):
            assert not connection.in_transaction
        else:
            assert db._connection_state().tx_depth == 0
        with original_guard(*args, **kwargs) as guarded:
            yield guarded

    monkeypatch.setattr(adapter, "_guard", observed_guard)
    with read(adapter, cid, claim) as (descriptor, handle):
        assert handle.read() == payload
        assert descriptor["asset_id"] == claim["asset_id"] and descriptor["asset_revision"] == 1
        assert descriptor["content_hash"] == hashlib.sha256(payload).hexdigest()
        assert descriptor["size_bytes"] == len(payload) and descriptor["mime_type"] == "text/plain"
        assert descriptor["representation"] == "document_text_v1"
        connection = db.get_connection()
        if hasattr(connection, "in_transaction"):
            assert not connection.in_transaction
        else:
            assert db._connection_state().tx_depth == 0
    assert handle.closed
    assert row(db)["state"] == "adopted"
    assert (root / "chat_assets" / row(db)["storage_key"]).read_bytes() == payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "foreign_owner",
        "wrong_scope",
        "missing_child",
        "foreign_child",
        "deleted_child",
        "unprotected_child",
        "uncommitted",
        "wrong_operation_target",
        "released_claim",
        "candidate_prepared",
        "candidate_generation",
        "release_started",
        "quota_pending",
        "wrong_hash",
        "wrong_size",
        "wrong_mime",
        "wrong_representation",
        "wrong_storage_key",
        "wrong_revision",
        "noncanonical_revision",
        "integer_revision",
        "boolean_revision",
    ],
)
async def test_invalid_claim_read_never_yields_handle(native_fork_db, tmp_path, monkeypatch, fault):
    db = native_fork_db
    adapter, root, cid, claim, payload = await committed(db, tmp_path, monkeypatch)
    before = row(db)
    kwargs = {}
    if fault == "foreign_owner":
        kwargs["authorized"] = AuthorizedNativeOwner("1", "native:foreign", owner().scope)
    elif fault == "wrong_scope":
        kwargs["authorized"] = AuthorizedNativeOwner(
            "1", owner().owner_key, NativeScopeV1(kind="workspace", workspace_id="foreign")
        )
    elif fault == "missing_child":
        cid = "missing"
    elif fault.endswith("revision"):
        kwargs["revision"] = {
            "wrong_revision": "2",
            "noncanonical_revision": "01",
            "integer_revision": 1,
            "boolean_revision": True,
        }[fault]
    else:
        with db.transaction() as conn:
            if fault == "foreign_child":
                conn.execute("UPDATE conversations SET client_id = '2' WHERE id = ?", (cid,))
            elif fault == "deleted_child":
                conn.execute("UPDATE conversations SET deleted = ? WHERE id = ?", (True, cid))
            elif fault == "unprotected_child":
                conn.execute("UPDATE conversations SET required_projection_version = NULL WHERE id = ?", (cid,))
            elif fault == "uncommitted":
                conn.execute("UPDATE native_chat_operations SET state = 'preparing'")
            elif fault == "wrong_operation_target":
                conn.execute("UPDATE native_chat_operations SET child_conversation_id = 'foreign'")
            elif fault == "released_claim":
                conn.execute("UPDATE native_chat_asset_claims SET state = 'released'")
            elif fault == "candidate_prepared":
                conn.execute("UPDATE native_chat_asset_candidates SET state = 'prepared'")
            elif fault == "candidate_generation":
                conn.execute("UPDATE native_chat_asset_candidates SET attempt_generation = attempt_generation + 1")
            elif fault == "release_started":
                conn.execute(
                    "INSERT INTO native_chat_quota_intents (client_id,candidate_id,intent_kind,owner_key,size_bytes,state,created_at,updated_at) SELECT client_id,candidate_id,'release',owner_key,size_bytes,'pending',created_at,updated_at FROM native_chat_quota_intents WHERE intent_kind = 'reserve'"
                )
            elif fault == "quota_pending":
                conn.execute("UPDATE native_chat_quota_intents SET state = 'pending'")
            elif fault == "wrong_hash":
                conn.execute("UPDATE native_chat_asset_candidates SET expected_hash = ?", ("a" * 64,))
            elif fault == "wrong_size":
                conn.execute("UPDATE native_chat_asset_candidates SET expected_size_bytes = expected_size_bytes + 1")
            elif fault == "wrong_mime":
                conn.execute("UPDATE native_chat_asset_candidates SET mime_type = 'image/png'")
            elif fault == "wrong_representation":
                conn.execute("UPDATE native_chat_asset_candidates SET representation = 'embedded_image_v1'")
            elif fault == "wrong_storage_key":
                conn.execute("UPDATE native_chat_asset_candidates SET storage_key = '../foreign'")
    with pytest.raises((ValueError, SyncBlobStoreError, OSError)):
        with read(adapter, cid, claim, **kwargs):
            pytest.fail("invalid admission yielded an asset handle")
    assert (root / "chat_assets" / before["storage_key"]).read_bytes() == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["missing", "tampered", "truncated", "symlink", "namespace_symlink"])
async def test_wrong_physical_bytes_never_yield_handle(native_fork_db, tmp_path, monkeypatch, fault):
    db = native_fork_db
    adapter, root, cid, claim, payload = await committed(db, tmp_path, monkeypatch)
    target = root / "chat_assets" / row(db)["storage_key"]
    if fault == "missing":
        target.unlink()
    elif fault == "tampered":
        target.write_bytes(b"x" * len(payload))
    elif fault == "truncated":
        target.write_bytes(payload[:-1])
    elif fault == "symlink":
        target.unlink()
        foreign = tmp_path / "foreign.txt"
        foreign.write_bytes(payload)
        target.symlink_to(foreign)
    else:
        namespace = target.parent
        moved = namespace.with_name(namespace.name + "-moved")
        namespace.rename(moved)
        namespace.symlink_to(moved, target_is_directory=True)
    with pytest.raises((ValueError, SyncBlobStoreError, OSError)):
        with read(adapter, cid, claim):
            pytest.fail("wrong physical bytes yielded an asset handle")
    assert row(db)["state"] == "adopted"


@pytest.mark.asyncio
async def test_owned_read_refuses_nested_transaction_before_byte_guard(native_fork_db, tmp_path, monkeypatch):
    db = native_fork_db
    adapter, root, cid, claim, payload = await committed(db, tmp_path, monkeypatch)
    with db.transaction():
        with pytest.raises(ValueError, match="native_assets_outermost_required"):
            with read(adapter, cid, claim):
                pytest.fail("nested reader yielded a handle")
    assert (root / "chat_assets" / row(db)["storage_key"]).read_bytes() == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["root", "db_owner", "db_client"])
async def test_final_real_descriptor_boundary_rechecks_retired_identity_before_yield(
    native_fork_db, tmp_path, monkeypatch, fault
):
    db = native_fork_db
    adapter, root, cid, claim, payload = await committed(db, tmp_path, monkeypatch)
    entered, release, yielded = threading.Event(), threading.Event(), threading.Event()
    original = db.native_assets.read_committed_asset
    calls = 0

    def held_descriptor(*args, **kwargs):
        nonlocal calls
        current = original(*args, **kwargs)
        calls += 1
        if calls == 3:
            entered.set()
            assert release.wait(8), "final descriptor was not released"
        return current

    monkeypatch.setattr(db.native_assets, "read_committed_asset", held_descriptor)

    def consume():
        try:
            with read(adapter, cid, claim) as (_, handle):
                yielded.set()
                return handle.read()
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(consume)
        try:
            assert entered.wait(5), "actual final descriptor boundary not reached"
            if fault == "root":
                (root / "chat_assets").rename(root / "retired-chat-assets")
                (root / "chat_assets").mkdir()
            elif fault == "db_owner":
                monkeypatch.setattr(db, "_owner_user_id", "foreign")
            else:
                monkeypatch.setattr(db, "client_id", "2")
            release.set()
            with pytest.raises((ValueError, SyncBlobStoreError, OSError)):
                pending.result(timeout=8)
            assert not yielded.is_set(), "retired identity exposed an asset handle"
        finally:
            release.set()
    monkeypatch.undo()
    assert row(db)["state"] == "adopted"

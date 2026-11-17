"""Actual prepared fork bytes become exact live claims only in the original child."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.Chat.native_fork_projection import AuthorizedNativeOwner
from tldw_Server_API.tests.DB_Management.test_native_chat_asset_io import _expire, owner
from tldw_Server_API.tests.DB_Management.test_native_chat_asset_preparation import case, prepare, row

pytest_plugins = ("tldw_Server_API.tests.DB_Management.test_native_chat_asset_io",)
pytestmark = pytest.mark.integration


def child(db, *, operation="fork-one"):
    cid = db.add_conversation({"title": "Private preparing child", "character_id": None})
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET required_projection_version = 'native-fork-v1', "
            "native_creation_operation_kind = 'native_fork_v1', native_creation_operation_id = ?, "
            "native_bundle_json = ? WHERE id = ? AND client_id = ?",
            (operation, json.dumps({"version": 1, "state": "preparing"}), cid, db.client_id),
        )
    return cid


def adopt(db, generation, cid, *, authorized=None, candidate="candidate-one", operation="fork-one", conn):
    return db.native_assets.adopt_fork_candidate(
        authorized or owner(),
        operation,
        "sha256:one",
        generation,
        candidate,
        cid,
        now=datetime.now(timezone.utc),
        conn=conn,
    )


def claims(db):
    with db.transaction() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM native_chat_asset_claims").fetchall()]


@pytest.mark.asyncio
async def test_real_prepared_bytes_adopt_once_into_original_protected_child(native_fork_db, tmp_path, monkeypatch):
    db = native_fork_db
    payload = "Résumé 😀".encode()
    adapter, generation, root = case(db, tmp_path, monkeypatch, payload, "text/plain", "document_text_v1")
    await prepare(adapter, generation, payload, "processed.txt")
    cid = child(db)
    before = row(db)
    with db.transaction() as conn:
        first = adopt(db, generation, cid, conn=conn)
        replay = adopt(db, generation, cid, conn=conn)
    assert first == replay == claims(db)[0]
    assert first["conversation_id"] == cid and first["candidate_id"] == "candidate-one"
    assert first["owner_key"] == owner().owner_key and first["state"] == "live"
    assert first["content_hash"] == hashlib.sha256(payload).hexdigest()
    assert first["size_bytes"] == len(payload) and first["mime_type"] == "text/plain"
    assert first["representation"] == "document_text_v1" and first["asset_revision"] == 1
    assert row(db)["state"] == "adopted"
    assert (root / "chat_assets" / before["storage_key"]).read_bytes() == payload
    with db.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM native_chat_quota_intents").fetchone()["n"] == 1
        assert conn.execute("SELECT COUNT(*) AS n FROM native_chat_asset_references").fetchone()["n"] == 0
        assert db.native_forks._row(owner(), "native_fork_v1", "fork-one", conn=conn)["state"] == "preparing"


@pytest.mark.asyncio
async def test_caller_failure_rolls_back_claim_and_adoption_without_touching_bytes(
    native_fork_db, tmp_path, monkeypatch
):
    db = native_fork_db
    payload = b"exact bytes"
    adapter, generation, root = case(db, tmp_path, monkeypatch, payload, "text/plain", "document_text_v1")
    await prepare(adapter, generation, payload, "processed.txt")
    cid = child(db)
    before = row(db)
    with pytest.raises(RuntimeError, match="graph_write_failed"):
        with db.transaction() as conn:
            adopt(db, generation, cid, conn=conn)
            raise RuntimeError("graph_write_failed")
    assert not claims(db)
    assert row(db)["state"] == "prepared"
    assert (root / "chat_assets" / before["storage_key"]).read_bytes() == payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "foreign_owner",
        "wrong_operation",
        "wrong_generation",
        "missing_child",
        "deleted_child",
        "foreign_child",
        "wrong_scope",
        "unprotected_child",
        "wrong_creation",
        "expired",
        "quota_pending",
        "release_started",
        "reclaiming",
        "invalid_key",
        "unsupported_representation",
    ],
)
async def test_invalid_admission_installs_no_claim(native_fork_db, tmp_path, monkeypatch, fault):
    db = native_fork_db
    payload = b"exact bytes"
    adapter, generation, root = case(db, tmp_path, monkeypatch, payload, "text/plain", "document_text_v1")
    await prepare(adapter, generation, payload, "processed.txt")
    cid = child(db)
    before = row(db)
    kwargs = {}
    if fault == "foreign_owner":
        kwargs["authorized"] = AuthorizedNativeOwner("1", "native:another", owner().scope)
    elif fault == "wrong_operation":
        kwargs["operation"] = "another-operation"
    elif fault == "wrong_generation":
        generation += 1
    elif fault == "missing_child":
        cid = "missing"
    elif fault == "expired":
        _expire(db)
    else:
        if fault == "wrong_scope":
            db.upsert_workspace("other", "Other owned workspace")
        with db.transaction() as conn:
            if fault == "deleted_child":
                conn.execute("UPDATE conversations SET deleted = ? WHERE id = ?", (True, cid))
            elif fault == "foreign_child":
                conn.execute("UPDATE conversations SET client_id = '2' WHERE id = ?", (cid,))
            elif fault == "wrong_scope":
                conn.execute(
                    "UPDATE conversations SET scope_type = 'workspace', workspace_id = 'other' WHERE id = ?", (cid,)
                )
            elif fault == "unprotected_child":
                conn.execute("UPDATE conversations SET required_projection_version = NULL WHERE id = ?", (cid,))
            elif fault == "wrong_creation":
                conn.execute("UPDATE conversations SET native_creation_operation_id = 'other' WHERE id = ?", (cid,))
            elif fault == "quota_pending":
                conn.execute("UPDATE native_chat_quota_intents SET state = 'pending'")
            elif fault == "release_started":
                stamp = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO native_chat_quota_intents (client_id,candidate_id,intent_kind,owner_key,size_bytes,state,created_at,updated_at) VALUES (?,?, 'release',?,?, 'pending',?,?)",
                    ("1", "candidate-one", owner().owner_key, len(payload), stamp, stamp),
                )
            elif fault == "reclaiming":
                _expire(db)
                db.native_assets.claim_reclamation(
                    owner(),
                    "native_fork_v1",
                    "fork-one",
                    "sha256:one",
                    "candidate-one",
                    now=datetime.now(timezone.utc),
                    conn=conn,
                )
            elif fault == "invalid_key":
                conn.execute("UPDATE native_chat_asset_candidates SET storage_key = '../foreign'")
            elif fault == "unsupported_representation":
                conn.execute("UPDATE native_chat_asset_candidates SET representation = 'file_v1'")
    with pytest.raises(ValueError):
        with db.transaction() as conn:
            adopt(db, generation, cid, conn=conn, **kwargs)
    assert not claims(db)
    assert (root / "chat_assets" / before["storage_key"]).read_bytes() == payload
    assert row(db)["state"] == ("reclaiming" if fault == "reclaiming" else "prepared")


@pytest.mark.asyncio
async def test_live_claim_replay_cannot_retarget_or_revive_released_claim(native_fork_db, tmp_path, monkeypatch):
    db = native_fork_db
    payload = b"exact bytes"
    adapter, generation, _ = case(db, tmp_path, monkeypatch, payload, "text/plain", "document_text_v1")
    await prepare(adapter, generation, payload, "processed.txt")
    cid, other = child(db), child(db)
    with db.transaction() as conn:
        first = adopt(db, generation, cid, conn=conn)
    with pytest.raises(ValueError):
        with db.transaction() as conn:
            adopt(db, generation, other, conn=conn)
    with db.transaction() as conn:
        conn.execute("UPDATE native_chat_asset_claims SET state = 'released' WHERE claim_id = ?", (first["claim_id"],))
    with pytest.raises(ValueError):
        with db.transaction() as conn:
            adopt(db, generation, cid, conn=conn)
    assert len(claims(db)) == 1 and claims(db)[0]["state"] == "released"
    assert row(db)["state"] == "adopted"


@pytest.mark.asyncio
async def test_adoption_requires_actual_owning_transaction(native_fork_db, tmp_path, monkeypatch):
    db = native_fork_db
    payload = b"exact bytes"
    adapter, generation, _ = case(db, tmp_path, monkeypatch, payload, "text/plain", "document_text_v1")
    await prepare(adapter, generation, payload, "processed.txt")
    cid = child(db)
    with pytest.raises(ValueError, match="active_transaction_required"):
        adopt(db, generation, cid, conn=db.get_connection())
    assert not claims(db)
    assert row(db)["state"] == "prepared"


@pytest.mark.asyncio
async def test_cleanup_winner_fences_adoption_before_actual_byte_removal(native_fork_db, tmp_path, monkeypatch):
    db = native_fork_db
    payload = b"exact bytes"
    adapter, generation, root = case(db, tmp_path, monkeypatch, payload, "text/plain", "document_text_v1")
    await prepare(adapter, generation, payload, "processed.txt")
    cid = child(db)
    before = row(db)
    entered, release = threading.Event(), threading.Event()
    original_guard = adapter._guard

    @contextmanager
    def paused_guard(*args, **kwargs):
        connection = db.get_connection()
        if hasattr(connection, "in_transaction"):
            assert not connection.in_transaction
        else:
            assert db._connection_state().tx_depth == 0
        entered.set()
        assert release.wait(8), "cleanup was not released"
        with original_guard(*args, **kwargs) as guarded:
            yield guarded

    monkeypatch.setattr(adapter, "_guard", paused_guard)
    _expire(db)

    def cleanup():
        try:
            return adapter.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one")
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(cleanup)
        try:
            assert entered.wait(5), "cleanup did not commit its fence"
            assert row(db)["state"] == "reclaiming"
            assert (root / "chat_assets" / before["storage_key"]).read_bytes() == payload
            with pytest.raises(ValueError, match="stale_attempt|candidate_state_conflict"):
                with db.transaction() as conn:
                    adopt(db, generation, cid, conn=conn)
            assert not claims(db)
            release.set()
            with pytest.raises(ValueError, match="quota_release_unconfirmed"):
                pending.result(timeout=8)
        finally:
            release.set()
    assert not (root / "chat_assets" / before["storage_key"]).exists()
    assert row(db)["state"] == "reclaiming"
    with db.transaction() as conn:
        assert (
            conn.execute("SELECT state FROM native_chat_quota_intents WHERE intent_kind = 'release'").fetchone()[
                "state"
            ]
            == "pending"
        )


@pytest.mark.asyncio
async def test_adoption_winner_blocks_cleanup_under_real_transaction(native_fork_db, tmp_path, monkeypatch):
    db = native_fork_db
    payload = b"exact bytes"
    adapter, generation, root = case(db, tmp_path, monkeypatch, payload, "text/plain", "document_text_v1")
    await prepare(adapter, generation, payload, "processed.txt")
    cid = child(db)
    before = row(db)
    installed, release, cleanup_started = threading.Event(), threading.Event(), threading.Event()
    original_transaction = db.transaction
    worker = threading.local()

    @contextmanager
    def observed_transaction(**kwargs):
        if getattr(worker, "cleanup", False):
            cleanup_started.set()
        with original_transaction(**kwargs) as conn:
            yield conn

    monkeypatch.setattr(db, "transaction", observed_transaction)

    def action(kind):
        try:
            if kind == "adopt":
                with db.transaction() as conn:
                    claim = adopt(db, generation, cid, conn=conn)
                    conn.execute(
                        "UPDATE native_chat_operations SET lease_expires_at = ? WHERE operation_id = ?",
                        ("2000-01-01T00:00:00+00:00", "fork-one"),
                    )
                    installed.set()
                    assert release.wait(8), "adoption was not released"
                    return claim
            worker.cleanup = True
            return adapter.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one")
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=2) as workers:
        pending = workers.submit(action, "adopt")
        try:
            assert installed.wait(5), "adoption did not reach its transaction barrier"
            cleanup = workers.submit(action, "cleanup")
            assert cleanup_started.wait(5), "cleanup did not attempt admission"
            assert not cleanup.done()
            release.set()
            claim = pending.result(timeout=8)
            with pytest.raises(ValueError, match="candidate_still_claimed"):
                cleanup.result(timeout=8)
        finally:
            release.set()
    assert claims(db) == [claim]
    assert row(db)["state"] == "adopted"
    assert (root / "chat_assets" / before["storage_key"]).read_bytes() == payload
    with db.transaction() as conn:
        assert conn.execute("SELECT 1 FROM native_chat_quota_intents WHERE intent_kind = 'release'").fetchone() is None

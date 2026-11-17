"""Ordinary retention source revisions are not native ready manifests."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeScopeV1
from tldw_Server_API.app.core.Chat.native_fork_projection import AuthorizedNativeOwner
from tldw_Server_API.tests.DB_Management.test_native_fork_transactions import native_fork_db as native_fork_db
from tldw_Server_API.tests.DB_Management.test_native_fork_transactions import owner

pytestmark = pytest.mark.integration


def capture(db, cid, *, authorized=None):
    with db.transaction() as conn:
        return db.native_forks.read_retention_source_revision(authorized or owner(), cid, conn=conn)


def source(db):
    cid = db.add_conversation({"character_id": None, "title": "Ordinary source"})
    mid = db.add_message(
        {
            "conversation_id": cid,
            "sender": "user",
            "content": "Résumé 😀",
            "images": [{"data": b"first-image", "mime": "image/png"}],
        }
    )
    return cid, mid


def test_source_revision_is_stable_bound_and_selects_no_bytes(native_fork_db, monkeypatch):
    db = native_fork_db
    cid, _ = source(db)
    before = dict(db.get_conversation_by_id(cid))
    original = db.message_store._read_history_snapshot
    calls = []

    def observed(*args, **kwargs):
        assert not kwargs.get("selected_ids")
        result = original(*args, **kwargs)
        assert result[1] == ()
        calls.append(result[0])
        return result

    monkeypatch.setattr(db.message_store, "_read_history_snapshot", observed)
    revision = capture(db, cid)
    assert revision.startswith("sha256:") and len(revision) == 71
    assert capture(db, cid) == revision
    assert calls and revision != calls[0].source_digest
    assert (
        capture(db, cid, authorized=AuthorizedNativeOwner("alice", "different-view-owner", owner().scope)) != revision
    )
    other, _ = source(db)
    assert capture(db, other) != revision
    assert dict(db.get_conversation_by_id(cid)) == before


@pytest.mark.parametrize("mutation", ["message", "metadata", "settings", "image"])
def test_real_source_mutations_retire_original_revision(native_fork_db, mutation):
    db = native_fork_db
    cid, mid = source(db)
    original = capture(db, cid)
    if mutation == "message":
        assert db.update_message(mid, {"content": "Changed"}, 1)
    elif mutation == "metadata":
        assert db.add_message_metadata(mid, extra={"reviewed_context": "changed"})
    elif mutation == "settings":
        assert db.upsert_conversation_settings(cid, {"temperature": 0.3})
    else:
        with db.transaction() as conn:
            conn.execute("UPDATE message_images SET image_data = ? WHERE message_id = ?", (b"other-image", mid))
    assert capture(db, cid) != original


@pytest.mark.parametrize(
    "fault", ["foreign_client", "scope", "deleted", "protected", "bundle", "creation", "claim", "closed_workspace"]
)
def test_nonordinary_or_wrong_owned_source_refuses_without_writes(native_fork_db, fault, monkeypatch):
    db = native_fork_db
    cid, _ = source(db)
    authorized = owner()
    if fault == "foreign_client":
        authorized = AuthorizedNativeOwner("bob", owner().owner_key, owner().scope)
    elif fault == "scope":
        authorized = AuthorizedNativeOwner(
            "alice", owner().owner_key, NativeScopeV1(kind="workspace", workspace_id="foreign")
        )
    elif fault == "deleted":
        assert db.soft_delete_conversation(cid, db.get_conversation_by_id(cid)["version"])
    else:
        with db.transaction() as conn:
            if fault == "protected":
                conn.execute(
                    "UPDATE conversations SET required_projection_version = 'native-fork-v1' WHERE id = ?", (cid,)
                )
            elif fault == "bundle":
                conn.execute("UPDATE conversations SET native_bundle_json = '{}' WHERE id = ?", (cid,))
            elif fault == "creation":
                conn.execute("UPDATE conversations SET native_creation_operation_id = 'original' WHERE id = ?", (cid,))
            elif fault == "claim":
                conn.execute(
                    "INSERT INTO native_chat_asset_claims (client_id,claim_id,owner_key,conversation_id,asset_id,asset_revision,content_hash,size_bytes,mime_type,representation,state,created_at,updated_at) VALUES ('alice','legacy','native:alice',?,'legacy',1,?,1,'image/png','embedded_image_v1','live','2026-01-01','2026-01-01')",
                    (cid, "a" * 64),
                )
        if fault == "closed_workspace":
            db.upsert_workspace("workspace", "Owned")
            authorized = AuthorizedNativeOwner(
                "alice", owner().owner_key, NativeScopeV1(kind="workspace", workspace_id="workspace")
            )
            with db.transaction() as conn:
                conn.execute(
                    "UPDATE conversations SET scope_type = 'workspace',workspace_id = 'workspace' WHERE id = ?", (cid,)
                )

            def fail(*args, **kwargs):
                raise RuntimeError("held cascade")

            monkeypatch.setattr(db, "soft_delete_conversation", fail)
            with pytest.raises(Exception, match="workspace_delete_incomplete"):
                db.delete_workspace("workspace", expected_version=1)
    before = dict(db.execute_query("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone())
    with pytest.raises(ValueError):
        capture(db, cid, authorized=authorized)
    assert dict(db.execute_query("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()) == before


def test_capture_requires_owned_active_transaction(native_fork_db):
    with pytest.raises(ValueError, match="active_transaction_required"):
        native_fork_db.native_forks.read_retention_source_revision(
            owner(), "missing", conn=native_fork_db.get_connection()
        )


@pytest.mark.parametrize("change", ["metadata", "carrier"])
def test_source_capture_refuses_state_changed_during_snapshot(native_fork_db, monkeypatch, change):
    db = native_fork_db
    cid, mid = source(db)
    original = db.message_store._read_history_snapshot

    def changed(*args, **kwargs):
        snapshot = original(*args, **kwargs)
        if change == "metadata":
            assert db.add_message_metadata(mid, extra={"race": "changed"}, conn=kwargs["conn"])
        else:
            kwargs["conn"].execute(
                "INSERT INTO native_chat_asset_claims (client_id,claim_id,owner_key,conversation_id,asset_id,asset_revision,content_hash,size_bytes,mime_type,representation,state,created_at,updated_at) VALUES ('alice','racing','native:alice',?,'racing',1,?,1,'image/png','embedded_image_v1','live','2026-01-01','2026-01-01')",
                (cid, "a" * 64),
            )
        return snapshot

    monkeypatch.setattr(db.message_store, "_read_history_snapshot", changed)
    with pytest.raises(ValueError, match="stale_retention_source|native_retention_source_unsupported"):
        capture(db, cid)


def test_actual_history_aba_cannot_reuse_original_source_revision(native_fork_db):
    db = native_fork_db
    cid, mid = source(db)
    before = capture(db, cid)
    assert db.update_message(mid, {"content": "Changed"}, 1)
    assert db.update_message(mid, {"content": "Résumé 😀"}, 2)
    assert capture(db, cid) != before


def test_hard_delete_same_id_same_clock_recreation_retires_original_source(native_fork_db, monkeypatch):
    db = native_fork_db
    monkeypatch.setattr(db, "_get_current_utc_timestamp_iso", lambda: "2026-10-12T12:00:00.000Z")
    cid = db.add_conversation({"id": "reused-source", "character_id": None, "title": "Empty"})
    original = capture(db, cid)
    assert db.hard_delete_conversation(cid)
    assert db.add_conversation({"id": cid, "character_id": None, "title": "Empty"}) == cid
    assert capture(db, cid) != original


def test_missing_explicit_owner_key_never_manufactures_source_namespace(native_fork_db):
    cid, _ = source(native_fork_db)
    with pytest.raises(ValueError):
        capture(native_fork_db, cid, authorized=AuthorizedNativeOwner("alice", None, owner().scope))

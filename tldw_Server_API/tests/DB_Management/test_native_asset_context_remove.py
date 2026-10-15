"""Reviewed native context removal preserves bytes and shared references."""

from __future__ import annotations

import json

import pytest

from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeAssetContextUpdateV1
from tldw_Server_API.tests.DB_Management.test_native_asset_context_read import read, reference_case
from tldw_Server_API.tests.DB_Management.test_native_chat_asset_io import owner

pytest_plugins = ("tldw_Server_API.tests.DB_Management.test_native_chat_asset_io",)
pytestmark = pytest.mark.integration
TABLES = (
    "conversations",
    "native_chat_asset_references",
    "native_chat_asset_claims",
    "native_chat_asset_candidates",
    "native_chat_quota_intents",
    "native_chat_operations",
)


def snapshot(db):
    return {table: [dict(row) for row in db.execute_query("SELECT * FROM " + table).fetchall()] for table in TABLES}


def request_for(context, **changes):
    target = context.asset_manifest[0]
    return NativeAssetContextUpdateV1.model_validate(
        {
            "expected_asset_manifest_revision": context.asset_manifest_revision,
            "target_reference_id": target.reference_id,
            "expected_reference_revision": target.reference_revision,
            "expected_asset_revision": target.revision,
            "action": "remove_from_context",
        }
        | changes
    )


def update(db, cid, request, *, authorized=None):
    with db.transaction() as conn:
        return db.native_assets.update_native_asset_context(authorized or owner(), cid, request, conn=conn)


@pytest.mark.asyncio
async def test_removal_changes_only_selected_reference_and_atomic_context_fences(native_fork_db, tmp_path, monkeypatch):
    db = native_fork_db
    adapter, cid, claim, payload = await reference_case(db, tmp_path, monkeypatch)
    original = read(db, cid)
    # Both shared references participate before choosing only the first.
    with db.transaction() as conn:
        conn.execute("UPDATE native_chat_asset_references SET context_enabled = ?", (True,))
    original = read(db, cid)
    before = snapshot(db)
    result = update(db, cid, request_for(original))
    after = snapshot(db)
    target, peer = result.asset_manifest
    assert target.reference_id == "first" and target.context_enabled is False and target.reference_revision == "4"
    assert peer == original.asset_manifest[1] and peer.context_enabled is True
    assert result.asset_manifest_revision != original.asset_manifest_revision
    assert after["conversations"][0]["history_version"] == before["conversations"][0]["history_version"] + 1
    old_bundle = json.loads(before["conversations"][0]["native_bundle_json"])
    new_bundle = json.loads(after["conversations"][0]["native_bundle_json"])
    assert new_bundle == old_bundle | {"asset_manifest_revision": result.asset_manifest_revision}
    for table in TABLES[2:]:
        assert after[table] == before[table]
    assert read(db, cid) == result
    with adapter.read_owned_asset(owner(), cid, claim["asset_id"], "1") as (_, handle):
        assert handle.read() == payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "manifest",
        "reference_revision",
        "asset_revision",
        "target",
        "owner",
        "replace",
        "restore",
        "raw_request",
        "unchecked_request",
    ],
)
async def test_invalid_reviewed_removal_writes_nothing(native_fork_db, tmp_path, monkeypatch, fault):
    db = native_fork_db
    _, cid, claim, _ = await reference_case(db, tmp_path, monkeypatch)
    current = read(db, cid)
    changes = {}
    if fault == "manifest":
        changes["expected_asset_manifest_revision"] = "different"
    elif fault == "reference_revision":
        changes["expected_reference_revision"] = "2"
    elif fault == "asset_revision":
        changes["expected_asset_revision"] = "2"
    elif fault == "target":
        changes["target_reference_id"] = "foreign"
    elif fault in {"replace", "restore"}:
        changes.update(action=fault, replacement={"asset_id": claim["asset_id"], "revision": "1"})
        if fault == "restore":
            changes["expected_original_hash"] = claim["content_hash"]
    reviewed = request_for(current, **changes)
    if fault == "raw_request":
        reviewed = reviewed.model_dump(mode="json")
    elif fault == "unchecked_request":
        reviewed = NativeAssetContextUpdateV1.model_construct(
            **(reviewed.model_dump(mode="python") | {"expected_reference_revision": 3})
        )
    before = snapshot(db)
    from tldw_Server_API.app.core.Chat.native_fork_projection import AuthorizedNativeOwner

    authorized = AuthorizedNativeOwner("1", "native:foreign", owner().scope) if fault == "owner" else owner()
    with pytest.raises(ValueError):
        update(db, cid, reviewed, authorized=authorized)
    assert snapshot(db) == before


@pytest.mark.asyncio
async def test_lost_result_reloads_original_owner_context_without_replaying_stale_request(
    native_fork_db, tmp_path, monkeypatch
):
    db = native_fork_db
    _, cid, _, _ = await reference_case(db, tmp_path, monkeypatch)
    reviewed = request_for(read(db, cid))
    saved = update(db, cid, reviewed)
    assert read(db, cid) == saved
    before = snapshot(db)
    with pytest.raises(ValueError, match="stale_native_asset_context"):
        update(db, cid, reviewed)
    assert snapshot(db) == before
    # Current already-excluded reference is a no-op only with the exact current review.
    assert update(db, cid, request_for(saved)) == saved
    assert snapshot(db) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["reference", "bundle", "history"])
async def test_failed_mutation_tail_rolls_back_even_when_caller_catches_error(
    native_fork_db, tmp_path, monkeypatch, boundary
):
    db = native_fork_db
    _, cid, _, _ = await reference_case(db, tmp_path, monkeypatch)
    reviewed = request_for(read(db, cid))
    before = snapshot(db)

    def fail(*args, **kwargs):
        raise RuntimeError("history fence interrupted")

    if boundary == "reference":
        monkeypatch.setattr(db, "_generate_uuid", fail)
    elif boundary == "bundle":
        monkeypatch.setattr(db.message_store, "_advance_history_version", fail)
    else:
        original_reader = db.native_assets.read_native_asset_context
        calls = 0

        def fail_final_read(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                fail()
            return original_reader(*args, **kwargs)

        monkeypatch.setattr(db.native_assets, "read_native_asset_context", fail_final_read)
    with db.transaction() as conn:
        caller_id = db.add_conversation(
            {"id": "caller-work", "character_id": None, "title": "Preserved caller work"}, conn=conn
        )
        with pytest.raises(RuntimeError, match="history fence interrupted"):
            db.native_assets.update_native_asset_context(owner(), cid, reviewed, conn=conn)
    assert db.get_conversation_by_id(caller_id)["title"] == "Preserved caller work"
    after = snapshot(db)
    after["conversations"] = [row for row in after["conversations"] if row["id"] != caller_id]
    assert after == before


@pytest.mark.asyncio
async def test_caller_rollback_reverts_successful_local_removal(native_fork_db, tmp_path, monkeypatch):
    db = native_fork_db
    _, cid, _, _ = await reference_case(db, tmp_path, monkeypatch)
    reviewed = request_for(read(db, cid))
    before = snapshot(db)
    with pytest.raises(RuntimeError, match="caller failed"):
        with db.transaction() as conn:
            db.native_assets.update_native_asset_context(owner(), cid, reviewed, conn=conn)
            raise RuntimeError("caller failed")
    assert snapshot(db) == before


def test_removal_requires_original_active_transaction(native_fork_db):
    reviewed = NativeAssetContextUpdateV1(
        expected_asset_manifest_revision="1",
        target_reference_id="first",
        expected_reference_revision="1",
        expected_asset_revision="1",
        action="remove_from_context",
    )
    with pytest.raises(ValueError, match="active_transaction_required"):
        native_fork_db.native_assets.update_native_asset_context(
            owner(), "missing", reviewed, conn=native_fork_db.get_connection()
        )


@pytest.mark.asyncio
async def test_removal_after_actual_restore_preserves_gone_creation_key(native_fork_db, tmp_path, monkeypatch):
    db = native_fork_db
    _, cid, _, _ = await reference_case(db, tmp_path, monkeypatch)
    version = db.get_conversation_by_id(cid)["version"]
    assert db.soft_delete_conversation(cid, version)
    assert db.restore_conversation(cid, version + 1)
    receipt = dict(db.execute_query("SELECT * FROM native_chat_operations").fetchone())
    assert receipt["state"] == "gone" and receipt["result_json"] is None
    result = update(db, cid, request_for(read(db, cid)))
    assert result.asset_manifest[0].context_enabled is False
    assert dict(db.execute_query("SELECT * FROM native_chat_operations").fetchone()) == receipt

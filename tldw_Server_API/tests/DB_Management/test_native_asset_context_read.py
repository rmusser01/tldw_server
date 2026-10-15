"""Private canonical native context reads refuse ambiguous owned metadata."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.schemas import native_fork_schemas as schemas
from tldw_Server_API.tests.DB_Management.test_native_chat_asset_io import owner
from tldw_Server_API.tests.DB_Management.test_native_chat_asset_read import committed

pytest_plugins = ("tldw_Server_API.tests.DB_Management.test_native_chat_asset_io",)
pytestmark = pytest.mark.integration


def read(db, cid, *, authorized=None):
    with db.transaction() as conn:
        return db.native_assets.read_native_asset_context(authorized or owner(), cid, conn=conn)


async def reference_case(db, tmp_path, monkeypatch):
    """Use real prepared/adopted bytes; seed metadata only, not rich publication."""
    adapter, root, cid, claim, payload = await committed(db, tmp_path, monkeypatch)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET native_bundle_json = ? WHERE id = ?",
            (
                json.dumps(
                    {
                        "version": 1,
                        "state": "ready",
                        "required_projection_version": "native-fork-v1",
                        "settings_revision": 1,
                        "asset_manifest_revision": "original-opaque-😀",
                    }
                ),
                cid,
            ),
        )
        for reference, position, enabled in (("second", 9, False), ("first", 2, True)):
            conn.execute(
                "INSERT INTO native_chat_asset_references "
                "(client_id,reference_id,owner_key,conversation_id,claim_id,position,revision,context_enabled,"
                "created_at,updated_at,role) VALUES ('1',?,'native:1',?,?,?,3,?,'2026-01-01','2026-01-01','document_context')",
                (reference, cid, claim["claim_id"], position, enabled),
            )
    return adapter, cid, claim, payload


@pytest.mark.asyncio
async def test_ordered_shared_reference_context_preserves_original_revision_and_no_writes(
    native_fork_db,
    tmp_path,
    monkeypatch,
):
    db = native_fork_db
    _, cid, claim, _ = await reference_case(db, tmp_path, monkeypatch)
    before = [dict(row) for row in db.execute_query("SELECT * FROM native_chat_asset_references").fetchall()]
    monkeypatch.setattr(db.native_forks, "_row", lambda *a, **k: pytest.fail("operation lock after conversation"))
    monkeypatch.setattr(db.native_assets, "read_committed_asset", lambda *a, **k: pytest.fail("byte reader called"))
    context = read(db, cid)
    assert context.asset_manifest_revision == "original-opaque-😀"
    assert [
        (e.reference_id, e.reference_revision, e.asset_id, e.revision, e.role, e.context_enabled)
        for e in context.asset_manifest
    ] == [
        ("first", "3", claim["asset_id"], "1", "document_context", True),
        ("second", "3", claim["asset_id"], "1", "document_context", False),
    ]
    assert [dict(row) for row in db.execute_query("SELECT * FROM native_chat_asset_references").fetchall()] == before
    with pytest.raises(ValueError):
        context.asset_manifest_revision = "new"
    assert schemas.NativeAssetContextV1.model_validate_json(context.model_dump_json()) == context


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "foreign_owner",
        "wrong_scope",
        "deleted",
        "ordinary",
        "missing_bundle",
        "malformed_bundle",
        "preparing_bundle",
        "boolean_bundle_version",
        "invalid_manifest_revision",
        "null_role",
        "wrong_role",
        "duplicate_position",
        "foreign_reference",
        "missing_claim",
        "foreign_claim",
        "released_claim",
        "candidate_generation",
        "candidate_prepared",
        "candidate_hash",
        "quota_pending",
        "release_started",
        "operation_pending",
        "operation_wrong_projection",
        "operation_wrong_target",
        "retention_kind",
        "ambiguous_asset",
        "fractional_asset_revision",
    ],
)
async def test_invalid_context_refuses_without_mutating_records(native_fork_db, tmp_path, monkeypatch, fault):
    db = native_fork_db
    _, cid, claim, _ = await reference_case(db, tmp_path, monkeypatch)
    authorized = owner()
    with db.transaction() as conn:
        if fault in {"foreign_owner", "wrong_scope"}:
            authorized = SimpleNamespace(
                client_id="1",
                owner_key="native:foreign" if fault == "foreign_owner" else owner().owner_key,
                scope=(
                    schemas.NativeScopeV1(kind="workspace", workspace_id="foreign")
                    if fault == "wrong_scope"
                    else owner().scope
                ),
            )
        elif fault in {"deleted", "ordinary"}:
            conn.execute("UPDATE conversations SET deleted = ? WHERE id = ?", (fault == "deleted", cid))
            if fault == "ordinary":
                conn.execute("UPDATE conversations SET required_projection_version = NULL WHERE id = ?", (cid,))
        elif "bundle" in fault or fault == "invalid_manifest_revision":
            value = {
                "version": 1,
                "state": "ready",
                "required_projection_version": "native-fork-v1",
                "settings_revision": 1,
                "asset_manifest_revision": "original-opaque-😀",
            }
            if fault == "preparing_bundle":
                value["state"] = "preparing"
            elif fault == "boolean_bundle_version":
                value["version"] = True
            elif fault == "invalid_manifest_revision":
                value["asset_manifest_revision"] = 1
            raw = (
                None if fault == "missing_bundle" else "not-json" if fault == "malformed_bundle" else json.dumps(value)
            )
            conn.execute("UPDATE conversations SET native_bundle_json = ? WHERE id = ?", (raw, cid))
        elif fault in {"null_role", "wrong_role", "duplicate_position", "foreign_reference"}:
            if fault == "null_role":
                conn.execute("UPDATE native_chat_asset_references SET role = NULL")
            elif fault == "wrong_role":
                conn.execute("UPDATE native_chat_asset_references SET role = 'message_image'")
            elif fault == "duplicate_position":
                conn.execute("UPDATE native_chat_asset_references SET position = 2")
            else:
                conn.execute("UPDATE native_chat_asset_references SET owner_key = 'native:foreign'")
        elif fault in {"missing_claim", "foreign_claim", "released_claim"}:
            if fault == "missing_claim":
                conn.execute("UPDATE native_chat_asset_claims SET conversation_id = 'foreign'")
            elif fault == "foreign_claim":
                conn.execute("UPDATE native_chat_asset_claims SET owner_key = 'native:foreign'")
            else:
                conn.execute("UPDATE native_chat_asset_claims SET state = 'released'")
        elif fault.startswith("candidate_"):
            if fault == "candidate_generation":
                conn.execute("UPDATE native_chat_asset_candidates SET attempt_generation = attempt_generation + 1")
            elif fault == "candidate_prepared":
                conn.execute("UPDATE native_chat_asset_candidates SET state = 'prepared'")
            else:
                conn.execute("UPDATE native_chat_asset_candidates SET expected_hash = ?", ("a" * 64,))
        elif fault == "quota_pending":
            conn.execute("UPDATE native_chat_quota_intents SET state = 'pending'")
        elif fault == "release_started":
            conn.execute(
                "INSERT INTO native_chat_quota_intents (client_id,candidate_id,intent_kind,owner_key,size_bytes,state,created_at,updated_at) SELECT client_id,candidate_id,'release',owner_key,size_bytes,'pending',created_at,updated_at FROM native_chat_quota_intents"
            )
        elif fault == "operation_pending":
            conn.execute("UPDATE native_chat_operations SET state = 'preparing'")
        elif fault == "operation_wrong_projection":
            conn.execute("UPDATE native_chat_operations SET projection_version = 'unsupported'")
        elif fault == "operation_wrong_target":
            conn.execute("UPDATE native_chat_operations SET child_conversation_id = 'foreign'")
        elif fault == "retention_kind":
            conn.execute("UPDATE conversations SET native_creation_operation_kind = 'native_asset_retention_v1'")
        elif fault == "ambiguous_asset":
            conn.execute(
                "INSERT INTO native_chat_asset_claims "
                "(client_id,claim_id,owner_key,conversation_id,candidate_id,asset_id,asset_revision,content_hash,"
                "size_bytes,mime_type,representation,state,created_at,updated_at) "
                "SELECT client_id,'alias',owner_key,conversation_id,candidate_id,asset_id,asset_revision,content_hash,"
                "size_bytes,mime_type,representation,state,created_at,updated_at FROM native_chat_asset_claims"
            )
        elif fault == "fractional_asset_revision":
            if db.backend_type.value != "sqlite":
                pytest.skip("PostgreSQL INTEGER storage already refuses fractional revisions")
            conn.execute("UPDATE native_chat_asset_claims SET asset_revision = 1.5")
    before = {
        table: [dict(row) for row in db.execute_query("SELECT * FROM " + table).fetchall()]
        for table in (
            "conversations",
            "native_chat_asset_references",
            "native_chat_asset_claims",
            "native_chat_asset_candidates",
            "native_chat_operations",
            "native_chat_quota_intents",
        )
    }
    with pytest.raises(ValueError):
        read(db, cid, authorized=authorized)
    after = {table: [dict(row) for row in db.execute_query("SELECT * FROM " + table).fetchall()] for table in before}
    assert after == before


def test_context_requires_actual_owned_transaction(native_fork_db):
    with pytest.raises(ValueError, match="active_transaction_required"):
        native_fork_db.native_assets.read_native_asset_context(owner(), "missing", conn=native_fork_db.get_connection())


@pytest.mark.asyncio
async def test_workspace_context_denies_after_actual_closure_without_mutation(native_fork_db, tmp_path, monkeypatch):
    db = native_fork_db
    _, cid, _, _ = await reference_case(db, tmp_path, monkeypatch)
    db.upsert_workspace("workspace", "Owned workspace")
    authorized = SimpleNamespace(
        client_id="1",
        owner_key=owner().owner_key,
        scope=schemas.NativeScopeV1(kind="workspace", workspace_id="workspace"),
    )
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET scope_type = 'workspace',workspace_id = 'workspace' WHERE id = ?", (cid,)
        )
        conn.execute("UPDATE native_chat_operations SET scope_type = 'workspace',workspace_id = 'workspace'")
    assert len(read(db, cid, authorized=authorized).asset_manifest) == 2
    monkeypatch.setattr(
        db, "soft_delete_conversation", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("held cascade"))
    )
    with pytest.raises(Exception, match="workspace_delete_incomplete"):
        db.delete_workspace("workspace", expected_version=1)
    assert db.get_workspace("workspace")["native_chat_admission_closed"]
    before = [dict(row) for row in db.execute_query("SELECT * FROM native_chat_asset_references").fetchall()]
    with pytest.raises(ValueError, match="workspace_native_unavailable"):
        read(db, cid, authorized=authorized)
    assert [dict(row) for row in db.execute_query("SELECT * FROM native_chat_asset_references").fetchall()] == before


@pytest.mark.unit
@pytest.mark.parametrize("fault", ["duplicate_reference", "alias_hash", "numeric_revision", "unknown_field"])
def test_strict_context_model_refuses_ambiguous_wire_identity(fault):
    entry = {
        "reference_id": "first",
        "reference_revision": "1",
        "asset_id": "asset",
        "revision": "1",
        "representation": "embedded_image_v1",
        "role": "message_image",
        "context_enabled": True,
        "hash": "a" * 64,
        "fidelity_disposition": "retained",
    }
    other = entry | {"reference_id": "second", "role": "reference_image", "context_enabled": False}
    body = {"asset_manifest_revision": "opaque-😀", "asset_manifest": [entry, other]}
    if fault == "duplicate_reference":
        other["reference_id"] = "first"
    elif fault == "alias_hash":
        other["hash"] = "b" * 64
    elif fault == "numeric_revision":
        body["asset_manifest_revision"] = 1
    else:
        body["ready"] = True
    with pytest.raises(ValueError):
        schemas.NativeAssetContextV1.model_validate(body)


@pytest.mark.unit
def test_context_model_preserves_shared_bytes_with_distinct_roles():
    entry = {
        "reference_id": "first",
        "reference_revision": "1",
        "asset_id": "asset",
        "revision": "1",
        "representation": "embedded_image_v1",
        "role": "message_image",
        "context_enabled": True,
        "hash": "a" * 64,
        "fidelity_disposition": "retained",
    }
    context = schemas.NativeAssetContextV1.model_validate(
        {
            "asset_manifest_revision": "opaque-😀",
            "asset_manifest": [
                entry,
                entry | {"reference_id": "second", "role": "reference_image", "context_enabled": False},
            ],
        }
    )
    assert [(item.role, item.context_enabled) for item in context.asset_manifest] == [
        ("message_image", True),
        ("reference_image", False),
    ]
    assert schemas.NativeAssetContextV1.model_validate_json(context.model_dump_json()) == context


@pytest.mark.asyncio
async def test_actual_restore_preserves_shared_context_metadata_and_burned_key(native_fork_db, tmp_path, monkeypatch):
    """Restoration qualifies metadata, not the separately narrower byte reader."""
    db = native_fork_db
    _, cid, _, _ = await reference_case(db, tmp_path, monkeypatch)
    original = read(db, cid)
    claims = [dict(row) for row in db.execute_query("SELECT * FROM native_chat_asset_claims").fetchall()]
    version = db.get_conversation_by_id(cid)["version"]
    assert db.soft_delete_conversation(cid, version)
    with pytest.raises(ValueError):
        read(db, cid)
    assert db.restore_conversation(cid, version + 1)
    operation = dict(db.execute_query("SELECT * FROM native_chat_operations").fetchone())
    assert operation["state"] == "gone" and operation["result_json"] is None
    assert read(db, cid) == original
    assert [dict(row) for row in db.execute_query("SELECT * FROM native_chat_asset_claims").fetchall()] == claims
    assert dict(db.execute_query("SELECT * FROM native_chat_operations").fetchone()) == operation

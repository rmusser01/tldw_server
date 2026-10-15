"""Seeded retention receipt identity; no retention publisher, claims or byte fixture."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import (
    NativeAssetRetentionCommittedV1,
    NativeAssetRetentionRequestV1,
    NativeScopeV1,
)
from tldw_Server_API.app.core.Chat import native_fork_projection as projection
from tldw_Server_API.app.core.DB_Management.chacha.native_fork_store import NativeForkStoreError
from tldw_Server_API.tests.DB_Management.test_native_fork_transactions import native_fork_db as native_fork_db
from tldw_Server_API.tests.DB_Management.test_native_fork_transactions import owner, workspace_owner

OPERATION = "11111111-1111-4111-8111-111111111111"
KIND = "native_asset_retention_v1"


def payload(source: str = "source", *, scope: Any = None) -> dict[str, Any]:
    """A frozen two-reference input, compatible with one shared retained asset."""
    first = {
        "reference_id": "ref-a",
        "reference_revision": "ref-a-original",
        "asset_id": "source-asset",
        "revision": "source-revision",
        "representation": "embedded_image_v1",
        "role": "message_image",
        "context_enabled": True,
        "hash": "0" * 64,
        "fidelity_disposition": "retained",
    }
    second = {
        **first,
        "reference_id": "ref-b",
        "reference_revision": "ref-b-original",
        "role": "reference_image",
        "context_enabled": False,
    }
    return {
        "operation_kind": KIND,
        "operation_id": OPERATION,
        "owner_key": owner().owner_key,
        "scope": (scope or NativeScopeV1(kind="global")).model_dump(mode="json"),
        "source_conversation_id": source,
        "expected_asset_manifest_revision": "opaque-original",
        "asset_manifest": [first, second],
        "request_digest": "0" * 64,
    }


def expected_digest(data: dict[str, Any]) -> str:
    """Independent fixed-tuple fixture encoding, excluding its supplied hash."""
    fields = (
        "reference_id",
        "reference_revision",
        "asset_id",
        "revision",
        "representation",
        "role",
        "context_enabled",
        "hash",
        "fidelity_disposition",
    )
    value = [
        "tldw-native-asset-retention",
        1,
        data["operation_id"],
        data["owner_key"],
        [data["scope"]["kind"], data["scope"].get("workspace_id")],
        data["source_conversation_id"],
        data["expected_asset_manifest_revision"],
        [[asset[field] for field in fields] for asset in data["asset_manifest"]],
    ]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def request(data: dict[str, Any]) -> NativeAssetRetentionRequestV1:
    """Validate exact synthetic input, then install its independently computed digest."""
    return NativeAssetRetentionRequestV1.model_validate({**data, "request_digest": expected_digest(data)})


def digest(value: NativeAssetRetentionRequestV1) -> str:
    return projection.native_asset_retention_request_digest(value)


def seeded(db, *, workspace: bool = False, alias_change: str | None = None):
    """Seed a typed stored result through real DB transactions, not an actual publication."""
    authorized = workspace_owner() if workspace else owner()
    if workspace:
        db.upsert_workspace("workspace-one", "Workspace One")
    source = db.add_conversation(
        {
            "title": "Receipt source",
            "character_id": None,
            "scope_type": authorized.scope.kind,
            "workspace_id": authorized.scope.workspace_id,
        }
    )
    data = payload(source, scope=authorized.scope)
    if alias_change == "hash":
        data["asset_manifest"][1]["hash"] = "1" * 64
    elif alias_change == "representation":
        data["asset_manifest"][1]["representation"] = "generated_image_revision_v1"
    elif alias_change == "fidelity":
        data["asset_manifest"][1].update(representation="unavailable_v1", hash=None, fidelity_disposition="unavailable")
    original = request(data)
    result = NativeAssetRetentionCommittedV1(
        operation_id=OPERATION,
        owner_key=authorized.owner_key,
        state="committed",
        asset_manifest_revision="opaque-committed",
        asset_map={key: {"asset_id": "retained-shared", "revision": "opaque-retained"} for key in ("ref-a", "ref-b")},
    )
    with db.transaction() as conn:
        db.native_forks.reserve_operation(
            authorized, KIND, OPERATION, original.request_digest, original.model_dump(mode="json"), conn=conn
        )
        conn.execute("UPDATE conversations SET required_projection_version = 'native-fork-v1' WHERE id = ?", (source,))
        conn.execute(
            "UPDATE native_chat_operations SET state = 'committed', source_conversation_id = ?, "
            "child_conversation_id = ?, result_json = ? WHERE client_id = ? AND operation_kind = ? AND operation_id = ?",
            (source, source, result.model_dump_json(), authorized.client_id, KIND, OPERATION),
        )
    return authorized, original, result


def resolve(db, authorized, original):
    with db.transaction() as conn:
        return db.native_forks.resolve_operation(authorized, KIND, OPERATION, original.request_digest, conn=conn)


@pytest.mark.unit
@pytest.mark.parametrize(
    "field",
    [
        "operation_id",
        "owner_key",
        "scope",
        "source_conversation_id",
        "expected_asset_manifest_revision",
        "reference_id",
        "reference_revision",
        "asset_id",
        "revision",
        "representation",
        "role",
        "context_enabled",
        "hash",
    ],
)
def test_retention_digest_binds_each_original_semantic_field(field):
    data = payload()
    original = request(data)
    changed = deepcopy(data)
    if field == "operation_id":
        changed[field] = "22222222-2222-4222-8222-222222222222"
    elif field == "scope":
        changed[field] = {"kind": "workspace", "workspace_id": "workspace-other"}
    elif field in changed:
        changed[field] += "-changed"
    elif field == "representation":
        changed["asset_manifest"][1][field] = "generated_image_revision_v1"
    elif field == "role":
        changed["asset_manifest"][0][field] = "reference_image"
    elif field == "context_enabled":
        changed["asset_manifest"][0][field] = False
    elif field == "hash":
        changed["asset_manifest"][0][field] = "1" * 64
    else:
        changed["asset_manifest"][0][field] += "-changed"
    assert digest(original) != digest(request(changed))


@pytest.mark.unit
def test_retention_digest_preserves_reference_order_unicode_and_ignores_supplied_digest():
    original = request(payload("é"))
    reordered = original.model_dump(mode="json")
    reordered["asset_manifest"].reverse()
    assert digest(original) == expected_digest(original.model_dump(mode="json"))
    assert digest(original) != digest(request(reordered))
    assert digest(original) != digest(request(payload("e\u0301")))
    assert digest(original) == digest(original.model_copy(update={"request_digest": "f" * 64}))


@pytest.mark.unit
@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=20))
def test_retention_digest_exact_unicode_survives_wire_roundtrip(source):
    original = request(payload(source))
    assert digest(original) == digest(NativeAssetRetentionRequestV1.model_validate_json(original.model_dump_json()))


@pytest.mark.unit
def test_retention_request_rejects_duplicate_reference_identity():
    data = payload()
    data["asset_manifest"][1]["reference_id"] = "ref-a"
    with pytest.raises(ValueError, match="duplicate_asset_reference"):
        request(data)


@pytest.mark.integration
@pytest.mark.parametrize("workspace", [False, True])
def test_original_typed_retention_receipt_resolves_with_compatible_shared_asset(native_fork_db, workspace):
    db = native_fork_db
    authorized, original, expected = seeded(db, workspace=workspace)
    result = resolve(db, authorized, original)
    assert isinstance(result.receipt, NativeAssetRetentionCommittedV1)
    assert result.receipt == expected
    assert result.child_conversation_id == original.source_conversation_id
    with pytest.raises(TypeError):
        result.receipt.asset_map["ref-a"] = result.receipt.asset_map["ref-b"]
    with pytest.raises(ValueError):
        result.receipt.asset_map["ref-a"].revision = "changed"
    assert resolve(db, authorized, original) == result


@pytest.mark.integration
@pytest.mark.parametrize(
    "corruption",
    [
        "missing_result",
        "bad_json",
        "wrong_kind",
        "receipt_owner",
        "receipt_operation",
        "projection",
        "missing_ref",
        "extra_ref",
        "canonical_digest",
        "canonical_owner",
        "canonical_operation",
        "canonical_scope",
        "canonical_source",
        "stored_source",
        "duplicate_ref",
        "target_substitution",
    ],
)
def test_retention_result_refuses_corrupt_original_binding_before_lifecycle_mutation(native_fork_db, corruption):
    db = native_fork_db
    authorized, original, expected = seeded(db)
    canonical = original.model_dump(mode="json")
    receipt = expected.model_dump(mode="json")
    with db.transaction() as conn:
        if corruption == "missing_result":
            conn.execute("UPDATE native_chat_operations SET result_json = NULL WHERE operation_id = ?", (OPERATION,))
        elif corruption == "bad_json":
            conn.execute(
                "UPDATE native_chat_operations SET result_json = 'invalid-json' WHERE operation_id = ?", (OPERATION,)
            )
        elif corruption == "stored_source":
            conn.execute(
                "UPDATE native_chat_operations SET source_conversation_id = 'other-source' WHERE operation_id = ?",
                (OPERATION,),
            )
        elif corruption == "target_substitution":
            conn.execute(
                "UPDATE native_chat_operations SET child_conversation_id = 'missing-substitute' WHERE operation_id = ?",
                (OPERATION,),
            )
        else:
            if corruption == "wrong_kind":
                receipt["operation_kind"] = "native_fork_v1"
            elif corruption == "receipt_owner":
                receipt["owner_key"] = "foreign-owner"
            elif corruption == "receipt_operation":
                receipt["operation_id"] = "22222222-2222-4222-8222-222222222222"
            elif corruption == "projection":
                receipt["projection_version"] = "unknown-projection"
            elif corruption == "missing_ref":
                receipt["asset_map"].pop("ref-a")
            elif corruption == "extra_ref":
                receipt["asset_map"]["extra-ref"] = receipt["asset_map"]["ref-a"]
            elif corruption == "canonical_digest":
                canonical["request_digest"] = "f" * 64
            elif corruption == "canonical_owner":
                canonical["owner_key"] = "foreign-owner"
            elif corruption == "canonical_operation":
                canonical["operation_id"] = "22222222-2222-4222-8222-222222222222"
            elif corruption == "canonical_scope":
                canonical["scope"] = {"kind": "workspace", "workspace_id": "other-workspace"}
            elif corruption == "canonical_source":
                canonical["source_conversation_id"] = "other-source"
            elif corruption == "duplicate_ref":
                canonical["asset_manifest"][1]["reference_id"] = "ref-a"
            conn.execute(
                "UPDATE native_chat_operations SET canonical_request_json = ?, result_json = ? WHERE operation_id = ?",
                (json.dumps(canonical), json.dumps(receipt), OPERATION),
            )
    with pytest.raises(NativeForkStoreError, match="operation_receipt_invalid"):
        resolve(db, authorized, original)
    rows = db.execute_query(
        "SELECT state, result_json FROM native_chat_operations WHERE operation_id = ?", (OPERATION,), read_only=True
    ).fetchall()
    assert rows[0]["state"] == "committed"


@pytest.mark.integration
@pytest.mark.parametrize("change", ["hash", "representation", "fidelity"])
def test_shared_output_asset_refuses_incompatible_frozen_descriptors(native_fork_db, change):
    authorized, original, _ = seeded(native_fork_db, alias_change=change)
    with pytest.raises(NativeForkStoreError, match="operation_receipt_invalid"):
        resolve(native_fork_db, authorized, original)


@pytest.mark.integration
def test_retention_resolution_keeps_owner_digest_and_deleted_target_boundaries(native_fork_db):
    db = native_fork_db
    authorized, original, _ = seeded(db)
    with pytest.raises(NativeForkStoreError, match="operation_owner_mismatch"):
        resolve(db, owner("foreign-owner"), original)
    with db.transaction() as conn:
        with pytest.raises(NativeForkStoreError, match="operation_id_conflict"):
            db.native_forks.resolve_operation(authorized, KIND, OPERATION, "f" * 64, conn=conn)
    version = db.execute_query(
        "SELECT version FROM conversations WHERE id = ?", (original.source_conversation_id,), read_only=True
    ).fetchone()["version"]
    assert db.soft_delete_conversation(original.source_conversation_id, version)
    assert resolve(db, authorized, original).state == "gone"

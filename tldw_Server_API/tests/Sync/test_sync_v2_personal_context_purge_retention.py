"""Retention boundaries for authorized Personal Context global purge."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.personal_context_deps import (
    get_personal_context_service,
)
from tldw_Server_API.app.api.v1.endpoints.personal_context import router
from tldw_Server_API.app.core.DB_Management.Personal_Context_Repository import (
    DirectPurgeCleanupIntent,
    _VerifiedDirectPurgeCleanupClaim,
)
from tldw_Server_API.app.core.DB_Management.Personalization_DB import (
    PersonalizationDB,
)
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Personalization.personal_context_crypto import (
    EnvelopeAuthenticationError,
)
from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    PersonalContextPublicationJournal,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository import (
    PersonalContextRepository,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    ProfileIntegrityError,
)
from tldw_Server_API.app.core.Personalization.personal_context_service import (
    PersonalContextService,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncConflictCreate,
    SyncDatasetCreate,
    SyncEnvelopeCreate,
    SyncKeyRecordCreate,
)
from tldw_Server_API.app.core.Sync.v2.personal_context_ongoing_contract import (
    PersonalContextAuthorityMetadata,
)
from tldw_Server_API.app.core.Sync.v2.personal_context_relay import (
    PersonalContextRelay,
)
from tldw_Server_API.app.core.Sync.v2.store import SyncStoreError, SyncV2Store
from tldw_Server_API.tests.Sync.test_sync_v2_personal_context_authority_identity import (
    AuthorityHarness,
)

pytestmark = pytest.mark.unit


_AUTHORITY_CANARY = "authority-retention-canary-8e531c31"
_INGRESS_CANARY = "ingress-retention-canary-c94fe170"
_CONFLICT_CANARY = "conflict-retention-canary-771c341a"
_KEY_CANARY = "key-retention-canary-d158e4bb"
_ORPHAN_CANARY = "orphan-retention-canary-e2e81718"
_RECEIPT_CANARY = "receipt-retention-canary-8251be46"
_CANONICAL_RECEIPT_CANARY = "canonical-receipt-retention-canary-b819d05a"
_FREELIST_CANARY = "freelist-retention-canary-1d9a967c"
_SECOND_DATASET_CANARY = "second-dataset-retention-canary-f3f319ad"
_SOURCE_CANARY = "source-retention-canary-70eaf67d"


def _matches_canary(value: str | None, canary: str) -> bool:
    """Compare secret test markers without exposing them in assertion output."""

    if value is None:
        return False
    return hashlib.sha256(value.encode()).digest() == hashlib.sha256(
        canary.encode()
    ).digest()


def _drain(runtime: AuthorityHarness) -> None:
    relay = PersonalContextRelay(
        publications=runtime.publications,
        stage_authority=runtime.service.stage_personal_context_authority,
        finalize_authority=runtime.service.finalize_personal_context_authority,
        cancel_authority=runtime.service.cancel_personal_context_authority,
    )
    for _attempt in range(20):
        result = relay.relay_profile(
            user_id="user-a",
            profile_id=runtime.manifest.profile_id,
            dataset_id="dataset-a",
            after_server_cursor=None,
            wall_time_ms=5_000,
        )
        if result.continuation == "complete":
            return
    raise AssertionError("authority relay did not drain")


def _intent_rows(database: PersonalizationDB) -> list[sqlite3.Row]:
    with database.transaction() as connection:
        return connection.execute(
            "SELECT * FROM personal_context_purge_cleanup_intents ORDER BY created_at"
        ).fetchall()


def _sqlite_artifact_bytes(*database_paths: Path) -> bytes:
    """Read active application-owned SQLite artifacts for opaque canary checks."""

    return b"".join(
        artifact.read_bytes()
        for database_path in database_paths
        for artifact in (
            database_path,
            Path(f"{database_path}-wal"),
            Path(f"{database_path}-shm"),
        )
        if artifact.exists()
    )


def _sqlite_artifacts_contain(database_path: Path, canary: str) -> bool:
    """Check opaque marker presence without exposing its value on assertion failure."""

    return canary.encode() in _sqlite_artifact_bytes(database_path)


def _install_cleanup_callback(runtime: AuthorityHarness) -> None:
    def cleanup(intent: DirectPurgeCleanupIntent) -> None:
        claim = runtime.canonical._repository.verify_direct_purge_cleanup_claim(
            intent,
            user_id="user-a",
            dataset_id="dataset-a",
            store=runtime.store,
            database=runtime.store.db,
        )
        runtime.service.shred_authorized_personal_context_history(claim)

    runtime.canonical.set_after_commit_purge_cleanup(cleanup)


def _execute_cleanup_layer(
    runtime: AuthorityHarness,
    layer: str,
    claim: object,
) -> object:
    """Invoke one destructive boundary directly for authorization negatives."""

    if layer == "service":
        return runtime.service.shred_authorized_personal_context_history(claim)
    if layer == "store":
        return runtime.store._shred_authorized_personal_context_history(claim)
    if layer == "database":
        return runtime.store.db._shred_authorized_personal_context_profile_history(
            claim
        )
    raise AssertionError("unknown cleanup layer")


def _issue_cleanup_capability(
    runtime: AuthorityHarness,
    *,
    owner_token: str = "verified-owner",
) -> tuple[object, DirectPurgeCleanupIntent, object]:
    """Create one direct purge and issue its target-bound cleanup capability."""

    runtime.canonical.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )
    repository = runtime.canonical._repository
    intent = repository.claim_direct_purge_cleanup(owner_token=owner_token)
    assert intent is not None
    capability = repository.verify_direct_purge_cleanup_claim(
        intent,
        user_id="user-a",
        dataset_id="dataset-a",
        store=runtime.store,
        database=runtime.store.db,
    )
    return repository, intent, capability


def _enroll_matching_dataset(runtime: AuthorityHarness, dataset_id: str) -> None:
    """Enroll another dataset for the same user and Personal Context profile."""

    original = runtime.store.get_dataset("dataset-a")
    runtime.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=dataset_id,
            owner_user_id="user-a",
            encryption_policy="server_trusted_v1",
            domains=list(original.domains),
            metadata={
                "personal_context": {
                    **original.metadata["personal_context"],
                    "profile_id": runtime.manifest.profile_id,
                    "purge_generation": 0,
                }
            },
        )
    )


def _stage_retention_canaries(runtime: AuthorityHarness) -> tuple[int, int, sqlite3.Row]:
    """Install one old authority, ingress, conflict, and recovery-key package."""

    scope_id = runtime.canonical.list_scopes()[0].scope_id
    runtime.canonical.create_manual_record(
        scope_id=scope_id,
        payload={
            "kind": "preference",
            "subject": "purge.retention",
            "polarity": "like",
            "value": _SOURCE_CANARY,
        },
        semantic_key={"namespace": "preference", "subject": "purge.retention"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    keys = runtime.canonical._repository.key_material_for_test(runtime.manifest.profile_id)
    with runtime.personal_db.transaction() as connection:
        source_before = connection.execute(
            """SELECT rows.* FROM personal_context_publication_rows rows
               JOIN personal_context_publication_batches batches
                 ON batches.profile_id = rows.profile_id
                AND batches.profile_publication_sequence = rows.profile_publication_sequence
               WHERE rows.profile_id = ? AND rows.role = 'semantic'
                 AND batches.purge_generation = 0
               ORDER BY rows.profile_publication_sequence DESC LIMIT 1""",
            (runtime.manifest.profile_id,),
        ).fetchone()
        source_manifest_row = connection.execute(
            """SELECT * FROM personal_context_publication_rows
               WHERE profile_id = ? AND profile_publication_sequence = ?
                 AND publication_batch_id = ? AND role = 'manifest'""",
            (
                runtime.manifest.profile_id,
                source_before["profile_publication_sequence"],
                source_before["publication_batch_id"],
            ),
        ).fetchone()
    assert source_before is not None
    assert source_manifest_row is not None
    assert PersonalContextPublicationJournal(keys).decrypt_row(source_before)[0] == "personal_context.record"
    with runtime.personal_db.transaction() as connection:
        connection.execute(
            """
            INSERT INTO personal_context_ingress_receipts(
                dataset_id, device_id, client_envelope_id,
                canonical_payload_digest, purge_generation, wire_entity_version,
                resulting_object_id, resulting_version_id,
                resulting_manifest_revision, resulting_manifest_version_id,
                publication_batch_id, profile_publication_sequence,
                receipt_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "dataset-a",
                "device-a",
                "canonical-old-ingress",
                _CANONICAL_RECEIPT_CANARY,
                0,
                "canonical-old-wire-version",
                str(source_before["opaque_object_id"]),
                str(source_before["opaque_version_id"]),
                1,
                str(source_manifest_row["opaque_version_id"]),
                str(source_before["publication_batch_id"]),
                int(source_before["profile_publication_sequence"]),
                "canonical-old-receipt",
                "2026-09-04T00:00:00Z",
            ),
        )

    _drain(runtime)
    authority = runtime.store.list_envelopes_for_entity(
        "dataset-a",
        "personal_context.record",
        entity_id="record-1",
        limit=10,
    )
    if not authority:
        authority = runtime.store.list_envelopes_after(
            "dataset-a",
            0,
            limit=100,
            domains=["personal_context.record"],
        )
    authority_cursor = authority[-1].server_sequence

    dataset = runtime.store.get_dataset("dataset-a")
    ingress = SyncEnvelopeCreate(
        dataset_id="dataset-a",
        client_envelope_id="old-client-ingress",
        device_id="device-a",
        domain="personal_context.record",
        operation="upsert",
        object_id="old-ingress-object",
        object_revision=1,
        schema_version=1,
        adapter_version=1,
        payload={"opaque": "placeholder"},
        payload_hash="hmac-sha256-v1:" + "0" * 64,
        payload_size_bytes=24,
        entity_version="old-ingress-version",
        encryption_metadata={"policy": "server_trusted_v1"},
        routing_metadata={
            "profile_id": runtime.manifest.profile_id,
            "integrity_key_id": dataset.metadata["personal_context"]["integrity_key_id"],
            "purge_generation": 0,
            "personal_context_authority": PersonalContextAuthorityMetadata(
                role="client_ingress"
            ).model_dump(mode="json"),
        },
    )
    ingress = runtime.service._protect_personal_context_for_storage(dataset, ingress)
    stored_ingress = runtime.store.insert_envelope(ingress)
    orphan = runtime.store.insert_envelope(
        replace(
            ingress,
            client_envelope_id="old-orphan",
            object_id="old-orphan-object",
            entity_version="old-orphan-version",
        )
    )

    runtime.store.insert_conflict(
        SyncConflictCreate(
            conflict_id="old-personal-context-conflict",
            dataset_id="dataset-a",
            domain="personal_context.record",
            entity_id="old-ingress-object",
            conflict_type="personal_context_base_conflict",
            local_envelope_id="old-client-ingress",
            server_sequence=stored_ingress.server_sequence,
            metadata={"protected_candidate": _CONFLICT_CANARY},
        )
    )
    runtime.store.store_key_record(
        SyncKeyRecordCreate(
            key_record_id="old-personal-context-key",
            dataset_id="dataset-a",
            user_id="user-a",
            device_id="device-a",
            key_purpose="personal_context_integrity",
            wrapped_key_blob=_KEY_CANARY,
            kdf_metadata={"algorithm": "test"},
            encryption_policy="server_trusted_v1",
            key_epoch=1,
        )
    )
    runtime.store.store_key_record(
        SyncKeyRecordCreate(
            key_record_id="rotated-personal-context-key",
            dataset_id="dataset-a",
            user_id="user-a",
            device_id="device-a",
            key_purpose="personal_context_integrity",
            wrapped_key_blob=_KEY_CANARY + "-rotated",
            kdf_metadata={"algorithm": "test"},
            encryption_policy="server_trusted_v1",
            key_epoch=2,
            rotation_of_key_record_id="old-personal-context-key",
            rotation_source_key_record_ids=("old-personal-context-key",),
        )
    )
    with runtime.store.db.backend.transaction() as connection:
        for cursor, canary in (
            (authority_cursor, _AUTHORITY_CANARY),
            (stored_ingress.server_sequence, _INGRESS_CANARY),
            (orphan.server_sequence, _ORPHAN_CANARY),
        ):
            runtime.store.db.execute(
                """UPDATE sync_envelopes
                   SET payload_ciphertext = ?, encryption_metadata_json = ?
                   WHERE dataset_id = ? AND server_sequence = ?""",
                (
                    canary,
                    json.dumps(
                        {
                            "personal_context_at_rest": {
                                "version": 1,
                                "algorithm": "AES-256-GCM",
                                "nonce": canary + "-nonce",
                                "wrapped_dek": canary + "-dek",
                                "wrapped_dek_nonce": canary + "-dek-nonce",
                                "key_version": 1,
                            }
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "dataset-a",
                    cursor,
                ),
                connection=connection,
            )
        runtime.store.db.execute(
            """UPDATE sync_envelopes SET apply_status = 'superseded'
               WHERE dataset_id = ? AND server_sequence = ?""",
            ("dataset-a", orphan.server_sequence),
            connection=connection,
        )
        runtime.store.db.execute(
            """
            INSERT INTO sync_personal_context_ingress_receipts(
                server_sequence, dataset_id, device_id, client_envelope_id,
                canonical_payload_digest, purge_generation, resulting_object_id,
                resulting_internal_version_id, manifest_revision,
                manifest_version_id, publication_batch_id,
                profile_publication_sequence, receipt_id, wire_entity_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stored_ingress.server_sequence,
                "dataset-a",
                "device-a",
                stored_ingress.client_envelope_id,
                _RECEIPT_CANARY,
                0,
                "old-ingress-object",
                "old-internal-version",
                1,
                "old-manifest-version",
                "old-publication-batch",
                1,
                "old-receipt",
                "old-wire-version",
            ),
            connection=connection,
        )
    return authority_cursor, stored_ingress.server_sequence, source_before


def _stage_dataset_canary(
    runtime: AuthorityHarness,
    *,
    dataset_id: str,
    client_envelope_id: str,
    canary: str,
) -> int:
    """Stage one protected old-generation envelope in an enrolled dataset."""

    dataset = runtime.store.get_dataset(dataset_id)
    envelope = SyncEnvelopeCreate(
        dataset_id=dataset_id,
        client_envelope_id=client_envelope_id,
        device_id="device-a",
        domain="personal_context.record",
        operation="upsert",
        object_id=f"{client_envelope_id}-object",
        object_revision=1,
        schema_version=1,
        adapter_version=1,
        payload={"opaque": "placeholder"},
        payload_hash="hmac-sha256-v1:" + "0" * 64,
        payload_size_bytes=24,
        entity_version=f"{client_envelope_id}-version",
        encryption_metadata={"policy": "server_trusted_v1"},
        routing_metadata={
            "profile_id": runtime.manifest.profile_id,
            "integrity_key_id": dataset.metadata["personal_context"][
                "integrity_key_id"
            ],
            "purge_generation": 0,
            "personal_context_authority": PersonalContextAuthorityMetadata(
                role="client_ingress"
            ).model_dump(mode="json"),
        },
    )
    protected = runtime.service._protect_personal_context_for_storage(dataset, envelope)
    stored = runtime.store.insert_envelope(protected)
    with runtime.store.db.backend.transaction() as connection:
        updated = runtime.store.db.execute(
            """UPDATE sync_envelopes
               SET payload_ciphertext = ?, encryption_metadata_json = ?
               WHERE dataset_id = ? AND server_sequence = ?""",
            (
                canary,
                json.dumps(
                    {
                        "personal_context_at_rest": {
                            "version": 1,
                            "algorithm": "AES-256-GCM",
                            "nonce": canary + "-nonce",
                            "wrapped_dek": canary + "-dek",
                            "wrapped_dek_nonce": canary + "-dek-nonce",
                            "key_version": 1,
                        }
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                dataset_id,
                stored.server_sequence,
            ),
            connection=connection,
        )
        assert updated.rowcount == 1
    return stored.server_sequence


def test_only_confirmed_direct_purge_mints_cleanup_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="confirmation"):
        runtime.canonical.purge_profile(
            mode="everywhere",
            confirmation="delete",
            expected_purge_generation=0,
        )
    assert _intent_rows(runtime.personal_db) == []

    runtime.canonical.create_manual_record(
        scope_id=runtime.canonical.list_scopes()[0].scope_id,
        payload={
            "kind": "preference",
            "subject": "non.purge",
            "polarity": "like",
            "value": "ordinary mutation",
        },
        semantic_key=None,
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    runtime.canonical._repository.compact_pre_activation(
        runtime.manifest.profile_id,
        through_sequence=2,
    )
    assert _intent_rows(runtime.personal_db) == []

    runtime.canonical.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )
    intents = _intent_rows(runtime.personal_db)
    assert len(intents) == 1
    assert (
        intents[0]["profile_id"],
        intents[0]["old_generation_through"],
        intents[0]["purge_generation"],
        intents[0]["state"],
    ) == (runtime.manifest.profile_id, 0, 1, "pending")
    repository = runtime.canonical._repository
    claimed = repository.claim_direct_purge_cleanup(owner_token="owner-a")
    assert claimed is not None
    claimed_again = repository.claim_direct_purge_cleanup(owner_token="owner-a")
    assert claimed_again == claimed
    with pytest.raises(ProfileIntegrityError):
        repository.complete_direct_purge_cleanup(
            replace(claimed, owner_token="owner-b")
        )
    repository.complete_direct_purge_cleanup(claimed)
    repository.complete_direct_purge_cleanup(claimed)


def test_remote_purge_application_never_mints_cleanup_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)

    runtime.canonical.apply_sync_object(
        domain="personal_context.purge",
        value={
            "schema_version": 1,
            "profile_id": runtime.manifest.profile_id,
            "purge_generation": 1,
        },
        actor_type="sync",
        actor_id="device-a",
    )

    assert _intent_rows(runtime.personal_db) == []


def test_expired_verified_claim_cannot_execute_or_complete_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    runtime.canonical.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )
    repository = runtime.canonical._repository
    intent = repository.claim_direct_purge_cleanup(owner_token="expired-owner")
    assert intent is not None
    claim = repository.verify_direct_purge_cleanup_claim(
        intent,
        user_id="user-a",
        dataset_id="dataset-a",
        store=runtime.store,
        database=runtime.store.db,
    )
    with runtime.personal_db.transaction(immediate=True) as connection:
        connection.execute(
            """UPDATE personal_context_purge_cleanup_intents
               SET claim_expires_at_ns = 0 WHERE intent_id = ?""",
            (intent.intent_id,),
        )

    with pytest.raises(SyncStoreError, match="unauthorized"):
        runtime.service.shred_authorized_personal_context_history(claim)
    with pytest.raises(ProfileIntegrityError, match="lost ownership"):
        repository.complete_direct_purge_cleanup(intent)


def test_forged_same_profile_claim_cannot_execute_remote_purge_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    authority_cursor, _ingress_cursor, _source = _stage_retention_canaries(runtime)
    runtime.canonical.apply_sync_object(
        domain="personal_context.purge",
        value={
            "schema_version": 1,
            "profile_id": runtime.manifest.profile_id,
            "purge_generation": 1,
        },
        actor_type="sync",
        actor_id="future-signed-purge",
    )
    assert _intent_rows(runtime.personal_db) == []

    forged = DirectPurgeCleanupIntent(
        intent_id="forged-remote-cleanup",
        profile_id=runtime.manifest.profile_id,
        old_generation_through=0,
        purge_generation=1,
        state="claimed",
        owner_token="forged-owner",
    )
    with pytest.raises(SyncStoreError, match="unauthorized"):
        runtime.service.shred_authorized_personal_context_history(forged)

    retained = runtime.store.get_envelope_by_server_cursor(authority_cursor)
    assert retained is not None
    assert _matches_canary(retained.payload_ciphertext, _AUTHORITY_CANARY)


@pytest.mark.parametrize("layer", ["store", "database"])
def test_lower_cleanup_layers_reject_duck_typed_same_profile_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layer: str,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    authority_cursor, _ingress_cursor, _source = _stage_retention_canaries(runtime)
    runtime.canonical.apply_sync_object(
        domain="personal_context.purge",
        value={
            "schema_version": 1,
            "profile_id": runtime.manifest.profile_id,
            "purge_generation": 1,
        },
        actor_type="sync",
        actor_id="future-signed-purge",
    )
    assert _intent_rows(runtime.personal_db) == []

    class ForgedCleanupClaim:
        dataset_id = "dataset-a"
        user_id = "user-a"
        profile_id = runtime.manifest.profile_id
        old_generation_through = 0
        purge_generation = 1

        @staticmethod
        def _require_live_execution(*, store: object, database: object) -> None:
            return None

        @staticmethod
        def _require_database_execution(database: object) -> None:
            return None

    with pytest.raises(SyncStoreError, match="unauthorized|authority is invalid"):
        _execute_cleanup_layer(runtime, layer, ForgedCleanupClaim())

    retained = runtime.store.get_envelope_by_server_cursor(authority_cursor)
    assert retained is not None
    assert _matches_canary(retained.payload_ciphertext, _AUTHORITY_CANARY)


def test_cleanup_capability_rejects_normal_target_field_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    authority_cursor, _ingress_cursor, _source = _stage_retention_canaries(runtime)
    repository, intent, capability = _issue_cleanup_capability(runtime)
    replacements = {
        "_dataset_id": "dataset-b",
        "_user_id": "user-b",
        "_store": object(),
        "_database": object(),
        "_repository": object(),
        "_intent": replace(intent, owner_token="other-owner"),
        "_provenance": object(),
        "_authentication_tag": b"other-tag",
    }

    for attribute, value in replacements.items():
        fresh = repository.verify_direct_purge_cleanup_claim(
            intent,
            user_id="user-a",
            dataset_id="dataset-a",
            store=runtime.store,
            database=runtime.store.db,
        )
        with pytest.raises((AttributeError, TypeError)):
            setattr(fresh, attribute, value)

    for field_name, value in {
        "profile_id": "other-profile",
        "old_generation_through": 1,
        "purge_generation": 2,
        "intent_id": "other-intent",
        "owner_token": "other-owner",
    }.items():
        with pytest.raises((AttributeError, TypeError)):
            setattr(intent, field_name, value)

    retained = runtime.store.get_envelope_by_server_cursor(authority_cursor)
    assert retained is not None
    assert _matches_canary(retained.payload_ciphertext, _AUTHORITY_CANARY)


@pytest.mark.parametrize("layer", ["service", "store", "database"])
def test_forced_cleanup_capability_dataset_retarget_is_rejected_at_every_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layer: str,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    _stage_retention_canaries(runtime)
    _enroll_matching_dataset(runtime, "dataset-b")
    second_cursor = _stage_dataset_canary(
        runtime,
        dataset_id="dataset-b",
        client_envelope_id="retargeted-dataset-envelope",
        canary=_SECOND_DATASET_CANARY,
    )
    _repository, _intent, capability = _issue_cleanup_capability(runtime)
    object.__setattr__(capability, "_dataset_id", "dataset-b")

    with pytest.raises(SyncStoreError, match="unauthorized|authority is invalid"):
        _execute_cleanup_layer(runtime, layer, capability)

    retained = runtime.store.get_envelope_by_server_cursor(second_cursor)
    assert retained is not None
    assert _matches_canary(retained.payload_ciphertext, _SECOND_DATASET_CANARY)


@pytest.mark.parametrize(
    "mutation",
    [
        "user",
        "store",
        "database",
        "repository",
        "provenance",
        "authentication_tag",
        "profile",
        "old_generation",
        "purge_generation",
        "intent_id",
        "owner",
    ],
)
def test_forced_cleanup_capability_identity_tampering_is_rejected_by_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    authority_cursor, _ingress_cursor, _source = _stage_retention_canaries(runtime)
    _repository, intent, capability = _issue_cleanup_capability(runtime)

    class ForgedRepository:
        @staticmethod
        def _require_live_direct_purge_cleanup_claim(
            cleanup_intent: DirectPurgeCleanupIntent,
        ) -> None:
            return None

    if mutation == "user":
        attribute, value = "_user_id", "user-b"
    elif mutation == "store":
        attribute, value = "_store", object()
    elif mutation == "database":
        attribute, value = "_database", object()
    elif mutation == "repository":
        attribute, value = "_repository", ForgedRepository()
    elif mutation == "provenance":
        attribute, value = "_provenance", object()
    elif mutation == "authentication_tag":
        attribute, value = "_authentication_tag", b"other-tag"
    else:
        intent_changes = {
            "profile": {"profile_id": "other-profile"},
            "old_generation": {"old_generation_through": 1},
            "purge_generation": {"purge_generation": 2},
            "intent_id": {"intent_id": "other-intent"},
            "owner": {"owner_token": "other-owner"},
        }
        attribute, value = "_intent", replace(intent, **intent_changes[mutation])
    object.__setattr__(capability, attribute, value)

    with pytest.raises(SyncStoreError, match="unauthorized|authority is invalid"):
        _execute_cleanup_layer(runtime, "database", capability)

    retained = runtime.store.get_envelope_by_server_cursor(authority_cursor)
    assert retained is not None
    assert _matches_canary(retained.payload_ciphertext, _AUTHORITY_CANARY)


@pytest.mark.parametrize("layer", ["service", "store", "database"])
def test_cleanup_layers_reject_subclass_without_invoking_spoofed_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layer: str,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    authority_cursor, _ingress_cursor, _source = _stage_retention_canaries(runtime)
    validator_calls: list[str] = []

    class SpoofedCapability(_VerifiedDirectPurgeCleanupClaim):
        @property
        def dataset_id(self) -> str:
            return "dataset-a"

        @property
        def user_id(self) -> str:
            return "user-a"

        @property
        def profile_id(self) -> str:
            return runtime.manifest.profile_id

        @property
        def old_generation_through(self) -> int:
            return 0

        @property
        def purge_generation(self) -> int:
            return 1

        @staticmethod
        def _require_live_execution(*, store: object, database: object) -> None:
            validator_calls.append("store")

        @staticmethod
        def _require_database_execution(database: object) -> None:
            validator_calls.append("database")

    spoofed = object.__new__(SpoofedCapability)
    with pytest.raises(SyncStoreError, match="unauthorized|authority is invalid"):
        _execute_cleanup_layer(runtime, layer, spoofed)
    assert validator_calls == []

    retained = runtime.store.get_envelope_by_server_cursor(authority_cursor)
    assert retained is not None
    assert _matches_canary(retained.payload_ciphertext, _AUTHORITY_CANARY)


@pytest.mark.parametrize("claim_state", ["expired", "complete"])
@pytest.mark.parametrize("layer", ["service", "store", "database"])
def test_cleanup_layers_reject_capability_after_live_claim_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    claim_state: str,
    layer: str,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    authority_cursor, _ingress_cursor, _source = _stage_retention_canaries(runtime)
    repository, intent, capability = _issue_cleanup_capability(runtime)
    if claim_state == "expired":
        with runtime.personal_db.transaction(immediate=True) as connection:
            connection.execute(
                """UPDATE personal_context_purge_cleanup_intents
                   SET claim_expires_at_ns = 0 WHERE intent_id = ?""",
                (intent.intent_id,),
            )
    else:
        repository.complete_direct_purge_cleanup(intent)

    with pytest.raises(SyncStoreError, match="unauthorized|authority is invalid"):
        _execute_cleanup_layer(runtime, layer, capability)

    retained = runtime.store.get_envelope_by_server_cursor(authority_cursor)
    assert retained is not None
    assert _matches_canary(retained.payload_ciphertext, _AUTHORITY_CANARY)


def test_cleanup_failure_is_restartable_but_generic_relay_and_scan_are_non_destructive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    authority_cursor, _ingress_cursor, _source = _stage_retention_canaries(runtime)

    def fail_cleanup(_intent: object) -> None:
        raise RuntimeError("simulated cleanup failure")

    runtime.canonical.set_after_commit_purge_cleanup(fail_cleanup)
    runtime.canonical.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )
    assert _intent_rows(runtime.personal_db)[0]["state"] == "pending"

    before = runtime.store.get_envelope_by_server_cursor(authority_cursor)
    assert before is not None
    assert _matches_canary(before.payload_ciphertext, _AUTHORITY_CANARY)
    runtime.resume_relay(row_budget=10)
    runtime.store.scan_personal_context_authority(
        "dataset-a",
        after_server_cursor=0,
        limit=10,
        profile_id=runtime.manifest.profile_id,
        integrity_key_id=runtime.store.get_dataset("dataset-a").metadata[
            "personal_context"
        ]["integrity_key_id"],
        purge_generation=1,
    )
    after = runtime.store.get_envelope_by_server_cursor(authority_cursor)
    assert after is not None
    assert _matches_canary(after.payload_ciphertext, _AUTHORITY_CANARY)

    restarted = PersonalContextService(
        PersonalContextRepository(PersonalizationDB.for_path(runtime.personal_db.db_path))
    )

    def cleanup(intent: DirectPurgeCleanupIntent) -> None:
        claim = restarted._repository.verify_direct_purge_cleanup_claim(
            intent,
            user_id="user-a",
            dataset_id="dataset-a",
            store=runtime.store,
            database=runtime.store.db,
        )
        runtime.service.shred_authorized_personal_context_history(claim)

    restarted.set_after_commit_purge_cleanup(cleanup)
    recovered = restarted.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )
    assert recovered.purge_generation == 1
    assert _intent_rows(runtime.personal_db)[0]["state"] == "complete"


def test_restarted_endpoint_exact_direct_purge_retry_reclaims_expired_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    _stage_retention_canaries(runtime)

    def fail_cleanup(_intent: DirectPurgeCleanupIntent) -> None:
        raise RuntimeError("cleanup unavailable")

    runtime.canonical.set_after_commit_purge_cleanup(fail_cleanup)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/personal-context")
    app.dependency_overrides[get_personal_context_service] = lambda: runtime.canonical
    request = {
        "mode": "everywhere",
        "confirmation": "DELETE EVERYWHERE",
        "expected_purge_generation": 0,
    }

    with TestClient(app) as client:
        first = client.post("/api/v1/personal-context/purge", json=request)
        assert first.status_code == 200
        abandoned = runtime.canonical._repository.claim_direct_purge_cleanup(
            owner_token="abandoned-owner"
        )
        assert abandoned is not None
        with runtime.personal_db.transaction(immediate=True) as connection:
            connection.execute(
                """UPDATE personal_context_purge_cleanup_intents
                   SET claim_expires_at_ns = 0 WHERE intent_id = ?""",
                (abandoned.intent_id,),
            )

        restarted = PersonalContextService(
            PersonalContextRepository(
                PersonalizationDB.for_path(runtime.personal_db.db_path)
            )
        )

        def cleanup(intent: DirectPurgeCleanupIntent) -> None:
            claim = restarted._repository.verify_direct_purge_cleanup_claim(
                intent,
                user_id="user-a",
                dataset_id="dataset-a",
                store=runtime.store,
                database=runtime.store.db,
            )
            runtime.service.shred_authorized_personal_context_history(claim)

        restarted.set_after_commit_purge_cleanup(cleanup)
        app.dependency_overrides[get_personal_context_service] = lambda: restarted
        wrong_confirmation = client.post(
            "/api/v1/personal-context/purge",
            json={**request, "confirmation": "delete"},
        )
        assert wrong_confirmation.status_code == 422
        different_generation = client.post(
            "/api/v1/personal-context/purge",
            json={**request, "expected_purge_generation": 1},
        )
        assert different_generation.status_code == 409

        retried = client.post("/api/v1/personal-context/purge", json=request)

    assert retried.status_code == 200
    assert retried.json()["purge_generation"] == 1
    assert _intent_rows(runtime.personal_db)[0]["state"] == "complete"


@pytest.mark.parametrize("failure", ["non_wal", "secure_delete_unavailable"])
def test_direct_purge_refuses_unverified_canonical_sqlite_retention_prerequisites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    _stage_retention_canaries(runtime)
    _install_cleanup_callback(runtime)
    original_connect = runtime.personal_db._connect

    def restricted_connect() -> sqlite3.Connection:
        connection = original_connect()
        if failure == "non_wal":
            connection.execute("PRAGMA journal_mode=DELETE")
        else:
            connection.set_authorizer(
                lambda action, name, _argument, _database, _source: (
                    sqlite3.SQLITE_DENY
                    if action == sqlite3.SQLITE_PRAGMA
                    and str(name).lower() == "secure_delete"
                    else sqlite3.SQLITE_OK
                )
            )
        return connection

    monkeypatch.setattr(runtime.personal_db, "_connect", restricted_connect)
    with pytest.raises(ProfileIntegrityError, match="retention prerequisites"):
        runtime.canonical.purge_profile(
            mode="everywhere",
            confirmation="DELETE EVERYWHERE",
            expected_purge_generation=0,
        )

    assert runtime.canonical.get_manifest().purge_generation == 0
    assert _intent_rows(runtime.personal_db) == []


@pytest.mark.parametrize("failure", ["non_wal", "secure_delete_unavailable"])
def test_unverified_sync_retention_prerequisite_leaves_cleanup_pending_until_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    authority_cursor, _ingress_cursor, _source = _stage_retention_canaries(runtime)
    _install_cleanup_callback(runtime)
    connection = runtime.store.db.backend.get_pool().get_connection()
    if failure == "non_wal":
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    else:
        connection.set_authorizer(
            lambda action, name, _argument, _database, _source: (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_PRAGMA
                and str(name).lower() == "secure_delete"
                else sqlite3.SQLITE_OK
            )
        )

    runtime.canonical.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )

    assert _intent_rows(runtime.personal_db)[0]["state"] == "pending"
    retained = runtime.store.get_envelope_by_server_cursor(authority_cursor)
    assert retained is not None
    assert _matches_canary(retained.payload_ciphertext, _AUTHORITY_CANARY)

    if failure == "non_wal":
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    else:
        connection.set_authorizer(None)
    runtime.canonical.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )
    assert _intent_rows(runtime.personal_db)[0]["state"] == "complete"


def test_busy_wal_checkpoint_leaves_intent_pending_until_reader_releases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    _stage_retention_canaries(runtime)
    _install_cleanup_callback(runtime)
    sync_path = Path(runtime.store.db.backend.config.sqlite_path)
    reader = sqlite3.connect(sync_path, isolation_level=None)
    reader.execute("PRAGMA journal_mode=WAL")
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM sync_envelopes").fetchone()
    try:
        runtime.canonical.purge_profile(
            mode="everywhere",
            confirmation="DELETE EVERYWHERE",
            expected_purge_generation=0,
        )
        assert _intent_rows(runtime.personal_db)[0]["state"] == "pending"
    finally:
        reader.rollback()
        reader.close()

    runtime.canonical.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )
    assert _intent_rows(runtime.personal_db)[0]["state"] == "complete"


def test_restart_recovery_rewrites_main_database_freelist_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    _stage_retention_canaries(runtime)
    sync_path = Path(runtime.store.db.backend.config.sqlite_path)

    def cleanup_then_leave_residue(intent: DirectPurgeCleanupIntent) -> None:
        claim = runtime.canonical._repository.verify_direct_purge_cleanup_claim(
            intent,
            user_id="user-a",
            dataset_id="dataset-a",
            store=runtime.store,
            database=runtime.store.db,
        )
        runtime.service.shred_authorized_personal_context_history(claim)
        connection = runtime.store.db.backend.get_pool().get_connection()
        connection.execute("PRAGMA secure_delete=OFF")
        runtime.store.store_key_record(
            SyncKeyRecordCreate(
                key_record_id="freelist-residue",
                dataset_id="dataset-a",
                user_id="user-a",
                device_id="device-a",
                key_purpose="personal_context_integrity",
                wrapped_key_blob=_FREELIST_CANARY * 8_192,
                kdf_metadata={"algorithm": "test"},
                encryption_policy="server_trusted_v1",
                key_epoch=3,
            )
        )
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        with runtime.store.db.backend.transaction(connection) as transaction:
            runtime.store.db.execute(
                "DELETE FROM sync_key_records WHERE key_record_id = ?",
                ("freelist-residue",),
                connection=transaction,
            )
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA secure_delete=ON")
        raise RuntimeError("maintenance interrupted after logical deletion")

    runtime.canonical.set_after_commit_purge_cleanup(cleanup_then_leave_residue)
    runtime.canonical.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )
    assert _intent_rows(runtime.personal_db)[0]["state"] == "pending"
    assert _sqlite_artifacts_contain(sync_path, _FREELIST_CANARY)

    restarted = PersonalContextService(
        PersonalContextRepository(PersonalizationDB.for_path(runtime.personal_db.db_path))
    )

    def recover_cleanup(intent: DirectPurgeCleanupIntent) -> None:
        claim = restarted._repository.verify_direct_purge_cleanup_claim(
            intent,
            user_id="user-a",
            dataset_id="dataset-a",
            store=runtime.store,
            database=runtime.store.db,
        )
        runtime.service.shred_authorized_personal_context_history(claim)

    restarted.set_after_commit_purge_cleanup(recover_cleanup)
    restarted.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )
    assert _intent_rows(runtime.personal_db)[0]["state"] == "complete"
    assert runtime.store.db.backend.get_pool().get_connection().execute(
        "PRAGMA freelist_count"
    ).fetchone()[0] == 0
    assert not _sqlite_artifacts_contain(sync_path, _FREELIST_CANARY)


def test_partial_multi_dataset_cleanup_is_retry_safe_and_converges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    first_cursor, _ingress_cursor, _source = _stage_retention_canaries(runtime)
    original_dataset = runtime.store.get_dataset("dataset-a")
    runtime.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-b",
            owner_user_id="user-a",
            encryption_policy="server_trusted_v1",
            domains=list(original_dataset.domains),
            metadata={
                "personal_context": {
                    **original_dataset.metadata["personal_context"],
                    "profile_id": runtime.manifest.profile_id,
                    "purge_generation": 0,
                }
            },
        )
    )
    second_cursor = _stage_dataset_canary(
        runtime,
        dataset_id="dataset-b",
        client_envelope_id="second-dataset-old-envelope",
        canary=_SECOND_DATASET_CANARY,
    )
    fail_after_first = True

    def cleanup(intent: DirectPurgeCleanupIntent) -> None:
        nonlocal fail_after_first
        for dataset_id in ("dataset-a", "dataset-b"):
            claim = runtime.canonical._repository.verify_direct_purge_cleanup_claim(
                intent,
                user_id="user-a",
                dataset_id=dataset_id,
                store=runtime.store,
                database=runtime.store.db,
            )
            runtime.service.shred_authorized_personal_context_history(claim)
            if fail_after_first:
                fail_after_first = False
                raise RuntimeError("second dataset unavailable")

    runtime.canonical.set_after_commit_purge_cleanup(cleanup)
    runtime.canonical.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )
    assert _intent_rows(runtime.personal_db)[0]["state"] == "pending"
    first = runtime.store.get_envelope_by_server_cursor(first_cursor)
    second = runtime.store.get_envelope_by_server_cursor(second_cursor)
    assert first is not None and first.payload_ciphertext is None
    assert second is not None
    assert _matches_canary(second.payload_ciphertext, _SECOND_DATASET_CANARY)

    runtime.canonical.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )
    assert _intent_rows(runtime.personal_db)[0]["state"] == "complete"
    second = runtime.store.get_envelope_by_server_cursor(second_cursor)
    assert second is not None and second.payload_ciphertext is None


def test_authorized_cleanup_scrubs_old_material_and_preserves_unrelated_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    authority_cursor, ingress_cursor, source_before = _stage_retention_canaries(runtime)
    keys = runtime.canonical._repository.key_material_for_test(runtime.manifest.profile_id)
    runtime.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-b",
            owner_user_id="user-a",
            encryption_policy="server_trusted_v1",
            domains=list(runtime.store.get_dataset("dataset-a").domains),
            metadata={
                "personal_context": {
                    "profile_id": "other-profile",
                    "integrity_key_id": "other-key",
                    "purge_generation": 0,
                    "ongoing_sync_version": 1,
                }
            },
        )
    )
    with runtime.store.db.backend.transaction() as connection:
        runtime.store.db.execute(
            """INSERT INTO sync_envelopes(
                   dataset_id, domain, entity_id, operation, client_envelope_id,
                   server_timestamp, payload_json, payload_clear_json,
                   routing_metadata_json, encryption_metadata_json,
                   adapter_version, status, apply_status
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "dataset-a",
                "notes.note",
                "unrelated-note",
                "upsert",
                "unrelated-note-envelope",
                "2026-09-04T00:00:00Z",
                '{"body":"unchanged"}',
                '{"body":"unchanged"}',
                "{}",
                "{}",
                1,
                "accepted",
                "applied",
            ),
            connection=connection,
        )
        unrelated_before = runtime.store.db.execute(
            "SELECT * FROM sync_envelopes WHERE client_envelope_id = ?",
            ("unrelated-note-envelope",),
            connection=connection,
        ).rows[0]
        runtime.store.db.execute(
            """INSERT INTO sync_envelopes(
                   dataset_id, domain, entity_id, operation, client_envelope_id,
                   server_timestamp, payload_json, payload_clear_json,
                   routing_metadata_json, encryption_metadata_json,
                   adapter_version, status, apply_status
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "dataset-a",
                "personal_context.record",
                "current-record",
                "upsert",
                "current-generation-envelope",
                "2026-09-04T00:00:01Z",
                '{"body":"current"}',
                '{"body":"current"}',
                json.dumps(
                    {
                        "profile_id": runtime.manifest.profile_id,
                        "purge_generation": 1,
                    }
                ),
                "{}",
                1,
                "accepted",
                "applied",
            ),
            connection=connection,
        )
        runtime.store.db.execute(
            """INSERT INTO sync_envelopes(
                   dataset_id, domain, entity_id, operation, client_envelope_id,
                   server_timestamp, payload_json, payload_clear_json,
                   routing_metadata_json, encryption_metadata_json,
                   adapter_version, status, apply_status
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "dataset-b",
                "personal_context.record",
                "other-profile-record",
                "upsert",
                "other-profile-envelope",
                "2026-09-04T00:00:02Z",
                '{"body":"other"}',
                '{"body":"other"}',
                '{"profile_id":"other-profile","purge_generation":0}',
                "{}",
                1,
                "accepted",
                "applied",
            ),
            connection=connection,
        )
        current_before = runtime.store.db.execute(
            "SELECT * FROM sync_envelopes WHERE client_envelope_id = ?",
            ("current-generation-envelope",),
            connection=connection,
        ).rows[0]
        other_profile_before = runtime.store.db.execute(
            "SELECT * FROM sync_envelopes WHERE client_envelope_id = ?",
            ("other-profile-envelope",),
            connection=connection,
        ).rows[0]

    with runtime.personal_db.transaction() as connection:
        connection.execute(
            """
            INSERT INTO personal_context_ingress_receipts(
                dataset_id, device_id, client_envelope_id,
                canonical_payload_digest, purge_generation, wire_entity_version,
                resulting_object_id, resulting_version_id,
                resulting_manifest_revision, resulting_manifest_version_id,
                publication_batch_id, profile_publication_sequence,
                receipt_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "unrelated-dataset",
                "unrelated-device",
                "unrelated-ingress",
                "unrelated-digest",
                0,
                "unrelated-wire-version",
                "unrelated-object",
                "unrelated-version",
                1,
                "unrelated-manifest-version",
                "unrelated-batch",
                1,
                "unrelated-receipt",
                "2026-09-04T00:00:00Z",
            ),
        )
        unrelated_receipt_before = connection.execute(
            """SELECT * FROM personal_context_ingress_receipts
               WHERE receipt_id = ?""",
            ("unrelated-receipt",),
        ).fetchone()

    _install_cleanup_callback(runtime)
    runtime.canonical.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )

    assert _intent_rows(runtime.personal_db)[0]["state"] == "complete"
    for cursor in (authority_cursor, ingress_cursor):
        stored = runtime.store.get_envelope_by_server_cursor(cursor)
        assert stored is not None
        assert stored.payload_ciphertext is None
        assert stored.payload == {}
        assert stored.encryption_metadata == {}
        assert stored.routing_metadata == {
            "profile_id": runtime.manifest.profile_id,
            "purge_generation": 0,
            "retention_state": "shredded",
        }
    conflict = runtime.store.get_conflict("old-personal-context-conflict")
    assert conflict is not None and conflict.status == "superseded"
    assert conflict.metadata == {}
    assert runtime.store.list_key_records(
        "dataset-a",
        user_id="user-a",
        key_purpose="personal_context_integrity",
    ) == []
    receipt = runtime.store.get_personal_context_ingress_receipt(ingress_cursor)
    assert receipt is not None and receipt["canonical_payload_digest"] == "shredded"
    with runtime.personal_db.transaction() as connection:
        source_after = connection.execute(
            """SELECT * FROM personal_context_publication_rows
               WHERE profile_id = ? AND profile_publication_sequence = ?
                 AND batch_ordinal = ?""",
            (
                source_before["profile_id"],
                source_before["profile_publication_sequence"],
                source_before["batch_ordinal"],
            ),
        ).fetchone()
        old_canonical_receipt = connection.execute(
            """SELECT * FROM personal_context_ingress_receipts
               WHERE receipt_id = ?""",
            ("canonical-old-receipt",),
        ).fetchone()
        unrelated_receipt_after = connection.execute(
            """SELECT * FROM personal_context_ingress_receipts
               WHERE receipt_id = ?""",
            ("unrelated-receipt",),
        ).fetchone()
    assert old_canonical_receipt is None
    assert unrelated_receipt_after == unrelated_receipt_before
    with pytest.raises(EnvelopeAuthenticationError):
        PersonalContextPublicationJournal(keys).decrypt_row(source_after)

    with runtime.store.db.backend.transaction() as connection:
        unrelated_after = runtime.store.db.execute(
            "SELECT * FROM sync_envelopes WHERE client_envelope_id = ?",
            ("unrelated-note-envelope",),
            connection=connection,
        ).rows[0]
        current_after = runtime.store.db.execute(
            "SELECT * FROM sync_envelopes WHERE client_envelope_id = ?",
            ("current-generation-envelope",),
            connection=connection,
        ).rows[0]
        other_profile_after = runtime.store.db.execute(
            "SELECT * FROM sync_envelopes WHERE client_envelope_id = ?",
            ("other-profile-envelope",),
            connection=connection,
        ).rows[0]
    assert unrelated_after == unrelated_before
    assert current_after == current_before
    assert other_profile_after == other_profile_before
    state = runtime.store.get_dataset("dataset-a").metadata["personal_context"]
    assert state["purge_generation"] == 1
    assert state["ongoing_sync_version"] == 1

    sync_path = Path(runtime.store.db.backend.config.sqlite_path)
    personalization_path = Path(runtime.personal_db.db_path)
    artifact_bytes = b"".join(
        path.read_bytes()
        for path in (
            sync_path,
            Path(f"{sync_path}-wal"),
            Path(f"{sync_path}-shm"),
            personalization_path,
            Path(f"{personalization_path}-wal"),
            Path(f"{personalization_path}-shm"),
        )
        if path.exists()
    )
    assert not any(
        token.encode("utf-8") in artifact_bytes
        for token in (
            _AUTHORITY_CANARY,
            _INGRESS_CANARY,
            _CONFLICT_CANARY,
            _KEY_CANARY,
            _ORPHAN_CANARY,
            _RECEIPT_CANARY,
            _CANONICAL_RECEIPT_CANARY,
            _SOURCE_CANARY,
        )
    )


def test_cleanup_is_idempotent_and_rejects_wrong_profile_or_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AuthorityHarness(tmp_path, monkeypatch)
    authority_cursor, _ingress_cursor, _source = _stage_retention_canaries(runtime)
    _install_cleanup_callback(runtime)
    runtime.canonical.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )
    intent = runtime.canonical._repository.claim_direct_purge_cleanup(
        owner_token="replay-owner"
    )
    assert intent is None

    completed = runtime.canonical._repository.completed_direct_purge_cleanup(
        runtime.manifest.profile_id,
        purge_generation=1,
    )
    assert completed is not None
    first = runtime.store.get_envelope_by_server_cursor(authority_cursor)
    repeated = runtime.canonical.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )
    second = runtime.store.get_envelope_by_server_cursor(authority_cursor)
    assert repeated.purge_generation == 1
    assert first == second
    with pytest.raises(SyncStoreError, match="unauthorized"):
        runtime.service.shred_authorized_personal_context_history(completed)

    wrong = completed.__class__(
        intent_id=completed.intent_id,
        profile_id="other-profile",
        old_generation_through=completed.old_generation_through,
        purge_generation=completed.purge_generation,
        state=completed.state,
        owner_token=completed.owner_token,
    )
    with pytest.raises(SyncStoreError):
        runtime.service.shred_authorized_personal_context_history(wrong)

    sync_db = SyncDatabase(sqlite_path=tmp_path / "sync.db")
    assert SyncV2Store(sync_db).get_dataset("dataset-a").metadata[
        "personal_context"
    ]["purge_generation"] == 1

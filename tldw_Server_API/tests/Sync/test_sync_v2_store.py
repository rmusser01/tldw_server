from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from typing import Any, cast

import pytest

import tldw_Server_API.app.core.Sync.v2.store as store_module
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    QueryResult,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import (
    DatabaseBackendFactory,
)
from tldw_Server_API.app.core.DB_Management.Sync_DB import (
    SYNC_POSTGRES_SCHEMA,
    SYNC_SQLITE_SCHEMA,
    SYNC_VERSIONED_DEVICE_STATE_POSTGRES_SCHEMA,
    SYNC_VERSIONED_DEVICE_STATE_SQLITE_SCHEMA,
    SyncDatabase,
    _envelope_fingerprint_from_create,
    _envelope_fingerprint_from_row,
    _envelope_from_row,
    utcnow_iso,
)
from tldw_Server_API.app.core.Sync.v2.errors import (
    SyncDatasetNotFoundError,
    SyncIdempotencyConflictError,
    SyncInvalidDomainError,
    SyncMaterializationPredecessorError,
    SyncStoreError,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncAttachmentCreate,
    SyncAttachmentRevisionBinding,
    SyncAttachmentRevisionBindingCreate,
    SyncBackgroundLeaseCreate,
    SyncBackgroundPolicyUpsert,
    SyncBlobChunkCreate,
    SyncBlobObjectCreate,
    SyncBlobUploadSessionCreate,
    SyncConflictCreate,
    SyncDatasetCreate,
    SyncDeviceAuthorizationCreate,
    SyncDeviceBlobAckCreate,
    SyncDeviceBlobIdAckCreate,
    SyncDeviceCursor,
    SyncDeviceDomainAckCreate,
    SyncDeviceUpsert,
    SyncEnvelopeCreate,
    SyncKeyRecordCreate,
    PERSONAL_CONTEXT_SYNC_DOMAINS,
)
from tldw_Server_API.app.core.Sync.v2.mutation_group_validation import (
    mutation_group_plan_hash,
    validate_stored_mutation_group,
)
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


@pytest.fixture()
def sync_store(tmp_path: Path) -> SyncV2Store:
    db = SyncDatabase(sqlite_path=tmp_path / "sync_v2.db")
    return SyncV2Store(db)


def _device(**overrides) -> SyncDeviceUpsert:
    payload = {
        "device_id": "device-1",
        "user_id": "user-1",
        "display_name": "Laptop",
        "client_type": "chatbook",
        "client_version": "0.1.0",
        "capabilities": {"domains": ["notes.note"]},
    }
    payload.update(overrides)
    return SyncDeviceUpsert(**payload)


def _dataset(**overrides) -> SyncDatasetCreate:
    payload = {
        "dataset_id": "dataset-1",
        "owner_user_id": "user-1",
        "scope_type": "personal",
        "encryption_policy": "server_trusted_v1",
        "domains": [
            "notes.note",
            "chat.conversation",
            "chat.message",
            "attachment.ref",
        ],
        "metadata": {"label": "Personal research"},
    }
    payload.update(overrides)
    return SyncDatasetCreate(**payload)


_TASK_CURSOR_1 = "00000000-0000-4000-8000-000000000001"
_TASK_CURSOR_2 = "00000000-0000-4000-8000-000000000002"
_TASK_CURSOR_3 = "00000000-0000-4000-8000-000000000003"
_ACTIVITY_CURSOR_1 = (
    "2026-08-13T00:00:00+00:00|00000000-0000-4000-8000-000000000011"
)
_ACTIVITY_CURSOR_2 = (
    "2026-08-13T00:00:01+00:00|00000000-0000-4000-8000-000000000012"
)
_MOODBOARD_CURSOR_1 = "00000000-0000-4000-8000-000000000101"
_MOODBOARD_CURSOR_2 = "00000000-0000-4000-8000-000000000102"
_PLACEMENT_CURSOR_1 = (
    "00000000-0000-4000-8000-000000000101|"
    "00000000-0000-4000-8000-000000000201"
)
_PLACEMENT_CURSOR_2 = (
    "00000000-0000-4000-8000-000000000102|"
    "00000000-0000-4000-8000-000000000202"
)
_STUDIO_CURSOR_1 = "00000000-0000-4000-8000-000000000301"
_STUDIO_CURSOR_2 = "00000000-0000-4000-8000-000000000302"


def _readiness_record(
    *,
    state: str,
    source_cursor: str | None = None,
    source_count: int = 0,
    source_fingerprint: str | None = None,
    reason_code: str | None = None,
    resume_phase: str | None = None,
) -> dict[str, object]:
    return {
        "state": state,
        "source_cursor": source_cursor,
        "source_count": source_count,
        "source_fingerprint": source_fingerprint,
        "reason_code": reason_code,
        "resume_phase": resume_phase,
    }


def _envelope(**overrides) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-1",
        "domain": "notes.note",
        "object_id": "note-1",
        "operation": "upsert",
        "device_id": "device-1",
        "client_profile_id": "chatbook-profile-1",
        "client_sequence": None,
        "base_server_cursor": None,
        "base_object_revision": None,
        "base_object_hash": None,
        "object_revision": None,
        "parent_id": None,
        "schema_version": 1,
        "payload": {"status": "active"},
        "payload_hash": "sha256:note-1",
        "payload_size_bytes": 24,
        "created_at_client": "2026-05-10T00:00:00+00:00",
        "deleted": False,
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "status": "accepted",
        "apply_status": "pending",
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _mutation_group_envelopes(
    *,
    mutation_group_id: str = "mutation-group-1",
    mutation_plan_hash: str = "a" * 64,
    count: int = 3,
) -> list[SyncEnvelopeCreate]:
    return [
        _envelope(
            client_envelope_id=f"env-group-{step}",
            object_id=f"note-{step + 1}",
            payload_hash=f"sha256:note-{step + 1}",
            mutation_group_id=mutation_group_id,
            mutation_step=step,
            mutation_step_count=count,
            mutation_plan_hash=mutation_plan_hash,
        )
        for step in range(count)
    ]


class _PostgresUniqueDiagnostics:
    constraint_name = "uq_sync_envelopes_dataset_mutation_group_step"


class _PostgresMutationGroupUniqueViolation(Exception):
    sqlstate = "23505"
    diag = _PostgresUniqueDiagnostics()


class _PostgresDeviceLockBackend:
    config = DatabaseConfig(backend_type=BackendType.POSTGRESQL)

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...] | None, Any]] = []
        self.row = {
            "device_id": "device-1",
            "user_id": "user-1",
            "display_name": "Laptop",
            "client_type": "chatbook",
            "client_version": "0.1.0",
            "capabilities_json": '{"domains":["notes.note"]}',
            "registered_at": "2026-05-10T00:00:00+00:00",
            "last_seen_at": "2026-05-10T00:00:00+00:00",
            "status": "active",
            "user_label": None,
            "authorized_at": None,
            "revoked_at": None,
            "revoked_reason": None,
        }

    @contextmanager
    def transaction(self, connection=None):
        yield connection or object()

    def execute(
        self,
        statement: str,
        params: tuple[Any, ...] | None = None,
        connection: Any = None,
    ) -> QueryResult:
        normalized = " ".join(statement.split())
        self.calls.append((normalized, params, connection))
        if normalized.startswith("SELECT * FROM sync_devices"):
            return QueryResult(rows=[dict(self.row)], rowcount=1)
        if normalized.startswith("UPDATE sync_devices"):
            assert params is not None
            self.row["display_name"] = params[1]
            self.row["capabilities_json"] = params[4]
            self.row["last_seen_at"] = params[5]
            return QueryResult(rows=[], rowcount=1)
        return QueryResult(rows=[], rowcount=1)


class _PostgresPersonalContextReceiptBackend:
    config = DatabaseConfig(backend_type=BackendType.POSTGRESQL)

    def __init__(self, *, stale_binding: bool = False) -> None:
        self.calls: list[tuple[str, tuple[Any, ...] | None, Any]] = []
        integrity_key_id = (
            "personal-context-integrity-vstale"
            if stale_binding
            else "personal-context-integrity-v1"
        )
        self.dataset_row = {
            "dataset_id": "dataset-1",
            "owner_user_id": "user-1",
            "metadata_json": json.dumps(
                {
                    "personal_context": {
                        "profile_id": "profile-1",
                        "integrity_key_id": integrity_key_id,
                        "purge_generation": 0,
                    }
                }
            ),
        }

    @contextmanager
    def transaction(self, connection=None):
        yield connection or object()

    def execute(
        self,
        statement: str,
        params: tuple[Any, ...] | None = None,
        connection: Any = None,
    ) -> QueryResult:
        normalized = " ".join(statement.split())
        self.calls.append((normalized, params, connection))
        if normalized.startswith("SELECT * FROM sync_datasets"):
            return QueryResult(rows=[dict(self.dataset_row)], rowcount=1)
        return QueryResult(rows=[], rowcount=1)


def _inject_postgres_mutation_group_race(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    original_list_group_rows = sync_store.db._list_mutation_group_rows
    original_transaction = sync_store.db.backend.transaction
    calls = {"list": 0, "transactions": 0}

    def hide_group_until_race_rollback(
        dataset_id: str,
        mutation_group_id: str,
        *,
        connection=None,
    ):
        calls["list"] += 1
        if calls["list"] == 1:
            return []
        return original_list_group_rows(
            dataset_id,
            mutation_group_id,
            connection=connection,
        )

    @contextmanager
    def track_transaction(connection=None):
        calls["transactions"] += 1
        with original_transaction(connection) as conn:
            yield conn

    def lose_group_step_unique_race(envelope, *, connection):
        raise BackendDatabaseError("injected PostgreSQL unique race") from (
            _PostgresMutationGroupUniqueViolation()
        )

    monkeypatch.setattr(
        sync_store.db,
        "_list_mutation_group_rows",
        hide_group_until_race_rollback,
    )
    monkeypatch.setattr(
        sync_store.db,
        "_find_existing_envelope_for_idempotency",
        lambda envelope, *, connection: None,
    )
    monkeypatch.setattr(
        sync_store.db,
        "_insert_envelope_in_transaction",
        lose_group_step_unique_race,
    )
    monkeypatch.setattr(sync_store.db.backend, "transaction", track_transaction)
    return calls


def _conflict(**overrides) -> SyncConflictCreate:
    payload = {
        "conflict_id": "conflict-1",
        "dataset_id": "dataset-1",
        "domain": "notes.note",
        "object_id": "note-1",
        "conflict_type": "version_divergence",
        "base_envelope_id": "env-base",
        "local_envelope_id": "env-local",
        "remote_envelope_id": "env-remote",
        "server_cursor": 3,
        "metadata": {"reason": "same entity changed on two devices"},
    }
    payload.update(overrides)
    return SyncConflictCreate(**payload)


def _key_record(**overrides) -> SyncKeyRecordCreate:
    payload = {
        "key_record_id": "key-1",
        "dataset_id": "dataset-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "key_purpose": "dataset_recovery",
        "wrapped_key_blob": "wrapped:opaque",
        "kdf_metadata": {"algorithm": "argon2id"},
        "recovery_hint": "personal laptop",
    }
    payload.update(overrides)
    return SyncKeyRecordCreate(**payload)


def _attachment(**overrides) -> SyncAttachmentCreate:
    payload = {
        "attachment_id": "attachment-1",
        "dataset_id": "dataset-1",
        "domain": "attachment.ref",
        "object_id": "attachment-1",
        "content_type": "application/octet-stream",
        "size_bytes": 512,
        "payload_ciphertext": "ciphertext:attachment",
        "payload_hash": "sha256:attachment",
        "encryption_policy": "server_trusted_v1",
        "metadata": {
            "parent_domain": "notes.note",
            "parent_object_id": "note-1",
            "availability": "available",
        },
    }
    payload.update(overrides)
    return SyncAttachmentCreate(**payload)


def _attachment_binding(**overrides) -> SyncAttachmentRevisionBindingCreate:
    payload = {
        "dataset_id": "dataset-1",
        "attachment_id": "11111111-1111-4111-8111-111111111111",
        "attachment_revision": 1,
        "blob_hash": "sha256:" + "a" * 64,
        "size_bytes": 2048,
        "establishing_server_cursor": 7,
        "availability_at_acceptance": "metadata_only",
    }
    payload.update(overrides)
    return SyncAttachmentRevisionBindingCreate(**payload)


def _insert_attachment_binding_for_schema_test(
    sync_store: SyncV2Store,
    binding: SyncAttachmentRevisionBindingCreate,
) -> SyncAttachmentRevisionBinding:
    """Exercise the private append seam for schema/query-plan tests only."""

    with sync_store.db.backend.transaction() as connection:
        return sync_store.db._create_attachment_revision_binding(
            binding,
            connection=connection,
        )


def _attachment_v2_envelope(**overrides) -> SyncEnvelopeCreate:
    from tldw_Server_API.app.core.Sync.v2.attachment_refs_v2 import (
        attachment_ref_v2_object_hash,
        parse_attachment_ref_v2_payload,
    )

    attachment_id = "11111111-1111-4111-8111-111111111111"
    payload = {
        "attachment_id": attachment_id,
        "parent_domain": "notes.note",
        "parent_object_id": "22222222-2222-4222-8222-222222222222",
        "file_name": "report.pdf",
        "original_file_name": "report.pdf",
        "content_type": "application/pdf",
        "size_bytes": 2048,
        "blob_hash": "sha256:" + "a" * 64,
        "created_at": "2026-08-11T20:30:00+00:00",
        "last_modified": "2026-08-11T20:30:00+00:00",
        "created_by": "device-1",
    }
    parsed = parse_attachment_ref_v2_payload("upsert", payload)
    values = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-attachment-v2",
        "domain": "attachment.ref",
        "object_id": attachment_id,
        "operation": "upsert",
        "device_id": "device-1",
        "client_sequence": 1,
        "schema_version": 2,
        "adapter_version": 2,
        "object_revision": 1,
        "payload": payload,
        "payload_hash": attachment_ref_v2_object_hash(
            "upsert",
            parsed,
            object_revision=1,
        ),
        "created_at_client": "2026-08-11T20:30:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "status": "accepted",
        "apply_status": "pending",
    }
    values.update(overrides)
    return SyncEnvelopeCreate(**values)


def test_sync_database_rejects_unsupported_database_url_scheme(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SYNC_V2_DATABASE_URL", "mysql://sync.example/sync_v2")
    monkeypatch.delenv("SYNC_V2_SQLITE_PATH", raising=False)

    with pytest.raises(SyncStoreError, match="Unsupported SYNC_V2_DATABASE_URL scheme"):
        SyncDatabase(sqlite_path=tmp_path / "ignored.db")


def test_sync_database_bootstrap_creates_required_tables(sync_store: SyncV2Store):
    required_tables = {
        "sync_devices",
        "sync_datasets",
        "sync_device_authorizations",
        "sync_device_domain_acks",
        "sync_device_blob_acks",
        "sync_background_policies",
        "sync_background_leases",
        "sync_domain_state",
        "sync_envelopes",
        "sync_object_state",
        "sync_materialization_locks",
        "sync_device_cursors",
        "sync_device_adapter_cursors",
        "sync_device_adapter_domain_acks",
        "sync_device_blob_id_acks",
        "sync_conflicts",
        "sync_key_records",
        "sync_attachments",
        "sync_blob_objects",
        "sync_blob_upload_sessions",
        "sync_blob_chunks",
        "sync_attachment_revision_bindings",
        "sync_dataset_storage_namespaces",
    }

    for table_name in required_tables:
        assert sync_store.db.backend.table_exists(table_name)


def test_adapter_cursor_and_version_ack_schema_has_fresh_backend_parity(
    sync_store: SyncV2Store,
) -> None:
    for schema in (
        SYNC_SQLITE_SCHEMA + SYNC_VERSIONED_DEVICE_STATE_SQLITE_SCHEMA,
        SYNC_POSTGRES_SCHEMA + SYNC_VERSIONED_DEVICE_STATE_POSTGRES_SCHEMA,
    ):
        assert "sync_device_adapter_cursors" in schema
        assert "sync_device_adapter_domain_acks" in schema
        assert "sync_device_blob_id_acks" in schema
        assert "PRIMARY KEY(dataset_id, device_id, domain, adapter_version)" in schema
        assert "PRIMARY KEY(dataset_id, device_id, blob_id)" in schema

    cursor_columns = {
        row["name"] for row in sync_store.db.execute(
            "PRAGMA table_info(sync_device_adapter_cursors)"
        ).rows
    }
    assert {"adapter_version", "last_pulled_sequence", "max_delivered_sequence"}.issubset(
        cursor_columns
    )


def test_adapter_state_migration_records_completion_and_exact_sqlite_catalog(
    sync_store: SyncV2Store,
) -> None:
    marker = sync_store.db.execute(
        "SELECT completed_at FROM sync_schema_migrations WHERE migration_id = ?",
        ("adapter_cursor_ack_blob_id_v1",),
    ).rows
    assert len(marker) == 1 and marker[0]["completed_at"]

    expected_primary_keys = {
        "sync_device_adapter_cursors": {
            "dataset_id": 1,
            "device_id": 2,
            "domain": 3,
            "adapter_version": 4,
        },
        "sync_device_adapter_domain_acks": {
            "dataset_id": 1,
            "device_id": 2,
            "domain": 3,
            "adapter_version": 4,
        },
        "sync_device_blob_id_acks": {
            "dataset_id": 1,
            "device_id": 2,
            "blob_id": 3,
        },
    }
    for table_name, expected_pk in expected_primary_keys.items():
        info = sync_store.db.execute(f"PRAGMA table_info({table_name})").rows
        assert {row["name"]: row["pk"] for row in info if row["pk"]} == expected_pk

    indexes = {
        row["name"]
        for row in sync_store.db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).rows
    }
    assert {
        "idx_sync_device_adapter_cursors_device",
        "idx_sync_device_adapter_domain_acks_device",
        "idx_sync_device_blob_id_acks_device",
    }.issubset(indexes)


def test_adapter_state_migration_seed_failure_rolls_back_ddl_and_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "atomic-adapter-migration.db"
    db = SyncDatabase(sqlite_path=path)
    for table_name in (
        "sync_device_adapter_cursors",
        "sync_device_adapter_domain_acks",
        "sync_device_blob_id_acks",
    ):
        db.execute(f"DROP TABLE {table_name}")
    db.execute(
        "DELETE FROM sync_schema_migrations WHERE migration_id = ?",
        ("adapter_cursor_ack_blob_id_v1",),
    )
    original_execute = db.execute

    def fail_seed(statement, params=None, *, connection=None):
        if "INSERT INTO sync_device_adapter_domain_acks" in statement:
            raise SyncStoreError("injected atomic seed failure")
        return original_execute(statement, params, connection=connection)

    monkeypatch.setattr(db, "execute", fail_seed)
    with pytest.raises(SyncStoreError, match="injected atomic seed failure"):
        db.ensure_schema()
    monkeypatch.setattr(db, "execute", original_execute)

    committed_tables = {
        row["name"]
        for row in original_execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).rows
    }
    assert "sync_device_adapter_cursors" not in committed_tables
    assert "sync_device_adapter_domain_acks" not in committed_tables
    assert "sync_device_blob_id_acks" not in committed_tables
    assert original_execute(
        "SELECT COUNT(*) AS count FROM sync_schema_migrations WHERE migration_id = ?",
        ("adapter_cursor_ack_blob_id_v1",),
    ).rows[0]["count"] == 0


def test_adapter_state_migration_serializes_concurrent_sqlite_initializers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "concurrent-adapter-migration.db"
    with ThreadPoolExecutor(max_workers=2) as executor:
        databases = list(executor.map(lambda _index: SyncDatabase(sqlite_path=path), range(2)))

    assert databases[0].execute(
        "SELECT COUNT(*) AS count FROM sync_schema_migrations WHERE migration_id = ? "
        "AND completed_at IS NOT NULL",
        ("adapter_cursor_ack_blob_id_v1",),
    ).rows[0]["count"] == 1


def test_adapter_state_migration_completed_authority_fails_closed_on_catalog_drift(
    sync_store: SyncV2Store,
) -> None:
    sync_store.db.execute(
        "ALTER TABLE sync_schema_migrations ADD COLUMN unexpected_state TEXT"
    )

    with pytest.raises(SyncStoreError, match="catalog is incompatible"):
        sync_store.db.ensure_schema()


def test_adapter_state_migration_fails_closed_on_side_table_type_drift(
    sync_store: SyncV2Store,
) -> None:
    sync_store.db.execute("DROP TABLE sync_device_adapter_cursors")
    sync_store.db.execute(
        """
        CREATE TABLE sync_device_adapter_cursors (
            dataset_id TEXT NOT NULL,
            device_id TEXT NOT NULL,
            domain TEXT NOT NULL,
            adapter_version TEXT NOT NULL,
            last_pulled_sequence INTEGER NOT NULL DEFAULT 0,
            max_delivered_sequence INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(dataset_id, device_id, domain, adapter_version)
        )
        """
    )
    sync_store.db.execute(
        "CREATE INDEX idx_sync_device_adapter_cursors_device "
        "ON sync_device_adapter_cursors(device_id, dataset_id)"
    )

    with pytest.raises(SyncStoreError, match="catalog is incompatible"):
        sync_store.db.ensure_schema()


def test_adapter_state_migration_fails_closed_when_side_checks_are_missing(
    sync_store: SyncV2Store,
) -> None:
    sync_store.db.execute("DROP TABLE sync_device_adapter_cursors")
    sync_store.db.execute(
        """
        CREATE TABLE sync_device_adapter_cursors (
            dataset_id TEXT NOT NULL,
            device_id TEXT NOT NULL,
            domain TEXT NOT NULL,
            adapter_version INTEGER NOT NULL,
            last_pulled_sequence INTEGER NOT NULL DEFAULT 0,
            max_delivered_sequence INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(dataset_id, device_id, domain, adapter_version)
        )
        """
    )
    sync_store.db.execute(
        "CREATE INDEX idx_sync_device_adapter_cursors_device "
        "ON sync_device_adapter_cursors(device_id, dataset_id)"
    )

    with pytest.raises(SyncStoreError, match="catalog is incompatible"):
        sync_store.db.ensure_schema()


def test_adapter_state_migration_postgres_contract_locks_before_exact_catalog_check() -> None:
    source = " ".join(
        inspect.getsource(SyncDatabase._migrate_versioned_device_state).split()
    )

    assert "pg_advisory_xact_lock" in source
    assert source.index("pg_advisory_xact_lock") < source.index(
        "CREATE TABLE IF NOT EXISTS sync_schema_migrations"
    )
    assert "FOR UPDATE" in source
    assert source.index("FOR UPDATE") < source.index("_verify_versioned_device_state_catalog")
    assert source.index("_verify_versioned_device_state_catalog") < source.index(
        "_reconcile_versioned_device_state"
    )
    assert source.index("_reconcile_versioned_device_state") < source.index(
        "UPDATE sync_schema_migrations"
    )


def test_adapter_state_migration_serializes_concurrent_fresh_postgres_authority(
    pg_database_config: DatabaseConfig,
) -> None:
    setup = SyncDatabase(
        backend=DatabaseBackendFactory.create_backend(pg_database_config)
    )
    with setup.backend.transaction() as connection:
        for table_name in (
            "sync_device_adapter_cursors",
            "sync_device_adapter_domain_acks",
            "sync_device_blob_id_acks",
            "sync_schema_migrations",
        ):
            setup.execute(f"DROP TABLE {table_name}", connection=connection)

    with ThreadPoolExecutor(max_workers=2) as executor:
        databases = list(
            executor.map(
                lambda _index: SyncDatabase(
                    backend=DatabaseBackendFactory.create_backend(pg_database_config)
                ),
                range(2),
            )
        )

    assert databases[0].execute(
        "SELECT COUNT(*) AS count FROM sync_schema_migrations "
        "WHERE migration_id = ? AND completed_at IS NOT NULL",
        ("adapter_cursor_ack_blob_id_v1",),
    ).rows[0]["count"] == 1


def test_v2_attachment_history_lookup_is_owner_scoped_indexed_and_bounded(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset(domains=["attachment.ref"]))
    sync_store.insert_envelope(_attachment_v2_envelope())

    assert sync_store.has_attachment_ref_v2_history(
        "dataset-1",
        "11111111-1111-4111-8111-111111111111",
        owner_user_id="user-1",
    )
    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.has_attachment_ref_v2_history(
            "dataset-1",
            "11111111-1111-4111-8111-111111111111",
            owner_user_id="other-user",
        )

    plan = sync_store.db.execute(
        "EXPLAIN QUERY PLAN SELECT 1 FROM sync_envelopes "
        "WHERE dataset_id = ? AND domain = 'attachment.ref' AND entity_id = ? "
        "AND adapter_version = 2 AND status = 'accepted' LIMIT 1",
        ("dataset-1", "11111111-1111-4111-8111-111111111111"),
    ).rows
    assert any("USING INDEX idx_sync_envelopes_dataset_domain" in row["detail"] for row in plan)


def test_blob_binding_page_is_owner_scoped_bounded_and_index_backed(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset(domains=["attachment.ref"]))
    sync_store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-page",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="legacy-provenance-not-authority",
            payload_hash="sha256:" + "a" * 64,
            content_type="application/pdf",
            size_bytes=2048,
            storage_backend="local_fs",
            storage_key="blob-page.bin",
        )
    )
    accepted = sync_store.insert_envelope(_attachment_v2_envelope())
    binding = sync_store.get_attachment_revision_binding(
        accepted.dataset_id,
        accepted.object_id,
        accepted.object_revision or 0,
        owner_user_id="user-1",
    )
    assert binding is not None and binding.resolved_blob_id is not None

    page = sync_store.list_attachment_revision_bindings_for_blob(
        "dataset-1",
        binding.resolved_blob_id,
        owner_user_id="user-1",
        after_establishing_server_cursor=0,
        limit=10_000,
    )
    assert page == [binding]
    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.list_attachment_revision_bindings_for_blob(
            "dataset-1",
            binding.resolved_blob_id,
            owner_user_id="other-user",
        )

    plan = " ".join(
        str(row["detail"])
        for row in sync_store.db.execute(
            "EXPLAIN QUERY PLAN SELECT attachment_id FROM "
            "sync_attachment_revision_bindings WHERE dataset_id = ? "
            "AND resolved_blob_id = ? AND retention_released_at IS NULL "
            "AND establishing_server_cursor > ? "
            "ORDER BY establishing_server_cursor, attachment_id, attachment_revision "
            "LIMIT ?",
            (
                "dataset-1",
                binding.resolved_blob_id,
                0,
                1_000,
            ),
        ).rows
    )
    assert "idx_sync_attachment_bindings_blob_retention" in plan
    assert "USE TEMP B-TREE" not in plan.upper()
    assert "idx_sync_attachment_bindings_blob_retention" in SYNC_POSTGRES_SCHEMA
    assert "idx_sync_attachment_bindings_blob_retention" in inspect.getsource(
        SyncDatabase._ensure_attachment_binding_tables
    )


@pytest.mark.unit
def test_blob_binding_page_uses_compound_keyset_for_shared_cursor(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset(domains=["attachment.ref"]))
    sync_store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-shared-cursor",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="legacy-provenance-not-authority",
            payload_hash="sha256:" + "a" * 64,
            content_type="application/pdf",
            size_bytes=2048,
            storage_backend="local_fs",
            storage_key="blob-shared-cursor.bin",
        )
    )
    bindings = [
        _insert_attachment_binding_for_schema_test(
            sync_store,
            _attachment_binding(
                attachment_id=attachment_id,
                establishing_server_cursor=7,
                availability_at_acceptance="available",
                resolved_blob_id="blob-shared-cursor",
            ),
        )
        for attachment_id in (
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
            "33333333-3333-4333-8333-333333333333",
        )
    ]

    first = sync_store.list_attachment_revision_bindings_for_blob(
        "dataset-1",
        "blob-shared-cursor",
        owner_user_id="user-1",
        limit=2,
    )
    second = sync_store.list_attachment_revision_bindings_for_blob(
        "dataset-1",
        "blob-shared-cursor",
        owner_user_id="user-1",
        after_establishing_server_cursor=first[-1].establishing_server_cursor,
        after_attachment_id=first[-1].attachment_id,
        after_attachment_revision=first[-1].attachment_revision,
        limit=2,
    )

    assert first == bindings[:2]
    assert second == bindings[2:]


def test_available_blob_page_is_keyset_bounded_and_index_backed(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset(domains=["attachment.ref"]))
    for index in range(3):
        sync_store.complete_blob_upload(
            SyncBlobObjectCreate(
                blob_id=f"blob-page-{index}",
                dataset_id="dataset-1",
                owner_user_id="user-1",
                attachment_id=f"legacy-provenance-{index}",
                payload_hash="sha256:" + str(index) * 64,
                content_type="application/octet-stream",
                size_bytes=1,
                storage_backend="local_fs",
                storage_key=f"blob-page-{index}.bin",
            )
        )

    first = sync_store.list_blob_objects_for_dataset_page(
        "dataset-1",
        limit=2,
    )
    second = sync_store.list_blob_objects_for_dataset_page(
        "dataset-1",
        after_updated_at=first[-1].updated_at,
        after_blob_id=first[-1].blob_id,
        limit=2,
    )

    assert len(first) == 2
    assert len(second) == 1
    assert {blob.blob_id for blob in [*first, *second]} == {
        "blob-page-0",
        "blob-page-1",
        "blob-page-2",
    }
    plan = " ".join(
        str(row["detail"])
        for row in sync_store.db.execute(
            "EXPLAIN QUERY PLAN SELECT blob_id FROM sync_blob_objects "
            "WHERE dataset_id = ? AND status = ? "
            "ORDER BY updated_at, blob_id LIMIT ?",
            ("dataset-1", "available", 2),
        ).rows
    )
    assert "idx_sync_blob_objects_retention" in plan
    assert "USE TEMP B-TREE" not in plan.upper()
    assert "idx_sync_blob_objects_retention" in SYNC_POSTGRES_SCHEMA


def test_adapter_cursor_and_version_ack_upgrade_seeds_v1_and_reconciles_maxima(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sync-upgrade.db"
    store = SyncV2Store(SyncDatabase(sqlite_path=path))
    store.upsert_device(_device())
    store.enroll_dataset(_dataset())
    store.db.execute("DROP TABLE sync_device_adapter_cursors")
    store.db.execute("DROP TABLE sync_device_adapter_domain_acks")
    store.db.execute("DROP TABLE sync_device_blob_id_acks")
    store.db.execute(
        "DELETE FROM sync_schema_migrations WHERE migration_id = ?",
        ("adapter_cursor_ack_blob_id_v1",),
    )
    now = utcnow_iso()
    store.db.execute(
        "INSERT INTO sync_device_cursors "
        "(dataset_id, device_id, domain, last_pulled_sequence, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("dataset-1", "device-1", "notes.note", 7, now),
    )
    store.db.execute(
        "INSERT INTO sync_device_domain_acks "
        "(dataset_id, device_id, domain, through_server_sequence, applied_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("dataset-1", "device-1", "notes.note", 6, now, now),
    )

    store.db.ensure_schema()

    cursor = store.get_device_cursor(
        "dataset-1", "device-1", "notes.note", adapter_version=1
    )
    ack = store.get_device_domain_ack(
        "dataset-1", "device-1", "notes.note", adapter_version=1
    )
    assert cursor is not None and cursor.last_pulled_sequence == 7
    assert ack is not None and ack.through_server_sequence == 6
    assert store.db.execute("SELECT COUNT(*) AS count FROM sync_device_blob_id_acks").rows[0][
        "count"
    ] == 0

    store.db.execute(
        "DELETE FROM sync_device_cursors WHERE dataset_id = ? AND device_id = ? AND domain = ?",
        ("dataset-1", "device-1", "notes.note"),
    )
    store.db.execute(
        "DELETE FROM sync_device_domain_acks "
        "WHERE dataset_id = ? AND device_id = ? AND domain = ?",
        ("dataset-1", "device-1", "notes.note"),
    )
    store.db.ensure_schema()
    assert store.db.execute(
        "SELECT last_pulled_sequence FROM sync_device_cursors "
        "WHERE dataset_id = ? AND device_id = ? AND domain = ?",
        ("dataset-1", "device-1", "notes.note"),
    ).rows[0]["last_pulled_sequence"] == 7
    assert store.db.execute(
        "SELECT through_server_sequence FROM sync_device_domain_acks "
        "WHERE dataset_id = ? AND device_id = ? AND domain = ?",
        ("dataset-1", "device-1", "notes.note"),
    ).rows[0]["through_server_sequence"] == 6

    store.db.execute(
        "UPDATE sync_device_cursors SET last_pulled_sequence = 11 "
        "WHERE dataset_id = ? AND device_id = ? AND domain = ?",
        ("dataset-1", "device-1", "notes.note"),
    )
    store.db.execute(
        "UPDATE sync_device_adapter_domain_acks SET through_server_sequence = 10 "
        "WHERE dataset_id = ? AND device_id = ? AND domain = ? AND adapter_version = 1",
        ("dataset-1", "device-1", "notes.note"),
    )
    store.db.ensure_schema()

    legacy_cursor = store.db.execute(
        "SELECT last_pulled_sequence FROM sync_device_cursors "
        "WHERE dataset_id = ? AND device_id = ? AND domain = ?",
        ("dataset-1", "device-1", "notes.note"),
    ).rows[0]
    version_cursor = store.db.execute(
        "SELECT last_pulled_sequence FROM sync_device_adapter_cursors "
        "WHERE dataset_id = ? AND device_id = ? AND domain = ? AND adapter_version = 1",
        ("dataset-1", "device-1", "notes.note"),
    ).rows[0]
    legacy_ack = store.db.execute(
        "SELECT through_server_sequence FROM sync_device_domain_acks "
        "WHERE dataset_id = ? AND device_id = ? AND domain = ?",
        ("dataset-1", "device-1", "notes.note"),
    ).rows[0]
    assert legacy_cursor["last_pulled_sequence"] == 11
    assert version_cursor["last_pulled_sequence"] == 11
    assert legacy_ack["through_server_sequence"] == 10

    store.db.execute(
        "UPDATE sync_device_domain_acks SET through_server_sequence = 11 "
        "WHERE dataset_id = ? AND device_id = ? AND domain = ?",
        ("dataset-1", "device-1", "notes.note"),
    )
    summary = store.list_device_acknowledgments("dataset-1", "device-1")
    assert summary.domain_acks["notes.note"].through_server_sequence == 11
    assert summary.version_acks[0].through_server_sequence == 11


def test_adapter_cursor_and_version_ack_partial_seed_rolls_back(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_store.upsert_device(_device())
    sync_store.enroll_dataset(_dataset())
    sync_store.db.execute(
        "INSERT INTO sync_device_cursors "
        "(dataset_id, device_id, domain, last_pulled_sequence, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("dataset-1", "device-1", "notes.note", 4, utcnow_iso()),
    )
    sync_store.db.execute("DELETE FROM sync_device_adapter_cursors")
    original_execute = sync_store.db.execute

    def fail_second_seed(statement, params=None, *, connection=None):
        if "INSERT INTO sync_device_adapter_domain_acks" in statement:
            raise SyncStoreError("injected version ack seed failure")
        return original_execute(statement, params, connection=connection)

    monkeypatch.setattr(sync_store.db, "execute", fail_second_seed)
    with pytest.raises(SyncStoreError, match="injected version ack seed failure"):
        sync_store.db.ensure_schema()
    monkeypatch.setattr(sync_store.db, "execute", original_execute)

    assert original_execute(
        "SELECT COUNT(*) AS count FROM sync_device_adapter_cursors"
    ).rows[0]["count"] == 0


def test_attachment_binding_schema_has_exact_constraints_and_indexes(
    sync_store: SyncV2Store,
) -> None:
    binding_sql = sync_store.db.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'sync_attachment_revision_bindings'"
    ).rows[0]["sql"]
    namespace_sql = sync_store.db.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'sync_dataset_storage_namespaces'"
    ).rows[0]["sql"]
    binding_indexes = {
        row["name"]
        for row in sync_store.db.execute(
            "PRAGMA index_list(sync_attachment_revision_bindings)"
        ).rows
    }
    namespace_indexes = {
        row["name"]
        for row in sync_store.db.execute(
            "PRAGMA index_list(sync_dataset_storage_namespaces)"
        ).rows
    }

    assert "PRIMARY KEY (dataset_id, attachment_id, attachment_revision)" in binding_sql
    assert "attachment_revision > 0" in binding_sql
    assert "size_bytes > 0" in binding_sql
    assert "availability_at_acceptance IN ('available', 'metadata_only')" in binding_sql
    assert "length(blob_hash) = 71" in binding_sql
    assert "length(storage_namespace_id) = 32" in namespace_sql
    assert {
        "idx_sync_attachment_bindings_unresolved",
        "idx_sync_attachment_bindings_pending_digest",
        "idx_sync_attachment_bindings_blob",
        "idx_sync_attachment_bindings_retention_release",
    }.issubset(binding_indexes)
    assert {
        "uq_sync_dataset_storage_namespace_id",
        "idx_sync_dataset_storage_namespaces_owner",
    }.issubset(namespace_indexes)


@pytest.mark.parametrize(
    ("dataset_id", "attachment_id"),
    [
        ("", "11111111-1111-4111-8111-111111111111"),
        ("dataset-1", "-1111111-1111-4111-8111-111111111111"),
    ],
)
def test_attachment_binding_schema_rejects_noncanonical_raw_identity(
    sync_store: SyncV2Store,
    dataset_id: str,
    attachment_id: str,
) -> None:
    with pytest.raises(BackendDatabaseError):
        sync_store.db.execute(
            """
            INSERT INTO sync_attachment_revision_bindings (
                dataset_id, attachment_id, attachment_revision, blob_hash,
                size_bytes, establishing_server_cursor,
                availability_at_acceptance, resolved_blob_id,
                retention_released_at, created_at
            ) VALUES (?, ?, 1, ?, 1, 1, 'metadata_only', NULL, NULL, ?)
            """,
            (dataset_id, attachment_id, "sha256:" + "a" * 64, utcnow_iso()),
        )


def test_storage_namespace_schema_rejects_null_dataset_authority(
    sync_store: SyncV2Store,
) -> None:
    with pytest.raises(BackendDatabaseError):
        sync_store.db.execute(
            """
            INSERT INTO sync_dataset_storage_namespaces (
                dataset_id, owner_user_id, storage_namespace_id, created_at
            ) VALUES (NULL, ?, ?, ?)
            """,
            ("user-1", "1" * 32, "2026-08-11T21:00:00+00:00"),
        )


def test_storage_namespace_schema_rejects_empty_dataset_authority(
    sync_store: SyncV2Store,
) -> None:
    with pytest.raises(BackendDatabaseError):
        sync_store.db.execute(
            """
            INSERT INTO sync_dataset_storage_namespaces (
                dataset_id, owner_user_id, storage_namespace_id, created_at
            ) VALUES ('', 'user-1', ?, ?)
            """,
            ("1" * 32, utcnow_iso()),
        )


def test_existing_current_schema_additively_ensures_binding_and_namespace_tables(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())
    sync_store.db.execute("DROP TABLE sync_attachment_revision_bindings")
    sync_store.db.execute("DROP TABLE sync_dataset_storage_namespaces")

    sync_store.db.ensure_schema()

    assert sync_store.db.backend.table_exists("sync_attachment_revision_bindings")
    assert sync_store.db.backend.table_exists("sync_dataset_storage_namespaces")
    binding_indexes = {
        row["name"]
        for row in sync_store.db.execute(
            "PRAGMA index_list(sync_attachment_revision_bindings)"
        ).rows
    }
    assert "idx_sync_attachment_bindings_pending_digest" in binding_indexes
    assert sync_store.get_dataset("dataset-1") is not None


@pytest.mark.parametrize(
    ("table_name", "weak_schema"),
    [
        (
            "sync_attachment_revision_bindings",
            """
            CREATE TABLE sync_attachment_revision_bindings (
                dataset_id TEXT NOT NULL,
                attachment_id TEXT NOT NULL,
                attachment_revision INTEGER NOT NULL,
                blob_hash TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                establishing_server_cursor INTEGER NOT NULL,
                availability_at_acceptance TEXT NOT NULL,
                resolved_blob_id TEXT,
                retention_released_at TEXT,
                created_at TEXT NOT NULL
            )
            """,
        ),
        (
            "sync_dataset_storage_namespaces",
            """
            CREATE TABLE sync_dataset_storage_namespaces (
                dataset_id TEXT,
                owner_user_id TEXT,
                storage_namespace_id TEXT,
                created_at TEXT
            )
            """,
        ),
    ],
)
def test_existing_current_schema_rejects_weak_attachment_authority_table(
    sync_store: SyncV2Store,
    table_name: str,
    weak_schema: str,
) -> None:
    sync_store.db.execute(f"DROP TABLE {table_name}")  # nosec B608 - fixed parameters.
    sync_store.db.execute(weak_schema)

    with pytest.raises(SyncStoreError, match="catalog"):
        sync_store.db.ensure_schema()

    stored_sql = sync_store.db.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).rows[0]["sql"]
    assert "CHECK" not in stored_sql


def test_existing_current_schema_rejects_weak_attachment_authority_index(
    sync_store: SyncV2Store,
) -> None:
    sync_store.db.execute("DROP INDEX idx_sync_attachment_bindings_pending_digest")
    sync_store.db.execute(
        "CREATE INDEX idx_sync_attachment_bindings_pending_digest "
        "ON sync_attachment_revision_bindings(dataset_id)"
    )

    with pytest.raises(SyncStoreError, match="catalog"):
        sync_store.db.ensure_schema()


def test_attachment_binding_creation_is_append_transaction_internal() -> None:
    assert not hasattr(SyncDatabase, "create_attachment_revision_binding")
    assert not hasattr(SyncV2Store, "create_attachment_revision_binding")
    append_source = inspect.getsource(SyncDatabase._insert_envelope_in_transaction)
    assert "_create_attachment_binding_for_envelope" in append_source


def test_attachment_binding_create_requires_consistent_acceptance_availability() -> None:
    with pytest.raises(ValueError, match="available.*resolved_blob_id"):
        _attachment_binding(availability_at_acceptance="available")
    with pytest.raises(ValueError, match="metadata_only.*resolved_blob_id"):
        _attachment_binding(
            availability_at_acceptance="metadata_only",
            resolved_blob_id="blob-impossible-at-acceptance",
        )

    create = _attachment_binding()
    stored = SyncAttachmentRevisionBinding(
        dataset_id=create.dataset_id,
        attachment_id=create.attachment_id,
        attachment_revision=create.attachment_revision,
        blob_hash=create.blob_hash,
        size_bytes=create.size_bytes,
        establishing_server_cursor=create.establishing_server_cursor,
        availability_at_acceptance=create.availability_at_acceptance,
        resolved_blob_id="blob-late",
        retention_released_at=None,
        created_at="2026-08-11T21:00:00+00:00",
    )
    assert stored.availability_at_acceptance == "metadata_only"
    assert stored.resolved_blob_id == "blob-late"


def test_attachment_binding_identity_is_immutable_and_pending_resolution_is_exact(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())
    created = _insert_attachment_binding_for_schema_test(
        sync_store, _attachment_binding()
    )
    replay = _insert_attachment_binding_for_schema_test(
        sync_store, _attachment_binding()
    )

    assert replay == created
    assert created.resolved_blob_id is None
    with pytest.raises(SyncIdempotencyConflictError):
        _insert_attachment_binding_for_schema_test(
            sync_store, _attachment_binding(size_bytes=4096)
        )

    matching = sync_store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-match",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="creation-provenance-only",
            payload_hash=created.blob_hash,
            content_type="application/octet-stream",
            size_bytes=created.size_bytes,
            storage_backend="local_fs",
            storage_key="blobs/v2/" + "1" * 32 + "/" + "a" * 64 + ".blob",
        )
    )
    resolved = sync_store.get_attachment_revision_binding(
        created.dataset_id,
        created.attachment_id,
        created.attachment_revision,
        owner_user_id="user-1",
    )
    assert resolved is not None
    assert resolved.resolved_blob_id == matching.blob_id
    assert resolved.availability_at_acceptance == "metadata_only"

    other = sync_store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-other",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="different-provenance",
            payload_hash="sha256:" + "b" * 64,
            content_type="application/octet-stream",
            size_bytes=created.size_bytes,
            storage_backend="local_fs",
            storage_key="blobs/v2/" + "1" * 32 + "/" + "b" * 64 + ".blob",
        )
    )
    with pytest.raises(SyncStoreError, match="binding"):
        sync_store.resolve_attachment_revision_binding(
            created.dataset_id,
            created.attachment_id,
            created.attachment_revision,
            blob_id=other.blob_id,
            owner_user_id="user-1",
        )


def test_public_attachment_binding_methods_deny_wrong_dataset_owner(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_store.enroll_dataset(_dataset())
    accepted = sync_store.insert_envelope(_attachment_v2_envelope())
    binding = sync_store.get_attachment_revision_binding(
        accepted.dataset_id,
        accepted.object_id,
        accepted.object_revision or 0,
        owner_user_id="user-1",
    )
    assert binding is not None
    statements: list[tuple[str, tuple[Any, ...]]] = []
    original_execute = sync_store.db.execute

    def record_execute(statement, params=None, *, connection=None):
        statements.append((" ".join(statement.split()), tuple(params or ())))
        return original_execute(statement, params, connection=connection)

    monkeypatch.setattr(sync_store.db, "execute", record_execute)
    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.get_attachment_revision_binding(
            binding.dataset_id,
            binding.attachment_id,
            binding.attachment_revision,
            owner_user_id="user-2",
        )
    assert "WHERE dataset_id = ? AND owner_user_id = ?" in statements[0][0]
    assert statements[0][1] == (binding.dataset_id, "user-2")
    assert "FOR UPDATE" not in statements[0][0]
    assert all(
        "FROM sync_attachment_revision_bindings" not in statement
        for statement, _params in statements
    )
    statements.clear()
    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.list_unresolved_attachment_revision_bindings(
            binding.dataset_id,
            owner_user_id="user-2",
        )
    assert "WHERE dataset_id = ? AND owner_user_id = ?" in statements[0][0]
    assert statements[0][1] == (binding.dataset_id, "user-2")
    assert "FOR UPDATE" not in statements[0][0]
    statements.clear()
    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.resolve_attachment_revision_binding(
            binding.dataset_id,
            binding.attachment_id,
            binding.attachment_revision,
            blob_id="blob-not-authorized",
            owner_user_id="user-2",
        )
    assert "WHERE dataset_id = ? AND owner_user_id = ?" in statements[0][0]
    assert statements[0][1] == (binding.dataset_id, "user-2")
    statements.clear()
    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.release_attachment_revision_binding(
            binding.dataset_id,
            binding.attachment_id,
            binding.attachment_revision,
            released_at="2026-08-11T21:00:00+00:00",
            owner_user_id="user-2",
        )
    assert "WHERE dataset_id = ? AND owner_user_id = ?" in statements[0][0]
    assert statements[0][1] == (binding.dataset_id, "user-2")


def test_attachment_binding_retention_release_is_monotonic_and_idempotent(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())
    accepted = sync_store.insert_envelope(_attachment_v2_envelope())
    binding = sync_store.get_attachment_revision_binding(
        accepted.dataset_id,
        accepted.object_id,
        accepted.object_revision or 0,
        owner_user_id="user-1",
    )
    assert binding is not None
    released = sync_store.release_attachment_revision_binding(
        binding.dataset_id,
        binding.attachment_id,
        binding.attachment_revision,
        released_at="2026-08-11T21:00:00+00:00",
        owner_user_id="user-1",
    )
    replay = sync_store.release_attachment_revision_binding(
        binding.dataset_id,
        binding.attachment_id,
        binding.attachment_revision,
        released_at="2026-08-12T21:00:00+00:00",
        owner_user_id="user-1",
    )

    assert released.retention_released_at == "2026-08-11T21:00:00+00:00"
    assert replay == released
    assert replay.blob_hash == binding.blob_hash
    assert replay.establishing_server_cursor == binding.establishing_server_cursor


def test_attachment_binding_release_candidate_page_is_bounded_and_indexed(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())
    timestamp = "2026-08-11T21:00:00+00:00"
    rows = [
        (
            "dataset-1",
            f"{index:08x}-1111-4111-8111-{index:012x}",
            1,
            "sha256:" + f"{index:064x}",
            1,
            index,
            "metadata_only",
            timestamp,
        )
        for index in range(1, 1003)
    ]
    with sync_store.db.backend.transaction() as connection:
        for row in rows:
            sync_store.db.execute(
                """
                INSERT INTO sync_attachment_revision_bindings (
                    dataset_id, attachment_id, attachment_revision, blob_hash,
                    size_bytes, establishing_server_cursor,
                    availability_at_acceptance, resolved_blob_id,
                    retention_released_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
                """,
                row,
                connection=connection,
            )

    first = sync_store.list_unreleased_attachment_revision_bindings(
        "dataset-1",
        owner_user_id="user-1",
        limit=10_000,
    )
    second = sync_store.list_unreleased_attachment_revision_bindings(
        "dataset-1",
        owner_user_id="user-1",
        after_establishing_server_cursor=first[-1].establishing_server_cursor,
        after_attachment_id=first[-1].attachment_id,
        after_attachment_revision=first[-1].attachment_revision,
        limit=10_000,
    )
    assert len(first) == 1000
    assert [item.establishing_server_cursor for item in second] == [1001, 1002]

    plan = " ".join(
        str(row["detail"])
        for row in sync_store.db.execute(
            "EXPLAIN QUERY PLAN SELECT attachment_id FROM "
            "sync_attachment_revision_bindings AS binding WHERE binding.dataset_id = ? "
            "AND binding.retention_released_at IS NULL AND NOT EXISTS (SELECT 1 "
            "FROM sync_current_heads AS head JOIN sync_envelopes AS envelope ON "
            "envelope.server_sequence = head.latest_server_cursor WHERE "
            "head.dataset_id = binding.dataset_id AND head.domain = 'attachment.ref' "
            "AND head.object_id = binding.attachment_id AND envelope.adapter_version = 2 "
            "AND envelope.object_revision = "
            "binding.attachment_revision AND envelope.operation <> 'tombstone') AND "
            "(binding.establishing_server_cursor, binding.attachment_id, "
            "binding.attachment_revision) > (?, ?, ?) ORDER BY "
            "binding.establishing_server_cursor, binding.attachment_id, "
            "binding.attachment_revision LIMIT ?",
            ("dataset-1", 0, "", 0, 1000),
        ).rows
    )
    assert "idx_sync_attachment_bindings_retention_release" in plan
    assert "USE TEMP B-TREE" not in plan.upper()


def test_storage_namespace_is_server_issued_owner_scoped_and_stable(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_store.enroll_dataset(_dataset())
    first = sync_store.get_or_create_storage_namespace(
        "dataset-1",
        owner_user_id="user-1",
    )
    replay = sync_store.get_or_create_storage_namespace(
        "dataset-1",
        owner_user_id="user-1",
    )

    assert replay == first
    assert len(first.storage_namespace_id) == 32
    assert first.storage_namespace_id.isascii()
    assert first.storage_namespace_id == first.storage_namespace_id.lower()
    assert all(character in "0123456789abcdef" for character in first.storage_namespace_id)
    statements: list[tuple[str, tuple[Any, ...]]] = []
    original_execute = sync_store.db.execute

    def record_execute(statement, params=None, *, connection=None):
        statements.append((" ".join(statement.split()), tuple(params or ())))
        return original_execute(statement, params, connection=connection)

    monkeypatch.setattr(sync_store.db, "execute", record_execute)
    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.get_or_create_storage_namespace(
            "dataset-1",
            owner_user_id="user-2",
        )
    assert "WHERE dataset_id = ? AND owner_user_id = ?" in statements[0][0]
    assert statements[0][1] == ("dataset-1", "user-2")
    assert all(
        "sync_dataset_storage_namespaces" not in statement
        for statement, _params in statements
    )


def test_legacy_blob_relocation_denies_wrong_owner_before_namespace_or_blob_query(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_store.enroll_dataset(_dataset())
    statements: list[tuple[str, tuple[Any, ...]]] = []
    original_execute = sync_store.db.execute

    def record_execute(statement, params=None, *, connection=None):
        statements.append((" ".join(statement.split()), tuple(params or ())))
        return original_execute(statement, params, connection=connection)

    monkeypatch.setattr(sync_store.db, "execute", record_execute)

    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.relocate_legacy_blob(
            object(),
            dataset_id="dataset-1",
            owner_user_id="user-2",
            blob_id="blob-not-authorized",
        )

    assert "WHERE dataset_id = ? AND owner_user_id = ?" in statements[0][0]
    assert statements[0][1] == ("dataset-1", "user-2")
    assert all(
        "sync_dataset_storage_namespaces" not in statement
        and "sync_blob_objects" not in statement
        for statement, _params in statements
    )


def test_legacy_blob_relocation_cas_updates_storage_and_resolves_matching_binding(
    sync_store: SyncV2Store,
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore

    sync_store.enroll_dataset(_dataset())
    blob_store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"shared legacy bytes"
    payload_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    blob_store.write_upload_chunk(
        upload_id="legacy-upload",
        chunk_index=0,
        payload=payload,
        expected_hash=payload_hash,
    )
    legacy_key = blob_store.commit_upload(
        upload_id="legacy-upload",
        payload_hash=payload_hash,
        chunk_indexes=[0],
    )
    blob = sync_store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-legacy",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="legacy-creation-provenance",
            payload_hash=payload_hash,
            content_type="application/octet-stream",
            size_bytes=len(payload),
            storage_backend="local_fs",
            storage_key=legacy_key,
        )
    )
    envelope = _attachment_v2_envelope(
        payload={
            **(_attachment_v2_envelope().payload or {}),
            "blob_hash": payload_hash,
            "size_bytes": len(payload),
        },
    )
    from tldw_Server_API.app.core.Sync.v2.attachment_refs_v2 import (
        attachment_ref_v2_object_hash,
        parse_attachment_ref_v2_payload,
    )

    parsed = parse_attachment_ref_v2_payload("upsert", envelope.payload or {})
    accepted = sync_store.insert_envelope(
        replace(
            envelope,
            payload_hash=attachment_ref_v2_object_hash(
                "upsert", parsed, object_revision=envelope.object_revision or 0
            ),
        )
    )
    binding = sync_store.get_attachment_revision_binding(
        accepted.dataset_id,
        accepted.object_id,
        accepted.object_revision or 0,
        owner_user_id="user-1",
    )
    assert binding is not None

    relocated = sync_store.relocate_legacy_blob(
        blob_store,
        dataset_id="dataset-1",
        owner_user_id="user-1",
        blob_id=blob.blob_id,
    )
    replay = sync_store.relocate_legacy_blob(
        blob_store,
        dataset_id="dataset-1",
        owner_user_id="user-1",
        blob_id=blob.blob_id,
    )
    resolved = sync_store.get_attachment_revision_binding(
        binding.dataset_id,
        binding.attachment_id,
        binding.attachment_revision,
        owner_user_id="user-1",
    )

    assert replay == relocated
    assert relocated.storage_key.startswith("blobs/v2/")
    assert "dataset-1" not in relocated.storage_key
    assert "legacy-creation-provenance" not in relocated.storage_key
    assert blob_store.read_blob(relocated.storage_key) == payload
    assert blob_store.read_blob(legacy_key) == payload
    assert resolved is not None
    assert resolved.resolved_blob_id == blob.blob_id


def test_legacy_blob_relocation_corrupt_target_does_not_advance_storage_cas(
    sync_store: SyncV2Store,
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.Sync.v2.blob_store import (
        LocalSyncBlobStore,
        SyncBlobStoreError,
    )

    sync_store.enroll_dataset(_dataset())
    blob_store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"shared legacy bytes blocked by partial target"
    payload_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    blob_store.write_upload_chunk(
        upload_id="legacy-partial-upload",
        chunk_index=0,
        payload=payload,
        expected_hash=payload_hash,
    )
    legacy_key = blob_store.commit_upload(
        upload_id="legacy-partial-upload",
        payload_hash=payload_hash,
        chunk_indexes=[0],
    )
    blob = sync_store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-legacy-partial",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="legacy-partial-provenance",
            payload_hash=payload_hash,
            content_type="application/octet-stream",
            size_bytes=len(payload),
            storage_backend="local_fs",
            storage_key=legacy_key,
        )
    )
    namespace = sync_store.get_or_create_storage_namespace(
        "dataset-1",
        owner_user_id="user-1",
    )
    target_key = blob_store.namespace_storage_key(
        namespace.storage_namespace_id,
        payload_hash,
    )
    target = blob_store.root / target_key
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload[:4])

    with pytest.raises(SyncBlobStoreError):
        sync_store.relocate_legacy_blob(
            blob_store,
            dataset_id="dataset-1",
            owner_user_id="user-1",
            blob_id=blob.blob_id,
        )

    stored = sync_store.get_blob_object(
        "dataset-1",
        blob_id=blob.blob_id,
        owner_user_id="user-1",
    )
    assert stored is not None
    assert stored.storage_key == legacy_key
    assert target.read_bytes() == payload[:4]
    assert blob_store.read_blob(legacy_key) == payload


def test_attachment_binding_lookup_and_unresolved_page_are_bounded_and_indexed(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_store.enroll_dataset(_dataset())
    for revision in range(1, 1_006):
        _insert_attachment_binding_for_schema_test(
            sync_store,
            _attachment_binding(
                attachment_revision=revision,
                establishing_server_cursor=revision,
            ),
        )
    current_attachment_id = "33333333-3333-4333-8333-333333333333"
    current = _insert_attachment_binding_for_schema_test(
        sync_store,
        _attachment_binding(
            attachment_id=current_attachment_id,
            attachment_revision=1,
            establishing_server_cursor=2_000,
        ),
    )
    sync_store.db.execute(
        """
        INSERT INTO sync_current_heads (
            dataset_id, domain, object_id, latest_server_cursor
        ) VALUES (?, 'attachment.ref', ?, ?)
        """,
        ("dataset-1", current_attachment_id, current.establishing_server_cursor),
    )

    statements: list[str] = []
    original_execute = sync_store.db.execute

    def counted_execute(statement, params=None, *, connection=None):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(" ".join(statement.split()))
        return original_execute(statement, params, connection=connection)

    monkeypatch.setattr(sync_store.db, "execute", counted_execute)
    detail = sync_store.get_attachment_revision_binding(
        "dataset-1",
        _attachment_binding().attachment_id,
        1,
        owner_user_id="user-1",
    )
    detail_query_count = len(statements)
    statements.clear()
    page = sync_store.list_unresolved_attachment_revision_bindings(
        "dataset-1",
        owner_user_id="user-1",
        after_establishing_server_cursor=0,
        limit=10_000,
    )
    page_query_count = len(statements)

    assert detail is not None
    assert detail_query_count == 2
    assert len(page) == 1_000
    assert page_query_count == 2
    plan = " ".join(
        str(row["detail"])
        for row in original_execute(
            "EXPLAIN QUERY PLAN SELECT attachment_id FROM "
            "sync_attachment_revision_bindings WHERE dataset_id = ? "
            "AND resolved_blob_id IS NULL AND retention_released_at IS NULL "
            "AND establishing_server_cursor > ? "
            "ORDER BY establishing_server_cursor, attachment_id, attachment_revision LIMIT ?",
            ("dataset-1", 0, 1000),
        ).rows
    )
    assert any(
        index_name in plan
        for index_name in (
            "idx_sync_attachment_bindings_unresolved",
            "idx_sync_attachment_bindings_blob_retention",
        )
    )
    assert "USE TEMP B-TREE" not in plan.upper()
    digest_plan = " ".join(
        str(row["detail"])
        for row in original_execute(
            "EXPLAIN QUERY PLAN SELECT attachment_id FROM "
            "sync_attachment_revision_bindings WHERE dataset_id = ? AND blob_hash = ? "
            "AND size_bytes = ? AND resolved_blob_id IS NULL "
            "AND retention_released_at IS NULL "
            "ORDER BY establishing_server_cursor, attachment_id, attachment_revision LIMIT 1000",
            ("dataset-1", current.blob_hash, current.size_bytes),
        ).rows
    )
    assert "idx_sync_attachment_bindings_pending_digest" in digest_plan
    assert "USE TEMP B-TREE" not in digest_plan.upper()

    blob = sync_store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-auto-resolve",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="legacy-provenance-must-not-select",
            payload_hash=current.blob_hash,
            content_type="application/octet-stream",
            size_bytes=current.size_bytes,
            storage_backend="local_fs",
            storage_key="blobs/v2/" + "8" * 32 + "/" + "a" * 64 + ".blob",
        )
    )
    resolved_current = sync_store.get_attachment_revision_binding(
        "dataset-1",
        current_attachment_id,
        1,
        owner_user_id="user-1",
    )
    resolution_counts = original_execute(
        """
        SELECT
            SUM(CASE WHEN resolved_blob_id = ? THEN 1 ELSE 0 END) AS resolved_count,
            SUM(CASE WHEN resolved_blob_id IS NULL THEN 1 ELSE 0 END) AS unresolved_count
        FROM sync_attachment_revision_bindings
        WHERE dataset_id = ?
        """,
        (blob.blob_id, "dataset-1"),
    ).rows[0]
    assert resolved_current is not None
    assert resolved_current.resolved_blob_id == blob.blob_id
    assert int(resolution_counts["resolved_count"]) == 1_001
    assert int(resolution_counts["unresolved_count"]) == 5


def test_blob_completion_bounds_current_and_historical_binding_repair_pages(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_store.enroll_dataset(_dataset())
    payload_hash = "sha256:" + "d" * 64
    with sync_store.db.backend.transaction() as connection:
        for index in range(1, 1_006):
            attachment_id = f"{index:08x}-0000-4000-8000-{index:012x}"
            sync_store.db._create_attachment_revision_binding(
                _attachment_binding(
                    attachment_id=attachment_id,
                    blob_hash=payload_hash,
                    size_bytes=17,
                    establishing_server_cursor=index,
                ),
                connection=connection,
            )
            sync_store.db.execute(
                """
                INSERT INTO sync_current_heads (
                    dataset_id, domain, object_id, latest_server_cursor
                ) VALUES (?, 'attachment.ref', ?, ?)
                """,
                ("dataset-1", attachment_id, index),
                connection=connection,
            )
        historical_id = "ffffffff-ffff-4fff-8fff-ffffffffffff"
        for revision in range(1, 1_006):
            sync_store.db._create_attachment_revision_binding(
                _attachment_binding(
                    attachment_id=historical_id,
                    attachment_revision=revision,
                    blob_hash=payload_hash,
                    size_bytes=17,
                    establishing_server_cursor=2_000 + revision,
                ),
                connection=connection,
            )

    selected_page_sizes: list[int] = []
    original_execute = sync_store.db.execute

    def record_resolution_pages(statement, params=None, *, connection=None):
        result = original_execute(statement, params, connection=connection)
        compact = " ".join(statement.split())
        if (
            compact.startswith("SELECT dataset_id, attachment_id, attachment_revision")
            and "sync_attachment_revision_bindings" in compact
        ):
            selected_page_sizes.append(len(result.rows))
        return result

    monkeypatch.setattr(sync_store.db, "execute", record_resolution_pages)
    blob_create = SyncBlobObjectCreate(
        blob_id="blob-bounded-repair",
        dataset_id="dataset-1",
        owner_user_id="user-1",
        attachment_id="legacy-provenance-is-irrelevant",
        payload_hash=payload_hash,
        content_type="application/octet-stream",
        size_bytes=17,
        storage_backend="local_fs",
        storage_key="blobs/v2/" + "d" * 32 + "/" + "d" * 64 + ".blob",
    )

    sync_store.complete_blob_upload(blob_create)
    first_counts = original_execute(
        """
        SELECT COUNT(*) AS resolved
          FROM sync_attachment_revision_bindings
         WHERE resolved_blob_id = ?
        """,
        (blob_create.blob_id,),
    ).rows[0]
    assert selected_page_sizes == [1_000, 1_000]
    assert int(first_counts["resolved"]) == 2_000

    selected_page_sizes.clear()
    sync_store.complete_blob_upload(blob_create)
    final_counts = original_execute(
        """
        SELECT COUNT(*) AS unresolved
          FROM sync_attachment_revision_bindings
         WHERE resolved_blob_id IS NULL
        """
    ).rows[0]
    assert selected_page_sizes == [5, 5]
    assert int(final_counts["unresolved"]) == 0


def test_blob_owner_drift_never_satisfies_attachment_binding_resolution(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())
    envelope = _attachment_v2_envelope()
    blob_create = SyncBlobObjectCreate(
        blob_id="blob-owner-drift",
        dataset_id="dataset-1",
        owner_user_id="user-1",
        attachment_id="creation-provenance-is-irrelevant",
        payload_hash=str((envelope.payload or {})["blob_hash"]),
        content_type="application/pdf",
        size_bytes=int((envelope.payload or {})["size_bytes"]),
        storage_backend="local_fs",
        storage_key="blobs/v2/" + "e" * 32 + "/" + "a" * 64 + ".blob",
    )
    sync_store.complete_blob_upload(blob_create)
    sync_store.db.execute(
        "UPDATE sync_blob_objects SET owner_user_id = ? WHERE blob_id = ?",
        ("user-2", blob_create.blob_id),
    )

    accepted = sync_store.insert_envelope(envelope)
    binding = sync_store.get_attachment_revision_binding(
        accepted.dataset_id,
        accepted.object_id,
        accepted.object_revision or 0,
        owner_user_id="user-1",
    )
    assert binding is not None
    assert binding.availability_at_acceptance == "metadata_only"
    assert binding.resolved_blob_id is None

    with pytest.raises(SyncStoreError, match="exact available blob"):
        sync_store.resolve_attachment_revision_binding(
            binding.dataset_id,
            binding.attachment_id,
            binding.attachment_revision,
            blob_id=blob_create.blob_id,
            owner_user_id="user-1",
        )
    with pytest.raises(SyncStoreError, match="owner authority"):
        sync_store.complete_blob_upload(blob_create)
    replayed = sync_store.get_attachment_revision_binding(
        binding.dataset_id,
        binding.attachment_id,
        binding.attachment_revision,
        owner_user_id="user-1",
    )
    assert replayed is not None and replayed.resolved_blob_id is None


def test_attachment_binding_acceptance_observation_does_not_change_envelope_identity(
    tmp_path: Path,
) -> None:
    stores = [
        SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "present.db")),
        SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "absent.db")),
    ]
    envelope = _attachment_v2_envelope()
    for store in stores:
        store.enroll_dataset(_dataset())
    stores[0].complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-present",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="unrelated-creation-provenance",
            payload_hash=envelope.payload["blob_hash"],
            content_type=envelope.payload["content_type"],
            size_bytes=envelope.payload["size_bytes"],
            storage_backend="local_fs",
            storage_key="blobs/v2/" + "1" * 32 + "/" + "a" * 64 + ".blob",
        )
    )

    accepted = [store.insert_envelope(envelope) for store in stores]
    replayed = [store.insert_envelope(envelope) for store in stores]
    bindings = [
        store.get_attachment_revision_binding(
            "dataset-1",
            envelope.object_id,
            envelope.object_revision or 0,
            owner_user_id="user-1",
        )
        for store in stores
    ]

    assert envelope.payload == accepted[0].payload == accepted[1].payload
    assert accepted[0].payload_hash == accepted[1].payload_hash == envelope.payload_hash
    assert _envelope_fingerprint_from_create(envelope) == _envelope_fingerprint_from_row(
        stores[0].db.execute(
            "SELECT * FROM sync_envelopes WHERE dataset_id = ? AND client_envelope_id = ?",
            ("dataset-1", envelope.client_envelope_id),
        ).rows[0]
    )
    assert _envelope_fingerprint_from_create(envelope) == _envelope_fingerprint_from_row(
        stores[1].db.execute(
            "SELECT * FROM sync_envelopes WHERE dataset_id = ? AND client_envelope_id = ?",
            ("dataset-1", envelope.client_envelope_id),
        ).rows[0]
    )
    assert [item.client_envelope_id for item in accepted] == [
        envelope.client_envelope_id,
        envelope.client_envelope_id,
    ]
    assert [item.server_sequence for item in accepted] == [1, 1]
    assert replayed == accepted
    assert bindings[0] is not None and bindings[1] is not None
    assert bindings[0].availability_at_acceptance == "available"
    assert bindings[0].resolved_blob_id == "blob-present"
    assert bindings[1].availability_at_acceptance == "metadata_only"
    assert bindings[1].resolved_blob_id is None

    absent_store = stores[1]
    absent_store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-late",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="different-provenance",
            payload_hash=envelope.payload["blob_hash"],
            content_type=envelope.payload["content_type"],
            size_bytes=envelope.payload["size_bytes"],
            storage_backend="local_fs",
            storage_key="blobs/v2/" + "2" * 32 + "/" + "a" * 64 + ".blob",
        )
    )
    resolved = absent_store.get_attachment_revision_binding(
        "dataset-1",
        envelope.object_id,
        envelope.object_revision or 0,
        owner_user_id="user-1",
    )
    assert resolved is not None
    assert resolved.availability_at_acceptance == "metadata_only"
    assert resolved.resolved_blob_id == "blob-late"
    assert absent_store.insert_envelope(envelope) == accepted[1]


def test_attachment_binding_creation_failure_rolls_back_envelope_acceptance(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_store.enroll_dataset(_dataset())

    def fail_binding(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected binding failure")

    monkeypatch.setattr(
        sync_store.db,
        "_create_attachment_binding_for_envelope",
        fail_binding,
    )

    with pytest.raises(RuntimeError, match="injected binding failure"):
        sync_store.insert_envelope(_attachment_v2_envelope())

    assert sync_store.list_envelopes_after("dataset-1", 0) == []
    assert (
        sync_store.get_attachment_revision_binding(
            "dataset-1",
            "11111111-1111-4111-8111-111111111111",
            1,
            owner_user_id="user-1",
        )
        is None
    )


def test_background_policy_and_lease_lifecycle(sync_store: SyncV2Store):
    sync_store.upsert_device(_device())
    sync_store.enroll_dataset(_dataset())

    assert sync_store.get_background_policy("dataset-1", "device-1") is None

    policy = sync_store.upsert_background_policy(
        SyncBackgroundPolicyUpsert(
            dataset_id="dataset-1",
            device_id="device-1",
            enabled=False,
            minimum_interval_seconds=900,
            backoff_floor_seconds=120,
            max_batch_size=25,
            max_blob_bytes_per_run=4096,
            respect_metered_networks=False,
            maintenance_window={"start": "01:00", "end": "03:00"},
            paused_reason="user_paused",
            pending_local_changes=True,
        )
    )
    retry = sync_store.upsert_background_policy(
        SyncBackgroundPolicyUpsert(
            dataset_id="dataset-1",
            device_id="device-1",
            enabled=False,
            minimum_interval_seconds=900,
            backoff_floor_seconds=120,
            max_batch_size=25,
            max_blob_bytes_per_run=4096,
            respect_metered_networks=False,
            maintenance_window={"start": "01:00", "end": "03:00"},
            paused_reason="user_paused",
            pending_local_changes=True,
        )
    )

    first_lease = sync_store.acquire_background_lease(
        SyncBackgroundLeaseCreate(
            dataset_id="dataset-1",
            device_id="device-1",
            lease_id="lease-1",
            ttl_seconds=120,
            requested_at="2026-05-23T18:00:00+00:00",
        )
    )
    refreshed = sync_store.acquire_background_lease(
        SyncBackgroundLeaseCreate(
            dataset_id="dataset-1",
            device_id="device-1",
            lease_id="lease-1",
            ttl_seconds=180,
            requested_at="2026-05-23T18:01:00+00:00",
        )
    )
    held = sync_store.acquire_background_lease(
        SyncBackgroundLeaseCreate(
            dataset_id="dataset-1",
            device_id="device-1",
            lease_id="lease-2",
            ttl_seconds=120,
            requested_at="2026-05-23T18:02:00+00:00",
        )
    )
    after_expiry = sync_store.acquire_background_lease(
        SyncBackgroundLeaseCreate(
            dataset_id="dataset-1",
            device_id="device-1",
            lease_id="lease-2",
            ttl_seconds=120,
            requested_at="2026-05-23T18:10:00+00:00",
        )
    )

    assert policy.enabled is False
    assert policy.pending_local_changes is True
    assert retry.updated_at >= policy.updated_at
    assert first_lease.status == "acquired"
    assert first_lease.acquired is True
    assert refreshed.status == "refreshed"
    assert refreshed.lease_id == "lease-1"
    assert held.status == "held_by_other"
    assert held.acquired is False
    assert held.lease_id == "lease-1"
    assert after_expiry.status == "acquired"
    assert after_expiry.lease_id == "lease-2"

    with pytest.raises(SyncStoreError, match="not registered"):
        sync_store.upsert_background_policy(
            SyncBackgroundPolicyUpsert(
                dataset_id="dataset-1",
                device_id="missing-device",
                enabled=True,
            )
        )


def test_blob_upload_sessions_are_idempotent_and_release_reserved_quota(
    sync_store: SyncV2Store,
):
    sync_store.upsert_device(_device())
    sync_store.enroll_dataset(_dataset())

    session = sync_store.create_blob_upload_session(
        SyncBlobUploadSessionCreate(
            upload_id="upload-1",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            device_id="device-1",
            attachment_id="attachment-1",
            domain="attachment.ref",
            object_id="attachment-1",
            content_type="application/octet-stream",
            size_bytes=2048,
            payload_hash="sha256:" + "a" * 64,
            chunk_size=1024,
            chunk_count=2,
            reserved_quota_bytes=2048,
            idempotency_key="same-upload",
        )
    )
    retry = sync_store.create_blob_upload_session(
        SyncBlobUploadSessionCreate(
            upload_id="upload-retry",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            device_id="device-1",
            attachment_id="attachment-1",
            domain="attachment.ref",
            object_id="attachment-1",
            content_type="application/octet-stream",
            size_bytes=2048,
            payload_hash="sha256:" + "a" * 64,
            chunk_size=1024,
            chunk_count=2,
            reserved_quota_bytes=2048,
            idempotency_key="same-upload",
        )
    )

    assert retry.upload_id == session.upload_id
    assert retry.missing_chunks == [0, 1]
    assert sync_store.summarize_blob_quota("user-1").reserved_blob_bytes == 2048

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.create_blob_upload_session(
            SyncBlobUploadSessionCreate(
                upload_id="upload-drift",
                dataset_id="dataset-1",
                owner_user_id="user-1",
                device_id="device-1",
                attachment_id="attachment-1",
                domain="attachment.ref",
                object_id="attachment-1",
                content_type="application/octet-stream",
                size_bytes=2048,
                payload_hash="sha256:" + "b" * 64,
                chunk_size=1024,
                chunk_count=2,
                reserved_quota_bytes=2048,
                idempotency_key="same-upload",
            )
        )

    cancelled = sync_store.cancel_blob_upload_session("upload-1", dataset_id="dataset-1")

    assert cancelled.status == "cancelled"
    assert sync_store.summarize_blob_quota("user-1").reserved_blob_bytes == 0


def test_blob_chunks_and_completion_validate_idempotency_and_dedupe(
    sync_store: SyncV2Store,
):
    sync_store.upsert_device(_device())
    sync_store.enroll_dataset(_dataset())
    payload_hash = "sha256:" + "a" * 64

    sync_store.create_blob_upload_session(
        SyncBlobUploadSessionCreate(
            upload_id="upload-1",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            device_id="device-1",
            attachment_id="attachment-1",
            domain="attachment.ref",
            object_id="attachment-1",
            content_type="application/octet-stream",
            size_bytes=2048,
            payload_hash=payload_hash,
            chunk_size=1024,
            chunk_count=2,
            reserved_quota_bytes=2048,
        )
    )
    first_chunk = sync_store.record_blob_chunk(
        SyncBlobChunkCreate(
            upload_id="upload-1",
            dataset_id="dataset-1",
            chunk_index=0,
            offset_bytes=0,
            size_bytes=1024,
            chunk_hash="sha256:" + "1" * 64,
            storage_key="uploads/upload-1/0.part",
        )
    )
    retry_chunk = sync_store.record_blob_chunk(
        SyncBlobChunkCreate(
            upload_id="upload-1",
            dataset_id="dataset-1",
            chunk_index=0,
            offset_bytes=0,
            size_bytes=1024,
            chunk_hash="sha256:" + "1" * 64,
            storage_key="uploads/upload-1/0.part",
        )
    )

    assert first_chunk.chunk_hash == retry_chunk.chunk_hash
    assert sync_store.get_blob_upload_session("upload-1").missing_chunks == [1]

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.record_blob_chunk(
            SyncBlobChunkCreate(
                upload_id="upload-1",
                dataset_id="dataset-1",
                chunk_index=0,
                offset_bytes=0,
                size_bytes=1024,
                chunk_hash="sha256:" + "2" * 64,
                storage_key="uploads/upload-1/0.part",
            )
        )

    sync_store.record_blob_chunk(
        SyncBlobChunkCreate(
            upload_id="upload-1",
            dataset_id="dataset-1",
            chunk_index=1,
            offset_bytes=1024,
            size_bytes=1024,
            chunk_hash="sha256:" + "3" * 64,
            storage_key="uploads/upload-1/1.part",
        )
    )
    blob = sync_store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-1",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="attachment-1",
            payload_hash=payload_hash,
            content_type="application/octet-stream",
            size_bytes=2048,
            storage_backend="local_fs",
            storage_key="blobs/sha256/aa/blob.bin",
        )
    )
    duplicate = sync_store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-duplicate",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="attachment-2",
            payload_hash=payload_hash,
            content_type="application/octet-stream",
            size_bytes=2048,
            storage_backend="local_fs",
            storage_key="blobs/sha256/aa/blob.bin",
        )
    )

    quota = sync_store.summarize_blob_quota("user-1")

    assert blob.status == "available"
    assert duplicate.blob_id == blob.blob_id
    assert quota.used_blob_bytes == 2048
    assert quota.reserved_blob_bytes == 0


def test_sync_envelope_mutation_group_schema_contains_m1_columns_and_indexes(
    sync_store: SyncV2Store,
):
    envelope_columns = {
        column["name"]
        for column in sync_store.db.backend.get_table_info("sync_envelopes")
    }
    object_state_columns = {
        column["name"]
        for column in sync_store.db.backend.get_table_info("sync_object_state")
    }
    indexes = {
        row["name"]
        for row in sync_store.db.execute("PRAGMA index_list(sync_envelopes)").rows
    }
    conflict_indexes = {
        row["name"]
        for row in sync_store.db.execute("PRAGMA index_list(sync_conflicts)").rows
    }
    history_index_columns = [
        row["name"]
        for row in sync_store.db.execute(
            "PRAGMA index_info(idx_sync_envelopes_dataset_domain_entity_status_sequence)"
        ).rows
    ]

    assert {
        "client_sequence",
        "base_server_cursor",
        "base_object_revision",
        "base_object_hash",
        "object_revision",
        "parent_id",
        "schema_version",
        "payload_json",
        "payload_hash",
        "created_at_client",
        "received_at_server",
        "deleted",
        "encryption_metadata_json",
        "apply_status",
        "apply_error_code",
        "apply_error_message",
        "applied_at",
        "client_profile_id",
        "mutation_group_id",
        "mutation_step",
        "mutation_step_count",
        "mutation_plan_hash",
    }.issubset(envelope_columns)
    assert {
        "dataset_id",
        "domain",
        "object_id",
        "object_revision",
        "object_hash",
        "latest_server_cursor",
        "deleted",
        "updated_at",
    }.issubset(object_state_columns)
    assert {
        "idx_sync_envelopes_dataset_sequence",
        "idx_sync_envelopes_dataset_domain_object",
        "idx_sync_envelopes_dataset_device_client_sequence",
        "idx_sync_envelopes_payload_hash",
        "idx_sync_envelopes_failed_apply",
        "idx_sync_envelopes_outstanding_apply",
        "uq_sync_envelopes_dataset_mutation_group_step",
        "idx_sync_envelopes_dataset_mutation_group_step",
        "idx_sync_envelopes_dataset_domain_entity_status_sequence",
    }.issubset(indexes)
    assert "uq_sync_conflicts_dataset_envelope_cursor" in conflict_indexes
    assert history_index_columns == [
        "dataset_id",
        "domain",
        "entity_id",
        "status",
        "server_sequence",
    ]


def _insert_legacy_duplicate_conflict(
    sync_store: SyncV2Store,
    *,
    conflict_id: str,
) -> None:
    sync_store.db.execute(
        """
        INSERT INTO sync_conflicts (
            conflict_id, dataset_id, domain, entity_id, conflict_type, status,
            base_envelope_id, local_envelope_id, remote_envelope_id,
            server_sequence, metadata_json, resolved_by_envelope_id,
            resolved_by_device_id, resolution_action, resolution_notes,
            created_at, resolved_at
        )
        SELECT ?, dataset_id, domain, entity_id, conflict_type, status,
               base_envelope_id, local_envelope_id, remote_envelope_id,
               server_sequence, metadata_json, resolved_by_envelope_id,
               resolved_by_device_id, resolution_action, resolution_notes,
               created_at, resolved_at
          FROM sync_conflicts
         WHERE conflict_id = 'conflict-1'
        """,
        (conflict_id,),
    )


def test_sync_database_dedupes_compatible_legacy_conflicts_before_unique_index(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())
    sync_store.insert_conflict(_conflict())
    sync_store.db.execute("DROP INDEX uq_sync_conflicts_dataset_envelope_cursor")
    _insert_legacy_duplicate_conflict(sync_store, conflict_id="conflict-duplicate")

    sync_store.db.ensure_schema()

    conflicts = sync_store.list_conflicts("dataset-1")
    indexes = {
        row["name"]
        for row in sync_store.db.execute("PRAGMA index_list(sync_conflicts)").rows
    }
    assert [conflict.conflict_id for conflict in conflicts] == ["conflict-1"]
    assert "uq_sync_conflicts_dataset_envelope_cursor" in indexes


def test_sync_database_rejects_incompatible_legacy_conflict_duplicates(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())
    sync_store.insert_conflict(_conflict())
    sync_store.db.execute("DROP INDEX uq_sync_conflicts_dataset_envelope_cursor")
    _insert_legacy_duplicate_conflict(sync_store, conflict_id="conflict-divergent")
    sync_store.db.execute(
        """
        UPDATE sync_conflicts
           SET status = 'resolved',
               resolution_action = 'skip',
               resolved_by_device_id = 'device-2',
               resolved_at = '2026-05-10T13:00:00+00:00'
         WHERE conflict_id = 'conflict-divergent'
        """
    )

    with pytest.raises(SyncStoreError, match="incompatible legacy duplicates"):
        sync_store.db.ensure_schema()

    assert {
        conflict.conflict_id for conflict in sync_store.list_conflicts("dataset-1")
    } == {"conflict-1", "conflict-divergent"}


def test_existing_sqlite_conflict_identity_index_skips_legacy_scan(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_scan(*, connection):
        raise AssertionError("existing unique index must skip the legacy duplicate scan")

    monkeypatch.setattr(
        sync_store.db,
        "_dedupe_legacy_conflict_identities",
        fail_scan,
    )
    with sync_store.db.backend.transaction() as connection:
        sync_store.db._ensure_conflict_indexes(connection=connection)


def test_sync_database_migrates_pre_m1_sqlite_schema_for_mutation_groups_before_index_creation(
    tmp_path: Path,
):
    db_path = tmp_path / "pre_m1_sync_v2.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_envelopes (
                server_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                stable_key TEXT,
                operation TEXT NOT NULL,
                client_envelope_id TEXT NOT NULL,
                device_id TEXT,
                client_timestamp TEXT,
                server_timestamp TEXT NOT NULL,
                base_version TEXT,
                entity_version TEXT,
                dependency_json TEXT NOT NULL DEFAULT '[]',
                routing_metadata_json TEXT NOT NULL DEFAULT '{}',
                payload_ciphertext TEXT,
                payload_clear_json TEXT NOT NULL DEFAULT '{}',
                payload_hash TEXT,
                payload_size_bytes INTEGER,
                adapter_version INTEGER NOT NULL,
                status TEXT NOT NULL,
                UNIQUE (dataset_id, client_envelope_id)
            );
            """
        )

    db = SyncDatabase(sqlite_path=db_path)
    envelope_columns = {
        column["name"]
        for column in db.backend.get_table_info("sync_envelopes")
    }
    indexes = {
        row["name"]
        for row in db.execute("PRAGMA index_list(sync_envelopes)").rows
    }

    assert "client_sequence" in envelope_columns
    assert "apply_status" in envelope_columns
    assert "mutation_group_id" in envelope_columns
    assert "mutation_step" in envelope_columns
    assert "mutation_step_count" in envelope_columns
    assert "mutation_plan_hash" in envelope_columns
    assert "idx_sync_envelopes_dataset_device_client_sequence" in indexes
    assert "idx_sync_envelopes_failed_apply" in indexes
    assert "idx_sync_envelopes_outstanding_apply" in indexes
    assert "uq_sync_envelopes_dataset_mutation_group_step" in indexes
    assert "idx_sync_envelopes_dataset_mutation_group_step" in indexes
    assert "idx_sync_envelopes_dataset_domain_entity_status_sequence" in indexes


def test_sync_timestamps_are_timezone_aware_utc():
    timestamp = utcnow_iso()

    parsed = datetime.fromisoformat(timestamp)

    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)
    assert parsed.tzinfo == timezone.utc


def test_sync_store_facade_does_not_embed_sql_statements():
    source = inspect.getsource(store_module.SyncV2Store)

    assert "SELECT " not in source
    assert "INSERT " not in source
    assert "UPDATE " not in source
    assert "DELETE " not in source
    assert "CREATE TABLE" not in source
    assert "ALTER TABLE" not in source


def test_core_models_import_without_api_schema_module():
    code = """
import sys
from tldw_Server_API.app.core.Sync.v2.models import SyncKeyRecordCreate

assert "tldw_Server_API.app.api.v1.schemas.sync_v2_models" not in sys.modules
record = SyncKeyRecordCreate(
    key_record_id="key-1",
    dataset_id="dataset-1",
    user_id="user-1",
    key_purpose="dataset_recovery",
    wrapped_key_blob="wrapped:opaque",
)
assert record.user_id == "user-1"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_device_upsert_is_idempotent(sync_store: SyncV2Store):
    first = sync_store.upsert_device(_device())
    second = sync_store.upsert_device(
        _device(
            display_name="Renamed Laptop",
            capabilities={"domains": ["notes.note", "chat.conversation"]},
        )
    )

    assert second.device_id == first.device_id
    assert second.registered_at == first.registered_at
    assert second.display_name == "Renamed Laptop"
    assert second.capabilities == {"domains": ["notes.note", "chat.conversation"]}
    assert second.last_seen_at >= first.last_seen_at


def test_postgres_device_upsert_locks_existing_row_before_update() -> None:
    backend = _PostgresDeviceLockBackend()
    db = SyncDatabase.__new__(SyncDatabase)
    db.backend = cast(Any, backend)

    updated = db.upsert_device(
        _device(capabilities={"domains": ["notes.note", "attachment.ref"]})
    )

    assert updated.capabilities == {
        "domains": ["notes.note", "attachment.ref"]
    }
    first_select = next(
        statement
        for statement, _params, _connection in backend.calls
        if statement.startswith("SELECT * FROM sync_devices")
    )
    assert first_select.endswith("FOR UPDATE")


def test_postgres_personal_context_receipt_locks_binding_before_upsert() -> None:
    """The receipt CAS holds the dataset lock through its durable upsert."""

    backend = _PostgresPersonalContextReceiptBackend()
    db = SyncDatabase.__new__(SyncDatabase)
    db.backend = cast(Any, backend)

    db.complete_personal_context_link_receipt(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        profile_id="profile-1",
        integrity_key_id="personal-context-integrity-v1",
        purge_generation=0,
        bootstrap_cursor="bootstrap-a",
    )

    statements = [statement for statement, _params, _connection in backend.calls]
    lock_index = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("SELECT * FROM sync_datasets")
    )
    upsert_index = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("INSERT INTO sync_personal_context_link_receipts")
    )
    assert statements[lock_index].endswith("FOR UPDATE")
    assert lock_index < upsert_index
    assert len({connection for _statement, _params, connection in backend.calls}) == 1


def test_postgres_personal_context_receipt_rejects_transition_observed_under_lock() -> None:
    """A binding changed before lock acquisition cannot receive a stale receipt."""

    backend = _PostgresPersonalContextReceiptBackend(stale_binding=True)
    db = SyncDatabase.__new__(SyncDatabase)
    db.backend = cast(Any, backend)

    with pytest.raises(SyncStoreError, match="personal_context_link_binding_stale"):
        db.complete_personal_context_link_receipt(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            profile_id="profile-1",
            integrity_key_id="personal-context-integrity-v1",
            purge_generation=0,
            bootstrap_cursor="bootstrap-a",
        )

    assert not any(
        statement.startswith("INSERT INTO sync_personal_context_link_receipts")
        for statement, _params, _connection in backend.calls
    )


def test_device_upsert_rejects_cross_user_takeover(sync_store: SyncV2Store):
    sync_store.upsert_device(_device(device_id="device-shared", user_id="user-1"))

    with pytest.raises(SyncStoreError):
        sync_store.upsert_device(_device(device_id="device-shared", user_id="user-2"))

    assert [device.user_id for device in sync_store.list_devices_for_user("user-1")] == ["user-1"]
    assert sync_store.list_devices_for_user("user-2") == []


def test_device_lifecycle_status_authorization_and_acknowledgments(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    pending = sync_store.upsert_device(
        _device(
            status="pending_authorization",
            user_label="new laptop",
            authorized_at=None,
        )
    )

    assert pending.status == "pending_authorization"
    assert pending.user_label == "new laptop"
    assert pending.authorized_at is None

    authorization = sync_store.create_device_authorization(
        SyncDeviceAuthorizationCreate(
            authorization_id="auth-1",
            dataset_id="dataset-1",
            user_id="user-1",
            device_id="device-1",
            authorization_method="existing_device",
            idempotency_key="authorize-device-1",
        )
    )
    retry = sync_store.create_device_authorization(
        SyncDeviceAuthorizationCreate(
            authorization_id="auth-retry",
            dataset_id="dataset-1",
            user_id="user-1",
            device_id="device-1",
            authorization_method="existing_device",
            idempotency_key="authorize-device-1",
        )
    )

    assert retry.authorization_id == authorization.authorization_id
    assert authorization.status == "pending"

    approved = sync_store.approve_device_authorization(
        authorization.authorization_id,
        user_id="user-1",
        dataset_id="dataset-1",
        approving_device_id="device-1",
        idempotency_key="approve-device-1",
    )
    active = sync_store.get_device("user-1", "device-1")

    assert approved.status == "approved"
    assert approved.approving_device_id == "device-1"
    assert active is not None
    assert active.status == "active"
    assert active.authorized_at is not None

    sync_store.update_device_cursor(
        SyncDeviceCursor(
            dataset_id="dataset-1",
            device_id="device-1",
            domain="notes.note",
            last_pulled_sequence=3,
            max_delivered_sequence=3,
        )
    )
    domain_ack = sync_store.upsert_device_domain_ack(
        SyncDeviceDomainAckCreate(
            dataset_id="dataset-1",
            device_id="device-1",
            domain="notes.note",
            through_server_sequence=3,
            applied_at="2026-05-23T18:30:00+00:00",
            idempotency_key="domain-ack-3",
        )
    )
    stale_domain_ack = sync_store.upsert_device_domain_ack(
        SyncDeviceDomainAckCreate(
            dataset_id="dataset-1",
            device_id="device-1",
            domain="notes.note",
            through_server_sequence=2,
            applied_at="2026-05-23T18:29:00+00:00",
            idempotency_key="domain-ack-stale",
        )
    )
    blob_ack = sync_store.upsert_device_blob_ack(
        SyncDeviceBlobAckCreate(
            dataset_id="dataset-1",
            device_id="device-1",
            attachment_id="attachment-1",
            payload_hash="sha256:" + "a" * 64,
            verified_at="2026-05-23T18:31:00+00:00",
            idempotency_key="blob-ack-1",
        )
    )
    summary = sync_store.list_device_acknowledgments("dataset-1", "device-1")

    assert domain_ack.through_server_sequence == 3
    assert stale_domain_ack.through_server_sequence == 3
    assert blob_ack.attachment_id == "attachment-1"
    assert summary.domain_acks["notes.note"].through_server_sequence == 3
    assert summary.blob_acks[0].payload_hash == "sha256:" + "a" * 64


def test_revoked_device_is_hidden_by_default_but_auditable(sync_store: SyncV2Store):
    sync_store.upsert_device(_device(device_id="device-1"))

    revoked = sync_store.revoke_device(
        user_id="user-1",
        device_id="device-1",
        reason="lost_device",
        revoke_key_records=True,
    )

    assert revoked.status == "revoked"
    assert revoked.revoked_at is not None
    assert revoked.revoked_reason == "lost_device"
    assert sync_store.list_devices_for_user("user-1") == []
    assert [device.device_id for device in sync_store.list_devices_for_user("user-1", include_revoked=True)] == [
        "device-1"
    ]


def test_dataset_enrollment_is_idempotent(sync_store: SyncV2Store):
    first = sync_store.enroll_dataset(_dataset())
    second = sync_store.enroll_dataset(
        _dataset(
            domains=["notes.note", "chat.conversation", "chat.message"],
            metadata={"label": "Updated"},
        )
    )
    fetched = sync_store.get_dataset("dataset-1")

    assert second.dataset_id == first.dataset_id
    assert second.created_at == first.created_at
    assert second.domains == ["notes.note", "chat.conversation", "chat.message"]
    assert second.metadata == {"label": "Updated"}
    assert fetched == second


def test_dataset_enrollment_rejects_cross_user_takeover(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset(dataset_id="dataset-shared", owner_user_id="user-1"))

    with pytest.raises(SyncStoreError):
        sync_store.enroll_dataset(_dataset(dataset_id="dataset-shared", owner_user_id="user-2"))

    dataset = sync_store.get_dataset("dataset-shared")
    assert dataset is not None
    assert dataset.owner_user_id == "user-1"


def test_dataset_enrollment_rejects_non_m1_domain_and_encryption_policy(
    sync_store: SyncV2Store,
):
    with pytest.raises(SyncInvalidDomainError):
        sync_store.enroll_dataset(_dataset(domains=["media"]))

    with pytest.raises(SyncStoreError):
        sync_store.enroll_dataset(
            _dataset(dataset_id="dataset-2", encryption_policy="client_private_v1")
        )


def test_dataset_enrollment_supports_workspace_metadata_domains(sync_store: SyncV2Store):
    dataset = sync_store.enroll_dataset(
        _dataset(
            dataset_id="workspace-dataset",
            scope_type="workspace",
            workspace_id="workspace-1",
            domains=["workspaces.workspace", "workspaces.source_ref"],
            metadata={"label": "Shared research workspace"},
        )
    )

    assert dataset.dataset_id == "workspace-dataset"
    assert dataset.scope_type == "workspace"
    assert dataset.workspace_id == "workspace-1"
    assert dataset.domains == ["workspaces.workspace", "workspaces.source_ref"]
    assert dataset.metadata == {"label": "Shared research workspace"}


def test_dataset_enrollment_supports_source_cache_in_personal_and_workspace_scopes(
    sync_store: SyncV2Store,
):
    personal = sync_store.enroll_dataset(
        _dataset(
            dataset_id="personal-source-cache",
            domains=["notes.note", "source_cache.entry"],
        )
    )
    workspace = sync_store.enroll_dataset(
        _dataset(
            dataset_id="workspace-source-cache",
            scope_type="workspace",
            workspace_id="workspace-1",
            domains=["workspaces.source_ref", "source_cache.entry"],
        )
    )

    assert personal.domains == ["notes.note", "source_cache.entry"]
    assert workspace.domains == ["workspaces.source_ref", "source_cache.entry"]


def test_dataset_enrollment_supports_media_metadata_in_personal_and_workspace_scopes(
    sync_store: SyncV2Store,
):
    media_domains = ["media.item", "media.keyword", "media.keyword_link"]
    personal = sync_store.enroll_dataset(
        _dataset(
            dataset_id="personal-media-metadata",
            domains=["notes.note", *media_domains],
        )
    )
    workspace = sync_store.enroll_dataset(
        _dataset(
            dataset_id="workspace-media-metadata",
            scope_type="workspace",
            workspace_id="workspace-1",
            domains=["workspaces.source_ref", *media_domains],
        )
    )

    assert personal.domains == ["notes.note", *media_domains]
    assert workspace.domains == ["workspaces.source_ref", *media_domains]


def test_personal_dataset_enrollment_accepts_notes_link_domain(
    sync_store: SyncV2Store,
) -> None:
    enrolled = sync_store.enroll_dataset(
        _dataset(
            dataset_id="personal-notes-link",
            domains=["notes.note", "notes.link"],
        )
    )

    assert enrolled.domains == ["notes.note", "notes.link"]


def test_dataset_enrollment_rejects_scope_domain_mismatches(sync_store: SyncV2Store):
    with pytest.raises(SyncInvalidDomainError):
        sync_store.enroll_dataset(_dataset(domains=["workspaces.workspace"]))

    with pytest.raises(SyncInvalidDomainError):
        sync_store.enroll_dataset(
            _dataset(
                scope_type="workspace",
                workspace_id="workspace-1",
                domains=["notes.note"],
            )
        )

    with pytest.raises(SyncStoreError):
        sync_store.enroll_dataset(
            _dataset(
                scope_type="workspace",
                workspace_id=None,
                domains=["workspaces.workspace"],
            )
        )


def test_get_dataset_can_be_scoped_by_owner(sync_store: SyncV2Store):
    dataset = sync_store.enroll_dataset(_dataset())

    assert sync_store.get_dataset("dataset-1", owner_user_id="user-1") == dataset
    assert sync_store.get_dataset("dataset-1", owner_user_id="user-2") is None


def test_get_or_create_default_personal_dataset_is_idempotent(sync_store: SyncV2Store):
    first = sync_store.get_or_create_default_personal_dataset("user-1")
    second = sync_store.get_or_create_default_personal_dataset("user-1")

    assert second.dataset_id == first.dataset_id
    assert second.created_at == first.created_at
    assert second.owner_user_id == "user-1"
    assert second.scope_type == "personal"
    assert second.encryption_policy == "server_trusted_v1"
    assert second.domains == [
        "notes.note",
        "chat.conversation",
        "chat.message",
        "attachment.ref",
    ]
    assert second.metadata["default_personal"] is True
    assert second.metadata["client_family"] == "chatbook"
    assert sync_store.list_datasets_for_user("user-1") == [second]


def _task_readiness_at_first_bootstrap_page(sync_store: SyncV2Store) -> None:
    sync_store.enroll_dataset(_dataset())
    sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
    )
    sync_store.transition_notes_task_activity_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
        task_activity_capture_enabled=True,
    )
    sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="enrolling",
        state="bootstrapping",
        source_dataset_id="dataset-1",
        source_cursor=_TASK_CURSOR_1,
        source_count=1,
        source_fingerprint="a" * 64,
    )


def test_notes_task_readiness_blocked_retains_last_verified_progress(
    sync_store: SyncV2Store,
) -> None:
    _task_readiness_at_first_bootstrap_page(sync_store)

    with pytest.raises(SyncStoreError, match="notes_task_readiness_source_changed"):
        sync_store.transition_notes_task_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="bootstrapping",
            state="blocked",
            source_dataset_id="dataset-1",
            source_cursor=_TASK_CURSOR_2,
            source_count=2,
            source_fingerprint="b" * 64,
            reason_code="notes_task_source_invalid",
        )


def test_notes_task_readiness_progress_requires_new_aggregate_fingerprint(
    sync_store: SyncV2Store,
) -> None:
    _task_readiness_at_first_bootstrap_page(sync_store)

    with pytest.raises(SyncStoreError, match="notes_task_readiness_source_changed"):
        sync_store.transition_notes_task_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="bootstrapping",
            state="bootstrapping",
            source_dataset_id="dataset-1",
            source_cursor=_TASK_CURSOR_2,
            source_count=2,
            source_fingerprint="a" * 64,
        )


def test_notes_task_readiness_rejects_corrupt_dataset_metadata_json(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())
    sync_store.db.execute(
        "UPDATE sync_datasets SET metadata_json = ? WHERE dataset_id = ?",
        ("{not-json", "dataset-1"),
    )

    with pytest.raises(SyncStoreError, match="notes_task_readiness_state_invalid"):
        sync_store.transition_notes_task_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="not_enrolled",
            state="enrolling",
            source_dataset_id="dataset-1",
            source_cursor=None,
            source_count=0,
            source_fingerprint=None,
        )

    stored = sync_store.db.execute(
        "SELECT metadata_json FROM sync_datasets WHERE dataset_id = ?",
        ("dataset-1",),
    ).rows[0]
    assert stored["metadata_json"] == "{not-json"


def test_notes_task_readiness_sanitizes_oversized_json_integer(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())
    sync_store.db.execute(
        "UPDATE sync_datasets SET metadata_json = ? WHERE dataset_id = ?",
        ('{"oversized":' + "1" * 4_301 + "}", "dataset-1"),
    )

    with pytest.raises(SyncStoreError, match="notes_task_readiness_state_invalid"):
        sync_store.transition_notes_task_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="not_enrolled",
            state="enrolling",
            source_dataset_id="dataset-1",
            source_cursor=None,
            source_count=0,
            source_fingerprint=None,
        )


def test_notes_task_readiness_rejects_unpaired_surrogate_cursor(
    sync_store: SyncV2Store,
) -> None:
    _task_readiness_at_first_bootstrap_page(sync_store)

    with pytest.raises(SyncStoreError, match="notes_task_readiness_cursor_invalid"):
        sync_store.transition_notes_task_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="bootstrapping",
            state="bootstrapping",
            source_dataset_id="dataset-1",
            source_cursor="\ud800",
            source_count=2,
            source_fingerprint="b" * 64,
        )


def test_notes_task_readiness_rejects_other_domain_reason_code(
    sync_store: SyncV2Store,
) -> None:
    _task_readiness_at_first_bootstrap_page(sync_store)

    with pytest.raises(SyncStoreError, match="notes_task_readiness_reason_invalid"):
        sync_store.transition_notes_task_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="bootstrapping",
            state="blocked",
            source_dataset_id="dataset-1",
            source_cursor=_TASK_CURSOR_1,
            source_count=1,
            source_fingerprint="a" * 64,
            reason_code="notes_task_activity_source_invalid",
        )


def test_notes_task_readiness_rejects_explicit_null_stored_state(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset(metadata={"notes_task_v1": None}))

    with pytest.raises(SyncStoreError, match="notes_task_readiness_state_invalid"):
        sync_store.transition_notes_task_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="not_enrolled",
            state="enrolling",
            source_dataset_id="dataset-1",
            source_cursor=None,
            source_count=0,
            source_fingerprint=None,
        )


def test_notes_task_readiness_rejects_other_domain_reason_in_stored_state(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(
        _dataset(
            metadata={
                "notes_task_v1": {
                    "state": "blocked",
                    "source_cursor": _TASK_CURSOR_1,
                    "source_count": 1,
                    "source_fingerprint": "a" * 64,
                    "reason_code": "notes_task_activity_source_invalid",
                },
                "notes_task_activity_v1": {
                    "state": "enrolling",
                    "source_cursor": None,
                    "source_count": 0,
                    "source_fingerprint": None,
                    "reason_code": None,
                },
                "task_activity_capture_enabled": True,
            }
        )
    )

    with pytest.raises(SyncStoreError, match="notes_task_readiness_state_invalid"):
        sync_store.transition_notes_task_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="blocked",
            state="verifying",
            source_dataset_id="dataset-1",
            source_cursor=_TASK_CURSOR_1,
            source_count=1,
            source_fingerprint="a" * 64,
        )


def test_notes_task_readiness_domains_advance_independently(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())

    task = sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
    )
    activity = sync_store.transition_notes_task_activity_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
        task_activity_capture_enabled=True,
    )

    assert task.metadata["notes_task_v1"]["state"] == "enrolling"
    assert "notes_task_activity_v1" not in task.metadata
    assert activity.metadata["notes_task_v1"]["state"] == "enrolling"
    assert activity.metadata["notes_task_activity_v1"]["state"] == "enrolling"
    assert activity.metadata["task_activity_capture_enabled"] is True
    assert "notes.task" not in activity.domains
    assert "notes.task_activity" not in activity.domains

    task = sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="enrolling",
        state="bootstrapping",
        source_dataset_id="dataset-1",
        source_cursor=_TASK_CURSOR_1,
        source_count=1,
        source_fingerprint="a" * 64,
    )
    task = sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="bootstrapping",
        state="verifying",
        source_dataset_id="dataset-1",
        source_cursor=_TASK_CURSOR_2,
        source_count=2,
        source_fingerprint="b" * 64,
    )
    task = sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="verifying",
        state="ready",
        source_dataset_id="dataset-1",
        source_cursor=_TASK_CURSOR_2,
        source_count=2,
        source_fingerprint="b" * 64,
    )

    assert task.metadata["notes_task_v1"] == {
        "state": "ready",
        "source_cursor": _TASK_CURSOR_2,
        "source_count": 2,
        "source_fingerprint": "b" * 64,
        "reason_code": None,
        "resume_phase": None,
    }
    assert task.metadata["notes_task_activity_v1"]["state"] == "enrolling"

    activity = sync_store.transition_notes_task_activity_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="enrolling",
        state="bootstrapping",
        source_dataset_id="dataset-1",
        source_cursor=_ACTIVITY_CURSOR_1,
        source_count=1,
        source_fingerprint="c" * 64,
    )
    blocked = sync_store.transition_notes_task_activity_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="bootstrapping",
        state="blocked",
        source_dataset_id="dataset-1",
        source_cursor=_ACTIVITY_CURSOR_1,
        source_count=1,
        source_fingerprint="c" * 64,
        reason_code="notes_task_activity_source_invalid",
    )
    resumed = sync_store.transition_notes_task_activity_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="blocked",
        state="bootstrapping",
        source_dataset_id="dataset-1",
        source_cursor=_ACTIVITY_CURSOR_1,
        source_count=1,
        source_fingerprint="c" * 64,
    )
    verifying = sync_store.transition_notes_task_activity_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="bootstrapping",
        state="verifying",
        source_dataset_id="dataset-1",
        source_cursor=_ACTIVITY_CURSOR_1,
        source_count=1,
        source_fingerprint="c" * 64,
    )

    assert activity.metadata["notes_task_activity_v1"]["state"] == "bootstrapping"
    assert blocked.metadata["notes_task_activity_v1"]["reason_code"] == (
        "notes_task_activity_source_invalid"
    )
    assert resumed.metadata["notes_task_activity_v1"]["state"] == "bootstrapping"
    assert resumed.metadata["notes_task_activity_v1"]["reason_code"] is None
    assert verifying.metadata["notes_task_activity_v1"]["state"] == "verifying"


@pytest.mark.parametrize(
    ("changes", "error_code"),
    [
        ({"source_count": 0}, "notes_task_readiness_progress_regressed"),
        ({"source_cursor": _TASK_CURSOR_1}, "notes_task_readiness_progress_regressed"),
        ({"source_fingerprint": "b" * 64}, "notes_task_readiness_source_changed"),
        ({"source_fingerprint": "not-a-hash"}, "notes_task_readiness_fingerprint_invalid"),
        ({"reason_code": "raw private task text"}, "notes_task_readiness_reason_invalid"),
    ],
)
def test_notes_task_readiness_rejects_regression_and_malformed_progress(
    sync_store: SyncV2Store,
    changes: dict[str, object],
    error_code: str,
) -> None:
    sync_store.enroll_dataset(_dataset())
    sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
    )
    sync_store.transition_notes_task_activity_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
        task_activity_capture_enabled=True,
    )
    sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="enrolling",
        state="bootstrapping",
        source_dataset_id="dataset-1",
        source_cursor=_TASK_CURSOR_2,
        source_count=1,
        source_fingerprint="a" * 64,
    )
    arguments: dict[str, object] = {
        "owner_user_id": "user-1",
        "expected_state": "bootstrapping",
        "state": "bootstrapping",
        "source_dataset_id": "dataset-1",
        "source_cursor": _TASK_CURSOR_2,
        "source_count": 1,
        "source_fingerprint": "a" * 64,
        "reason_code": None,
    }
    arguments.update(changes)

    with pytest.raises(SyncStoreError, match=error_code):
        sync_store.transition_notes_task_readiness("dataset-1", **arguments)


def test_notes_task_readiness_capture_change_is_atomic_and_rolls_back(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_store.enroll_dataset(_dataset())
    sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
    )
    sync_store.transition_notes_task_activity_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
    )
    original = sync_store.db._get_dataset_row

    def fail_after_update(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("forced post-update failure")

    monkeypatch.setattr(sync_store.db, "_get_dataset_row", fail_after_update)
    with pytest.raises(RuntimeError, match="forced post-update failure"):
        sync_store.transition_notes_task_activity_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="enrolling",
            state="bootstrapping",
            source_dataset_id="dataset-1",
            source_cursor=None,
            source_count=0,
            source_fingerprint=None,
            task_activity_capture_enabled=True,
        )
    monkeypatch.setattr(sync_store.db, "_get_dataset_row", original)

    stored = sync_store.get_dataset("dataset-1", owner_user_id="user-1")
    assert stored is not None
    assert stored.metadata["notes_task_activity_v1"]["state"] == "enrolling"
    assert stored.metadata.get("task_activity_capture_enabled") is not True


def test_notes_task_readiness_capture_requires_both_domains_and_empty_reset_is_safe(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())
    initial = sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="not_enrolled",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
    )
    assert initial.metadata["notes_task_v1"]["state"] == "not_enrolled"

    with pytest.raises(
        SyncStoreError,
        match="notes_task_readiness_capture_incomplete",
    ):
        sync_store.transition_notes_task_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="not_enrolled",
            state="enrolling",
            source_dataset_id="dataset-1",
            source_cursor=None,
            source_count=0,
            source_fingerprint=None,
            task_activity_capture_enabled=True,
        )

    sync_store.transition_notes_task_activity_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
    )
    sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
        task_activity_capture_enabled=True,
    )
    sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="enrolling",
        state="bootstrapping",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
    )
    with pytest.raises(
        SyncStoreError,
        match="notes_task_readiness_capture_required",
    ):
        sync_store.transition_notes_task_activity_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="enrolling",
            state="enrolling",
            source_dataset_id="dataset-1",
            source_cursor=None,
            source_count=0,
            source_fingerprint=None,
            task_activity_capture_enabled=False,
        )
    sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="bootstrapping",
        state="verifying",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=hashlib.sha256(b"").hexdigest(),
    )

    reset = sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="verifying",
        state="not_enrolled",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
        task_activity_capture_enabled=False,
    )

    assert reset.metadata["notes_task_v1"] == {
        "state": "not_enrolled",
        "source_cursor": None,
        "source_count": 0,
        "source_fingerprint": None,
        "reason_code": None,
        "resume_phase": None,
    }
    assert reset.metadata["notes_task_activity_v1"]["state"] == "enrolling"
    assert reset.metadata["task_activity_capture_enabled"] is False


def test_notes_task_readiness_ready_state_is_terminal(sync_store: SyncV2Store) -> None:
    sync_store.enroll_dataset(_dataset())
    sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
    )
    sync_store.transition_notes_task_activity_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
        task_activity_capture_enabled=True,
    )
    for expected_state, state in (
        ("enrolling", "bootstrapping"),
        ("bootstrapping", "verifying"),
        ("verifying", "ready"),
    ):
        sync_store.transition_notes_task_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state=expected_state,
            state=state,
            source_dataset_id="dataset-1",
            source_cursor=_TASK_CURSOR_1,
            source_count=1,
            source_fingerprint="a" * 64,
        )

    with pytest.raises(SyncStoreError, match="notes_task_readiness_source_changed"):
        sync_store.transition_notes_task_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="ready",
            state="ready",
            source_dataset_id="dataset-1",
            source_cursor=_TASK_CURSOR_2,
            source_count=2,
            source_fingerprint="b" * 64,
        )


def test_notes_task_readiness_rejects_wrong_owner_local_unbound_and_malformed_state(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())

    for owner_user_id, source_dataset_id in (
        ("other-user", "dataset-1"),
        ("user-1", "local-unbound"),
    ):
        with pytest.raises(SyncStoreError):
            sync_store.transition_notes_task_readiness(
                "dataset-1",
                owner_user_id=owner_user_id,
                expected_state="not_enrolled",
                state="enrolling",
                source_dataset_id=source_dataset_id,
                source_cursor=None,
                source_count=0,
                source_fingerprint=None,
            )

    sync_store.db.execute(
        "UPDATE sync_datasets SET metadata_json = ? WHERE dataset_id = ?",
        (
            '{"notes_task_v1":{"state":"ready",'
            '"source_cursor":"private task title","source_count":"many",'
            '"source_fingerprint":"not-a-hash",'
            '"reason_code":"private failure detail"}}',
            "dataset-1",
        ),
    )
    with pytest.raises(SyncStoreError, match="notes_task_readiness_state_invalid"):
        sync_store.transition_notes_task_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="ready",
            state="ready",
            source_dataset_id="dataset-1",
            source_cursor="private task title",
            source_count=1,
            source_fingerprint="a" * 64,
        )


def test_dataset_reenrollment_preserves_server_owned_task_readiness_metadata(
    sync_store: SyncV2Store,
) -> None:
    server_metadata = {
        "notes_task_v1": _readiness_record(
            state="ready",
            source_cursor=_TASK_CURSOR_2,
            source_count=2,
            source_fingerprint="a" * 64,
        ),
        "notes_task_activity_v1": _readiness_record(
            state="blocked",
            source_cursor=_ACTIVITY_CURSOR_1,
            source_count=1,
            source_fingerprint="b" * 64,
            reason_code="notes_task_activity_source_invalid",
            resume_phase="bootstrapping",
        ),
        "task_activity_capture_enabled": True,
    }
    sync_store.enroll_dataset(
        _dataset(metadata={"label": "before", **server_metadata})
    )

    overwritten = sync_store.enroll_dataset(
        _dataset(
            metadata={
                "label": "after",
                "notes_task_v1": _readiness_record(state="not_enrolled"),
                "notes_task_activity_v1": _readiness_record(state="not_enrolled"),
                "task_activity_capture_enabled": False,
            }
        )
    )
    erased = sync_store.enroll_dataset(_dataset(metadata={}))

    assert overwritten.metadata == {"label": "after", **server_metadata}
    assert erased.metadata == server_metadata


def test_dataset_reenrollment_preserves_personal_context_domains_and_metadata(
    sync_store: SyncV2Store,
) -> None:
    """Generic dataset rewrites cannot erase server-owned Personal Context state."""

    binding = {
        "profile_id": "profile-1",
        "authority_id": "authority-a",
        "integrity_key_id": "personal-context-integrity-v1",
        "purge_generation": 0,
        "link_state": "complete",
    }
    sync_store.enroll_dataset(
        _dataset(
            domains=["notes.note", *PERSONAL_CONTEXT_SYNC_DOMAINS],
            metadata={"label": "before", "personal_context": binding},
        )
    )

    reenrolled = sync_store.enroll_dataset(
        _dataset(domains=["notes.note"], metadata={"label": "old-client"})
    )
    attempted_overwrite = sync_store.enroll_dataset(
        _dataset(
            domains=["notes.note"],
            metadata={
                "label": "overwrite-attempt",
                "personal_context": {"profile_id": "attacker"},
            },
        )
    )

    assert set(PERSONAL_CONTEXT_SYNC_DOMAINS).issubset(reenrolled.domains)
    assert reenrolled.metadata["personal_context"] == binding
    assert set(PERSONAL_CONTEXT_SYNC_DOMAINS).issubset(attempted_overwrite.domains)
    assert attempted_overwrite.metadata["personal_context"] == binding


def test_personal_context_receipt_lookup_surfaces_storage_failure(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An outage is not downgraded into a false link-incomplete instruction."""

    def fail_lookup(*_args: object, **_kwargs: object):
        raise SyncStoreError("receipt storage unavailable")

    monkeypatch.setattr(sync_store.db, "execute", fail_lookup)

    with pytest.raises(SyncStoreError, match="receipt storage unavailable"):
        sync_store.has_personal_context_link_receipt(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            profile_id="profile-1",
            integrity_key_id="personal-context-integrity-v1",
            purge_generation=0,
        )


@pytest.mark.parametrize(
    ("readiness_key", "raw", "error_code"),
    [
        ("notes_task_v1", None, "notes_task_readiness_state_invalid"),
        ("notes_task_v1", [], "notes_task_readiness_state_invalid"),
        ("notes_task_v1", {}, "notes_task_readiness_state_invalid"),
        (
            "notes_task_v1",
            {**_readiness_record(state="not_enrolled"), "extra": "private"},
            "notes_task_readiness_state_invalid",
        ),
        (
            "notes_task_v1",
            _readiness_record(
                state="ready",
                source_cursor=_TASK_CURSOR_1,
                source_count=1,
                source_fingerprint=None,
            ),
            "notes_task_readiness_fingerprint_invalid",
        ),
        (
            "notes_task_v1",
            _readiness_record(
                state="bootstrapping",
                source_cursor=_TASK_CURSOR_1,
                source_count=1,
                source_fingerprint=[],  # type: ignore[arg-type]
            ),
            "notes_task_readiness_fingerprint_invalid",
        ),
        (
            "notes_task_v1",
            _readiness_record(
                state="bootstrapping",
                source_cursor=_TASK_CURSOR_1,
                source_count=9_223_372_036_854_775_808,
                source_fingerprint="a" * 64,
            ),
            "notes_task_readiness_progress_invalid",
        ),
        (
            "notes_task_v1",
            _readiness_record(
                state="bootstrapping",
                source_cursor="not-a-uuid",
                source_count=1,
                source_fingerprint="a" * 64,
            ),
            "notes_task_readiness_cursor_invalid",
        ),
        (
            "notes_task_activity_v1",
            _readiness_record(
                state="bootstrapping",
                source_cursor=_TASK_CURSOR_1,
                source_count=1,
                source_fingerprint="a" * 64,
            ),
            "notes_task_readiness_cursor_invalid",
        ),
        (
            "notes_task_activity_v1",
            _readiness_record(
                state="bootstrapping",
                source_cursor=(
                    "0001-01-01T00:00:00+14:00|"
                    "00000000-0000-4000-8000-000000000011"
                ),
                source_count=1,
                source_fingerprint="a" * 64,
            ),
            "notes_task_readiness_cursor_invalid",
        ),
        (
            "notes_task_activity_v1",
            _readiness_record(
                state="bootstrapping",
                source_cursor=(
                    "9999-12-31T23:59:59-14:00|"
                    "00000000-0000-4000-8000-000000000011"
                ),
                source_count=1,
                source_fingerprint="a" * 64,
            ),
            "notes_task_readiness_cursor_invalid",
        ),
        (
            "notes_task_v1",
            {
                **_readiness_record(
                    state="blocked",
                    source_fingerprint="a" * 64,
                    reason_code="notes_task_source_invalid",
                ),
                "resume_phase": [],
            },
            "notes_task_readiness_state_invalid",
        ),
        (
            "notes_task_activity_v1",
            {
                **_readiness_record(
                    state="blocked",
                    source_fingerprint="a" * 64,
                    reason_code="notes_task_activity_source_invalid",
                ),
                "resume_phase": {},
            },
            "notes_task_readiness_state_invalid",
        ),
        (
            "notes_task_v1",
            _readiness_record(
                state="blocked",
                reason_code="notes_task_verification_failed",
                resume_phase="verifying",
            ),
            "notes_task_readiness_state_invalid",
        ),
        (
            "notes_task_activity_v1",
            _readiness_record(
                state="blocked",
                reason_code="notes_task_activity_verification_failed",
                resume_phase="verifying",
            ),
            "notes_task_readiness_state_invalid",
        ),
    ],
)
def test_notes_task_readiness_shared_parser_is_total_and_exact(
    readiness_key: str,
    raw: object,
    error_code: str,
) -> None:
    from tldw_Server_API.app.core.Sync.v2.notes_task_readiness import (
        parse_notes_task_readiness_record,
    )

    result = parse_notes_task_readiness_record(raw, readiness_key=readiness_key)

    assert result.record is None
    assert result.error_code == error_code


def test_notes_task_readiness_shared_parser_accepts_signed_int64_boundary() -> None:
    from tldw_Server_API.app.core.Sync.v2.notes_task_readiness import (
        parse_notes_task_readiness_record,
    )

    result = parse_notes_task_readiness_record(
        _readiness_record(
            state="bootstrapping",
            source_cursor=_TASK_CURSOR_3,
            source_count=9_223_372_036_854_775_807,
            source_fingerprint="f" * 64,
        ),
        readiness_key="notes_task_v1",
    )

    assert result.error_code is None
    assert result.record is not None
    assert result.record.source_count == 9_223_372_036_854_775_807


@pytest.mark.parametrize(
    ("readiness_key", "resume_phase"),
    [
        ("notes_task_v1", []),
        ("notes_task_activity_v1", {}),
    ],
)
def test_notes_task_readiness_transition_rejects_unhashable_resume_phase(
    sync_store: SyncV2Store,
    readiness_key: str,
    resume_phase: object,
) -> None:
    reason_code = (
        "notes_task_source_invalid"
        if readiness_key == "notes_task_v1"
        else "notes_task_activity_source_invalid"
    )
    sync_store.enroll_dataset(
        _dataset(
            metadata={
                readiness_key: {
                    **_readiness_record(
                        state="blocked",
                        source_fingerprint="a" * 64,
                        reason_code=reason_code,
                    ),
                    "resume_phase": resume_phase,
                }
            }
        )
    )
    method = (
        sync_store.transition_notes_task_readiness
        if readiness_key == "notes_task_v1"
        else sync_store.transition_notes_task_activity_readiness
    )

    with pytest.raises(SyncStoreError, match="notes_task_readiness_state_invalid"):
        method(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="blocked",
            state="bootstrapping",
            source_dataset_id="dataset-1",
            source_cursor=None,
            source_count=0,
            source_fingerprint="a" * 64,
        )


def test_notes_task_readiness_shared_parser_exposes_domain_order_keys() -> None:
    from datetime import datetime
    from uuid import UUID

    from tldw_Server_API.app.core.Sync.v2.notes_task_readiness import (
        parse_notes_task_readiness_record,
    )

    task = parse_notes_task_readiness_record(
        _readiness_record(
            state="bootstrapping",
            source_cursor=_TASK_CURSOR_1,
            source_count=1,
            source_fingerprint="a" * 64,
        ),
        readiness_key="notes_task_v1",
    )
    activity = parse_notes_task_readiness_record(
        _readiness_record(
            state="bootstrapping",
            source_cursor=_ACTIVITY_CURSOR_1,
            source_count=1,
            source_fingerprint="b" * 64,
        ),
        readiness_key="notes_task_activity_v1",
    )

    assert task.record is not None
    assert task.record.source_cursor_key == UUID(_TASK_CURSOR_1)
    assert activity.record is not None
    assert activity.record.source_cursor_key == (
        datetime.fromisoformat("2026-08-13T00:00:00+00:00"),
        UUID("00000000-0000-4000-8000-000000000011"),
    )


@pytest.mark.parametrize("source_fingerprint", [[], {}])
def test_notes_task_readiness_non_string_fingerprint_is_bounded_error(
    sync_store: SyncV2Store,
    source_fingerprint: object,
) -> None:
    _task_readiness_at_first_bootstrap_page(sync_store)

    with pytest.raises(
        SyncStoreError,
        match="notes_task_readiness_fingerprint_invalid",
    ):
        sync_store.transition_notes_task_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="bootstrapping",
            state="bootstrapping",
            source_dataset_id="dataset-1",
            source_cursor=_TASK_CURSOR_2,
            source_count=2,
            source_fingerprint=source_fingerprint,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("transition", "source_cursor"),
    [
        ("task", "not-a-uuid"),
        ("task", _ACTIVITY_CURSOR_1),
        ("activity", _TASK_CURSOR_1),
        (
            "activity",
            "2026-08-13T00:00:00-07:00|00000000-0000-4000-8000-000000000011",
        ),
    ],
)
def test_notes_task_readiness_rejects_noncanonical_domain_cursor(
    sync_store: SyncV2Store,
    transition: str,
    source_cursor: str,
) -> None:
    sync_store.enroll_dataset(_dataset())
    sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
    )
    sync_store.transition_notes_task_activity_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
        task_activity_capture_enabled=True,
    )
    method = (
        sync_store.transition_notes_task_readiness
        if transition == "task"
        else sync_store.transition_notes_task_activity_readiness
    )

    with pytest.raises(SyncStoreError, match="notes_task_readiness_cursor_invalid"):
        method(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="enrolling",
            state="bootstrapping",
            source_dataset_id="dataset-1",
            source_cursor=source_cursor,
            source_count=1,
            source_fingerprint="a" * 64,
        )


def _enroll_both_dormant_task_readiness_domains(sync_store: SyncV2Store) -> None:
    sync_store.enroll_dataset(_dataset())
    sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
    )
    sync_store.transition_notes_task_activity_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
        task_activity_capture_enabled=True,
    )


def test_notes_task_readiness_blocked_from_enrolling_requires_bootstrap_resume(
    sync_store: SyncV2Store,
) -> None:
    _enroll_both_dormant_task_readiness_domains(sync_store)
    blocked = sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="enrolling",
        state="blocked",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
        reason_code="notes_task_source_invalid",
    )

    assert blocked.metadata["notes_task_v1"]["resume_phase"] == "bootstrapping"
    with pytest.raises(SyncStoreError, match="notes_task_readiness_transition_invalid"):
        sync_store.transition_notes_task_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="blocked",
            state="verifying",
            source_dataset_id="dataset-1",
            source_cursor=None,
            source_count=0,
            source_fingerprint=hashlib.sha256(b"").hexdigest(),
        )
    resumed = sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="blocked",
        state="bootstrapping",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
    )
    assert resumed.metadata["notes_task_v1"]["resume_phase"] is None


def test_notes_task_readiness_blocked_from_bootstrap_resumes_without_advancing(
    sync_store: SyncV2Store,
) -> None:
    _task_readiness_at_first_bootstrap_page(sync_store)
    blocked = sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="bootstrapping",
        state="blocked",
        source_dataset_id="dataset-1",
        source_cursor=_TASK_CURSOR_1,
        source_count=1,
        source_fingerprint="a" * 64,
        reason_code="notes_task_source_invalid",
    )

    assert blocked.metadata["notes_task_v1"]["resume_phase"] == "bootstrapping"
    with pytest.raises(SyncStoreError, match="notes_task_readiness_transition_invalid"):
        sync_store.transition_notes_task_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="blocked",
            state="verifying",
            source_dataset_id="dataset-1",
            source_cursor=_TASK_CURSOR_1,
            source_count=1,
            source_fingerprint="a" * 64,
        )
    resumed = sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="blocked",
        state="bootstrapping",
        source_dataset_id="dataset-1",
        source_cursor=_TASK_CURSOR_1,
        source_count=1,
        source_fingerprint="a" * 64,
    )
    advanced = sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="bootstrapping",
        state="verifying",
        source_dataset_id="dataset-1",
        source_cursor=_TASK_CURSOR_1,
        source_count=1,
        source_fingerprint="a" * 64,
    )
    assert resumed.metadata["notes_task_v1"]["state"] == "bootstrapping"
    assert advanced.metadata["notes_task_v1"]["state"] == "verifying"


def test_notes_task_readiness_blocked_from_verifying_preserves_resume_progress(
    sync_store: SyncV2Store,
) -> None:
    _task_readiness_at_first_bootstrap_page(sync_store)
    sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="bootstrapping",
        state="verifying",
        source_dataset_id="dataset-1",
        source_cursor=_TASK_CURSOR_1,
        source_count=1,
        source_fingerprint="a" * 64,
    )
    blocked = sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="verifying",
        state="blocked",
        source_dataset_id="dataset-1",
        source_cursor=_TASK_CURSOR_1,
        source_count=1,
        source_fingerprint="a" * 64,
        reason_code="notes_task_verification_failed",
    )

    assert blocked.metadata["notes_task_v1"]["resume_phase"] == "verifying"
    with pytest.raises(SyncStoreError, match="notes_task_readiness_source_changed"):
        sync_store.transition_notes_task_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="blocked",
            state="verifying",
            source_dataset_id="dataset-1",
            source_cursor=_TASK_CURSOR_2,
            source_count=2,
            source_fingerprint="b" * 64,
        )
    resumed = sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="blocked",
        state="verifying",
        source_dataset_id="dataset-1",
        source_cursor=_TASK_CURSOR_1,
        source_count=1,
        source_fingerprint="a" * 64,
    )
    ready = sync_store.transition_notes_task_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="verifying",
        state="ready",
        source_dataset_id="dataset-1",
        source_cursor=_TASK_CURSOR_1,
        source_count=1,
        source_fingerprint="a" * 64,
    )
    assert resumed.metadata["notes_task_v1"]["state"] == "verifying"
    assert ready.metadata["notes_task_v1"]["state"] == "ready"


def _moodboard_dataset(**metadata_overrides: object) -> SyncDatasetCreate:
    return _dataset(
        metadata={
            "default_personal": True,
            "client_family": "chatbook",
            **metadata_overrides,
        }
    )


def test_moodboard_graph_and_studio_readiness_persist_privately_without_capture(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_moodboard_dataset())

    graph = sync_store.transition_notes_moodboard_graph_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        moodboard_source_cursor=None,
        moodboard_source_count=0,
        moodboard_source_fingerprint=None,
        placement_source_cursor=None,
        placement_source_count=0,
        placement_source_fingerprint=None,
    )
    assert graph.metadata["notes_moodboard_v1"]["state"] == "enrolling"
    assert graph.metadata["notes_moodboard_note_v1"]["state"] == "enrolling"
    assert graph.metadata["moodboard_capture_enabled"] is False
    assert graph.metadata["studio_document_capture_enabled"] is False

    for expected_state, state in (
        ("enrolling", "bootstrapping"),
        ("bootstrapping", "verifying"),
        ("verifying", "ready"),
    ):
        graph = sync_store.transition_notes_moodboard_graph_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state=expected_state,
            state=state,
            source_dataset_id="dataset-1",
            moodboard_source_cursor=_MOODBOARD_CURSOR_1,
            moodboard_source_count=1,
            moodboard_source_fingerprint="a" * 64,
            placement_source_cursor=_PLACEMENT_CURSOR_1,
            placement_source_count=1,
            placement_source_fingerprint="b" * 64,
        )

    assert graph.metadata["notes_moodboard_v1"]["state"] == "ready"
    assert graph.metadata["notes_moodboard_note_v1"]["state"] == "ready"
    assert "notes.studio_document" not in graph.domains

    studio = sync_store.transition_notes_studio_document_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
    )
    assert studio.metadata["notes_studio_document_v1"]["state"] == "enrolling"
    assert studio.metadata["notes_moodboard_v1"]["state"] == "ready"
    assert studio.metadata["studio_document_capture_enabled"] is False


def test_moodboard_graph_readiness_is_coupled_atomic_and_cannot_enable_capture(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_store.enroll_dataset(_moodboard_dataset())
    sync_store.transition_notes_moodboard_graph_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        moodboard_source_cursor=None,
        moodboard_source_count=0,
        moodboard_source_fingerprint=None,
        placement_source_cursor=None,
        placement_source_count=0,
        placement_source_fingerprint=None,
    )

    with pytest.raises(
        SyncStoreError,
        match="notes_moodboard_studio_readiness_capture_forbidden",
    ):
        sync_store.transition_notes_moodboard_graph_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="enrolling",
            state="bootstrapping",
            source_dataset_id="dataset-1",
            moodboard_source_cursor=None,
            moodboard_source_count=0,
            moodboard_source_fingerprint=None,
            placement_source_cursor=None,
            placement_source_count=0,
            placement_source_fingerprint=None,
            moodboard_capture_enabled=True,
        )

    original = sync_store.db._get_dataset_row

    def fail_after_update(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("forced moodboard rollback")

    monkeypatch.setattr(sync_store.db, "_get_dataset_row", fail_after_update)
    with pytest.raises(RuntimeError, match="forced moodboard rollback"):
        sync_store.transition_notes_moodboard_graph_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="enrolling",
            state="bootstrapping",
            source_dataset_id="dataset-1",
            moodboard_source_cursor=None,
            moodboard_source_count=0,
            moodboard_source_fingerprint=None,
            placement_source_cursor=None,
            placement_source_count=0,
            placement_source_fingerprint=None,
        )
    monkeypatch.setattr(sync_store.db, "_get_dataset_row", original)

    stored = sync_store.get_dataset("dataset-1", owner_user_id="user-1")
    assert stored is not None
    assert stored.metadata["notes_moodboard_v1"]["state"] == "enrolling"
    assert stored.metadata["notes_moodboard_note_v1"]["state"] == "enrolling"
    assert stored.metadata["moodboard_capture_enabled"] is False


def test_moodboard_studio_readiness_rejects_scope_and_cas_races(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_moodboard_dataset())
    sync_store.enroll_dataset(
        _dataset(
            dataset_id="workspace-1",
            scope_type="workspace",
            workspace_id="workspace-1",
            domains=["workspaces.workspace"],
            metadata={"default_personal": True, "client_family": "chatbook"},
        )
    )
    sync_store.enroll_dataset(
        _dataset(
            dataset_id="bad-policy",
            metadata={"default_personal": True, "client_family": "chatbook"},
        )
    )
    sync_store.db.execute(
        "UPDATE sync_datasets SET encryption_policy = ? WHERE dataset_id = ?",
        ("client_private_v1", "bad-policy"),
    )

    for dataset_id, owner_user_id, source_dataset_id in (
        ("dataset-1", "other-user", "dataset-1"),
        ("dataset-1", "user-1", "local-unbound"),
        ("workspace-1", "user-1", "workspace-1"),
        ("bad-policy", "user-1", "bad-policy"),
    ):
        with pytest.raises(SyncStoreError):
            sync_store.transition_notes_studio_document_readiness(
                dataset_id,
                owner_user_id=owner_user_id,
                expected_state="not_enrolled",
                state="enrolling",
                source_dataset_id=source_dataset_id,
                source_cursor=None,
                source_count=0,
                source_fingerprint=None,
            )

    sync_store.transition_notes_studio_document_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
    )
    with pytest.raises(
        SyncStoreError,
        match="notes_moodboard_studio_readiness_compare_and_set_failed",
    ):
        sync_store.transition_notes_studio_document_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="not_enrolled",
            state="enrolling",
            source_dataset_id="dataset-1",
            source_cursor=None,
            source_count=0,
            source_fingerprint=None,
        )


def test_moodboard_studio_readiness_blocks_and_resumes_without_raw_source_leak(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_moodboard_dataset())
    sync_store.transition_notes_moodboard_graph_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        moodboard_source_cursor=None,
        moodboard_source_count=0,
        moodboard_source_fingerprint=None,
        placement_source_cursor=None,
        placement_source_count=0,
        placement_source_fingerprint=None,
    )
    blocked = sync_store.transition_notes_moodboard_graph_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="enrolling",
        state="blocked",
        source_dataset_id="dataset-1",
        moodboard_source_cursor=None,
        moodboard_source_count=0,
        moodboard_source_fingerprint=None,
        placement_source_cursor=None,
        placement_source_count=0,
        placement_source_fingerprint=None,
        moodboard_reason_code="notes_moodboard_source_invalid",
        placement_reason_code="notes_moodboard_note_source_invalid",
    )
    assert blocked.metadata["notes_moodboard_v1"]["resume_phase"] == "bootstrapping"
    assert blocked.metadata["notes_moodboard_note_v1"]["resume_phase"] == "bootstrapping"

    with pytest.raises(
        SyncStoreError,
        match="notes_moodboard_studio_readiness_transition_invalid",
    ):
        sync_store.transition_notes_moodboard_graph_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="blocked",
            state="verifying",
            source_dataset_id="dataset-1",
            moodboard_source_cursor=None,
            moodboard_source_count=0,
            moodboard_source_fingerprint=hashlib.sha256(b"").hexdigest(),
            placement_source_cursor=None,
            placement_source_count=0,
            placement_source_fingerprint=hashlib.sha256(b"").hexdigest(),
        )

    resumed = sync_store.transition_notes_moodboard_graph_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="blocked",
        state="bootstrapping",
        source_dataset_id="dataset-1",
        moodboard_source_cursor=None,
        moodboard_source_count=0,
        moodboard_source_fingerprint=None,
        placement_source_cursor=None,
        placement_source_count=0,
        placement_source_fingerprint=None,
    )
    assert resumed.metadata["notes_moodboard_v1"]["reason_code"] is None
    assert resumed.metadata["notes_moodboard_note_v1"]["reason_code"] is None


@pytest.mark.parametrize(
    ("readiness_key", "raw", "error_code"),
    [
        ("notes_moodboard_v1", None, "notes_moodboard_studio_readiness_state_invalid"),
        (
            "notes_moodboard_v1",
            {**_readiness_record(state="not_enrolled"), "private": "board name"},
            "notes_moodboard_studio_readiness_state_invalid",
        ),
        (
            "notes_moodboard_v1",
            _readiness_record(
                state="ready",
                source_cursor=_MOODBOARD_CURSOR_1,
                source_count=1,
                source_fingerprint=None,
            ),
            "notes_moodboard_studio_readiness_fingerprint_invalid",
        ),
        (
            "notes_moodboard_note_v1",
            _readiness_record(
                state="bootstrapping",
                source_cursor=_MOODBOARD_CURSOR_1,
                source_count=1,
                source_fingerprint="b" * 64,
            ),
            "notes_moodboard_studio_readiness_cursor_invalid",
        ),
        (
            "notes_studio_document_v1",
            _readiness_record(
                state="blocked",
                source_fingerprint="c" * 64,
                reason_code="studio provider prompt leaked",
                resume_phase="bootstrapping",
            ),
            "notes_moodboard_studio_readiness_reason_invalid",
        ),
    ],
)
def test_moodboard_studio_readiness_parser_is_total_and_exact(
    readiness_key: str,
    raw: object,
    error_code: str,
) -> None:
    from tldw_Server_API.app.core.Sync.v2.notes_moodboard_studio_readiness import (
        parse_notes_moodboard_studio_readiness_record,
    )

    result = parse_notes_moodboard_studio_readiness_record(
        raw,
        readiness_key=readiness_key,
    )

    assert result.record is None
    assert result.error_code == error_code


def test_moodboard_studio_readiness_rejects_progress_regression_and_source_drift(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_moodboard_dataset())
    sync_store.transition_notes_studio_document_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="not_enrolled",
        state="enrolling",
        source_dataset_id="dataset-1",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
    )
    sync_store.transition_notes_studio_document_readiness(
        "dataset-1",
        owner_user_id="user-1",
        expected_state="enrolling",
        state="bootstrapping",
        source_dataset_id="dataset-1",
        source_cursor=_STUDIO_CURSOR_2,
        source_count=2,
        source_fingerprint="c" * 64,
    )

    with pytest.raises(
        SyncStoreError,
        match="notes_moodboard_studio_readiness_progress_regressed",
    ):
        sync_store.transition_notes_studio_document_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="bootstrapping",
            state="bootstrapping",
            source_dataset_id="dataset-1",
            source_cursor=_STUDIO_CURSOR_1,
            source_count=1,
            source_fingerprint="d" * 64,
        )
    with pytest.raises(
        SyncStoreError,
        match="notes_moodboard_studio_readiness_source_changed",
    ):
        sync_store.transition_notes_studio_document_readiness(
            "dataset-1",
            owner_user_id="user-1",
            expected_state="bootstrapping",
            state="bootstrapping",
            source_dataset_id="dataset-1",
            source_cursor=_STUDIO_CURSOR_2,
            source_count=2,
            source_fingerprint="d" * 64,
        )


def test_insert_envelope_is_idempotent_by_dataset_and_client_envelope(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    envelope = _envelope(client_envelope_id="env-1")

    first = sync_store.insert_envelope(envelope)
    second = sync_store.insert_envelope(envelope)

    assert second.server_cursor == first.server_cursor
    assert second.received_at_server == first.received_at_server
    assert sync_store.list_envelopes_after("dataset-1", 0) == [first]


def test_insert_envelope_idempotent_retry_ignores_mutable_apply_state(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())
    envelope = _envelope(client_envelope_id="env-1")

    first = sync_store.insert_envelope(envelope)
    applied = sync_store.mark_envelope_apply_status(
        first.server_cursor,
        apply_status="applied",
    )
    retried = sync_store.insert_envelope(envelope)

    assert applied.apply_status == "applied"
    assert retried.server_cursor == first.server_cursor
    assert retried.apply_status == "applied"


def test_insert_envelope_idempotency_uses_envelope_key_after_other_insert(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    envelope = _envelope(client_envelope_id="env-1")

    first = sync_store.insert_envelope(envelope)
    sync_store.insert_envelope(
        _envelope(
            client_envelope_id="env-2",
            object_id="note-2",
            payload_hash="sha256:note-2",
        )
    )
    duplicate = sync_store.insert_envelope(envelope)

    assert duplicate.server_cursor == first.server_cursor


def test_insert_envelope_persists_m1_fields_and_aliases(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    created = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="env-created",
            object_revision=1,
            payload_hash="sha256:note-v1",
        )
    )
    prior = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="env-prior",
            object_revision=2,
            payload_hash="sha256:note-v2",
            base_server_cursor=created.server_cursor,
            base_object_revision=created.object_revision,
            base_object_hash=created.payload_hash,
        )
    )

    stored = sync_store.insert_envelope(
        _envelope(
            client_sequence=1,
            base_server_cursor=prior.server_cursor,
            base_object_revision=2,
            base_object_hash="sha256:note-v2",
            object_revision=3,
            parent_id="folder-1",
            schema_version=2,
            payload={"title": "Changed"},
            payload_hash="sha256:note-v3",
            deleted=True,
            encryption_metadata={"policy": "server_trusted_v1", "key": "server"},
        )
    )

    assert stored.server_cursor >= 1
    assert stored.server_sequence == stored.server_cursor
    assert stored.object_id == "note-1"
    assert stored.entity_id == "note-1"
    assert stored.base_server_cursor == prior.server_cursor
    assert stored.base_object_revision == 2
    assert stored.base_object_hash == "sha256:note-v2"
    assert stored.object_revision == 3
    assert stored.parent_id == "folder-1"
    assert stored.schema_version == 2
    assert stored.payload == {"title": "Changed"}
    assert stored.payload_clear == {"title": "Changed"}
    assert stored.payload_hash == "sha256:note-v3"
    assert stored.created_at_client == "2026-05-10T00:00:00+00:00"
    assert stored.received_at_server is not None
    assert stored.deleted is True
    assert stored.encryption_metadata == {"policy": "server_trusted_v1", "key": "server"}
    assert stored.apply_status == "pending"


def test_insert_envelopes_atomic_persists_complete_ordered_mutation_group(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())

    inserted = sync_store.insert_envelopes_atomic(_mutation_group_envelopes())

    assert [envelope.mutation_step for envelope in inserted] == [0, 1, 2]
    assert [envelope.server_cursor for envelope in inserted] == [1, 2, 3]
    assert all(envelope.mutation_step_count == 3 for envelope in inserted)
    assert all(envelope.mutation_plan_hash == "a" * 64 for envelope in inserted)
    assert sync_store.list_mutation_group("dataset-1", "mutation-group-1") == inserted


def test_insert_envelopes_atomic_does_not_queue_behind_unresolved_materialization_conflict(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())
    source = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="env-group-blocker",
            object_id="note-blocker",
            payload_hash="sha256:blocker",
        )
    )
    source = sync_store.mark_envelope_apply_status(
        source.server_cursor,
        apply_status="conflict",
        apply_error_code="projection_conflict",
    )
    sync_store.insert_conflict(
        _conflict(
            object_id=source.object_id,
            local_envelope_id=source.client_envelope_id,
            server_cursor=source.server_cursor,
        )
    )

    with pytest.raises(SyncMaterializationPredecessorError):
        sync_store.insert_envelopes_atomic(_mutation_group_envelopes())

    assert sync_store.list_mutation_group("dataset-1", "mutation-group-1") == []


def test_guarded_conflict_bookkeeping_reuses_materialization_transaction(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_store.enroll_dataset(_dataset())
    envelope = sync_store.insert_envelope(_envelope())
    observed_connections: list[object | None] = []

    def record_conflict(conflict, *, connection=None):
        observed_connections.append(connection)
        return conflict

    monkeypatch.setattr(sync_store.db, "insert_conflict", record_conflict)

    with sync_store.materialization_guard([envelope]) as guarded_store:
        guarded_store.insert_conflict(
            _conflict(server_cursor=envelope.server_cursor)
        )
        guarded_connection = guarded_store._connection

    assert observed_connections == [guarded_connection]


def test_conflict_insert_rolls_back_on_crash_and_retries_idempotently_by_envelope_cursor(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())
    envelope = sync_store.insert_envelope(_envelope())
    conflict = _conflict(
        local_envelope_id=envelope.client_envelope_id,
        server_cursor=envelope.server_cursor,
    )

    with pytest.raises(RuntimeError, match="crash before commit"):
        with sync_store.materialization_guard([envelope]) as guarded_store:
            guarded_store.insert_conflict(conflict)
            raise RuntimeError("crash before commit")

    assert sync_store.list_conflicts("dataset-1") == []

    with sync_store.materialization_guard([envelope]) as guarded_store:
        first = guarded_store.insert_conflict(conflict)
        replay = guarded_store.insert_conflict(
            replace(conflict, conflict_id="conflict-retry")
        )

    assert replay == first
    assert [item.conflict_id for item in sync_store.list_conflicts("dataset-1")] == [
        "conflict-1"
    ]


def test_direct_bootstrap_bookkeeping_acquires_dataset_guard(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_store.enroll_dataset(_dataset())
    stale = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="bootstrap-stale",
            object_revision=1,
            routing_metadata={"source": "notes-organization-bootstrap"},
        )
    )
    correction = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="bootstrap-correction",
            object_revision=2,
            payload_hash="sha256:correction",
            base_server_cursor=stale.server_cursor,
            base_object_revision=stale.object_revision,
            base_object_hash=stale.payload_hash,
            routing_metadata={"source": "notes-organization-bootstrap"},
        )
    )
    assert correction.server_cursor is not None
    sync_store.mark_envelope_apply_status(
        correction.server_cursor,
        apply_status="applied",
    )
    pending = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="bootstrap-pending",
            object_id="note-2",
            payload_hash="sha256:pending",
            routing_metadata={"source": "notes-organization-bootstrap"},
        )
    )
    acquired: list[str] = []
    original_lock = sync_store.db._lock_materialization_dataset

    def record_lock(dataset_id: str, *, connection):
        acquired.append(dataset_id)
        return original_lock(dataset_id, connection=connection)

    monkeypatch.setattr(sync_store.db, "_lock_materialization_dataset", record_lock)

    assert pending.server_cursor is not None
    sync_store.mark_bootstrap_envelope_verified(
        pending.server_cursor,
        bootstrap_id="bootstrap-1",
    )
    assert stale.server_cursor is not None
    sync_store.reconcile_bootstrap_envelope_superseded(
        stale.server_cursor,
        bootstrap_id="bootstrap-1",
        superseded_by_cursor=correction.server_cursor,
    )

    assert acquired == ["dataset-1", "dataset-1"]


def test_code_quality_round2_atomic_append_rejects_oversized_group(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())

    with pytest.raises(SyncStoreError, match="sync_restore_group_limit_exceeded"):
        sync_store.insert_envelopes_atomic(_mutation_group_envelopes(count=1_001))

    assert sync_store.list_mutation_group("dataset-1", "mutation-group-1") == []


def test_code_quality_round2_group_read_uses_bounded_max_plus_one_query(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def oversized_query(sql, params, *, connection=None):
        captured.update(sql=sql, params=params, connection=connection)
        return type("Result", (), {"rows": [{}] * 1_001})()

    monkeypatch.setattr(sync_store.db, "execute", oversized_query)

    with pytest.raises(SyncStoreError, match="sync_restore_group_limit_exceeded"):
        sync_store.list_mutation_group("dataset-1", "mutation-group-1")

    assert "LIMIT ?" in str(captured["sql"])
    assert captured["params"] == ("dataset-1", "mutation-group-1", 1_001)


def test_insert_envelopes_atomic_returns_identical_mutation_group_replay(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())
    envelopes = _mutation_group_envelopes()

    first = sync_store.insert_envelopes_atomic(envelopes)
    replay = sync_store.insert_envelopes_atomic(envelopes)

    assert replay == first
    assert sync_store.list_envelopes_after("dataset-1", 0) == first


def test_code_quality_i2_native_postgres_timestamp_preserves_group_fingerprint(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())
    placeholder = _mutation_group_envelopes(mutation_plan_hash="0" * 64)
    plan_hash = mutation_group_plan_hash(placeholder)
    plan = [replace(envelope, mutation_plan_hash=plan_hash) for envelope in placeholder]
    sync_store.insert_envelopes_atomic(plan)
    rows = sync_store.db.execute(
        "SELECT * FROM sync_envelopes WHERE mutation_group_id = ? "
        "ORDER BY mutation_step ASC",
        ("mutation-group-1",),
    ).rows
    native_timestamp = datetime(2026, 5, 10, tzinfo=timezone.utc)
    native_rows = [
        {
            **row,
            "created_at_client": native_timestamp,
            "client_timestamp": native_timestamp,
        }
        for row in rows
    ]

    restored = [_envelope_from_row(row) for row in native_rows]

    assert [envelope.created_at_client for envelope in restored] == [
        "2026-05-10T00:00:00+00:00"
    ] * 3
    assert _envelope_fingerprint_from_row(native_rows[0]) == (
        _envelope_fingerprint_from_create(plan[0])
    )
    validate_stored_mutation_group(
        restored,
        dataset_id="dataset-1",
        mutation_group_id="mutation-group-1",
    )


@pytest.mark.parametrize(
    ("timestamp", "legacy_plan_hash"),
    [
        (
            "2026-05-10T00:00:00Z",
            "c320a2ff060ffe677c1e89fef9b69add90e27ea25226ecc23091971e1435c665",
        ),
        (
            "2026-05-09T17:00:00-07:00",
            "7bd3699f323f6409d188e41c93ae1a5570c37e3ea367814c2f1cd229efe4ca51",
        ),
    ],
)
def test_code_quality_round2_accepts_genuine_legacy_sqlite_timestamp_hashes(
    sync_store: SyncV2Store,
    timestamp: str,
    legacy_plan_hash: str,
) -> None:
    sync_store.enroll_dataset(_dataset())
    legacy_plan = _mutation_group_envelopes(mutation_plan_hash=legacy_plan_hash)
    for envelope in legacy_plan:
        object.__setattr__(envelope, "created_at_client", timestamp)
        object.__setattr__(envelope, "client_timestamp", timestamp)
    sync_store.insert_envelopes_atomic(legacy_plan)

    restored = sync_store.list_mutation_group("dataset-1", "mutation-group-1")

    assert [envelope.created_at_client for envelope in restored] == [timestamp] * 3
    validate_stored_mutation_group(
        restored,
        dataset_id="dataset-1",
        mutation_group_id="mutation-group-1",
    )


def test_code_quality_round2_accepts_legacy_utc_z_hash_from_postgres_datetime(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())
    legacy_plan_hash = (
        "c320a2ff060ffe677c1e89fef9b69add90e27ea25226ecc23091971e1435c665"
    )
    legacy_plan = _mutation_group_envelopes(mutation_plan_hash=legacy_plan_hash)
    for envelope in legacy_plan:
        object.__setattr__(envelope, "created_at_client", "2026-05-10T00:00:00Z")
        object.__setattr__(envelope, "client_timestamp", "2026-05-10T00:00:00Z")
    sync_store.insert_envelopes_atomic(legacy_plan)
    rows = sync_store.db.execute(
        "SELECT * FROM sync_envelopes WHERE mutation_group_id = ? "
        "ORDER BY mutation_step ASC",
        ("mutation-group-1",),
    ).rows
    native_rows = [
        {
            **row,
            "created_at_client": datetime(2026, 5, 10, tzinfo=timezone.utc),
            "client_timestamp": datetime(2026, 5, 10, tzinfo=timezone.utc),
        }
        for row in rows
    ]

    restored = [_envelope_from_row(row) for row in native_rows]

    assert [envelope.created_at_client for envelope in restored] == [
        "2026-05-10T00:00:00+00:00"
    ] * 3
    validate_stored_mutation_group(
        restored,
        dataset_id="dataset-1",
        mutation_group_id="mutation-group-1",
    )


def test_insert_envelopes_atomic_rejects_mutation_group_replay_drift(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())
    sync_store.insert_envelopes_atomic(_mutation_group_envelopes())
    changed = _mutation_group_envelopes()
    changed[1] = _envelope(
        client_envelope_id="env-group-1",
        object_id="note-2",
        payload_hash="sha256:changed",
        mutation_group_id="mutation-group-1",
        mutation_step=1,
        mutation_step_count=3,
        mutation_plan_hash="a" * 64,
    )

    with pytest.raises(
        SyncIdempotencyConflictError,
        match="Sync mutation group idempotency key was reused with different content",
    ):
        sync_store.insert_envelopes_atomic(changed)


@pytest.mark.parametrize(
    "replay_factory",
    [
        pytest.param(lambda plan: plan[:2], id="missing-step"),
        pytest.param(
            lambda plan: [
                *plan,
                replace(
                    plan[2],
                    client_envelope_id="env-group-extra",
                    object_id="note-extra",
                    payload_hash="sha256:note-extra",
                ),
            ],
            id="extra-duplicate-step",
        ),
        pytest.param(
            lambda plan: [plan[0], plan[2], plan[1]],
            id="reordered-steps",
        ),
    ],
)
def test_insert_envelopes_atomic_rejects_incomplete_mutation_group_replays_as_conflicts(
    sync_store: SyncV2Store,
    replay_factory,
) -> None:
    sync_store.enroll_dataset(_dataset())
    plan = _mutation_group_envelopes()
    sync_store.insert_envelopes_atomic(plan)

    with pytest.raises(
        SyncIdempotencyConflictError,
        match="Sync mutation group idempotency key was reused with different content",
    ):
        sync_store.insert_envelopes_atomic(replay_factory(plan))


def test_insert_envelopes_atomic_keeps_shape_error_for_new_incomplete_mutation_group(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())

    with pytest.raises(
        SyncStoreError,
        match="Sync mutation group steps must exactly match the ordered complete plan",
    ):
        sync_store.insert_envelopes_atomic(_mutation_group_envelopes()[:2])


def test_insert_envelopes_atomic_returns_identical_postgres_mutation_group_race_winner(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_store.enroll_dataset(_dataset())
    plan = _mutation_group_envelopes()
    stored = sync_store.insert_envelopes_atomic(plan)
    calls = _inject_postgres_mutation_group_race(sync_store, monkeypatch)

    replay = sync_store.insert_envelopes_atomic(plan)

    assert replay == stored
    assert calls == {"list": 2, "transactions": 2}


def test_insert_envelopes_atomic_rejects_drifted_postgres_mutation_group_race_winner(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_store.enroll_dataset(_dataset())
    plan = _mutation_group_envelopes()
    sync_store.insert_envelopes_atomic(plan)
    changed = list(plan)
    changed[1] = replace(changed[1], payload_hash="sha256:changed")
    calls = _inject_postgres_mutation_group_race(sync_store, monkeypatch)

    with pytest.raises(
        SyncIdempotencyConflictError,
        match="Sync mutation group idempotency key was reused with different content",
    ):
        sync_store.insert_envelopes_atomic(changed)

    assert calls == {"list": 2, "transactions": 2}


def test_insert_envelopes_atomic_does_not_swallow_unrelated_postgres_unique_failure(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OtherUniqueDiagnostics:
        constraint_name = "sync_envelopes_dataset_id_client_envelope_id_key"

    class OtherPostgresUniqueViolation(Exception):
        sqlstate = "23505"
        diag = OtherUniqueDiagnostics()

    def fail_unrelated_unique_insert(envelope, *, connection):
        raise BackendDatabaseError("unrelated PostgreSQL unique failure") from (
            OtherPostgresUniqueViolation()
        )

    sync_store.enroll_dataset(_dataset())
    monkeypatch.setattr(
        sync_store.db,
        "_insert_envelope_in_transaction",
        fail_unrelated_unique_insert,
    )

    with pytest.raises(
        BackendDatabaseError,
        match="unrelated PostgreSQL unique failure",
    ):
        sync_store.insert_envelopes_atomic(_mutation_group_envelopes())


def test_insert_envelopes_atomic_rolls_back_mutation_group_after_step_two_failure(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_store.enroll_dataset(_dataset())
    original_execute = sync_store.db.execute
    insert_calls = 0

    def fail_second_envelope_insert(query, params=None, *, connection=None):
        nonlocal insert_calls
        if query.lstrip().startswith("INSERT INTO sync_envelopes"):
            insert_calls += 1
            if insert_calls == 2:
                raise RuntimeError("injected mutation-group step 2 failure")
        return original_execute(query, params, connection=connection)

    monkeypatch.setattr(sync_store.db, "execute", fail_second_envelope_insert)

    with pytest.raises(RuntimeError, match="injected mutation-group step 2 failure"):
        sync_store.insert_envelopes_atomic(_mutation_group_envelopes())

    assert sync_store.list_mutation_group("dataset-1", "mutation-group-1") == []
    assert sync_store.list_envelopes_after("dataset-1", 0) == []


def test_insert_envelopes_atomic_preserves_legacy_single_mutation_group_absence_behavior(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(_dataset())

    inserted = sync_store.insert_envelope(_envelope(client_envelope_id="env-legacy"))
    replay = sync_store.insert_envelope(_envelope(client_envelope_id="env-legacy"))

    assert replay == inserted
    assert inserted.mutation_group_id is None
    assert inserted.mutation_step is None
    assert inserted.mutation_step_count is None
    assert inserted.mutation_plan_hash is None


def test_insert_envelope_is_idempotent_by_dataset_device_and_client_sequence(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())

    first = sync_store.insert_envelope(_envelope(client_sequence=11))
    duplicate = sync_store.insert_envelope(
        _envelope(client_envelope_id="env-1-retry", client_sequence=11)
    )

    assert duplicate.server_cursor == first.server_cursor
    assert duplicate.client_envelope_id == first.client_envelope_id

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.insert_envelope(
            _envelope(
                client_envelope_id="env-1-conflict",
                client_sequence=11,
                payload_hash="sha256:changed",
            )
        )


def test_insert_envelope_rejects_duplicate_drift(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    envelope = _envelope(client_envelope_id="env-1")

    sync_store.insert_envelope(envelope)

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.insert_envelope(
            _envelope(client_envelope_id="env-1", payload_hash="sha256:changed")
        )


def test_insert_envelope_rejects_domain_not_enrolled(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset(domains=["notes.note"]))

    with pytest.raises(SyncInvalidDomainError):
        sync_store.insert_envelope(
            _envelope(
                domain="chat.conversation",
                object_id="conversation-1",
                payload_hash="sha256:chat-1",
            )
        )


def test_insert_envelope_rejects_unsupported_operation_and_encryption_policy(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(_envelope(operation="delete"))

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(
            _envelope(encryption_metadata={"policy": "client_private_v1"})
        )


def test_insert_envelope_rejects_direct_core_payload_hash_bypass(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(_envelope(payload_hash=None))

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(
            _envelope(client_envelope_id="env-blank-hash", payload_hash="   ")
        )


def test_insert_envelope_rejects_direct_core_whole_object_base_metadata_bypass(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(
            _envelope(
                client_envelope_id="env-update-no-base",
                object_revision=2,
            )
        )

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(
            _envelope(
                client_envelope_id="env-partial-base",
                base_server_cursor=1,
            )
        )

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(
            _envelope(
                client_envelope_id="env-tombstone-no-base",
                operation="tombstone",
                deleted=True,
                payload_hash="sha256:tombstone",
            )
        )


def test_insert_envelope_rejects_direct_core_chat_message_identity_and_hash_bypass(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(
            _envelope(
                client_envelope_id="env-message-blank-id",
                domain="chat.message",
                operation="append",
                object_id="   ",
                parent_id="conversation-1",
                payload_hash="sha256:message",
            )
        )

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(
            _envelope(
                client_envelope_id="env-message-blank-hash",
                domain="chat.message",
                operation="append",
                object_id="message-1",
                parent_id="conversation-1",
                payload_hash="",
            )
        )


def test_insert_envelope_rejects_direct_core_attachment_ref_metadata_bypass(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(
            _envelope(
                client_envelope_id="env-attachment-ref",
                domain="attachment.ref",
                object_id="attachment-1",
                payload={"attachment_id": "attachment-1"},
                payload_hash="sha256:attachment-ref",
            )
        )


def test_list_envelopes_after_cursor_is_ordered_and_domain_filterable(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    first = sync_store.insert_envelope(_envelope(client_envelope_id="env-1", object_id="note-1"))
    second = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="env-2",
            domain="chat.conversation",
            object_id="conversation-1",
            payload_hash="sha256:chat-1",
        )
    )
    third = sync_store.insert_envelope(
        _envelope(client_envelope_id="env-3", object_id="note-2", payload_hash="sha256:note-2")
    )

    assert [row.server_cursor for row in sync_store.list_envelopes_after("dataset-1", first.server_cursor)] == [
        second.server_cursor,
        third.server_cursor,
    ]
    assert sync_store.list_envelopes_after("dataset-1", 0, domains=["notes.note"]) == [first, third]


def test_list_envelopes_after_filters_status_and_excluded_device_in_sql(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())
    own = sync_store.insert_envelope(_envelope(client_envelope_id="own-accepted"))
    remote = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="remote-accepted",
            object_id="note-remote",
            device_id="device-2",
            payload_hash="sha256:remote",
        )
    )
    sync_store.insert_envelope(
        _envelope(
            client_envelope_id="remote-conflict",
            object_id="note-conflict",
            device_id="device-2",
            payload_hash="sha256:conflict",
            status="conflict",
        )
    )

    visible = sync_store.list_envelopes_after(
        "dataset-1",
        0,
        status="accepted",
        exclude_device_id="device-1",
    )

    assert own not in visible
    assert remote in visible
    assert [envelope.client_envelope_id for envelope in visible] == ["remote-accepted"]


def test_apply_status_lifecycle_and_replay_listing(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    first = sync_store.insert_envelope(_envelope(client_envelope_id="env-1"))
    second = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="env-2",
            object_id="note-2",
            client_sequence=2,
            payload_hash="sha256:note-2",
        )
    )
    sync_store.insert_envelope(
        _envelope(
            client_envelope_id="env-conflict",
            object_id="note-conflict",
            client_sequence=3,
            payload_hash="sha256:note-conflict",
            status="conflict",
            apply_status="conflict",
        )
    )

    applied = sync_store.mark_envelope_apply_status(
        first.server_cursor,
        apply_status="applied",
    )
    failed = sync_store.mark_envelope_apply_status(
        second.server_cursor,
        apply_status="failed",
        apply_error_code="projection_write_failed",
        apply_error_message="projection database is locked",
    )

    assert applied.apply_status == "applied"
    assert applied.applied_at is not None
    assert failed.apply_status == "failed"
    assert failed.apply_error_code == "projection_write_failed"
    assert failed.apply_error_message == "projection database is locked"
    assert sync_store.list_failed_applies("dataset-1") == [failed]
    assert sync_store.list_accepted_envelopes_for_replay("dataset-1", since_cursor=0) == [
        applied,
        failed,
    ]


def test_device_cursor_upsert_and_fetch(sync_store: SyncV2Store):
    sync_store.upsert_device(_device())
    sync_store.enroll_dataset(_dataset())

    cursor = sync_store.update_device_cursor(
        SyncDeviceCursor(
            dataset_id="dataset-1",
            device_id="device-1",
            domain="notes.note",
            last_pulled_sequence=42,
        )
    )
    fetched = sync_store.get_device_cursor("dataset-1", "device-1", "notes.note")

    assert fetched == cursor
    assert fetched.last_pulled_sequence == 42


def test_device_cursor_rejects_missing_dataset_and_unenrolled_domain(sync_store: SyncV2Store):
    sync_store.upsert_device(_device())
    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.update_device_cursor(
            SyncDeviceCursor(
                dataset_id="missing-dataset",
                device_id="device-1",
                domain="notes.note",
                last_pulled_sequence=1,
            )
        )

    sync_store.enroll_dataset(_dataset(domains=["notes.note"]))

    with pytest.raises(SyncInvalidDomainError):
        sync_store.update_device_cursor(
            SyncDeviceCursor(
                dataset_id="dataset-1",
                device_id="device-1",
                domain="chat.conversation",
                last_pulled_sequence=1,
            )
        )


def test_adapter_cursor_v1_dual_writes_but_v2_does_not_project_legacy(
    sync_store: SyncV2Store,
) -> None:
    sync_store.upsert_device(_device())
    sync_store.enroll_dataset(_dataset())

    v1 = sync_store.update_device_cursor(
        SyncDeviceCursor(
            dataset_id="dataset-1",
            device_id="device-1",
            domain="attachment.ref",
            adapter_version=1,
            last_pulled_sequence=8,
            max_delivered_sequence=7,
        )
    )
    v2 = sync_store.update_device_cursor(
        SyncDeviceCursor(
            dataset_id="dataset-1",
            device_id="device-1",
            domain="attachment.ref",
            adapter_version=2,
            last_pulled_sequence=3,
            max_delivered_sequence=3,
        )
    )

    legacy = sync_store.db.execute(
        "SELECT last_pulled_sequence FROM sync_device_cursors "
        "WHERE dataset_id = ? AND device_id = ? AND domain = ?",
        ("dataset-1", "device-1", "attachment.ref"),
    ).rows[0]
    assert v1.adapter_version == 1
    assert v2.adapter_version == 2
    assert legacy["last_pulled_sequence"] == 8
    assert sync_store.get_device_cursor(
        "dataset-1", "device-1", "attachment.ref", adapter_version=1
    ) == v1
    assert sync_store.get_device_cursor(
        "dataset-1", "device-1", "attachment.ref", adapter_version=2
    ) == v2


def test_version_ack_is_monotonic_bounded_by_exact_delivered_watermark_and_dual_writes_v1(
    sync_store: SyncV2Store,
) -> None:
    sync_store.upsert_device(_device())
    sync_store.enroll_dataset(_dataset())
    for adapter_version, scanned, delivered in ((1, 9, 7), (2, 4, 4)):
        sync_store.update_device_cursor(
            SyncDeviceCursor(
                dataset_id="dataset-1",
                device_id="device-1",
                domain="attachment.ref",
                adapter_version=adapter_version,
                last_pulled_sequence=scanned,
                max_delivered_sequence=delivered,
            )
        )

    v1 = sync_store.upsert_device_domain_ack(
        SyncDeviceDomainAckCreate(
            dataset_id="dataset-1",
            device_id="device-1",
            domain="attachment.ref",
            adapter_version=1,
            through_server_sequence=7,
            applied_at="2026-05-23T18:30:00+00:00",
        )
    )
    v2 = sync_store.upsert_device_domain_ack(
        SyncDeviceDomainAckCreate(
            dataset_id="dataset-1",
            device_id="device-1",
            domain="attachment.ref",
            adapter_version=2,
            through_server_sequence=4,
            applied_at="2026-05-23T18:31:00+00:00",
        )
    )
    stale_v2 = sync_store.upsert_device_domain_ack(
        SyncDeviceDomainAckCreate(
            dataset_id="dataset-1",
            device_id="device-1",
            domain="attachment.ref",
            adapter_version=2,
            through_server_sequence=2,
            applied_at="2026-05-23T18:29:00+00:00",
        )
    )

    with pytest.raises(SyncStoreError, match="delivered watermark"):
        sync_store.upsert_device_domain_ack(
            SyncDeviceDomainAckCreate(
                dataset_id="dataset-1",
                device_id="device-1",
                domain="attachment.ref",
                adapter_version=1,
                through_server_sequence=8,
                applied_at="2026-05-23T18:32:00+00:00",
            )
        )
    with pytest.raises(SyncStoreError, match="delivered watermark"):
        sync_store.upsert_device_domain_ack(
            SyncDeviceDomainAckCreate(
                dataset_id="dataset-1",
                device_id="device-1",
                domain="attachment.ref",
                adapter_version=2,
                through_server_sequence=5,
                applied_at="2026-05-23T18:32:00+00:00",
            )
        )

    legacy = sync_store.db.execute(
        "SELECT through_server_sequence FROM sync_device_domain_acks "
        "WHERE dataset_id = ? AND device_id = ? AND domain = ?",
        ("dataset-1", "device-1", "attachment.ref"),
    ).rows[0]
    assert v1.through_server_sequence == 7
    assert v2.through_server_sequence == 4
    assert stale_v2.through_server_sequence == 4
    assert stale_v2.applied_at == v2.applied_at
    assert legacy["through_server_sequence"] == 7


def test_adapter_cursor_and_version_ack_postgres_concurrent_writes_are_monotonic(
    pg_database_config: DatabaseConfig,
) -> None:
    backends = [
        DatabaseBackendFactory.create_backend(pg_database_config)
        for _ in range(3)
    ]
    stores = [SyncV2Store(SyncDatabase(backend=backend)) for backend in backends]
    setup, low, high = stores
    setup.upsert_device(_device())
    setup.enroll_dataset(_dataset())

    def force_low_write_after_high_commit(
        store: SyncV2Store,
        *,
        select_marker: str,
        low_write,
        high_write,
    ) -> None:
        low_selected = Event()
        high_committed = Event()
        original_execute = store.db.execute
        paused = False

        def delayed_execute(statement, params=None, *, connection=None):
            nonlocal paused
            result = original_execute(statement, params, connection=connection)
            if select_marker in " ".join(statement.split()) and not paused:
                paused = True
                low_selected.set()
                assert high_committed.wait(10)
            return result

        store.db.execute = delayed_execute  # type: ignore[method-assign]
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                low_future = executor.submit(low_write)

                def commit_high():
                    assert low_selected.wait(10)
                    try:
                        return high_write()
                    finally:
                        high_committed.set()

                high_future = executor.submit(commit_high)
                high_future.result(timeout=15)
                low_future.result(timeout=15)
        finally:
            store.db.execute = original_execute  # type: ignore[method-assign]

    force_low_write_after_high_commit(
        low,
        select_marker="SELECT * FROM sync_device_adapter_cursors",
        low_write=lambda: low.update_device_cursor(
            SyncDeviceCursor(
                dataset_id="dataset-1",
                device_id="device-1",
                domain="notes.note",
                adapter_version=2,
                last_pulled_sequence=10,
                max_delivered_sequence=10,
            )
        ),
        high_write=lambda: high.update_device_cursor(
            SyncDeviceCursor(
                dataset_id="dataset-1",
                device_id="device-1",
                domain="notes.note",
                adapter_version=2,
                last_pulled_sequence=20,
                max_delivered_sequence=20,
            )
        ),
    )
    cursor = setup.get_device_cursor(
        "dataset-1", "device-1", "notes.note", adapter_version=2
    )
    assert cursor is not None and cursor.last_pulled_sequence == 20
    assert cursor.max_delivered_sequence == 20

    force_low_write_after_high_commit(
        low,
        select_marker="SELECT * FROM sync_device_adapter_domain_acks",
        low_write=lambda: low.upsert_device_domain_ack(
            SyncDeviceDomainAckCreate(
                dataset_id="dataset-1",
                device_id="device-1",
                domain="notes.note",
                adapter_version=2,
                through_server_sequence=10,
                applied_at="2026-05-23T18:29:00+00:00",
            )
        ),
        high_write=lambda: high.upsert_device_domain_ack(
            SyncDeviceDomainAckCreate(
                dataset_id="dataset-1",
                device_id="device-1",
                domain="notes.note",
                adapter_version=2,
                through_server_sequence=20,
                applied_at="2026-05-23T18:30:00+00:00",
            )
        ),
    )
    ack = setup.get_device_domain_ack(
        "dataset-1", "device-1", "notes.note", adapter_version=2
    )
    assert ack is not None and ack.through_server_sequence == 20

    force_low_write_after_high_commit(
        low,
        select_marker="SELECT * FROM sync_device_adapter_cursors",
        low_write=lambda: low.update_device_cursor(
            SyncDeviceCursor(
                dataset_id="dataset-1",
                device_id="device-1",
                domain="notes.note",
                adapter_version=1,
                last_pulled_sequence=30,
                max_delivered_sequence=30,
            )
        ),
        high_write=lambda: high.update_device_cursor(
            SyncDeviceCursor(
                dataset_id="dataset-1",
                device_id="device-1",
                domain="notes.note",
                adapter_version=1,
                last_pulled_sequence=40,
                max_delivered_sequence=40,
            )
        ),
    )
    cursor = setup.get_device_cursor(
        "dataset-1", "device-1", "notes.note", adapter_version=1
    )
    legacy_cursor = setup.db.execute(
        "SELECT last_pulled_sequence FROM sync_device_cursors "
        "WHERE dataset_id = ? AND device_id = ? AND domain = ?",
        ("dataset-1", "device-1", "notes.note"),
    ).rows[0]
    assert cursor is not None and cursor.last_pulled_sequence == 40
    assert cursor.max_delivered_sequence == 40
    assert legacy_cursor["last_pulled_sequence"] == 40

    force_low_write_after_high_commit(
        low,
        select_marker="SELECT * FROM sync_device_adapter_domain_acks",
        low_write=lambda: low.upsert_device_domain_ack(
            SyncDeviceDomainAckCreate(
                dataset_id="dataset-1",
                device_id="device-1",
                domain="notes.note",
                adapter_version=1,
                through_server_sequence=30,
                applied_at="2026-05-23T18:31:00+00:00",
            )
        ),
        high_write=lambda: high.upsert_device_domain_ack(
            SyncDeviceDomainAckCreate(
                dataset_id="dataset-1",
                device_id="device-1",
                domain="notes.note",
                adapter_version=1,
                through_server_sequence=40,
                applied_at="2026-05-23T18:32:00+00:00",
            )
        ),
    )
    ack = setup.get_device_domain_ack(
        "dataset-1", "device-1", "notes.note", adapter_version=1
    )
    legacy_ack = setup.db.execute(
        "SELECT through_server_sequence FROM sync_device_domain_acks "
        "WHERE dataset_id = ? AND device_id = ? AND domain = ?",
        ("dataset-1", "device-1", "notes.note"),
    ).rows[0]
    assert ack is not None and ack.through_server_sequence == 40
    assert legacy_ack["through_server_sequence"] == 40

    for backend in backends:
        backend.get_pool().close_all()


def test_adapter_cursor_and_version_ack_upserts_apply_monotonic_max_in_database() -> None:
    cursor_source = " ".join(inspect.getsource(SyncDatabase.update_device_cursor).split())
    ack_source = " ".join(inspect.getsource(SyncDatabase.upsert_device_domain_ack).split())

    assert (
        "last_pulled_sequence = CASE WHEN excluded.last_pulled_sequence > "
        "sync_device_adapter_cursors.last_pulled_sequence"
    ) in cursor_source
    assert (
        "last_pulled_sequence = CASE WHEN excluded.last_pulled_sequence > "
        "sync_device_cursors.last_pulled_sequence"
    ) in cursor_source
    assert (
        "through_server_sequence = CASE WHEN excluded.through_server_sequence > "
        "sync_device_adapter_domain_acks.through_server_sequence"
    ) in ack_source
    assert (
        "through_server_sequence = CASE WHEN excluded.through_server_sequence > "
        "sync_device_domain_acks.through_server_sequence"
    ) in ack_source


def test_blob_id_ack_requires_authorized_immutable_digest_and_preserves_replacement_evidence(
    sync_store: SyncV2Store,
) -> None:
    sync_store.upsert_device(_device())
    sync_store.enroll_dataset(_dataset())
    digest_one = "sha256:" + "a" * 64
    digest_two = "sha256:" + "b" * 64
    wrong_owner_digest = "sha256:" + "c" * 64
    for blob_id, digest in (("blob-1", digest_one), ("blob-2", digest_two)):
        sync_store.complete_blob_upload(
            SyncBlobObjectCreate(
                blob_id=blob_id,
                dataset_id="dataset-1",
                owner_user_id="user-1",
                attachment_id="legacy-provenance",
                payload_hash=digest,
                content_type="application/octet-stream",
                size_bytes=1,
                storage_backend="local_fs",
                storage_key=f"blobs/{blob_id}",
            )
        )
    now = utcnow_iso()
    sync_store.db.execute(
        "INSERT INTO sync_blob_objects "
        "(blob_id, dataset_id, owner_user_id, attachment_id, payload_hash, content_type, "
        "size_bytes, encryption_policy, storage_backend, storage_key, status, ref_count, "
        "metadata_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "blob-wrong-owner",
            "dataset-1",
            "user-2",
            "legacy-provenance",
            wrong_owner_digest,
            "application/octet-stream",
            1,
            "server_trusted_v1",
            "local_fs",
            "blobs/blob-wrong-owner",
            "available",
            1,
            "{}",
            now,
            now,
        ),
    )

    first = sync_store.upsert_device_blob_id_ack(
        SyncDeviceBlobIdAckCreate(
            dataset_id="dataset-1",
            device_id="device-1",
            blob_id="blob-1",
            payload_hash=digest_one,
            verified_at="2026-05-23T18:31:00+00:00",
        )
    )
    second = sync_store.upsert_device_blob_id_ack(
        SyncDeviceBlobIdAckCreate(
            dataset_id="dataset-1",
            device_id="device-1",
            blob_id="blob-2",
            payload_hash=digest_two,
            verified_at="2026-05-23T18:32:00+00:00",
        )
    )
    with pytest.raises(SyncStoreError, match="digest"):
        sync_store.upsert_device_blob_id_ack(
            SyncDeviceBlobIdAckCreate(
                dataset_id="dataset-1",
                device_id="device-1",
                blob_id="blob-1",
                payload_hash=digest_two,
                verified_at="2026-05-23T18:33:00+00:00",
            )
        )
    with pytest.raises(SyncStoreError, match="not_authorized"):
        sync_store.upsert_device_blob_id_ack(
            SyncDeviceBlobIdAckCreate(
                dataset_id="dataset-1",
                device_id="device-1",
                blob_id="blob-wrong-owner",
                payload_hash=wrong_owner_digest,
                verified_at="2026-05-23T18:33:00+00:00",
            )
        )

    summary = sync_store.list_device_acknowledgments("dataset-1", "device-1")
    assert first.blob_id == "blob-1"
    assert second.blob_id == "blob-2"
    assert [(ack.blob_id, ack.payload_hash) for ack in summary.blob_id_acks] == [
        ("blob-1", digest_one),
        ("blob-2", digest_two),
    ]

def test_conflict_insert_list_and_resolve_lifecycle(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())

    inserted = sync_store.insert_conflict(_conflict())

    assert inserted.status == "unresolved"
    assert sync_store.get_conflict("conflict-1") == inserted
    assert sync_store.get_conflict("missing-conflict") is None
    assert sync_store.list_conflicts("dataset-1") == [inserted]
    assert sync_store.list_conflicts("dataset-1", status="resolved") == []

    resolved = sync_store.resolve_conflict(
        "conflict-1",
        status="resolved",
        resolved_by_envelope_id="env-resolution",
        resolved_by_device_id="device-1",
        resolution_action="merge",
        resolution_notes="Merged locally",
    )

    assert resolved.status == "resolved"
    assert resolved.resolved_by_envelope_id == "env-resolution"
    assert resolved.resolved_at is not None
    assert sync_store.list_conflicts("dataset-1", status="unresolved") == []
    assert sync_store.list_conflicts("dataset-1", status="resolved") == [resolved]


def test_resolve_conflict_preserves_existing_resolution_metadata(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    sync_store.insert_conflict(_conflict())

    first = sync_store.resolve_conflict(
        "conflict-1",
        server_cursor=10,
        status="resolved",
        resolved_by_envelope_id="srv_env_first",
        resolved_by_device_id="device-1",
        resolution_action="overwrite",
        resolution_notes="first decision",
    )
    replayed = sync_store.resolve_conflict(
        "conflict-1",
        server_cursor=10,
        status="resolved",
        resolved_by_envelope_id="srv_env_first",
        resolved_by_device_id="device-1",
        resolution_action="overwrite",
        resolution_notes="first decision",
    )

    assert replayed == first

    with pytest.raises(SyncStoreError, match="already resolved"):
        sync_store.resolve_conflict(
            "conflict-1",
            server_cursor=11,
            status="resolved",
            resolved_by_envelope_id="srv_env_second",
            resolved_by_device_id="device-2",
            resolution_action="duplicate_rename",
            resolution_notes="second decision",
        )

    assert sync_store.get_conflict("conflict-1") == first


def test_conflict_resolution_claim_and_finalize_lifecycle(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    sync_store.insert_conflict(_conflict())

    claimed = sync_store.claim_conflict_resolution(
        "conflict-1",
        dataset_id="dataset-1",
        resolved_by_device_id="device-1",
        resolution_action="overwrite",
        resolution_notes="first decision",
    )

    assert claimed.status == "unresolved"
    assert claimed.resolution_action == "overwrite"
    assert claimed.resolved_by_device_id == "device-1"
    assert claimed.resolution_notes == "first decision"
    assert claimed.resolved_by_envelope_id is None

    with pytest.raises(SyncStoreError, match="already claimed"):
        sync_store.claim_conflict_resolution(
            "conflict-1",
            dataset_id="dataset-1",
            resolved_by_device_id="device-2",
            resolution_action="duplicate_rename",
            resolution_notes="second decision",
        )

    assert sync_store.get_conflict("conflict-1") == claimed

    resolved = sync_store.resolve_conflict(
        "conflict-1",
        dataset_id="dataset-1",
        server_cursor=10,
        status="resolved",
        resolved_by_envelope_id="srv_env_first",
        resolved_by_device_id="device-1",
        resolution_action="overwrite",
        resolution_notes="first decision",
    )
    replayed = sync_store.resolve_conflict(
        "conflict-1",
        dataset_id="dataset-1",
        server_cursor=10,
        status="resolved",
        resolved_by_envelope_id="srv_env_first",
        resolved_by_device_id="device-1",
        resolution_action="overwrite",
        resolution_notes="first decision",
    )

    assert resolved.status == "resolved"
    assert resolved.resolution_action == "overwrite"
    assert resolved.resolved_by_envelope_id == "srv_env_first"
    assert replayed == resolved

    with pytest.raises(SyncStoreError, match="already resolved"):
        sync_store.resolve_conflict(
            "conflict-1",
            dataset_id="dataset-1",
            server_cursor=11,
            status="resolved",
            resolved_by_envelope_id="srv_env_second",
            resolved_by_device_id="device-2",
            resolution_action="duplicate_rename",
            resolution_notes="second decision",
        )

    assert sync_store.get_conflict("conflict-1") == resolved


def test_conflict_rejects_missing_dataset_and_unenrolled_domain(sync_store: SyncV2Store):
    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.insert_conflict(_conflict(dataset_id="missing-dataset"))

    sync_store.enroll_dataset(_dataset(domains=["notes.note"]))

    with pytest.raises(SyncInvalidDomainError):
        sync_store.insert_conflict(_conflict(domain="chat.conversation"))


def test_key_records_store_wrapped_blobs_without_plaintext_keys(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())

    stored = sync_store.store_key_record(_key_record())
    duplicate = sync_store.store_key_record(_key_record())
    records = sync_store.list_key_records("dataset-1", user_id="user-1")
    columns = {column["name"] for column in sync_store.db.backend.get_table_info("sync_key_records")}

    assert duplicate.key_record_id == stored.key_record_id
    assert duplicate.created_at == stored.created_at
    assert duplicate.recovery_hint == "personal laptop"
    assert duplicate.user_id == "user-1"
    assert records == [duplicate]
    assert records[0].user_id == "user-1"
    assert records[0].wrapped_key_blob == "wrapped:opaque"
    assert not hasattr(records[0], "plaintext_key")
    assert "user_id" in columns
    assert all("plaintext" not in column_name.lower() for column_name in columns)

    with pytest.raises(TypeError):
        sync_store.list_key_records("dataset-1")

    with pytest.raises(SyncStoreError):
        sync_store.list_key_records("dataset-1", user_id="")


def test_key_records_store_epoch_and_rotation_state_metadata(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())

    stored = sync_store.store_key_record(
        _key_record(
            encryption_policy="passphrase_wrapped_v1",
            key_epoch=2,
            active_from_server_sequence=7,
            superseded_at="2026-05-10T12:30:00+00:00",
            wrapped_for="passphrase",
            rewrap_status="complete",
        )
    )
    duplicate = sync_store.store_key_record(
        _key_record(
            encryption_policy="passphrase_wrapped_v1",
            key_epoch=2,
            active_from_server_sequence=7,
            superseded_at="2026-05-10T12:30:00+00:00",
            wrapped_for="passphrase",
            rewrap_status="complete",
        )
    )
    records = sync_store.list_key_records("dataset-1", user_id="user-1")

    assert duplicate == stored
    assert records == [stored]
    assert stored.encryption_policy == "passphrase_wrapped_v1"
    assert stored.key_epoch == 2
    assert stored.active_from_server_sequence == 7
    assert stored.superseded_at == "2026-05-10T12:30:00+00:00"
    assert stored.wrapped_for == "passphrase"
    assert stored.rewrap_status == "complete"

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.store_key_record(_key_record(key_epoch=3))

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.store_key_record(_key_record(rewrap_status="pending"))


def test_existing_key_record_rows_receive_safe_epoch_defaults(tmp_path: Path):
    db_path = tmp_path / "sync_v2_legacy_key_records.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE sync_key_records (
                key_record_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                device_id TEXT,
                key_purpose TEXT NOT NULL,
                wrapped_key_blob TEXT NOT NULL,
                kdf_metadata_json TEXT NOT NULL DEFAULT '{}',
                recovery_hint TEXT,
                rotation_of_key_record_id TEXT,
                created_at TEXT NOT NULL,
                revoked_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sync_key_records (
                key_record_id, dataset_id, user_id, device_id, key_purpose,
                wrapped_key_blob, kdf_metadata_json, recovery_hint,
                rotation_of_key_record_id, created_at, revoked_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-key-1",
                "dataset-1",
                "user-1",
                "device-1",
                "dataset_recovery",
                "wrapped:legacy",
                '{"algorithm": "scrypt", "salt": "legacy-salt"}',
                "legacy recovery",
                None,
                "2026-05-10T12:00:00+00:00",
                None,
            ),
        )

    store = SyncV2Store(SyncDatabase(sqlite_path=db_path))
    store.enroll_dataset(_dataset())

    columns = {column["name"] for column in store.db.backend.get_table_info("sync_key_records")}
    records = store.list_key_records("dataset-1", user_id="user-1")

    assert {
        "encryption_policy",
        "key_epoch",
        "active_from_server_sequence",
        "superseded_at",
        "wrapped_for",
        "rewrap_status",
    }.issubset(columns)
    assert len(records) == 1
    assert records[0].key_record_id == "legacy-key-1"
    assert records[0].encryption_policy == "server_trusted_v1"
    assert records[0].key_epoch == 1
    assert records[0].active_from_server_sequence is None
    assert records[0].superseded_at is None
    assert records[0].wrapped_for == "recovery"
    assert records[0].rewrap_status == "not_required"


def test_key_records_are_scoped_by_user(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())

    sync_store.store_key_record(_key_record(key_record_id="key-user-1", user_id="user-1"))
    sync_store.store_key_record(_key_record(key_record_id="key-user-2", user_id="user-2"))

    user_1_records = sync_store.list_key_records("dataset-1", user_id="user-1")
    user_2_records = sync_store.list_key_records("dataset-1", user_id="user-2")

    assert [record.key_record_id for record in user_1_records] == ["key-user-1"]
    assert [record.key_record_id for record in user_2_records] == ["key-user-2"]


def test_key_record_rejects_missing_dataset(sync_store: SyncV2Store):
    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.store_key_record(_key_record(dataset_id="missing-dataset"))


def test_key_record_duplicate_drift_raises(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())

    sync_store.store_key_record(_key_record())

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.store_key_record(_key_record(wrapped_key_blob="wrapped:changed"))

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.store_key_record(_key_record(user_id="user-2"))


def test_attachment_store_is_idempotent_and_keeps_ciphertext_out_of_manifest(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())

    stored = sync_store.store_attachment(_attachment())
    duplicate = sync_store.store_attachment(_attachment())
    stats = sync_store.summarize_restore_manifest_dataset(
        "dataset-1",
        user_id="user-1",
        domains=["attachment.ref"],
    )

    assert stored.stored is True
    assert duplicate.stored is False
    assert duplicate.attachment_id == stored.attachment_id
    assert duplicate.payload_hash == "sha256:attachment"
    assert duplicate.payload_ciphertext == "ciphertext:attachment"
    assert stats.attachment_availability == {"available": 1}
    assert stats.attachment_size_classes == {"small": 1}
    assert "ciphertext:attachment" not in repr(stats)


def test_attachment_store_rejects_duplicate_drift_and_unenrolled_domain(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset(domains=["attachment.ref"]))
    sync_store.store_attachment(_attachment())

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.store_attachment(_attachment(payload_hash="sha256:changed"))

    with pytest.raises(SyncInvalidDomainError):
        sync_store.store_attachment(
            _attachment(
                attachment_id="attachment-chat",
                domain="chat.conversation",
                object_id="conversation-1",
                payload_hash="sha256:chat-attachment",
            )
        )


def test_attachment_store_restore_summary_respects_domain_filter(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset(domains=["attachment.ref", "chat.conversation"]))
    sync_store.store_attachment(_attachment(domain="attachment.ref", size_bytes=512))
    sync_store.store_attachment(
        _attachment(
            attachment_id="attachment-chat",
            domain="chat.conversation",
            object_id="conversation-1",
            size_bytes=2_097_152,
            payload_hash="sha256:chat-attachment",
        )
    )

    notes_stats = sync_store.summarize_restore_manifest_dataset(
        "dataset-1",
        user_id="user-1",
        domains=["attachment.ref"],
    )
    all_stats = sync_store.summarize_restore_manifest_dataset(
        "dataset-1",
        user_id="user-1",
        domains=["attachment.ref", "chat.conversation"],
    )

    assert notes_stats.attachment_availability == {"available": 1}
    assert notes_stats.attachment_size_classes == {"small": 1}
    assert all_stats.attachment_availability == {"available": 2}
    assert all_stats.attachment_size_classes == {"medium": 1, "small": 1}

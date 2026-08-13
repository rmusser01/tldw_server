from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from uuid import UUID

import pytest

import tldw_Server_API.app.core.DB_Management.Sync_DB as sync_db_module
from tldw_Server_API.app.core.DB_Management.backends.base import QueryResult
from tldw_Server_API.app.core.DB_Management.Sync_DB import (
    SYNC_POSTGRES_SCHEMA,
    SYNC_SQLITE_SCHEMA,
    SyncDatabase,
)
from tldw_Server_API.app.core.Sync.v2.errors import (
    SyncDatasetNotFoundError,
    SyncStoreError,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncDatasetCreate,
    SyncEnvelopeCreate,
    sync_v2_attachment_ref_v2_is_writable,
)
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


@pytest.fixture()
def sync_store(tmp_path: Path) -> SyncV2Store:
    return SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db"))


def _enroll_notes_dataset(sync_store: SyncV2Store) -> None:
    sync_store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="owner-1",
            domains=["notes.note"],
        )
    )


@pytest.mark.unit
def test_attachment_bootstrap_begin_is_idempotent_and_enrolls_v2(
    sync_store: SyncV2Store,
) -> None:
    _enroll_notes_dataset(sync_store)

    begun = sync_store.begin_notes_attachment_bootstrap(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
    )
    replay = sync_store.begin_notes_attachment_bootstrap(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-replacement-must-not-win",
    )

    assert begun.metadata["notes_attachment_v2"] == {
        "bootstrap_id": "bootstrap-stable",
        "state": "initializing",
        "target_adapter_version": 2,
        "captured_count": 0,
        "expected_count": 0,
        "source_hash": None,
        "source_cursor": None,
        "error_code": None,
    }
    assert replay.metadata["notes_attachment_v2"] == begun.metadata["notes_attachment_v2"]
    assert "attachment.ref" in replay.domains
    states = sync_store.db.execute(
        "SELECT adapter_version, server_sequence FROM sync_domain_state "
        "WHERE dataset_id = ? AND domain = ? ORDER BY adapter_version",
        ("dataset-1", "attachment.ref"),
    ).rows
    assert states == [{"adapter_version": 2, "server_sequence": 0}]


@pytest.mark.unit
def test_attachment_bootstrap_transition_is_cas_and_failure_code_is_safe(
    sync_store: SyncV2Store,
) -> None:
    _enroll_notes_dataset(sync_store)
    sync_store.begin_notes_attachment_bootstrap(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
    )
    source_hash = hashlib.sha256(b"source-set").hexdigest()

    progressed = sync_store.transition_notes_attachment_bootstrap(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
        expected_state="initializing",
        state="initializing",
        captured_count=1,
        expected_count=2,
        source_hash=source_hash,
        source_cursor='{"note_id":"note-1","source_key_hash":"safe"}',
    )

    assert progressed.metadata["notes_attachment_v2"]["captured_count"] == 1
    assert progressed.metadata["notes_attachment_v2"]["source_hash"] == source_hash
    with pytest.raises(
        SyncStoreError,
        match="notes_attachment_bootstrap_compare_and_set_failed",
    ):
        sync_store.transition_notes_attachment_bootstrap(
            "dataset-1",
            owner_user_id="owner-1",
            bootstrap_id="wrong-bootstrap",
            expected_state="initializing",
            state="failed",
            captured_count=1,
            expected_count=2,
            source_hash=source_hash,
            source_cursor=None,
            error_code="notes_attachment_source_changed",
        )
    with pytest.raises(SyncStoreError, match="failure code is invalid"):
        sync_store.transition_notes_attachment_bootstrap(
            "dataset-1",
            owner_user_id="owner-1",
            bootstrap_id="bootstrap-stable",
            expected_state="initializing",
            state="failed",
            captured_count=1,
            expected_count=2,
            source_hash=source_hash,
            source_cursor=None,
            error_code="/private/owner/notes/file.txt",
        )
    failed = sync_store.transition_notes_attachment_bootstrap(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
        expected_state="initializing",
        state="failed",
        captured_count=1,
        expected_count=2,
        source_hash=source_hash,
        source_cursor=None,
        error_code="notes_attachment_source_changed",
    )
    assert failed.metadata["notes_attachment_v2"]["error_code"] == (
        "notes_attachment_source_changed"
    )


@pytest.mark.unit
def test_attachment_bootstrap_progress_and_source_hash_are_monotonic(
    sync_store: SyncV2Store,
) -> None:
    _enroll_notes_dataset(sync_store)
    sync_store.begin_notes_attachment_bootstrap(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
    )
    first_hash = hashlib.sha256(b"source-set").hexdigest()
    sync_store.transition_notes_attachment_bootstrap(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
        expected_state="initializing",
        state="initializing",
        captured_count=1,
        expected_count=2,
        source_hash=first_hash,
        source_cursor='{"source_key_hash":"safe"}',
    )

    with pytest.raises(SyncStoreError, match="progress_regressed"):
        sync_store.transition_notes_attachment_bootstrap(
            "dataset-1",
            owner_user_id="owner-1",
            bootstrap_id="bootstrap-stable",
            expected_state="initializing",
            state="initializing",
            captured_count=0,
            expected_count=2,
            source_hash=first_hash,
            source_cursor=None,
        )
    with pytest.raises(SyncStoreError, match="source_changed"):
        sync_store.transition_notes_attachment_bootstrap(
            "dataset-1",
            owner_user_id="owner-1",
            bootstrap_id="bootstrap-stable",
            expected_state="initializing",
            state="initializing",
            captured_count=1,
            expected_count=2,
            source_hash=hashlib.sha256(b"different").hexdigest(),
            source_cursor=None,
        )


@pytest.mark.unit
def test_attachment_bootstrap_ready_requires_exact_counts_and_verification(
    sync_store: SyncV2Store,
) -> None:
    _enroll_notes_dataset(sync_store)
    sync_store.begin_notes_attachment_bootstrap(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
    )
    source_hash = hashlib.sha256(b"source-set").hexdigest()

    with pytest.raises(
        SyncStoreError,
        match="notes_attachment_bootstrap_verification_failed",
    ):
        sync_store.transition_notes_attachment_bootstrap(
            "dataset-1",
            owner_user_id="owner-1",
            bootstrap_id="bootstrap-stable",
            expected_state="initializing",
            state="ready",
            captured_count=2,
            expected_count=2,
            source_hash=source_hash,
            source_cursor=None,
            ready_verifier=lambda: False,
        )
    ready = sync_store.transition_notes_attachment_bootstrap(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
        expected_state="initializing",
        state="ready",
        captured_count=2,
        expected_count=2,
        source_hash=source_hash,
        source_cursor=None,
        ready_verifier=lambda: True,
    )
    assert ready.metadata["notes_attachment_v2"]["state"] == "ready"
    assert ready.metadata["notes_attachment_v2"]["target_adapter_version"] == 2


@pytest.mark.unit
def test_attachment_bootstrap_transition_requires_owner_at_sql_boundary(
    sync_store: SyncV2Store,
) -> None:
    _enroll_notes_dataset(sync_store)
    sync_store.begin_notes_attachment_bootstrap(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
    )

    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.transition_notes_attachment_bootstrap(
            "dataset-1",
            owner_user_id="owner-2",
            bootstrap_id="bootstrap-stable",
            expected_state="initializing",
            state="failed",
            captured_count=0,
            expected_count=0,
            source_hash=None,
            source_cursor=None,
            error_code="notes_attachment_source_changed",
        )


@pytest.mark.unit
def test_attachment_bootstrap_keeps_v1_pullable_and_v2_closed_until_ready(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="owner-1",
            domains=["notes.note", "attachment.ref"],
        )
    )
    legacy = sync_store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id="dataset-1",
            client_envelope_id="legacy-ref-1",
            domain="attachment.ref",
            operation="upsert",
            object_id="8b61cb20-92b5-4a41-8f38-763b24e7c2d0",
            adapter_version=1,
            payload={
                "attachment_id": "8b61cb20-92b5-4a41-8f38-763b24e7c2d0",
                "parent_domain": "notes.note",
                "parent_object_id": "note-1",
                "content_type": "application/pdf",
                "size_bytes": 7,
                "payload_hash": "sha256:" + "a" * 64,
                "availability": "available",
            },
            payload_hash="sha256:" + "a" * 64,
        )
    )

    initializing = sync_store.begin_notes_attachment_bootstrap(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
    )

    assert sync_store.list_envelopes_after(
        "dataset-1",
        0,
        domains=["attachment.ref"],
        adapter_versions=[1],
    ) == [legacy]
    assert not sync_v2_attachment_ref_v2_is_writable(
        initializing,
        notes_attachment_sync_enabled=True,
        supports_attachments=True,
    )


@pytest.mark.unit
def test_source_map_allocates_one_uuid4_per_source_key(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enroll_notes_dataset(sync_store)
    sync_store.begin_notes_attachment_bootstrap(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
    )
    allocated: list[str] = []
    real_uuid4 = sync_db_module.uuid4

    def tracked_uuid4():
        value = real_uuid4()
        allocated.append(str(value))
        return value

    monkeypatch.setattr(sync_db_module, "uuid4", tracked_uuid4)

    first = sync_store.resolve_notes_attachment_source_map(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
        note_id="note-1",
        source_key="notes_attachments/note-1/report.pdf",
    )
    replay = sync_store.resolve_notes_attachment_source_map(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
        note_id="note-1",
        source_key="notes_attachments/note-1/report.pdf",
    )
    second = sync_store.resolve_notes_attachment_source_map(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
        note_id="note-1",
        source_key="notes_attachments/note-1/appendix.pdf",
    )

    assert replay == first
    assert len(allocated) == 2
    assert first.attachment_id != second.attachment_id
    assert UUID(first.attachment_id).version == 4
    assert first.source_key_hash == "sha256:" + hashlib.sha256(
        b"notes_attachments/note-1/report.pdf"
    ).hexdigest()
    columns = sync_store.db.execute(
        "PRAGMA table_info(sync_notes_attachment_source_map)"
    ).rows
    assert "source_key" not in {row["name"] for row in columns}


@pytest.mark.unit
def test_cleanup_candidate_keeps_internal_path_out_of_repr_and_pages_bounded(
    sync_store: SyncV2Store,
) -> None:
    _enroll_notes_dataset(sync_store)
    sync_store.begin_notes_attachment_bootstrap(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
    )
    mapping = sync_store.resolve_notes_attachment_source_map(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
        note_id="note-1",
        source_key="notes_attachments/note-1/report.pdf",
    )
    candidate = sync_store.record_notes_attachment_cleanup_candidate(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
        source_key="notes_attachments/note-1/report.pdf",
        source_relative_path="notes_attachments/note-1/report.pdf",
        source_blob_hash="sha256:" + hashlib.sha256(b"payload").hexdigest(),
        source_size_bytes=7,
        source_modified_ns=123,
    )
    replay = sync_store.record_notes_attachment_cleanup_candidate(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
        source_key="notes_attachments/note-1/report.pdf",
        source_relative_path="notes_attachments/note-1/report.pdf",
        source_blob_hash="sha256:" + hashlib.sha256(b"payload").hexdigest(),
        source_size_bytes=7,
        source_modified_ns=123,
    )

    assert replay == candidate
    assert candidate.attachment_id == mapping.attachment_id
    assert candidate.source_path_hash == mapping.source_key_hash
    assert candidate.source_relative_path == "notes_attachments/note-1/report.pdf"
    assert candidate.source_relative_path not in repr(candidate)
    assert candidate.source_path_hash in repr(candidate)
    assert sync_store.list_notes_attachment_cleanup_candidates(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
        limit=1,
    ) == (candidate,)
    with pytest.raises(ValueError, match="1..1000"):
        sync_store.list_notes_attachment_cleanup_candidates(
            "dataset-1",
            owner_user_id="owner-1",
            bootstrap_id="bootstrap-stable",
            limit=1_001,
        )


@pytest.mark.unit
def test_cleanup_candidate_rejects_source_key_path_mismatch(
    sync_store: SyncV2Store,
) -> None:
    _enroll_notes_dataset(sync_store)
    sync_store.begin_notes_attachment_bootstrap(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
    )
    sync_store.resolve_notes_attachment_source_map(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
        note_id="note-1",
        source_key="notes_attachments/note-1/report.pdf",
    )

    with pytest.raises(SyncStoreError, match="source path does not match"):
        sync_store.record_notes_attachment_cleanup_candidate(
            "dataset-1",
            owner_user_id="owner-1",
            bootstrap_id="bootstrap-stable",
            source_key="notes_attachments/note-1/report.pdf",
            source_relative_path="notes_attachments/note-1/other.pdf",
            source_blob_hash="sha256:" + hashlib.sha256(b"payload").hexdigest(),
            source_size_bytes=7,
            source_modified_ns=123,
        )


@pytest.mark.unit
def test_cleanup_candidate_schema_rejects_path_hash_identity_drift(
    sync_store: SyncV2Store,
) -> None:
    _enroll_notes_dataset(sync_store)
    sync_store.begin_notes_attachment_bootstrap(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
    )
    mapping = sync_store.resolve_notes_attachment_source_map(
        "dataset-1",
        owner_user_id="owner-1",
        bootstrap_id="bootstrap-stable",
        note_id="note-1",
        source_key="notes_attachments/note-1/report.pdf",
    )

    with pytest.raises(Exception, match="CHECK constraint failed"):
        sync_store.db.execute(
            "INSERT INTO sync_notes_attachment_cleanup_candidates "
            "(dataset_id, bootstrap_id, source_key_hash, attachment_id, "
            "source_relative_path, source_path_hash, source_blob_hash, "
            "source_size_bytes, source_modified_ns, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "dataset-1",
                "bootstrap-stable",
                mapping.source_key_hash,
                mapping.attachment_id,
                "notes_attachments/note-1/report.pdf",
                "sha256:" + "b" * 64,
                "sha256:" + "a" * 64,
                7,
                123,
                "2026-08-13T00:00:00+00:00",
            ),
        )


@pytest.mark.unit
def test_attachment_bootstrap_tables_are_in_both_fresh_schema_contracts() -> None:
    for schema in (SYNC_SQLITE_SCHEMA, SYNC_POSTGRES_SCHEMA):
        assert "sync_notes_attachment_source_map" in schema
        assert "sync_notes_attachment_cleanup_candidates" in schema
        assert "uq_sync_notes_attachment_source_id" in schema
        assert "idx_sync_notes_attachment_cleanup_page" in schema


@pytest.mark.unit
@pytest.mark.parametrize(
    "malformed_table",
    [
        "sync_notes_attachment_source_map",
        "sync_notes_attachment_cleanup_candidates",
    ],
)
def test_attachment_bootstrap_catalog_rejects_malformed_existing_table(
    tmp_path: Path,
    malformed_table: str,
) -> None:
    database_path = tmp_path / "malformed.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(f"CREATE TABLE {malformed_table} (dataset_id TEXT)")

    with pytest.raises(SyncStoreError, match="bootstrap catalog is malformed"):
        SyncDatabase(sqlite_path=database_path)


@pytest.mark.unit
@pytest.mark.parametrize("weaken_constraint", [False, True])
def test_postgres_attachment_bootstrap_catalog_checks_are_exact(
    weaken_constraint: bool,
) -> None:
    database = object.__new__(SyncDatabase)
    columns = [
        {
            "table_name": table,
            "column_name": name,
            "data_type": data_type,
            "is_not_null": True,
        }
        for table, specs in {
            "sync_notes_attachment_cleanup_candidates": [
                ("dataset_id", "text"),
                ("bootstrap_id", "text"),
                ("source_key_hash", "text"),
                ("attachment_id", "text"),
                ("source_relative_path", "text"),
                ("source_path_hash", "text"),
                ("source_blob_hash", "text"),
                ("source_size_bytes", "bigint"),
                ("source_modified_ns", "bigint"),
                ("created_at", "timestamp with time zone"),
            ],
            "sync_notes_attachment_source_map": [
                ("dataset_id", "text"),
                ("bootstrap_id", "text"),
                ("source_key_hash", "text"),
                ("note_id", "text"),
                ("attachment_id", "text"),
                ("created_at", "timestamp with time zone"),
            ],
        }.items()
        for name, data_type in specs
    ]
    uuid_check = (
        "CHECK (attachment_id ~ "
        "'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'::text)"
    )
    hash_check = "CHECK ({column} ~ '^sha256:[0-9a-f]{{64}}$'::text)"
    constraints = [
        {
            "table_name": table,
            "kind": kind,
            "is_validated": True,
            "definition": definition,
        }
        for table, definitions in {
            "sync_notes_attachment_source_map": [
                ("p", "PRIMARY KEY (dataset_id, bootstrap_id, source_key_hash)"),
                ("c", "CHECK (length(dataset_id) > 0)"),
                (
                    "c",
                    "CHECK ((length(bootstrap_id) >= 1) AND "
                    "(length(bootstrap_id) <= 128))",
                ),
                ("c", hash_check.format(column="source_key_hash")),
                ("c", "CHECK (length(note_id) > 0)"),
                ("c", uuid_check),
            ],
            "sync_notes_attachment_cleanup_candidates": [
                ("p", "PRIMARY KEY (dataset_id, bootstrap_id, source_key_hash)"),
                ("c", "CHECK (length(dataset_id) > 0)"),
                (
                    "c",
                    "CHECK ((length(bootstrap_id) >= 1) AND "
                    "(length(bootstrap_id) <= 128))",
                ),
                ("c", hash_check.format(column="source_key_hash")),
                ("c", uuid_check),
                (
                    "c",
                    "CHECK ((length(source_relative_path) >= 1) AND "
                    "(length(source_relative_path) <= 4096))",
                ),
                ("c", hash_check.format(column="source_path_hash")),
                ("c", "CHECK (source_path_hash = source_key_hash)"),
                ("c", hash_check.format(column="source_blob_hash")),
                ("c", "CHECK (source_size_bytes > 0)"),
                ("c", "CHECK (source_modified_ns >= 0)"),
            ],
        }.items()
        for kind, definition in definitions
    ]
    if weaken_constraint:
        constraints[-2]["definition"] += " OR true"
    indexes = [
        {
            "index_name": "idx_sync_notes_attachment_cleanup_page",
            "table_name": "sync_notes_attachment_cleanup_candidates",
            "is_unique": False,
            "is_valid": True,
            "is_ready": True,
            "definition": "CREATE INDEX idx_sync_notes_attachment_cleanup_page "
            "ON public.sync_notes_attachment_cleanup_candidates "
            "USING btree (dataset_id, bootstrap_id, source_key_hash)",
        },
        {
            "index_name": "uq_sync_notes_attachment_source_id",
            "table_name": "sync_notes_attachment_source_map",
            "is_unique": True,
            "is_valid": True,
            "is_ready": True,
            "definition": "CREATE UNIQUE INDEX uq_sync_notes_attachment_source_id "
            "ON public.sync_notes_attachment_source_map "
            "USING btree (dataset_id, attachment_id)",
        },
    ]

    def fake_execute(query: str, *_args, **_kwargs) -> QueryResult:
        if "pg_catalog.pg_attribute" in query:
            rows = columns
        elif "pg_catalog.pg_constraint" in query:
            rows = constraints
        else:
            rows = indexes
        return QueryResult(rows=rows, rowcount=len(rows))

    database.execute = fake_execute
    if not weaken_constraint:
        database._verify_notes_attachment_bootstrap_tables_postgres(
            connection=object()
        )
        return
    with pytest.raises(SyncStoreError, match="bootstrap catalog is malformed"):
        database._verify_notes_attachment_bootstrap_tables_postgres(
            connection=object()
        )

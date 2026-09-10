"""Live PostgreSQL tenancy proof for the Notes attachment registry."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User
from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoint
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore
from tldw_Server_API.app.core.Sync.v2.domain_adapters.attachment_refs import (
    AttachmentRefDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.materializers import AttachmentRefMaterializer
from tldw_Server_API.app.core.Sync.v2.models import SyncDatasetCreate
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.integration


class _NoopRateLimiter:
    async def check_user_rate_limit(
        self,
        user_id: int,
        endpoint: str,
        role: str = "user",
    ) -> tuple[bool, dict[str, object]]:
        return True, {}


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _create_attachment(
    db: CharactersRAGDB,
    *,
    dataset_id: str,
    attachment_id: str,
    note_id: str,
):
    return db.note_attachment_store.create(
        dataset_id=dataset_id,
        attachment_id=attachment_id,
        note_id=note_id,
        file_name="same-name.pdf",
        original_file_name="same-name.pdf",
        content_type="application/pdf",
        size_bytes=10,
        blob_hash="sha256:" + "a" * 64,
        object_hash="sha256:" + "b" * 64,
        created_at="2026-08-11T12:00:00+00:00",
        last_modified="2026-08-11T12:00:00+00:00",
        created_by="device-postgres",
        source_kind="sync",
    )


def test_postgres_v4_bootstrap_namespaces_keyword_indexes(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = object.__new__(CharactersRAGDB)
    db._backend = backend
    db._local = threading.local()
    db._uses_shared_content_backend = False

    try:
        with backend.transaction() as conn:
            db._apply_schema_v4_postgres(conn)
            assert backend.table_exists("chacha_keywords", connection=conn)
            assert not backend.table_exists("keywords", connection=conn)
    finally:
        backend.get_pool().close_all()


def test_postgres_note_attachments_are_two_owner_isolated_and_indexed(
    pg_database_config: DatabaseConfig,
) -> None:
    owner_a = "930001"
    owner_b = "930002"
    dataset_id = f"dataset-{uuid4()}"
    note_a = str(uuid4())
    note_b = str(uuid4())
    attachment_a = str(uuid4())
    attachment_b = str(uuid4())
    backend_a = DatabaseBackendFactory.create_backend(pg_database_config)
    backend_b = DatabaseBackendFactory.create_backend(pg_database_config)
    db_a = CharactersRAGDB(":memory:", client_id=owner_a, backend=backend_a)
    db_b = CharactersRAGDB(":memory:", client_id=owner_b, backend=backend_b)
    ident = backend_a.escape_identifier  # type: ignore[attr-defined]
    role_name = f"note_attachment_rls_{uuid4().hex[:8]}"
    role_created = False

    try:
        db_a.add_note("Owner A", "Body", note_id=note_a)
        db_b.add_note("Owner B", "Body", note_id=note_b)
        row_a = _create_attachment(
            db_a,
            dataset_id=dataset_id,
            attachment_id=attachment_a,
            note_id=note_a,
        )
        row_b = _create_attachment(
            db_b,
            dataset_id=dataset_id,
            attachment_id=attachment_b,
            note_id=note_b,
        )

        assert row_a.file_name == row_b.file_name == "same-name.pdf"
        assert db_a.note_attachment_store.get(dataset_id, attachment_a) == row_a
        assert db_b.note_attachment_store.get(dataset_id, attachment_b) == row_b
        assert db_a.note_attachment_store.get(dataset_id, attachment_b) is None
        assert db_b.note_attachment_store.get(dataset_id, attachment_a) is None

        tombstone = db_a.note_attachment_store.tombstone(
            dataset_id=dataset_id,
            attachment_id=attachment_a,
            expected_version=row_a.version,
            expected_object_hash=row_a.object_hash,
            object_hash="sha256:" + "e" * 64,
            last_modified="2026-08-11T12:01:00+00:00",
            deleted_at="2026-08-11T12:01:00+00:00",
            delete_reason="live PostgreSQL lifecycle proof",
        )
        assert tombstone.deleted is True
        assert db_b.note_attachment_store.get(dataset_id, attachment_a) is None
        restored = db_a.note_attachment_store.restore(
            dataset_id=dataset_id,
            attachment_id=attachment_a,
            expected_version=tombstone.version,
            expected_object_hash=tombstone.object_hash,
            object_hash="sha256:" + "f" * 64,
            last_modified="2026-08-11T12:02:00+00:00",
        )
        assert restored.deleted is False
        assert restored.version == 3
        assert db_b.note_attachment_store.get(dataset_id, attachment_b) == row_b

        with backend_a.transaction() as conn:
            backend_a.execute(
                f"CREATE ROLE {ident(role_name)} NOLOGIN NOSUPERUSER NOBYPASSRLS",
                connection=conn,
            )
            backend_a.execute(
                f"GRANT USAGE ON SCHEMA public TO {ident(role_name)}",
                connection=conn,
            )
            backend_a.execute(
                f"GRANT SELECT ON notes TO {ident(role_name)}",
                connection=conn,
            )
            backend_a.execute(
                f"GRANT SELECT, INSERT, UPDATE ON note_attachments TO {ident(role_name)}",
                connection=conn,
            )
            backend_a.execute(
                f"GRANT {ident(role_name)} TO CURRENT_USER",
                connection=conn,
            )
        role_created = True

        with backend_a.transaction() as conn:
            backend_a.execute(f"SET LOCAL ROLE {ident(role_name)}", connection=conn)
            backend_a.execute(
                "SELECT set_config('app.current_user_id', ?, true)",
                (owner_a,),
                connection=conn,
            )
            principal = backend_a.execute(
                "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user",
                connection=conn,
            ).rows[0]
            assert principal["rolsuper"] is False
            assert principal["rolbypassrls"] is False
            policy = backend_a.execute(
                "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE oid = 'note_attachments'::regclass",
                connection=conn,
            ).rows[0]
            assert policy["relrowsecurity"] and policy["relforcerowsecurity"]
            hidden_update = backend_a.execute(
                "UPDATE note_attachments SET file_name = ? "
                "WHERE client_id = ? AND dataset_id = ? AND attachment_id = ?",
                ("overwrite.pdf", owner_b, dataset_id, attachment_b),
                connection=conn,
            )
            assert hidden_update.rowcount == 0

        with pytest.raises(DatabaseError):
            with backend_a.transaction() as conn:
                backend_a.execute(f"SET LOCAL ROLE {ident(role_name)}", connection=conn)
                backend_a.execute(
                    "SELECT set_config('app.current_user_id', ?, true)",
                    (owner_a,),
                    connection=conn,
                )
                backend_a.execute(
                    "INSERT INTO note_attachments("
                    "client_id, dataset_id, attachment_id, note_id, file_name, "
                    "normalized_file_name, original_file_name, content_type, size_bytes, "
                    "blob_hash, object_hash, version, deleted, created_at, last_modified, "
                    "created_by, source_kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, "
                    "FALSE, ?, ?, ?, ?)",
                    (
                        owner_a,
                        dataset_id,
                        str(uuid4()),
                        note_b,
                        "cross-owner.pdf",
                        "cross-owner.pdf",
                        "cross-owner.pdf",
                        "application/pdf",
                        1,
                        "sha256:" + "c" * 64,
                        "sha256:" + "d" * 64,
                        "2026-08-11T12:00:00+00:00",
                        "2026-08-11T12:00:00+00:00",
                        "device-postgres",
                        "sync",
                    ),
                    connection=conn,
                )

        with db_a.transaction() as conn:
            conn.execute("SET LOCAL enable_seqscan = off")
            plan_rows = conn.execute(
                "EXPLAIN SELECT deleted, attachment_id FROM note_attachments "
                "WHERE client_id = ? AND dataset_id = ? AND note_id = ? "
                "AND deleted <= FALSE AND attachment_id > ? "
                "ORDER BY deleted, attachment_id LIMIT ?",
                (owner_a, dataset_id, note_a, "", 50),
            ).fetchall()
            plan = " ".join(str(next(iter(dict(row).values()))) for row in plan_rows)
        assert "idx_note_attachments_owner_dataset_note_page" in plan

        with db_a.transaction() as conn:
            conn.execute("SET LOCAL enable_seqscan = off")
            all_state_plan_rows = conn.execute(
                "EXPLAIN SELECT attachment_id FROM note_attachments "
                "WHERE client_id = ? AND dataset_id = ? AND note_id = ? "
                "AND attachment_id > ? ORDER BY attachment_id LIMIT ?",
                (owner_a, dataset_id, note_a, "", 50),
            ).fetchall()
            all_state_plan = " ".join(str(next(iter(dict(row).values()))) for row in all_state_plan_rows)
        assert "idx_note_attachments_owner_dataset_note_all_page" in all_state_plan

        with db_a.transaction() as conn:
            conn.execute("SET LOCAL enable_seqscan = off")
            name_plan_rows = conn.execute(
                "EXPLAIN SELECT attachment_id FROM note_attachments "
                "WHERE client_id = ? AND dataset_id = ? AND note_id = ? "
                "AND normalized_file_name = ? AND deleted = FALSE",
                (owner_a, dataset_id, note_a, "same-name.pdf"),
            ).fetchall()
            name_plan = " ".join(str(next(iter(dict(row).values()))) for row in name_plan_rows)
        assert "uq_note_attachments_live_name" in name_plan
    finally:
        if role_created:
            with backend_a.transaction() as conn:
                backend_a.execute(
                    f"REVOKE {ident(role_name)} FROM CURRENT_USER",
                    connection=conn,
                )
                backend_a.execute(
                    f"DROP OWNED BY {ident(role_name)}",
                    connection=conn,
                )
                backend_a.execute(f"DROP ROLE {ident(role_name)}", connection=conn)
        db_a.close_all_connections()
        db_b.close_all_connections()
        backend_a.get_pool().close_all()
        backend_b.get_pool().close_all()


def test_postgres_canonical_content_supports_all_single_byte_range_forms(
    pg_database_config: DatabaseConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "940001"
    dataset_id = f"dataset-{uuid4()}"
    note_id = str(uuid4())
    attachment_id = str(uuid4())
    payload = b"0123456789"
    note_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    sync_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    note_db = CharactersRAGDB(":memory:", client_id=owner, backend=note_backend)
    sync_db = SyncDatabase(backend=sync_backend)
    store = SyncV2Store(sync_db)
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=dataset_id,
            owner_user_id=owner,
            domains=["notes.note", "attachment.ref"],
            metadata={
                "default_personal": True,
                "client_family": "chatbook",
                "notes_attachment_v2": {"state": "ready"},
            },
        )
    )
    note_db.add_note("Range proof", "Body", note_id=note_id)
    service = SyncV2Service(
        store=store,
        adapters=SyncAdapterRegistry([AttachmentRefDomainAdapter(v2_writes_enabled=True)]),
        materializers={"attachment.ref": AttachmentRefMaterializer(note_db)},
        blob_store=LocalSyncBlobStore(tmp_path / "postgres-range-blobs"),
        settings=SyncV2Settings(
            supports_attachments=True,
            max_blob_bytes=1024 * 1024,
            max_chunk_bytes=64 * 1024,
            pull_token_signing_secret="postgres-attachment-range-test",  # nosec B106
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="managed_storage",
                server_trusted_enabled=True,
                auth_mode="multi_user",
            ),
        ),
        clock=lambda: "2026-08-11T20:30:00+00:00",
    )
    app = FastAPI()
    app.include_router(notes_endpoint.router, prefix="/api/v1/notes")

    async def _db_override() -> CharactersRAGDB:
        return note_db

    async def _user_override() -> User:
        return User(id=owner, username=owner, is_admin=True)

    app.dependency_overrides[notes_endpoint.get_chacha_db_for_user] = _db_override
    app.dependency_overrides[notes_endpoint.get_request_user] = _user_override
    app.dependency_overrides[notes_endpoint.get_rate_limiter_dep] = lambda: _NoopRateLimiter()
    monkeypatch.setattr(
        notes_endpoint,
        "get_active_server_origin_sync_service_for_user",
        lambda user_id: service,
    )

    try:
        session = service.create_blob_upload_session(
            user_id=owner,
            dataset_id=dataset_id,
            device_id=None,
            domain="attachment.ref",
            entity_id=attachment_id,
            attachment_id=attachment_id,
            content_type="application/pdf",
            size_bytes=len(payload),
            payload_hash=_sha256(payload),
            chunk_size=len(payload),
            chunk_count=1,
            idempotency_key="postgres-range-upload",
            metadata={
                "notes_attachment_intent": {
                    "intent": "create",
                    "note_id": note_id,
                    "attachment_id": attachment_id,
                    "file_name": "Range.pdf",
                }
            },
        )
        service.upload_blob_chunk(
            user_id=owner,
            dataset_id=dataset_id,
            upload_id=session.upload_id,
            chunk_index=0,
            offset_bytes=0,
            chunk_payload=payload,
            chunk_hash=_sha256(payload),
        )
        service.complete_blob_upload(
            user_id=owner,
            dataset_id=dataset_id,
            upload_id=session.upload_id,
        )
        path = f"/api/v1/notes/{note_id}/attachments/by-id/{attachment_id}/content"
        with TestClient(app) as client:
            created = client.post(
                f"/api/v1/notes/{note_id}/attachments/from-upload",
                params={"dataset_id": dataset_id},
                headers={"Idempotency-Key": "postgres-range-attach"},
                json={"upload_id": session.upload_id},
            )
            assert created.status_code == 201, created.text
            bounded = client.get(
                path,
                params={"dataset_id": dataset_id},
                headers={"Range": "bytes=2-5"},
            )
            suffix = client.get(
                path,
                params={"dataset_id": dataset_id},
                headers={"Range": "bytes=-3"},
            )
            open_ended = client.get(
                path,
                params={"dataset_id": dataset_id},
                headers={"Range": "bytes=7-"},
            )
            unsatisfied = client.get(
                path,
                params={"dataset_id": dataset_id},
                headers={"Range": "bytes=99-"},
            )

        assert bounded.status_code == 206 and bounded.content == b"2345"
        assert bounded.headers["content-range"] == "bytes 2-5/10"
        assert bounded.headers["content-length"] == "4"
        assert bounded.headers["accept-ranges"] == "bytes"
        assert suffix.status_code == 206 and suffix.content == b"789"
        assert suffix.headers["content-range"] == "bytes 7-9/10"
        assert open_ended.status_code == 206 and open_ended.content == b"789"
        assert open_ended.headers["content-range"] == "bytes 7-9/10"
        assert unsatisfied.status_code == 416
        assert unsatisfied.headers["content-range"] == "bytes */10"
    finally:
        note_db.close_all_connections()
        note_backend.get_pool().close_all()
        sync_backend.get_pool().close_all()

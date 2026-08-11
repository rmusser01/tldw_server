"""Live PostgreSQL tenancy proof for the Notes attachment registry."""

from __future__ import annotations

from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


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

        with db_a.transaction() as conn:
            policy = conn.execute(
                "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                "WHERE oid = 'note_attachments'::regclass"
            ).fetchone()
            assert policy["relrowsecurity"] and policy["relforcerowsecurity"]
            hidden_update = conn.execute(
                "UPDATE note_attachments SET file_name = ? "
                "WHERE client_id = ? AND dataset_id = ? AND attachment_id = ?",
                ("overwrite.pdf", owner_b, dataset_id, attachment_b),
            )
            assert hidden_update.rowcount == 0

        with pytest.raises(DatabaseError):
            with db_a.transaction() as conn:
                conn.execute(
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
                )

        with db_a.transaction() as conn:
            conn.execute("SET LOCAL enable_seqscan = off")
            plan_rows = conn.execute(
                "EXPLAIN SELECT attachment_id FROM note_attachments "
                "WHERE client_id = ? AND dataset_id = ? AND note_id = ? "
                "AND deleted = FALSE AND attachment_id > ? ORDER BY attachment_id LIMIT ?",
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
            all_state_plan = " ".join(
                str(next(iter(dict(row).values()))) for row in all_state_plan_rows
            )
        assert "idx_note_attachments_owner_dataset_note_all_page" in all_state_plan

        with db_a.transaction() as conn:
            conn.execute("SET LOCAL enable_seqscan = off")
            name_plan_rows = conn.execute(
                "EXPLAIN SELECT attachment_id FROM note_attachments "
                "WHERE client_id = ? AND dataset_id = ? AND note_id = ? "
                "AND normalized_file_name = ? AND deleted = FALSE",
                (owner_a, dataset_id, note_a, "same-name.pdf"),
            ).fetchall()
            name_plan = " ".join(
                str(next(iter(dict(row).values()))) for row in name_plan_rows
            )
        assert "uq_note_attachments_live_name" in name_plan
    finally:
        db_a.close_all_connections()
        db_b.close_all_connections()
        backend_a.get_pool().close_all()
        backend_b.get_pool().close_all()

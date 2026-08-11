"""Owner- and dataset-scoped behavior for the canonical attachment store."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)

pytestmark = pytest.mark.unit

OWNER = "attachment-store-owner"
OTHER_OWNER = "attachment-store-other-owner"
DATASET = "dataset-default-notes"
OTHER_DATASET = "dataset-other-notes"
NOTE_ID = "11111111-1111-4111-8111-111111111111"
OTHER_NOTE_ID = "22222222-2222-4222-8222-222222222222"
ATTACHMENT_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
ATTACHMENT_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
ATTACHMENT_C = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
CREATED_AT = "2026-08-11T12:00:00+00:00"
BLOB_A = "sha256:" + "1" * 64
OBJECT_A = "sha256:" + "2" * 64


@pytest.fixture
def attachment_db(tmp_path: Path) -> CharactersRAGDB:
    db = CharactersRAGDB(str(tmp_path / "attachment-store.sqlite"), client_id=OWNER)
    db.add_note("Parent", "Body", note_id=NOTE_ID)
    db.add_note("Other parent", "Body", note_id=OTHER_NOTE_ID)
    yield db
    db.close_all_connections()


def _store(db: CharactersRAGDB) -> Any:
    assert hasattr(db, "note_attachment_store")
    return db.note_attachment_store


def _create(
    db: CharactersRAGDB,
    *,
    attachment_id: str = ATTACHMENT_A,
    dataset_id: str = DATASET,
    note_id: str = NOTE_ID,
    file_name: str = "Report.pdf",
    original_file_name: str | None = None,
    object_hash: str = OBJECT_A,
    source_kind: str = "sync",
):
    return _store(db).create(
        dataset_id=dataset_id,
        attachment_id=attachment_id,
        note_id=note_id,
        file_name=file_name,
        original_file_name=original_file_name or file_name,
        content_type="application/pdf",
        size_bytes=42,
        blob_hash=BLOB_A,
        object_hash=object_hash,
        created_at=CREATED_AT,
        last_modified=CREATED_AT,
        created_by="device-a",
        source_kind=source_kind,
    )


def test_create_get_and_list_are_owner_dataset_and_note_scoped(
    attachment_db: CharactersRAGDB,
) -> None:
    created = _create(attachment_db)

    assert created.client_id == OWNER
    assert created.dataset_id == DATASET
    assert created.attachment_id == ATTACHMENT_A
    assert created.note_id == NOTE_ID
    assert created.file_name == "Report.pdf"
    assert created.normalized_file_name == "report.pdf"
    assert created.version == 1
    assert created.deleted is False
    assert _store(attachment_db).get(DATASET, ATTACHMENT_A) == created
    assert _store(attachment_db).get(OTHER_DATASET, ATTACHMENT_A) is None
    assert _store(attachment_db).list_page(
        DATASET,
        NOTE_ID,
        after_attachment_id=None,
        limit=50,
    ) == (created,)
    assert _store(attachment_db).list_page(
        DATASET,
        OTHER_NOTE_ID,
        after_attachment_id=None,
        limit=50,
    ) == ()


def test_create_rejects_cross_owner_or_missing_note_endpoint(
    attachment_db: CharactersRAGDB,
) -> None:
    with attachment_db.transaction() as conn:
        conn.execute(
            "UPDATE notes SET client_id = ? WHERE id = ?",
            (OTHER_OWNER, OTHER_NOTE_ID),
        )

    with pytest.raises(InputError, match="owned note|endpoint"):
        _create(attachment_db, note_id=OTHER_NOTE_ID)
    with pytest.raises(InputError, match="owned note|endpoint"):
        _create(
            attachment_db,
            note_id="33333333-3333-4333-8333-333333333333",
        )


def test_existing_row_is_hidden_and_mutation_conflicts_for_another_owner(
    attachment_db: CharactersRAGDB,
) -> None:
    created = _create(attachment_db)
    other_owner = CharactersRAGDB(attachment_db.db_path_str, client_id=OTHER_OWNER)
    try:
        other_store = _store(other_owner)
        assert other_store.get(DATASET, ATTACHMENT_A) is None
        assert other_store.list_page(
            DATASET,
            NOTE_ID,
            after_attachment_id=None,
            limit=50,
        ) == ()
        with pytest.raises(ConflictError) as exc_info:
            other_store.tombstone(
                dataset_id=DATASET,
                attachment_id=ATTACHMENT_A,
                expected_version=created.version,
                expected_object_hash=created.object_hash,
                object_hash="sha256:" + "f" * 64,
                last_modified="2026-08-11T12:01:00+00:00",
                deleted_at="2026-08-11T12:01:00+00:00",
            )
        assert OWNER not in str(exc_info.value)
        assert NOTE_ID not in str(exc_info.value)
    finally:
        other_owner.close_all_connections()


def test_live_name_uniqueness_uses_unicode_normalization_and_casefold(
    attachment_db: CharactersRAGDB,
) -> None:
    first = _create(attachment_db, file_name="Résumé.PDF")
    assert first.normalized_file_name == "résumé.pdf"

    with pytest.raises(ConflictError, match="name|filename"):
        _create(
            attachment_db,
            attachment_id=ATTACHMENT_B,
            file_name="RÉSUMÉ.pdf",
        )

    other_dataset = _create(
        attachment_db,
        attachment_id=ATTACHMENT_C,
        dataset_id=OTHER_DATASET,
        file_name="RÉSUMÉ.pdf",
    )
    assert other_dataset.dataset_id == OTHER_DATASET


def test_compare_and_set_updates_mutable_fields_and_rejects_stale_base(
    attachment_db: CharactersRAGDB,
) -> None:
    created = _create(attachment_db)
    object_b = "sha256:" + "3" * 64
    updated = _store(attachment_db).compare_and_set(
        dataset_id=DATASET,
        attachment_id=ATTACHMENT_A,
        expected_version=created.version,
        expected_object_hash=created.object_hash,
        file_name="Renamed.pdf",
        content_type="application/pdf",
        size_bytes=created.size_bytes,
        blob_hash=created.blob_hash,
        object_hash=object_b,
        last_modified="2026-08-11T12:01:00+00:00",
    )

    assert updated.version == 2
    assert updated.file_name == "Renamed.pdf"
    assert updated.normalized_file_name == "renamed.pdf"
    assert updated.original_file_name == created.original_file_name
    assert updated.note_id == created.note_id
    assert updated.created_at == created.created_at
    assert updated.created_by == created.created_by
    with pytest.raises(ConflictError, match="version|base|conflict"):
        _store(attachment_db).compare_and_set(
            dataset_id=DATASET,
            attachment_id=ATTACHMENT_A,
            expected_version=1,
            expected_object_hash=OBJECT_A,
            file_name="Stale.pdf",
            content_type="application/pdf",
            size_bytes=42,
            blob_hash=BLOB_A,
            object_hash="sha256:" + "4" * 64,
            last_modified="2026-08-11T12:02:00+00:00",
        )
    with pytest.raises(ConflictError, match="hash|base|conflict"):
        _store(attachment_db).compare_and_set(
            dataset_id=DATASET,
            attachment_id=ATTACHMENT_A,
            expected_version=2,
            expected_object_hash=OBJECT_A,
            file_name="Stale.pdf",
            content_type="application/pdf",
            size_bytes=42,
            blob_hash=BLOB_A,
            object_hash="sha256:" + "4" * 64,
            last_modified="2026-08-11T12:02:00+00:00",
        )


def test_tombstone_releases_live_name_and_restore_rechecks_collision(
    attachment_db: CharactersRAGDB,
) -> None:
    first = _create(attachment_db)
    tombstoned = _store(attachment_db).tombstone(
        dataset_id=DATASET,
        attachment_id=ATTACHMENT_A,
        expected_version=first.version,
        expected_object_hash=first.object_hash,
        object_hash="sha256:" + "5" * 64,
        last_modified="2026-08-11T12:03:00+00:00",
        deleted_at="2026-08-11T12:03:00+00:00",
        delete_reason="user_deleted",
    )
    assert tombstoned.version == 2
    assert tombstoned.deleted is True
    assert tombstoned.deleted_at == "2026-08-11T12:03:00+00:00"
    assert tombstoned.delete_reason == "user_deleted"

    replacement = _create(
        attachment_db,
        attachment_id=ATTACHMENT_B,
        file_name="REPORT.PDF",
        object_hash="sha256:" + "6" * 64,
    )
    with pytest.raises(ConflictError, match="name|filename"):
        _store(attachment_db).restore(
            dataset_id=DATASET,
            attachment_id=ATTACHMENT_A,
            expected_version=tombstoned.version,
            expected_object_hash=tombstoned.object_hash,
            object_hash="sha256:" + "7" * 64,
            last_modified="2026-08-11T12:04:00+00:00",
        )

    replacement_tombstone = _store(attachment_db).tombstone(
        dataset_id=DATASET,
        attachment_id=ATTACHMENT_B,
        expected_version=replacement.version,
        expected_object_hash=replacement.object_hash,
        object_hash="sha256:" + "8" * 64,
        last_modified="2026-08-11T12:05:00+00:00",
        deleted_at="2026-08-11T12:05:00+00:00",
    )
    assert replacement_tombstone.deleted is True
    restored = _store(attachment_db).restore(
        dataset_id=DATASET,
        attachment_id=ATTACHMENT_A,
        expected_version=tombstoned.version,
        expected_object_hash=tombstoned.object_hash,
        object_hash="sha256:" + "9" * 64,
        last_modified="2026-08-11T12:06:00+00:00",
    )
    assert restored.version == 3
    assert restored.deleted is False
    assert restored.deleted_at is None
    assert restored.delete_reason is None


def test_list_page_state_filter_and_attachment_id_keyset(
    attachment_db: CharactersRAGDB,
) -> None:
    a = _create(attachment_db, attachment_id=ATTACHMENT_A, file_name="a.pdf")
    b = _create(
        attachment_db,
        attachment_id=ATTACHMENT_B,
        file_name="b.pdf",
        object_hash="sha256:" + "a" * 64,
    )
    c = _create(
        attachment_db,
        attachment_id=ATTACHMENT_C,
        file_name="c.pdf",
        object_hash="sha256:" + "b" * 64,
    )
    b = _store(attachment_db).tombstone(
        dataset_id=DATASET,
        attachment_id=b.attachment_id,
        expected_version=b.version,
        expected_object_hash=b.object_hash,
        object_hash="sha256:" + "c" * 64,
        last_modified="2026-08-11T12:07:00+00:00",
        deleted_at="2026-08-11T12:07:00+00:00",
    )

    assert _store(attachment_db).list_page(
        DATASET,
        NOTE_ID,
        after_attachment_id=None,
        limit=1,
    ) == (a,)
    assert _store(attachment_db).list_page(
        DATASET,
        NOTE_ID,
        after_attachment_id=a.attachment_id,
        limit=50,
    ) == (c,)
    assert _store(attachment_db).list_page(
        DATASET,
        NOTE_ID,
        after_attachment_id=None,
        limit=50,
        state="tombstoned",
    ) == (b,)
    assert _store(attachment_db).list_page(
        DATASET,
        NOTE_ID,
        after_attachment_id=None,
        limit=50,
        state="all",
    ) == (a, b, c)


@pytest.mark.parametrize("limit", [0, 201])
def test_list_page_enforces_canonical_bound(
    attachment_db: CharactersRAGDB,
    limit: int,
) -> None:
    with pytest.raises(InputError, match="limit"):
        _store(attachment_db).list_page(
            DATASET,
            NOTE_ID,
            after_attachment_id=None,
            limit=limit,
        )


def test_detail_and_list_use_one_bounded_indexed_query_each(
    attachment_db: CharactersRAGDB,
) -> None:
    _create(attachment_db)
    traced: list[str] = []
    connection = attachment_db.get_connection()
    assert isinstance(connection, sqlite3.Connection)
    with attachment_db.transaction() as conn:
        connection.set_trace_callback(traced.append)
        try:
            assert _store(attachment_db).get(DATASET, ATTACHMENT_A, conn=conn) is not None
            detail_selects = [sql for sql in traced if sql.lstrip().upper().startswith("SELECT")]
            traced.clear()
            assert len(
                _store(attachment_db).list_page(
                    DATASET,
                    NOTE_ID,
                    after_attachment_id=None,
                    limit=50,
                    conn=conn,
                )
            ) == 1
            list_selects = [sql for sql in traced if sql.lstrip().upper().startswith("SELECT")]
        finally:
            connection.set_trace_callback(None)

    assert len(detail_selects) == 1
    assert len(list_selects) == 1
    with attachment_db.transaction() as conn:
        plan = " ".join(
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT attachment_id FROM note_attachments "
                "WHERE client_id = ? AND dataset_id = ? AND note_id = ? "
                "AND deleted = 0 AND attachment_id > ? ORDER BY attachment_id LIMIT ?",
                (OWNER, DATASET, NOTE_ID, "", 50),
            ).fetchall()
        )
    assert "idx_note_attachments_owner_dataset_note_page" in plan


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_id", ""),
        ("attachment_id", "not-a-uuid"),
        ("note_id", "not-a-uuid"),
        ("file_name", "../report.pdf"),
        ("file_name", "report\\escape.pdf"),
        ("file_name", "report／escape.pdf"),
        ("file_name", "x" * 181),
        ("original_file_name", "x" * 256),
        ("content_type", ""),
        ("size_bytes", 0),
        ("blob_hash", "sha256:" + "A" * 64),
        ("object_hash", "sha256:" + "B" * 64),
        ("source_kind", "legacy"),
    ],
)
def test_create_validates_registry_boundary(
    attachment_db: CharactersRAGDB,
    field: str,
    value: object,
) -> None:
    kwargs = {
        "dataset_id": DATASET,
        "attachment_id": ATTACHMENT_A,
        "note_id": NOTE_ID,
        "file_name": "Report.pdf",
        "original_file_name": "Report.pdf",
        "content_type": "application/pdf",
        "size_bytes": 42,
        "blob_hash": BLOB_A,
        "object_hash": OBJECT_A,
        "created_at": CREATED_AT,
        "last_modified": CREATED_AT,
        "created_by": "device-a",
        "source_kind": "sync",
    }
    kwargs[field] = value
    with pytest.raises(InputError):
        _store(attachment_db).create(**kwargs)

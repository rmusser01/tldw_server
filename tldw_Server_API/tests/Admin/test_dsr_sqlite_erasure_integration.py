"""Real SQLite erasure boundaries, transaction recovery, and safe failure logs."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.services import admin_data_subject_requests_service as service

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

NOTE_ID = "11111111-1111-4111-8111-111111111111"
TARGET_ID = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
def user_databases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build the actual current notes schema for two isolated owners."""
    monkeypatch.setenv("TLDW_DB_ALLOWED_BASE_DIRS", str(tmp_path))
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    databases = {owner: CharactersRAGDB(tmp_path / f"notes-{owner}.db", client_id=str(owner)) for owner in (7, 8)}
    monkeypatch.setattr(
        service.DatabasePaths,
        "get_chacha_db_path",
        lambda owner: databases[owner].db_path,
    )
    for db in databases.values():
        db.add_note("Target", "target body", note_id=TARGET_ID)
        db.add_note("Private note", f"[[id:{TARGET_ID}]]", note_id=NOTE_ID)
    yield databases
    for db in databases.values():
        db.close_all_connections()


def _rows(db: CharactersRAGDB, table: str) -> list:
    """Read only fixed test-owned tables through the repository connection."""
    queries = {
        "notes": "SELECT * FROM notes",
        "note_graph_note_state": "SELECT * FROM note_graph_note_state",
        "note_wikilink_edges": "SELECT * FROM note_wikilink_edges",
    }
    return db.execute_query(queries[table]).fetchall()


async def test_notes_erasure_cascades_projection_state_and_preserves_other_user(
    user_databases,
) -> None:
    own, other = user_databases[7], user_databases[8]
    other_before = _rows(other, "notes")
    assert _rows(own, "note_graph_note_state")

    await service._erase_notes(7)

    assert _rows(own, "notes") == []
    assert _rows(own, "note_graph_note_state") == []
    assert _rows(other, "notes") == other_before
    assert await service._erase_notes(7) == 0


async def test_notes_erasure_preserves_attachment_parent_on_restricted_delete(
    user_databases,
) -> None:
    db = user_databases[7]
    db.note_attachment_store.create(
        dataset_id="default",
        attachment_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        note_id=NOTE_ID,
        file_name="attachment.txt",
        original_file_name="attachment.txt",
        content_type="text/plain",
        size_bytes=42,
        blob_hash="sha256:" + "1" * 64,
        object_hash="sha256:" + "2" * 64,
        created_at="2026-09-10T12:00:00+00:00",
        last_modified="2026-09-10T12:00:00+00:00",
        created_by="7",
        source_kind="sync",
    )
    before = _rows(db, "notes")
    edges_before = _rows(db, "note_wikilink_edges")

    with pytest.raises(sqlite3.IntegrityError):
        await service._erase_notes(7)

    assert _rows(db, "notes") == before
    assert _rows(db, "note_wikilink_edges") == edges_before


async def test_notes_erasure_rolls_back_and_retry_succeeds(user_databases) -> None:
    db = user_databases[7]
    before = _rows(db, "notes")
    edges_before = _rows(db, "note_wikilink_edges")
    with db.transaction() as conn:
        conn.execute(
            "CREATE TRIGGER reject_erasure BEFORE DELETE ON notes "
            "BEGIN SELECT RAISE(ABORT, 'temporary deletion failure'); END"
        )

    with pytest.raises(sqlite3.IntegrityError):
        await service._erase_notes(7)

    assert _rows(db, "notes") == before
    assert _rows(db, "note_wikilink_edges") == edges_before
    with db.transaction() as conn:
        conn.execute("DROP TRIGGER reject_erasure")
    await service._erase_notes(7)
    assert _rows(db, "notes") == []


async def test_erasure_failure_logs_exclude_database_exception_content(
    user_databases,
) -> None:
    db = user_databases[7]
    private_marker = "private-subject-content"
    with db.transaction() as conn:
        conn.execute(
            "CREATE TRIGGER reject_erasure BEFORE DELETE ON notes "
            "BEGIN SELECT RAISE(ABORT, 'private-subject-content'); END"
        )
    messages: list[str] = []
    sink = service.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        result = await service.execute_dsr_erasure(
            request_id=1,
            user_id=7,
            selected_categories=["notes"],
            dsr_repo=AsyncMock(),
        )
    finally:
        service.logger.remove(sink)

    assert result["status"] == "failed"
    assert "DSR erasure failed" in "\n".join(messages)
    assert private_marker not in "\n".join(messages)
    assert private_marker not in repr(result)

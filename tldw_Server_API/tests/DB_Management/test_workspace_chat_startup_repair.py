"""Current-schema repair cannot change accepted Workspace startup authority."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.workspace_chat_startup_store import WorkspaceStartupError
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.DB_Backups import create_backup, restore_single_db_backup
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import db_factory as db_factory
from tldw_Server_API.tests.DB_Management.test_workspace_assistant_creation_atomic import creation_db as creation_db
from tldw_Server_API.tests.DB_Management.test_workspace_chat_startup_acceptance import _start
from tldw_Server_API.tests.DB_Management.test_workspace_chat_startup_lifecycle import _receipt

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("selection", ["inherit", "none"])
@pytest.mark.parametrize("missing", ["kind", "id", "both"])
def test_current_schema_character_repair_preserves_accepted_startup(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB],
    selection: str, missing: str,
) -> None:
    """Reopening repairs historical Character omissions without touching strict chats."""
    payload = {
        "scope_type": "workspace", "workspace_id": "ws", "workspace_assistant_selection": selection,
    }
    if selection == "inherit":
        payload["workspace_assistant_default_version"] = 2
    first = _start(creation_db, payload=payload)
    receipt = _receipt(creation_db)
    character = creation_db.add_character_card({"name": "Historical Character"})
    legacy = creation_db.add_conversation({"title": "Legacy", "character_id": character})
    with creation_db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET assistant_kind = ?, assistant_id = ? WHERE id = ?",
            (None if missing in ("kind", "both") else "character",
             None if missing in ("id", "both") else str(character), legacy),
        )
    creation_db.close_all_connections()
    reopened = db_factory()
    with reopened.transaction():
        repaired = reopened.get_conversation_by_id(legacy)
        accepted = reopened.get_conversation_by_id(first.conversation["id"])
    assert (repaired["assistant_kind"], repaired["assistant_id"]) == ("character", str(character))
    assert _receipt(reopened) == receipt
    assert accepted["assistant_startup_json"] == first.conversation["assistant_startup_json"]
    assert accepted["assistant_id"] == first.conversation["assistant_id"]
    assert _start(reopened, payload=payload).replayed


def test_cached_unhooked_writer_requires_offline_drain(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB],
) -> None:
    """An already-open driver is not fenced; legacy-style SQL can bypass writer hooks."""
    cached = db_factory()
    with cached.transaction() as conn:
        conn.execute("SELECT 1").fetchone()
    first = _start(creation_db)
    cid = first.conversation["id"]
    # Deliberately bypass modern writer hooks to demonstrate the unsupported
    # cached-old-writer hazard, not to claim old-binary compatibility.
    with cached.transaction() as conn:
        conn.execute("UPDATE conversations SET assistant_id = ? WHERE id = ?", ("persona-b", cid))
    assert _receipt(creation_db)["invalidated_at"] is None
    with pytest.raises(WorkspaceStartupError) as changed:
        _start(creation_db)
    assert changed.value.code == "workspace_chat_startup_changed"
    with cached.transaction() as conn:
        conn.execute("UPDATE conversations SET assistant_id = ? WHERE id = ?", ("persona-a", cid))
    assert _start(creation_db).replayed


def test_sqlite_offline_backup_restore_retains_private_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing physical backup/restore retains live keys and non-recyclable tombstones."""
    from tldw_Server_API.app.core import feature_flags

    monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: True)
    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", str(tmp_path))
    monkeypatch.setenv("TLDW_DB_ALLOWED_BASE_DIRS", str(tmp_path))
    database_path, backup_dir = tmp_path / "offline.db", tmp_path / "backups"
    db = CharactersRAGDB(database_path, client_id="user-1")
    payload = {"scope_type": "workspace", "workspace_id": "ws", "workspace_assistant_selection": "none"}
    try:
        db.upsert_workspace("ws", "Workspace")
        live = _start(db, payload=payload, receipt_limit=2)
        deleted = _start(db, payload=payload, idempotency_key="deleted-key", receipt_limit=2)
        db.hard_delete_conversation(deleted.conversation["id"])
        with db.transaction() as conn:
            before = [dict(row) for row in conn.execute(
                "SELECT * FROM workspace_chat_startup_receipts ORDER BY key_digest"
            ).fetchall()]
        db.close_all_connections()
        assert create_backup(str(database_path), str(backup_dir), "offline").startswith("Backup created")
        backup = next(backup_dir.glob("offline_backup_*.db"))
        restored_path = tmp_path / "restored.db"
        assert not restored_path.exists()
        assert restore_single_db_backup(
            str(restored_path), str(backup_dir), "offline", backup.name,
        ).startswith("Database restored")
        assert restored_path.exists()
        reopened = CharactersRAGDB(restored_path, client_id="user-1")
        try:
            with reopened.transaction() as conn:
                assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                after = [dict(row) for row in conn.execute(
                    "SELECT * FROM workspace_chat_startup_receipts ORDER BY key_digest"
                ).fetchall()]
            assert after == before
            assert _start(reopened, payload=payload, receipt_limit=2).conversation["id"] == live.conversation["id"]
            with pytest.raises(WorkspaceStartupError) as tombstone:
                _start(reopened, payload=payload, idempotency_key="deleted-key", receipt_limit=2)
            assert tombstone.value.code == "workspace_chat_deleted"
            with pytest.raises(WorkspaceStartupError) as capacity:
                _start(reopened, payload=payload, idempotency_key="new-key", receipt_limit=2)
            assert capacity.value.code == "workspace_chat_receipt_capacity_exceeded"
        finally:
            reopened.close_all_connections()
    finally:
        db.close_all_connections()

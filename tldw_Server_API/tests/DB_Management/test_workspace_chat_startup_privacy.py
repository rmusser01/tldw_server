"""Chatbook and erasure boundaries do not export or forge startup authority."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    ChatbookContent,
    ChatbookManifest,
    ChatbookVersion,
    ConflictResolution,
    ImportJob,
    ImportStatus,
)
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.services import admin_data_subject_requests_service as dsr_service
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import db_factory as db_factory
from tldw_Server_API.tests.DB_Management.test_workspace_assistant_creation_atomic import creation_db as creation_db
from tldw_Server_API.tests.DB_Management.test_workspace_chat_startup_acceptance import _start
from tldw_Server_API.tests.DB_Management.test_workspace_chat_startup_lifecycle import _receipt

pytestmark = pytest.mark.integration


def _export(
    db: CharactersRAGDB, work_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[ChatbookService, ChatbookManifest, dict[str, Any]]:
    """Export an accepted chat through the real Chatbook conversation collector."""
    monkeypatch.setenv("USER_DB_BASE_DIR", str(work_dir / "users"))
    result = _start(db)
    # Collector/import privacy does not depend on background job-table initialization.
    service = ChatbookService.__new__(ChatbookService)
    service.user_id = "user-1"
    service.db = db
    manifest = ChatbookManifest(version=ChatbookVersion.V1, name="Export", description="")
    content = ChatbookContent()
    service._collect_conversations([result.conversation["id"]], work_dir, manifest, content)
    assert list(content.conversations) == [result.conversation["id"]]
    return service, manifest, content.conversations[result.conversation["id"]]


def test_chatbook_export_never_serializes_startup_receipt(
    creation_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conversation payload may carry display data, never the owner retry secret."""
    _, manifest, payload = _export(creation_db, tmp_path, monkeypatch)
    cid = payload["id"]
    exported = (tmp_path / manifest.content_items[0].file_path).read_text(encoding="utf-8")
    receipt = _receipt(creation_db)
    assert set(payload) == {"id", "name", "created_at", "character_id", "attachments_path", "messages"}
    assert json.loads(exported) == payload
    assert receipt["key_digest"] not in exported
    assert receipt["binding_digest"] not in exported
    assert receipt["request_fingerprint"] not in exported
    assert cid == receipt["conversation_id"]


def test_chatbook_import_ignores_forged_startup_receipt_fields(
    creation_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An archive cannot create a replayable receipt or transplant its binding."""
    service, manifest, payload = _export(creation_db, tmp_path, monkeypatch)
    original = _receipt(creation_db)
    character_id = creation_db.add_character_card({"name": "Imported source"})
    forged = {
        **payload,
        "name": "Forged import",
        "character_id": character_id,
        "scope_type": "workspace",
        "workspace_id": "ws",
        "assistant_kind": "persona",
        "assistant_id": "persona-a",
        "assistant_startup_json": creation_db.get_conversation_by_id(payload["id"])["assistant_startup_json"],
        "idempotency_key": "accepted-1",
        "workspace_chat_startup_receipts": [original],
    }
    (tmp_path / manifest.content_items[0].file_path).write_text(json.dumps(forged), encoding="utf-8")
    status = ImportJob(
        job_id="import-1", user_id="user-1", status=ImportStatus.IN_PROGRESS, chatbook_path=str(tmp_path),
    )
    service._import_conversations(
        tmp_path, manifest, [payload["id"]], ConflictResolution.RENAME, False, status,
    )
    assert status.successful_items == 1, status.warnings
    with creation_db.transaction() as conn:
        receipts = conn.execute("SELECT * FROM workspace_chat_startup_receipts").fetchall()
        imported = conn.execute("SELECT * FROM conversations WHERE title = ?", ("Forged import",)).fetchone()
    assert len(receipts) == 1
    assert receipts[0]["conversation_id"] == original["conversation_id"]
    assert imported is not None
    assert imported["id"] != original["conversation_id"]
    assert imported["assistant_startup_json"] is None
    assert imported["workspace_id"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_device", [False, True])
async def test_chat_erasure_keeps_unbound_receipt_tombstone(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch, sync_device: bool,
) -> None:
    """Erasure follows immutable receipt ownership even after Sync changes device."""
    if creation_db.backend_type.value != "sqlite":
        pytest.skip("The data-subject eraser operates on a SQLite per-user file")
    result = _start(creation_db)
    cid = result.conversation["id"]
    if sync_device:
        assert creation_db.upsert_conversation_from_sync(
            conversation_id=cid, title="Different device", sync_client_id="other-device",
            object_revision=2, object_hash="h", assistant_kind="persona",
            assistant_id="persona-a", persona_memory_mode="read_only",
            scope_type="workspace", workspace_id="ws",
        )
        assert _receipt(creation_db)["invalidated_at"] is None
    creation_db.add_message({"conversation_id": cid, "sender": "user", "content": "Private text"})
    monkeypatch.setattr(dsr_service.DatabasePaths, "get_chacha_db_path", lambda _user_id: creation_db.db_path)
    assert await dsr_service._count_chat_messages("user-1") == 1
    creation_db.close_all_connections()
    assert await dsr_service._erase_chat_messages("user-1") == 2
    with db_factory().transaction() as conn:
        receipt = conn.execute("SELECT * FROM workspace_chat_startup_receipts").fetchone()
        conversation = conn.execute("SELECT id FROM conversations WHERE id = ?", (cid,)).fetchone()
        message = conn.execute("SELECT id FROM messages WHERE conversation_id = ?", (cid,)).fetchone()
    assert conversation is None
    assert message is None
    assert receipt["conversation_id"] is None
    assert receipt["invalidated_at"] is None
    assert receipt["owner_user_id"] == "user-1"


@pytest.mark.asyncio
async def test_chat_erasure_preserves_pre_receipt_database_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older ChaCha files without the receipt table retain their original DSR path."""
    path = tmp_path / "old-chacha.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE conversations (id TEXT PRIMARY KEY, client_id TEXT, deleted INTEGER)")
        conn.execute("CREATE TABLE messages (id TEXT PRIMARY KEY, conversation_id TEXT, deleted INTEGER)")
        conn.execute("INSERT INTO conversations VALUES ('chat-1', 'user-1', 0)")
        conn.execute("INSERT INTO messages VALUES ('message-1', 'chat-1', 0)")
    monkeypatch.setattr(dsr_service.DatabasePaths, "get_chacha_db_path", lambda _user_id: path)
    assert await dsr_service._count_chat_messages("user-1") == 1
    assert await dsr_service._erase_chat_messages("user-1") == 2
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_chat_receipt_probe_preserves_coverage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken store follows the existing bounded DSR preview error contract."""
    path = tmp_path / "broken-chacha.db"
    path.write_bytes(b"not a sqlite database")
    monkeypatch.setattr(dsr_service.DatabasePaths, "get_chacha_db_path", lambda _user_id: path)
    with pytest.raises(dsr_service.DataSubjectRequestCoverageUnavailableError):
        await dsr_service._count_chat_messages("user-1")

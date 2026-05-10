import json
import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Chatbooks.import_adapters.openwebui_db import (
    extract_openwebui_db_user,
    preview_openwebui_db,
)


pytestmark = pytest.mark.unit


def _message_tree(*, content: str = "secret user content", branched: bool = False) -> dict:
    messages = {
        "user-root": {
            "id": "user-root",
            "role": "user",
            "content": content,
            "timestamp": 1700000000,
            "childrenIds": ["assistant-main", "assistant-alt"] if branched else ["assistant-main"],
        },
        "assistant-main": {
            "id": "assistant-main",
            "role": "assistant",
            "content": "main answer",
            "parentId": "user-root",
            "timestamp": 1700000001,
            "model": "gpt-4o",
            "files": [{"id": "file-1", "name": "notes.pdf"}],
        },
    }
    if branched:
        messages["assistant-alt"] = {
            "id": "assistant-alt",
            "role": "assistant",
            "content": "alternative answer",
            "parentId": "user-root",
            "timestamp": 1700000002,
        }
    return {
        "history": {
            "currentId": "assistant-main",
            "messages": messages,
        },
        "models": ["gpt-4o"],
    }


def _write_openwebui_db(
    path: Path,
    *,
    users: list[dict],
    chats: list[dict],
    folders: list[dict] | None = None,
) -> Path:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE user (
                id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                role TEXT,
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE folder (
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                user_id TEXT,
                name TEXT,
                items TEXT,
                meta TEXT,
                is_expanded INTEGER,
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE chat (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT,
                chat TEXT,
                created_at INTEGER,
                updated_at INTEGER,
                share_id TEXT,
                archived INTEGER,
                pinned INTEGER,
                meta TEXT,
                folder_id TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO user (id, name, email, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    user["id"],
                    user.get("name"),
                    user.get("email"),
                    user.get("role", "user"),
                    user.get("created_at", 1700000000),
                    user.get("updated_at", 1700000001),
                )
                for user in users
            ],
        )
        conn.executemany(
            """
            INSERT INTO folder (id, parent_id, user_id, name, items, meta, is_expanded, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    folder["id"],
                    folder.get("parent_id"),
                    folder["user_id"],
                    folder["name"],
                    json.dumps(folder.get("items", [])),
                    json.dumps(folder.get("meta", {})),
                    1,
                    folder.get("created_at", 1700000000),
                    folder.get("updated_at", 1700000001),
                )
                for folder in folders or []
            ],
        )
        conn.executemany(
            """
            INSERT INTO chat (id, user_id, title, chat, created_at, updated_at, share_id, archived, pinned, meta, folder_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    chat["id"],
                    chat["user_id"],
                    chat["title"],
                    json.dumps(chat["chat"]),
                    chat.get("created_at", 1700000100),
                    chat.get("updated_at", 1700000200),
                    chat.get("share_id"),
                    int(bool(chat.get("archived", False))),
                    int(bool(chat.get("pinned", False))),
                    json.dumps(chat.get("meta", {})),
                    chat.get("folder_id"),
                )
                for chat in chats
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return path


def _standard_db(tmp_path: Path) -> Path:
    return _write_openwebui_db(
        tmp_path / "webui.db",
        users=[
            {"id": "user-a", "name": "Alice", "email": "alice@example.test"},
            {"id": "user-b", "name": "Bob", "email": "bob@example.test"},
        ],
        folders=[
            {"id": "folder-root", "user_id": "user-a", "name": "Research", "items": ["chat-other"]},
            {"id": "folder-child", "parent_id": "folder-root", "user_id": "user-a", "name": "Papers"},
        ],
        chats=[
            {
                "id": "chat-a",
                "user_id": "user-a",
                "title": "Alice research",
                "chat": _message_tree(branched=True),
                "folder_id": "folder-child",
                "archived": True,
                "pinned": True,
                "meta": {"project": "Migration"},
            },
            {
                "id": "chat-b",
                "user_id": "user-b",
                "title": "Bob notes",
                "chat": _message_tree(content="bob private content"),
            },
        ],
    )


def test_preview_openwebui_db_rejects_non_sqlite_file(tmp_path):
    db_path = tmp_path / "webui.db"
    db_path.write_bytes(b"not a sqlite database")

    with pytest.raises(ValueError, match="Invalid OpenWebUI SQLite database"):
        preview_openwebui_db(db_path)


def test_preview_openwebui_db_rejects_missing_required_schema(tmp_path):
    db_path = tmp_path / "missing-schema.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE user (id TEXT PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(ValueError, match="missing required OpenWebUI table"):
        preview_openwebui_db(db_path)


def test_preview_openwebui_db_lists_users_counts_and_hides_content(tmp_path):
    db_path = _standard_db(tmp_path)

    preview = preview_openwebui_db(db_path, duplicate_lookup=lambda external_ref: external_ref == "chat-a")
    data = preview.to_dict()

    assert data["user_count"] == 2
    alice = next(user for user in data["users"] if user["source_user_id"] == "user-a")
    bob = next(user for user in data["users"] if user["source_user_id"] == "user-b")
    assert alice["display_label"] == "Alice"
    assert alice["chat_count"] == 1
    assert alice["folder_count"] == 2
    assert alice["message_count"] == 3
    assert alice["branched_chat_count"] == 1
    assert alice["duplicate_chat_count"] == 1
    assert alice["archived_chat_count"] == 1
    assert alice["pinned_chat_count"] == 1
    assert alice["attachment_reference_count"] == 1
    assert bob["chat_count"] == 1
    assert "secret user content" not in json.dumps(data)
    assert "bob private content" not in json.dumps(data)


def test_extract_openwebui_db_user_imports_only_selected_user_and_folder_plan(tmp_path):
    db_path = _standard_db(tmp_path)

    result = extract_openwebui_db_user(db_path, selected_user_id="user-a")

    assert result.selected_user_id == "user-a"
    assert result.selected_user_label == "Alice"
    assert [chat.external_ref for chat in result.chats] == ["chat-a"]
    chat = result.chats[0]
    assert chat.title == "Alice research"
    assert chat.is_branched is True
    assert [message.source_id for message in chat.messages] == [
        "user-root",
        "assistant-main",
        "assistant-alt",
    ]
    assert chat.source_metadata["source_kind"] == "openwebui_db"
    assert chat.source_metadata["source_user_id"] == "user-a"
    assert chat.source_metadata["folder_id"] == "folder-child"
    assert chat.source_metadata["meta"] == {"project": "Migration"}
    folder_plan = result.folder_plans_by_external_ref["chat-a"]
    assert folder_plan.source_folder_id == "folder-child"
    assert folder_plan.source_path == ["Research", "Papers"]
    assert any("folder.items" in warning for warning in result.warnings)


def test_extract_openwebui_db_user_routes_folder_cycles_to_unfiled(tmp_path):
    db_path = _write_openwebui_db(
        tmp_path / "cycle.db",
        users=[{"id": "user-a", "name": "Alice"}],
        folders=[
            {"id": "folder-a", "parent_id": "folder-b", "user_id": "user-a", "name": "A"},
            {"id": "folder-b", "parent_id": "folder-a", "user_id": "user-a", "name": "B"},
        ],
        chats=[
            {
                "id": "chat-cycle",
                "user_id": "user-a",
                "title": "Cycle",
                "chat": _message_tree(),
                "folder_id": "folder-a",
            }
        ],
    )

    result = extract_openwebui_db_user(db_path, selected_user_id="user-a")

    folder_plan = result.folder_plans_by_external_ref["chat-cycle"]
    assert folder_plan.source_path == ["Unfiled"]
    assert any("cycle" in warning.lower() for warning in result.warnings)

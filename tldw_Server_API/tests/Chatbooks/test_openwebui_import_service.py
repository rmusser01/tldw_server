import json
import sqlite3

import pytest

from tldw_Server_API.app.core.Chatbooks.exceptions import DatabaseError
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.Chatbooks.chatbook_models import ConflictResolution
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit


@pytest.fixture()
def openwebui_service(tmp_path):
    db = CharactersRAGDB(
        db_path=str(tmp_path / "openwebui-import.sqlite"),
        client_id="openwebui-import-user",
    )
    db.add_character_card({"name": "OpenWebUI Import Assistant"})
    service = ChatbookService(user_id="1", db=db)
    service.temp_dir = tmp_path / "chatbooks-temp"
    service.import_dir = tmp_path / "chatbooks-imports"
    service.temp_dir.mkdir(parents=True, exist_ok=True)
    service.import_dir.mkdir(parents=True, exist_ok=True)
    return service


def _write_export(service: ChatbookService, payload: list[dict]) -> str:
    path = service.temp_dir / "openwebui.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _branched_export() -> list[dict]:
    return [
        {
            "id": "chat-branched",
            "chat": {
                "title": "Branch preserving import",
                "models": ["gpt-4o"],
                "history": {
                    "currentId": "assistant-main",
                    "messages": {
                        "user-root": {
                            "id": "user-root",
                            "role": "user",
                            "content": "Compare two retrieval strategies.",
                            "timestamp": 1700000000,
                            "childrenIds": ["assistant-main", "assistant-alt"],
                        },
                        "assistant-main": {
                            "id": "assistant-main",
                            "role": "assistant",
                            "content": "Use hybrid retrieval.",
                            "parentId": "user-root",
                            "timestamp": 1700000001,
                            "model": "gpt-4o",
                        },
                        "assistant-alt": {
                            "id": "assistant-alt",
                            "role": "assistant",
                            "content": "Use keyword search first.",
                            "parentId": "user-root",
                            "timestamp": 1700000002,
                            "files": [{"id": "file-1", "name": "notes.pdf"}],
                        },
                    },
                },
            },
        }
    ]


def _write_openwebui_db(service: ChatbookService) -> str:
    path = service.temp_dir / "webui.db"
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
                ("user-a", "Alice", "alice@example.test", "user", 1700000000, 1700000001),
                ("user-b", "Bob", "bob@example.test", "user", 1700000000, 1700000001),
            ],
        )
        conn.execute(
            """
            INSERT INTO folder (id, parent_id, user_id, name, items, meta, is_expanded, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "folder-a",
                None,
                "user-a",
                "Research",
                json.dumps(["chat-a"]),
                json.dumps({"color": "blue"}),
                1,
                1700000000,
                1700000001,
            ),
        )
        conn.executemany(
            """
            INSERT INTO chat (id, user_id, title, chat, created_at, updated_at, share_id, archived, pinned, meta, folder_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "chat-a",
                    "user-a",
                    "Alice DB chat",
                    json.dumps(_branched_export()[0]["chat"]),
                    1700000100,
                    1700000200,
                    None,
                    0,
                    1,
                    json.dumps({"project": "Migration"}),
                    "folder-a",
                ),
                (
                    "chat-b",
                    "user-b",
                    "Bob DB chat",
                    json.dumps(_branched_export()[0]["chat"]),
                    1700000100,
                    1700000200,
                    None,
                    0,
                    0,
                    json.dumps({}),
                    None,
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return str(path)


def test_import_openwebui_json_preserves_branch_parent_links_and_metadata(openwebui_service):
    path = _write_export(openwebui_service, _branched_export())

    success, message, result = openwebui_service.import_openwebui_json(path, ConflictResolution.SKIP)

    assert success is True, message
    assert result is not None
    assert result["imported_chats"] == 1
    assert result["imported_messages"] == 3

    db = openwebui_service.db
    conversation = db.get_conversation_by_source_ref(
        "openwebui",
        "chat-branched",
        client_id=db.client_id,
    )
    assert conversation is not None
    assert conversation["title"] == "Branch preserving import"

    messages = db.get_messages_for_conversation(conversation["id"])
    by_content = {message["content"]: message for message in messages}
    root = by_content["Compare two retrieval strategies."]
    main = by_content["Use hybrid retrieval."]
    alt = by_content["Use keyword search first."]
    assert root["parent_message_id"] is None
    assert main["parent_message_id"] == root["id"]
    assert alt["parent_message_id"] == root["id"]

    settings = db.get_conversation_settings(conversation["id"])
    assert settings is not None
    assert settings["settings"]["openwebui_import"]["history_current_id"] == "assistant-main"

    alt_metadata = db.get_message_metadata(alt["id"])
    assert alt_metadata is not None
    assert alt_metadata["extra"]["openwebui_import"]["source_message_id"] == "assistant-alt"
    assert alt_metadata["extra"]["openwebui_import"]["attachment_refs"] == [
        {"id": "file-1", "name": "notes.pdf"}
    ]


def test_preview_openwebui_db_lists_source_users(openwebui_service):
    path = _write_openwebui_db(openwebui_service)

    preview, error = openwebui_service.preview_openwebui_db(path)

    assert error is None
    assert preview["user_count"] == 2
    alice = next(user for user in preview["users"] if user["source_user_id"] == "user-a")
    assert alice["display_label"] == "Alice"
    assert alice["chat_count"] == 1
    assert alice["pinned_chat_count"] == 1


def test_import_openwebui_db_imports_only_selected_user(openwebui_service):
    path = _write_openwebui_db(openwebui_service)

    success, message, result = openwebui_service.import_openwebui_db(
        path,
        selected_user_id="user-a",
        conflict_resolution=ConflictResolution.SKIP,
    )

    assert success is True, message
    assert result["selected_user_id"] == "user-a"
    assert result["selected_user_label"] == "Alice"
    assert result["imported_chats"] == 1
    assert result["imported_messages"] == 3
    assert result["mirrored_folders"] == 1
    assert result["folder_links"] == 1

    db = openwebui_service.db
    imported = db.get_conversation_by_source_ref("openwebui", "chat-a", client_id=db.client_id)
    assert imported is not None
    assert db.get_conversation_by_source_ref("openwebui", "chat-b", client_id=db.client_id) is None

    settings = db.get_conversation_settings(imported["id"])
    metadata = settings["settings"]["openwebui_import"]["metadata"]
    assert metadata["source_kind"] == "openwebui_db"
    assert metadata["source_user_id"] == "user-a"
    assert metadata["folder_id"] == "folder-a"

    openwebui_root = db.get_keyword_collection_by_name("OpenWebUI")
    user_folder = db.get_keyword_collection_by_name("Alice (user-a)")
    research_folder = db.get_keyword_collection_by_name("Research")
    assert openwebui_root is not None
    assert user_folder is not None
    assert research_folder is not None
    collection_keyword_ids = {
        keyword["id"] for keyword in db.get_keywords_for_collection(research_folder["id"])
    }
    conversation_keyword_ids = {
        keyword["id"] for keyword in db.get_keywords_for_conversation(imported["id"])
    }
    assert collection_keyword_ids & conversation_keyword_ids


@pytest.mark.asyncio
async def test_import_chatbook_dispatches_openwebui_db_selected_user(openwebui_service):
    path = _write_openwebui_db(openwebui_service)

    success, message, result = await openwebui_service.import_chatbook(
        path,
        source_format="openwebui_db",
        selected_openwebui_user_id="user-a",
        conflict_resolution=ConflictResolution.SKIP,
    )

    assert success is True, message
    assert result["selected_user_id"] == "user-a"
    assert result["imported_chats"] == 1


def test_import_openwebui_json_duplicate_skip_uses_source_external_ref(openwebui_service):
    first_path = _write_export(openwebui_service, _branched_export())
    success, _, first_result = openwebui_service.import_openwebui_json(first_path, ConflictResolution.SKIP)
    assert success is True
    assert first_result["imported_chats"] == 1

    second_path = _write_export(openwebui_service, _branched_export())
    success, message, second_result = openwebui_service.import_openwebui_json(second_path, ConflictResolution.SKIP)

    assert success is True, message
    assert second_result["duplicate_chats"] == 1
    assert second_result["skipped_chats"] == 1
    assert second_result["imported_chats"] == 0


def test_preview_openwebui_json_returns_error_for_unsafe_upload_path(openwebui_service, tmp_path):
    outside_path = tmp_path / "outside-openwebui.json"
    outside_path.write_text("[]", encoding="utf-8")

    preview, error = openwebui_service.preview_openwebui_json(str(outside_path))

    assert preview is None
    assert error == "Invalid or potentially malicious import file"


def test_import_openwebui_json_rolls_back_when_message_insert_fails(openwebui_service, monkeypatch):
    payload = _branched_export()
    path = _write_export(openwebui_service, payload)

    original_add_message = openwebui_service.db.add_message

    def flaky_add_message(message_data):
        if message_data["content"] == "Use keyword search first.":
            return None
        return original_add_message(message_data)

    monkeypatch.setattr(openwebui_service.db, "add_message", flaky_add_message)

    success, message, result = openwebui_service.import_openwebui_json(path, ConflictResolution.SKIP)

    assert success is True, message
    assert result["imported_chats"] == 0
    assert result["imported_messages"] == 0
    assert result["failed_chats"] == 1

    assert openwebui_service.db.get_conversation_by_source_ref(
        "openwebui",
        "chat-branched",
        client_id=openwebui_service.db.client_id,
        include_deleted=True,
    ) is None


def test_preview_openwebui_json_returns_error_for_non_utf8_json(openwebui_service):
    path = openwebui_service.temp_dir / "bad-encoding.json"
    path.write_bytes(b"\xff\xfe")

    preview, error = openwebui_service.preview_openwebui_json(str(path))

    assert preview is None
    assert error == "Malformed OpenWebUI JSON export"


def test_import_openwebui_json_rolls_back_when_metadata_insert_fails(openwebui_service, monkeypatch):
    path = _write_export(openwebui_service, _branched_export())

    monkeypatch.setattr(openwebui_service.db, "set_message_metadata_extra", lambda *_args, **_kwargs: False)

    success, message, result = openwebui_service.import_openwebui_json(path, ConflictResolution.SKIP)

    assert success is True, message
    assert result["imported_chats"] == 0
    assert result["failed_chats"] == 1
    assert openwebui_service.db.get_conversation_by_source_ref(
        "openwebui",
        "chat-branched",
        client_id=openwebui_service.db.client_id,
        include_deleted=True,
    ) is None


def test_import_openwebui_json_rolls_back_conversation_when_chat_fails_after_create(
    openwebui_service,
    monkeypatch,
):
    path = _write_export(openwebui_service, _branched_export())

    def fail_settings(*_args, **_kwargs):
        raise DatabaseError("settings write failed")

    monkeypatch.setattr(openwebui_service, "_store_openwebui_conversation_settings", fail_settings)

    success, message, result = openwebui_service.import_openwebui_json(path, ConflictResolution.SKIP)

    assert success is True, message
    assert result["imported_chats"] == 0
    assert result["failed_chats"] == 1
    assert openwebui_service.db.get_conversation_by_source_ref(
        "openwebui",
        "chat-branched",
        client_id=openwebui_service.db.client_id,
        include_deleted=True,
    ) is None


def test_openwebui_copy_title_lookup_uses_db_title_abstraction():
    class FakeDB:
        client_id = "postgres-client"
        call = None

        def conversation_title_exists(self, title, *, client_id, include_deleted=False):
            self.call = {
                "title": title,
                "client_id": client_id,
                "include_deleted": include_deleted,
            }
            return True

    fake_db = FakeDB()
    service = object.__new__(ChatbookService)
    service.db = fake_db

    assert service._openwebui_conversation_title_exists("Existing title") is True
    assert fake_db.call == {
        "title": "Existing title",
        "client_id": "postgres-client",
        "include_deleted": False,
    }


def test_import_openwebui_json_duplicate_rename_creates_intentional_copy(openwebui_service):
    first_path = _write_export(openwebui_service, _branched_export())
    success, _, first_result = openwebui_service.import_openwebui_json(first_path, ConflictResolution.SKIP)
    assert success is True
    assert first_result["imported_chats"] == 1

    second_path = _write_export(openwebui_service, _branched_export())
    success, message, second_result = openwebui_service.import_openwebui_json(
        second_path,
        ConflictResolution.RENAME,
    )

    assert success is True, message
    assert second_result["duplicate_chats"] == 1
    assert second_result["skipped_chats"] == 0
    assert second_result["imported_chats"] == 1
    conversations = openwebui_service.db.get_conversations_for_user(openwebui_service.db.client_id)
    openwebui_conversations = [
        conversation
        for conversation in conversations
        if conversation["source"] == "openwebui"
    ]
    assert len(openwebui_conversations) == 2
    external_refs = {conversation["external_ref"] for conversation in openwebui_conversations}
    assert "chat-branched" in external_refs
    assert any(ref.startswith("chat-branched#copy:") for ref in external_refs)

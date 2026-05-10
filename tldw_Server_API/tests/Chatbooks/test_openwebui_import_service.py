import json

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

import sqlite3
from typing import Any

import pytest

from tldw_Server_API.app.core.Chatbooks import openwebui_hydration as hydration


pytestmark = pytest.mark.unit


class FakeChaChaDB:
    def __init__(
        self,
        *,
        messages_by_conversation: dict[str, list[dict[str, Any]]],
        metadata_by_message_id: dict[str, dict[str, Any]],
        settings_by_conversation: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.messages_by_conversation = messages_by_conversation
        self.metadata_by_message_id = metadata_by_message_id
        self.settings_by_conversation = settings_by_conversation or {}

    def get_messages_for_conversation(
        self,
        conversation_id: str,
        limit: int = 1000,
        offset: int = 0,
        order_by_timestamp: str = "ASC",
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        del limit, offset, order_by_timestamp, include_deleted
        return list(self.messages_by_conversation.get(conversation_id, []))

    def get_message_metadata_map(self, message_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {
            message_id: self.metadata_by_message_id[message_id]
            for message_id in message_ids
            if message_id in self.metadata_by_message_id
        }

    def get_conversation_settings(self, conversation_id: str) -> dict[str, Any] | None:
        return self.settings_by_conversation.get(conversation_id)


def _metadata(*, refs: list[Any] | None = None, source_message_id: str = "source-msg-a") -> dict[str, Any]:
    return {
        "extra": {
            "openwebui_import": {
                "source_message_id": source_message_id,
                "attachment_refs": list(refs or []),
                "metadata": {},
            }
        }
    }


def _chat_file_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE chat_file (
            id TEXT PRIMARY KEY,
            chat_id TEXT,
            file_id TEXT,
            message_id TEXT,
            user_id TEXT,
            created_at INTEGER,
            updated_at INTEGER
        )
        """
    )
    return conn


def test_imported_message_metadata_refs_are_extracted_from_openwebui_extra():
    db = FakeChaChaDB(
        messages_by_conversation={"conv-a": [{"id": "msg-a", "conversation_id": "conv-a"}]},
        metadata_by_message_id={
            "msg-a": _metadata(
                refs=[
                    {"id": "file-a", "name": "notes.pdf"},
                    "file-b",
                    {"file_id": "file-c"},
                    {"fileId": "file-d"},
                ]
            )
        },
    )

    preview = hydration.extract_openwebui_hydration_references(
        db,
        hydration.OpenWebUIHydrationScope(conversation_ids=("conv-a",)),
    )

    assert [ref.file_id for ref in preview.references] == ["file-a", "file-b", "file-c", "file-d"]
    assert [ref.raw_ref_index for ref in preview.references] == [0, 1, 2, 3]
    assert all(ref.message_id == "msg-a" for ref in preview.references)


def test_unsupported_reference_shapes_are_reported_as_preview_items():
    db = FakeChaChaDB(
        messages_by_conversation={"conv-a": [{"id": "msg-a", "conversation_id": "conv-a"}]},
        metadata_by_message_id={
            "msg-a": _metadata(refs=[{"name": "missing-id"}, [], 42, ""]),
        },
    )

    preview = hydration.extract_openwebui_hydration_references(
        db,
        hydration.OpenWebUIHydrationScope(conversation_ids=("conv-a",)),
    )

    assert preview.references == ()
    assert [item.status for item in preview.items] == [
        "unsupported_reference_shape",
        "unsupported_reference_shape",
        "unsupported_reference_shape",
        "unsupported_reference_shape",
    ]
    assert {item.warning_code for item in preview.items} == {"unsupported_reference_shape"}


def test_db_chat_file_fallback_uses_preserved_openwebui_source_chat_row_id():
    conn = _chat_file_connection()
    conn.executemany(
        """
        INSERT INTO chat_file (id, chat_id, file_id, message_id, user_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("link-a", "source-chat-a", "file-db", "source-msg-a", "owui-user", 1, 2),
            ("link-b", "source-chat-b", "file-other", "source-msg-a", "owui-user", 3, 4),
            ("link-c", "source-chat-a", "file-wrong-user", "source-msg-a", "other-user", 5, 6),
        ],
    )
    db = FakeChaChaDB(
        messages_by_conversation={"conv-a": [{"id": "msg-a", "conversation_id": "conv-a"}]},
        metadata_by_message_id={"msg-a": _metadata(refs=[])},
        settings_by_conversation={
            "conv-a": {
                "settings": {
                    "openwebui_import": {
                        "metadata": {"row_id": "source-chat-a"},
                    }
                }
            }
        },
    )

    preview = hydration.extract_openwebui_hydration_references(
        db,
        hydration.OpenWebUIHydrationScope(
            conversation_ids=("conv-a",),
            openwebui_user_id="owui-user",
        ),
        openwebui_conn=conn,
    )

    assert [ref.file_id for ref in preview.references] == ["file-db"]
    assert preview.references[0].message_id == "msg-a"
    assert preview.references[0].source == "chat_file"


def test_db_chat_file_fallback_skips_conversations_without_source_row_id():
    conn = _chat_file_connection()
    conn.execute(
        """
        INSERT INTO chat_file (id, chat_id, file_id, message_id, user_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("link-a", "source-chat-a", "file-db", "source-msg-a", "owui-user", 1, 2),
    )
    db = FakeChaChaDB(
        messages_by_conversation={"conv-a": [{"id": "msg-a", "conversation_id": "conv-a"}]},
        metadata_by_message_id={"msg-a": _metadata(refs=[])},
        settings_by_conversation={
            "conv-a": {
                "settings": {
                    "openwebui_import": {
                        "external_ref": "source-chat-a",
                        "metadata": {},
                    }
                }
            }
        },
    )

    preview = hydration.extract_openwebui_hydration_references(
        db,
        hydration.OpenWebUIHydrationScope(
            conversation_ids=("conv-a",),
            openwebui_user_id="owui-user",
        ),
        openwebui_conn=conn,
    )

    assert preview.references == ()
    assert preview.items == ()

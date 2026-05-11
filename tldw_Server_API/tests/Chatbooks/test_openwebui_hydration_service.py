import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Chatbooks import openwebui_hydration as hydration
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError


pytestmark = pytest.mark.unit

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png"


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


@pytest.fixture()
def real_hydration_db(tmp_path):
    db = CharactersRAGDB(
        db_path=str(tmp_path / "openwebui-hydration.sqlite"),
        client_id="openwebui-hydration-test",
    )
    character_id = db.add_character_card({"name": "Hydration Assistant"})
    conversation_id = db.add_conversation({"id": "conv-a", "character_id": character_id, "title": "Hydration"})
    message_id = db.add_message(
        {
            "id": "msg-a",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "message with existing image",
            "images": [{"data": PNG_BYTES, "mime": "image/png"}],
        }
    )
    db.add_message_metadata(
        message_id,
        extra={
            "openwebui_import": {
                "source_message_id": "source-msg-a",
                "source_parent_id": None,
                "source_children_ids": [],
                "role": "user",
                "model": "model-a",
                "attachment_refs": [{"id": "file-image"}],
                "metadata": {"done": True},
            }
        },
    )
    return db


def _resolved_file(path: Path, *, file_id: str = "file-image") -> hydration.OpenWebUIHydrationResolvedFile:
    file_kind, mime_type = hydration.classify_openwebui_file(path)
    return hydration.OpenWebUIHydrationResolvedFile(
        file_id=file_id,
        filename=path.name,
        path=path,
        status="resolved",
        source="file_path",
        file_kind=file_kind,
        mime_type=mime_type,
    )


def _reference(*, file_id: str = "file-image") -> hydration.OpenWebUIHydrationReference:
    return hydration.OpenWebUIHydrationReference(
        conversation_id="conv-a",
        message_id="msg-a",
        file_id=file_id,
        raw_ref_index=0,
        raw_ref={"id": file_id},
        source="message_metadata",
        source_message_id="source-msg-a",
    )


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


def test_hydrate_png_ref_appends_image_and_preserves_openwebui_metadata(real_hydration_db, tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(PNG_BYTES)

    item = hydration.hydrate_image_reference(
        real_hydration_db,
        _reference(),
        _resolved_file(image_path),
        job_id="job-a",
    )

    images = real_hydration_db.get_message_images("msg-a")
    metadata = real_hydration_db.get_message_metadata("msg-a")
    openwebui_import = metadata["extra"]["openwebui_import"]

    assert item.status == "hydrated_image"
    assert item.file_id == "file-image"
    assert item.job_id == "job-a"
    assert images[1]["position"] == 1
    assert images[1]["image_data"] == PNG_BYTES
    assert openwebui_import["source_message_id"] == "source-msg-a"
    assert openwebui_import["source_parent_id"] is None
    assert openwebui_import["source_children_ids"] == []
    assert openwebui_import["role"] == "user"
    assert openwebui_import["model"] == "model-a"
    assert openwebui_import["attachment_refs"] == [{"id": "file-image"}]
    assert openwebui_import["metadata"] == {"done": True}
    assert openwebui_import["hydration"]["last_job_id"] == "job-a"
    assert openwebui_import["hydration"]["items"][0]["status"] == "hydrated_image"
    assert openwebui_import["hydration"]["items"][0]["message_image_position"] == 1


def test_hydrate_png_ref_is_idempotent_for_existing_source_key(real_hydration_db, tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(PNG_BYTES)
    resolved = _resolved_file(image_path)
    reference = _reference()

    first = hydration.hydrate_image_reference(real_hydration_db, reference, resolved, job_id="job-a")
    second = hydration.hydrate_image_reference(real_hydration_db, reference, resolved, job_id="job-b")

    images = real_hydration_db.get_message_images("msg-a")
    metadata = real_hydration_db.get_message_metadata("msg-a")
    hydration_items = metadata["extra"]["openwebui_import"]["hydration"]["items"]

    assert first.status == "hydrated_image"
    assert second.status == "already_hydrated"
    assert len(images) == 2
    assert len(hydration_items) == 1
    assert hydration_items[0]["message_image_position"] == 1


def test_hydrate_image_ref_rolls_back_when_metadata_update_fails(real_hydration_db, tmp_path, monkeypatch):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(PNG_BYTES)

    def fail_metadata_update(*_args, **_kwargs):
        return False

    monkeypatch.setattr(real_hydration_db, "add_message_metadata", fail_metadata_update)

    item = hydration.hydrate_image_reference(
        real_hydration_db,
        _reference(file_id="file-rollback"),
        _resolved_file(image_path, file_id="file-rollback"),
        job_id="job-a",
    )

    assert item.status == "metadata_update_failed"
    assert len(real_hydration_db.get_message_images("msg-a")) == 1


def test_append_message_image_enforces_existing_message_image_byte_cap(real_hydration_db):
    oversized = b"x" * ((5 * 1024 * 1024) + 1)

    with pytest.raises(InputError, match="maximum size"):
        real_hydration_db.append_message_image("msg-a", oversized, "image/png")


def test_hydrate_image_ref_reports_oversized_without_inserting(real_hydration_db, tmp_path):
    image_path = tmp_path / "large.png"
    image_path.write_bytes(PNG_BYTES + b"x" * 32)

    item = hydration.hydrate_image_reference(
        real_hydration_db,
        _reference(file_id="file-large"),
        _resolved_file(image_path, file_id="file-large"),
        job_id="job-a",
        max_image_bytes=8,
    )

    assert item.status == "oversized"
    assert len(real_hydration_db.get_message_images("msg-a")) == 1


def test_hydrate_image_ref_rejects_png_extension_with_non_image_bytes(real_hydration_db, tmp_path):
    image_path = tmp_path / "fake.png"
    image_path.write_bytes(b"not actually an image")

    item = hydration.hydrate_image_reference(
        real_hydration_db,
        _reference(file_id="file-fake"),
        _resolved_file(image_path, file_id="file-fake"),
        job_id="job-a",
    )

    assert item.status == "unsupported_file_type"
    assert len(real_hydration_db.get_message_images("msg-a")) == 1

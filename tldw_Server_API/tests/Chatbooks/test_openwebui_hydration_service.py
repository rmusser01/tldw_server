import hashlib
import json
import sqlite3
import stat
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Chatbooks import openwebui_hydration as hydration
from tldw_Server_API.app.core.Chatbooks import chatbook_service as chatbook_service_mod
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


pytestmark = pytest.mark.unit

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png"
PDF_BYTES = b"%PDF-1.4\nfake-pdf\n%%EOF"


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


def _patch_allowed_roots(monkeypatch: pytest.MonkeyPatch, allowed_root: Path) -> None:
    monkeypatch.setattr(
        hydration,
        "get_ingestion_source_allowed_roots",
        lambda *, reload=False: (allowed_root.resolve(strict=False),),
    )


def _write_openwebui_hydration_db(data_root: Path) -> None:
    conn = sqlite3.connect(data_root / "webui.db")
    try:
        conn.execute(
            """
            CREATE TABLE user (
                id TEXT,
                name TEXT,
                email TEXT,
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE folder (
                id TEXT,
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
                id TEXT,
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
        conn.execute(
            """
            CREATE TABLE file (
                id TEXT,
                user_id TEXT,
                hash TEXT,
                filename TEXT,
                path TEXT,
                data TEXT,
                meta TEXT,
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE chat_file (
                id TEXT,
                chat_id TEXT,
                file_id TEXT,
                message_id TEXT,
                user_id TEXT,
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO file (id, user_id, hash, filename, path, data, meta, created_at, updated_at)
            VALUES ('file-image', 'ow-user', 'hash-image', 'image.png', 'uploads/file-image_image.png', '{}', '{}', 1, 1)
            """
        )
        conn.commit()
    finally:
        conn.close()


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


@pytest.fixture()
def media_db(tmp_path):
    db = MediaDatabase(
        db_path=str(tmp_path / "openwebui-media.sqlite"),
        client_id="101",
    )
    yield db
    db.close_connection()


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


def _reference(
    *,
    file_id: str = "file-image",
    message_id: str | None = "msg-a",
    source_message_id: str | None = "source-msg-a",
) -> hydration.OpenWebUIHydrationReference:
    return hydration.OpenWebUIHydrationReference(
        conversation_id="conv-a",
        message_id=message_id,
        file_id=file_id,
        raw_ref_index=0,
        raw_ref={"id": file_id},
        source="message_metadata",
        source_message_id=source_message_id,
    )


def _add_hydration_message(db: CharactersRAGDB, message_id: str) -> str | None:
    return db.add_message(
        {
            "id": message_id,
            "conversation_id": "conv-a",
            "sender": "user",
            "content": f"message {message_id}",
        }
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


def test_db_chat_file_fallback_batches_source_chat_rows(monkeypatch):
    conn = _chat_file_connection()
    calls: list[tuple[tuple[str, ...], str | None]] = []

    def fake_load_chat_file_rows(_conn, chat_ids, user_id=None):
        calls.append((tuple(chat_ids), user_id))
        return [
            {
                "id": "link-a",
                "chat_id": "source-chat-a",
                "file_id": "file-a",
                "message_id": "source-msg-a",
                "user_id": "owui-user",
                "created_at": 1,
                "updated_at": 1,
            },
            {
                "id": "link-b",
                "chat_id": "source-chat-b",
                "file_id": "file-b",
                "message_id": "source-msg-b",
                "user_id": "owui-user",
                "created_at": 2,
                "updated_at": 2,
            },
        ]

    monkeypatch.setattr(hydration, "load_openwebui_chat_file_rows_for_chats", fake_load_chat_file_rows)
    db = FakeChaChaDB(
        messages_by_conversation={
            "conv-a": [{"id": "msg-a", "conversation_id": "conv-a"}],
            "conv-b": [{"id": "msg-b", "conversation_id": "conv-b"}],
        },
        metadata_by_message_id={
            "msg-a": _metadata(refs=[], source_message_id="source-msg-a"),
            "msg-b": _metadata(refs=[], source_message_id="source-msg-b"),
        },
        settings_by_conversation={
            "conv-a": {"settings": {"openwebui_import": {"metadata": {"row_id": "source-chat-a"}}}},
            "conv-b": {"settings": {"openwebui_import": {"metadata": {"row_id": "source-chat-b"}}}},
        },
    )

    preview = hydration.extract_openwebui_hydration_references(
        db,
        hydration.OpenWebUIHydrationScope(
            conversation_ids=("conv-a", "conv-b"),
            openwebui_user_id="owui-user",
        ),
        openwebui_conn=conn,
    )

    assert calls == [(("source-chat-a", "source-chat-b"), "owui-user")]
    assert [(ref.conversation_id, ref.message_id, ref.file_id) for ref in preview.references] == [
        ("conv-a", "msg-a", "file-a"),
        ("conv-b", "msg-b", "file-b"),
    ]


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


def test_append_message_image_retries_transient_position_conflict(real_hydration_db, monkeypatch):
    original_execute_query = real_hydration_db.execute_query
    conflicts = {"count": 0}

    def flaky_execute_query(query, *args, **kwargs):
        if "INSERT INTO message_images" in str(query) and conflicts["count"] == 0:
            conflicts["count"] += 1
            raise sqlite3.IntegrityError(
                "UNIQUE constraint failed: message_images.message_id, message_images.position"
            )
        return original_execute_query(query, *args, **kwargs)

    monkeypatch.setattr(real_hydration_db, "execute_query", flaky_execute_query)

    position = real_hydration_db.append_message_image("msg-a", PNG_BYTES, "image/png")

    assert conflicts["count"] == 1
    assert position == 1
    assert len(real_hydration_db.get_message_images("msg-a")) == 2


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


def test_hydrate_image_ref_checks_size_with_bounded_read(real_hydration_db, tmp_path, monkeypatch):
    image_path = tmp_path / "large.png"
    image_path.write_bytes(PNG_BYTES + b"x" * 32)
    original_read_bytes = Path.read_bytes

    def fail_unbounded_read(path: Path) -> bytes:
        if path == image_path:
            raise AssertionError("image hydration must not read the full file before size checks")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_unbounded_read)

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


def test_register_pdf_ref_creates_owned_media_and_media_file(real_hydration_db, media_db, tmp_path):
    source_root = tmp_path / "openwebui-source"
    source_root.mkdir()
    pdf_path = source_root / "source.pdf"
    pdf_path.write_bytes(PDF_BYTES)
    storage_root = tmp_path / "tldw-owned-storage"
    expected_hash = hashlib.sha256(PDF_BYTES).hexdigest()

    item = hydration.register_non_image_reference(
        real_hydration_db,
        media_db,
        _reference(file_id="file-pdf"),
        _resolved_file(pdf_path, file_id="file-pdf"),
        owner_user_id=101,
        storage_root=storage_root,
        job_id="job-media",
    )

    media = media_db.get_media_by_id(item.media_id)
    media_file = media_db.get_media_file(item.media_id, "original")
    metadata = real_hydration_db.get_message_metadata("msg-a")
    hydration_item = metadata["extra"]["openwebui_import"]["hydration"]["items"][0]
    safe_metadata = json.loads(media_db.get_all_document_versions(item.media_id)[0]["safe_metadata"])

    assert item.status == "registered_media"
    assert item.media_id is not None
    assert item.media_file_id == media_file["uuid"]
    assert item.checksum == expected_hash
    assert media["url"] == "openwebui://user/101/file/file-pdf"
    assert media["source_hash"] == expected_hash
    assert media["owner_user_id"] == 101
    assert media["visibility"] == "personal"
    assert media_file["checksum"] == expected_hash
    assert media_file["mime_type"] == "application/pdf"
    assert str(source_root) not in media_file["storage_path"]
    assert (storage_root / media_file["storage_path"]).read_bytes() == PDF_BYTES
    assert hydration_item["status"] == "registered_media"
    assert hydration_item["media_id"] == item.media_id
    assert hydration_item["media_file_id"] == item.media_file_id
    assert safe_metadata["source"] == "openwebui"
    assert safe_metadata["source_file_id"] == "file-pdf"
    assert safe_metadata["sha256"] == expected_hash
    assert "source_path" not in safe_metadata


def test_register_same_source_file_reuses_owned_media_link(real_hydration_db, media_db, tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(PDF_BYTES)
    storage_root = tmp_path / "tldw-owned-storage"
    reference = _reference(file_id="file-reused")
    resolved = _resolved_file(pdf_path, file_id="file-reused")

    first = hydration.register_non_image_reference(
        real_hydration_db,
        media_db,
        reference,
        resolved,
        owner_user_id=101,
        storage_root=storage_root,
        job_id="job-media",
    )
    second = hydration.register_non_image_reference(
        real_hydration_db,
        media_db,
        reference,
        resolved,
        owner_user_id=101,
        storage_root=storage_root,
        job_id="job-media",
    )

    assert first.media_id == second.media_id
    assert second.status == "already_registered_media"
    assert len(media_db.get_media_files(first.media_id)) == 1


def test_register_same_source_file_does_not_cross_tldw_users(real_hydration_db, media_db, tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(PDF_BYTES)
    storage_root = tmp_path / "tldw-owned-storage"
    _add_hydration_message(real_hydration_db, "msg-b")

    first = hydration.register_non_image_reference(
        real_hydration_db,
        media_db,
        _reference(file_id="file-shared", message_id="msg-a"),
        _resolved_file(pdf_path, file_id="file-shared"),
        owner_user_id=101,
        storage_root=storage_root,
        job_id="job-media",
    )
    second = hydration.register_non_image_reference(
        real_hydration_db,
        media_db,
        _reference(file_id="file-shared", message_id="msg-b", source_message_id="source-msg-b"),
        _resolved_file(pdf_path, file_id="file-shared"),
        owner_user_id=202,
        storage_root=storage_root,
        job_id="job-media",
    )

    assert first.media_id != second.media_id
    assert media_db.get_media_by_id(first.media_id)["url"] != media_db.get_media_by_id(second.media_id)["url"]
    assert media_db.get_media_by_id(first.media_id)["owner_user_id"] == 101
    assert media_db.get_media_by_id(second.media_id)["owner_user_id"] == 202


def test_register_source_id_less_files_do_not_share_placeholder_content_hash(real_hydration_db, media_db, tmp_path):
    first_path = tmp_path / "alpha.txt"
    second_path = tmp_path / "beta.txt"
    first_path.write_bytes(b"alpha")
    second_path.write_bytes(b"beta")
    storage_root = tmp_path / "tldw-owned-storage"
    _add_hydration_message(real_hydration_db, "msg-b")

    first = hydration.register_non_image_reference(
        real_hydration_db,
        media_db,
        _reference(file_id="", message_id="msg-a"),
        _resolved_file(first_path, file_id=""),
        owner_user_id=101,
        storage_root=storage_root,
        job_id="job-media",
    )
    second = hydration.register_non_image_reference(
        real_hydration_db,
        media_db,
        _reference(file_id="", message_id="msg-b", source_message_id="source-msg-b"),
        _resolved_file(second_path, file_id=""),
        owner_user_id=101,
        storage_root=storage_root,
        job_id="job-media",
    )

    assert first.media_id != second.media_id
    assert media_db.get_media_by_id(first.media_id)["content_hash"] != media_db.get_media_by_id(
        second.media_id
    )["content_hash"]


def test_register_non_image_processing_hook_is_optional_and_failure_keeps_media_file(
    real_hydration_db,
    media_db,
    tmp_path,
):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(PDF_BYTES)
    storage_root = tmp_path / "tldw-owned-storage"
    calls: list[int] = []

    def failing_processor(**kwargs):
        calls.append(kwargs["media_id"])
        raise RuntimeError("processor failed")

    skipped = hydration.register_non_image_reference(
        real_hydration_db,
        media_db,
        _reference(file_id="file-no-process"),
        _resolved_file(pdf_path, file_id="file-no-process"),
        owner_user_id=101,
        storage_root=storage_root,
        job_id="job-media",
        process_supported_files=False,
        processing_hook=failing_processor,
    )
    _add_hydration_message(real_hydration_db, "msg-b")
    processed = hydration.register_non_image_reference(
        real_hydration_db,
        media_db,
        _reference(file_id="file-process", message_id="msg-b", source_message_id="source-msg-b"),
        _resolved_file(pdf_path, file_id="file-process"),
        owner_user_id=101,
        storage_root=storage_root,
        job_id="job-media",
        process_supported_files=True,
        processing_hook=failing_processor,
    )

    assert calls == [processed.media_id]
    assert skipped.processing_status == "skipped"
    assert processed.status == "registered_media"
    assert processed.warning_code == "processing_failed"
    assert media_db.get_media_file(processed.media_id, "original") is not None


def test_register_non_image_reports_copy_failure_without_terminating(
    real_hydration_db,
    media_db,
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(PDF_BYTES)
    storage_root = tmp_path / "tldw-owned-storage"

    def fail_copy(**_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(hydration, "_copy_openwebui_attachment_to_storage", fail_copy)

    item = hydration.register_non_image_reference(
        real_hydration_db,
        media_db,
        _reference(file_id="file-copy-fails"),
        _resolved_file(pdf_path, file_id="file-copy-fails"),
        owner_user_id=101,
        storage_root=storage_root,
        job_id="job-media",
    )

    assert item.status == "media_registration_failed"
    assert item.warning_code == "media_registration_failed"
    assert item.media_id is not None


def test_register_non_image_reports_media_file_insert_failure_without_terminating(
    real_hydration_db,
    media_db,
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(PDF_BYTES)
    storage_root = tmp_path / "tldw-owned-storage"

    def fail_insert_media_file(**_kwargs):
        raise RuntimeError("insert failed")

    monkeypatch.setattr(media_db, "insert_media_file", fail_insert_media_file)

    item = hydration.register_non_image_reference(
        real_hydration_db,
        media_db,
        _reference(file_id="file-insert-fails"),
        _resolved_file(pdf_path, file_id="file-insert-fails"),
        owner_user_id=101,
        storage_root=storage_root,
        job_id="job-media",
    )

    assert item.status == "media_registration_failed"
    assert item.warning_code == "media_registration_failed"
    assert item.media_id is not None
    assert item.storage_path is not None


def test_preview_and_run_hydration_cap_response_items(tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    data_root = allowed_root / "openwebui"
    uploads_dir = data_root / "uploads"
    uploads_dir.mkdir(parents=True)
    _write_openwebui_hydration_db(data_root)
    _patch_allowed_roots(monkeypatch, allowed_root)
    monkeypatch.setattr(chatbook_service_mod, "MAX_OPENWEBUI_HYDRATION_RESPONSE_ITEMS", 3)
    monkeypatch.setattr(chatbook_service_mod, "MAX_PREVIEW_WARNING_ITEMS", 3)
    refs = [{"id": f"missing-{index}"} for index in range(5)]
    db = FakeChaChaDB(
        messages_by_conversation={"conv-a": [{"id": "msg-a", "conversation_id": "conv-a"}]},
        metadata_by_message_id={"msg-a": _metadata(refs=refs)},
    )
    service = ChatbookService("101", db, user_id_int=101)
    monkeypatch.setattr(service, "_get_media_db", lambda: None)

    preview = service.preview_openwebui_attachment_hydration(
        openwebui_data_root=str(data_root),
        scope={"conversation_ids": ["conv-a"], "source_user_id": "ow-user"},
    )
    result = service.run_openwebui_attachment_hydration(
        openwebui_data_root=str(data_root),
        scope={"conversation_ids": ["conv-a"], "source_user_id": "ow-user"},
        job_id="job-run",
    )

    assert preview["summary"]["referenced_files"] == 5
    assert preview["summary"]["returned_items"] == 3
    assert preview["summary"]["omitted_items"] == 2
    assert len(preview["items"]) == 3
    assert result["summary"]["referenced_files"] == 5
    assert result["summary"]["returned_items"] == 3
    assert result["summary"]["omitted_items"] == 2
    assert result["summary"]["warning_count"] == 5
    assert len(result["items"]) == 3
    assert len(result["warnings"]) == 3


def test_openwebui_attachment_storage_root_is_private(tmp_path, monkeypatch):
    storage_root = tmp_path / "media-storage"
    storage_root.mkdir(mode=0o755)
    storage_root.chmod(0o755)
    monkeypatch.setenv("OPENWEBUI_HYDRATION_MEDIA_STORAGE_PATH", str(storage_root))
    service = ChatbookService(
        "101",
        FakeChaChaDB(messages_by_conversation={}, metadata_by_message_id={}),
        user_id_int=101,
    )

    resolved = service._openwebui_attachment_storage_root()

    assert resolved == storage_root
    assert stat.S_IMODE(resolved.stat().st_mode) == 0o700


def test_run_openwebui_attachment_hydration_hydrates_resolved_image(
    real_hydration_db,
    tmp_path,
    monkeypatch,
):
    allowed_root = tmp_path / "allowed"
    data_root = allowed_root / "openwebui"
    uploads_dir = data_root / "uploads"
    uploads_dir.mkdir(parents=True)
    (uploads_dir / "file-image_image.png").write_bytes(PNG_BYTES)
    _write_openwebui_hydration_db(data_root)
    _patch_allowed_roots(monkeypatch, allowed_root)
    service = ChatbookService("101", real_hydration_db, user_id_int=101)

    result = service.run_openwebui_attachment_hydration(
        openwebui_data_root=str(data_root),
        scope={"conversation_ids": ["conv-a"], "source_user_id": "ow-user"},
        job_id="job-run",
    )

    assert result["summary"]["hydrated_images"] == 1
    assert result["summary"]["resolved_files"] == 1
    assert len(real_hydration_db.get_message_images("msg-a")) == 2

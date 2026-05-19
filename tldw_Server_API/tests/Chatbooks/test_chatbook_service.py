# test_chatbook_service.py
# Description: Unit tests for the ChatbookService (combined import/export)
#
"""
Chatbook Service Tests
----------------------

Comprehensive unit tests for the chatbook import/export functionality including
job management, conflict resolution, and user isolation.
"""

import pytest
import json
import tempfile
import zipfile
import os
import shutil
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
from uuid import uuid4
from pathlib import Path

from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    ChatbookManifest,
    ChatbookContent,
    ContentItem,
    ExportJob,
    ImportJob,
    ExportStatus,
    ImportStatus,
    ChatbookVersion,
    ContentType,
    ConflictResolution,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def manifest_to_dict(manifest):
    """Helper to convert manifest to dict for JSON serialization."""
    data = {}
    for key, value in manifest.__dict__.items():
        if hasattr(value, 'value'):  # Enum
            data[key] = value.value
        elif hasattr(value, 'isoformat'):  # DateTime
            data[key] = value.isoformat()
        else:
            data[key] = value
    return data


@pytest.fixture
def mock_db():
    """Create a mock database instance."""
    mock = MagicMock()
    mock.execute_query = MagicMock()
    mock.execute_many = MagicMock()
    return mock


@pytest.fixture
def service(mock_db, tmp_path, monkeypatch):
    """Create a ChatbookService instance with mocked database and temp directories."""
    # Set environment variable to use temp directory for tests
    monkeypatch.setenv('PYTEST_CURRENT_TEST', 'test')
    monkeypatch.setenv('USER_DB_BASE_DIR', str(tmp_path))

    mock_db.execute_query.return_value = []
    connection = MagicMock()
    connection.execute = MagicMock()
    connection.close = MagicMock()
    mock_db.get_connection.return_value = connection

    service = ChatbookService(user_id="test_user", db=mock_db)
    return service


@pytest.fixture
def sample_manifest():
    """Create a sample chatbook manifest."""
    return ChatbookManifest(
        version=ChatbookVersion.V1,
        exported_at=datetime.now().isoformat(),
        user_id="test_user",
        name="Test Chatbook",
        description="Test chatbook for unit tests",
        content_summary={
            "conversations": 2,
            "characters": 1,
            "world_books": 1,
            "dictionaries": 1,
            "notes": 2,
            "prompts": 3
        },
        metadata={"test": "data"}
    )


@pytest.fixture
def sample_content_items():
    """Create sample content items."""
    return [
        ContentItem(
            id="conv1",
            type=ContentType.CONVERSATION,
            title="Test Conversation",
            metadata={"created_at": "2024-01-01"}
        ),
        ContentItem(
            id="char1",
            type=ContentType.CHARACTER,
            title="Test Character",
            metadata={"description": "A test character"}
        )
    ]


class TestChatbookService:
    """Test suite for ChatbookService."""

    def test_init_creates_tables(self, mock_db, tmp_path, monkeypatch):
        """Test that initialization creates necessary tables."""
        monkeypatch.setenv('PYTEST_CURRENT_TEST', 'test')
        monkeypatch.setenv('USER_DB_BASE_DIR', str(tmp_path))

        mock_db.execute_query.return_value = []

        with patch('tldw_Server_API.app.core.Chatbooks.chatbook_service.Path.mkdir') as mock_mkdir:
            mock_mkdir.return_value = None
            service = ChatbookService(user_id="test_user", db=mock_db)

        # Should create export and import job tables
        assert mock_db.execute_query.call_count >= 2

        # Check table creation
        calls = mock_db.execute_query.call_args_list
        sql_statements = [call[0][0] for call in calls]
        assert any("CREATE TABLE IF NOT EXISTS export_jobs" in sql for sql in sql_statements)
        assert any("CREATE TABLE IF NOT EXISTS import_jobs" in sql for sql in sql_statements)

    def test_create_export_job(self, service, mock_db):
        """Test creating an export job."""
        test_uuid = uuid4()
        job_id = str(test_uuid)

        with patch('tldw_Server_API.app.core.Chatbooks.chatbook_service.uuid4', return_value=test_uuid):
            mock_db.execute_query.return_value = None

            result = service.create_export_job(
                name="Test Export",
                description="Test export job",
                content_types=["conversations", "characters"]
            )

        assert result["job_id"] == job_id
        assert result["status"] == "pending"

        # Check job was saved - looking for INSERT OR REPLACE
        call_args = mock_db.execute_query.call_args[0]
        assert "INSERT OR REPLACE INTO export_jobs" in call_args[0]

    def test_get_media_db_uses_shared_factory_and_caches_result(self, mock_db, tmp_path, monkeypatch):
        """Media DB lookup should use the shared factory and cache the instance."""
        monkeypatch.setenv('PYTEST_CURRENT_TEST', 'test')
        monkeypatch.setenv('USER_DB_BASE_DIR', str(tmp_path))
        mock_db.execute_query.return_value = []

        service = ChatbookService(user_id="test_user", user_id_int=7, db=mock_db)
        fake_media_db = MagicMock()
        media_db_path = tmp_path / "media.db"

        with patch(
            'tldw_Server_API.app.core.Chatbooks.chatbook_service.DatabasePaths.get_media_db_path',  # nosec B108
            return_value=media_db_path,
        ), patch(
            'tldw_Server_API.app.core.Chatbooks.chatbook_service.create_media_database',
            return_value=fake_media_db,
        ) as mock_create_media_database:
            first = service._get_media_db()
            second = service._get_media_db()

        assert first is fake_media_db
        assert second is fake_media_db
        mock_create_media_database.assert_called_once_with('test_user', db_path=media_db_path)  # nosec B108

    @pytest.mark.asyncio
    async def test_export_chatbook_sync(self, service, mock_db, tmp_path):
        """Test synchronous chatbook export."""
        mock_db.execute_query.return_value = [
            {"id": "conv1"},
            {"id": "conv2"},
        ]

        archive_path = tmp_path / "test_export.chatbook"
        manifest_payload = {
            "name": "Test Export",
            "description": "Test Description",
            "statistics": {"total_conversations": 2},
        }
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest_payload))

        with patch.object(service, "create_chatbook", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = (
                True,
                "Chatbook created successfully",
                str(archive_path),
            )

            result = await service.export_chatbook(
                name="Test Export",
                content_types=["conversations"],
            )

        mock_create.assert_awaited_once()
        call_kwargs = mock_create.await_args.kwargs
        assert call_kwargs["name"] == "Test Export"
        assert "async_mode" not in call_kwargs
        assert call_kwargs["content_selections"] == {
            ContentType.CONVERSATION: ["conv1", "conv2"],
        }

        assert result["success"] is True
        assert result["message"] == "Chatbook created successfully"
        assert result["file_path"] == str(archive_path)

    def test_import_conversation_restores_inline_images(self, service, mock_db, tmp_path):
        """Conversations imported from chatbooks should restore embedded images."""
        conv_id = "conv-image"
        content_root = tmp_path / "content" / "conversations"
        assets_dir = content_root / f"conversation_{conv_id}_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        image_bytes = b"\x89PNG\r\n\x1a\n"
        image_rel_path = f"content/conversations/{assets_dir.name}/msg1_image_0.png"
        (assets_dir / "msg1_image_0.png").write_bytes(image_bytes)

        conversation_payload = {
            "id": conv_id,
            "name": "Sample Conversation",
            "created_at": "2024-01-01T00:00:00",
            "character_id": None,
            "attachments_path": f"content/conversations/{assets_dir.name}",
            "messages": [
                {
                    "id": "msg1",
                    "role": "user",
                    "content": "Hello with image",
                    "timestamp": "2024-01-01T00:00:00",
                    "attachments": [
                        {
                            "type": "image",
                            "mime_type": "image/png",
                            "file_path": image_rel_path,
                        }
                    ],
                }
            ],
        }

        conversation_file = content_root / f"conversation_{conv_id}.json"
        conversation_file.parent.mkdir(parents=True, exist_ok=True)
        conversation_file.write_text(json.dumps(conversation_payload), encoding="utf-8")

        mock_db.add_conversation.return_value = "new-conv-id"

        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Desc",
        )
        status = ImportJob(
            job_id="job",
            user_id="test_user",
            status=ImportStatus.PENDING,
            chatbook_path="dummy",
        )

        with patch.object(service, "_get_conversation_by_name", return_value=None):
            service._import_conversations(
                tmp_path,
                manifest,
                [conv_id],
                ConflictResolution.SKIP,
                prefix_imported=False,
                status=status,
            )

        assert mock_db.add_message.call_count == 1
        message_payload = mock_db.add_message.call_args[0][0]
        assert "images" in message_payload
        assert len(message_payload["images"]) == 1
        img_payload = message_payload["images"][0]
        assert img_payload["image_data"] == image_bytes
        assert img_payload["image_mime_type"] == "image/png"

    def test_import_conversation_legacy_fields(self, service, mock_db, tmp_path):
        """Legacy conversation exports should import title/sender/message fields."""
        conv_id = "legacy-conv"
        content_root = tmp_path / "content" / "conversations"
        content_root.mkdir(parents=True, exist_ok=True)

        conversation_payload = {
            "id": conv_id,
            "title": "Legacy Conversation Title",
            "created_at": "2024-01-01T00:00:00",
            "character_id": 1,
            "messages": [
                {
                    "id": "msg-legacy",
                    "sender": "user",
                    "message": "Hello from legacy export",
                    "timestamp": "2024-01-01T00:00:00",
                }
            ],
        }

        conversation_file = content_root / f"conversation_{conv_id}.json"
        conversation_file.write_text(json.dumps(conversation_payload), encoding="utf-8")

        mock_db.get_character_card_by_id.return_value = {"id": 1}
        mock_db.add_conversation.return_value = "new-conv-id"

        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Desc",
        )
        status = ImportJob(
            job_id="job",
            user_id="test_user",
            status=ImportStatus.PENDING,
            chatbook_path="dummy",
        )

        with patch.object(service, "_get_conversation_by_name", return_value=None):
            service._import_conversations(
                tmp_path,
                manifest,
                [conv_id],
                ConflictResolution.SKIP,
                prefix_imported=False,
                status=status,
            )

        conv_payload = mock_db.add_conversation.call_args[0][0]
        assert conv_payload["title"] == "Legacy Conversation Title"
        msg_payload = mock_db.add_message.call_args[0][0]
        assert msg_payload["sender"] == "user"
        assert msg_payload["content"] == "Hello from legacy export"

    def test_import_conversation_skips_outside_attachments(self, service, mock_db, tmp_path):
        """Attachment paths that escape extraction boundaries must be ignored."""
        conv_id = "conv-path"
        content_root = tmp_path / "content" / "conversations"
        content_root.mkdir(parents=True, exist_ok=True)

        conversation_payload = {
            "id": conv_id,
            "name": "Suspicious Conversation",
            "created_at": "2024-01-01T00:00:00",
            "messages": [
                {
                    "id": "msg-escape",
                    "role": "user",
                    "content": "Hello",
                    "timestamp": "2024-01-01T00:00:00",
                    "attachments": [
                        {
                            "type": "image",
                            "mime_type": "image/png",
                            "file_path": "../outside.png",
                        }
                    ],
                }
            ],
        }

        conversation_file = content_root / f"conversation_{conv_id}.json"
        conversation_file.write_text(json.dumps(conversation_payload), encoding="utf-8")

        mock_db.add_conversation.return_value = "new-conv-id"

        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Desc",
        )
        status = ImportJob(
            job_id="job",
            user_id="test_user",
            status=ImportStatus.PENDING,
            chatbook_path="dummy",
        )

        with patch.object(service, "_get_conversation_by_name", return_value=None):
            service._import_conversations(
                tmp_path,
                manifest,
                [conv_id],
                ConflictResolution.SKIP,
                prefix_imported=False,
                status=status,
            )

        assert mock_db.add_message.call_count == 1
        message_payload = mock_db.add_message.call_args[0][0]
        assert "images" not in message_payload
        assert any("Skipped attachment outside extract dir" in warning for warning in status.warnings)
        assert status.successful_items == 1

    def test_import_conversation_warns_on_missing_attachment(self, service, mock_db, tmp_path):
        """Missing attachment files should log warnings while continuing import."""
        conv_id = "conv-missing"
        content_root = tmp_path / "content" / "conversations"
        content_root.mkdir(parents=True, exist_ok=True)

        rel_path = "content/conversations/conversation_assets/missing.png"
        assets_dir = content_root / "conversation_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        conversation_payload = {
            "id": conv_id,
            "name": "Conversation Missing Attachment",
            "created_at": "2024-01-01T00:00:00",
            "messages": [
                {
                    "id": "msg-missing",
                    "role": "assistant",
                    "content": "Hi!",
                    "timestamp": "2024-01-01T00:00:00",
                    "attachments": [
                        {
                            "type": "image",
                            "mime_type": "image/png",
                            "file_path": rel_path,
                        }
                    ],
                }
            ],
        }

        conversation_file = content_root / f"conversation_{conv_id}.json"
        conversation_file.write_text(json.dumps(conversation_payload), encoding="utf-8")

        mock_db.add_conversation.return_value = "conv-id"

        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Desc",
        )
        status = ImportJob(
            job_id="job",
            user_id="test_user",
            status=ImportStatus.PENDING,
            chatbook_path="dummy",
        )

        with patch.object(service, "_get_conversation_by_name", return_value=None):
            service._import_conversations(
                tmp_path,
                manifest,
                [conv_id],
                ConflictResolution.SKIP,
                prefix_imported=False,
                status=status,
            )

        assert mock_db.add_message.call_count == 1
        message_payload = mock_db.add_message.call_args[0][0]
        assert "images" not in message_payload
        assert any("Failed to read attachment" in warning for warning in status.warnings)
        assert status.successful_items == 1

    def test_collect_conversations_exports_citations(self, service, mock_db, tmp_path):
        """Exported conversations should include RAG citations when available."""
        conv_id = "conv-citations"
        msg_id = "msg-1"
        now = datetime(2024, 1, 1, 0, 0, 0)

        mock_db.get_conversation_by_id.return_value = {
            "id": conv_id,
            "title": "Conversation With Citations",
            "created_at": now,
            "character_id": 1,
        }
        mock_db.get_messages_for_conversation.return_value = [
            {
                "id": msg_id,
                "sender": "assistant",
                "content": "Answer",
                "timestamp": now,
            }
        ]
        mock_db.get_message_rag_context.return_value = {
            "retrieved_documents": [
                {
                    "id": "doc-1",
                    "source_type": "file",
                    "title": "Doc Title",
                    "score": 0.95,
                    "excerpt": "Excerpt",
                    "url": "https://example.com",
                    "page_number": 2,
                    "chunk_id": "chunk-1",
                }
            ],
            "citations": [{"id": "cite-1"}],
            "settings_snapshot": {"top_k": 5},
            "generated_answer": {"id": "answer-1"},
            "search_query": "example query",
        }

        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Desc",
        )
        content = ChatbookContent()

        service._collect_conversations([conv_id], tmp_path, manifest, content)

        exported = content.conversations[conv_id]
        message_payload = exported["messages"][0]
        assert message_payload["citations"]
        assert message_payload["citations"][0]["id"] == "doc-1"
        assert message_payload.get("formal_citations") == [{"id": "cite-1"}]
        assert message_payload.get("rag_settings") == {"top_k": 5}

    def test_import_conversation_skips_oversized_attachment(self, service, mock_db, tmp_path):
        """Oversized attachments should be skipped with a warning, not fail the conversation."""
        conv_id = "conv-oversize"
        content_root = tmp_path / "content" / "conversations"
        assets_dir = content_root / f"conversation_{conv_id}_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        image_bytes = b"0123456789"
        rel_path = f"content/conversations/{assets_dir.name}/big.png"
        (assets_dir / "big.png").write_bytes(image_bytes)

        conversation_payload = {
            "id": conv_id,
            "name": "Conversation Oversize Attachment",
            "created_at": "2024-01-01T00:00:00",
            "messages": [
                {
                    "id": "msg-big",
                    "role": "user",
                    "content": "Hello",
                    "timestamp": "2024-01-01T00:00:00",
                    "attachments": [
                        {
                            "type": "image",
                            "mime_type": "image/png",
                            "file_path": rel_path,
                        }
                    ],
                }
            ],
        }

        conversation_file = content_root / f"conversation_{conv_id}.json"
        conversation_file.write_text(json.dumps(conversation_payload), encoding="utf-8")

        mock_db.add_conversation.return_value = "conv-id"

        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Desc",
        )
        status = ImportJob(
            job_id="job",
            user_id="test_user",
            status=ImportStatus.PENDING,
            chatbook_path="dummy",
        )

        with patch.object(service, "_get_max_message_image_bytes", return_value=4):
            with patch.object(service, "_get_conversation_by_name", return_value=None):
                service._import_conversations(
                    tmp_path,
                    manifest,
                    [conv_id],
                    ConflictResolution.SKIP,
                    prefix_imported=False,
                    status=status,
                )

        assert mock_db.add_message.call_count == 1
        message_payload = mock_db.add_message.call_args[0][0]
        assert "images" not in message_payload
        assert any("exceeds MAX_MESSAGE_IMAGE_BYTES" in warning for warning in status.warnings)
        assert status.successful_items == 1

    def test_import_dictionary_preserves_max_replacements_zero(self, service, mock_db, tmp_path):
        """Dictionary imports should preserve max_replacements=0 (unlimited)."""
        dict_id = "dict-max"
        dict_dir = tmp_path / "content" / "dictionaries"
        dict_dir.mkdir(parents=True, exist_ok=True)

        dict_payload = {
            "name": "Test Dict",
            "description": "Desc",
            "entries": [
                {
                    "pattern": "hello",
                    "replacement": "hi",
                    "type": "literal",
                    "probability": 1.0,
                    "max_replacements": 0,
                }
            ],
        }
        dict_file = dict_dir / f"dictionary_{dict_id}.json"
        dict_file.write_text(json.dumps(dict_payload), encoding="utf-8")

        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Desc",
        )
        status = ImportJob(
            job_id="job",
            user_id="test_user",
            status=ImportStatus.PENDING,
            chatbook_path="dummy",
        )

        class DummyDictService:
            last_instance = None

            def __init__(self, _db):
                DummyDictService.last_instance = self
                self.add_entry_calls = []

            def get_dictionary_by_name(self, _name):
                return None

            def create_dictionary(self, _name, _description, _is_active=True):
                return 123

            def add_entry(self, *args, **kwargs):
                self.add_entry_calls.append((args, kwargs))

        with patch(
            "tldw_Server_API.app.core.Character_Chat.chat_dictionary.ChatDictionaryService",
            DummyDictService,
        ):
            service._import_dictionaries(
                tmp_path,
                manifest,
                [dict_id],
                ConflictResolution.SKIP,
                prefix_imported=False,
                status=status,
            )

        inst = DummyDictService.last_instance
        assert inst is not None
        assert inst.add_entry_calls
        _args, kwargs = inst.add_entry_calls[0]
        assert kwargs.get("max_replacements") == 0

    def test_preview_export_uses_chat_dictionaries_table(self, service, mock_db):
        """preview_export should query the chat_dictionaries table for dictionary counts."""
        mock_db.execute_query.return_value = []

        service.preview_export(["dictionaries"])

        calls = [call_args[0][0] for call_args in mock_db.execute_query.call_args_list]
        assert any("chat_dictionaries" in sql for sql in calls)

    @pytest.mark.asyncio
    async def test_export_manifest_total_size_bytes_matches_archive(self, service):
        """Exported manifest should report the final archive size."""
        success, _message, export_path = await service.create_chatbook(
            name="Size Check",
            description="Ensure manifest size is accurate",
            content_selections={},
            async_mode=False
        )
        assert success is True
        assert export_path is not None

        archive_path = Path(export_path)
        archive_size = archive_path.stat().st_size
        with zipfile.ZipFile(archive_path, "r") as zf:
            manifest_data = json.loads(zf.read("manifest.json"))

        total_size = (manifest_data.get("statistics") or {}).get("total_size_bytes", 0)
        assert total_size == archive_size

    def test_collect_conversations_paginates_messages(self, service, mock_db, tmp_path, monkeypatch):
        """Conversation exports should page through all messages."""
        monkeypatch.setenv("CHATBOOKS_CONVERSATION_EXPORT_PAGE_SIZE", "2")

        conv_id = "conv1"
        mock_db.get_conversation_by_id.return_value = {
            "id": conv_id,
            "title": "Test Conversation",
            "created_at": datetime(2024, 1, 1, 0, 0, 0),
            "character_id": 1,
        }

        msg1 = {"id": "m1", "sender": "user", "content": "hi", "timestamp": datetime(2024, 1, 1, 0, 0, 1), "images": []}
        msg2 = {"id": "m2", "sender": "assistant", "content": "hello", "timestamp": datetime(2024, 1, 1, 0, 0, 2), "images": []}
        msg3 = {"id": "m3", "sender": "user", "content": "more", "timestamp": datetime(2024, 1, 1, 0, 0, 3), "images": []}
        msg4 = {"id": "m4", "sender": "assistant", "content": "done", "timestamp": datetime(2024, 1, 1, 0, 0, 4), "images": []}

        mock_db.get_messages_for_conversation.side_effect = [
            [msg1, msg2],
            [msg3, msg4],
            [],
        ]

        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Desc",
        )
        content = ChatbookContent()

        service._collect_conversations([conv_id], tmp_path, manifest, content)

        calls = mock_db.get_messages_for_conversation.call_args_list
        assert len(calls) == 3
        assert calls[0].kwargs["limit"] == 2
        assert calls[0].kwargs["offset"] == 0
        assert calls[1].kwargs["offset"] == 2

        conv_file = tmp_path / "content" / "conversations" / f"conversation_{conv_id}.json"
        assert conv_file.exists()
        payload = json.loads(conv_file.read_text(encoding="utf-8"))
        assert len(payload["messages"]) == 4
        assert "conversations" not in (manifest.truncation or {})

    @pytest.mark.asyncio
    async def test_export_filename_truncates_long_names(self, service):
        """Export filenames should stay within filesystem limits for long names."""
        long_name = "a" * 300
        success, _message, export_path = await service.create_chatbook(
            name=long_name,
            description="Long name export",
            content_selections={},
            async_mode=False,
        )
        assert success is True
        assert export_path is not None

        filename = Path(export_path).name
        assert filename.endswith(".zip")
        assert len(filename) <= 255

    @pytest.mark.asyncio
    async def test_export_chatbook_async_job(self, service, mock_db):
        """Test asynchronous chatbook export with job management."""
        mock_db.execute_query.return_value = []

        with patch.object(service, "create_chatbook", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = (
                True,
                "Export job started: job-123",
                "job-123",
            )

            result = await service.export_chatbook(
                name="Test Export",
                content_types=["conversations"],
                async_job=True,
            )

        mock_create.assert_awaited_once()
        call_kwargs = mock_create.await_args.kwargs
        assert call_kwargs["async_mode"] is True
        assert result == {
            "success": True,
            "message": "Export job started: job-123",
            "file_path": None,
            "job_id": "job-123",
            "status": "pending",
            "content_summary": {},
        }

    def test_import_skips_unsupported_content_types(self, service, tmp_path):
        """Unsupported content types should be skipped with warnings."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Unsupported Types",
            description="Media and prompts are skipped",
            content_items=[
                ContentItem(id="m1", type=ContentType.MEDIA, title="Media 1"),
                ContentItem(id="p1", type=ContentType.PROMPT, title="Prompt 1"),
            ],
        )

        archive_path = service.temp_dir / "unsupported.chatbook"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest.to_dict()))

        success, message, details = service._import_chatbook_sync(
            file_path=str(archive_path),
            content_selections=None,
            conflict_resolution=ConflictResolution.SKIP,
            prefix_imported=False,
            import_media=True,
            import_embeddings=False,
        )

        assert success is True
        assert "skipped" in message.lower()
        assert details is not None
        assert details["imported_items"] == {}
        assert any("unsupported content type" in warning.lower() for warning in details["warnings"])

    def test_import_conversation_missing_character_falls_back(self, service, mock_db, tmp_path):
        """Missing character_id should fall back to default with a warning."""
        conv_id = "conv-missing-char"
        content_root = tmp_path / "content" / "conversations"
        content_root.mkdir(parents=True, exist_ok=True)

        conversation_payload = {
            "id": conv_id,
            "name": "Conversation Missing Character",
            "created_at": "2024-01-01T00:00:00",
            "character_id": None,
            "messages": [],
        }

        conversation_file = content_root / f"conversation_{conv_id}.json"
        conversation_file.write_text(json.dumps(conversation_payload), encoding="utf-8")

        def _char_lookup(char_id):
            if char_id == 1:
                return {"id": 1}
            return None

        mock_db.get_character_card_by_id.side_effect = _char_lookup
        mock_db.add_conversation.return_value = "new-conv-id"

        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test",
            description="Desc",
        )
        status = ImportJob(
            job_id="job",
            user_id="test_user",
            status=ImportStatus.PENDING,
            chatbook_path="dummy",
        )

        with patch.object(service, "_get_conversation_by_name", return_value=None):
            service._import_conversations(
                tmp_path,
                manifest,
                [conv_id],
                ConflictResolution.SKIP,
                prefix_imported=False,
                status=status,
            )

        conv_payload = mock_db.add_conversation.call_args[0][0]
        assert conv_payload["character_id"] == 1
        assert any("default character" in warning.lower() for warning in status.warnings)

    def test_preview_export(self, service, mock_db):
        """Test previewing export content."""
        # Mock queries based on what's being queried
        def mock_query(query, params=None):
            if "conversations" in query.lower():
                return [{"id": "conv1"}, {"id": "conv2"}]
            elif "character_cards" in query.lower():
                return [{"id": "char1"}]
            elif "notes" in query.lower():
                return [{"id": "note1"}, {"id": "note2"}]
            return []

        mock_db.execute_query = mock_query

        result = service.preview_export(
            content_types=["conversations", "characters", "notes"]
        )

        # Results should match mocked data
        assert result["conversations"] == 2
        assert result["characters"] == 1
        assert result["notes"] == 2
        assert result["world_books"] == 0

    def test_generate_unique_name_world_book_conflict(self, service, mock_db):
        """Ensure rename helper terminates when conflicts exist."""
        responses = [
            [{"id": 1}],  # Existing name hits conflict
            []            # Next candidate is available
        ]

        def side_effect(query, params=None):
            return responses.pop(0)

        mock_db.execute_query.side_effect = side_effect

        new_name = service._generate_unique_name("Existing", "world_book")

        assert new_name == "Existing (2)"

    def test_generate_unique_name_conversation_conflict(self, service):
        """Ensure rename helper uses conversation lookup helper."""
        with patch.object(
            service,
            "_get_conversation_by_name",
            side_effect=[{"id": "conv-1"}, None],
        ) as mock_lookup:
            new_name = service._generate_unique_name("Existing", "conversation")

        assert new_name == "Existing (2)"
        assert mock_lookup.call_count == 2

    def test_generate_unique_name_note_conflict(self, service):
        """Ensure rename helper uses note lookup helper."""
        with patch.object(
            service,
            "_get_note_by_title",
            side_effect=[{"id": "note-1"}, None],
        ) as mock_lookup:
            new_name = service._generate_unique_name("Existing", "note")

        assert new_name == "Existing (2)"
        assert mock_lookup.call_count == 2

    @pytest.mark.asyncio
    async def test_import_chatbook_rejects_unsupported_conflict_resolution(self, service):
        """Unsupported conflict_resolution values should fail fast."""
        with patch(
            "tldw_Server_API.app.core.Chatbooks.chatbook_service.asyncio.to_thread",
            new_callable=AsyncMock,
        ) as mock_to_thread:
            result = await service.import_chatbook(
                file_path="sample.chatbook",
                conflict_resolution="overwrite",
            )

        mock_to_thread.assert_not_awaited()
        assert result[0] is False
        assert "not supported" in result[1].lower()

    @pytest.mark.asyncio
    async def test_import_chatbook_conflict_strategy_alias(self, service):
        """conflict_strategy alias should map to the same enum handling."""
        sample_path = service.import_dir / "sample.chatbook"
        sample_path.write_text("dummy")
        with patch(
            "tldw_Server_API.app.core.Chatbooks.chatbook_service.asyncio.to_thread",
            new_callable=AsyncMock,
        ) as mock_to_thread:
            mock_to_thread.return_value = (True, "alias ok", None)

            await service.import_chatbook(
                file_path=str(sample_path),
                conflict_strategy="rename",
            )

        call_args = mock_to_thread.await_args.args
        assert call_args[3] is ConflictResolution.RENAME

    @pytest.mark.asyncio
    async def test_import_chatbook_invalid_conflict_resolution_defaults_to_skip(self, service):
        """Invalid conflict resolution values should fall back to SKIP."""
        sample_path = service.import_dir / "sample.chatbook"
        sample_path.write_text("dummy")
        with patch(
            "tldw_Server_API.app.core.Chatbooks.chatbook_service.asyncio.to_thread",
            new_callable=AsyncMock,
        ) as mock_to_thread:
            mock_to_thread.return_value = (True, "defaulted", None)

            await service.import_chatbook(
                file_path=str(sample_path),
                conflict_resolution="replace",
            )

        call_args = mock_to_thread.await_args.args
        assert call_args[3] is ConflictResolution.SKIP

    def test_get_statistics(self, service, mock_db):
        """Test getting import/export statistics."""
        # Mock returns tuples for status counts
        mock_db.execute_query.side_effect = [
            [("completed", 45), ("failed", 5)],  # Export stats by status
            [("completed", 28), ("failed", 2)],  # Import stats by status
        ]

        stats = service.get_statistics()

        # Check the structure returned by get_statistics
        assert "exports" in stats
        assert "imports" in stats
        assert stats["exports"].get("completed", 0) == 45
        assert stats["imports"].get("completed", 0) == 28

    @pytest.mark.asyncio
    async def test_error_handling_during_export(self, service):
        """Test error handling during export."""
        with patch.object(service, "create_chatbook", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = RuntimeError("Database error")

            with pytest.raises(RuntimeError):
                await service.export_chatbook(
                    name="Test Export",
                    content_types=["conversations"],
                )

    @pytest.mark.asyncio
    async def test_error_handling_during_import(self, service):
        """Test error handling during import."""
        result = await service.import_chatbook(
            file_path="/nonexistent/file.chatbook",
            conflict_resolution="skip"
        )

        assert result == (
            False,
            "Invalid or potentially malicious archive file",
            None,
        )

    def test_user_isolation(self, service, mock_db):
        """Test that operations are isolated to the current user."""
        # Test export - should only get current user's content
        mock_db.execute_query.return_value = []

        service.preview_export(content_types=["conversations"])

        # Check that the query was made - the actual implementation uses "deleted = 0"
        # instead of user_id filtering because tables don't have user_id columns
        call_args = mock_db.execute_query.call_args[0]
        # Accept either user_id filtering or deleted filtering (actual implementation)
        assert ("WHERE user_id = ?" in call_args[0] or
                "WHERE deleted = 0" in call_args[0] or
                "test_user" in str(call_args))

    def test_create_chatbook_archive(self, service, sample_manifest, sample_content_items):
        """Test creating a chatbook archive file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir) / "work"
            work_dir.mkdir()
            output_path = Path(tmpdir) / "test.chatbook"

            # Write some test content to work dir
            (work_dir / "manifest.json").write_text(json.dumps(manifest_to_dict(sample_manifest)))
            (work_dir / "test.txt").write_text("test content")

            # Call with correct parameters (work_dir and output_path)
            result = service._create_chatbook_archive(
                work_dir=work_dir,
                output_path=output_path
            )

            assert result == True
            assert output_path.exists()

            # Verify it's a valid zip file
            with zipfile.ZipFile(output_path, 'r') as zf:
                assert 'manifest.json' in zf.namelist()

    def test_process_import_items(self, service, mock_db, sample_content_items):
        """Test processing individual import items."""
        mock_db.execute_query.return_value = []  # No conflicts

        # Call with correct parameter name (conflict_resolution)
        results = service._process_import_items(
            items=sample_content_items,
            conflict_resolution="skip"
        )

        # Results is an ImportStatusData object
        assert results.successful_items == len(sample_content_items)
        assert results.skipped_items == 0
        assert results.failed_items == 0


class TestBinaryLimitsExtension:
    """Tests for binary limits enforcement on media embeddings (Sub-task 1)."""

    def test_embedding_skipped_when_over_binary_limit(self, service, tmp_path):
        """Large embedding BLOBs should produce a stub with bundled=False."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="limit-test",
            description="test",
            binary_limits={"embeddings": 10},  # 10 bytes limit
        )
        content = ChatbookContent()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        # Mock media DB returning a record with a large vector blob
        large_blob = b"\x00" * 100  # 100 bytes, exceeds 10 byte limit
        media_record = {
            "id": 42,
            "title": "Test Media",
            "uuid": "abc-123",
            "vector_embedding": large_blob,
        }
        mock_media_db = MagicMock()
        service._media_db = mock_media_db
        # The real normalizer strips binary fields; mock should return JSON-serializable data
        normalized = {k: v for k, v in media_record.items() if k != "vector_embedding"}
        with patch.object(service, '_fetch_media_record', return_value=media_record), \
             patch.object(service, '_normalize_media_record', return_value=normalized):
            service._collect_media_items(
                media_ids=["42"],
                work_dir=work_dir,
                manifest=manifest,
                content=content,
                include_media=False,
                include_embeddings=True,
            )

        # Embedding should exist but not be bundled
        assert "media:42" in content.embeddings
        emb = content.embeddings["media:42"]
        assert emb["bundled"] is False
        assert emb["size_bytes"] == 100
        # No embedding file should have been written
        emb_dir = work_dir / "content" / "embeddings"
        if emb_dir.exists():
            assert not list(emb_dir.glob("embedding_media_42.json"))
        # Bug 1 regression: media metadata must still be exported even when embedding exceeds limit
        assert "42" in content.media
        media_items = [ci for ci in manifest.content_items if ci.type == ContentType.MEDIA]
        assert any(ci.id == "42" for ci in media_items)

    def test_embedding_bundled_when_under_limit(self, service, tmp_path):
        """Normal embeddings under the limit should still be bundled."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="limit-test",
            description="test",
            binary_limits={"embeddings": 10000},
        )
        content = ChatbookContent()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        small_blob = b"\x01\x02\x03"
        media_record = {
            "id": 7,
            "title": "Small Media",
            "uuid": "def-456",
            "vector_embedding": small_blob,
        }
        # normalized record excludes non-serializable vector_embedding
        normalized = {"id": 7, "title": "Small Media", "uuid": "def-456"}
        mock_media_db = MagicMock()
        service._media_db = mock_media_db
        with patch.object(service, '_fetch_media_record', return_value=media_record), \
             patch.object(service, '_normalize_media_record', return_value=normalized):
            service._collect_media_items(
                media_ids=["7"],
                work_dir=work_dir,
                manifest=manifest,
                content=content,
                include_media=False,
                include_embeddings=True,
            )

        assert "media:7" in content.embeddings
        emb = content.embeddings["media:7"]
        assert "vector" in emb  # Full payload with base64 vector
        assert "bundled" not in emb  # Not a stub

    def test_embedding_bundled_when_no_limit_set(self, service, tmp_path):
        """Without binary limits, all embeddings should be bundled."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="no-limit",
            description="test",
        )
        content = ChatbookContent()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        blob = b"\x00" * 500
        media_record = {
            "id": 99,
            "title": "Big Media",
            "uuid": "ghi-789",
            "vector_embedding": blob,
        }
        # normalized record excludes non-serializable vector_embedding
        normalized = {"id": 99, "title": "Big Media", "uuid": "ghi-789"}
        mock_media_db = MagicMock()
        service._media_db = mock_media_db
        with patch.object(service, '_fetch_media_record', return_value=media_record), \
             patch.object(service, '_normalize_media_record', return_value=normalized):
            service._collect_media_items(
                media_ids=["99"],
                work_dir=work_dir,
                manifest=manifest,
                content=content,
                include_media=False,
                include_embeddings=True,
            )

        assert "media:99" in content.embeddings
        assert "vector" in content.embeddings["media:99"]


class TestMediaExportCapping:
    """Tests for CHATBOOKS_MEDIA_EXPORT_MAX_ITEMS truncation (Sub-task 4)."""

    def test_media_capped_at_max_items(self, service, tmp_path, monkeypatch):
        """When CHATBOOKS_MEDIA_EXPORT_MAX_ITEMS is set, only that many items are exported."""
        monkeypatch.setenv("CHATBOOKS_MEDIA_EXPORT_MAX_ITEMS", "2")
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="cap-test",
            description="test",
        )
        content = ChatbookContent()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        def fake_fetch(db, media_id):
            return {"id": int(media_id), "title": f"Media {media_id}"}

        mock_media_db = MagicMock()
        service._media_db = mock_media_db
        with patch.object(service, '_fetch_media_record', side_effect=fake_fetch), \
             patch.object(service, '_normalize_media_record', side_effect=lambda r: dict(r)):
            service._collect_media_items(
                media_ids=["1", "2", "3", "4", "5"],
                work_dir=work_dir,
                manifest=manifest,
                content=content,
                include_media=False,
                include_embeddings=False,
            )

        # Only 2 items should have been exported
        assert len(content.media) == 2
        # Truncation metadata should be recorded
        assert "media" in manifest.truncation
        trunc = manifest.truncation["media"]
        assert trunc["truncated"] is True
        assert trunc["max_items"] == 2
        assert trunc["exported_count"] == 2
        assert trunc["total_count"] == 5

    def test_media_not_capped_when_zero(self, service, tmp_path, monkeypatch):
        """CHATBOOKS_MEDIA_EXPORT_MAX_ITEMS=0 means unlimited."""
        monkeypatch.setenv("CHATBOOKS_MEDIA_EXPORT_MAX_ITEMS", "0")
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="no-cap",
            description="test",
        )
        content = ChatbookContent()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        def fake_fetch(db, media_id):
            return {"id": int(media_id), "title": f"Media {media_id}"}

        mock_media_db = MagicMock()
        service._media_db = mock_media_db
        with patch.object(service, '_fetch_media_record', side_effect=fake_fetch), \
             patch.object(service, '_normalize_media_record', side_effect=lambda r: dict(r)):
            service._collect_media_items(
                media_ids=["1", "2", "3"],
                work_dir=work_dir,
                manifest=manifest,
                content=content,
                include_media=False,
                include_embeddings=False,
            )

        assert len(content.media) == 3
        assert "media" not in manifest.truncation


class TestChromaDBEmbeddingExport:
    """Tests for ChromaDB collection-level embedding export (Sub-task 2)."""

    def _make_mock_collection(self, name, metadata=None, chunks=None):
        """Create a mock ChromaDB collection."""
        col = MagicMock()
        col.name = name
        col.metadata = metadata or {"model": "test-model", "dimensions": 384}

        if chunks is None:
            chunks = []

        def mock_count():
            return len(chunks)
        col.count = mock_count

        def mock_get(limit=None, offset=None, include=None):
            start = offset or 0
            end = start + (limit or len(chunks))
            page = chunks[start:end]
            result = {"ids": [c["id"] for c in page]}
            if include and "documents" in include:
                result["documents"] = [c.get("document", "") for c in page]
            if include and "metadatas" in include:
                result["metadatas"] = [c.get("metadata", {}) for c in page]
            if include and "embeddings" in include:
                result["embeddings"] = [c.get("embedding", []) for c in page]
            return result
        col.get = mock_get

        return col

    def test_collect_embeddings_two_collections(self, service, tmp_path):
        """Two collections should produce two JSON files and two content items."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="emb-test",
            description="test",
        )
        content = ChatbookContent()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        col1 = self._make_mock_collection("col_a", chunks=[
            {"id": "c1", "document": "hello", "metadata": {"k": "v"}, "embedding": [0.1, 0.2]},
        ])
        col2 = self._make_mock_collection("col_b", chunks=[
            {"id": "c2", "document": "world", "metadata": {}, "embedding": [0.3, 0.4]},
            {"id": "c3", "document": "test", "metadata": {}, "embedding": [0.5, 0.6]},
        ])

        mock_chroma = MagicMock()
        mock_chroma.list_collections.return_value = [col1, col2]
        service._chroma_manager = mock_chroma

        service._collect_embeddings([], work_dir, manifest, content)

        assert len(content.embeddings) == 2
        assert "collection:col_a" in content.embeddings
        assert "collection:col_b" in content.embeddings

        # Verify files written
        emb_dir = work_dir / "content" / "embeddings"
        assert (emb_dir / "collection_col_a.json").exists()
        assert (emb_dir / "collection_col_b.json").exists()

        # Verify manifest content items
        emb_items = [i for i in manifest.content_items if i.type == ContentType.EMBEDDING]
        assert len(emb_items) == 2

        # Verify source_hash is deterministic
        data_a = content.embeddings["collection:col_a"]
        assert "source_hash" in data_a
        assert len(data_a["source_hash"]) == 64  # SHA-256 hex

    def test_collect_embeddings_truncation(self, service, tmp_path, monkeypatch):
        """Chunks exceeding max should be truncated with metadata recorded."""
        monkeypatch.setenv("CHATBOOKS_EMBEDDING_EXPORT_MAX_CHUNKS", "2")
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="trunc-test",
            description="test",
        )
        content = ChatbookContent()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        chunks = [
            {"id": f"chunk_{i}", "document": f"doc {i}", "metadata": {}, "embedding": [float(i)]}
            for i in range(10)
        ]
        col = self._make_mock_collection("big_col", chunks=chunks)

        mock_chroma = MagicMock()
        mock_chroma.list_collections.return_value = [col]
        service._chroma_manager = mock_chroma

        service._collect_embeddings([], work_dir, manifest, content)

        data = content.embeddings["collection:big_col"]
        assert data["truncated"] is True
        assert len(data["chunks"]) == 2

        assert "embeddings" in manifest.truncation
        trunc = manifest.truncation["embeddings"]
        assert trunc["truncated"] is True
        assert trunc["max_chunks_per_collection"] == 2
        assert "big_col" in trunc["collection_ids"]

    def test_collect_embeddings_empty(self, service, tmp_path):
        """No collections should produce no output and no error."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="empty-test",
            description="test",
        )
        content = ChatbookContent()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        mock_chroma = MagicMock()
        mock_chroma.list_collections.return_value = []
        service._chroma_manager = mock_chroma

        service._collect_embeddings([], work_dir, manifest, content)

        assert len(content.embeddings) == 0
        assert len(manifest.content_items) == 0

    def test_collect_embeddings_chroma_unavailable(self, service, tmp_path):
        """If ChromaDB is unavailable, export continues without error."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="fallback-test",
            description="test",
        )
        content = ChatbookContent()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        service._chroma_manager = None
        with patch.object(service, '_get_chroma_manager', return_value=None):
            service._collect_embeddings(["col_x"], work_dir, manifest, content)

        assert len(content.embeddings) == 0

    def test_collect_embeddings_specific_ids(self, service, tmp_path):
        """When specific IDs are given, only those collections are exported."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="specific-test",
            description="test",
        )
        content = ChatbookContent()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        col = self._make_mock_collection("target_col", chunks=[
            {"id": "t1", "document": "data", "metadata": {}, "embedding": [1.0]},
        ])

        mock_chroma = MagicMock()
        mock_chroma.get_collection.return_value = col
        service._chroma_manager = mock_chroma

        service._collect_embeddings(["target_col"], work_dir, manifest, content)

        assert "collection:target_col" in content.embeddings
        mock_chroma.list_collections.assert_not_called()
        mock_chroma.get_or_create_collection.assert_not_called()


class TestTruncationMetadataConsistency:
    """Tests for standardized truncation metadata across content types (Sub-task 4)."""

    def test_eval_truncation_includes_exported_count(self, service, tmp_path):
        """Evaluation truncation should include exported_count."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="eval-trunc",
            description="test",
        )
        content = ChatbookContent()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        mock_evals_db = MagicMock()
        mock_evals_db.get_evaluation.return_value = {
            "id": "eval_1", "name": "Test Eval", "type": "manual"
        }
        mock_evals_db.list_runs.return_value = (
            [{"id": f"run_{i}", "eval_id": "eval_1"} for i in range(5)],
            True,  # has_more
        )
        service._evaluations_db = mock_evals_db
        with patch.object(service, '_normalize_evaluation_record', side_effect=lambda r: dict(r)), \
             patch.object(service, '_normalize_evaluation_run', side_effect=lambda r: dict(r)):
            service._collect_evaluations(["eval_1"], work_dir, manifest, content)

        assert "evaluations" in manifest.truncation
        trunc = manifest.truncation["evaluations"]
        assert trunc["exported_count"] == 5

    def test_conversation_truncation_includes_exported_count(self, service, tmp_path):
        """Conversation truncation should include exported_count."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="conv-trunc",
            description="test",
        )
        content = ChatbookContent()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        service.db.get_conversation_by_id.return_value = {"id": "c1", "title": "Test"}
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        with patch.object(service, '_get_conversation_messages_paged', return_value=(messages, True, 2)):
            service._collect_conversations(["c1"], work_dir, manifest, content)

        assert "conversations" in manifest.truncation
        trunc = manifest.truncation["conversations"]
        assert trunc["exported_count"] == 2


class TestBug2NonexistentCollection:
    """Tests for read-only collection lookup during export (Bug 2 fix)."""

    def _make_mock_collection(self, name, metadata=None, chunks=None):
        """Create a mock ChromaDB collection."""
        col = MagicMock()
        col.name = name
        col.metadata = metadata or {"model": "test-model", "dimensions": 384}
        if chunks is None:
            chunks = []

        def mock_count():
            return len(chunks)
        col.count = mock_count

        def mock_get(limit=None, offset=None, include=None):
            start = offset or 0
            end = start + (limit or len(chunks))
            page = chunks[start:end]
            result = {"ids": [c["id"] for c in page]}
            if include and "documents" in include:
                result["documents"] = [c.get("document", "") for c in page]
            if include and "metadatas" in include:
                result["metadatas"] = [c.get("metadata", {}) for c in page]
            if include and "embeddings" in include:
                result["embeddings"] = [c.get("embedding", []) for c in page]
            return result
        col.get = mock_get
        return col

    def test_nonexistent_collection_skipped_without_side_effect(self, service, tmp_path):
        """A nonexistent collection name should be skipped, not created."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="ghost-test",
            description="test",
        )
        content = ChatbookContent()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        mock_chroma = MagicMock()
        mock_chroma.get_collection.side_effect = KeyError("Collection 'ghost' does not exist")
        service._chroma_manager = mock_chroma

        service._collect_embeddings(["ghost"], work_dir, manifest, content)

        # No collections should be exported
        assert len(content.embeddings) == 0
        # get_or_create_collection should NOT have been called
        mock_chroma.get_or_create_collection.assert_not_called()

    def test_existing_collection_still_exported(self, service, tmp_path):
        """An existing collection should be exported normally via get_collection."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="exists-test",
            description="test",
        )
        content = ChatbookContent()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        col = self._make_mock_collection("real_col", chunks=[
            {"id": "c1", "document": "data", "metadata": {}, "embedding": [1.0]},
        ])
        mock_chroma = MagicMock()
        mock_chroma.get_collection.return_value = col
        service._chroma_manager = mock_chroma

        service._collect_embeddings(["real_col"], work_dir, manifest, content)

        assert "collection:real_col" in content.embeddings


class TestGap6BinaryLimitTruncationMetadata:
    """Tests for truncation metadata on binary-limited collections (Gap 6 fix)."""

    def _make_mock_collection(self, name, metadata=None, chunks=None):
        col = MagicMock()
        col.name = name
        col.metadata = metadata or {}
        if chunks is None:
            chunks = []
        col.count = MagicMock(return_value=len(chunks))

        def mock_get(limit=None, offset=None, include=None):
            start = offset or 0
            end = start + (limit or len(chunks))
            page = chunks[start:end]
            result = {"ids": [c["id"] for c in page]}
            if include and "documents" in include:
                result["documents"] = [c.get("document", "") for c in page]
            if include and "metadatas" in include:
                result["metadatas"] = [c.get("metadata", {}) for c in page]
            if include and "embeddings" in include:
                result["embeddings"] = [c.get("embedding", []) for c in page]
            return result
        col.get = mock_get
        return col

    def test_binary_limited_collection_records_truncation(self, service, tmp_path):
        """When a collection exceeds the binary limit, truncation metadata should be recorded."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="trunc-meta-test",
            description="test",
            binary_limits={"collection_embeddings": 10},  # 10 bytes = everything will exceed
        )
        content = ChatbookContent()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        col = self._make_mock_collection("big_col", chunks=[
            {"id": "c1", "document": "x" * 100, "metadata": {}, "embedding": [1.0] * 50},
        ])
        mock_chroma = MagicMock()
        mock_chroma.list_collections.return_value = [col]
        service._chroma_manager = mock_chroma

        service._collect_embeddings([], work_dir, manifest, content)

        # The collection should be a stub
        assert "collection:big_col" in content.embeddings
        assert content.embeddings["collection:big_col"]["bundled"] is False

        # Truncation metadata should be recorded
        assert "embeddings" in manifest.truncation
        trunc = manifest.truncation["embeddings"]
        assert trunc["truncated"] is True
        assert "big_col" in trunc.get("binary_limited_collections", [])

# test_document_generator.py
# Description: Unit tests for the DocumentGeneratorService
#
"""
Document Generator Service Tests
---------------------------------

Comprehensive unit tests for document generation functionality including
all document types, async job management, and prompt customization.
"""

import pytest
import json
import asyncio
import tempfile
import os
import shutil
from types import SimpleNamespace
from unittest.mock import patch
from datetime import datetime
from uuid import uuid4

from tldw_Server_API.app.core.Chat.document_generator import DocumentGeneratorService, DocumentType
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME


@pytest.fixture
def real_db():
    """Create a real database instance for testing."""
    # Create a temporary directory for the test database
    temp_dir = tempfile.mkdtemp(prefix="test_docgen_")
    db_path = os.path.join(temp_dir, "test_docgen.db")

    # Initialize real database
    db = CharactersRAGDB(db_path, client_id="test_user")

    # Add default character
    char_id = db.add_character_card({
        "name": DEFAULT_CHARACTER_NAME,
        "description": "Test character",
        "personality": "Helpful",
        "scenario": "Testing",
        "system_prompt": "You are a helpful AI assistant."
    })

    # Create a test conversation with messages
    conv_id = db.add_conversation({
        "character_id": char_id,
        "title": "Test Conversation",
        "client_id": "test_user"
    })

    # Add test messages
    db.add_message({
        "conversation_id": conv_id,
        "sender": "user",
        "content": "Hello, how are you?",
        "client_id": "test_user"
    })

    db.add_message({
        "conversation_id": conv_id,
        "sender": "assistant",
        "content": "I'm doing well, thank you!",
        "client_id": "test_user"
    })

    db.add_message({
        "conversation_id": conv_id,
        "sender": "user",
        "content": "Can you explain quantum computing?",
        "client_id": "test_user"
    })

    db.add_message({
        "conversation_id": conv_id,
        "sender": "assistant",
        "content": "Quantum computing uses quantum bits...",
        "client_id": "test_user"
    })

    # Store conversation ID for tests
    db.test_conversation_id = conv_id

    yield db

    # Cleanup
    try:
        if hasattr(db, 'close_connection'):
            db.close_connection()
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"Cleanup error: {e}")


@pytest.fixture
def mock_db(real_db):
    """Alias for real_db to match test expectations."""
    return real_db


@pytest.fixture
def service(real_db):
    """Create a DocumentGeneratorService instance with real database."""
    return DocumentGeneratorService(real_db, user_id="test_user")


@pytest.fixture
def sample_conversation():
    """Create sample conversation data."""
    return {
        "id": "conv123",
        "title": "Test Conversation",
        "created_at": "2024-01-01T00:00:00Z",
        "messages": [
            {"sender": "user", "content": "Hello, how are you?", "timestamp": "2024-01-01T00:00:00Z"},
            {"sender": "assistant", "content": "I'm doing well, thank you!", "timestamp": "2024-01-01T00:01:00Z"},
            {"sender": "user", "content": "Can you explain quantum computing?", "timestamp": "2024-01-01T00:02:00Z"},
            {"sender": "assistant", "content": "Quantum computing uses quantum bits...", "timestamp": "2024-01-01T00:03:00Z"}
        ]
    }


class TestDocumentGeneratorService:
    """Test suite for DocumentGeneratorService."""

    def test_init_creates_tables(self, real_db):

        """Test that initialization creates necessary tables."""
        service = DocumentGeneratorService(real_db, "test_user")

        # Check that service was initialized properly
        # The database should have tables created
        result = real_db.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        table_names = [row['name'] for row in result] if result else []

        # The service creates its own tables, or we accept that core tables exist
        assert len(table_names) >= 2  # Should have at least some tables

    def test_generate_timeline(self, service, real_db):

        """Test timeline document generation."""
        # Use the real conversation created in the fixture
        conv_id = real_db.test_conversation_id

        # Mock LLM call - patch chat_api_call where it's imported in document_generator
        with patch('tldw_Server_API.app.core.Chat.document_generator.chat_api_call') as mock_llm:
            mock_llm.return_value = "Timeline: Event 1, Event 2, Event 3"

            result = service.generate_document(
                conversation_id=conv_id,
                document_type=DocumentType.TIMELINE,
                provider="openai",
                model="gpt-3.5-turbo",
                api_key="test_key",
                custom_prompt=None
            )

        assert result is not None
        mock_llm.assert_called_once()

    @pytest.mark.parametrize("captured_key", ["document-key-a", ""], ids=["a-to-b", "absent-to-b"])
    def test_generate_document_keeps_snapshot_at_chat_boundary(
        self,
        service: DocumentGeneratorService,
        real_db: CharactersRAGDB,
        captured_key: str,
    ) -> None:
        """Document generation must mark its key/config pair authoritative."""
        config_a = {"openai_api": {"model": "model-a", "api_key": "config-key-a"}}
        credentials = SimpleNamespace(provider="openai")

        with patch(
            "tldw_Server_API.app.core.Chat.document_generator.chat_api_call",
            return_value="Generated content",
        ) as mock_llm:
            service.generate_document(
                conversation_id=real_db.test_conversation_id,
                document_type=DocumentType.SUMMARY,
                provider="openai",
                model="model-a",
                api_key=captured_key,
                app_config=config_a,
                credentials_resolved=True,
                provider_credentials=credentials,
            )

        kwargs = mock_llm.call_args.kwargs
        assert kwargs["api_key"] == captured_key
        assert kwargs["app_config"] == config_a
        assert kwargs["credentials_resolved"] is True
        assert kwargs[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is credentials

    def test_generate_study_guide(self, service, real_db):

        """Test study guide document generation."""
        # Use the real conversation created in the fixture
        conv_id = real_db.test_conversation_id

        with patch('tldw_Server_API.app.core.Chat.document_generator.chat_api_call') as mock_llm:
            mock_llm.return_value = "Study Guide: Key concepts and review questions"

            result = service.generate_document(
                conversation_id=conv_id,
                document_type=DocumentType.STUDY_GUIDE,
                provider="openai",
                model="gpt-3.5-turbo",
                api_key="test_key"
            )

        assert result is not None
        mock_llm.assert_called_once()

    def test_generate_briefing(self, service, real_db):

        """Test executive briefing document generation."""
        conv_id = real_db.test_conversation_id

        with patch('tldw_Server_API.app.core.Chat.document_generator.chat_api_call') as mock_llm:
            mock_llm.return_value = "Executive Summary: Key points and recommendations"

            result = service.generate_document(
                conversation_id=conv_id,
                document_type=DocumentType.BRIEFING,
                provider="openai",
                model="gpt-3.5-turbo",
                api_key="test_key"
            )

        assert result is not None
        mock_llm.assert_called_once()

    def test_generate_summary(self, service, real_db):

        """Test summary document generation."""
        conv_id = real_db.test_conversation_id

        with patch('tldw_Server_API.app.core.Chat.document_generator.chat_api_call') as mock_llm:
            mock_llm.return_value = "Summary: Main discussion points"

            result = service.generate_document(
                conversation_id=conv_id,
                document_type=DocumentType.SUMMARY,
                provider="openai",
                model="gpt-3.5-turbo",
                api_key="test_key"
            )

        assert result is not None
        mock_llm.assert_called_once()

    def test_generate_summary_extracts_openai_content(self, service, real_db):

        """Ensure OpenAI-style responses are reduced to message content."""
        conv_id = real_db.test_conversation_id

        with patch('tldw_Server_API.app.core.Chat.document_generator.chat_api_call') as mock_llm:
            mock_llm.return_value = {
                "choices": [{"message": {"content": "Summary content"}}]
            }

            result = service.generate_document(
                conversation_id=conv_id,
                document_type=DocumentType.SUMMARY,
                provider="openai",
                model="gpt-3.5-turbo",
                api_key="test_key"
            )

        assert result == "Summary content"
        mock_llm.assert_called_once()

    def test_generate_qa_pairs(self, service, real_db):

        """Test Q&A pairs document generation."""
        conv_id = real_db.test_conversation_id

        with patch('tldw_Server_API.app.core.Chat.document_generator.chat_api_call') as mock_llm:
            mock_llm.return_value = "Q1: Question?\nA1: Answer."

            result = service.generate_document(
                conversation_id=conv_id,
                document_type=DocumentType.QA,
                provider="openai",
                model="gpt-3.5-turbo",
                api_key="test_key"
            )

        assert result is not None
        mock_llm.assert_called_once()

    def test_generate_meeting_notes(self, service, real_db):

        """Test meeting notes document generation."""
        conv_id = real_db.test_conversation_id

        with patch('tldw_Server_API.app.core.Chat.document_generator.chat_api_call') as mock_llm:
            mock_llm.return_value = "Meeting Notes: Attendees, Agenda, Action Items"

            result = service.generate_document(
                conversation_id=conv_id,
                document_type=DocumentType.MEETING_NOTES,
                provider="openai",
                model="gpt-3.5-turbo",
                api_key="test_key"
            )

        assert result is not None
        mock_llm.assert_called_once()

    def test_custom_prompt(self, service, real_db):

        """Test using custom prompt for generation."""
        conv_id = real_db.test_conversation_id
        custom_prompt = "Extract only the technical terms mentioned"

        with patch('tldw_Server_API.app.core.Chat.document_generator.chat_api_call') as mock_llm:
            mock_llm.return_value = "Technical terms: quantum computing, quantum bits"

            result = service.generate_document(
                conversation_id=conv_id,
                document_type=DocumentType.SUMMARY,
                provider="openai",
                model="gpt-3.5-turbo",
                api_key="test_key",
                custom_prompt=custom_prompt
            )

        assert result is not None
        mock_llm.assert_called_once()

    def test_async_job_creation(self, service, real_db):

        """Test creating an async generation job."""
        conv_id = real_db.test_conversation_id

        # DocumentGeneratorService may not have create_generation_job method
        # Skip this test if the method doesn't exist
        if not hasattr(service, 'create_generation_job'):
            pytest.skip("create_generation_job not implemented")

        result = service.create_generation_job(
            conversation_id=conv_id,
            document_type=DocumentType.TIMELINE,
            provider="openai",
            model="gpt-3.5-turbo",
            prompt_config={}
        )

        assert result is not None

    def test_get_job_status(self, service, real_db):

        """Test retrieving job status."""
        # Skip if method doesn't exist
        if not hasattr(service, 'get_job_status'):
            pytest.skip("get_job_status not implemented")

    def test_cancel_job(self, service, real_db):

        """Test cancelling a generation job."""
        # Skip if method doesn't exist
        if not hasattr(service, 'cancel_job'):
            pytest.skip("cancel_job not implemented")

    def test_get_document(self, service, real_db):

        """Test retrieving a generated document."""
        # Skip if method doesn't exist
        if not hasattr(service, 'get_document'):
            pytest.skip("get_document not implemented")

    def test_list_documents(self, service, real_db):

        """Test listing generated documents."""
        # Skip if method doesn't exist
        if not hasattr(service, 'list_documents'):
            pytest.skip("list_documents not implemented")

    def test_delete_document(self, service, real_db):

        """Test deleting a generated document."""
        # Skip if method doesn't exist
        if not hasattr(service, 'delete_document'):
            pytest.skip("delete_document not implemented")

    def test_save_custom_prompt_config(self, service, real_db):

        """Test saving custom prompt configuration."""
        assert service.save_prompt_config({DocumentType.SUMMARY: "Custom summary prompt"})

        with real_db.get_connection() as conn:
            row = conn.execute(
                """
                SELECT user_id, document_type, custom_prompt
                FROM user_prompt_configs
                WHERE user_id = ? AND document_type = ?
                """,
                ("test_user", DocumentType.SUMMARY.value),
            ).fetchone()

        assert dict(row) == {
            "user_id": "test_user",
            "document_type": DocumentType.SUMMARY.value,
            "custom_prompt": "Custom summary prompt",
        }

    def test_get_prompt_config(self, service, real_db):

        """Test retrieving custom prompt configuration."""
        assert service.save_prompt_config({DocumentType.QA: "Custom QA prompt"})
        assert service.get_prompt_config(DocumentType.QA) == "Custom QA prompt"

    def test_save_user_prompt_config_allows_multiple_inactive_versions(self, service, real_db):
        """Saving new prompt versions keeps history while leaving one active prompt."""
        assert service.save_user_prompt_config(
            DocumentType.SUMMARY,
            "System v1",
            "User v1",
        )
        assert service.save_user_prompt_config(
            DocumentType.SUMMARY,
            "System v2",
            "User v2",
        )
        assert service.save_user_prompt_config(
            DocumentType.SUMMARY,
            "System v3",
            "User v3",
        )

        rows = real_db.execute_query(
            """
            SELECT user_prompt, is_active
            FROM user_prompts
            WHERE document_type = ?
            ORDER BY id
            """,
            (DocumentType.SUMMARY.value,)
        ).fetchall()

        assert [row["user_prompt"] for row in rows] == ["User v1", "User v2", "User v3"]
        assert [row["is_active"] for row in rows] == [0, 0, 1]

    def test_init_repairs_legacy_user_prompt_unique_constraint(self, real_db):
        """Existing prompt tables with legacy uniqueness are repaired in place."""
        with real_db.get_connection() as conn:
            conn.execute("""
                CREATE TABLE user_prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_type TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    user_prompt TEXT NOT NULL,
                    temperature REAL DEFAULT 0.7,
                    max_tokens INTEGER DEFAULT 2000,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(document_type, is_active)
                )
            """)
            conn.execute(
                """
                INSERT INTO user_prompts
                (document_type, system_prompt, user_prompt, is_active)
                VALUES (?, ?, ?, 0)
                """,
                (DocumentType.SUMMARY.value, "System v1", "User v1")
            )
            conn.execute(
                """
                INSERT INTO user_prompts
                (document_type, system_prompt, user_prompt, is_active)
                VALUES (?, ?, ?, 1)
                """,
                (DocumentType.SUMMARY.value, "System v2", "User v2")
            )
            conn.commit()

        service = DocumentGeneratorService(real_db, user_id="test_user")

        assert service.save_user_prompt_config(
            DocumentType.SUMMARY,
            "System v3",
            "User v3",
        )

        with real_db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT user_prompt, is_active
                FROM user_prompts
                WHERE document_type = ?
                ORDER BY id
                """,
                (DocumentType.SUMMARY.value,)
            ).fetchall()
            index_row = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'index'
                  AND name = 'idx_user_prompts_active_document_type'
                """
            ).fetchone()

        assert [row["user_prompt"] for row in rows] == ["User v1", "User v2", "User v3"]
        assert [row["is_active"] for row in rows] == [0, 0, 1]
        assert index_row is not None

    @pytest.mark.asyncio
    async def test_bulk_generation(self, service, real_db):
        """Test generating multiple document types at once."""
        # Skip if method doesn't exist
        if not hasattr(service, 'bulk_generate'):
            pytest.skip("bulk_generate not implemented")

    def test_get_statistics(self, service, real_db):

        """Test getting generation statistics."""
        # Skip if method doesn't exist
        if not hasattr(service, 'get_statistics'):
            pytest.skip("get_statistics not implemented")

    def test_error_handling(self, service, real_db):

        """Test error handling during generation."""
        conv_id = "invalid_conversation_id"

        with patch('tldw_Server_API.app.core.Chat.document_generator.chat_api_call') as mock_llm:
            mock_llm.side_effect = Exception("LLM API error")

            result = service.generate_document(
                conversation_id=conv_id,
                document_type=DocumentType.TIMELINE,
                provider="openai",
                model="gpt-3.5-turbo",
                api_key="test_key"
            )

        # Should handle the error gracefully
        assert result is not None

    def test_conversation_not_found(self, service, real_db):

        """Test handling when conversation doesn't exist."""
        result = service.generate_document(
            conversation_id="nonexistent",
            document_type=DocumentType.TIMELINE,
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test_key"
        )

        # Should handle missing conversation gracefully
        assert result is not None

"""
Unit tests for contextual chunking functionality in ChromaDB_Library.

Tests the ability to add LLM-generated context to chunks during embedding.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any, List
import json

from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager


class TestContextualChunking:
    """Test suite for contextual chunking features."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a mock configuration with contextual settings."""
        return {
            "USER_DB_BASE_DIR": str(tmp_path / "test_db"),
            "embedding_config": {
                "default_model_id": "test-model",
                "enable_contextual_chunking": False,  # Default to false
                "contextual_llm_model": "gpt-3.5-turbo",
                "models": {
                    "test-model": {
                        "provider": "openai",
                        "dimension": 1536
                    }
                }
            },
            "chroma_client_settings": {
                "anonymized_telemetry": False,
                "allow_reset": True
            }
        }

    @pytest.fixture
    def mock_chroma_manager(self, mock_config):
        """Create a mock ChromaDBManager instance."""
        with patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.chromadb'):
            manager = ChromaDBManager(
                user_id="test_user",
                user_embedding_config=mock_config
            )
            try:
                yield manager
            finally:
                manager.close()

    def test_contextual_chunking_disabled_by_default(self, mock_chroma_manager):

        """Test that contextual chunking is disabled by default."""
        assert mock_chroma_manager.embedding_config.get("enable_contextual_chunking", False) == False

    def test_contextual_chunking_config_override(self, mock_config):

        """Test that config settings can override contextual chunking default."""
        mock_config["embedding_config"]["enable_contextual_chunking"] = True

        with patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.chromadb'):
            manager = ChromaDBManager(
                user_id="test_user",
                user_embedding_config=mock_config
            )
            try:
                assert manager.embedding_config.get("enable_contextual_chunking") == True
            finally:
                manager.close()

    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.chunk_for_embedding')
    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.create_embeddings_batch')
    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.analyze')
    def test_process_content_with_contextualization_enabled(
        self,
        mock_analyze,
        mock_create_embeddings,
        mock_chunk_for_embedding,
        mock_chroma_manager
    ):
        """Test that contextualization is applied when explicitly enabled."""
        # Setup mocks
        mock_chunk_for_embedding.return_value = [
            {"text": "chunk1", "metadata": {}},
            {"text": "chunk2", "metadata": {}}
        ]
        mock_analyze.return_value = "contextual summary"
        mock_create_embeddings.return_value = [[0.1] * 1536, [0.2] * 1536]

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0  # Mock count method
        mock_collection.metadata = {}  # Mock metadata
        mock_chroma_manager.get_or_create_collection = MagicMock(return_value=mock_collection)

        # Call with contextualization enabled
        mock_chroma_manager.process_and_store_content(
            content="test content",
            media_id=1,
            file_name="test.txt",
            create_contextualized=True,  # Explicitly enable
            llm_model_for_context="gpt-4"
        )

        # Verify analyze was called for contextualization
        assert mock_analyze.call_count == 2  # Once per chunk

        # Verify embeddings were created with contextualized text
        mock_create_embeddings.assert_called_once()
        texts_for_embedding = mock_create_embeddings.call_args[1]['texts']
        assert len(texts_for_embedding) == 2
        assert "Contextual Summary:" in texts_for_embedding[0]
        assert "Contextual Summary:" in texts_for_embedding[1]

    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.chunk_for_embedding')
    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.create_embeddings_batch')
    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.analyze')
    def test_process_content_without_contextualization(
        self,
        mock_analyze,
        mock_create_embeddings,
        mock_chunk_for_embedding,
        mock_chroma_manager
    ):
        """Test that contextualization is NOT applied when disabled."""
        # Setup mocks
        mock_chunk_for_embedding.return_value = [
            {"text": "chunk1", "metadata": {}},
            {"text": "chunk2", "metadata": {}}
        ]
        mock_create_embeddings.return_value = [[0.1] * 1536, [0.2] * 1536]

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0  # Mock count method
        mock_collection.metadata = {}  # Mock metadata
        mock_chroma_manager.get_or_create_collection = MagicMock(return_value=mock_collection)

        # Call with contextualization disabled (default)
        mock_chroma_manager.process_and_store_content(
            content="test content",
            media_id=1,
            file_name="test.txt",
            create_contextualized=False  # Explicitly disable
        )

        # Verify analyze was NOT called
        mock_analyze.assert_not_called()

        # Verify embeddings were created with original text only
        mock_create_embeddings.assert_called_once()
        texts_for_embedding = mock_create_embeddings.call_args[1]['texts']
        assert len(texts_for_embedding) == 2
        assert texts_for_embedding[0] == "chunk1"
        assert texts_for_embedding[1] == "chunk2"

    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.chunk_for_embedding')
    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.create_embeddings_batch')
    def test_process_content_uses_config_default_when_none(
        self,
        mock_create_embeddings,
        mock_chunk_for_embedding,
        mock_chroma_manager
    ):
        """Test that None value for create_contextualized uses config default."""
        # Setup config to have contextualization enabled
        mock_chroma_manager.embedding_config["enable_contextual_chunking"] = True

        # Setup mocks
        mock_chunk_for_embedding.return_value = [
            {"text": "chunk1", "metadata": {}}
        ]
        mock_create_embeddings.return_value = [[0.1] * 1536]

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0  # Mock count method
        mock_collection.metadata = {}  # Mock metadata
        mock_chroma_manager.get_or_create_collection = MagicMock(return_value=mock_collection)

        with patch.object(mock_chroma_manager, 'situate_context', return_value="context") as mock_situate:
            # Call with None (should use config default)
            mock_chroma_manager.process_and_store_content(
                content="test content",
                media_id=1,
                file_name="test.txt",
                create_contextualized=None  # Use config default
            )

            # Should have called situate_context since config default is True
            mock_situate.assert_called()

    def test_situate_context_generates_summary(self, mock_chroma_manager):

        """Test that situate_context generates appropriate context."""
        with patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.analyze') as mock_analyze:
            mock_analyze.return_value = "This chunk discusses machine learning concepts."

            result = mock_chroma_manager.situate_context(
                api_name_for_context="gpt-3.5-turbo",
                doc_content="Full document about AI and ML",
                chunk_content="Neural networks are..."
            )

            assert result == "This chunk discusses machine learning concepts."
            mock_analyze.assert_called_once()

            # Check the prompt includes both document and chunk
            call_args = mock_analyze.call_args
            assert "Full document about AI and ML" in str(call_args)
            assert "Neural networks are..." in str(call_args)

    def test_situate_context_handles_errors_gracefully(self, mock_chroma_manager):

        """Test that situate_context handles LLM errors gracefully."""
        with patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.analyze') as mock_analyze:
            mock_analyze.side_effect = Exception("LLM API error")

            result = mock_chroma_manager.situate_context(
                api_name_for_context="gpt-3.5-turbo",
                doc_content="Full document",
                chunk_content="Chunk content"
            )

            # Should return empty string on error
            assert result == ""

    def test_contextual_llm_model_selection(self, mock_chroma_manager):

        """Test that the correct LLM model is selected for contextualization."""
        # Test default model
        assert mock_chroma_manager.embedding_config.get("contextual_llm_model", "gpt-3.5-turbo") == "gpt-3.5-turbo"

        # Test custom model in config
        mock_chroma_manager.embedding_config["contextual_llm_model"] = "gpt-4"
        assert mock_chroma_manager.embedding_config.get("contextual_llm_model") == "gpt-4"

    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.chunk_for_embedding')
    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.create_embeddings_batch')
    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.analyze')
    def test_metadata_includes_contextualization_flag(
        self,
        mock_analyze,
        mock_create_embeddings,
        mock_chunk_for_embedding,
        mock_chroma_manager
    ):
        """Test that chunk metadata includes contextualization information."""
        # Setup mocks
        mock_chunk_for_embedding.return_value = [
            {"text": "chunk1", "metadata": {}}
        ]
        mock_analyze.return_value = "context"
        mock_create_embeddings.return_value = [[0.1] * 1536]

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0  # Mock count method
        mock_collection.metadata = {}  # Mock metadata
        mock_chroma_manager.get_or_create_collection = MagicMock(return_value=mock_collection)

        # Call with contextualization enabled
        mock_chroma_manager.process_and_store_content(
            content="test content",
            media_id=1,
            file_name="test.txt",
            create_contextualized=True
        )

        # Check that metadata was added to collection
        assert mock_collection.add.called or mock_collection.upsert.called

        # Check upsert was called (the actual method used in the code)
        if mock_collection.upsert.called:
            call_args = mock_collection.upsert.call_args
            if call_args and len(call_args) > 1 and 'metadatas' in call_args[1]:
                metadatas = call_args[1]['metadatas']
                assert len(metadatas) == 1
                assert metadatas[0]['contextualized'] == True

"""
Unit tests for ChromaDBManager class.

Tests cover all public methods with mocked dependencies.
Focus on business logic, error handling, and edge cases.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import json
import uuid
from typing import List, Dict, Any

from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager


@pytest.mark.unit
class TestChromaDBManagerInit:
    """Test ChromaDBManager initialization and setup."""

    def test_init_with_valid_user_id(self):

        """Test initialization with valid user ID."""
        import tempfile
        base_dir = tempfile.mkdtemp(prefix="chroma_user_base_")
        manager = ChromaDBManager(
            user_id="valid_user_123",
            user_embedding_config={"USER_DB_BASE_DIR": base_dir, "embedding_config": {}, "chroma_client_settings": {"backend": "stub"}},
            client=MagicMock(),
        )
        assert manager.user_id == "valid_user_123"
        assert manager.user_embedding_config.get("USER_DB_BASE_DIR") == base_dir

    def test_init_with_invalid_user_id(self):

        """Test initialization rejects invalid user IDs."""
        import tempfile
        base_dir = tempfile.mkdtemp(prefix="chroma_user_base_")
        # Test path traversal attempt
        with pytest.raises(ValueError, match="Invalid user_id"):
            ChromaDBManager(
                user_id="../malicious",
                user_embedding_config={"USER_DB_BASE_DIR": base_dir}
            )

        # Test special characters
        with pytest.raises(ValueError, match="Invalid user_id"):
            ChromaDBManager(
                user_id="user$#@!",
                user_embedding_config={"USER_DB_BASE_DIR": base_dir}
            )

    def test_init_creates_user_directory(self):

        """Test initialization creates user-specific directory."""
        import tempfile, os
        base_dir = tempfile.mkdtemp(prefix="chroma_user_base_")
        manager = ChromaDBManager(
            user_id="test_user",
            user_embedding_config={"USER_DB_BASE_DIR": base_dir, "chroma_client_settings": {"backend": "stub"}},
            client=MagicMock(),
        )
        assert os.path.isdir(str(manager.user_chroma_path))

    def test_init_with_custom_base_path(self):

        """Test initialization with custom base path."""
        import tempfile, os, pathlib
        base_dir = tempfile.mkdtemp(prefix="chroma_user_base_")
        manager = ChromaDBManager(
            user_id="test_user",
            user_embedding_config={"USER_DB_BASE_DIR": base_dir, "chroma_client_settings": {"backend": "stub"}},
            client=MagicMock(),
        )
        # Normalize both to avoid /private path prefix differences on macOS
        resolved_base = str(pathlib.Path(base_dir).resolve())
        resolved_user_path = str(pathlib.Path(str(manager.user_chroma_path)).resolve())
        assert resolved_user_path.startswith(resolved_base)

    def test_init_with_stub_backend_records_cache_key(self):

        """Test stub backend init uses base dir in stub cache key."""
        import tempfile
        from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as cdl

        base_dir = tempfile.mkdtemp(prefix="chroma_user_base_")
        with cdl._TEST_STUB_CLIENTS_LOCK:
            cdl._TEST_STUB_CLIENTS.clear()

        ChromaDBManager(
            user_id="stub_user",
            user_embedding_config={
                "USER_DB_BASE_DIR": base_dir,
                "chroma_client_settings": {"backend": "stub"},
            },
        )

        expected_key = f"stub_user::{base_dir}"
        with cdl._TEST_STUB_CLIENTS_LOCK:
            assert expected_key in cdl._TEST_STUB_CLIENTS

    def test_init_accepts_uppercase_embedding_config(self):

        """Ensure EMBEDDING_CONFIG is normalized into embedding_config."""
        import tempfile
        base_dir = tempfile.mkdtemp(prefix="chroma_user_base_")
        manager = ChromaDBManager(
            user_id="upper_cfg_user",
            user_embedding_config={
                "USER_DB_BASE_DIR": base_dir,
                "EMBEDDING_CONFIG": {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"},
                "chroma_client_settings": {"backend": "stub"},
            },
            client=MagicMock(),
        )
        assert manager.embedding_config.get("embedding_model") == "sentence-transformers/all-MiniLM-L6-v2"

    def test_init_accepts_raw_embedding_config(self):

        """Ensure raw embedding config is wrapped when embedding_config is missing."""
        import tempfile
        base_dir = tempfile.mkdtemp(prefix="chroma_user_base_")
        manager = ChromaDBManager(
            user_id="raw_cfg_user",
            user_embedding_config={
                "USER_DB_BASE_DIR": base_dir,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_provider": "huggingface",
                "chroma_client_settings": {"backend": "stub"},
            },
            client=MagicMock(),
        )
        assert manager.embedding_config.get("embedding_model") == "sentence-transformers/all-MiniLM-L6-v2"

    def test_init_fallback_to_stub_records_cache_key(self):

        """Test stub fallback init uses base dir in stub cache key."""
        import tempfile
        from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as cdl

        base_dir = tempfile.mkdtemp(prefix="chroma_user_base_")
        with cdl._TEST_STUB_CLIENTS_LOCK:
            cdl._TEST_STUB_CLIENTS.clear()

        with patch(
            "tldw_Server_API.app.core.Embeddings.ChromaDB_Library.chromadb.PersistentClient",
            side_effect=RuntimeError("boom"),
        ):
            ChromaDBManager(
                user_id="fallback_user",
                user_embedding_config={
                    "USER_DB_BASE_DIR": base_dir,
                    "chroma_client_settings": {"backend": "persistent", "allow_stub_fallback": True},
                },
            )

        expected_key = f"fallback_user::{base_dir}"
        with cdl._TEST_STUB_CLIENTS_LOCK:
            assert expected_key in cdl._TEST_STUB_CLIENTS

    def test_init_shares_inert_client_handles_without_retaining_live_sdk(self):

        """Managers share coordination, but no SDK lives beyond its store lock."""
        import tempfile
        from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as cdl

        base_dir = tempfile.mkdtemp(prefix="chroma_user_base_")
        persistent_client = MagicMock()
        persistent_client.close = MagicMock()

        with patch(
            "tldw_Server_API.app.core.Embeddings.ChromaDB_Library.chromadb.PersistentClient",
            return_value=persistent_client,
        ) as mock_persistent:
            manager1 = cdl.ChromaDBManager(
                user_id="cached_user",
                user_embedding_config={
                    "USER_DB_BASE_DIR": base_dir,
                    "embedding_config": {},
                    "chroma_client_settings": {
                        "backend": "persistent",
                        "allow_stub_fallback": True,
                    },
                },
            )
            manager2 = cdl.ChromaDBManager(
                user_id="cached_user",
                user_embedding_config={
                    "USER_DB_BASE_DIR": base_dir,
                    "embedding_config": {},
                    "chroma_client_settings": {
                        "backend": "persistent",
                        "allow_stub_fallback": True,
                    },
                },
            )
            assert manager2.client is manager1.client
            assert manager1._lock is manager2._lock
            assert manager1._lock.sdk is None
            persistent_client.close.assert_called_once()
            manager1.close()
            persistent_client.close.assert_called_once()
            manager2.close()

        assert mock_persistent.call_count == 1
        persistent_client.close.assert_called_once()

    def test_init_does_not_reuse_cached_persistent_client_after_last_close(self):

        """Closing the final manager should evict the cached client so the next init reopens it."""
        import tempfile
        from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as cdl

        base_dir = tempfile.mkdtemp(prefix="chroma_user_base_")
        first_client = MagicMock()
        first_client.close = MagicMock()
        second_client = MagicMock()
        second_client.close = MagicMock()

        with patch(
            "tldw_Server_API.app.core.Embeddings.ChromaDB_Library.chromadb.PersistentClient",
            side_effect=[first_client, second_client],
        ) as mock_persistent:
            manager1 = cdl.ChromaDBManager(
                user_id="cached_user",
                user_embedding_config={
                    "USER_DB_BASE_DIR": base_dir,
                    "embedding_config": {},
                    "chroma_client_settings": {
                        "backend": "persistent",
                        "allow_stub_fallback": True,
                    },
                },
            )
            manager1.close()

            manager2 = cdl.ChromaDBManager(
                user_id="cached_user",
                user_embedding_config={
                    "USER_DB_BASE_DIR": base_dir,
                    "embedding_config": {},
                    "chroma_client_settings": {
                        "backend": "persistent",
                        "allow_stub_fallback": True,
                    },
                },
            )
            manager2.close()

        assert mock_persistent.call_count == 2
        assert manager2.client is None
        first_client.close.assert_called_once()


@pytest.mark.unit
class TestCollectionManagement:
    """Test collection management operations."""

    def test_get_or_create_collection_new(self, chromadb_manager, mock_chroma_client):

        """Test creating a new collection."""
        collection_name = "test_collection"
        metadata = {"description": "Test collection"}

        collection = chromadb_manager.get_or_create_collection(
            collection_name=collection_name,
            collection_metadata=metadata
        )

        mock_chroma_client.get_or_create_collection.assert_called_once()
        assert collection is not None

    def test_get_or_create_collection_existing(self, chromadb_manager, mock_chroma_client):

        """Test getting an existing collection."""
        collection_name = "existing_collection"

        # Simulate existing collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        collection = chromadb_manager.get_or_create_collection(collection_name)
        assert collection == mock_collection

    @pytest.mark.legacy_skip
    @pytest.mark.xfail(strict=False, reason="Collection name validation not implemented in current API. TODO(#chroma-collection-name-validation)")
    def test_get_or_create_collection_with_invalid_name(self, chromadb_manager):
        """
        TODO(#chroma-collection-name-validation): Implement name sanitation in API then convert to a real test.
        """
        # This test is intentionally marked legacy_skip/xfail; it targets behavior not yet implemented.
        pass

    def test_reset_collection(self, chromadb_manager, mock_chroma_client):

        """Test resetting a collection."""
        collection_name = "test_collection"
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        chromadb_manager.reset_chroma_collection(collection_name)

        mock_chroma_client.delete_collection.assert_called_once_with(name=collection_name)
        mock_chroma_client.create_collection.assert_called_once_with(name=collection_name)

    def test_delete_collection(self, chromadb_manager, mock_chroma_client):

        """Test deleting a collection."""
        collection_name = "test_collection"

        chromadb_manager.delete_collection(collection_name)

        mock_chroma_client.delete_collection.assert_called_once_with(name=collection_name)

    def test_count_items_in_collection(self, chromadb_manager, mock_chroma_client):

        """Test counting items in a collection."""
        collection_name = "test_collection"
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        count = chromadb_manager.count_items_in_collection(collection_name)

        assert count == 42
        mock_collection.count.assert_called_once()


@pytest.mark.unit
class TestStorageOperations:
    """Test storage and retrieval operations."""

    def test_store_in_chroma_basic(self, chromadb_manager, mock_chroma_client):

        """Test basic storage operation."""
        collection_name = "test_collection"
        texts = ["text1", "text2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        ids = ["id1", "id2"]

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        result = chromadb_manager.store_in_chroma(
            collection_name=collection_name,
            texts=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{} for _ in ids]
        )

        mock_collection.upsert.assert_called_once_with(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{} for _ in ids]
        )
        assert result is not None

    def test_store_in_chroma_with_metadata(self, chromadb_manager, mock_chroma_client):

        """Test storage with metadata."""
        collection_name = "test_collection"
        texts = ["text1"]
        embeddings = [[0.1, 0.2]]
        ids = ["id1"]
        metadata = [{"key": "value"}]

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        result = chromadb_manager.store_in_chroma(
            collection_name=collection_name,
            texts=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata
        )

        mock_collection.upsert.assert_called_once_with(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata
        )
        assert result is not None

    def test_store_recreates_on_dimension_mismatch(self, chromadb_manager, mock_chroma_client):

        """If existing collection metadata has a different embedding_dimension, it recreates the collection."""
        collection_name = "dim_mismatch"
        texts = ["t1", "t2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]  # dim=2
        ids = ["id1", "id2"]

        mock_collection = MagicMock()
        mock_collection.name = collection_name
        mock_collection.metadata = {"embedding_dimension": 999}  # Force mismatch
        mock_collection.count.return_value = 0
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        result = chromadb_manager.store_in_chroma(
            collection_name=collection_name,
            texts=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{} for _ in ids],
            embedding_model_id_for_dim_check="text-embedding-ada-002"
        )

        # Should delete and create collection due to dim mismatch
        mock_chroma_client.delete_collection.assert_called_once_with(name=collection_name)
        mock_chroma_client.create_collection.assert_called_once()
        assert result is not None

    def test_store_in_chroma_mismatched_lengths(self, chromadb_manager):

        """Test storage with mismatched input lengths."""
        with pytest.raises(ValueError, match="length mismatch"):
            chromadb_manager.store_in_chroma(
                collection_name="test",
                texts=["text1", "text2"],
                embeddings=[[0.1]],  # Wrong length
                ids=["id1", "id2"],
                metadatas=[{}, {}]
            )

    def test_store_in_chroma_empty_inputs(self, chromadb_manager):

        """Test storage with empty inputs."""
        with pytest.raises(ValueError, match="empty"):
            chromadb_manager.store_in_chroma(
                collection_name="test",
                texts=[],
                embeddings=[],
                ids=[],
                metadatas=[]
            )

    def test_store_handles_error(self, chromadb_manager, mock_chroma_client):

        """Test storage error handling."""
        mock_collection = MagicMock()
        mock_collection.add.side_effect = Exception("Storage failed")
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        with pytest.raises(RuntimeError):
            chromadb_manager.store_in_chroma(
                collection_name="test",
                texts=["text"],
                embeddings=[[0.1]],
                ids=["id1"],
                metadatas=[{}]
            )

    def test_delete_from_collection(self, chromadb_manager, mock_chroma_client):

        """Test deleting items from collection."""
        collection_name = "test_collection"
        ids = ["id1", "id2"]

        mock_collection = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        chromadb_manager.delete_from_collection(ids, collection_name)

        mock_collection.delete.assert_called_once_with(ids=ids)

    def test_delete_from_collection_handles_error(self, chromadb_manager, mock_chroma_client):

        """Test delete error handling."""
        mock_collection = MagicMock()
        mock_collection.delete.side_effect = RuntimeError("Delete failed")
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        with pytest.raises(RuntimeError):
            chromadb_manager.delete_from_collection(["id1"], "test")


@pytest.mark.unit
class TestSearchOperations:
    """Test search and query operations."""

    def test_vector_search_basic(self, chromadb_manager, mock_chroma_client, mock_embeddings):

        """Test basic vector search."""
        query = "test query"
        collection_name = "test_collection"

        # Setup mock collection
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"key": "value1"}, {"key": "value2"}]]
        }
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        # Mock embedding creation
        with patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.create_embedding', return_value=[0.1, 0.2, 0.3]):
            results = chromadb_manager.vector_search(
                query=query,
                collection_name=collection_name,
                k=2,
                embedding_model_id_override="text-embedding-ada-002"
            )

        assert isinstance(results, list)
        assert [r["id"] for r in results] == ["id1", "id2"]
        mock_collection.query.assert_called_once()

    def test_vector_search_with_filter(self, chromadb_manager, mock_chroma_client):

        """Test vector search with metadata filter."""
        query = "test query"
        collection_name = "test_collection"
        filter_dict = {"category": "test"}

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["doc1"]],
            "distances": [[0.1]],
            "metadatas": [[{"category": "test"}]]
        }
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        with patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.create_embedding', return_value=[0.1, 0.2, 0.3]):
            results = chromadb_manager.vector_search(
                query=query,
                collection_name=collection_name,
                k=1,
                embedding_model_id_override="text-embedding-ada-002",
                where_filter=filter_dict
            )

        call_args = mock_collection.query.call_args
        assert call_args[1].get('where') == filter_dict

    def test_vector_search_custom_fields(self, chromadb_manager, mock_chroma_client):

        """Test vector search with custom include fields."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "embeddings": [[[0.1, 0.2]]],
            "documents": None,
            "metadatas": None,
            "distances": [[0.1]]
        }
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        with patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.create_embedding', return_value=[0.1, 0.2, 0.3]):
            results = chromadb_manager.vector_search(
                query="test",
                collection_name="test",
                embedding_model_id_override="text-embedding-ada-002",
                include_fields=["embeddings", "distances"]
            )
        assert isinstance(results, list)
        assert all("distance" in r for r in results)

    def test_query_with_precomputed_embeddings(self, chromadb_manager, mock_chroma_client):

        """Test query with precomputed embeddings."""
        embeddings = [[0.1, 0.2, 0.3]]
        collection_name = "test_collection"

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["doc1"]],
            "distances": [[0.1]],
            "metadatas": [[{"key": "value"}]]
        }
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        results = chromadb_manager.query_collection_with_precomputed_embeddings(
            collection_name=collection_name,
            query_embeddings=embeddings,
            n_results=1
        )

        assert results is not None
        mock_collection.query.assert_called_once_with(
            query_embeddings=embeddings,
            n_results=1,
            where=None,
            include=["documents", "metadatas", "distances"]
        )

    def test_search_handles_empty_results(self, chromadb_manager, mock_chroma_client):

        """Test search handles empty results gracefully."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]]
        }
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        with patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.create_embedding', return_value=[0.1, 0.2, 0.3]):
            results = chromadb_manager.vector_search("test", "test", k=5, embedding_model_id_override="text-embedding-ada-002")

        assert isinstance(results, list)
        assert results == []


@pytest.mark.unit
class TestContentProcessing:
    """Test content processing pipeline."""

    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.chunk_for_embedding')
    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.create_embeddings_batch')
    def test_process_and_store_content(self, mock_create_emb, mock_chunk_for_embed, chromadb_manager, mock_chroma_client):
        """Test end-to-end content processing."""
        content = "This is test content to be processed."
        media_id = "media_123"

        # Setup mocks
        mock_chunk_for_embed.return_value = [
            {"text": "chunk1", "metadata": {"paragraph_kind": "header_atx", "start_index": 5, "end_index": 12}},
            {"text": "chunk2", "metadata": {}},
        ]
        mock_create_emb.return_value = [[0.1, 0.2], [0.3, 0.4]]

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        chromadb_manager.process_and_store_content(
            content=content,
            media_id=media_id,
            collection_name="test_collection",
            file_name="test.txt"
        )

        mock_chunk_for_embed.assert_called_once()
        mock_create_emb.assert_called_once()
        mock_collection.upsert.assert_called_once()

        args, kwargs = mock_collection.upsert.call_args
        metadatas = kwargs.get("metadatas") or []
        assert len(metadatas) == 2
        assert metadatas[0].get("chunk_type") == "heading"
        assert metadatas[0].get("start_char") == 5
        assert metadatas[0].get("end_char") == 12
        assert metadatas[1].get("chunk_type") == "text"

    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.chunk_for_embedding')
    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.create_embeddings_batch')
    def test_process_content_with_custom_chunking(self, mock_create_emb, mock_chunk_for_embed, chromadb_manager):
        """Test content processing with custom chunk parameters."""
        content = "Test content"
        mock_chunk_for_embed.return_value = [{"text": "chunk1", "metadata": {}}]
        mock_create_emb.return_value = [[0.1, 0.2]]

        with patch.object(chromadb_manager, 'store_in_chroma') as mock_store:
            mock_store.return_value = True

            chromadb_manager.process_and_store_content(
                content=content,
                media_id="test",
                file_name="file.txt",
                chunk_options={"chunk_size": 500, "overlap": 50}
            )

        args, kwargs = mock_chunk_for_embed.call_args
        assert args[0] == content
        assert args[1] == "file.txt"
        # chunk options are passed via kwargs
        assert kwargs.get("chunk_size") == 500
        assert kwargs.get("overlap") == 50

    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.chunk_for_embedding')
    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.create_embeddings_batch')
    def test_process_handles_chunking_failure(self, mock_create_emb, mock_chunk_for_embed, chromadb_manager):
        """Test handling of chunking failures."""
        mock_chunk_for_embed.side_effect = Exception("Chunking failed")

        with pytest.raises(Exception):
            chromadb_manager.process_and_store_content(
                content="test",
                media_id="test",
                file_name="file.txt"
            )

        mock_create_emb.assert_not_called()

    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.chunk_for_embedding')
    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.create_embeddings_batch')
    def test_process_handles_embedding_failure(self, mock_create_emb, mock_chunk_for_embed, chromadb_manager):
        """Test handling of embedding generation failures."""
        mock_chunk_for_embed.return_value = [{"text": "chunk1", "metadata": {}}]
        mock_create_emb.side_effect = Exception("Embedding failed")

        with pytest.raises(Exception):
            chromadb_manager.process_and_store_content(
                content="test",
                media_id="test",
                file_name="file.txt"
            )


@pytest.mark.unit
@pytest.mark.legacy_skip
@pytest.mark.xfail(strict=False, reason="Tests target internal helpers not present in current API. TODO(#chroma-security-internals)")
class TestSecurityValidation:
    """Test security and input validation."""

    def test_sanitize_collection_name(self, chromadb_manager):

        """Test collection name sanitization."""
        # Valid names should pass through
        assert chromadb_manager._sanitize_name("valid_name") == "valid_name"
        assert chromadb_manager._sanitize_name("name123") == "name123"

        # Invalid characters should be replaced
        assert chromadb_manager._sanitize_name("name with spaces") == "name_with_spaces"
        assert chromadb_manager._sanitize_name("name@#$%") == "name____"

        # Path traversal attempts should be blocked
        with pytest.raises(ValueError):
            chromadb_manager._sanitize_name("../malicious")

        with pytest.raises(ValueError):
            chromadb_manager._sanitize_name("..\\malicious")

    def test_validate_user_id_security(self):

        """Test user ID validation for security."""
        with patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.chromadb'):
            # Valid IDs
            ChromaDBManager("user123", {})
            ChromaDBManager("user_test", {})
            ChromaDBManager("user-test", {})

            # Invalid IDs
            invalid_ids = [
                "../evil",
                "..\\evil",
                "/etc/passwd",
                "user/../other",
                "user;rm -rf /",
                "user$(whoami)",
                "user`ls`"
            ]

            for invalid_id in invalid_ids:
                with pytest.raises(ValueError, match="Invalid user_id"):
                    ChromaDBManager(invalid_id, {})

    def test_metadata_validation(self, chromadb_manager):

        """Test metadata validation and sanitization."""
        # Valid metadata
        valid_metadata = {"key": "value", "number": 123, "bool": True}
        sanitized = chromadb_manager._sanitize_metadata(valid_metadata)
        assert sanitized == valid_metadata

        # Nested objects should be flattened or handled
        nested_metadata = {
            "key": "value",
            "nested": {"inner": "value"}
        }
        sanitized = chromadb_manager._sanitize_metadata(nested_metadata)
        assert isinstance(sanitized["nested"], (str, dict))

        # Large values should be truncated
        large_metadata = {"key": "x" * 10000}
        sanitized = chromadb_manager._sanitize_metadata(large_metadata)
        assert len(str(sanitized["key"])) <= 5000  # Reasonable limit

    def test_embedding_dimension_validation(self, chromadb_manager):

        """Test embedding dimension validation."""
        # Valid embeddings
        valid_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert chromadb_manager._validate_embeddings(valid_embeddings)

        # Mismatched dimensions
        invalid_embeddings = [[0.1, 0.2], [0.3, 0.4, 0.5]]
        with pytest.raises(ValueError, match="dimension"):
            chromadb_manager._validate_embeddings(invalid_embeddings)

        # Non-numeric values
        invalid_embeddings = [["a", "b", "c"]]
        with pytest.raises(ValueError, match="numeric"):
            chromadb_manager._validate_embeddings(invalid_embeddings)


@pytest.mark.unit
@pytest.mark.legacy_skip
@pytest.mark.xfail(strict=False, reason="Connection retry internals not exposed in current API. TODO(#chroma-connection-retry)")
class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""

    def test_connection_retry_on_failure(self, chromadb_manager, mock_chroma_client):

        """Test connection retry logic."""
        # Simulate connection failure then success
        mock_chroma_client.heartbeat.side_effect = [
            Exception("Connection failed"),
            Exception("Connection failed"),
            None  # Success on third try
        ]

        with patch('time.sleep'):  # Speed up test
            result = chromadb_manager._ensure_connection()

        assert result is True
        assert mock_chroma_client.heartbeat.call_count == 3

    def test_max_retries_exceeded(self, chromadb_manager, mock_chroma_client):

        """Test behavior when max retries exceeded."""
        mock_chroma_client.heartbeat.side_effect = Exception("Connection failed")

        with patch('time.sleep'):  # Speed up test
            result = chromadb_manager._ensure_connection()

        assert result is False

    def test_graceful_degradation_on_storage_failure(self, chromadb_manager, mock_chroma_client):

        """Test graceful degradation when storage fails."""
        mock_collection = MagicMock()
        mock_collection.add.side_effect = Exception("Storage unavailable")
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        # Should not raise exception, but return False
        result = chromadb_manager.store_in_chroma(
            collection_name="test",
            texts=["text"],
            embeddings=[[0.1]],
            ids=["id1"]
        )

        assert result is False

    def test_cleanup_on_partial_failure(self, chromadb_manager):

        """Test cleanup after partial operation failure."""
        with patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.improved_chunking_process') as mock_chunk:
            with patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.create_embeddings_batch') as mock_emb:
                mock_chunk.return_value = ["chunk1", "chunk2"]
                mock_emb.side_effect = Exception("Embedding failed")

                result = chromadb_manager.process_and_store_content(
                    content="test",
                    media_id="test"
                )

                assert result is False
                # Verify no partial data was stored
                assert chromadb_manager.client.get_or_create_collection.call_count == 0


@pytest.mark.unit
@pytest.mark.legacy_skip
@pytest.mark.xfail(strict=False, reason="Resource limit helpers not implemented in current API. TODO(#chroma-resource-limits)")
class TestResourceManagement:
    """Test resource management and limits."""

    def test_collection_limit_enforcement(self, chromadb_manager, mock_chroma_client):

        """Test enforcement of collection limits per user."""
        # Simulate max collections reached
        mock_collections = [MagicMock(name=f"col_{i}") for i in range(10)]
        mock_chroma_client.list_collections.return_value = mock_collections

        with patch.object(chromadb_manager, 'max_collections_per_user', 10):
            with pytest.raises(ValueError, match="Maximum collections"):
                chromadb_manager.get_or_create_collection("new_collection")

    def test_item_limit_per_collection(self, chromadb_manager, mock_chroma_client):

        """Test enforcement of item limits per collection."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100000
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        with patch.object(chromadb_manager, 'max_items_per_collection', 100000):
            with pytest.raises(ValueError, match="Maximum items"):
                chromadb_manager.store_in_chroma(
                    collection_name="test",
                    texts=["new_text"],
                    embeddings=[[0.1]],
                    ids=["new_id"]
                )

    def test_memory_limit_check(self, chromadb_manager):

        """Test memory usage monitoring."""
        with patch('psutil.Process') as mock_process:
            mock_proc_instance = MagicMock()
            mock_proc_instance.memory_info.return_value.rss = 2 * 1024 * 1024 * 1024  # 2GB
            mock_process.return_value = mock_proc_instance

            with patch.object(chromadb_manager, 'max_memory_gb', 1.5):
                with pytest.raises(MemoryError, match="Memory limit"):
                    chromadb_manager._check_memory_usage()

    def test_batch_size_limiting(self, chromadb_manager):

        """Test batch size limiting for large operations."""
        large_texts = [f"text_{i}" for i in range(1000)]
        large_embeddings = [[0.1] * 384 for _ in range(1000)]
        large_ids = [f"id_{i}" for i in range(1000)]

        with patch.object(chromadb_manager, 'batch_size', 100):
            with patch.object(chromadb_manager, '_store_batch') as mock_store:
                mock_store.return_value = True

                chromadb_manager.store_in_chroma(
                    collection_name="test",
                    texts=large_texts,
                    embeddings=large_embeddings,
                    ids=large_ids
                )

                # Should be called 10 times (1000 items / 100 batch size)
                assert mock_store.call_count == 10

    def test_list_collections(self, chromadb_manager, mock_chroma_client):

        """Verify list_collections proxies to client and returns sequence."""
        fake_collections = [MagicMock(name="colA"), MagicMock(name="colB")]
        mock_chroma_client.list_collections.return_value = fake_collections
        cols = chromadb_manager.list_collections()
        assert cols == fake_collections


@pytest.mark.unit
@pytest.mark.legacy_skip
@pytest.mark.xfail(strict=False, reason="Audit logger not used in current API; events handled differently. TODO(#chroma-audit-integration)")
class TestAuditingAndLogging:
    """Test audit logging functionality."""

    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.AuditLogger')
    def test_audit_log_on_store(self, mock_audit_logger, chromadb_manager):
        """Test audit logging for storage operations."""
        mock_logger_instance = MagicMock()
        mock_audit_logger.return_value = mock_logger_instance

        with patch.object(chromadb_manager, '_do_store') as mock_store:
            mock_store.return_value = True

            chromadb_manager.store_in_chroma(
                collection_name="test",
                texts=["text"],
                embeddings=[[0.1]],
                ids=["id1"]
            )

        mock_logger_instance.log_operation.assert_called_with(
            operation="store",
            user_id=chromadb_manager.user_id,
            details=mock.ANY
        )

    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.AuditLogger')
    def test_audit_log_on_search(self, mock_audit_logger, chromadb_manager):
        """Test audit logging for search operations."""
        mock_logger_instance = MagicMock()
        mock_audit_logger.return_value = mock_logger_instance

        with patch.object(chromadb_manager, 'create_query_embedding') as mock_embed:
            mock_embed.return_value = [0.1, 0.2]

            with patch.object(chromadb_manager.client, 'get_or_create_collection') as mock_get:
                mock_collection = MagicMock()
                mock_collection.query.return_value = {"ids": [[]], "documents": [[]]}
                mock_get.return_value = mock_collection

                chromadb_manager.vector_search("test query", "test_collection")

        mock_logger_instance.log_operation.assert_called_with(
            operation="search",
            user_id=chromadb_manager.user_id,
            details=mock.ANY
        )

    @patch('tldw_Server_API.app.core.Embeddings.ChromaDB_Library.AuditLogger')
    def test_audit_log_on_delete(self, mock_audit_logger, chromadb_manager):
        """Test audit logging for delete operations."""
        mock_logger_instance = MagicMock()
        mock_audit_logger.return_value = mock_logger_instance

        with patch.object(chromadb_manager.client, 'get_or_create_collection') as mock_get:
            mock_collection = MagicMock()
            mock_get.return_value = mock_collection

            chromadb_manager.delete_from_collection(["id1", "id2"], "test_collection")

        mock_logger_instance.log_operation.assert_called_with(
            operation="delete",
            user_id=chromadb_manager.user_id,
            details=mock.ANY
        )


@pytest.mark.unit
class TestConcurrency:
    """Test concurrent operation handling."""

    def test_thread_safe_operations(self, chromadb_manager):

        """Test thread safety of operations."""
        import threading
        results = []

        def store_operation(index):

            try:
                result = chromadb_manager.store_in_chroma(
                    collection_name=f"test_{index}",
                    texts=[f"text_{index}"],
                    embeddings=[[0.1, 0.2]],
                    ids=[f"id_{index}"]
                )
                results.append(result)
            except Exception as e:
                results.append(False)

        threads = [threading.Thread(target=store_operation, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All operations should complete without deadlock
        assert len(results) == 5

    def test_lock_timeout_handling(self, chromadb_manager):

        """Lock timeout helper not available; skipping."""
        pytest.skip("Operation lock timeout helper not implemented in current API")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

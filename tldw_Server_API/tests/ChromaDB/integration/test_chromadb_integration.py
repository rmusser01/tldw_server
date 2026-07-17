"""
Integration tests for ChromaDB functionality.

These tests use real ChromaDB instances and actual embedding generation
without mocking to verify end-to-end functionality.
"""

import pytest

pytestmark = pytest.mark.integration
import uuid

import chromadb
import numpy as np
from chromadb.config import Settings

from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager


@pytest.mark.integration
class TestChromaDBSetup:
    """Test ChromaDB setup and initialization."""

    def test_chromadb_client_creation(self, temp_chroma_path):

        """Test creating a ChromaDB client."""
        settings = Settings(
            persist_directory=temp_chroma_path,
            anonymized_telemetry=False,
            allow_reset=True
        )

        client = chromadb.PersistentClient(
            path=temp_chroma_path,
            settings=settings
        )
        try:
            # Verify client is functional
            assert client.heartbeat() is not None
            collections = client.list_collections()
            assert isinstance(collections, list)
        finally:
            # Best-effort shutdown
            try:
                if hasattr(client, "close"):
                    client.close()  # type: ignore[attr-defined]
                else:
                    system = getattr(client, "_system", None)
                    stop_fn = getattr(system, "stop", None) if system is not None else None
                    if callable(stop_fn):
                        stop_fn()
            except Exception:
                _ = None

    def test_chromadb_persistence(self, temp_chroma_path):

        """Test ChromaDB data persistence."""
        # Create client and add data
        settings = Settings(
            persist_directory=temp_chroma_path,
            anonymized_telemetry=False,
            allow_reset=True
        )
        client1 = chromadb.PersistentClient(path=temp_chroma_path, settings=settings)
        try:
            collection1 = client1.create_collection("test_persist")
            collection1.add(
                documents=["test document"],
                embeddings=[[0.1, 0.2, 0.3]],
                ids=["test_id"]
            )
        finally:
            try:
                if hasattr(client1, "close"):
                    client1.close()  # type: ignore[attr-defined]
                else:
                    sys1 = getattr(client1, "_system", None)
                    stop1 = getattr(sys1, "stop", None) if sys1 is not None else None
                    if callable(stop1):
                        stop1()
            except Exception:
                _ = None

        # Create new client and verify data persists
        try:
            client2 = chromadb.PersistentClient(path=temp_chroma_path, settings=settings)
        except Exception as e:
            # Some chromadb versions have a known tenant init bug in RustBindings
            pytest.skip(f"Skipping persistence re-open due to ChromaDB client init issue: {e}")
        try:
            collection2 = client2.get_collection("test_persist")
            result = collection2.get(ids=["test_id"])
            assert result["documents"][0] == "test document"
        finally:
            try:
                if hasattr(client2, "close"):
                    client2.close()  # type: ignore[attr-defined]
                else:
                    sys2 = getattr(client2, "_system", None)
                    stop2 = getattr(sys2, "stop", None) if sys2 is not None else None
                    if callable(stop2):
                        stop2()
            except Exception:
                _ = None

    def test_multiple_collections(self, chroma_client):

        """Test creating and managing multiple collections."""
        # Create multiple collections
        collections = []
        for i in range(5):
            col = chroma_client.create_collection(f"collection_{i}")
            collections.append(col)

        # Verify all collections exist
        all_collections = chroma_client.list_collections()
        assert len(all_collections) == 5

        # Add data to each collection
        for i, col in enumerate(collections):
            col.add(
                documents=[f"doc_{i}"],
                embeddings=[[float(i), 0.1, 0.2]],
                ids=[f"id_{i}"]
            )

        # Verify data isolation between collections
        for i, col in enumerate(collections):
            result = col.get(ids=[f"id_{i}"])
            assert result["documents"][0] == f"doc_{i}"

            # Should not find other collections' data
            wrong_result = col.get(ids=[f"id_{(i+1)%5}"])
            assert len(wrong_result["documents"]) == 0


@pytest.mark.integration
class TestChromaDBManagerIntegration:
    """Integration tests for ChromaDBManager with real components."""

    def test_manager_can_reopen_persistent_store_after_final_close(self, tmp_path):
        config = {
            "USER_DB_BASE_DIR": str(tmp_path),
            "embedding_config": {},
            "chroma_client_settings": {
                "backend": "persistent",
                "anonymized_telemetry": False,
                "allow_reset": True,
            },
        }
        first = ChromaDBManager(user_id="reopen_user", user_embedding_config=config)
        first.store_in_chroma(
            collection_name="reopen_collection",
            texts=["persistent document"],
            embeddings=[[0.1, 0.2, 0.3]],
            ids=["persistent-id"],
            metadatas=[{"source": "reopen-test"}],
        )
        first.close()

        second = ChromaDBManager(user_id="reopen_user", user_embedding_config=config)
        try:
            collection = second.get_collection("reopen_collection")
            result = collection.get(ids=["persistent-id"])
            assert result["documents"] == ["persistent document"]
        finally:
            second.close()

    def test_manager_initialization_with_real_chromadb(self, real_chromadb_manager):

        """Test ChromaDBManager basic initialization with real client."""
        assert real_chromadb_manager.client is not None
        collection = real_chromadb_manager.get_or_create_collection("test_collection")
        assert collection is not None

    def test_end_to_end_storage_and_retrieval(self, real_chromadb_manager, sample_texts):

        """Test complete storage and retrieval pipeline."""
        collection_name = "test_e2e"

        # Generate simple fixed-dimension embeddings (no external model)
        dim = 8
        rng = np.random.default_rng(0)
        embeddings = rng.normal(size=(len(sample_texts), dim)).tolist()

        # Store in ChromaDB
        ids = [f"doc_{i}" for i in range(len(sample_texts))]
        metadata = [{"index": i, "source": "test"} for i in range(len(sample_texts))]

        real_chromadb_manager.store_in_chroma(
            collection_name=collection_name,
            texts=sample_texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata
        )

        # Verify storage
        count = real_chromadb_manager.count_items_in_collection(collection_name)
        assert count == len(sample_texts)

        # Test retrieval with precomputed embeddings
        query_embedding = embeddings[0:1]  # Use first embedding as query
        results = real_chromadb_manager.query_collection_with_precomputed_embeddings(
            collection_name=collection_name,
            query_embeddings=query_embedding,
            n_results=3
        )

        assert len(results["ids"][0]) <= 3
        assert results["ids"][0][0] == "doc_0"  # Should find itself as most similar

    def test_vector_search_with_real_embeddings(self, real_chromadb_manager, hf_or_deterministic_embeddings):

        """Test vector search, preferring HF embeddings when online."""
        collection_name = "search_test"

        # Add test documents
        documents = [
            "The weather is sunny today",
            "Machine learning is fascinating",
            "The sun is shining brightly",
            "Deep learning transforms AI",
            "It's a beautiful sunny morning"
        ]
        # Choose embedding path: HF model if available; otherwise deterministic
        embed_func, used_real_model, dim = hf_or_deterministic_embeddings
        embeddings = embed_func(documents)

        # Store documents
        ids = [f"doc_{i}" for i in range(len(documents))]
        real_chromadb_manager.store_in_chroma(
            collection_name=collection_name,
            texts=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"i": i} for i in range(len(documents))]
        )

        # Use one of the stored vectors as query
        query_embedding = embeddings[0]

        results = real_chromadb_manager.query_collection_with_precomputed_embeddings(
            collection_name=collection_name,
            query_embeddings=[query_embedding],
            n_results=3
        )

        # Should return up to 3 documents
        found_docs = results["documents"][0]
        assert len(found_docs) == 3

    def test_metadata_filtering(self, real_chromadb_manager):

        """Test metadata filtering in searches."""
        collection_name = "metadata_test"

        # Create documents with different categories
        documents = []
        embeddings_list = []
        ids = []
        metadatas = []

        categories = ["science", "sports", "science", "sports", "science"]
        texts = [
            "Quantum physics discoveries",
            "Football match results",
            "Chemical reactions study",
            "Basketball tournament",
            "Astronomy observations"
        ]


        for i, (text, category) in enumerate(zip(texts, categories)):
            documents.append(text)
            ids.append(f"doc_{i}")
            metadatas.append({"category": category, "index": i})

        dim = 8
        rng = np.random.default_rng(123)
        embeddings_list = rng.normal(size=(len(documents), dim)).tolist()

        # Store documents
        real_chromadb_manager.store_in_chroma(
            collection_name=collection_name,
            texts=documents,
            embeddings=embeddings_list,
            ids=ids,
            metadatas=metadatas
        )

        # Search only in science category
        collection = real_chromadb_manager.get_or_create_collection(collection_name)
        results = collection.query(
            query_embeddings=[embeddings_list[0]],
            n_results=5,
            where={"category": "science"}
        )

        # Should only return science documents
        assert len(results["ids"][0]) == 3
        for metadata in results["metadatas"][0]:
            assert metadata["category"] == "science"

    def test_collection_deletion_and_recreation(self, real_chromadb_manager):

        """Test deleting and recreating collections."""
        collection_name = "delete_test"

        # Create and populate collection
        real_chromadb_manager.store_in_chroma(
            collection_name=collection_name,
            texts=["test1", "test2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            ids=["id1", "id2"],
            metadatas=[{"i": 0}, {"i": 1}]
        )

        # Verify data exists
        count = real_chromadb_manager.count_items_in_collection(collection_name)
        assert count == 2

        # Delete collection (no return in current API)
        real_chromadb_manager.delete_collection(collection_name)

        # Recreate collection
        collection = real_chromadb_manager.get_or_create_collection(collection_name)
        assert collection is not None

        # Should be empty
        new_count = real_chromadb_manager.count_items_in_collection(collection_name)
        assert new_count == 0

    def test_large_batch_processing(self, real_chromadb_manager):

        """Test processing large batches of documents."""
        collection_name = "large_batch_test"
        num_documents = 100

        # Generate large dataset
        texts = [f"Document {i}: This is test content for document number {i}"
                 for i in range(num_documents)]
        ids = [f"doc_{i}" for i in range(num_documents)]

        # Generate simple embeddings (normally would use real embedder)
        np.random.seed(42)
        embeddings = np.random.randn(num_documents, 384).tolist()

        # Store in batches
        batch_size = 20
        for i in range(0, num_documents, batch_size):
            batch_end = min(i + batch_size, num_documents)
            real_chromadb_manager.store_in_chroma(
                collection_name=collection_name,
                texts=texts[i:batch_end],
                embeddings=embeddings[i:batch_end],
                ids=ids[i:batch_end],
                metadatas=[{"i": j} for j in range(i, batch_end)]
            )

        # Verify all documents stored
        count = real_chromadb_manager.count_items_in_collection(collection_name)
        assert count == num_documents

        # Test searching in large collection
        results = real_chromadb_manager.query_collection_with_precomputed_embeddings(
            collection_name=collection_name,
            query_embeddings=[embeddings[50]],
            n_results=10
        )

        assert len(results["ids"][0]) == 10
        assert "doc_50" in results["ids"][0]  # Should find itself


@pytest.mark.unit
class TestEmbeddingGeneration:
    """Embedding generation tests that prefer HF when online, else fallback."""

    def test_embeddings_via_fixture(self, hf_or_deterministic_embeddings):

        embed, used_real_model, dim = hf_or_deterministic_embeddings
        texts = ["Hello world", "Testing embeddings", "ChromaDB integration"]
        embeddings = embed(texts)
        assert len(embeddings) == 3
        if used_real_model:
            # Strict assertions when HF model is used
            assert dim == 384
            assert len(embeddings[0]) == 384
            # Different texts should yield different vectors
            import numpy as _np
            assert not _np.allclose(embeddings[0], embeddings[1])
        else:
            # Looser checks offline / deterministic fallback
            assert len(embeddings[0]) > 0
        # Valid non-zero vectors
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert norm > 0.1

    def test_embeddings_are_deterministic(self, hf_or_deterministic_embeddings):

        embed, used_real_model, dim = hf_or_deterministic_embeddings
        texts = ["Hello world"]
        emb1 = embed(texts)[0]
        emb2 = embed(texts)[0]
        # Deterministic for the same text and backend
        if used_real_model:
            np.testing.assert_allclose(emb1, emb2, rtol=1e-4, atol=1e-5)
            assert dim == 384 and len(emb1) == 384
        else:
            np.testing.assert_allclose(emb1, emb2, rtol=1e-9, atol=1e-9)


@pytest.mark.integration
class TestDatabaseIntegration:
    """Test integration with MediaDatabase."""

    def test_media_db_with_chromadb(self, media_database, real_chromadb_manager, monkeypatch):

        """Test ChromaDB integration with MediaDatabase."""
        # Add media entry
        media_id = str(uuid.uuid4())
        content = "This is test content for ChromaDB integration"
        media_database.add_media_with_keywords(
            title="Test Document",
            content=content,
            media_type="document",
            author="Test Suite",
            keywords=["chromadb", "integration"],
        )

        # Process and store in ChromaDB
        # Directly process known content
        # Patch embedding creation to avoid external models
        from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as cdl
        monkeypatch.setattr(
            cdl,
            "create_embeddings_batch",
            lambda texts, user_app_config, model_id_override=None: [[0.1, 0.2, 0.3] for _ in texts],
        )
        real_chromadb_manager.process_and_store_content(
            content=content,
            media_id=media_id,
            file_name="test.txt",
            collection_name="media_collection"
        )

        # Verify in ChromaDB
        count = real_chromadb_manager.count_items_in_collection("media_collection")
        assert count > 0

    def test_chunk_storage_with_db_references(self, media_database, real_chromadb_manager):

        """Test storing chunks with database references."""
        media_id = str(uuid.uuid4())

        # Add media
        media_database.add_media_with_keywords(
            title="Chunked Document",
            content="First chunk content. Second chunk content. Third chunk content.",
            media_type="document",
            keywords=["test"],
        )

        # Manually chunk and store
        chunks = [
            "First chunk content.",
            "Second chunk content.",
            "Third chunk content."
        ]

        # Generate embeddings
        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(len(chunks), 8)).tolist()

        # Store with chunk metadata
        ids = [f"{media_id}_chunk_{i}" for i in range(len(chunks))]
        metadata = [
            {"media_id": media_id, "chunk_index": i, "chunk_text": chunk}
            for i, chunk in enumerate(chunks)
        ]

        real_chromadb_manager.store_in_chroma(
            collection_name="chunked_media",
            texts=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata
        )


        # Search and verify metadata
        results = real_chromadb_manager.query_collection_with_precomputed_embeddings(
            collection_name="chunked_media",
            query_embeddings=[embeddings[0]],
            n_results=1
        )

        assert results["metadatas"][0][0]["media_id"] == media_id
        assert results["metadatas"][0][0]["chunk_index"] == 0


@pytest.mark.integration
class TestConcurrentOperations:
    """Test concurrent ChromaDB operations."""

    @pytest.mark.concurrent
    def test_concurrent_writes(self, real_chromadb_manager):
        """Test concurrent write operations."""
        import concurrent.futures

        collection_name = "concurrent_test"
        num_threads = 5
        docs_per_thread = 10

        def write_documents(thread_id):

            """Write documents from a thread."""
            texts = [f"Thread {thread_id} doc {i}" for i in range(docs_per_thread)]
            ids = [f"t{thread_id}_d{i}" for i in range(docs_per_thread)]

            # Simple embeddings for testing
            embeddings = [[float(thread_id), float(i), 0.1]
                         for i in range(docs_per_thread)]

            return real_chromadb_manager.store_in_chroma(
                collection_name=collection_name,
                texts=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=[{"t": thread_id}] * len(ids)
            )

        # Execute concurrent writes
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(write_documents, i) for i in range(num_threads)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All writes should succeed
        assert all(results)

        # Verify all documents written
        count = real_chromadb_manager.count_items_in_collection(collection_name)
        assert count == num_threads * docs_per_thread

    @pytest.mark.concurrent
    def test_concurrent_reads(self, real_chromadb_manager):
        """Test concurrent read operations."""
        import concurrent.futures

        collection_name = "read_test"

        # Populate collection
        texts = [f"Document {i}" for i in range(20)]
        embeddings = [[float(i), 0.1, 0.2] for i in range(20)]
        ids = [f"doc_{i}" for i in range(20)]

        real_chromadb_manager.store_in_chroma(
            collection_name=collection_name,
            texts=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"i": i} for i in range(20)]
        )

        def search_documents(query_id):

            """Search documents from a thread."""
            query_embedding = [[float(query_id), 0.1, 0.2]]
            results = real_chromadb_manager.query_collection_with_precomputed_embeddings(
                collection_name=collection_name,
                query_embeddings=query_embedding,
                n_results=5
            )
            return len(results["ids"][0])

        # Execute concurrent reads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(search_documents, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All searches should return results
        assert all(r == 5 for r in results)


@pytest.mark.integration
class TestErrorRecovery:
    """Test error recovery in real scenarios."""

    def test_recovery_from_corrupted_collection(self, real_chromadb_manager):

        """Test recovery from corrupted collection."""
        collection_name = "recovery_test"

        # Create and populate collection
        real_chromadb_manager.store_in_chroma(
            collection_name=collection_name,
            texts=["test"],
            embeddings=[[0.1, 0.2]],
            ids=["id1"],
            metadatas=[{"i": 0}]
        )

        # Simulate corruption by directly manipulating ChromaDB
        # This is implementation-specific and might need adjustment
        collection = real_chromadb_manager.get_or_create_collection(collection_name)

        # Try to add invalid data
        with pytest.raises(Exception):
            collection.add(
                documents=["test"],
                embeddings=[[0.1]],  # Wrong dimension
                ids=["id2"]
            )

        # Should still be able to query existing data
        results = real_chromadb_manager.query_collection_with_precomputed_embeddings(
            collection_name=collection_name,
            query_embeddings=[[0.1, 0.2]],
            n_results=1
        )

        assert len(results["ids"][0]) == 1

    def test_recovery_from_connection_loss(self, temp_chroma_path):

        """Test recovery from temporary connection loss."""
        manager = ChromaDBManager(
            user_id="test_user",
            user_embedding_config={
                "USER_DB_BASE_DIR": temp_chroma_path,
                "embedding_config": {"default_model_id": "unused", "models": {}},
                "chroma_client_settings": {"anonymized_telemetry": False, "allow_reset": True},
            },
        )
        manager.persist_directory = temp_chroma_path

        # Connection already initialized in constructor

        try:
            # Simulate connection loss by setting client to None
            original_client = manager.client
            manager.client = None

            # Try operation that requires connection
            with pytest.raises(Exception):
                manager.count_items_in_collection("test")

            # Restore connection
            manager.client = original_client

            # Should work again
            collection = manager.get_or_create_collection("recovery_test")
            assert collection is not None
        finally:
            # Ensure resources are released even if assertions fail
            try:
                manager.close()
            except Exception:
                _ = None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

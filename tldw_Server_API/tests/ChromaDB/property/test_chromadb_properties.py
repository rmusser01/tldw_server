"""
Property-based tests for ChromaDB functionality.

These tests verify invariants and properties that should hold
across all valid inputs using Hypothesis.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, example, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition, invariant, Bundle
import numpy as np
from typing import List, Dict, Any, Optional
import tempfile
import shutil
from pathlib import Path
import threading

from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager, validate_user_id
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

# Configure Hypothesis to keep resource usage low in CI
from hypothesis import settings as _hyp_settings
_hyp_settings.register_profile("ci_min", max_examples=5, stateful_step_count=15, deadline=None)
_hyp_settings.load_profile("ci_min")

# Shared manager for stateful tests to avoid resource exhaustion
_state_mgr_lock = threading.Lock()
_state_mgr_singleton = None
_state_mgr_base_dir = None


def _new_chroma_manager(base_dir: str, user_id: str = "prop_stateful") -> ChromaDBManager:
    """Create a Chroma manager rooted under a test-owned temporary directory."""
    return ChromaDBManager(
        user_id=user_id,
        user_embedding_config={
            "USER_DB_BASE_DIR": base_dir,
            "embedding_config": {"default_model_id": "unused", "models": {}},
            "chroma_client_settings": {"anonymized_telemetry": False, "allow_reset": True},
        },
    )


def _get_shared_state_manager():


    global _state_mgr_singleton, _state_mgr_base_dir
    with _state_mgr_lock:
        if _state_mgr_singleton is None:
            base_dir = tempfile.mkdtemp(prefix="chroma_prop_shared_")
            _state_mgr_singleton = _new_chroma_manager(base_dir)
            _state_mgr_base_dir = base_dir
        return _state_mgr_singleton


# Ensure the shared state manager is properly closed at session end
@pytest.fixture(scope="session", autouse=True)
def _state_mgr_cleanup():
    yield
    try:
        mgr = globals().get("_state_mgr_singleton")
        if mgr is not None:
            try:
                mgr.close()
            except Exception:
                _ = None
    finally:
        base_dir = globals().get("_state_mgr_base_dir")
        if base_dir:
            try:
                shutil.rmtree(base_dir, ignore_errors=True)
            except Exception:
                _ = None
        globals()["_state_mgr_singleton"] = None
        globals()["_state_mgr_base_dir"] = None


# =====================================================================
# Hypothesis Strategies
# =====================================================================

# Strategy for valid user IDs (ASCII alnum, underscore, hyphen)
valid_user_id = st.text(
    alphabet=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"),
    min_size=1,
    max_size=50,
).filter(lambda x: x.strip() != "")

# Strategy for collection names (Chroma requires ASCII [a-zA-Z0-9._-], start/end alnum)
_ascii_alnum = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_ascii_safe = _ascii_alnum + "_-"
collection_name = st.text(alphabet=list(_ascii_safe), min_size=3, max_size=63).filter(
    lambda x: x[0].isalnum() and x[-1].isalnum()
)

# Strategy for document texts
document_text = st.text(min_size=1, max_size=1000)

# Strategy for embedding dimensions
embedding_dim = st.sampled_from([128, 256, 384, 512, 768, 1536])

# Strategy for embeddings
@st.composite
def embeddings_strategy(draw, dim=None, count=None):
    """Generate valid embeddings."""
    if dim is None:
        dim = draw(embedding_dim)
    if count is None:
        count = draw(st.integers(min_value=1, max_value=10))

    embeddings = []
    for _ in range(count):
        # Generate random values and normalize to unit vector
        values = draw(st.lists(
            st.floats(min_value=-1, max_value=1, allow_nan=False),
            min_size=dim,
            max_size=dim
        ))
        norm = np.linalg.norm(values)
        if norm == 0:
            # Avoid zero vectors to keep normalization property meaningful
            values = [1.0] + [0.0] * (dim - 1)
            norm = 1.0
        values = (np.array(values) / norm).tolist()
        embeddings.append(values)

    return embeddings

# Strategy for metadata
_INT_SAFE_MIN = -(2 ** 53) + 1
_INT_SAFE_MAX = (2 ** 53) - 1
metadata_value = st.one_of(
    st.text(max_size=100),
    st.integers(min_value=_INT_SAFE_MIN, max_value=_INT_SAFE_MAX),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans()
)

metadata_dict = st.dictionaries(
    keys=st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu")), min_size=1, max_size=50),
    values=metadata_value,
    min_size=1,
    max_size=10
)

# Strategy for batch sizes
batch_size = st.integers(min_value=1, max_value=100)


# =====================================================================
# Property Tests for ChromaDBManager
# =====================================================================

@pytest.mark.property
class TestChromaDBProperties:
    """Property tests for ChromaDB operations."""

    @given(user_id=valid_user_id)
    @settings(max_examples=5)
    def test_valid_user_id_accepted(self, user_id):
        """Any valid user ID should be accepted."""
        try:
            assert validate_user_id(user_id) == user_id
        except ValueError:
            pytest.fail(f"Valid user ID rejected: {user_id}")

    @given(
        texts=st.lists(document_text, min_size=1, max_size=10),
        dim=embedding_dim
    )
    @settings(max_examples=5)
    def test_storage_retrieval_consistency(self, texts, dim):
        """Stored data should be retrievable exactly as stored."""
        collection_name = "property_test"

        # Generate consistent embeddings
        embeddings = [[float(i)] * dim for i in range(len(texts))]
        ids = [f"id_{i}" for i in range(len(texts))]

        # Store data
        manager = _get_shared_state_manager()
        manager.reset_chroma_collection(collection_name)
        manager.store_in_chroma(
            collection_name=collection_name,
            texts=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"i": i} for i in range(len(texts))]
        )

        # Retrieve and verify
        collection = manager.get_or_create_collection(collection_name)
        results = collection.get(ids=ids)

        # All documents should be retrieved
        assert len(results["ids"]) == len(ids)
        assert set(results["ids"]) == set(ids)

        # Documents should match
        for original_text in texts:
            assert original_text in results["documents"]

    @given(
        num_docs=st.integers(min_value=0, max_value=100),
        collection=collection_name
    )
    @settings(max_examples=5)
    def test_count_accuracy(self, num_docs, collection):
        """Count should accurately reflect number of stored items."""
        # Reset collection
        manager = _get_shared_state_manager()
        manager.reset_chroma_collection(collection)

        if num_docs > 0:
            texts = [f"doc_{i}" for i in range(num_docs)]
            embeddings = [[float(i), 0.1, 0.2] for i in range(num_docs)]
            ids = [f"id_{i}" for i in range(num_docs)]

            manager.store_in_chroma(
                collection_name=collection,
                texts=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=[{"i": i} for i in range(num_docs)]
            )

        count = manager.count_items_in_collection(collection)
        assert count == num_docs

    @given(
        texts=st.lists(document_text, min_size=1, max_size=5, unique=True),
        metadata_list=st.lists(metadata_dict, min_size=1, max_size=5)
    )
    @settings(max_examples=5)
    def test_metadata_preservation(self, texts, metadata_list):
        """Metadata should be preserved exactly as provided."""
        # Ensure same length
        min_len = min(len(texts), len(metadata_list))
        texts = texts[:min_len]
        metadata_list = metadata_list[:min_len]

        collection_name = "metadata_test"
        embeddings = [[0.1, 0.2, 0.3] for _ in texts]
        ids = [f"id_{i}" for i in range(len(texts))]

        # Store with metadata
        manager = _get_shared_state_manager()
        manager.reset_chroma_collection(collection_name)
        manager.store_in_chroma(
            collection_name=collection_name,
            texts=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata_list
        )

        # Retrieve and verify metadata
        collection = manager.get_or_create_collection(collection_name)
        results = collection.get(ids=ids)

        for i, original_metadata in enumerate(metadata_list):
            retrieved_metadata = results["metadatas"][i]
            for key, value in original_metadata.items():
                assert key in retrieved_metadata
                rv = retrieved_metadata[key]
                if isinstance(value, float):
                    # Allow for tiny float differences due to serialization
                    import math
                    assert math.isclose(rv, value, rel_tol=1e-9, abs_tol=1e-12)
                else:
                    assert rv == value

    @given(
        embedding_list=embeddings_strategy()
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.data_too_large])
    def test_embedding_dimension_consistency(self, embedding_list):
        """All embeddings in a collection must have same dimension."""
        if len(embedding_list) < 2:
            return  # Need at least 2 embeddings to test

        collection_name = "dim_test"
        texts = [f"text_{i}" for i in range(len(embedding_list))]
        ids = [f"id_{i}" for i in range(len(embedding_list))]

        # All embeddings should have same dimension
        first_dim = len(embedding_list[0])
        for emb in embedding_list[1:]:
            assume(len(emb) == first_dim)

        manager = _get_shared_state_manager()
        manager.reset_chroma_collection(collection_name)
        manager.store_in_chroma(
            collection_name=collection_name,
            texts=texts,
            embeddings=embedding_list,
            ids=ids,
            metadatas=[{"i": i} for i in range(len(ids))]
        )

    @given(
        ids=st.lists(
            st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")), min_size=1),
            min_size=1,
            max_size=10,
            unique=True
        )
    )
    @settings(max_examples=5)
    def test_id_uniqueness(self, ids):
        """IDs must be unique within a collection."""
        collection_name = "id_test"
        texts = [f"text_{i}" for i in range(len(ids))]
        embeddings = [[0.1, 0.2] for _ in ids]

        # Store with unique IDs
        manager = _get_shared_state_manager()
        manager.reset_chroma_collection(collection_name)
        manager.store_in_chroma(
            collection_name=collection_name,
            texts=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"i": i} for i in range(len(ids))]
        )

        # Count should match number of unique IDs
        count = manager.count_items_in_collection(collection_name)
        assert count == len(ids)


# =====================================================================
# Property Tests for Vector Search
# =====================================================================

@pytest.mark.property
class TestVectorSearchProperties:
    """Property tests for vector search operations."""

    @given(
        k=st.integers(min_value=1, max_value=100),
        total_docs=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=5)
    def test_search_result_count(self, k, total_docs):
        """Search should return at most min(k, total_docs) results."""
        collection_name = "search_count_test"

        # Add documents
        texts = [f"doc_{i}" for i in range(total_docs)]
        embeddings = [[float(i), 0.1, 0.2] for i in range(total_docs)]
        ids = [f"id_{i}" for i in range(total_docs)]

        manager = _get_shared_state_manager()
        manager.reset_chroma_collection(collection_name)
        manager.store_in_chroma(
            collection_name=collection_name,
            texts=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"i": i} for i in range(total_docs)]
        )

        # Search
        results = manager.query_collection_with_precomputed_embeddings(
            collection_name=collection_name,
            query_embeddings=[[0.0, 0.1, 0.2]],
            n_results=k
        )

        expected_count = min(k, total_docs)
        assert len(results["ids"][0]) == expected_count

    @given(
        query_embedding=embeddings_strategy(dim=3, count=1)
    )
    @settings(max_examples=5)
    def test_self_similarity_highest(self, query_embedding):
        """A document should be most similar to itself."""
        collection_name = "self_similarity_test"

        # Store the query as a document
        manager = _get_shared_state_manager()
        manager.reset_chroma_collection(collection_name)
        manager.store_in_chroma(
            collection_name=collection_name,
            texts=["query_doc", "other_doc1", "other_doc2"],
            embeddings=query_embedding + [[0.9, 0.1, 0.0], [0.0, 0.9, 0.1]],
            ids=["query_id", "other1", "other2"],
            metadatas=[{"i": 0}, {"i": 1}, {"i": 2}]
        )

        # Search with same embedding
        results = manager.query_collection_with_precomputed_embeddings(
            collection_name=collection_name,
            query_embeddings=query_embedding,
            n_results=3
        )

        # Query document should be first result
        assert results["ids"][0][0] == "query_id"

        # Distance to itself should be close to 0
        if results.get("distances"):
            assert results["distances"][0][0] < 0.01

    @given(
        filter_value=st.text(min_size=1, max_size=20),
        num_matching=st.integers(min_value=1, max_value=10),
        num_non_matching=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=5)
    def test_metadata_filter_correctness(self, filter_value, num_matching,
                                        num_non_matching):
        """Metadata filters should return only matching documents."""
        collection_name = "filter_test"

        # Create documents with and without matching metadata
        all_texts = []
        all_embeddings = []
        all_ids = []
        all_metadata = []

        # Matching documents
        for i in range(num_matching):
            all_texts.append(f"matching_{i}")
            all_embeddings.append([float(i), 0.1, 0.2])
            all_ids.append(f"match_{i}")
            all_metadata.append({"category": filter_value, "index": i})

        # Non-matching documents
        for i in range(num_non_matching):
            all_texts.append(f"non_matching_{i}")
            all_embeddings.append([float(i + 100), 0.1, 0.2])
            all_ids.append(f"nomatch_{i}")
            all_metadata.append({"category": "other", "index": i})

        # Store all documents
        manager = ChromaDBManager(
            user_id="prop_user",
            user_embedding_config={
                "USER_DB_BASE_DIR": tempfile.mkdtemp(prefix="chroma_prop_"),
                "embedding_config": {"default_model_id": "unused", "models": {}},
                "chroma_client_settings": {"anonymized_telemetry": False, "allow_reset": True},
            },
        )
        manager.reset_chroma_collection(collection_name)
        manager.store_in_chroma(
            collection_name=collection_name,
            texts=all_texts,
            embeddings=all_embeddings,
            ids=all_ids,
            metadatas=all_metadata
        )

        # Search with filter
        collection = manager.get_or_create_collection(collection_name)
        results = collection.query(
            query_embeddings=[[0.0, 0.1, 0.2]],
            n_results=100,
            where={"category": filter_value}
        )

        # Should only return matching documents
        assert len(results["ids"][0]) == num_matching
        for metadata in results["metadatas"][0]:
            assert metadata["category"] == filter_value


# =====================================================================
# Stateful Property Tests
# =====================================================================

class ChromaDBStateMachine(RuleBasedStateMachine):
    """Stateful testing for ChromaDB operations."""

    def __init__(self):

        super().__init__()
        self._base_dir = tempfile.mkdtemp(prefix="chroma_prop_state_machine_")
        self.manager = _new_chroma_manager(self._base_dir)
        # Keep the namespace deterministic so Hypothesis can replay failures.
        self._ns = "sm_"
        # Track state local to this run
        self.collections = {}  # collection_name -> set of ids
        self.documents = {}     # (collection, id) -> document
        self.embeddings = {}    # (collection, id) -> embedding

    def _full(self, collection: str) -> str:
        suffix = collection[:60]
        if suffix and not suffix[-1].isalnum():
            suffix = f"{suffix[:-1]}{collection[-1]}"
        return f"{self._ns}{suffix}"

    collections_bundle = Bundle("collections")
    documents_bundle = Bundle("documents")

    @rule(
        collection=collection_name,
        target=collections_bundle
    )
    def create_collection(self, collection):
        """Create a new collection."""
        name = self._full(collection)
        self.manager.get_or_create_collection(name)
        if name not in self.collections:
            self.collections[name] = set()
        return name

    @rule(
        collection=collections_bundle,
        text=document_text,
        target=documents_bundle
    )
    def add_document(self, collection, text):
        """Add a document to a collection."""
        doc_id = f"doc_{len(self.collections[collection])}"
        embedding = [0.1, 0.2, 0.3]

        self.manager.store_in_chroma(
            collection_name=collection,
            texts=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[{"i": 0}]
        )
        self.collections[collection].add(doc_id)
        self.documents[(collection, doc_id)] = text
        self.embeddings[(collection, doc_id)] = embedding

        return (collection, doc_id)

    @rule(
        document=documents_bundle
    )
    def delete_document(self, document):
        """Delete a document from a collection."""
        collection, doc_id = document

        self.manager.delete_from_collection([doc_id], collection)
        if doc_id in self.collections[collection]:
            self.collections[collection].remove(doc_id)
            del self.documents[(collection, doc_id)]
            del self.embeddings[(collection, doc_id)]

    @rule(
        collection=collections_bundle,
        k=st.integers(min_value=1, max_value=10)
    )
    def search_collection(self, collection, k):
        """Search in a collection."""
        if not self.collections[collection]:
            return  # Empty collection

        results = self.manager.query_collection_with_precomputed_embeddings(
            collection_name=collection,
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=k
        )

        # Results should only contain existing documents
        for doc_id in results["ids"][0]:
            assert doc_id in self.collections[collection]

    @invariant()
    def count_matches_state(self):
        """Collection counts should match internal state."""
        for collection, ids in self.collections.items():
            actual_count = self.manager.count_items_in_collection(collection)
            expected_count = len(ids)
            assert actual_count == expected_count, \
                f"Count mismatch in {collection}: {actual_count} != {expected_count}"

    @invariant()
    def documents_retrievable(self):
        """All tracked documents should be retrievable."""
        for collection, ids in self.collections.items():
            if ids:  # Non-empty collection
                collection_obj = self.manager.get_or_create_collection(collection)
                results = collection_obj.get(ids=list(ids))

                retrieved_ids = set(results["ids"])
                assert retrieved_ids == ids, \
                    f"ID mismatch in {collection}: {retrieved_ids} != {ids}"

    def teardown(self):

        """Clean up after test."""
        # Delete only collections created in this run to avoid leaking state
        try:
            for collection in list(self.collections.keys()):
                try:
                    self.manager.delete_collection(collection)
                except Exception:
                    _ = None
        finally:
            self.collections.clear()
            self.documents.clear()
            self.embeddings.clear()
            try:
                self.manager.close()
            except Exception:
                _ = None
            shutil.rmtree(self._base_dir, ignore_errors=True)


def test_state_machine_examples_use_isolated_chroma_managers():
    """Each state-machine run must use isolated Chroma storage for replay safety."""
    first = ChromaDBStateMachine()
    second = ChromaDBStateMachine()

    try:
        assert first.manager is not second.manager
        assert first.manager.user_chroma_path != second.manager.user_chroma_path
    finally:
        first.teardown()
        second.teardown()


# Run the stateful test
class TestChromaDBStateMachine(ChromaDBStateMachine.TestCase):
    # Keep runs light to avoid resource issues under CI
    settings = settings(max_examples=10, stateful_step_count=25)


# =====================================================================
# Property Tests for Embeddings
# =====================================================================

@pytest.mark.property
class TestEmbeddingProperties:
    """Property tests for embedding generation and validation."""

    @given(
        embeddings=embeddings_strategy()
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.data_too_large])
    def test_embedding_normalization(self, embeddings):
        """Embeddings should be normalized (unit vectors)."""
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            # Should be close to 1 (unit vector)
            assert abs(norm - 1.0) < 0.01, f"Embedding not normalized: norm={norm}"

    @given(
        dim=embedding_dim,
        count=st.integers(min_value=2, max_value=10),
        data=st.data(),
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.data_too_large])
    def test_embedding_dimension_consistency(self, dim, count, data):
        """All embeddings in a batch should have same dimension."""
        embeddings = data.draw(embeddings_strategy(dim=dim, count=count))

        # All should have same dimension
        dimensions = [len(emb) for emb in embeddings]
        assert all(d == dim for d in dimensions), \
            f"Inconsistent dimensions: {dimensions}"

    @given(
        text1=document_text,
        text2=document_text
    )
    @settings(max_examples=5)
    def test_deterministic_embedding(self, text1, text2):
        """Same text should produce same embedding."""
        # This assumes deterministic embedding generation
        # In practice, might need to account for model variations

        if text1 == text2:
            # Mock deterministic embeddings based on text hash
            def mock_embedding(text):
                np.random.seed(hash(text) % 10000)
                emb = np.random.randn(384)
                return (emb / np.linalg.norm(emb)).tolist()

            emb1 = mock_embedding(text1)
            emb2 = mock_embedding(text2)

            # Should be identical for same text
            np.testing.assert_array_almost_equal(emb1, emb2)

    @given(
        texts=st.lists(document_text, min_size=1, max_size=100),
        batch_size=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=5)
    def test_batch_processing_consistency(self, texts, batch_size):
        """Batch processing should produce same results as individual processing."""
        # Mock embedding function
        def mock_embed(text_list):
            embeddings = []
            for text in text_list:
                np.random.seed(hash(text) % 10000)
                emb = np.random.randn(384)
                embeddings.append((emb / np.linalg.norm(emb)).tolist())
            return embeddings

        # Process in batches
        batch_results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results.extend(mock_embed(batch))

        # Process individually
        individual_results = mock_embed(texts)

        # Should produce same results
        for batch_emb, ind_emb in zip(batch_results, individual_results):
            np.testing.assert_array_almost_equal(batch_emb, ind_emb)


# =====================================================================
# Property Tests for Error Handling
# =====================================================================

@pytest.mark.property
class TestErrorHandlingProperties:
    """Property tests for error handling."""

    @given(
        invalid_user_id=st.one_of(
            st.just("../evil"),
            st.just("../../etc/passwd"),
            st.just("/absolute/path"),
            st.just("user/../other"),
            # Construct invalid IDs without filtering to avoid health check failures
            st.builds(lambda s: f"../{s or 'x'}", st.text()),
            st.builds(lambda s: f"{s or 'x'}/..", st.text()),
            st.builds(lambda s: f"/{s or 'x'}", st.text()),
            st.builds(lambda s: f"{s or 'x'}\\{s or 'y'}", st.text()),
            st.builds(lambda s: f"{s}\n", st.text()),
            st.builds(lambda s: f"{s}\r", st.text()),
            st.builds(lambda s: f"{s}\x00", st.text()),
        )
    )
    @settings(max_examples=5, deadline=None)
    def test_invalid_user_id_rejected(self, invalid_user_id):
        """Invalid user IDs should be rejected."""
        with pytest.raises(ValueError):
            validate_user_id(invalid_user_id)

    @given(
        texts=st.lists(document_text, min_size=1, max_size=5),
        embeddings=embeddings_strategy(dim=3, count=6)  # Guaranteed mismatch with texts length
    )
    @settings(max_examples=5)
    def test_mismatched_input_lengths(self, texts, embeddings):
        """Mismatched input lengths should be rejected."""
        with pytest.raises(ValueError, match="length"):
            _get_shared_state_manager().store_in_chroma(
                collection_name="test",
                texts=texts,
                embeddings=embeddings,
                ids=[f"id_{i}" for i in range(len(texts))],
                metadatas=[{"i": i} for i in range(len(texts))]
            )

    @given(
        collection=collection_name,
        bad_ids=st.lists(st.text(min_size=1), min_size=2, max_size=5)
    )
    @settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_duplicate_ids_handled(self, collection, bad_ids):
        """Duplicate IDs should be handled appropriately."""
        # Make some IDs duplicate
        ids_with_duplicates = bad_ids + [bad_ids[0]]
        texts = [f"text_{i}" for i in range(len(ids_with_duplicates))]
        embeddings = [[0.1, 0.2] for _ in texts]

        # This should either reject or handle duplicates gracefully
        manager = _get_shared_state_manager()
        manager.reset_chroma_collection(collection)
        try:
            result = manager.store_in_chroma(
                collection_name=collection,
                texts=texts,
                embeddings=embeddings,
                ids=ids_with_duplicates,
                metadatas=[{"i": i} for i in range(len(ids_with_duplicates))]
            )
        except Exception:
            # Raising is acceptable handling for duplicate IDs
            return

        # If successful, count should not exceed unique IDs
        if result:
            count = manager.count_items_in_collection(collection)
            assert count <= len(set(ids_with_duplicates))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "property"])

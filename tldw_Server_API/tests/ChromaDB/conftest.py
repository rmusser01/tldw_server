"""
ChromaDB Test Configuration and Fixtures

This module provides comprehensive fixtures for testing ChromaDB functionality,
including both unit tests (with mocking) and integration tests (with real ChromaDB).
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Any, Generator, Optional
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import json
import sqlite3
from queue import Queue

import pytest
import chromadb
from chromadb.config import Settings
try:
    from chromadb.api import API  # legacy export
except Exception:
    # Create a minimal stub to satisfy imports; tests patch chroma client directly
    class API:  # type: ignore
        pass
import numpy as np
import sys
import importlib
# Provide alias for legacy import path expected by tests
try:
    sys.modules['tldw_Server_API.app.core.Embeddings.Embeddings_Create']
except KeyError:
    _mod = importlib.import_module('tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create')
    sys.modules['tldw_Server_API.app.core.Embeddings.Embeddings_Create'] = _mod

# Import the MediaDatabase class for proper database creation
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

# Import the modules we're testing
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
    create_embeddings_batch,
)
from tldw_Server_API.app.core.Embeddings.connection_pool import ConnectionPool
from tldw_Server_API.app.core.Infrastructure.circuit_breaker import CircuitBreaker
from tldw_Server_API.app.core.Embeddings.error_recovery import ErrorRecoveryManager
# Legacy Embeddings audit logger was removed. Provide a minimal stub to satisfy tests
class AuditLogger:  # type: ignore
    def __init__(self, *args, **kwargs):
        pass
    def log_event(self, *args, **kwargs):
        return None
    def log_security_event(self, *args, **kwargs):
        return None
    def log_resource_event(self, *args, **kwargs):
        return None
    def log_admin_operation(self, *args, **kwargs):
        return None

# Ensure unit tests in this package use the patched PersistentClient path,
# not the internal in-memory stub forced via env in other suites.
@pytest.fixture(autouse=True)
def _disable_chromadb_force_stub_for_chromadb_unit_tests(monkeypatch):
    """Unset CHROMADB_FORCE_STUB so ChromaDBManager uses the patched client.

    Some other test packages enable CHROMADB_FORCE_STUB globally; that causes
    ChromaDBManager to bypass chromadb.PersistentClient, defeating our mocks.
    Unsetting here ensures unit tests exercise the expected client path.
    """
    monkeypatch.delenv("CHROMADB_FORCE_STUB", raising=False)
    yield


@pytest.fixture(autouse=True)
def _clear_chromadb_persistent_client_cache():
    """Isolate process-level persistent client reuse across ChromaDB tests."""
    from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as cdl

    cache_lock = getattr(cdl, "_PERSISTENT_CLIENTS_LOCK", None)
    cache = getattr(cdl, "_PERSISTENT_CLIENTS", None)
    if cache_lock is not None and cache is not None:
        with cache_lock:
            cache.clear()
    yield
    if cache_lock is not None and cache is not None:
        with cache_lock:
            cache.clear()

# =====================================================================
# Test Markers
# =====================================================================

def pytest_configure(config):

    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "unit: Unit tests with mocking")
    config.addinivalue_line("markers", "integration: Integration tests with real ChromaDB")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "slow: Tests that take > 1 second")
    config.addinivalue_line("markers", "requires_model: Tests that need ML models")
    config.addinivalue_line("markers", "concurrent: Tests with concurrent operations")
    # Register collection-level skipping for unstable suites
    config.addinivalue_line("markers", "legacy_skip: Skip tests targeting legacy behaviors")


def pytest_addoption(parser):


    """Add CLI option to run external model tests."""
    parser.addoption(
        "--run-model-tests",
        action="store_true",
        default=False,
        help="Run tests marked with requires_model (downloads lightweight HF model)",
    )

def pytest_collection_modifyitems(config, items):

    """Conditionally skip tests requiring external models based on flag/env."""
    run_models_flag = config.getoption("--run-model-tests") or os.getenv("RUN_MODEL_TESTS") == "1"
    for item in items:
        if item.get_closest_marker("requires_model") and not run_models_flag:
            item.add_marker(pytest.mark.skip(reason="Requires external model; set RUN_MODEL_TESTS=1 or --run-model-tests to enable"))

# =====================================================================
# Database Fixtures using MediaDatabase
# =====================================================================

@pytest.fixture
def temp_media_db():
    """Create a temporary MediaDatabase instance for testing."""
    # Create temporary directory for database
    db_dir = tempfile.mkdtemp()
    db_path = os.path.join(db_dir, "test_media.db")

    # Initialize MediaDatabase with test client ID
    db = MediaDatabase(
        db_path=db_path,
        client_id="test_client"
    )

    # Initialize the database schema
    db.initialize_db()

    yield db_path

    # Cleanup
    try:
        db.close_connection()
    except:
        _ = None
    shutil.rmtree(db_dir, ignore_errors=True)

@pytest.fixture
def media_database(temp_media_db):
    """Create a MediaDatabase instance for testing."""
    db = MediaDatabase(
        db_path=temp_media_db,
        client_id="test_client"
    )
    yield db
    db.close_connection()

# =====================================================================
# ChromaDB Fixtures
# =====================================================================

@pytest.fixture
def temp_chroma_path():
    """Create a temporary directory for ChromaDB storage."""
    temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def chroma_client(temp_chroma_path):
    """Create a ChromaDB client for integration tests."""
    settings = Settings(
        persist_directory=temp_chroma_path,
        anonymized_telemetry=False,
        allow_reset=True
    )
    client = chromadb.PersistentClient(path=temp_chroma_path, settings=settings)
    try:
        yield client
    finally:
        # Best-effort shutdown to release file descriptors
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

@pytest.fixture
def mock_chroma_client():
    """Create a mock ChromaDB client for unit tests."""
    mock_client = MagicMock()
    mock_collection = MagicMock()

    # Setup collection behavior
    mock_collection.count.return_value = 0
    mock_collection.add.return_value = None
    mock_collection.upsert.return_value = None
    mock_collection.query.return_value = {
        "ids": [["test_id_1", "test_id_2"]],
        "embeddings": None,
        "documents": [["Test document 1", "Test document 2"]],
        "metadatas": [[{"source": "test"}, {"source": "test"}]],
        "distances": [[0.1, 0.2]]
    }
    mock_collection.delete.return_value = None
    mock_collection.get.return_value = {
        "ids": ["test_id_1"],
        "embeddings": None,
        "documents": ["Test document 1"],
        "metadatas": [{"source": "test"}]
    }

    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.list_collections.return_value = []
    mock_client.delete_collection.return_value = None
    mock_client.reset.return_value = None

    return mock_client

@pytest.fixture
def chromadb_manager(mock_chroma_client, temp_media_db):
    """Create a ChromaDBManager instance with mocked dependencies (constructor injection)."""
    base_dir = tempfile.mkdtemp(prefix="chroma_user_base_")
    manager = ChromaDBManager(
        user_id="test_user",
        user_embedding_config={
            "USER_DB_BASE_DIR": base_dir,
            "embedding_config": {
                "default_model_id": "text-embedding-ada-002",
                "models": {
                    "text-embedding-ada-002": {
                        "provider": "openai",
                        "model_name_or_path": "text-embedding-ada-002"
                    }
                }
            },
            # Ensure no accidental persistent client usage in this unit fixture
            "chroma_client_settings": {"backend": "stub"},
        },
        client=mock_chroma_client,
    )
    manager.db_path = temp_media_db
    yield manager

@pytest.fixture
def real_chromadb_manager(chroma_client, temp_media_db, temp_chroma_path):
    """Create a ChromaDBManager instance with real ChromaDB for integration tests."""
    manager = ChromaDBManager(
        user_id="test_user",
        user_embedding_config={
            "USER_DB_BASE_DIR": temp_chroma_path,
            "embedding_config": {"default_model_id": "unused", "models": {}},
            "chroma_client_settings": {"anonymized_telemetry": False, "allow_reset": True},
        },
    )
    # Use the provided client to share the same persistent dir
    manager.client = chroma_client
    manager.db_path = temp_media_db
    try:
        yield manager
    finally:
        try:
            manager.close()
        except Exception:
            _ = None

# =====================================================================
# Embedding Fixtures
# =====================================================================

@pytest.fixture
def mock_embeddings():
    """Generate mock embeddings for testing."""
    def _create_embeddings(texts: List[str], dim: int = 384) -> List[List[float]]:
        """Create deterministic mock embeddings based on text content."""
        embeddings = []
        for text in texts:
            # Create a deterministic embedding based on text hash
            seed = hash(text) % 10000
            np.random.seed(seed)
            embedding = np.random.randn(dim).tolist()
            # Normalize to unit vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (np.array(embedding) / norm).tolist()
            embeddings.append(embedding)
        return embeddings
    return _create_embeddings

@pytest.fixture
def mock_embedding_provider(mock_embeddings):
    """Mock embedding provider for unit tests."""
    mock_provider = MagicMock()
    mock_provider.create_embeddings = Mock(side_effect=lambda texts: mock_embeddings(texts))
    mock_provider.embedding_dimension = 384
    mock_provider.max_batch_size = 100
    return mock_provider

def _hf_connectivity_ok() -> bool:

    """Best-effort online check to Hugging Face Hub."""
    try:
        import requests  # Local import to avoid hard dependency if unused
        resp = requests.head(
            "https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2",
            timeout=2.0,
            allow_redirects=True,
        )
        return resp.status_code < 400
    except Exception:
        return False


@pytest.fixture(scope="session")
def hf_or_deterministic_embeddings(pytestconfig):
    """
    Provide an embedding function that uses a lightweight HF model if internet
    is available; otherwise returns deterministic numeric embeddings.
    Returns a tuple (embed_func, is_real_model: bool, dim: int).
    """
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
        get_embedding_config,
        create_embeddings_batch as _create_batch,
    )

    hf_model_id = "sentence-transformers/all-MiniLM-L6-v2"
    run_models = pytestconfig.getoption("--run-model-tests") or os.getenv("RUN_MODEL_TESTS") == "1"
    is_online = run_models and _hf_connectivity_ok()

    if is_online:
        try:
            cfg = get_embedding_config()
            probe = _create_batch(["probe"], user_app_config=cfg, model_id_override=hf_model_id)
            if probe and isinstance(probe[0], list) and len(probe[0]) > 0:
                dim = len(probe[0])

                def _embed(texts: List[str]) -> List[List[float]]:
                    return _create_batch(texts, user_app_config=cfg, model_id_override=hf_model_id)

                return _embed, True, dim
        except Exception:
            _ = None  # Fall through to deterministic path

    def _det_embed(texts: List[str], dim: int = 384) -> List[List[float]]:
        vecs: List[List[float]] = []
        for t in texts:
            seed = hash(t) % 10_000
            np.random.seed(seed)
            v = np.random.randn(dim)
            v = v / (np.linalg.norm(v) + 1e-12)
            vecs.append(v.tolist())
        return vecs

    return (lambda texts: _det_embed(texts, 384)), False, 384

# =====================================================================
# Error Recovery Fixtures
# =====================================================================

@pytest.fixture
def circuit_breaker():
    """Create a circuit breaker for testing."""
    return CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=1.0,
        half_open_attempts=1
    )

@pytest.fixture
def error_recovery_manager():
    """Create an error recovery manager for testing."""
    return ErrorRecoveryManager(
        max_retries=3,
        base_delay=0.1,
        max_delay=1.0
    )

# =====================================================================
# Test Data Fixtures
# =====================================================================

@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we process information.",
        "ChromaDB provides efficient vector similarity search.",
        "Testing is essential for maintaining code quality.",
        "Python is a versatile programming language."
    ]

@pytest.fixture
def sample_documents():
    """Sample documents with metadata for testing."""
    return [
        {
            "id": "doc_1",
            "text": "Introduction to vector databases and their applications.",
            "metadata": {"source": "tutorial", "chapter": 1, "author": "TestAuthor"}
        },
        {
            "id": "doc_2",
            "text": "ChromaDB architecture and performance optimization.",
            "metadata": {"source": "documentation", "version": "1.0", "tags": ["chromadb", "performance"]}
        },
        {
            "id": "doc_3",
            "text": "Best practices for embedding generation and storage.",
            "metadata": {"source": "guide", "difficulty": "intermediate", "updated": "2024-01-01"}
        }
    ]

@pytest.fixture
def sample_media_content():
    """Sample media content for end-to-end testing."""
    return {
        "media_id": str(uuid.uuid4()),
        "title": "Test Media Content",
        "content": """This is a comprehensive test document for ChromaDB integration.
        It contains multiple paragraphs to test chunking functionality.

        The document includes various topics to test semantic search:
        - Vector databases and their importance
        - Machine learning applications
        - Natural language processing
        - Information retrieval systems

        This content should be properly chunked and embedded for testing.""",
        "author": "Test Suite",
        "media_type": "document",
        "metadata": {
            "test": True,
            "version": "1.0",
            "tags": ["test", "chromadb", "embeddings"]
        }
    }

# =====================================================================
# Connection Pool Fixtures
# =====================================================================

@pytest.fixture
def connection_pool(temp_media_db):
    """Create a connection pool for testing."""
    pool = ConnectionPool(
        db_path=temp_media_db,
        min_connections=1,
        max_connections=5,
        timeout=1.0
    )
    yield pool
    pool.close()

# =====================================================================
# Audit and Metrics Fixtures
# =====================================================================

@pytest.fixture
def audit_logger(temp_media_db):
    """Create an audit logger for testing (stubbed)."""
    return AuditLogger()

@pytest.fixture
def mock_metrics():
    """Mock metrics collector for testing."""
    mock_metrics = MagicMock()
    mock_metrics.record_embedding_generation.return_value = None
    mock_metrics.record_search_query.return_value = None
    mock_metrics.record_storage_operation.return_value = None
    mock_metrics.record_error.return_value = None
    return mock_metrics

# =====================================================================
# Configuration Fixtures
# =====================================================================

@pytest.fixture
def test_config(tmp_path):
    """Test configuration for ChromaDB module."""
    return {
        "chroma_base_path": str(tmp_path / "chroma_test"),
        "embedding_providers": {
            "openai": {
                "api_key": "test_key",
                "model": "text-embedding-ada-002",
                "dimension": 1536
            },
            "huggingface": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu",
                "dimension": 384
            }
        },
        "cache_config": {
            "enabled": True,
            "ttl": 3600,
            "max_size": 1000
        },
        "resource_limits": {
            "max_memory_gb": 2,
            "max_collections_per_user": 10,
            "max_items_per_collection": 100000
        },
        "security": {
            "validate_user_ids": True,
            "sanitize_inputs": True,
            "audit_logging": True
        }
    }

# =====================================================================
# Utility Functions
# =====================================================================

@pytest.fixture
def create_test_collection():
    """Factory fixture for creating test collections."""
    def _create(client, name="test_collection", metadata=None):
        """Create a test collection with optional metadata."""
        return client.get_or_create_collection(
            name=name,
            metadata=metadata or {"test": True}
        )
    return _create

@pytest.fixture
def populate_collection():
    """Factory fixture for populating collections with test data."""
    def _populate(collection, num_items=10, embedding_dim=384):
        """Populate a collection with test data."""
        ids = [f"item_{i}" for i in range(num_items)]
        texts = [f"Test document {i} with some content" for i in range(num_items)]
        embeddings = np.random.randn(num_items, embedding_dim).tolist()
        metadatas = [{"index": i, "test": True} for i in range(num_items)]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return ids, texts, embeddings, metadatas
    return _populate

@pytest.fixture
def assert_embeddings_similar():
    """Utility to assert embeddings are similar within tolerance."""
    def _assert(emb1: List[float], emb2: List[float], tolerance: float = 0.01):
        """Assert two embeddings are similar."""
        assert len(emb1) == len(emb2), f"Embedding dimensions don't match: {len(emb1)} vs {len(emb2)}"

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5

        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
            assert similarity > (1 - tolerance), f"Embeddings not similar enough: {similarity}"
    return _assert

# =====================================================================
# Cleanup and Resource Management
# =====================================================================

@pytest.fixture(autouse=True)
def cleanup_environment():
    """Automatically cleanup test environment after each test."""
    yield
    # Cleanup any temporary files or processes
    import gc
    gc.collect()

@pytest.fixture(scope="session")
def test_session_cleanup():
    """Session-level cleanup for all ChromaDB tests."""
    yield
    # Final cleanup
    temp_dirs = Path(tempfile.gettempdir()).glob("chroma_test_*")
    for temp_dir in temp_dirs:
        try:
            shutil.rmtree(temp_dir)
        except:
            _ = None

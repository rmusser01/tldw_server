import builtins
import sqlite3

import pytest

from tldw_Server_API.app.core.RAG.rag_service.health_check import (
    HealthStatus,
    RAGHealthChecker,
)


pytestmark = pytest.mark.unit


class _FailingVectorStore:
    async def list_collections(self):
        raise RuntimeError("vector store failed at /private/vector-store.db")


@pytest.mark.asyncio
async def test_vector_store_health_check_sanitizes_backend_failures():
    checker = RAGHealthChecker(vector_store=_FailingVectorStore())

    health = await checker.check_vector_store()

    assert health.status is HealthStatus.UNHEALTHY
    assert health.message == "Vector store connection failed"
    assert "vector store failed" not in health.message
    assert "/private/vector-store.db" not in health.message


@pytest.mark.asyncio
async def test_database_health_check_sanitizes_connection_failures(monkeypatch):
    def fail_connect(*_args, **_kwargs):
        raise sqlite3.Error("database opened /private/rag-health.db failed")

    monkeypatch.setattr(sqlite3, "connect", fail_connect)
    checker = RAGHealthChecker()

    health = await checker.check_database()

    assert health.status is HealthStatus.UNHEALTHY
    assert health.message == "Database connection failed"
    assert "database opened" not in health.message
    assert "/private/rag-health.db" not in health.message


@pytest.mark.asyncio
async def test_search_index_health_check_sanitizes_connection_failures(monkeypatch):
    def fail_connect(*_args, **_kwargs):
        raise sqlite3.Error("search index opened /private/rag-index.db failed")

    monkeypatch.setattr(sqlite3, "connect", fail_connect)
    checker = RAGHealthChecker()

    health = await checker.check_search_index()

    assert health.status is HealthStatus.UNHEALTHY
    assert health.message == "Search index check failed"
    assert "search index opened" not in health.message
    assert "/private/rag-index.db" not in health.message


@pytest.mark.asyncio
async def test_embedding_service_health_check_sanitizes_backend_failures(monkeypatch):
    original_import = builtins.__import__

    def fail_jobs_worker_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            "Embeddings.services" in name
            and "jobs_worker" in fromlist
        ) or name.endswith("Embeddings.services.jobs_worker"):
            raise RuntimeError("embedding worker failed at /private/embeddings.db")
        if (
            level > 0
            and name == "Embeddings.services"
            and "jobs_worker" in fromlist
        ):
            raise RuntimeError("embedding worker failed at /private/embeddings.db")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_jobs_worker_import)
    checker = RAGHealthChecker()

    health = await checker.check_embedding_service()

    assert health.status is HealthStatus.UNHEALTHY
    assert health.message == "Embedding service check failed"
    assert "embedding worker failed" not in health.message
    assert "/private/embeddings.db" not in health.message

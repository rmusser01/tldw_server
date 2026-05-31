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


class _LoggerStub:
    def __init__(self):
        self.debugs = []
        self.errors = []

    def debug(self, message, *args, **kwargs):
        self.debugs.append(str(message))

    def error(self, message, *args, **kwargs):
        self.errors.append(str(message))


def _assert_sanitized(messages):
    rendered = "\n".join(messages)
    assert "/private/" not in rendered
    assert "failed at" not in rendered
    assert "opened" not in rendered
    assert "secret-token" not in rendered


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


@pytest.mark.asyncio
async def test_vector_store_health_check_sanitizes_backend_failure_log(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.RAG.rag_service.health_check.logger",
        logger_stub,
    )
    checker = RAGHealthChecker(vector_store=_FailingVectorStore())

    await checker.check_vector_store()

    assert logger_stub.errors == ["Vector store health check failed"]
    _assert_sanitized(logger_stub.errors)


@pytest.mark.asyncio
async def test_database_health_check_sanitizes_connection_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    def fail_connect(*_args, **_kwargs):
        raise sqlite3.Error("database opened /private/rag-health.db failed")

    monkeypatch.setattr(sqlite3, "connect", fail_connect)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.RAG.rag_service.health_check.logger",
        logger_stub,
    )
    checker = RAGHealthChecker()

    await checker.check_database()

    assert logger_stub.errors == ["Database health check failed"]
    _assert_sanitized(logger_stub.errors)


@pytest.mark.asyncio
async def test_search_index_health_check_sanitizes_connection_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    def fail_connect(*_args, **_kwargs):
        raise sqlite3.Error("search index opened /private/rag-index.db failed")

    monkeypatch.setattr(sqlite3, "connect", fail_connect)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.RAG.rag_service.health_check.logger",
        logger_stub,
    )
    checker = RAGHealthChecker()

    await checker.check_search_index()

    assert logger_stub.errors == ["Search index health check failed"]
    _assert_sanitized(logger_stub.errors)


@pytest.mark.asyncio
async def test_embedding_service_health_check_sanitizes_backend_failure_log(monkeypatch):
    logger_stub = _LoggerStub()
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
    monkeypatch.setattr(
        "tldw_Server_API.app.core.RAG.rag_service.health_check.logger",
        logger_stub,
    )
    checker = RAGHealthChecker()

    await checker.check_embedding_service()

    assert logger_stub.errors == ["Embedding service health check failed"]
    _assert_sanitized(logger_stub.errors)


def test_sqlite_db_path_fallback_sanitizes_config_failure_log(monkeypatch):
    logger_stub = _LoggerStub()
    original_import = builtins.__import__

    def fail_config_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level > 0 and name == "core.config":
            raise RuntimeError("config failed at /private/config.txt?token=secret-token")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_config_import)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.RAG.rag_service.health_check.logger",
        logger_stub,
    )
    checker = RAGHealthChecker()

    assert checker._resolve_sqlite_db_path() == ":memory:"
    assert logger_stub.debugs == ["Health check db path fallback"]
    _assert_sanitized(logger_stub.debugs)


@pytest.mark.asyncio
async def test_periodic_health_check_sanitizes_internal_failure_log(monkeypatch):
    logger_stub = _LoggerStub()
    checker = RAGHealthChecker(check_interval=0)

    async def fail_check_all():
        checker._running = False
        raise RuntimeError("periodic check failed at /private/rag.db")

    async def no_sleep(_interval):
        return None

    checker._running = True
    monkeypatch.setattr(checker, "check_all", fail_check_all)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.RAG.rag_service.health_check.asyncio.sleep",
        no_sleep,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.RAG.rag_service.health_check.logger",
        logger_stub,
    )

    await checker._periodic_check()

    assert logger_stub.errors == ["Error in periodic health check"]
    _assert_sanitized(logger_stub.errors)

"""Sanitizer coverage for PGVector adapter fallback logs."""

import pytest

from tldw_Server_API.app.core.RAG.rag_service.vector_stores import pgvector_adapter
from tldw_Server_API.app.core.RAG.rag_service.vector_stores.base import (
    VectorStoreConfig,
    VectorStoreType,
)
from tldw_Server_API.app.core.RAG.rag_service.vector_stores.pgvector_adapter import (
    PGVectorAdapter,
)


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debug_records: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.error_records: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def debug(self, *args: object, **kwargs: object) -> None:
        self.debug_records.append((args, kwargs))

    def error(self, *args: object, **kwargs: object) -> None:
        self.error_records.append((args, kwargs))

    def info(self, *args: object, **kwargs: object) -> None:
        pass


def _adapter() -> PGVectorAdapter:
    return PGVectorAdapter(
        VectorStoreConfig(
            store_type=VectorStoreType.PGVECTOR,
            connection_params={},
            embedding_dim=3,
            user_id="sanitizer-test",
        )
    )


def _assert_no_sensitive_fragments(rendered_log: str) -> None:
    assert "topsecret" not in rendered_log
    assert "pgvector.sqlite" not in rendered_log
    assert "/var/lib/tldw/private" not in rendered_log
    assert "token=secret-token" not in rendered_log
    assert "postgresql://user:" not in rendered_log


@pytest.mark.asyncio
async def test_initialize_failure_log_omits_raw_exception_and_resets_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    adapter._conn = object()
    adapter._pool = object()

    def _raise_sensitive_dsn_error(_params: dict[str, object]) -> str:
        raise RuntimeError(
            "failed for postgresql://user:topsecret@db.example/app "
            "using /var/lib/tldw/private/pgvector.sqlite?token=secret-token"
        )

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    monkeypatch.setattr(adapter, "_build_dsn", _raise_sensitive_dsn_error)

    await adapter.initialize()

    assert adapter._conn is None
    assert adapter._pool is None
    assert adapter._initialized is False
    assert logger_stub.error_records
    _assert_no_sensitive_fragments(repr(logger_stub.error_records))


@pytest.mark.asyncio
async def test_register_vector_failure_log_omits_raw_exception_and_stays_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    adapter._conn = object()

    class _FakeVector:
        pass

    def _raise_sensitive_registration_error(_conn: object) -> None:
        raise RuntimeError(
            "registration failed for postgresql://user:topsecret@db.example/app "
            "using /var/lib/tldw/private/pgvector.sqlite?token=secret-token"
        )

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    monkeypatch.setattr(pgvector_adapter, "_PgVector", _FakeVector)
    monkeypatch.setattr(pgvector_adapter, "_register_pgvector", _raise_sensitive_registration_error)

    await adapter._register_vector_support()

    assert adapter._vector_cls is None
    assert logger_stub.debug_records
    _assert_no_sensitive_fragments(repr(logger_stub.debug_records))

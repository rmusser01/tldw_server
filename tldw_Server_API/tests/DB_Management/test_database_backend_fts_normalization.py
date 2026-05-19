from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    DatabaseError,
    FTSQuery,
    QueryResult,
)
from tldw_Server_API.app.core.DB_Management.backends.fts_translator import FTSQueryTranslator
from tldw_Server_API.app.core.DB_Management.backends.postgresql_backend import PostgreSQLBackend
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend


def _expect_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        pytest.fail(f"{message}: expected {expected!r}, got {actual!r}")


def test_sqlite_backend_fts_search_normalizes_query_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    backend = SQLiteBackend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(tmp_path / "fts-normalize.db"),
            client_id="sqlite-fts-test",
        )
    )
    calls: list[tuple[str, str]] = []
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        FTSQueryTranslator,
        "normalize_query",
        staticmethod(lambda query, backend_name: calls.append((query, backend_name)) or "hello OR world"),
    )

    def fake_execute(sql: str, params=(), connection=None):
        captured["sql"] = sql
        captured["params"] = params
        return QueryResult(rows=[], rowcount=0)

    monkeypatch.setattr(backend, "execute", fake_execute)

    backend.fts_search(FTSQuery(query="hello world", table="docs_fts"))

    _expect_equal(calls, [("hello world", "sqlite")], "expected sqlite query normalization call")
    _expect_equal(captured["params"], ("hello OR world",), "expected normalized sqlite query param")


def test_postgresql_backend_fts_search_raises_when_normalized_query_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = PostgreSQLBackend(DatabaseConfig(backend_type=BackendType.POSTGRESQL, client_id="pg-fts-test"))

    monkeypatch.setattr(
        FTSQueryTranslator,
        "normalize_query",
        staticmethod(lambda query, backend_name: ""),
    )

    with pytest.raises(DatabaseError, match="normalized to empty"):
        backend.fts_search(FTSQuery(query="!!!", table="docs_fts"))

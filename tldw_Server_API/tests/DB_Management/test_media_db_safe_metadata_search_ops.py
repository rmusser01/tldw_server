from __future__ import annotations

import importlib
import importlib.util
import sqlite3
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError

pytestmark = pytest.mark.unit


def _load_safe_metadata_search_ops_module():
    module_name = (
        "tldw_Server_API.app.core.DB_Management.media_db.runtime.safe_metadata_search_ops"
    )
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(module_name)


def test_search_by_safe_metadata_rebinds_on_media_database() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.media_database import MediaDatabase

    safe_metadata_search_ops_module = _load_safe_metadata_search_ops_module()

    assert (
        MediaDatabase.search_by_safe_metadata
        is safe_metadata_search_ops_module.search_by_safe_metadata
    )


def test_search_by_safe_metadata_uses_identifier_join_and_grouped_count() -> None:
    safe_metadata_search_ops_module = _load_safe_metadata_search_ops_module()

    queries: list[tuple[str, tuple[object, ...]]] = []

    class _Cursor:
        def __init__(self, rows):
            self._rows = rows

        def fetchone(self):
            return self._rows[0] if self._rows else None

        def fetchall(self):
            return self._rows

    def _execute_query(query, params):
        normalized = " ".join(query.split())
        queries.append((normalized, params))
        if "COUNT(DISTINCT m.id)" in normalized:
            return _Cursor([{"total_count": 2}])
        return _Cursor(
            [
                {
                    "media_id": 1,
                    "title": "Alpha",
                    "type": "pdf",
                    "version_number": 2,
                    "created_at": "2026-01-10T00:00:00.000Z",
                    "safe_metadata": '{"doi":"10.1000/xyz"}',
                }
            ]
        )

    db = SimpleNamespace(
        backend_type=BackendType.SQLITE,
        execute_query=_execute_query,
    )

    rows, total = safe_metadata_search_ops_module.search_by_safe_metadata(
        db,
        filters=[{"field": "doi", "op": "eq", "value": "10.1000/xyz"}],
        match_all=True,
        page=2,
        per_page=10,
        group_by_media=True,
        sort_by="title_asc",
    )

    assert total == 2
    assert rows == [
        {
            "media_id": 1,
            "title": "Alpha",
            "type": "pdf",
            "version_number": 2,
            "created_at": "2026-01-10T00:00:00.000Z",
            "safe_metadata": '{"doi":"10.1000/xyz"}',
        }
    ]
    count_sql, count_params = queries[0]
    result_sql, result_params = queries[1]
    assert "LEFT JOIN DocumentVersionIdentifiers dvi ON dvi.dv_id = dv.id" in count_sql
    assert "COUNT(DISTINCT m.id) AS total_count" in count_sql
    assert "dvi.doi = ?" in count_sql
    assert count_params == ("10.1000/xyz",)
    assert "GROUP BY m.id" in result_sql
    assert "ORDER BY m.title COLLATE NOCASE ASC, m.id ASC" in result_sql
    assert result_params == ("10.1000/xyz", 10, 10)


def test_search_by_safe_metadata_uses_json_fallback_and_zero_result_fast_return() -> None:
    safe_metadata_search_ops_module = _load_safe_metadata_search_ops_module()

    queries: list[tuple[str, tuple[object, ...]]] = []

    class _Cursor:
        def fetchone(self):
            return {"total_count": 0}

        def fetchall(self):
            return []

    def _execute_query(query, params):
        queries.append((" ".join(query.split()), params))
        return _Cursor()

    db = SimpleNamespace(
        backend_type=BackendType.SQLITE,
        execute_query=_execute_query,
    )

    rows, total = safe_metadata_search_ops_module.search_by_safe_metadata(
        db,
        filters=[{"field": "journal", "op": "icontains", "value": "nature"}],
        match_all=True,
        page=1,
        per_page=20,
        group_by_media=True,
    )

    assert rows == []
    assert total == 0
    assert len(queries) == 1
    count_sql, count_params = queries[0]
    assert "LEFT JOIN DocumentVersionIdentifiers" not in count_sql
    assert "LOWER(dv.safe_metadata) LIKE ?" in count_sql
    assert count_params == ("%nature%",)


def test_search_by_safe_metadata_eq_matches_default_json_spacing() -> None:
    safe_metadata_search_ops_module = _load_safe_metadata_search_ops_module()
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE Media (
            id INTEGER PRIMARY KEY,
            title TEXT,
            type TEXT,
            deleted INTEGER NOT NULL DEFAULT 0,
            last_modified TEXT,
            system_operation_id TEXT
        );
        CREATE TABLE DocumentVersions (
            id INTEGER PRIMARY KEY,
            media_id INTEGER NOT NULL,
            version_number INTEGER NOT NULL,
            created_at TEXT,
            safe_metadata TEXT,
            deleted INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO Media (id, title, type, last_modified)
        VALUES (7, 'Provider paper', 'pdf', '2026-07-12T00:00:00Z');
        INSERT INTO DocumentVersions (
            id, media_id, version_number, created_at, safe_metadata
        ) VALUES (
            1,
            7,
            1,
            '2026-07-12T00:00:00Z',
            '{"provider_ids": {"openalex_id": "W123"}}'
        );
        """
    )
    db = SimpleNamespace(
        backend_type=BackendType.SQLITE,
        execute_query=connection.execute,
    )

    try:
        rows, total = safe_metadata_search_ops_module.search_by_safe_metadata(
            db,
            filters=[{"field": "openalex_id", "op": "eq", "value": "W123"}],
        )
    finally:
        connection.close()

    assert total == 1
    assert rows[0]["media_id"] == 7


def test_search_by_safe_metadata_ungrouped_query_preserves_standard_constraints_and_postgres_title_sort() -> None:
    safe_metadata_search_ops_module = _load_safe_metadata_search_ops_module()

    queries: list[tuple[str, tuple[object, ...]]] = []

    class _Cursor:
        def __init__(self, rows):
            self._rows = rows

        def fetchone(self):
            return self._rows[0] if self._rows else None

        def fetchall(self):
            return self._rows

    def _execute_query(query, params):
        normalized = " ".join(query.split())
        queries.append((normalized, params))
        if "COUNT(*) AS total_count" in normalized:
            return _Cursor([{"total_count": 1}])
        return _Cursor(
            [
                {
                    "media_id": 3,
                    "title": "Zulu Study",
                    "type": "pdf",
                    "version_number": 5,
                    "created_at": "2026-01-15T00:00:00.000Z",
                    "safe_metadata": '{"journal":"Nature Medicine"}',
                }
            ]
        )

    db = SimpleNamespace(
        backend_type=BackendType.POSTGRESQL,
        execute_query=_execute_query,
    )

    rows, total = safe_metadata_search_ops_module.search_by_safe_metadata(
        db,
        filters=[{"field": "journal", "op": "contains", "value": "Nature"}],
        match_all=False,
        page=3,
        per_page=5,
        group_by_media=False,
        text_query="biology",
        media_types=["pdf"],
        must_have_keywords=["review", "biology"],
        must_not_have_keywords=["private"],
        date_start="2026-01-01T00:00:00.000Z",
        date_end="2026-01-31T23:59:59.999Z",
        sort_by="title_desc",
    )

    assert total == 1
    assert rows[0]["media_id"] == 3
    count_sql, count_params = queries[0]
    result_sql, result_params = queries[1]
    assert "COUNT(*) AS total_count" in count_sql
    assert "GROUP BY m.id" not in result_sql
    assert "ORDER BY LOWER(m.title) DESC, m.title DESC, m.id DESC" in result_sql
    assert "LOWER(COALESCE(m.title, '')) LIKE ?" in count_sql
    assert "LOWER(COALESCE(dv.safe_metadata, '')) LIKE ?" in count_sql
    assert "LOWER(m.type) IN (?)" in count_sql
    assert "COUNT(DISTINCT k_mh.id)" in count_sql
    assert "NOT EXISTS" in count_sql
    assert "dv.created_at >= ?" in count_sql
    assert "dv.created_at <= ?" in count_sql
    assert count_params == (
        "%Nature%",
        "%biology%",
        "%biology%",
        "pdf",
        "review",
        "biology",
        2,
        "private",
        "2026-01-01T00:00:00.000Z",
        "2026-01-31T23:59:59.999Z",
    )
    assert result_params == count_params + (5, 10)


def test_search_by_safe_metadata_wraps_query_errors() -> None:
    safe_metadata_search_ops_module = _load_safe_metadata_search_ops_module()

    db = SimpleNamespace(
        backend_type=BackendType.SQLITE,
        execute_query=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("bad query")
        ),
    )

    with pytest.raises(DatabaseError, match="Failed metadata search"):
        safe_metadata_search_ops_module.search_by_safe_metadata(
            db,
            filters=[{"field": "doi", "op": "eq", "value": "10.1000/xyz"}],
        )

"""Actual SQLite fallback filtering and scope/pagination regressions."""
import sqlite3
from datetime import datetime
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_search_repository import MediaSearchRepository
from tldw_Server_API.app.core.DB_Management.scope_context import reset_scope, set_scope

pytestmark = pytest.mark.unit


@pytest.fixture
def search_db():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE Media(id INTEGER PRIMARY KEY, uuid TEXT, url TEXT, title TEXT,
          content TEXT, type TEXT, author TEXT, ingestion_date TEXT, transcription_model TEXT,
          is_trash INTEGER DEFAULT 0, trash_date TEXT, chunking_status TEXT, vector_processing TEXT,
          content_hash TEXT, last_modified TEXT, version INTEGER DEFAULT 1, client_id TEXT,
          deleted INTEGER DEFAULT 0, system_operation_id TEXT, visibility TEXT,
          owner_user_id INTEGER, team_id INTEGER, org_id INTEGER);
        CREATE TABLE DocumentVersions(id INTEGER PRIMARY KEY, media_id INTEGER,
          version_number INTEGER, safe_metadata TEXT, deleted INTEGER);
        CREATE VIRTUAL TABLE media_fts USING fts5(title, content);
        CREATE TABLE Keywords(id INTEGER PRIMARY KEY, keyword TEXT, deleted INTEGER DEFAULT 0);
        CREATE TABLE MediaKeywords(media_id INTEGER, keyword_id INTEGER);
    """)

    def add(row_id, text, *, owner=1, title="Source", **values):
        columns = dict(id=row_id, uuid=f"source-{row_id}", title=title, content=text,
                       owner_user_id=owner, client_id=str(owner), type="document",
                       ingestion_date="2026-09-16", last_modified="2026-09-16", **values)
        connection.execute(f"INSERT INTO Media({','.join(columns)}) VALUES({','.join('?' for _ in columns)})", tuple(columns.values()))
        connection.execute("INSERT INTO media_fts(rowid,title,content) VALUES(?,?,?)", (row_id, title, text))

    add(1, "AURORA-CYCLE5-MULTI-23 blue harbor")
    add(2, "unrelated violet orchard")
    add(3, "AURORA-CYCLE5-MULTI-23 second harbor")
    add(4, "AURORA-CYCLE5-MULTI-23 foreign harbor", owner=2)
    add(5, "AURORA-CYCLE5-MULTI-23 trashed", is_trash=1)
    add(6, "AURORA-CYCLE5-MULTI-23 deleted", deleted=1)
    db = SimpleNamespace(backend_type=BackendType.SQLITE, db_path_str="in-memory-test",
                         execute_query=connection.execute)
    scope = set_scope(user_id=1)
    try:
        yield MediaSearchRepository(db), connection, add, db
    finally:
        reset_scope(scope)
        connection.close()


@pytest.mark.parametrize("query,expected", [
    ("AURORA-CYCLE5-MULTI-23", [3, 1]),
    ("FOREIGN-CYCLE5-MARKER", []),
    ('"AURORA-CYCLE5-MULTI-23"', [3, 1]),
    ("harbor NOT second", [1]),
    ("unmatched", []),
    ("", [3, 2, 1]),
])
def test_actual_fts_and_literal_fallback_preserve_query_and_owner(search_db, query, expected):
    rows, total = search_db[0].search(query)
    assert ([row["id"] for row in rows], total) == (expected, len(expected))


def test_fallback_pagination_counts_only_literal_matches(search_db):
    rows, total = search_db[0].search("AURORA-CYCLE5-MULTI-23", page=2, results_per_page=1)
    assert ([row["id"] for row in rows], total) == ([1], 2)


@pytest.mark.parametrize("fields,expected", [(["title"], [7]), (["content"], [3, 1]), (["author"], [7])])
def test_fallback_respects_selected_fields(search_db, fields, expected):
    repo, _, add, _ = search_db
    add(7, "unrelated", title="AURORA-CYCLE5-MULTI-23", author="AURORA-CYCLE5-MULTI-23")
    rows, total = repo.search("AURORA-CYCLE5-MULTI-23", search_fields=fields)
    assert ([row["id"] for row in rows], total) == (expected, len(expected))


def test_fallback_retains_media_ids_keywords_and_trash_controls(search_db):
    repo, connection, _, _ = search_db
    connection.execute("INSERT INTO Keywords VALUES(1,'retain',0)")
    connection.execute("INSERT INTO MediaKeywords VALUES(1,1)")
    rows, total = repo.search("AURORA-CYCLE5-MULTI-23", media_ids_filter=[1, 2, 3, 4],
                             media_types=["document"], must_have_keywords=["retain"])
    assert ([row["id"] for row in rows], total) == ([1], 1)
    rows, total = repo.search("AURORA-CYCLE5-MULTI-23", include_trash=True, include_deleted=True)
    assert ([row["id"] for row in rows], total) == ([6, 5, 3, 1], 4)


def test_result_only_fts_failure_recounts_literal_results(search_db):
    repo, connection, _, db = search_db
    def execute(sql, params):
        if "media_fts MATCH" in sql and "SELECT DISTINCT" in sql:
            raise sqlite3.OperationalError("fts5: syntax error")
        return connection.execute(sql, params)
    db.execute_query = execute
    # FTS matches separate tokens; literal fallback must not reuse that count.
    rows, total = repo.search("blue AND harbor")
    assert (rows, total) == ([], 0)


def test_fallback_punctuation_does_not_turn_into_like_wildcards(search_db):
    repo, _, add, _ = search_db
    add(7, "LITERAL-50%_VALUE")
    add(8, "LITERAL-5000XVALUE")
    rows, total = repo.search("LITERAL-50%_VALUE")
    assert ([row["id"] for row in rows], total) == ([7], 1)


def test_missing_fts_table_uses_same_filtered_fallback(search_db):
    repo, connection, _, _ = search_db
    connection.execute("DROP TABLE media_fts")
    rows, total = repo.search("harbor", sort_by="relevance")
    assert ([row["id"] for row in rows], total) == ([3, 1], 2)


def test_fallback_retains_team_org_visibility_and_date_exclusion(search_db):
    repo, _, add, _ = search_db
    add(7, "AURORA-CYCLE5-MULTI-23 team", owner=2, visibility="team", team_id=10)
    add(8, "AURORA-CYCLE5-MULTI-23 org", owner=2, visibility="org", org_id=20)
    add(9, "AURORA-CYCLE5-MULTI-23 foreign team", owner=2, visibility="team", team_id=11)
    scope = set_scope(user_id=1, team_ids=[10], org_ids=[20])
    try:
        rows, total = repo.search("AURORA-CYCLE5-MULTI-23", media_ids_filter=[7, 8, 9])
        assert ([row["id"] for row in rows], total) == ([8, 7], 2)
        assert repo.search("AURORA-CYCLE5-MULTI-23", date_range={"start_date": datetime(2026, 9, 17)}) == ([], 0)
    finally:
        reset_scope(scope)

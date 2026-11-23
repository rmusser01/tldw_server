"""ChaChaContext: the stores' one view of the CharactersRAGDB they sit on (TASK-13950.22 PR C1)."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    InputError,
    OwnerScope,
    OwnerScopeRequiredError,
)

pytestmark = pytest.mark.unit


class _PostgresDB(CharactersRAGDB):
    """A DB that reports PostgreSQL without a server: only the backend-dependent helpers are used."""

    backend_type = BackendType.POSTGRESQL


@pytest.fixture
def db():
    database = CharactersRAGDB(":memory:", client_id="ctx-test")
    try:
        yield database
    finally:
        database.close_connection()


@pytest.fixture
def pg() -> CharactersRAGDB:
    return _PostgresDB.__new__(_PostgresDB)


def test_db_builds_one_context_before_its_stores(db) -> None:
    assert db.ctx.db is db
    assert db.conversation_store._ctx is db.ctx
    assert db.message_store._ctx is db.ctx


def test_context_reads_the_db_at_call_time(db, monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(db, "execute_query", lambda query, *a, **k: seen.append(query) or "patched")

    assert db.ctx.execute_query("SELECT 1") == "patched"
    assert seen == ["SELECT 1"]
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 7)
    assert db.ctx.code_schema_version == 7


def test_context_exposes_the_db_state(db) -> None:
    ctx = db.ctx
    assert (ctx.client_id, ctx.owner_user_id, ctx.is_memory_db) == ("ctx-test", "ctx-test", True)
    assert ctx.backend is db.backend and ctx.backend_type == BackendType.SQLITE
    assert ctx.db_path_str == ":memory:" and ctx.schema_name == db._SCHEMA_NAME


def test_map_table_and_owner_filter_by_backend(db, pg) -> None:
    assert db.ctx.map_table("keywords") == "keywords"
    assert db.ctx.owner_filter("1") == ("", ())
    assert pg.ctx.map_table("keywords") == "chacha_keywords"
    assert pg.ctx.map_table("notes") == "notes"
    with pytest.raises(OwnerScopeRequiredError, match="_selected_owner_filter"):
        pg.ctx.owner_filter(None, table="notes")
    assert pg.ctx.owner_filter(OwnerScope.NOT_APPLICABLE) == ("", ())
    assert pg.ctx.owner_filter("7", "n") == (" AND n.client_id = ?", ("7",))


def test_prepare_statement_matches_the_db(db, pg) -> None:
    query, params = "SELECT * FROM t WHERE a = ? AND b = ?", (1, 2)
    assert pg.ctx.prepare_statement(query, params) == pg._prepare_backend_statement(query, params)
    assert pg.ctx.prepare_statement(query, params)[0] != query
    assert db.ctx.prepare_statement(query, params) == db._prepare_backend_statement(query, params)


def test_helpers_match_the_db_forwarders(db) -> None:
    assert db.ctx.normalize_nullable_text("  x ") == "x"
    assert db.ctx.normalize_nullable_text("  ") is None
    assert db.ctx.normalize_nullable_text(5) == "5"
    assert db.ctx.utc_now_iso().endswith("Z")
    assert len(db.ctx.new_uuid()) == 36
    assert db._generate_uuid() != db._generate_uuid()


def test_db_forwarders_follow_a_context_patch(db, monkeypatch) -> None:
    monkeypatch.setattr(db.ctx, "utc_now_iso", lambda: "T")
    assert db._get_current_utc_timestamp_iso() == "T"


def test_keyword_link_filter_and_require_owner_row_by_backend(db, pg) -> None:
    assert db.ctx.keyword_link_filter("note_keywords", "nk", owner_client_id="1") == ("", ())
    sql, params = pg.ctx.keyword_link_filter("note_keywords", "nk", owner_client_id="7")
    assert params == ("7", "7")
    assert "owner_parent.id = nk.note_id" in sql and "chacha_keywords owner_keyword" in sql
    with pytest.raises(OwnerScopeRequiredError, match="_selected_keyword_link_filter"):
        pg.ctx.keyword_link_filter("note_keywords", "nk", owner_client_id=None)
    with pytest.raises(InputError):
        pg.ctx.keyword_link_filter("note_keywords", "n k", owner_client_id="7")
    assert db.ctx.require_owner_row(None, "notes", 1, None) is None  # SQLite: no row lock, conn untouched
    assert pg.ctx.require_owner_row(None, "notes", None, None) is None  # no item, nothing to lock
    with pytest.raises(OwnerScopeRequiredError, match="_require_selected_owner_row"):
        pg.ctx.require_owner_row(None, "notes", 1, None)
    with pytest.raises(InputError):
        pg.ctx.require_owner_row(None, "widgets", 1, "7")


def test_order_and_match_helpers_by_backend(db, pg) -> None:
    assert db.ctx.case_insensitive_order_expression("kw.keyword", "DESC") == "kw.keyword COLLATE NOCASE DESC"
    assert pg.ctx.case_insensitive_order_expression("kw.keyword") == "LOWER(kw.keyword)"
    assert pg.ctx.case_insensitive_order_clause("path", "ASC") == "ORDER BY LOWER(path) ASC"
    with pytest.raises(InputError):
        db.ctx.case_insensitive_order_expression("name; DROP TABLE x")
    assert db.ctx.unique_text_match_sql("keywords", "keyword") == "keyword = ?"
    assert pg.ctx.unique_text_match_sql("keywords", "keyword") == "LOWER(keyword) = LOWER(?)"
    assert pg.ctx.unique_text_match_sql("notes", "title") == "title = ?"


def test_pure_helpers_match_the_db_forwarders(db) -> None:
    assert db.ctx.chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert db.ctx.note_task_v60_hash({"a": 1}) == db._note_task_v60_hash({"a": 1}) == CharactersRAGDB._note_task_v60_hash({"a": 1})
    assert db.ctx.canonical_json_text_v61({"b": 1, "a": 2}) == db._canonical_json_text_v61({"b": 1, "a": 2})
    assert db.ctx.is_unique_violation(Exception("UNIQUE constraint failed: x")) is db._is_unique_violation(Exception("UNIQUE constraint failed: x"))
    assert db.ctx.is_unique_violation(ValueError("other")) is False
    row = {"id": 1, "blob": '{"k": 1}', "bad": "{"}
    assert db.ctx.deserialize_row_fields(row, ["blob", "bad"]) == {"id": 1, "blob": {"k": 1}, "bad": None}
    assert db.ctx.deserialize_row_fields(None, ["blob"]) is None


def test_runtime_schema_version_reads_the_db(db) -> None:
    assert db.ctx.runtime_schema_version is None
    db.runtime_schema_version = 81
    assert db.ctx.runtime_schema_version == 81

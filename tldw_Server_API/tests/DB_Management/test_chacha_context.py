"""ChaChaContext: the stores' one view of the CharactersRAGDB they sit on (TASK-13950.22 PR C1)."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
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

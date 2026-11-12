"""Real conversation search regressions for literal punctuation and FTS controls."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

pytestmark = pytest.mark.integration


@pytest.fixture
def db(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    """Yield an isolated SQLite search store and close every connection afterward."""
    database = CharactersRAGDB(db_path=str(tmp_path / "punctuation.sqlite"), client_id="alice")
    yield database
    database.close_all_connections()


def _add(db: CharactersRAGDB, conversation_id: str, title: str, *, owner: str="alice", workspace: str | None=None) -> str:
    """Create an owned persona conversation in the requested global or workspace scope."""
    return db.add_conversation(
        {
            "id": conversation_id,
            "root_id": conversation_id,
            "assistant_kind": "persona",
            "assistant_id": "persona-search",
            "persona_memory_mode": "read_only",
            "title": title,
            "client_id": owner,
            "scope_type": "workspace" if workspace else "global",
            "workspace_id": workspace,
        }
    )


def _ids(db: CharactersRAGDB, query: str, *, paged: bool, **kwargs: Any) -> set[str]:
    """Compare both search APIs through their returned conversation IDs and total."""
    if paged:
        rows, total, _ = db.search_conversations_page(query, **kwargs)
        assert total == len(rows)
    else:
        rows = db.search_conversations(query, **kwargs)
    return {row["id"] for row in rows}


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize(
    "query",
    [
        "ALICE-CEDAR-SIDEPANEL-742",
        "cedar/maple",
        "O'Reilly",
        "note:cedar",
        'cedar"branch',
        '"cedar',
        "!!!",
    ],
)
def test_punctuation_is_a_literal_search_when_fts_rejects_it(db: CharactersRAGDB, paged: bool, query: str) -> None:
    """Treat rejected FTS syntax as literal text without exposing foreign conversations."""
    _add(db, "own", query)
    _add(db, "foreign", query, owner="bob")
    _add(db, "unrelated", "other conversation")

    assert _ids(db, query, paged=paged, client_id="alice", scope_type="global") == (
        set() if query == "!!!" else {"own"}
    )


@pytest.mark.parametrize("order_by", ["recency", "bm25", "hybrid", "topic"])
def test_punctuation_fallback_keeps_count_ranking_pagination_and_scope(db: CharactersRAGDB, order_by: str) -> None:
    """Keep counts, score normalization, pages and workspace ownership after fallback."""
    marker = "ALICE-CEDAR-SIDEPANEL-742"
    db.upsert_workspace("ws-a", "A")
    db.upsert_workspace("ws-b", "B")
    _add(db, "own-a", marker, workspace="ws-a")
    _add(db, "own-b", f"{marker} extended title", workspace="ws-a")
    _add(db, "foreign", marker, owner="bob", workspace="ws-a")
    _add(db, "other-workspace", marker, workspace="ws-b")
    _add(db, "global", marker)
    _add(db, "deleted", marker, workspace="ws-a")
    deleted = db.get_conversation_by_id("deleted")
    db.soft_delete_conversation("deleted", expected_version=deleted["version"])
    options = {"client_id": "alice", "scope_type": "workspace", "workspace_id": "ws-a", "order_by": order_by}

    first, total, maximum = db.search_conversations_page(marker, limit=1, offset=0, **options)
    second, second_total, second_maximum = db.search_conversations_page(marker, limit=1, offset=1, **options)
    beyond, beyond_total, _ = db.search_conversations_page(marker, limit=1, offset=2, **options)

    assert {row["id"] for row in first + second} == {"own-a", "own-b"}
    assert total == second_total == beyond_total == 2
    assert beyond == []
    assert maximum == pytest.approx(second_maximum)
    if order_by != "recency":
        assert maximum > 0
        assert max(row["bm25_norm"] for row in first + second) == pytest.approx(1.0)


@pytest.mark.parametrize("paged", [False, True])
def test_deleted_punctuation_search_keeps_existing_text_search_and_owner_scope(db: CharactersRAGDB, paged: bool) -> None:
    """Include deleted rows only when requested while preserving owner isolation."""
    marker = "ALICE-CEDAR-SIDEPANEL-742"
    for conversation_id, owner in (("live", "alice"), ("deleted", "alice"), ("foreign", "bob")):
        _add(db, conversation_id, marker, owner=owner)
    deleted = db.get_conversation_by_id("deleted")
    db.soft_delete_conversation("deleted", expected_version=deleted["version"])

    assert _ids(db, marker, paged=paged, client_id="alice", include_deleted=True) == {"live", "deleted"}
    assert _ids(db, marker, paged=paged, client_id="alice", deleted_only=True) == {"deleted"}


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize(
    "query,expected",
    [
        ("cedar", {"near", "far"}),
        ("ced*", {"near", "far"}),
        ('"cedar side"*', {"near"}),
        ('"cedar sidepanel"', {"near"}),
        ("NEAR(cedar sidepanel, 1)", {"near"}),
        ("(cedar OR birch) AND sidepanel", {"near", "far", "birch"}),
        ("cedar NOT maple", {"near"}),
        ("title:cedar", {"near", "far"}),
    ],
)
def test_valid_fts_expressions_keep_their_meaning(db: CharactersRAGDB, paged: bool, query: str, expected: set[str]) -> None:
    """Preserve valid phrases, prefixes, proximity, boolean and column expressions."""
    _add(db, "near", "cedar sidepanel")
    _add(db, "far", "cedar maple one two three four sidepanel")
    _add(db, "birch", "birch sidepanel")

    assert _ids(db, query, paged=paged, client_id="alice") == expected


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize(
    "message",
    [
        "database is locked",
        "database disk image is malformed",
        "no such column: c.missing_column",
        "unable to use function MATCH in the requested context",
    ],
)
def test_unrelated_database_errors_are_not_retried(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, paged: bool, message: str) -> None:
    """Propagate unrelated storage errors after exactly one database attempt."""
    attempts = []

    def fail(query: str, params: Any) -> NoReturn:
        """Inject a non-parser storage failure and record its sole attempted query."""
        attempts.append(query)
        raise CharactersRAGDBError("storage failure") from sqlite3.OperationalError(message)

    monkeypatch.setattr(db, "execute_query", fail)
    with pytest.raises(CharactersRAGDBError, match="storage failure"):
        _ids(db, "cedar-maple", paged=paged)
    assert len(attempts) == 1


@pytest.mark.parametrize("paged", [False, True])
def test_literal_retry_failure_is_propagated_without_further_retries(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, paged: bool) -> None:
    """Stop after the literal fallback itself fails rather than retrying indefinitely."""
    _add(db, "own", "cedar-maple")
    original = db.execute_query
    attempts = []

    def fail_after_parse_error(query: str, params: Any) -> Any:
        """Allow the original FTS parse error, then fail the single literal fallback."""
        attempts.append(params)
        if len(attempts) == 1:
            return original(query, params)
        raise CharactersRAGDBError("literal read failed") from sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "execute_query", fail_after_parse_error)
    with pytest.raises(CharactersRAGDBError, match="literal read failed"):
        _ids(db, "cedar-maple", paged=paged)
    assert len(attempts) == 2


@pytest.mark.postgres
def test_postgres_punctuation_preserves_scoped_search(pg_database_config: DatabaseConfig) -> None:
    """Use the official temporary PostgreSQL database for punctuation and scope controls."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    database = CharactersRAGDB(db_path=":memory:", client_id="alice", backend=backend)
    try:
        marker = "ALICE-CEDAR-SIDEPANEL-742"
        database.upsert_workspace("ws-a", "A")
        database.upsert_workspace("ws-b", "B")
        _add(database, "own", marker, workspace="ws-a")
        _add(database, "foreign", marker, owner="bob", workspace="ws-a")
        _add(database, "other-workspace", marker, workspace="ws-b")
        _add(database, "deleted", marker, workspace="ws-a")
        deleted = database.get_conversation_by_id("deleted")
        database.soft_delete_conversation("deleted", expected_version=deleted["version"])
        options = {"client_id": "alice", "scope_type": "workspace", "workspace_id": "ws-a"}
        for paged in (False, True):
            assert _ids(database, marker, paged=paged, **options) == {"own"}
            assert _ids(database, "cedar", paged=paged, **options) == {"own"}
            assert _ids(database, marker, paged=paged, include_deleted=True, **options) == {"own", "deleted"}
            assert _ids(database, marker, paged=paged, deleted_only=True, **options) == {"deleted"}
    finally:
        database.close_all_connections()

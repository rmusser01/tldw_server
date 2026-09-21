"""Real conversation search regressions for literal punctuation and FTS controls."""

import sqlite3

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(db_path=str(tmp_path / "punctuation.sqlite"), client_id="alice")
    yield database
    database.close_all_connections()


def _add(db, conversation_id, title, *, owner="alice", workspace=None):
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


def _ids(db, query, *, paged, **kwargs):
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
def test_punctuation_is_a_literal_search_when_fts_rejects_it(db, paged, query):
    _add(db, "own", query)
    _add(db, "foreign", query, owner="bob")
    _add(db, "unrelated", "other conversation")

    assert _ids(db, query, paged=paged, client_id="alice", scope_type="global") == (
        set() if query == "!!!" else {"own"}
    )


@pytest.mark.parametrize("order_by", ["recency", "bm25", "hybrid", "topic"])
def test_punctuation_fallback_keeps_count_ranking_pagination_and_scope(db, order_by):
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
def test_deleted_punctuation_search_keeps_existing_text_search_and_owner_scope(db, paged):
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
def test_valid_fts_expressions_keep_their_meaning(db, paged, query, expected):
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
def test_unrelated_database_errors_are_not_retried(db, monkeypatch, paged, message):
    attempts = []

    def fail(query, params):
        attempts.append(query)
        raise CharactersRAGDBError("storage failure") from sqlite3.OperationalError(message)

    monkeypatch.setattr(db, "execute_query", fail)
    with pytest.raises(CharactersRAGDBError, match="storage failure"):
        _ids(db, "cedar-maple", paged=paged)
    assert len(attempts) == 1


@pytest.mark.parametrize("paged", [False, True])
def test_literal_retry_failure_is_propagated_without_further_retries(db, monkeypatch, paged):
    _add(db, "own", "cedar-maple")
    original = db.execute_query
    attempts = []

    def fail_after_parse_error(query, params):
        attempts.append(params)
        if len(attempts) == 1:
            return original(query, params)
        raise CharactersRAGDBError("literal read failed") from sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "execute_query", fail_after_parse_error)
    with pytest.raises(CharactersRAGDBError, match="literal read failed"):
        _ids(db, "cedar-maple", paged=paged)
    assert len(attempts) == 2


@pytest.mark.integration
def test_postgres_punctuation_preserves_scoped_search(pg_database_config):
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

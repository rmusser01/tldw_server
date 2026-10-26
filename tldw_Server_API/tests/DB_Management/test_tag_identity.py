"""Case-insensitive Collections and Watchlists tags (TASK-13950.5, slice 3).

Both stores keep per-owner tags: Collections ``collection_tags`` (the Media DB
schema defines the same physical table) and Watchlists ``tags``, each scoped by
``user_id``. Every lookup, the unique index and the duplicate merge group by
(owner, identity), so one owner's "Foo" never meets another owner's "foo".
Names keep their first-seen casing and compare by ``normalize_keyword_identity``
(NFC plus casefold).

SQLite's unique index is ``(user_id, name COLLATE NOCASE)`` and Postgres's is
``(user_id, LOWER(name))``. Neither folds ß/SS or NFC forms (SQLite NOCASE also
leaves accented capitals apart), so lookups compare identities over the owner's
non-ASCII names, and the merge groups by identity before the SQL fold. Postgres
LOWER() folds the dotted capital I ("İstanbul" and "istanbul" share a LOWER slot)
while the identity keeps the dot, so they are one tag on Postgres and two on
SQLite: a documented backend difference.

Postgres cases run as a NOBYPASSRLS role (``pg_restricted_backend``), so the
bootstrap merge is proven to see every owner's rows without a user scope.
"""

from __future__ import annotations

import re
from functools import cached_property
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management import schema_once
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseBackend,
    DatabaseConfig,
    DatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.content_backend import backend_target_key
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

NON_ASCII_VARIANTS = [
    pytest.param("Straße", "STRASSE", id="sharp-s"),
    pytest.param("Éclair", "éclair", id="accented-capital"),
    pytest.param("éclair", "éclair", id="nfc-composition"),
]


@pytest.fixture(
    params=[
        pytest.param("sqlite", marks=pytest.mark.unit),
        pytest.param("postgres", marks=pytest.mark.integration),
    ]
)
def backend(request, tmp_path: Path):
    if request.param == "postgres":
        yield request.getfixturevalue("pg_restricted_backend")
        return
    sqlite = SQLiteBackend(DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(tmp_path / "Media_DB_v2.db")))
    yield sqlite
    sqlite.get_pool().close_all()


class _TagStore:
    """One tag store behind a common surface: tag ids, tagged rows, the tag filter and raw rows."""

    table: str
    link_table: str
    link_column: str
    index: str

    def __init__(self, backend: DatabaseBackend, user_id: str = "1") -> None:
        self.backend = backend
        self.user_id = user_id

    def as_user(self, user_id: str) -> _TagStore:
        return type(self)(self.backend, user_id)

    def names(self) -> list[str]:
        rows = self.backend.execute(f"SELECT name FROM {self.table} WHERE user_id = ? ORDER BY id", (self.user_id,)).rows
        return [str(row["name"]) for row in rows]

    def link_count(self, row_id: int) -> int:
        sql = f"SELECT COUNT(*) AS n FROM {self.link_table} WHERE {self.link_column} = ?"
        return int(self.backend.execute(sql, (row_id,)).rows[0]["n"])

    def insert_tag(self, user_id: str, name: str) -> int:
        """Insert a raw tag row the way a pre-migration database could hold it."""
        self.backend.execute(f"INSERT INTO {self.table} (user_id, name) VALUES (?, ?)", (user_id, name))
        row = self.backend.execute(f"SELECT id FROM {self.table} WHERE user_id = ? AND name = ?", (user_id, name)).first
        assert row is not None
        return int(row["id"])

    def link(self, row_id: int, tag_id: int) -> None:
        self.backend.execute(f"INSERT INTO {self.link_table} ({self.link_column}, tag_id) VALUES (?, ?)", (row_id, tag_id))

    def drop_identity_index(self) -> None:
        self.backend.execute(f"DROP INDEX IF EXISTS {self.index}")

    def reopen(self) -> _TagStore:
        """Construct the store again as a new process would, re-running its schema bootstrap."""
        self.forget_schema()
        fresh = self.as_user(self.user_id)
        assert fresh.db is not None
        return fresh

    @property
    def db(self) -> CollectionsDatabase | WatchlistsDatabase:
        raise NotImplementedError

    def forget_schema(self) -> None:
        raise NotImplementedError

    def ensure(self, names: list[str]) -> list[int]:
        raise NotImplementedError

    def tag_new(self, key: str, names: list[str]) -> int:
        raise NotImplementedError

    def retag(self, row_id: int, names: list[str]) -> list[str]:
        raise NotImplementedError

    def tags_of(self, row_id: int) -> list[str]:
        raise NotImplementedError

    def find(self, names: list[str]) -> list[int]:
        raise NotImplementedError


class CollectionsTags(_TagStore):
    table, link_table, link_column = "collection_tags", "content_item_tags", "item_id"
    index = "ux_collection_tags_user_name_identity"

    @cached_property
    def db(self) -> CollectionsDatabase:
        return CollectionsDatabase.from_backend(self.user_id, self.backend)

    def ensure(self, names: list[str]) -> list[int]:
        return self.db.ensure_collection_tag_ids(names)

    def tag_new(self, key: str, names: list[str]) -> int:
        return self.db.upsert_content_item(
            origin="reading",
            url=f"https://example.com/{key}",
            canonical_url=None,
            domain="example.com",
            title=key,
            summary=None,
            content_hash=None,
            word_count=None,
            published_at=None,
            tags=names,
        ).id

    def retag(self, row_id: int, names: list[str]) -> list[str]:
        return self.db.update_content_item(row_id, tags=names).tags

    def tags_of(self, row_id: int) -> list[str]:
        return self.db.get_content_item(row_id).tags

    def find(self, names: list[str]) -> list[int]:
        rows, _total = self.db.list_content_items(tags=names, size=100)
        return sorted(row.id for row in rows)

    def forget_schema(self) -> None:
        schema_once.reset("collections")
        CollectionsDatabase._bootstrapped_backend_targets.discard(backend_target_key(self.backend))


class WatchlistsTags(_TagStore):
    table, link_table, link_column = "tags", "source_tags", "source_id"
    index = "ux_tags_user_name_identity"

    @cached_property
    def db(self) -> WatchlistsDatabase:
        return WatchlistsDatabase(self.user_id, backend=self.backend)

    def ensure(self, names: list[str]) -> list[int]:
        return self.db.ensure_tag_ids(names)

    def tag_new(self, key: str, names: list[str]) -> int:
        return self.db.create_source(name=key, url=f"https://example.com/{key}.xml", source_type="rss", tags=names).id

    def retag(self, row_id: int, names: list[str]) -> list[str]:
        return sorted(self.db.set_source_tags(row_id, names))

    def tags_of(self, row_id: int) -> list[str]:
        return sorted(self.db.get_source(row_id).tags)

    def find(self, names: list[str]) -> list[int]:
        rows, _total = self.db.list_sources(q=None, tag_names=names, limit=100, offset=0)
        return sorted(row.id for row in rows)

    def forget_schema(self) -> None:
        schema_once.reset("watchlists")
        WatchlistsDatabase._schema_init_keys.discard(backend_target_key(self.backend))


@pytest.fixture(params=[CollectionsTags, WatchlistsTags], ids=["collections", "watchlists"])
def store(request, backend) -> _TagStore:
    tag_store: _TagStore = request.param(backend)
    return tag_store


def test_adding_foo_then_foo_keeps_one_tag_found_in_either_case(store: _TagStore) -> None:
    [tag_id] = store.ensure(["Foo"])

    assert store.ensure(["foo"]) == [tag_id]
    assert store.ensure(["FOO", "Foo"]) == [tag_id]
    assert store.names() == ["Foo"]

    row_id = store.tag_new("a", ["FOO"])
    assert store.tags_of(row_id) == ["Foo"]
    for probe in ("Foo", "foo", "FOO"):
        assert store.find([probe]) == [row_id]


def test_linking_case_variants_to_a_row_that_has_the_tag_keeps_one_link(store: _TagStore) -> None:
    row_id = store.tag_new("a", ["foo"])
    assert store.retag(row_id, ["Foo", "foo"]) == ["foo"]
    assert store.link_count(row_id) == 1

    sharp = store.tag_new("b", ["Straße"])
    assert store.retag(sharp, ["STRASSE", "straße"]) == ["Straße"]
    assert store.link_count(sharp) == 1


def test_first_seen_display_casing_is_stored(store: _TagStore) -> None:
    row_id = store.tag_new("a", ["Machine Learning", "machine learning", "NLP"])

    assert store.tags_of(row_id) == ["Machine Learning", "NLP"]
    assert store.names() == ["Machine Learning", "NLP"]


@pytest.mark.parametrize(("first", "second"), NON_ASCII_VARIANTS)
def test_non_ascii_case_variants_are_one_tag(store: _TagStore, first: str, second: str) -> None:
    [tag_id] = store.ensure([first])

    assert store.ensure([second]) == [tag_id]
    assert store.names() == [first]
    row_id = store.tag_new("a", [second])
    assert store.tags_of(row_id) == [first]
    assert store.find([second]) == [row_id]
    assert store.find([first]) == [row_id]


def test_tag_filters_match_by_identity_with_and_semantics(store: _TagStore) -> None:
    both = store.tag_new("a", ["Foo", "Bar"])
    foo_only = store.tag_new("b", ["foo"])

    assert store.find(["FOO"]) == sorted([both, foo_only])
    assert store.find(["Foo", "foo"]) == sorted([both, foo_only])
    assert store.find(["foo", "BAR"]) == [both]
    assert store.find(["foo", "missing"]) == []


def test_tags_never_cross_owners(store: _TagStore) -> None:
    other = store.as_user("2")

    [mine] = store.ensure(["Foo"])
    [theirs] = other.ensure(["foo"])

    assert mine != theirs
    assert store.names() == ["Foo"]
    assert other.names() == ["foo"]
    row_id = other.tag_new("b", ["FOO"])
    assert other.tags_of(row_id) == ["foo"]
    assert store.find(["foo"]) == []
    assert other.find(["Foo"]) == [row_id]


def test_the_dotted_capital_i_follows_each_backends_fold(store: _TagStore) -> None:
    [first] = store.ensure(["İstanbul"])
    second = store.ensure(["istanbul"])

    if store.backend.backend_type != BackendType.POSTGRESQL:
        assert second != [first]
        return
    assert second == [first]
    assert store.names() == ["İstanbul"]
    row_id = store.tag_new("a", ["ISTANBUL"])
    assert store.find(["istanbul"]) == [row_id]


def test_reopening_merges_each_owners_case_variant_duplicates(store: _TagStore) -> None:
    other = store.as_user("2")
    a = store.tag_new("a", ["Foo"])
    b = store.tag_new("b", ["Bar"])
    c = other.tag_new("c", ["Bar"])
    store.drop_identity_index()
    foo_lower = store.insert_tag("1", "foo")
    foo_upper = store.insert_tag("1", "FOO")
    sharp = store.insert_tag("1", "straße")
    sharp_ascii = store.insert_tag("1", "strasse")
    theirs = store.insert_tag("2", "foo")
    store.link(a, foo_lower)
    store.link(b, foo_upper)
    store.link(b, sharp)
    store.link(b, sharp_ascii)
    store.link(c, theirs)

    reopened = store.reopen()

    assert reopened.names() == ["Foo", "Bar", "straße"]
    assert other.names() == ["Bar", "foo"]
    assert reopened.tags_of(a) == ["Foo"]
    assert reopened.link_count(a) == 1
    assert reopened.tags_of(b) == ["Bar", "Foo", "straße"]
    assert reopened.link_count(b) == 3
    assert other.tags_of(c) == ["Bar", "foo"]
    assert reopened.find(["STRASSE"]) == [b]
    assert other.find(["FOO"]) == [c]
    with pytest.raises(DatabaseError):
        store.insert_tag("1", "fOO")
    store.insert_tag("3", "fOO")

    again = reopened.reopen()
    assert again.names() == ["Foo", "Bar", "straße"]
    assert other.names() == ["Bar", "foo"]


def _plan(backend: DatabaseBackend, query: str, params: Any) -> str:
    if backend.backend_type != BackendType.POSTGRESQL:
        return " | ".join(str(row["detail"]) for row in backend.execute("EXPLAIN QUERY PLAN " + query, params).rows)
    with backend.transaction() as conn:
        backend.execute("SET LOCAL enable_seqscan = off", connection=conn)
        rows = backend.execute("EXPLAIN " + query, params, connection=conn).rows
    return " | ".join(str(next(iter(row.values()))) for row in rows)


def test_tag_lookups_on_a_miss_use_an_index(store: _TagStore, monkeypatch: pytest.MonkeyPatch) -> None:
    store.ensure(["Foo", "Straße"])
    lookups: list[tuple[str, Any]] = []
    execute = store.backend.execute

    def recording_execute(query, params=None, *args, **kwargs):
        if query.lstrip().startswith("SELECT id") and f"FROM {store.table} " in query:
            lookups.append((query, params))
        return execute(query, params, *args, **kwargs)

    monkeypatch.setattr(store.backend, "execute", recording_execute)
    store.ensure(["missing", "STRASSE"])
    monkeypatch.undo()

    assert len(lookups) >= 3
    for query, params in lookups:
        plan = _plan(store.backend, query, params)
        assert re.search(r"Index Cond: \(+user_id|SEARCH .*\(user_id=\?", plan), (query, plan)


def test_collections_bootstrap_merges_duplicates_in_the_media_db_copy(backend: DatabaseBackend, tmp_path: Path) -> None:
    postgres = backend.backend_type == BackendType.POSTGRESQL
    media = MediaDatabase(db_path=str(tmp_path / "Media_DB_v2.db"), client_id="1", backend=backend if postgres else None)
    try:
        raw = CollectionsTags(backend)
        foo = raw.insert_tag("1", "Foo")
        foo_lower = raw.insert_tag("1", "foo")
        theirs = raw.insert_tag("2", "foo")
        raw.link(1, foo)
        raw.link(1, foo_lower)
        raw.link(2, foo_lower)
        raw.link(3, theirs)

        assert raw.db is not None

        tags = backend.execute("SELECT id, user_id, name FROM collection_tags ORDER BY id").rows
        assert [(row["id"], row["user_id"], row["name"]) for row in tags] == [(foo, "1", "Foo"), (theirs, "2", "foo")]
        links = backend.execute("SELECT item_id, tag_id FROM content_item_tags ORDER BY item_id, tag_id").rows
        assert [(row["item_id"], row["tag_id"]) for row in links] == [(1, foo), (2, foo), (3, theirs)]
    finally:
        media.close_connection()


def test_watchlist_tag_search_is_case_insensitive(backend: DatabaseBackend) -> None:
    store = WatchlistsTags(backend)
    store.ensure(["Foo", "bar"])

    for q in ("foo", "FOO", "Fo"):
        rows, total = store.db.list_tags(q=q, limit=10, offset=0)
        assert ([row.name for row in rows], total) == (["Foo"], 1)
    rows, _total = store.db.list_tags(q=None, limit=10, offset=0)
    assert [row.name for row in rows] == ["bar", "Foo"]

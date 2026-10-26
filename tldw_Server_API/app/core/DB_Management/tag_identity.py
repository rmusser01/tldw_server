"""Per-owner, case-insensitive tag identity for Collections and Watchlists tags (TASK-13950.5).

Collections ``collection_tags`` (the Media DB schema defines the same physical table) and
Watchlists ``tags`` share one shape: ``<table>(id, user_id, name)``, linked to tagged rows
through ``<link_table>(<link_column>, tag_id)``. Names keep their first-seen display casing
and compare by ``normalize_keyword_identity`` (NFC plus casefold), always within one owner:
lookups, the unique index and the duplicate merge all group by ``user_id`` first.

``ensure_tag_identity`` merges each owner's existing case-variant duplicates and then adds a
unique ``(user_id, name COLLATE NOCASE)`` index on SQLite or ``(user_id, LOWER(name))`` on
Postgres. Neither SQL fold equals the identity (NOCASE folds ASCII only; LOWER keeps the
German sharp s and folds the dotted capital I, which casefold keeps), so lookups take the SQL
matches for each name and its identity, compare identities over the owner's non-ASCII names
for the rest, and on Postgres finally reuse the row holding the name's LOWER() slot. That makes
"İstanbul" and "istanbul" one tag on Postgres and two on SQLite (a documented difference).
Filters resolve every matching id, since concurrent first inserts of two spellings the fold
index tells apart can both succeed; substring search (``tags_containing``) compares identities.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseBackend, DatabaseError
from tldw_Server_API.app.core.Utils.keyword_identity import dedupe_keywords, normalize_keyword_identity


@dataclass(frozen=True)
class TagTable:
    """A tag table, the table linking it to tagged rows, and that link table's row column."""

    table: str
    link_table: str
    link_column: str

    @property
    def identity_index(self) -> str:
        return f"ux_{self.table}_user_name_identity"


COLLECTION_TAGS = TagTable("collection_tags", "content_item_tags", "item_id")
WATCHLIST_TAGS = TagTable("tags", "source_tags", "source_id")


def _postgres(backend: DatabaseBackend) -> bool:
    return backend.backend_type == BackendType.POSTGRESQL


def _ids_by_identity(wanted: Mapping[str, str], rows: Iterable[Mapping[str, Any]]) -> dict[str, set[int]]:
    found: dict[str, set[int]] = {}
    for row in rows:
        identity = normalize_keyword_identity(str(row["name"]))
        if identity in wanted:
            found.setdefault(identity, set()).add(int(row["id"]))
    return found


def match_tag_ids(
    backend: DatabaseBackend,
    tags: TagTable,
    user_id: str,
    names: Iterable[object],
    *,
    all_matches: bool = False,
    connection: Any | None = None,
) -> dict[str, list[int]]:
    """Map the identity of each of ``names`` to the owner's matching tag ids, earliest first.

    Names with no tag are left out. An identity has one id unless concurrent first inserts of
    two spellings the fold index tells apart ("Straße", "STRASSE") both succeeded; filters pass
    ``all_matches`` so the non-ASCII names are read even when every identity already matched.
    """
    wanted = {normalize_keyword_identity(name): name for name in dedupe_keywords(names)}
    if not wanted:
        return {}
    postgres = _postgres(backend)
    terms = list(dict.fromkeys([*wanted.values(), *wanted]))
    if postgres:
        # Each row carries the term whose LOWER() slot it holds, so the slot holder needs no re-query.
        values = ", ".join("(?, LOWER(?))" for _ in terms)
        sql = (
            f"WITH q(term, folded) AS (VALUES {values}) "  # nosec B608 - fixed table names
            f"SELECT t.id, t.name, q.term FROM {tags.table} t JOIN q ON LOWER(t.name) = q.folded WHERE t.user_id = ?"
        )
        params: tuple[Any, ...] = (*(value for term in terms for value in (term, term)), user_id)
    else:
        marks = ", ".join("?" for _ in terms)
        sql = f"SELECT id, name FROM {tags.table} WHERE user_id = ? AND name COLLATE NOCASE IN ({marks})"  # nosec B608
        params = (user_id, *terms)
    rows = backend.execute(sql, params, connection=connection).rows
    found = _ids_by_identity(wanted, rows)
    if all_matches or len(found) < len(wanted):
        # ponytail: reads the owner's non-ASCII tag names (an index range scan on user_id) on a
        # miss or for a filter; a stored identity column if vocabularies grow large.
        non_ascii = "name ~ '[^[:ascii:]]'" if postgres else "name GLOB '*[^ -~]*'"
        extra = backend.execute(
            f"SELECT id, name FROM {tags.table} WHERE user_id = ? AND {non_ascii}",  # nosec B608
            (user_id,),
            connection=connection,
        ).rows
        found = _ids_by_identity(wanted, [*rows, *extra])
    for identity in wanted.keys() - found.keys():
        holders = {int(row["id"]) for row in rows if row.get("term") in (wanted[identity], identity)}
        if holders:
            found[identity] = holders
    return {identity: sorted(ids) for identity, ids in found.items()}


def tags_containing(
    backend: DatabaseBackend,
    tags: TagTable,
    user_id: str,
    text: str,
    *,
    connection: Any | None = None,
) -> list[dict[str, Any]]:
    """Return the owner's tag rows whose identity contains the identity of ``text``, by identity then id."""
    # ponytail: filters the owner's whole tag vocabulary in Python on each search; per-user
    # vocabularies are small, and SQL LIKE/LOWER cannot casefold on SQLite.
    needle = normalize_keyword_identity(text)
    rows = backend.execute(
        f"SELECT id, user_id, name FROM {tags.table} WHERE user_id = ?",  # nosec B608
        (user_id,),
        connection=connection,
    ).rows
    keyed = sorted((normalize_keyword_identity(str(row["name"])), int(row["id"]), dict(row)) for row in rows)
    return [row for identity, _tag_id, row in keyed if needle in identity]


def ensure_tag_ids(
    backend: DatabaseBackend,
    tags: TagTable,
    user_id: str,
    names: Iterable[object],
    *,
    connection: Any | None = None,
) -> list[int]:
    """Return the owner's tag ids for ``names`` (one per tag), creating missing tags with their given casing."""
    display = dedupe_keywords(names)
    found = match_tag_ids(backend, tags, user_id, display, connection=connection)
    ids = [
        (found.get(normalize_keyword_identity(name)) or [_insert_tag(backend, tags, user_id, name, connection)])[0]
        for name in display
    ]
    return list(dict.fromkeys(ids))


def _insert_tag(backend: DatabaseBackend, tags: TagTable, user_id: str, name: str, connection: Any | None) -> int:
    """Insert ``name``, or reuse the row already holding its SQL fold (a concurrent writer, or a
    Postgres LOWER() slot the identity keeps apart); DO NOTHING keeps a Postgres transaction alive."""
    fold = "LOWER(name) = LOWER(?)" if _postgres(backend) else "name COLLATE NOCASE = ?"
    backend.execute(
        f"INSERT INTO {tags.table} (user_id, name) VALUES (?, ?) ON CONFLICT DO NOTHING",  # nosec B608
        (user_id, name),
        connection=connection,
    )
    row = backend.execute(
        f"SELECT id FROM {tags.table} WHERE user_id = ? AND {fold} ORDER BY id LIMIT 1",  # nosec B608
        (user_id, name),
        connection=connection,
    ).first
    if row is None:
        raise DatabaseError(f"Failed to ensure a {tags.table} row")
    return int(row["id"])


def ensure_tag_identity(backend: DatabaseBackend, tags: TagTable) -> int:
    """Merge each owner's case-variant duplicate tags, add the case-insensitive unique index, and
    return how many rows were merged (0 once the index exists).

    Duplicates group by (owner, identity) and then by (owner, SQL fold), the key the index
    enforces; the earliest row of a group survives with its casing. The others' links move to it
    without duplicates and they are deleted: tag rows have no soft-delete, version or sync-log
    columns. The index gates this, so it runs once per database; on Postgres the table lock
    serializes concurrent bootstraps. The tables carry no row-level security policy, so an
    unscoped bootstrap sees every owner's rows.
    """
    postgres = _postgres(backend)
    exists = (
        "SELECT 1 AS found FROM pg_indexes WHERE schemaname = current_schema() AND indexname = ?"
        if postgres
        else "SELECT 1 AS found FROM sqlite_master WHERE type = 'index' AND name = ?"
    )
    if backend.execute(exists, (tags.identity_index,)).first:
        return 0
    with backend.transaction() as conn:
        if postgres:
            backend.execute(f"LOCK TABLE {tags.table} IN SHARE ROW EXCLUSIVE MODE", connection=conn)
        merged = _merge_duplicates(backend, tags, conn, lambda row: normalize_keyword_identity(str(row["name"])))
        merged += _merge_duplicates(backend, tags, conn, lambda row: str(row["folded"]))
        fold = "LOWER(name)" if postgres else "name COLLATE NOCASE"
        backend.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {tags.identity_index} ON {tags.table} (user_id, {fold})",
            connection=conn,
        )
    if merged:
        logger.info("Merged {} case-variant {} rows into their earliest variant", merged, tags.table)
    return merged


def _merge_duplicates(
    backend: DatabaseBackend,
    tags: TagTable,
    conn: Any,
    key: Callable[[Mapping[str, Any]], str],
) -> int:
    rows = backend.execute(
        f"SELECT id, user_id, name, LOWER(name) AS folded FROM {tags.table} ORDER BY id",  # nosec B608
        connection=conn,
    ).rows
    survivors: dict[tuple[str, str], int] = {}
    merged = 0
    for row in rows:
        tag_id = int(row["id"])
        survivor = survivors.setdefault((str(row["user_id"]), key(row)), tag_id)
        if survivor == tag_id:
            continue
        backend.execute(
            f"INSERT INTO {tags.link_table} ({tags.link_column}, tag_id) "  # nosec B608
            f"SELECT {tags.link_column}, ? FROM {tags.link_table} WHERE tag_id = ? ON CONFLICT DO NOTHING",
            (survivor, tag_id),
            connection=conn,
        )
        backend.execute(f"DELETE FROM {tags.link_table} WHERE tag_id = ?", (tag_id,), connection=conn)  # nosec B608
        backend.execute(f"DELETE FROM {tags.table} WHERE id = ?", (tag_id,), connection=conn)  # nosec B608
        merged += 1
    return merged

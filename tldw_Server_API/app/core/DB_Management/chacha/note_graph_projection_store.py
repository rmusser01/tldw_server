"""Persistence for bounded, local-only Notes graph projections."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.Notes.wikilinks import (
    MAX_WIKILINK_RENAME_NOTES,
    WikilinkProjection,
    WikilinkTitleCandidate,
    is_wikilink_title_reference_key,
    normalize_wikilink_title,
    parse_wikilinks,
    select_wikilink_title_target,
    wikilink_title_reference_key,
)

from ..ChaChaNotes_DB import BackendConnectionWrapper, BackendType

if TYPE_CHECKING:
    import sqlite3

    from ..ChaChaNotes_DB import CharactersRAGDB


@dataclass(frozen=True, slots=True)
class DirtyProjection:
    note_id: str
    generation: int


@dataclass(frozen=True, slots=True)
class NoteProjectionState:
    note_id: str
    source_version: int
    parser_version: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class NoteProjectionSource:
    note_id: str
    content: str
    version: int


@dataclass(frozen=True, slots=True)
class ProjectionStatus:
    parser_version: int
    rebuild_state: str
    rebuild_cursor: str | None


@dataclass(frozen=True, slots=True)
class WikilinkProjectionEdge:
    source_note_id: str
    target_note_id: str


@dataclass(frozen=True, slots=True)
class WikilinkTitleResolution:
    """How one ``[[Title]]`` link resolves against the owner's live notes."""

    note_id: str | None
    note_title: str | None
    candidate_count: int


@dataclass(frozen=True, slots=True)
class WikilinkTitleReferrer:
    """One live note whose text holds a ``[[Title]]`` link to some title."""

    note_id: str
    title: str
    version: int


@dataclass(frozen=True, slots=True)
class WikilinkTitleReferrerPage:
    """One page of referrers, ordered by note id, and the count across all pages."""

    total: int
    notes: tuple[WikilinkTitleReferrer, ...]
    # More referrers follow the last note of this page.
    has_more: bool = False


# Titles are matched in batches of LIKE prefilters; the exact rule runs in Python.
_TITLE_LOOKUP_BATCH = 100
# Notes one referrer page may list: what one rename rewrite request may name.
MAX_TITLE_REFERRER_PAGE = MAX_WIKILINK_RENAME_NOTES
# Notes re-projected inline when a title changes. Any remainder is queued for
# the maintenance worker, so one rename can't make a request unbounded.
MAX_INLINE_TITLE_REFERRER_REFRESH = 200


def _title_like_pattern(title_key: str) -> str:
    """Build a LIKE superset pattern for one normalized title key.

    LIKE wildcards are escaped, whitespace matches any run, and non-ASCII
    characters match any one character, because SQLite only lower-cases ASCII.
    """

    parts: list[str] = []
    for char in title_key:
        if char in "\\%_":
            parts.append("\\" + char)
        elif char == " ":
            parts.append("%")
        elif ord(char) > 127:
            parts.append("_")
        else:
            parts.append(char)
    return "%" + "".join(parts) + "%"


class NoteGraphProjectionStore:
    """Owner-bound projection, dirty-generation, and graph-revision store.

    ``note_wikilink_edges`` holds, per source note, its ``[[id:UUID]]`` targets,
    the note ids its ``[[Title]]`` links resolved to, and one title reference
    key per ``[[Title]]`` link (``wikilink_title_reference_key``). Reference
    keys never join a note row, so graph reads ignore them; they let a title
    change find the notes whose links must be re-resolved
    (``refresh_title_referrers``).
    """

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    @property
    def _postgres(self) -> bool:
        return self._db.backend_type == BackendType.POSTGRESQL

    def claim_dirty(
        self,
        *,
        limit: int,
        conn: sqlite3.Connection | BackendConnectionWrapper | None = None,
    ) -> tuple[DirtyProjection, ...]:
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        query = "SELECT note_id, generation FROM note_graph_dirty"
        params: tuple[object, ...]
        if self._postgres:
            query += " WHERE owner_user_id = ? ORDER BY note_id LIMIT ? FOR UPDATE SKIP LOCKED"
            params = (self._db.client_id, limit)
        else:
            query += " ORDER BY note_id LIMIT ?"
            params = (limit,)

        def execute(inner_conn: sqlite3.Connection | BackendConnectionWrapper):
            return tuple(
                DirtyProjection(str(row["note_id"]), int(row["generation"]))
                for row in inner_conn.execute(query, params).fetchall()
            )

        if conn is not None:
            return execute(conn)
        with self._db.transaction() as transaction_conn:
            return execute(transaction_conn)

    def replace_projection(
        self,
        *,
        note_id: str,
        source_version: int,
        projection: WikilinkProjection,
        claimed_generation: int | None = None,
        parser_version: int | None = None,
        bump_revision: bool = False,
        conn: sqlite3.Connection | BackendConnectionWrapper | None = None,
    ) -> bool:
        effective_parser_version = parser_version or projection.parser_version
        if source_version < 1 or effective_parser_version < 1:
            raise ValueError("source and parser versions must be positive")

        def execute(inner_conn: sqlite3.Connection | BackendConnectionWrapper) -> bool:
            owner_clause = " AND owner_user_id = ?" if self._postgres else ""
            delete_params: tuple[object, ...] = (note_id, self._db.client_id) if self._postgres else (note_id,)
            projected_targets = self._projected_targets(inner_conn, note_id=note_id, projection=projection)
            inner_conn.execute(
                f"DELETE FROM note_wikilink_edges WHERE source_note_id = ?{owner_clause}",  # nosec B608
                delete_params,
            )
            for target_note_id in projected_targets:
                if self._postgres:
                    inner_conn.execute(
                        "INSERT INTO note_wikilink_edges (owner_user_id, source_note_id, "
                        "target_note_id, source_version, parser_version) VALUES (?, ?, ?, ?, ?)",
                        (
                            self._db.client_id,
                            note_id,
                            target_note_id,
                            source_version,
                            effective_parser_version,
                        ),
                    )
                else:
                    inner_conn.execute(
                        "INSERT INTO note_wikilink_edges (source_note_id, target_note_id, "
                        "source_version, parser_version) VALUES (?, ?, ?, ?)",
                        (note_id, target_note_id, source_version, effective_parser_version),
                    )
            if self._postgres:
                inner_conn.execute(
                    "INSERT INTO note_graph_note_state (owner_user_id, note_id, source_version, "
                    "parser_version, truncated, updated_at) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(owner_user_id, note_id) DO UPDATE SET "
                    "source_version = excluded.source_version, parser_version = excluded.parser_version, "
                    "truncated = excluded.truncated, updated_at = CURRENT_TIMESTAMP",
                    (
                        self._db.client_id,
                        note_id,
                        source_version,
                        effective_parser_version,
                        projection.truncated,
                    ),
                )
            else:
                inner_conn.execute(
                    "INSERT INTO note_graph_note_state (note_id, source_version, parser_version, "
                    "truncated, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(note_id) DO UPDATE SET source_version = excluded.source_version, "
                    "parser_version = excluded.parser_version, truncated = excluded.truncated, "
                    "updated_at = CURRENT_TIMESTAMP",
                    (
                        note_id,
                        source_version,
                        effective_parser_version,
                        int(projection.truncated),
                    ),
                )
            cleared = self._clear_dirty(
                inner_conn,
                note_id=note_id,
                claimed_generation=claimed_generation,
            )
            if bump_revision:
                self._bump_revision(inner_conn)
            return cleared

        if conn is not None:
            return execute(conn)
        with self._db.transaction() as transaction_conn:
            return execute(transaction_conn)

    def _projected_targets(
        self,
        conn: sqlite3.Connection | BackendConnectionWrapper,
        *,
        note_id: str,
        projection: WikilinkProjection,
    ) -> tuple[str, ...]:
        """Return id targets, resolved title targets, and title reference keys."""

        targets = list(projection.target_note_ids)
        if projection.target_titles:
            resolved = self.resolve_wikilink_titles(
                projection.target_titles,
                exclude_note_id=note_id,
                conn=conn,
            )
            for title in projection.target_titles:
                match = resolved.get(title)
                if match is not None and match.note_id is not None:
                    targets.append(match.note_id)
            targets.extend(wikilink_title_reference_key(title) for title in projection.target_titles)
        return tuple(dict.fromkeys(targets))

    def resolve_wikilink_titles(
        self,
        titles: Iterable[str],
        *,
        exclude_note_id: str | None = None,
        conn: sqlite3.Connection | BackendConnectionWrapper | None = None,
    ) -> dict[str, WikilinkTitleResolution]:
        """Resolve ``[[Title]]`` link texts against the owner's live notes.

        Results are keyed by the given title text; blank titles are skipped.
        ``exclude_note_id`` (the linking note) is never a candidate. Ambiguity
        follows ``select_wikilink_title_target``: exact title, then the oldest
        note, then the lowest id.
        """

        requested = [
            title
            for title in dict.fromkeys(str(title) for title in titles)
            if normalize_wikilink_title(title)
        ]
        if not requested:
            return {}
        execute = conn.execute if conn is not None else self._db.execute_query
        candidates = self._title_candidates(
            execute,
            [normalize_wikilink_title(title) for title in requested],
            exclude_note_id=exclude_note_id,
        )
        resolutions: dict[str, WikilinkTitleResolution] = {}
        for title in requested:
            matches = candidates.get(normalize_wikilink_title(title), [])
            target = select_wikilink_title_target(title, matches)
            target_title = next((match.title for match in matches if match.note_id == target), None)
            resolutions[title] = WikilinkTitleResolution(target, target_title, len(matches))
        return resolutions

    def _title_candidates(
        self,
        execute: Callable[[str, tuple[object, ...]], Any],
        title_keys: list[str],
        *,
        exclude_note_id: str | None,
    ) -> dict[str, list[WikilinkTitleCandidate]]:
        ordered_keys = list(dict.fromkeys(title_keys))
        wanted = set(ordered_keys)
        by_key: dict[str, list[WikilinkTitleCandidate]] = {}
        seen_note_ids: set[str] = set()
        for offset in range(0, len(ordered_keys), _TITLE_LOOKUP_BATCH):
            batch = ordered_keys[offset : offset + _TITLE_LOOKUP_BATCH]
            title_clauses = " OR ".join("LOWER(title) LIKE ? ESCAPE '\\'" for _ in batch)
            query = f"SELECT id, title, created_at FROM notes WHERE deleted = ? AND ({title_clauses})"  # nosec B608
            params: list[object] = [False if self._postgres else 0]
            params.extend(_title_like_pattern(key) for key in batch)
            if self._postgres:
                query += " AND client_id = ?"
                params.append(self._db.client_id)
            if exclude_note_id:
                query += " AND id <> ?"
                params.append(exclude_note_id)
            for row in execute(query, tuple(params)).fetchall():
                candidate_id = str(row["id"])
                key = normalize_wikilink_title(row["title"])
                if candidate_id in seen_note_ids or key not in wanted:
                    continue
                seen_note_ids.add(candidate_id)
                by_key.setdefault(key, []).append(
                    WikilinkTitleCandidate(
                        note_id=candidate_id,
                        title=str(row["title"] or ""),
                        created_at=row["created_at"],
                    )
                )
        return by_key

    def get_live_note_titles(self, note_ids: Iterable[str]) -> dict[str, str]:
        """Map each live, owner-bound note id in ``note_ids`` to its title."""

        normalized = tuple(dict.fromkeys(str(note_id) for note_id in note_ids if note_id))
        if len(normalized) > 1_000:
            raise ValueError("note title lookup is limited to 1000 note IDs")
        titles: dict[str, str] = {}
        for offset in range(0, len(normalized), 400):
            batch = normalized[offset : offset + 400]
            placeholders = ",".join("?" for _ in batch)
            query = f"SELECT id, title FROM notes WHERE id IN ({placeholders}) AND deleted = ?"  # nosec B608
            params: list[object] = [*batch, False if self._postgres else 0]
            if self._postgres:
                query += " AND client_id = ?"
                params.append(self._db.client_id)
            for row in self._db.execute_query(query, tuple(params)).fetchall():
                titles[str(row["id"])] = str(row["title"] or "")
        return titles

    def get_note_title(
        self,
        note_id: str,
        *,
        conn: sqlite3.Connection | BackendConnectionWrapper,
    ) -> str | None:
        """Read one owner-bound note title, live or deleted."""

        query = "SELECT title FROM notes WHERE id = ?"
        params: tuple[object, ...] = (note_id,)
        if self._postgres:
            query += " AND client_id = ?"
            params += (self._db.client_id,)
        row = conn.execute(query, params).fetchone()
        return None if row is None else str(row["title"] or "")

    def list_live_note_ids_titled(self, title: str | None) -> tuple[str, ...]:
        """List the owner's live notes whose title matches ``title`` as a link would.

        Matching ignores case and extra whitespace, like ``[[Title]]`` links.
        """

        key = normalize_wikilink_title(title)
        if not key:
            return ()
        candidates = self._title_candidates(self._db.execute_query, [key], exclude_note_id=None)
        return tuple(sorted(candidate.note_id for candidate in candidates.get(key, ())))

    def list_title_referrers(
        self,
        title: str | None,
        *,
        exclude_note_id: str | None = None,
        unresolved_only: bool = False,
        after_note_id: str | None = None,
        limit: int = MAX_TITLE_REFERRER_PAGE,
    ) -> WikilinkTitleReferrerPage:
        """Page through the owner's live notes that hold a ``[[Title]]`` link to ``title``.

        The projection's title reference keys answer this without reading note
        text. Trashed notes are never listed. ``unresolved_only`` keeps only
        notes where the link names no live note, which is what a rename leaves
        behind; a note never answers its own link. ``total`` ignores the
        ``after_note_id`` cursor.
        """

        if not 1 <= limit <= MAX_TITLE_REFERRER_PAGE:
            raise ValueError(f"limit must be between 1 and {MAX_TITLE_REFERRER_PAGE}")
        if not normalize_wikilink_title(title):
            return WikilinkTitleReferrerPage(0, ())
        clauses = ["edge.target_note_id = ?", "note.deleted = ?"]
        params: list[object] = [wikilink_title_reference_key(title), False if self._postgres else 0]
        if self._postgres:
            clauses.append("edge.owner_user_id = ? AND note.client_id = ?")
            params.extend((self._db.client_id,) * 2)
        if exclude_note_id:
            clauses.append("note.id <> ?")
            params.append(exclude_note_id)
        if unresolved_only:
            answering = self.list_live_note_ids_titled(title)
            if len(answering) > 1:
                return WikilinkTitleReferrerPage(0, ())
            if answering:
                # Only that note's own link is unresolved: it can't answer itself.
                clauses.append("note.id = ?")
                params.append(answering[0])
        source = (
            "FROM note_wikilink_edges edge JOIN notes note ON note.id = edge.source_note_id "
            f"WHERE {' AND '.join(clauses)}"
        )
        total_row = self._db.execute_query(
            f"SELECT COUNT(*) AS referrer_count {source}",  # nosec B608 - fixed fragments; values stay bound.
            tuple(params),
        ).fetchone()
        total = int(total_row["referrer_count"]) if total_row else 0
        if total == 0:
            return WikilinkTitleReferrerPage(0, ())
        page_query = f"SELECT note.id, note.title, note.version {source}"  # nosec B608
        page_params = list(params)
        if after_note_id:
            page_query += " AND note.id > ?"
            page_params.append(after_note_id)
        page_query += " ORDER BY note.id LIMIT ?"
        # One row beyond the page tells whether another page follows.
        page_params.append(limit + 1)
        notes = tuple(
            WikilinkTitleReferrer(str(row["id"]), str(row["title"] or ""), int(row["version"]))
            for row in self._db.execute_query(page_query, tuple(page_params)).fetchall()
        )
        return WikilinkTitleReferrerPage(total, notes[:limit], has_more=len(notes) > limit)

    def refresh_title_referrers(
        self,
        titles: Iterable[str | None],
        *,
        conn: sqlite3.Connection | BackendConnectionWrapper,
    ) -> int:
        """Re-resolve the notes whose ``[[Title]]`` links name any of ``titles``.

        Call it in the same transaction after a note with one of these titles
        is created, renamed, deleted or restored. Up to
        ``MAX_INLINE_TITLE_REFERRER_REFRESH`` notes are re-projected inline and
        the rest are queued dirty for the maintenance worker. Returns the
        number of notes re-projected inline.
        """

        keys = list(
            dict.fromkeys(
                wikilink_title_reference_key(title)
                for title in titles
                if normalize_wikilink_title(title)
            )
        )
        if not keys:
            return 0
        placeholders = ",".join("?" for _ in keys)
        query = (
            "SELECT DISTINCT source_note_id FROM note_wikilink_edges "
            f"WHERE target_note_id IN ({placeholders})"  # nosec B608
        )
        params: list[object] = list(keys)
        if self._postgres:
            query += " AND owner_user_id = ?"
            params.append(self._db.client_id)
        query += " ORDER BY source_note_id"
        source_ids = [str(row["source_note_id"]) for row in conn.execute(query, tuple(params)).fetchall()]
        refreshed = 0
        for index, source_id in enumerate(source_ids):
            if index >= MAX_INLINE_TITLE_REFERRER_REFRESH:
                self._enqueue_dirty(conn, source_id)
                continue
            source = self.get_projection_source(source_id, conn=conn)
            if source is None:
                continue
            self.replace_projection(
                note_id=source.note_id,
                source_version=source.version,
                projection=parse_wikilinks(source.content, source_note_id=source.note_id),
                conn=conn,
            )
            refreshed += 1
        if source_ids:
            self._bump_revision(conn)
        return refreshed

    def mark_lifecycle(
        self,
        *,
        note_id: str,
        source_version: int,
        conn: sqlite3.Connection | BackendConnectionWrapper,
    ) -> None:
        if self._postgres:
            cursor = conn.execute(
                "UPDATE note_graph_note_state SET source_version = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE owner_user_id = ? AND note_id = ?",
                (source_version, self._db.client_id, note_id),
            )
        else:
            cursor = conn.execute(
                "UPDATE note_graph_note_state SET source_version = ?, updated_at = CURRENT_TIMESTAMP WHERE note_id = ?",
                (source_version, note_id),
            )
        if cursor.rowcount > 0:
            self._clear_dirty(conn, note_id=note_id, claimed_generation=None)
        # Deleting or restoring a note changes which links its title answers.
        self.refresh_title_referrers((self.get_note_title(note_id, conn=conn),), conn=conn)

    def list_outgoing(self, note_id: str) -> tuple[str, ...]:
        """List projected note-id targets, including unresolved ``[[id:UUID]]`` ones."""

        query = "SELECT target_note_id FROM note_wikilink_edges WHERE source_note_id = ?"
        params: tuple[object, ...] = (note_id,)
        if self._postgres:
            query += " AND owner_user_id = ?"
            params += (self._db.client_id,)
        query += " ORDER BY target_note_id"
        return tuple(
            str(row["target_note_id"])
            for row in self._db.execute_query(query, params).fetchall()
            if not is_wikilink_title_reference_key(str(row["target_note_id"]))
        )

    def list_live_outgoing(self, note_id: str) -> tuple[str, ...]:
        query = (
            "SELECT edge.target_note_id FROM note_wikilink_edges edge "
            "JOIN notes source ON source.id = edge.source_note_id "
            "JOIN notes target ON target.id = edge.target_note_id "
            "WHERE edge.source_note_id = ? AND source.deleted = ? AND target.deleted = ?"
        )
        params: tuple[object, ...] = (
            note_id,
            False if self._postgres else 0,
            False if self._postgres else 0,
        )
        if self._postgres:
            query += " AND edge.owner_user_id = ? AND source.client_id = ? AND target.client_id = ?"
            params += (self._db.client_id,) * 3
        query += " ORDER BY edge.target_note_id"
        return tuple(str(row["target_note_id"]) for row in self._db.execute_query(query, params).fetchall())

    def list_live_edges_for_notes(
        self,
        note_ids: tuple[str, ...] | list[str],
    ) -> tuple[WikilinkProjectionEdge, ...]:
        """Return live projected edges touching a bounded set of live notes."""

        normalized = tuple(dict.fromkeys(str(note_id) for note_id in note_ids))
        if not normalized:
            return ()
        if len(normalized) > 1_000:
            raise ValueError("note graph projection query is limited to 1000 note IDs")
        results: set[tuple[str, str]] = set()
        for offset in range(0, len(normalized), 400):
            batch = normalized[offset : offset + 400]
            placeholders = ",".join("?" for _ in batch)
            query = (
                "SELECT edge.source_note_id, edge.target_note_id "
                "FROM note_wikilink_edges edge "
                "JOIN notes source ON source.id = edge.source_note_id "
                "JOIN notes target ON target.id = edge.target_note_id "
                f"WHERE (edge.source_note_id IN ({placeholders}) OR "  # nosec B608
                f"edge.target_note_id IN ({placeholders})) "  # nosec B608
                "AND source.deleted = ? AND target.deleted = ?"
            )
            params: list[object] = [*batch, *batch]
            params.extend((False if self._postgres else 0,) * 2)
            if self._postgres:
                query += " AND edge.owner_user_id = ? AND source.client_id = ? AND target.client_id = ?"
                params.extend((self._db.client_id,) * 3)
            for row in self._db.execute_query(query, tuple(params)).fetchall():
                results.add((str(row["source_note_id"]), str(row["target_note_id"])))
        return tuple(WikilinkProjectionEdge(*edge) for edge in sorted(results))

    def list_orphan_note_ids(
        self,
        *,
        after_note_id: str | None,
        limit: int,
    ) -> tuple[str, ...]:
        """List live notes without live manual or projected note relationships."""

        if not 1 <= limit <= 201:
            raise ValueError("orphan limit must be between 1 and 201")
        live = False if self._postgres else 0
        note_owner_clause = " AND note.client_id = ?" if self._postgres else ""
        query = (
            "SELECT note.id AS note_id FROM notes note "
            f"WHERE note.deleted = ? AND note.id > ?{note_owner_clause} "  # nosec B608
            "AND NOT EXISTS ("
            "SELECT 1 FROM note_edges manual "
            "JOIN notes other ON other.id = manual.to_note_id "
            "WHERE manual.deleted = ? AND other.deleted = ? "
            "AND manual.from_note_id = note.id AND manual.user_id = ?"
        )
        params: list[object] = [live, after_note_id or ""]
        if self._postgres:
            params.append(self._db.client_id)
        params.extend((live, live, self._db.client_id))
        if self._postgres:
            query += " AND other.client_id = ?"
            params.append(self._db.client_id)
        query += (
            ") AND NOT EXISTS ("
            "SELECT 1 FROM note_edges manual "
            "JOIN notes other ON other.id = manual.from_note_id "
            "WHERE manual.deleted = ? AND other.deleted = ? "
            "AND manual.to_note_id = note.id AND manual.user_id = ?"
        )
        params.extend((live, live, self._db.client_id))
        if self._postgres:
            query += " AND other.client_id = ?"
            params.append(self._db.client_id)
        query += (
            ") AND NOT EXISTS ("
            "SELECT 1 FROM note_wikilink_edges derived "
            "JOIN notes other ON other.id = derived.target_note_id "
            "WHERE other.deleted = ? AND derived.source_note_id = note.id"
        )
        params.append(live)
        if self._postgres:
            query += " AND derived.owner_user_id = ? AND other.client_id = ?"
            params.extend((self._db.client_id,) * 2)
        query += (
            ") AND NOT EXISTS ("
            "SELECT 1 FROM note_wikilink_edges derived "
            "JOIN notes other ON other.id = derived.source_note_id "
            "WHERE other.deleted = ? AND derived.target_note_id = note.id"
        )
        params.append(live)
        if self._postgres:
            query += " AND derived.owner_user_id = ? AND other.client_id = ?"
            params.extend((self._db.client_id,) * 2)
        query += ") ORDER BY note.id LIMIT ?"
        params.append(limit)
        return tuple(str(row["note_id"]) for row in self._db.execute_query(query, tuple(params)).fetchall())

    def get_projection_source(
        self,
        note_id: str,
        *,
        conn: sqlite3.Connection | BackendConnectionWrapper,
    ) -> NoteProjectionSource | None:
        """Read one owner-bound note through the projection persistence boundary."""

        query = "SELECT id AS note_id, content, version FROM notes WHERE id = ?"
        params: tuple[object, ...] = (note_id,)
        if self._postgres:
            query += " AND client_id = ?"
            params += (self._db.client_id,)
        row = conn.execute(query, params).fetchone()
        if row is None:
            return None
        return NoteProjectionSource(
            note_id=str(row["note_id"]),
            content=str(row["content"] or ""),
            version=int(row["version"]),
        )

    def get_note_state(self, note_id: str) -> NoteProjectionState | None:
        query = "SELECT note_id, source_version, parser_version, truncated FROM note_graph_note_state WHERE note_id = ?"
        params: tuple[object, ...] = (note_id,)
        if self._postgres:
            query += " AND owner_user_id = ?"
            params += (self._db.client_id,)
        row = self._db.execute_query(query, params).fetchone()
        if row is None:
            return None
        return NoteProjectionState(
            note_id=str(row["note_id"]),
            source_version=int(row["source_version"]),
            parser_version=int(row["parser_version"]),
            truncated=bool(row["truncated"]),
        )

    def count_dirty(self, *, conn: Any | None = None) -> int:
        query = "SELECT COUNT(*) AS dirty_count FROM note_graph_dirty"
        params: tuple[object, ...] = ()
        if self._postgres:
            query += " WHERE owner_user_id = ?"
            params = (self._db.client_id,)
        cursor = conn.execute(query, params) if conn is not None else self._db.execute_query(query, params)
        return int(cursor.fetchone()["dirty_count"])

    def get_revision(self) -> int:
        if self._postgres:
            row = self._db.execute_query(
                "SELECT revision FROM note_graph_revisions WHERE owner_user_id = ?",
                (self._db.client_id,),
            ).fetchone()
        else:
            row = self._db.execute_query("SELECT revision FROM note_graph_revisions WHERE singleton_id = 1").fetchone()
        return int(row["revision"]) if row else 0

    def get_projection_status(self) -> ProjectionStatus:
        if self._postgres:
            row = self._db.execute_query(
                "SELECT parser_version, rebuild_state, rebuild_cursor "
                "FROM note_graph_projection_state WHERE owner_user_id = ?",
                (self._db.client_id,),
            ).fetchone()
        else:
            row = self._db.execute_query(
                "SELECT parser_version, rebuild_state, rebuild_cursor "
                "FROM note_graph_projection_state WHERE singleton_id = 1"
            ).fetchone()
        if row is None:
            return ProjectionStatus(1, "ready", None)
        return ProjectionStatus(
            int(row["parser_version"]),
            str(row["rebuild_state"]),
            row["rebuild_cursor"],
        )

    def prepare_rebuild(
        self,
        *,
        parser_version: int,
        conn: sqlite3.Connection | BackendConnectionWrapper,
    ) -> bool:
        status = self._projection_status(conn)
        if status.parser_version == parser_version and status.rebuild_state == "ready":
            return False
        if self._postgres:
            conn.execute(
                "INSERT INTO note_graph_projection_state (owner_user_id, parser_version, rebuild_state, "
                "rebuild_cursor, updated_at) VALUES (?, ?, 'pending', NULL, CURRENT_TIMESTAMP) "
                "ON CONFLICT(owner_user_id) DO UPDATE SET parser_version = excluded.parser_version, "
                "rebuild_state = 'pending', rebuild_cursor = NULL, updated_at = CURRENT_TIMESTAMP",
                (self._db.client_id, parser_version),
            )
        else:
            conn.execute(
                "UPDATE note_graph_projection_state SET parser_version = ?, rebuild_state = 'pending', "
                "rebuild_cursor = NULL, updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1",
                (parser_version,),
            )
        return True

    def queue_rebuild_page(
        self,
        *,
        limit: int,
        conn: sqlite3.Connection | BackendConnectionWrapper,
    ) -> int:
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        status = self._projection_status(conn)
        query = "SELECT id AS note_id FROM notes WHERE id > ?"
        params: list[object] = [status.rebuild_cursor or ""]
        if self._postgres:
            query += " AND client_id = ?"
            params.append(self._db.client_id)
        query += " ORDER BY id LIMIT ?"
        params.append(limit)
        note_ids = [str(row["note_id"]) for row in conn.execute(query, tuple(params)).fetchall()]
        for note_id in note_ids:
            self._enqueue_dirty(conn, note_id)
        cursor = note_ids[-1] if note_ids else status.rebuild_cursor
        if self._postgres:
            conn.execute(
                "UPDATE note_graph_projection_state SET rebuild_state = 'running', rebuild_cursor = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE owner_user_id = ?",
                (cursor, self._db.client_id),
            )
        else:
            conn.execute(
                "UPDATE note_graph_projection_state SET rebuild_state = 'running', rebuild_cursor = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1",
                (cursor,),
            )
        return len(note_ids)

    def finish_rebuild_if_idle(
        self,
        *,
        conn: sqlite3.Connection | BackendConnectionWrapper,
    ) -> bool:
        status = self._projection_status(conn)
        if status.rebuild_state not in {"pending", "running"} or self.count_dirty(conn=conn):
            return False
        query = "SELECT 1 FROM notes WHERE id > ?"
        params: tuple[object, ...] = (status.rebuild_cursor or "",)
        if self._postgres:
            query += " AND client_id = ?"
            params += (self._db.client_id,)
        query += " LIMIT 1"
        if conn.execute(query, params).fetchone() is not None:
            return False
        if self._postgres:
            conn.execute(
                "UPDATE note_graph_projection_state SET rebuild_state = 'ready', rebuild_cursor = NULL, "
                "updated_at = CURRENT_TIMESTAMP WHERE owner_user_id = ?",
                (self._db.client_id,),
            )
        else:
            conn.execute(
                "UPDATE note_graph_projection_state SET rebuild_state = 'ready', rebuild_cursor = NULL, "
                "updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1"
            )
        self._bump_revision(conn)
        return True

    def _projection_status(self, conn: Any) -> ProjectionStatus:
        if self._postgres:
            row = conn.execute(
                "SELECT parser_version, rebuild_state, rebuild_cursor "
                "FROM note_graph_projection_state WHERE owner_user_id = ?",
                (self._db.client_id,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO note_graph_projection_state (owner_user_id) VALUES (?)",
                    (self._db.client_id,),
                )
                return ProjectionStatus(1, "ready", None)
        else:
            row = conn.execute(
                "SELECT parser_version, rebuild_state, rebuild_cursor "
                "FROM note_graph_projection_state WHERE singleton_id = 1"
            ).fetchone()
        return ProjectionStatus(
            int(row["parser_version"]),
            str(row["rebuild_state"]),
            row["rebuild_cursor"],
        )

    def _enqueue_dirty(self, conn: Any, note_id: str) -> None:
        if self._postgres:
            conn.execute(
                "INSERT INTO note_graph_dirty (owner_user_id, note_id, generation, last_modified) "
                "VALUES (?, ?, 1, CURRENT_TIMESTAMP) ON CONFLICT(owner_user_id, note_id) "
                "DO UPDATE SET generation = note_graph_dirty.generation + 1, "
                "last_modified = CURRENT_TIMESTAMP",
                (self._db.client_id, note_id),
            )
        else:
            conn.execute(
                "INSERT INTO note_graph_dirty (note_id, generation, last_modified) "
                "VALUES (?, 1, CURRENT_TIMESTAMP) ON CONFLICT(note_id) DO UPDATE SET "
                "generation = note_graph_dirty.generation + 1, last_modified = CURRENT_TIMESTAMP",
                (note_id,),
            )

    def _clear_dirty(
        self,
        conn: Any,
        *,
        note_id: str,
        claimed_generation: int | None,
    ) -> bool:
        query = "DELETE FROM note_graph_dirty WHERE note_id = ?"
        params: list[object] = [note_id]
        if self._postgres:
            query += " AND owner_user_id = ?"
            params.append(self._db.client_id)
        if claimed_generation is not None:
            query += " AND generation = ?"
            params.append(claimed_generation)
        cursor = conn.execute(query, tuple(params))
        return cursor.rowcount > 0

    def _bump_revision(self, conn: Any) -> None:
        if self._postgres:
            conn.execute("SELECT notes_graph_bump_revision(?)", (self._db.client_id,))
        else:
            conn.execute(
                "UPDATE note_graph_revisions SET revision = revision + 1, "
                "updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1"
            )


__all__ = [
    "MAX_INLINE_TITLE_REFERRER_REFRESH",
    "MAX_TITLE_REFERRER_PAGE",
    "DirtyProjection",
    "NoteGraphProjectionStore",
    "NoteProjectionSource",
    "NoteProjectionState",
    "ProjectionStatus",
    "WikilinkProjectionEdge",
    "WikilinkTitleReferrer",
    "WikilinkTitleReferrerPage",
    "WikilinkTitleResolution",
]

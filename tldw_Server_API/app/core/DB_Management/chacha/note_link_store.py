from __future__ import annotations

"""Owner-bound persistence for canonical explicit Notes links."""

import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Sync.v2.models import normalize_sync_timestamp
from tldw_Server_API.app.core.Sync.v2.notes_link import (
    NotesLinkValidationError,
    parse_notes_link_payload,
    validate_notes_link_object_id,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@dataclass(frozen=True)
class NotesLink:
    edge_id: str
    owner_user_id: str
    source_note_id: str
    target_note_id: str
    type: str
    directed: bool
    weight: float
    label: str | None
    properties: Mapping[str, object]
    created_at: str
    last_modified: str
    created_by: str
    version: int
    deleted: bool
    deleted_at: str | None


@dataclass(frozen=True)
class NotesLinkMutationResult:
    link: NotesLink
    changed: bool


class NotesLinkStore:
    """Apply canonical link lifecycle transitions in one owner's product DB."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    @property
    def _owner_id(self) -> str:
        return str(self._db.client_id)

    def _deleted_value(self, deleted: bool) -> bool | int:
        return deleted if self._db.backend_type == BackendType.POSTGRESQL else int(deleted)

    @staticmethod
    def _encode_properties(properties: object) -> str:
        """Encode one canonical properties object identically for every lifecycle write."""

        return json.dumps(
            properties,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @staticmethod
    def legacy_dict(link: NotesLink) -> dict[str, object]:
        """Return the historical manual-edge response shape."""

        metadata = dict(link.properties)
        if link.label is not None:
            metadata["label"] = link.label
        return {
            "edge_id": link.edge_id,
            "user_id": link.owner_user_id,
            "from_note_id": link.source_note_id,
            "to_note_id": link.target_note_id,
            "type": link.type,
            "directed": link.directed,
            "weight": link.weight,
            "created_at": link.created_at,
            "created_by": link.created_by,
            "metadata": metadata,
        }

    @staticmethod
    def _timestamp(value: object | None) -> str | None:
        return normalize_sync_timestamp(value)

    @classmethod
    def _from_row(cls, row: Mapping[str, object]) -> NotesLink:
        record = dict(row)
        raw_properties = record.get("properties")
        if isinstance(raw_properties, str):
            try:
                raw_properties = json.loads(raw_properties)
            except (json.JSONDecodeError, TypeError) as exc:
                raise CharactersRAGDBError("Stored notes.link properties are invalid") from exc
        if not isinstance(raw_properties, dict):
            raise CharactersRAGDBError("Stored notes.link properties are invalid")
        created_at = cls._timestamp(record.get("created_at"))
        last_modified = cls._timestamp(record.get("last_modified"))
        if created_at is None or last_modified is None:
            raise CharactersRAGDBError("Stored notes.link timestamps are invalid")
        return NotesLink(
            edge_id=str(record["edge_id"]),
            owner_user_id=str(record["user_id"]),
            source_note_id=str(record["from_note_id"]),
            target_note_id=str(record["to_note_id"]),
            type=str(record["type"]),
            directed=bool(record["directed"]),
            weight=float(record["weight"]),
            label=str(record["label"]) if record.get("label") is not None else None,
            properties=dict(raw_properties),
            created_at=created_at,
            last_modified=last_modified,
            created_by=str(record["created_by"]),
            version=int(record["version"]),
            deleted=bool(record["deleted"]),
            deleted_at=cls._timestamp(record.get("deleted_at")),
        )

    @staticmethod
    def _validated(
        operation: str,
        edge_id: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        try:
            validate_notes_link_object_id(edge_id)
            return parse_notes_link_payload(operation, payload)
        except NotesLinkValidationError as exc:
            raise InputError(str(exc)) from exc

    def _get_locked(
        self,
        conn: Any,
        edge_id: str,
        *,
        for_update: bool = False,
    ) -> NotesLink | None:
        query = (
            "SELECT edge.edge_id, edge.user_id, edge.from_note_id, edge.to_note_id, edge.type, "
            "edge.directed, edge.weight, edge.label, edge.properties, edge.created_at, "
            "edge.last_modified, edge.created_by, edge.version, edge.deleted, edge.deleted_at "
            "FROM note_edges edge JOIN notes source ON source.id = edge.from_note_id "
            "JOIN notes target ON target.id = edge.to_note_id "
            "WHERE edge.edge_id = ? AND edge.user_id = ? "
            "AND source.client_id = ? AND target.client_id = ?"
        )
        if for_update and self._db.backend_type == BackendType.POSTGRESQL:
            query += " FOR UPDATE OF edge"
        row = conn.execute(
            query,
            (edge_id, self._owner_id, self._owner_id, self._owner_id),
        ).fetchone()
        return self._from_row(row) if row else None

    def get(self, edge_id: str, *, conn: Any | None = None) -> NotesLink | None:
        """Return one owner-scoped link, including tombstones."""

        try:
            validate_notes_link_object_id(edge_id)
        except NotesLinkValidationError as exc:
            raise InputError(str(exc)) from exc
        context = nullcontext(conn) if conn is not None else self._db.transaction()
        with context as transaction_conn:
            return self._get_locked(transaction_conn, edge_id)

    def _validate_endpoints_locked(
        self,
        conn: Any,
        payload: Mapping[str, object],
        *,
        allow_deleted: bool,
        for_update: bool = False,
    ) -> None:
        query = "SELECT id, client_id, deleted FROM notes WHERE id IN (?, ?)"
        if for_update and self._db.backend_type == BackendType.POSTGRESQL:
            query += " FOR UPDATE"
        rows = conn.execute(
            query,
            (payload["source_note_id"], payload["target_note_id"]),
        ).fetchall()
        by_id = {str(row["id"]): dict(row) for row in rows}
        if set(by_id) != {payload["source_note_id"], payload["target_note_id"]}:
            raise InputError("notes.link requires two existing owned endpoints")
        if any(str(row["client_id"]) != self._owner_id for row in by_id.values()):
            raise InputError("notes.link requires two existing owned endpoints")
        if not allow_deleted and any(bool(row["deleted"]) for row in by_id.values()):
            raise InputError("notes.link public mutations require two live endpoints")

    def validate_public_endpoints(self, source_note_id: str, target_note_id: str) -> None:
        """Require two live owner-scoped endpoints without mutating product state."""

        with self._db.transaction() as conn:
            self._validate_endpoints_locked(
                conn,
                {
                    "source_note_id": source_note_id,
                    "target_note_id": target_note_id,
                },
                allow_deleted=False,
            )

    @staticmethod
    def _identity_matches(link: NotesLink, payload: Mapping[str, object]) -> bool:
        return (
            link.source_note_id == payload["source_note_id"]
            and link.target_note_id == payload["target_note_id"]
            and link.type == payload["type"]
            and link.directed is payload["directed"]
            and link.created_at == payload["created_at"]
            and link.created_by == payload["created_by"]
        )

    @staticmethod
    def _live_postcondition_matches(link: NotesLink, payload: Mapping[str, object]) -> bool:
        return (
            not link.deleted
            and NotesLinkStore._identity_matches(link, payload)
            and link.weight == payload["weight"]
            and link.label == payload["label"]
            and dict(link.properties) == payload["properties"]
            and link.last_modified == payload["last_modified"]
            and link.deleted_at is None
        )

    @staticmethod
    def _tombstone_postcondition_matches(
        link: NotesLink,
        payload: Mapping[str, object],
    ) -> bool:
        return (
            link.deleted
            and NotesLinkStore._identity_matches(link, payload)
            and link.weight == payload["weight"]
            and link.label == payload["label"]
            and dict(link.properties) == payload["properties"]
            and link.last_modified == payload["last_modified"]
            and link.deleted_at == payload["deleted_at"]
        )

    @staticmethod
    def _require_identity(link: NotesLink, payload: Mapping[str, object]) -> None:
        if not NotesLinkStore._identity_matches(link, payload):
            raise InputError("notes.link identity and creation provenance are immutable")

    @staticmethod
    def _require_version(link: NotesLink, expected_version: int | None) -> None:
        if expected_version != link.version:
            raise ConflictError(
                "notes.link version conflict",
                entity="note_edges",
                entity_id=link.edge_id,
            )

    @staticmethod
    def _require_cas_update(cursor: Any, edge_id: str) -> None:
        if cursor.rowcount != 1:
            raise ConflictError(
                "notes.link version conflict",
                entity="note_edges",
                entity_id=edge_id,
            )

    def upsert(
        self,
        *,
        edge_id: str,
        payload: Mapping[str, object],
        expected_version: int | None,
        allow_deleted_endpoints: bool = False,
        conn: Any | None = None,
        before: Callable[[Any], None] | None = None,
        after: Callable[[Any, str], None] | None = None,
    ) -> NotesLinkMutationResult:
        """Create or update a live link, with exact-replay short-circuiting."""

        normalized = self._validated("upsert", edge_id, payload)
        context = nullcontext(conn) if conn is not None else self._db.transaction()
        try:
            with context as transaction_conn:
                guarded = before is not None or after is not None
                existing = self._get_locked(
                    transaction_conn,
                    edge_id,
                    for_update=guarded,
                )
                self._validate_endpoints_locked(
                    transaction_conn,
                    normalized,
                    allow_deleted=allow_deleted_endpoints,
                    for_update=guarded,
                )
                if existing is None:
                    if expected_version not in {None, 0}:
                        raise ConflictError(
                            "notes.link version conflict",
                            entity="note_edges",
                            entity_id=edge_id,
                        )
                    duplicate = transaction_conn.execute(
                        "SELECT edge_id FROM note_edges WHERE user_id = ? AND type = ? AND directed = ? "
                        "AND from_note_id = ? AND to_note_id = ? LIMIT 1",
                        (
                            self._owner_id,
                            normalized["type"],
                            self._deleted_value(bool(normalized["directed"])),
                            normalized["source_note_id"],
                            normalized["target_note_id"],
                        ),
                    ).fetchone()
                    if duplicate:
                        raise ConflictError(
                            "notes.link logical identity already exists",
                            entity="note_edges",
                            entity_id=str(duplicate["edge_id"]),
                        )
                    if before is not None:
                        before(transaction_conn)
                    properties = self._encode_properties(normalized["properties"])
                    transaction_conn.execute(
                        "INSERT INTO note_edges(edge_id, user_id, from_note_id, to_note_id, type, "
                        "directed, weight, label, properties, created_at, last_modified, created_by, "
                        "version, deleted, deleted_at, metadata) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, NULL, ?)",
                        (
                            edge_id,
                            self._owner_id,
                            normalized["source_note_id"],
                            normalized["target_note_id"],
                            normalized["type"],
                            self._deleted_value(bool(normalized["directed"])),
                            normalized["weight"],
                            normalized["label"],
                            properties,
                            normalized["created_at"],
                            normalized["last_modified"],
                            normalized["created_by"],
                            self._deleted_value(False),
                            properties,
                        ),
                    )
                    created = self._get_locked(transaction_conn, edge_id)
                    if created is None or not self._live_postcondition_matches(
                        created, normalized
                    ):
                        raise CharactersRAGDBError("Inserted notes.link was not found")
                    if after is not None:
                        after(transaction_conn, edge_id)
                    return NotesLinkMutationResult(created, True)

                self._require_identity(existing, normalized)
                if self._live_postcondition_matches(existing, normalized):
                    if before is not None:
                        before(transaction_conn)
                    if after is not None:
                        after(transaction_conn, edge_id)
                    return NotesLinkMutationResult(existing, False)
                if existing.deleted:
                    raise ConflictError(
                        "notes.link is tombstoned; use restore",
                        entity="note_edges",
                        entity_id=edge_id,
                    )
                self._require_version(existing, expected_version)
                if before is not None:
                    before(transaction_conn)
                properties = self._encode_properties(normalized["properties"])
                cursor = transaction_conn.execute(
                    "UPDATE note_edges SET weight = ?, label = ?, properties = ?, metadata = ?, "
                    "last_modified = ?, version = ? WHERE edge_id = ? AND user_id = ? "
                    "AND version = ?",
                    (
                        normalized["weight"],
                        normalized["label"],
                        properties,
                        properties,
                        normalized["last_modified"],
                        existing.version + 1,
                        edge_id,
                        self._owner_id,
                        existing.version,
                    ),
                )
                self._require_cas_update(cursor, edge_id)
                updated = self._get_locked(transaction_conn, edge_id)
                if updated is None or not self._live_postcondition_matches(
                    updated, normalized
                ):
                    raise CharactersRAGDBError("Updated notes.link was not found")
                if after is not None:
                    after(transaction_conn, edge_id)
                return NotesLinkMutationResult(updated, True)
        except (sqlite3.IntegrityError, BackendDatabaseError) as exc:
            message = str(exc).lower()
            if "unique" in message or "duplicate" in message:
                raise ConflictError(
                    "notes.link logical identity already exists",
                    entity="note_edges",
                    entity_id=edge_id,
                ) from exc
            raise CharactersRAGDBError(f"Failed to persist notes.link: {exc}") from exc

    def tombstone(
        self,
        *,
        edge_id: str,
        payload: Mapping[str, object],
        expected_version: int,
        conn: Any | None = None,
    ) -> NotesLinkMutationResult:
        """Soft-delete one existing link without requiring live endpoints."""

        normalized = self._validated("tombstone", edge_id, payload)
        context = nullcontext(conn) if conn is not None else self._db.transaction()
        with context as transaction_conn:
            existing = self._get_locked(transaction_conn, edge_id)
            if existing is None:
                raise InputError("Cannot tombstone an unknown notes.link")
            self._require_identity(existing, normalized)
            if self._tombstone_postcondition_matches(existing, normalized):
                return NotesLinkMutationResult(existing, False)
            self._require_version(existing, expected_version)
            properties = self._encode_properties(normalized["properties"])
            cursor = transaction_conn.execute(
                "UPDATE note_edges SET weight = ?, label = ?, properties = ?, metadata = ?, "
                "last_modified = ?, deleted = ?, deleted_at = ?, version = ? "
                "WHERE edge_id = ? AND user_id = ? AND version = ?",
                (
                    normalized["weight"],
                    normalized["label"],
                    properties,
                    properties,
                    normalized["last_modified"],
                    self._deleted_value(True),
                    normalized["deleted_at"],
                    existing.version + 1,
                    edge_id,
                    self._owner_id,
                    existing.version,
                ),
            )
            self._require_cas_update(cursor, edge_id)
            updated = self._get_locked(transaction_conn, edge_id)
            if updated is None:
                raise CharactersRAGDBError("Tombstoned notes.link was not found")
            return NotesLinkMutationResult(updated, True)

    def restore(
        self,
        *,
        edge_id: str,
        payload: Mapping[str, object],
        expected_version: int,
        allow_deleted_endpoints: bool = False,
        conn: Any | None = None,
    ) -> NotesLinkMutationResult:
        """Restore one tombstoned link with its immutable identity intact."""

        normalized = self._validated("upsert", edge_id, payload)
        context = nullcontext(conn) if conn is not None else self._db.transaction()
        with context as transaction_conn:
            existing = self._get_locked(transaction_conn, edge_id)
            if existing is None:
                raise InputError("Cannot restore an unknown notes.link")
            self._require_identity(existing, normalized)
            if self._live_postcondition_matches(existing, normalized):
                return NotesLinkMutationResult(existing, False)
            if not existing.deleted:
                raise ConflictError(
                    "notes.link is not tombstoned",
                    entity="note_edges",
                    entity_id=edge_id,
                )
            self._require_version(existing, expected_version)
            self._validate_endpoints_locked(
                transaction_conn,
                normalized,
                allow_deleted=allow_deleted_endpoints,
            )
            properties = self._encode_properties(normalized["properties"])
            cursor = transaction_conn.execute(
                "UPDATE note_edges SET weight = ?, label = ?, properties = ?, metadata = ?, "
                "last_modified = ?, deleted = ?, deleted_at = NULL, version = ? "
                "WHERE edge_id = ? AND user_id = ? AND version = ?",
                (
                    normalized["weight"],
                    normalized["label"],
                    properties,
                    properties,
                    normalized["last_modified"],
                    self._deleted_value(False),
                    existing.version + 1,
                    edge_id,
                    self._owner_id,
                    existing.version,
                ),
            )
            self._require_cas_update(cursor, edge_id)
            restored = self._get_locked(transaction_conn, edge_id)
            if restored is None:
                raise CharactersRAGDBError("Restored notes.link was not found")
            return NotesLinkMutationResult(restored, True)

    def list_for_notes(
        self,
        note_ids: Sequence[str],
        *,
        include_deleted_links: bool = False,
        include_deleted_endpoints: bool = False,
        conn: Any | None = None,
    ) -> tuple[NotesLink, ...]:
        """Return owner-scoped links touching any requested note identity."""

        normalized_ids = tuple(dict.fromkeys(str(note_id) for note_id in note_ids))
        if not normalized_ids:
            return ()
        if len(normalized_ids) > 10_000:
            raise InputError("notes.link query may include at most 10000 note IDs")
        context = nullcontext(conn) if conn is not None else self._db.transaction()
        with context as transaction_conn:
            results: dict[str, NotesLink] = {}
            for offset in range(0, len(normalized_ids), 400):
                batch = normalized_ids[offset : offset + 400]
                placeholders = ",".join("?" for _ in batch)
                query = (
                    "SELECT edge.edge_id, edge.user_id, edge.from_note_id, edge.to_note_id, edge.type, "
                    "edge.directed, edge.weight, edge.label, edge.properties, edge.created_at, "
                    "edge.last_modified, edge.created_by, edge.version, edge.deleted, edge.deleted_at "
                    "FROM note_edges edge JOIN notes source ON source.id = edge.from_note_id "
                    "JOIN notes target ON target.id = edge.to_note_id WHERE edge.user_id = ? "
                    "AND source.client_id = ? AND target.client_id = ? "
                    f"AND (edge.from_note_id IN ({placeholders}) OR "  # nosec B608
                    f"edge.to_note_id IN ({placeholders}))"  # nosec B608
                )
                params: list[object] = [self._owner_id, self._owner_id, self._owner_id]
                params.extend(batch)
                params.extend(batch)
                if not include_deleted_links:
                    query += " AND edge.deleted = ?"
                    params.append(self._deleted_value(False))
                if not include_deleted_endpoints:
                    query += " AND source.deleted = ? AND target.deleted = ?"
                    params.extend((self._deleted_value(False), self._deleted_value(False)))
                for row in transaction_conn.execute(query, tuple(params)).fetchall():
                    link = self._from_row(row)
                    results[link.edge_id] = link
            return tuple(results[edge_id] for edge_id in sorted(results))

    def snapshot(self, *, conn: Any | None = None) -> tuple[NotesLink, ...]:
        """Return all live and tombstoned owner links for trusted bootstrap."""

        context = nullcontext(conn) if conn is not None else self._db.transaction()
        with context as transaction_conn:
            rows = transaction_conn.execute(
                "SELECT edge.edge_id, edge.user_id, edge.from_note_id, edge.to_note_id, edge.type, "
                "edge.directed, edge.weight, edge.label, edge.properties, edge.created_at, "
                "edge.last_modified, edge.created_by, edge.version, edge.deleted, edge.deleted_at "
                "FROM note_edges edge JOIN notes source ON source.id = edge.from_note_id "
                "JOIN notes target ON target.id = edge.to_note_id WHERE edge.user_id = ? "
                "AND source.client_id = ? AND target.client_id = ? ORDER BY edge.edge_id",
                (self._owner_id, self._owner_id, self._owner_id),
            ).fetchall()
            return tuple(self._from_row(row) for row in rows)

    def list_page(
        self,
        *,
        after_edge_id: str | None,
        limit: int,
        include_deleted_links: bool = False,
        include_deleted_endpoints: bool = False,
    ) -> tuple[NotesLink, ...]:
        """Return one keyset page of owner-scoped explicit links."""

        if not 1 <= limit <= 201:
            raise InputError("notes.link page limit must be between 1 and 201")
        query = (
            "SELECT edge.edge_id, edge.user_id, edge.from_note_id, edge.to_note_id, edge.type, "
            "edge.directed, edge.weight, edge.label, edge.properties, edge.created_at, "
            "edge.last_modified, edge.created_by, edge.version, edge.deleted, edge.deleted_at "
            "FROM note_edges edge JOIN notes source ON source.id = edge.from_note_id "
            "JOIN notes target ON target.id = edge.to_note_id WHERE edge.user_id = ? "
            "AND source.client_id = ? AND target.client_id = ? "
            "AND edge.type = 'manual' AND edge.edge_id > ?"
        )
        params: list[object] = [
            self._owner_id,
            self._owner_id,
            self._owner_id,
            after_edge_id or "",
        ]
        if not include_deleted_links:
            query += " AND edge.deleted = ?"
            params.append(self._deleted_value(False))
        if not include_deleted_endpoints:
            query += " AND source.deleted = ? AND target.deleted = ?"
            params.extend((self._deleted_value(False), self._deleted_value(False)))
        query += " ORDER BY edge.edge_id LIMIT ?"
        params.append(limit)
        rows = self._db.execute_query(query, tuple(params)).fetchall()
        return tuple(self._from_row(row) for row in rows)


__all__ = ["NotesLink", "NotesLinkMutationResult", "NotesLinkStore"]

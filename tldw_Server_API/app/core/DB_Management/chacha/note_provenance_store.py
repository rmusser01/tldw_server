"""Owner-scoped independently versioned Knowledge provenance sidecars."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError, InputError
from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import (
    canonical_notes_provenance_json,
    notes_provenance_object_hash,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def notes_provenance_schema_sql(*, postgres: bool) -> str:
    """Migration-owned one-to-one parent binding on both database engines."""
    version_type = "BIGINT" if postgres else "INTEGER"
    deleted_type = "BOOLEAN" if postgres else "INTEGER CHECK(deleted IN (0, 1))"
    deleted_default = "FALSE" if postgres else "0"
    return f"""CREATE TABLE IF NOT EXISTS notes_knowledge_provenance (
        owner_user_id TEXT NOT NULL,
        note_id TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        version {version_type} NOT NULL CHECK(version > 0),
        object_hash TEXT NOT NULL,
        deleted {deleted_type} NOT NULL DEFAULT {deleted_default},
        PRIMARY KEY(owner_user_id, note_id),
        FOREIGN KEY(owner_user_id, note_id) REFERENCES notes(client_id, id) ON DELETE CASCADE
    )"""


class NoteProvenanceStore:
    """All reads and writes bind the sidecar and its parent to the DB owner."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def _parent(self, note_id: str, conn: Any, *, lock: bool = False) -> dict[str, Any] | None:
        """Read or lock the retained parent with an explicit authenticated owner."""
        query = "SELECT id, deleted FROM notes WHERE id = ? AND client_id = ?"
        if lock and self._db.backend_type == BackendType.POSTGRESQL:
            query += " FOR UPDATE"
        row = conn.execute(query, (note_id, self._db.owner_user_id)).fetchone()
        return dict(row) if row else None

    def _record(self, note_id: str, conn: Any, *, lock: bool = False) -> dict[str, Any] | None:
        """Read the owner sidecar after the caller has checked its parent."""
        query = "SELECT payload_json, version, object_hash, deleted FROM notes_knowledge_provenance WHERE owner_user_id = ? AND note_id = ?"
        if lock and self._db.backend_type == BackendType.POSTGRESQL:
            query += " FOR UPDATE"
        row = conn.execute(query, (self._db.owner_user_id, note_id)).fetchone()
        if row is None:
            return None
        return {
            "payload": json.loads(row["payload_json"]),
            "version": row["version"],
            "object_hash": row["object_hash"],
            "deleted": bool(row["deleted"]),
        }

    def get(self, note_id: str, include_deleted: bool = False, conn: Any = None) -> dict[str, Any] | None:
        """Read owned provenance, hiding deleted records and parents by default."""
        with nullcontext(conn) if conn is not None else self._db.transaction() as connection:
            parent = self._parent(note_id, connection)
            if parent is None or (parent["deleted"] and not include_deleted):
                return None
            record = self._record(note_id, connection)
            return record if record and (include_deleted or not record["deleted"]) else None

    def list_parent_notes(self, *, after_note_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        """Page this owner's retained parents, including trash, in stable ID order."""
        if type(limit) is not int or not 1 <= limit <= 200:
            raise InputError("Knowledge provenance source page limit must be 1..200")
        with self._db.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM notes WHERE client_id = ? AND id > ? ORDER BY id LIMIT ?",
                (self._db.owner_user_id, after_note_id or "", limit),
            ).fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def _version(value: int, *, absent: bool = False) -> None:
        """Require independent positive safe integer revisions, or zero for absence."""
        if type(value) is not int or value < (0 if absent else 1) or value > 9_007_199_254_740_991:
            raise InputError("Invalid Knowledge provenance version")

    def _write(
        self, note_id: str, payload: Mapping[str, object], version: int, deleted: bool, conn: Any
    ) -> dict[str, Any]:
        """Write bounded evidence and its canonical lifecycle hash on caller connection."""
        encoded = canonical_notes_provenance_json(payload)
        digest = notes_provenance_object_hash(payload, deleted)
        flag = deleted if self._db.backend_type == BackendType.POSTGRESQL else int(deleted)
        conn.execute(
            """INSERT INTO notes_knowledge_provenance(owner_user_id, note_id, payload_json, version, object_hash, deleted)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_user_id, note_id) DO UPDATE SET payload_json = excluded.payload_json,
                version = excluded.version, object_hash = excluded.object_hash, deleted = excluded.deleted
            WHERE notes_knowledge_provenance.owner_user_id = excluded.owner_user_id""",
            (self._db.owner_user_id, note_id, encoded, version, digest, flag),
        )
        return {"payload": json.loads(encoded), "version": version, "object_hash": digest, "deleted": deleted}

    def put(
        self,
        note_id: str,
        payload: Mapping[str, object],
        expected_version: int,
        conn: Any = None,
        *,
        deleted: bool = False,
        restore: bool = False,
    ) -> dict[str, Any]:
        """Replace at an exact independent version, with explicit tombstone restore."""
        self._version(expected_version, absent=True)
        if type(deleted) is not bool or type(restore) is not bool or (deleted and restore):
            raise InputError("Invalid Knowledge provenance lifecycle flags")
        with nullcontext(conn) if conn is not None else self._db.transaction() as connection:
            parent = self._parent(note_id, connection, lock=True)
            if parent is None or (parent["deleted"] and not deleted):
                raise InputError("Active owned note required for Knowledge provenance")
            current = self._record(note_id, connection, lock=True)
            if (current["version"] if current else 0) != expected_version:
                raise ConflictError("Knowledge provenance version conflict")
            if current and current["deleted"] and not deleted and not restore:
                raise ConflictError("Explicit Knowledge provenance restore required")
            if (
                current
                and current["deleted"]
                and not deleted
                and canonical_notes_provenance_json(payload) != canonical_notes_provenance_json(current["payload"])
            ):
                raise ConflictError("Knowledge provenance restore must retain the tombstone payload")
            self._version(expected_version + 1)
            return self._write(note_id, payload, expected_version + 1, deleted, connection)

    def tombstone(self, note_id: str, expected_version: int, conn: Any = None) -> dict[str, Any]:
        """Retain the exact payload while advancing the sidecar tombstone version."""
        with nullcontext(conn) if conn is not None else self._db.transaction() as connection:
            self._parent(note_id, connection, lock=True)
            current = self.get(note_id, include_deleted=True, conn=connection)
            if current is None:
                raise InputError("Knowledge provenance does not exist")
            return self.put(note_id, current["payload"], expected_version, connection, deleted=True)

    def restore(self, note_id: str, expected_version: int, conn: Any = None) -> dict[str, Any]:
        """Explicitly restore retained evidence after its owned parent is active."""
        with nullcontext(conn) if conn is not None else self._db.transaction() as connection:
            self._parent(note_id, connection, lock=True)
            current = self.get(note_id, include_deleted=True, conn=connection)
            if current is None or not current["deleted"]:
                raise ConflictError("Knowledge provenance tombstone required")
            return self.put(note_id, current["payload"], expected_version, connection, restore=True)

    def apply_sync(
        self,
        note_id: str,
        payload: Mapping[str, object],
        object_revision: int,
        object_hash: str,
        deleted: bool = False,
        conn: Any = None,
        *,
        restore: bool = False,
    ) -> dict[str, Any]:
        """Project canonical revisions once; equal revision/hash is an idempotent replay."""
        self._version(object_revision)
        if type(deleted) is not bool or type(restore) is not bool or (deleted and restore):
            raise InputError("Invalid Knowledge provenance lifecycle flags")
        if object_hash != notes_provenance_object_hash(payload, deleted):
            raise InputError("Knowledge provenance hash mismatch")
        with nullcontext(conn) if conn is not None else self._db.transaction() as connection:
            parent = self._parent(note_id, connection, lock=True)
            if parent is None or (parent["deleted"] and not deleted):
                raise InputError("Active owned note required for Knowledge provenance")
            current = self._record(note_id, connection, lock=True)
            if current and object_revision <= current["version"]:
                if object_revision == current["version"] and object_hash == current["object_hash"]:
                    return current
                raise ConflictError("Knowledge provenance revision conflict")
            if current and current["deleted"] and not deleted and not restore:
                raise ConflictError("Explicit Knowledge provenance restore required")
            if (
                current
                and current["deleted"]
                and not deleted
                and canonical_notes_provenance_json(payload) != canonical_notes_provenance_json(current["payload"])
            ):
                raise ConflictError("Knowledge provenance restore must retain the tombstone payload")
            return self._write(note_id, payload, object_revision, deleted, connection)

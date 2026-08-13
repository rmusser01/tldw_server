"""Owner-bound persistence for canonical Notes attachment metadata."""

from __future__ import annotations

import re
import sqlite3
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, NoReturn
from uuid import RFC_4122, UUID

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.exceptions import NoteAttachmentPolicyError
from tldw_Server_API.app.core.Notes.attachment_policy import (
    canonicalize_note_attachment_file_name,
    validate_note_attachment_content_type,
    validate_note_attachment_original_file_name,
)
from tldw_Server_API.app.core.Sync.v2.models import normalize_sync_timestamp

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SOURCE_KINDS = frozenset({"upload", "sync", "legacy_bootstrap"})
_LIST_STATES = frozenset({"live", "tombstoned", "all"})


@dataclass(frozen=True)
class NoteAttachment:
    """One current canonical attachment-registry row."""

    client_id: str
    dataset_id: str
    attachment_id: str
    note_id: str
    file_name: str
    normalized_file_name: str
    original_file_name: str
    content_type: str
    size_bytes: int
    blob_hash: str
    object_hash: str
    version: int
    deleted: bool
    deleted_at: str | None
    delete_reason: str | None
    created_at: str
    last_modified: str
    created_by: str
    source_kind: str


class NoteAttachmentStore:
    """Persist attachment heads with explicit owner and dataset predicates."""

    _SELECT = (
        "SELECT attachment.client_id, attachment.dataset_id, attachment.attachment_id, "
        "attachment.note_id, attachment.file_name, attachment.normalized_file_name, "
        "attachment.original_file_name, attachment.content_type, attachment.size_bytes, "
        "attachment.blob_hash, attachment.object_hash, attachment.version, "
        "attachment.deleted, attachment.deleted_at, attachment.delete_reason, "
        "attachment.created_at, attachment.last_modified, attachment.created_by, "
        "attachment.source_kind FROM note_attachments AS attachment "
        "JOIN notes AS note ON note.id = attachment.note_id "
    )

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    @property
    def _owner_id(self) -> str:
        return str(self._db.client_id)

    def _deleted_value(self, deleted: bool) -> bool | int:
        return deleted if self._db.backend_type == BackendType.POSTGRESQL else int(deleted)

    @staticmethod
    def _uuid4(value: object, field_name: str) -> str:
        if not isinstance(value, str):
            raise InputError(f"{field_name} must be a canonical lowercase UUIDv4")
        try:
            parsed = UUID(value)
        except ValueError as exc:
            raise InputError(f"{field_name} must be a canonical lowercase UUIDv4") from exc
        if parsed.version != 4 or parsed.variant != RFC_4122 or str(parsed) != value:
            raise InputError(f"{field_name} must be a canonical lowercase UUIDv4")
        return value

    @staticmethod
    def _dataset_id(value: object) -> str:
        if not isinstance(value, str) or not value or value != value.strip() or len(value) > 255:
            raise InputError("dataset_id must be a non-empty bounded identifier")
        return value

    @classmethod
    def _normalized_filename(cls, value: object) -> tuple[str, str]:
        try:
            return canonicalize_note_attachment_file_name(value)
        except NoteAttachmentPolicyError as exc:
            raise InputError(str(exc)) from exc

    @staticmethod
    def _original_filename(value: object) -> str:
        try:
            return validate_note_attachment_original_file_name(value)
        except NoteAttachmentPolicyError as exc:
            raise InputError(str(exc)) from exc

    @staticmethod
    def _content_type(value: object) -> str:
        try:
            return validate_note_attachment_content_type(value)
        except NoteAttachmentPolicyError as exc:
            raise InputError(str(exc)) from exc

    @staticmethod
    def _positive_integer(value: object, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InputError(f"{field_name} must be a positive integer")
        return value

    @staticmethod
    def _digest(value: object, field_name: str) -> str:
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise InputError(f"{field_name} must be a canonical lowercase SHA-256 digest")
        return value

    @staticmethod
    def _timestamp(value: object, field_name: str) -> str:
        if not isinstance(value, str):
            raise InputError(f"{field_name} must be a canonical timestamp")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise InputError(f"{field_name} must be a canonical timestamp") from exc
        normalized = normalize_sync_timestamp(parsed)
        if parsed.tzinfo is None or normalized != value:
            raise InputError(f"{field_name} must be a canonical timestamp")
        return normalized

    @staticmethod
    def _created_by(value: object) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise InputError("created_by must be non-empty")
        return value

    @staticmethod
    def _source_kind(value: object) -> str:
        if not isinstance(value, str) or value not in _SOURCE_KINDS:
            raise InputError("source_kind is invalid")
        return value

    @classmethod
    def _from_row(cls, row: Any) -> NoteAttachment:
        record = dict(row)
        created_at = normalize_sync_timestamp(record.get("created_at"))
        last_modified = normalize_sync_timestamp(record.get("last_modified"))
        deleted_at = normalize_sync_timestamp(record.get("deleted_at"))
        if created_at is None or last_modified is None:
            raise CharactersRAGDBError("Stored attachment timestamps are invalid")
        return NoteAttachment(
            client_id=str(record["client_id"]),
            dataset_id=str(record["dataset_id"]),
            attachment_id=str(record["attachment_id"]),
            note_id=str(record["note_id"]),
            file_name=str(record["file_name"]),
            normalized_file_name=str(record["normalized_file_name"]),
            original_file_name=str(record["original_file_name"]),
            content_type=str(record["content_type"]),
            size_bytes=int(record["size_bytes"]),
            blob_hash=str(record["blob_hash"]),
            object_hash=str(record["object_hash"]),
            version=int(record["version"]),
            deleted=bool(record["deleted"]),
            deleted_at=deleted_at,
            delete_reason=(
                str(record["delete_reason"])
                if record.get("delete_reason") is not None
                else None
            ),
            created_at=created_at,
            last_modified=last_modified,
            created_by=str(record["created_by"]),
            source_kind=str(record["source_kind"]),
        )

    def _get_locked(
        self,
        conn: Any,
        dataset_id: str,
        attachment_id: str,
    ) -> NoteAttachment | None:
        row = conn.execute(
            self._SELECT
            + "WHERE attachment.client_id = ? AND attachment.dataset_id = ? "
            "AND attachment.attachment_id = ? AND note.client_id = ?",
            (self._owner_id, dataset_id, attachment_id, self._owner_id),
        ).fetchone()
        return self._from_row(row) if row is not None else None

    def _require_owned_note(
        self,
        conn: Any,
        note_id: str,
        *,
        allow_deleted: bool,
    ) -> None:
        row = conn.execute(
            "SELECT id, deleted FROM notes WHERE id = ? AND client_id = ?",
            (note_id, self._owner_id),
        ).fetchone()
        if row is None or (not allow_deleted and bool(row["deleted"])):
            raise InputError("Attachment requires an existing owned note endpoint")

    @staticmethod
    def _require_base(
        attachment: NoteAttachment | None,
        *,
        expected_version: int,
        expected_object_hash: str,
        attachment_id: str,
    ) -> NoteAttachment:
        if (
            attachment is None
            or attachment.version != expected_version
            or attachment.object_hash != expected_object_hash
        ):
            raise ConflictError(
                "Attachment optimistic base conflict",
                entity="note_attachments",
                entity_id=attachment_id,
            )
        return attachment

    @staticmethod
    def _require_updated(cursor: Any, attachment_id: str) -> None:
        if cursor.rowcount != 1:
            raise ConflictError(
                "Attachment optimistic base conflict",
                entity="note_attachments",
                entity_id=attachment_id,
            )

    @staticmethod
    def _write_constraint_name(exc: BaseException) -> str | None:
        current: BaseException | None = exc
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            diagnostic = getattr(current, "diag", None)
            constraint_name = getattr(diagnostic, "constraint_name", None)
            if isinstance(constraint_name, str) and constraint_name:
                return constraint_name.lower()
            current = current.__cause__ or current.__context__
        return None

    @staticmethod
    def _translate_write_error(exc: BaseException, attachment_id: str) -> NoReturn:
        message = str(exc).lower()
        constraint_name = NoteAttachmentStore._write_constraint_name(exc)
        if constraint_name == "uq_note_attachments_live_name" or (
            "unique constraint failed:" in message
            and "note_attachments.note_id" in message
            and "note_attachments.normalized_file_name" in message
        ):
            raise ConflictError(
                "Attachment live filename conflict",
                entity="note_attachments",
                entity_id=attachment_id,
            ) from exc
        if constraint_name == "note_attachments_pkey" or (
            "unique constraint failed:" in message
            and "note_attachments.client_id" in message
            and "note_attachments.dataset_id" in message
            and "note_attachments.attachment_id" in message
            and "note_attachments.note_id" not in message
        ):
            raise ConflictError(
                "Attachment identity conflict",
                entity="note_attachments",
                entity_id=attachment_id,
            ) from exc
        if "foreign key" in message:
            raise InputError("Attachment requires an existing owned note endpoint") from exc
        raise CharactersRAGDBError("Failed to persist canonical attachment metadata") from exc

    def get(
        self,
        dataset_id: str,
        attachment_id: str,
        *,
        conn: Any | None = None,
    ) -> NoteAttachment | None:
        """Return one owner/dataset-scoped row, including a tombstone."""

        dataset_id = self._dataset_id(dataset_id)
        attachment_id = self._uuid4(attachment_id, "attachment_id")
        context = nullcontext(conn) if conn is not None else self._db.transaction()
        with context as transaction_conn:
            return self._get_locked(transaction_conn, dataset_id, attachment_id)

    def list_page(
        self,
        dataset_id: str,
        note_id: str,
        *,
        after_attachment_id: str | None,
        limit: int,
        state: Literal["live", "tombstoned", "all"] = "live",
        include_deleted_note: bool = False,
        conn: Any | None = None,
    ) -> tuple[NoteAttachment, ...]:
        """Return one bounded attachment-ID keyset page."""

        dataset_id = self._dataset_id(dataset_id)
        note_id = self._uuid4(note_id, "note_id")
        if after_attachment_id is not None:
            after_attachment_id = self._uuid4(after_attachment_id, "after_attachment_id")
        if not 1 <= limit <= 200:
            raise InputError("Attachment page limit must be between 1 and 200")
        if state not in _LIST_STATES:
            raise InputError("Attachment list state is invalid")
        query = (
            self._SELECT
            + "WHERE attachment.client_id = ? AND attachment.dataset_id = ? "
            "AND attachment.note_id = ? AND attachment.attachment_id > ? "
            "AND note.client_id = ?"
        )
        params: list[object] = [
            self._owner_id,
            dataset_id,
            note_id,
            after_attachment_id or "",
            self._owner_id,
        ]
        if not include_deleted_note:
            query += " AND note.deleted = ?"
            params.append(self._deleted_value(False))
        if state != "all":
            query += " AND attachment.deleted = ?"
            params.append(self._deleted_value(state == "tombstoned"))
        query += " ORDER BY attachment.attachment_id LIMIT ?"
        params.append(limit)
        context = nullcontext(conn) if conn is not None else self._db.transaction()
        with context as transaction_conn:
            rows = transaction_conn.execute(query, tuple(params)).fetchall()
        return tuple(self._from_row(row) for row in rows)

    def create(
        self,
        *,
        dataset_id: str,
        attachment_id: str,
        note_id: str,
        file_name: str,
        original_file_name: str,
        content_type: str,
        size_bytes: int,
        blob_hash: str,
        object_hash: str,
        created_at: str,
        last_modified: str,
        created_by: str,
        source_kind: str,
        conn: Any | None = None,
    ) -> NoteAttachment:
        """Create a live revision-one attachment for one owned note."""

        dataset_id = self._dataset_id(dataset_id)
        attachment_id = self._uuid4(attachment_id, "attachment_id")
        note_id = self._uuid4(note_id, "note_id")
        file_name, normalized_file_name = self._normalized_filename(file_name)
        original_file_name = self._original_filename(original_file_name)
        content_type = self._content_type(content_type)
        size_bytes = self._positive_integer(size_bytes, "size_bytes")
        blob_hash = self._digest(blob_hash, "blob_hash")
        object_hash = self._digest(object_hash, "object_hash")
        created_at = self._timestamp(created_at, "created_at")
        last_modified = self._timestamp(last_modified, "last_modified")
        created_by = self._created_by(created_by)
        source_kind = self._source_kind(source_kind)
        context = nullcontext(conn) if conn is not None else self._db.transaction()
        try:
            with context as transaction_conn:
                self._require_owned_note(transaction_conn, note_id, allow_deleted=False)
                transaction_conn.execute(
                    "INSERT INTO note_attachments("
                    "client_id, dataset_id, attachment_id, note_id, file_name, "
                    "normalized_file_name, original_file_name, content_type, size_bytes, "
                    "blob_hash, object_hash, version, deleted, deleted_at, delete_reason, "
                    "created_at, last_modified, created_by, source_kind) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, NULL, NULL, ?, ?, ?, ?)",
                    (
                        self._owner_id,
                        dataset_id,
                        attachment_id,
                        note_id,
                        file_name,
                        normalized_file_name,
                        original_file_name,
                        content_type,
                        size_bytes,
                        blob_hash,
                        object_hash,
                        self._deleted_value(False),
                        created_at,
                        last_modified,
                        created_by,
                        source_kind,
                    ),
                )
                created = self._get_locked(transaction_conn, dataset_id, attachment_id)
                if created is None:
                    raise CharactersRAGDBError("Inserted attachment was not found")
                return created
        except (sqlite3.IntegrityError, BackendDatabaseError) as exc:
            self._translate_write_error(exc, attachment_id)

    def compare_and_set(
        self,
        *,
        dataset_id: str,
        attachment_id: str,
        expected_version: int,
        expected_object_hash: str,
        file_name: str,
        content_type: str,
        size_bytes: int,
        blob_hash: str,
        object_hash: str,
        last_modified: str,
        conn: Any | None = None,
    ) -> NoteAttachment:
        """Replace mutable live fields after matching the exact optimistic base."""

        dataset_id = self._dataset_id(dataset_id)
        attachment_id = self._uuid4(attachment_id, "attachment_id")
        expected_version = self._positive_integer(expected_version, "expected_version")
        expected_object_hash = self._digest(expected_object_hash, "expected_object_hash")
        file_name, normalized_file_name = self._normalized_filename(file_name)
        content_type = self._content_type(content_type)
        size_bytes = self._positive_integer(size_bytes, "size_bytes")
        blob_hash = self._digest(blob_hash, "blob_hash")
        object_hash = self._digest(object_hash, "object_hash")
        last_modified = self._timestamp(last_modified, "last_modified")
        context = nullcontext(conn) if conn is not None else self._db.transaction()
        try:
            with context as transaction_conn:
                existing = self._require_base(
                    self._get_locked(transaction_conn, dataset_id, attachment_id),
                    expected_version=expected_version,
                    expected_object_hash=expected_object_hash,
                    attachment_id=attachment_id,
                )
                if existing.deleted:
                    raise ConflictError(
                        "Attachment is tombstoned; use restore",
                        entity="note_attachments",
                        entity_id=attachment_id,
                    )
                self._require_owned_note(
                    transaction_conn,
                    existing.note_id,
                    allow_deleted=False,
                )
                cursor = transaction_conn.execute(
                    "UPDATE note_attachments SET file_name = ?, normalized_file_name = ?, "
                    "content_type = ?, size_bytes = ?, blob_hash = ?, object_hash = ?, "
                    "version = ?, last_modified = ? WHERE client_id = ? AND dataset_id = ? "
                    "AND attachment_id = ? AND version = ? AND object_hash = ? AND deleted = ?",
                    (
                        file_name,
                        normalized_file_name,
                        content_type,
                        size_bytes,
                        blob_hash,
                        object_hash,
                        existing.version + 1,
                        last_modified,
                        self._owner_id,
                        dataset_id,
                        attachment_id,
                        expected_version,
                        expected_object_hash,
                        self._deleted_value(False),
                    ),
                )
                self._require_updated(cursor, attachment_id)
                updated = self._get_locked(transaction_conn, dataset_id, attachment_id)
                if updated is None:
                    raise CharactersRAGDBError("Updated attachment was not found")
                return updated
        except (sqlite3.IntegrityError, BackendDatabaseError) as exc:
            self._translate_write_error(exc, attachment_id)

    def tombstone(
        self,
        *,
        dataset_id: str,
        attachment_id: str,
        expected_version: int,
        expected_object_hash: str,
        object_hash: str,
        last_modified: str,
        deleted_at: str,
        delete_reason: str | None = None,
        conn: Any | None = None,
    ) -> NoteAttachment:
        """Tombstone one live attachment without changing immutable identity."""

        dataset_id = self._dataset_id(dataset_id)
        attachment_id = self._uuid4(attachment_id, "attachment_id")
        expected_version = self._positive_integer(expected_version, "expected_version")
        expected_object_hash = self._digest(expected_object_hash, "expected_object_hash")
        object_hash = self._digest(object_hash, "object_hash")
        last_modified = self._timestamp(last_modified, "last_modified")
        deleted_at = self._timestamp(deleted_at, "deleted_at")
        if delete_reason is not None and (
            not isinstance(delete_reason, str) or len(delete_reason) > 256
        ):
            raise InputError("delete_reason exceeds its boundary")
        context = nullcontext(conn) if conn is not None else self._db.transaction()
        with context as transaction_conn:
            existing = self._require_base(
                self._get_locked(transaction_conn, dataset_id, attachment_id),
                expected_version=expected_version,
                expected_object_hash=expected_object_hash,
                attachment_id=attachment_id,
            )
            if existing.deleted:
                raise ConflictError(
                    "Attachment is already tombstoned",
                    entity="note_attachments",
                    entity_id=attachment_id,
                )
            self._require_owned_note(transaction_conn, existing.note_id, allow_deleted=True)
            cursor = transaction_conn.execute(
                "UPDATE note_attachments SET deleted = ?, deleted_at = ?, delete_reason = ?, "
                "object_hash = ?, version = ?, last_modified = ? WHERE client_id = ? "
                "AND dataset_id = ? AND attachment_id = ? AND version = ? "
                "AND object_hash = ? AND deleted = ?",
                (
                    self._deleted_value(True),
                    deleted_at,
                    delete_reason,
                    object_hash,
                    existing.version + 1,
                    last_modified,
                    self._owner_id,
                    dataset_id,
                    attachment_id,
                    expected_version,
                    expected_object_hash,
                    self._deleted_value(False),
                ),
            )
            self._require_updated(cursor, attachment_id)
            tombstone = self._get_locked(transaction_conn, dataset_id, attachment_id)
            if tombstone is None:
                raise CharactersRAGDBError("Tombstoned attachment was not found")
            return tombstone

    def restore(
        self,
        *,
        dataset_id: str,
        attachment_id: str,
        expected_version: int,
        expected_object_hash: str,
        object_hash: str,
        last_modified: str,
        conn: Any | None = None,
    ) -> NoteAttachment:
        """Restore one tombstone after rechecking its owned note and live name."""

        dataset_id = self._dataset_id(dataset_id)
        attachment_id = self._uuid4(attachment_id, "attachment_id")
        expected_version = self._positive_integer(expected_version, "expected_version")
        expected_object_hash = self._digest(expected_object_hash, "expected_object_hash")
        object_hash = self._digest(object_hash, "object_hash")
        last_modified = self._timestamp(last_modified, "last_modified")
        context = nullcontext(conn) if conn is not None else self._db.transaction()
        try:
            with context as transaction_conn:
                existing = self._require_base(
                    self._get_locked(transaction_conn, dataset_id, attachment_id),
                    expected_version=expected_version,
                    expected_object_hash=expected_object_hash,
                    attachment_id=attachment_id,
                )
                if not existing.deleted:
                    raise ConflictError(
                        "Attachment is not tombstoned",
                        entity="note_attachments",
                        entity_id=attachment_id,
                    )
                self._require_owned_note(transaction_conn, existing.note_id, allow_deleted=True)
                cursor = transaction_conn.execute(
                    "UPDATE note_attachments SET deleted = ?, deleted_at = NULL, "
                    "delete_reason = NULL, object_hash = ?, version = ?, last_modified = ? "
                    "WHERE client_id = ? AND dataset_id = ? AND attachment_id = ? "
                    "AND version = ? AND object_hash = ? AND deleted = ?",
                    (
                        self._deleted_value(False),
                        object_hash,
                        existing.version + 1,
                        last_modified,
                        self._owner_id,
                        dataset_id,
                        attachment_id,
                        expected_version,
                        expected_object_hash,
                        self._deleted_value(True),
                    ),
                )
                self._require_updated(cursor, attachment_id)
                restored = self._get_locked(transaction_conn, dataset_id, attachment_id)
                if restored is None:
                    raise CharactersRAGDBError("Restored attachment was not found")
                return restored
        except (sqlite3.IntegrityError, BackendDatabaseError) as exc:
            self._translate_write_error(exc, attachment_id)


__all__ = ["NoteAttachment", "NoteAttachmentStore"]

"""Repeatable source reads for cloneable Media rows and child collections."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, TypeVar
from uuid import UUID, uuid4

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import (
    MediaDbLike,
    require_media_database_like,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Sharing.clone_models import MediaCloneSnapshot

_SnapshotResult = TypeVar("_SnapshotResult")
_CLONE_OPERATION_KIND = "shared_workspace_clone"
_IDENTIFIER_MAX_LENGTH = 255
_PROVENANCE_URL_MAX_LENGTH = 4096
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CANONICAL_MAX_DEPTH = 64
_CANONICAL_MAX_CONTAINER_ITEMS = 1_000_000
_CANONICAL_HASH_DOMAIN = b"tldw.media-clone-snapshot.v1\x00"
_LOGICAL_COPY_PROJECTION_VERSION = 1


@dataclass(frozen=True, slots=True)
class OperationOwnedMediaResult:
    """Target identity plus whether this call created or replayed the owned row."""

    media_id: int
    media_uuid: str
    created: bool
    replayed: bool

    def __post_init__(self) -> None:
        if isinstance(self.media_id, bool) or not isinstance(self.media_id, int) or self.media_id <= 0:
            raise ValueError("media_id must be a positive integer")
        if not isinstance(self.media_uuid, str) or not self.media_uuid:
            raise ValueError("media_uuid must be a non-empty string")
        if not isinstance(self.created, bool) or not isinstance(self.replayed, bool):
            raise TypeError("created and replayed must be booleans")
        if self.created == self.replayed:
            raise ValueError("exactly one of created or replayed must be true")


@dataclass(frozen=True, slots=True)
class OperationOwnedMediaReference:
    """Bounded reconciliation identity for one pending clone Media row."""

    media_id: int
    media_uuid: str
    source_identity: str
    expected_content_hash: str

    def __post_init__(self) -> None:
        if isinstance(self.media_id, bool) or not isinstance(self.media_id, int) or self.media_id <= 0:
            raise ValueError("media_id must be a positive integer")
        if not isinstance(self.media_uuid, str) or not self.media_uuid:
            raise ValueError("media_uuid must be a non-empty string")
        _validate_identifier(self.source_identity, "source_identity")
        _validate_sha256(self.expected_content_hash, "expected_content_hash")


def _canonical_frame(tag: bytes, payload: bytes) -> bytes:
    return tag + len(payload).to_bytes(8, "big") + payload


def _canonical_parts(value: Any, *, depth: int = 0) -> Iterator[bytes]:
    """Yield an unambiguous type-tagged encoding for immutable snapshot values."""

    if depth > _CANONICAL_MAX_DEPTH:
        raise ValueError("clone snapshot exceeds the canonical nesting bound")
    if value is None:
        yield b"n"
        return
    if isinstance(value, bool):
        yield b"b1" if value else b"b0"
        return
    if isinstance(value, int):
        yield _canonical_frame(b"i", str(value).encode("ascii"))
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite clone snapshot floats are unsupported")
        yield _canonical_frame(b"f", value.hex().encode("ascii"))
        return
    if isinstance(value, str):
        yield _canonical_frame(b"s", value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        yield _canonical_frame(b"y", value)
        return
    if isinstance(value, datetime):
        yield _canonical_frame(b"z", value.isoformat(timespec="microseconds").encode("ascii"))
        return
    if isinstance(value, date):
        yield _canonical_frame(b"d", value.isoformat().encode("ascii"))
        return
    if isinstance(value, time):
        yield _canonical_frame(b"t", value.isoformat(timespec="microseconds").encode("ascii"))
        return
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite clone snapshot decimals are unsupported")
        yield _canonical_frame(b"m", str(value.normalize()).encode("ascii"))
        return
    if isinstance(value, UUID):
        yield _canonical_frame(b"u", value.bytes)
        return
    if isinstance(value, Mapping):
        if len(value) > _CANONICAL_MAX_CONTAINER_ITEMS:
            raise ValueError("clone snapshot mapping exceeds the item bound")
        encoded_keys = [
            (b"".join(_canonical_parts(key, depth=depth + 1)), key)
            for key in value
        ]
        encoded_keys.sort(key=lambda item: item[0])
        if any(
            encoded_keys[index - 1][0] == encoded_keys[index][0]
            for index in range(1, len(encoded_keys))
        ):
            raise ValueError("clone snapshot mapping has ambiguous canonical keys")
        yield b"p" + len(encoded_keys).to_bytes(8, "big")
        for encoded_key, key in encoded_keys:
            yield encoded_key
            yield from _canonical_parts(value[key], depth=depth + 1)
        return
    if isinstance(value, tuple):
        if len(value) > _CANONICAL_MAX_CONTAINER_ITEMS:
            raise ValueError("clone snapshot sequence exceeds the item bound")
        yield b"q" + len(value).to_bytes(8, "big")
        for item in value:
            yield from _canonical_parts(item, depth=depth + 1)
        return
    if isinstance(value, frozenset):
        if len(value) > _CANONICAL_MAX_CONTAINER_ITEMS:
            raise ValueError("clone snapshot set exceeds the item bound")
        encoded_items = sorted(
            b"".join(_canonical_parts(item, depth=depth + 1)) for item in value
        )
        yield b"r" + len(encoded_items).to_bytes(8, "big")
        yield from encoded_items
        return
    raise TypeError(f"unsupported clone snapshot hash value: {type(value).__name__}")


def _normalize_persisted_value(value: Any, *, depth: int = 0) -> Any:
    """Project supported values to their stable database representation."""
    if depth > _CANONICAL_MAX_DEPTH:
        raise ValueError("clone snapshot exceeds the canonical nesting bound")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite clone snapshot floats are unsupported")
        return value
    if isinstance(value, complex):
        raise TypeError("unsupported clone snapshot hash value: complex")
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (date, time, Decimal, UUID)):
        if isinstance(value, Decimal) and not value.is_finite():
            raise ValueError("non-finite clone snapshot decimals are unsupported")
        return str(value)
    if isinstance(value, Mapping):
        if len(value) > _CANONICAL_MAX_CONTAINER_ITEMS:
            raise ValueError("clone snapshot mapping exceeds the item bound")
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key in normalized:
                raise ValueError("clone snapshot mapping has ambiguous persisted keys")
            normalized[normalized_key] = _normalize_persisted_value(
                item,
                depth=depth + 1,
            )
        return normalized
    if isinstance(value, (list, tuple, frozenset)):
        if len(value) > _CANONICAL_MAX_CONTAINER_ITEMS:
            raise ValueError("clone snapshot sequence exceeds the item bound")
        normalized_items = tuple(
            _normalize_persisted_value(item, depth=depth + 1) for item in value
        )
        if isinstance(value, frozenset):
            return tuple(sorted(normalized_items, key=_canonical_bytes))
        return normalized_items
    raise TypeError(f"unsupported clone snapshot hash value: {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return b"".join(_canonical_parts(value))


def _bounded_source_url(value: Any) -> str | None:
    return value[:_PROVENANCE_URL_MAX_LENGTH] if isinstance(value, str) else None


def _normalize_keywords(keywords: Any) -> tuple[str, ...]:
    if not isinstance(keywords, tuple):
        raise InputError("snapshot Media keywords must be a tuple")
    normalized: set[str] = set()
    for keyword in keywords:
        if not isinstance(keyword, str):
            raise InputError("snapshot Media keywords must contain strings")
        value = keyword.strip().lower()
        if value:
            normalized.add(value)
    return tuple(sorted(normalized))


def _ordered_projection(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(sorted(rows, key=_canonical_bytes))


def _logical_copy_projection(
    *,
    media: Mapping[str, Any],
    keywords: tuple[str, ...],
    chunks: Sequence[Mapping[str, Any]],
    transcripts: Sequence[Mapping[str, Any]],
    document: Mapping[str, Any],
    source_rows: bool,
) -> Mapping[str, Any]:
    media_projection = {
        field: _normalize_persisted_value(media.get(field))
        for field in (
            "title",
            "type",
            "content",
            "author",
            "ingestion_date",
            "transcription_model",
            "content_hash",
            "source_hash",
            "latest_transcription_run_id",
        )
    }
    media_projection["chunking_status"] = _normalize_persisted_value(
        media.get("chunking_status") or "completed"
    )
    media_projection["next_transcription_run_id"] = _normalize_persisted_value(
        media.get("next_transcription_run_id") or 1
    )

    chunk_projection: list[Mapping[str, Any]] = []
    for row in chunks:
        projected = {
            field: _normalize_persisted_value(row.get(field))
            for field in (
                "chunk_text",
                "chunk_index",
                "start_char",
                "end_char",
                "chunk_type",
                "creation_date",
                "last_modified_orig",
                "metadata",
            )
        }
        projected["is_processed"] = (
            False if source_rows else bool(row.get("is_processed"))
        )
        chunk_projection.append(projected)

    transcript_projection = [
        {
            field: _normalize_persisted_value(row.get(field))
            for field in (
                "whisper_model",
                "transcription",
                "created_at",
                "transcription_run_id",
                "supersedes_run_id",
                "idempotency_key",
            )
        }
        for row in transcripts
    ]
    return {
        "projection_version": _LOGICAL_COPY_PROJECTION_VERSION,
        "media": media_projection,
        "document": {
            field: _normalize_persisted_value(document.get(field))
            for field in (
                "version_number",
                "prompt",
                "analysis_content",
                "source_url",
                "content",
            )
        },
        "keywords": _normalize_keywords(keywords),
        "chunks": _ordered_projection(chunk_projection),
        "transcripts": _ordered_projection(transcript_projection),
    }


def _snapshot_logical_copy_projection(snapshot: MediaCloneSnapshot) -> Mapping[str, Any]:
    content = snapshot.media.get("content")
    return _logical_copy_projection(
        media=snapshot.media,
        keywords=_normalize_keywords(snapshot.media.get("keywords", ())),
        chunks=snapshot.chunks,
        transcripts=snapshot.transcripts,
        document={
            "version_number": 1,
            "prompt": None,
            "analysis_content": None,
            "source_url": _bounded_source_url(snapshot.media.get("url")),
            "content": "" if content is None else content,
        },
        source_rows=True,
    )


def _hash_logical_copy_projection(projection: Mapping[str, Any]) -> str:
    hasher = hashlib.sha256()
    hasher.update(_CANONICAL_HASH_DOMAIN)
    for part in _canonical_parts(projection):
        hasher.update(part)
    return hasher.hexdigest()


def hash_media_clone_snapshot(snapshot: MediaCloneSnapshot) -> str:
    """Hash the versioned canonical logical-copy projection of a snapshot."""

    from tldw_Server_API.app.core.Sharing.clone_models import MediaCloneSnapshot

    if not isinstance(snapshot, MediaCloneSnapshot):
        raise TypeError("snapshot must be a MediaCloneSnapshot")
    return _hash_logical_copy_projection(_snapshot_logical_copy_projection(snapshot))


def _validate_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _IDENTIFIER_MAX_LENGTH:
        raise InputError(f"{field_name} must be a bounded non-empty ASCII identifier")
    if not value.isascii() or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        raise InputError(f"{field_name} must contain printable ASCII characters")
    return value


def _validate_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise InputError(f"{field_name} must be a lowercase SHA-256 hash")
    return value


def _operation_conflict() -> ConflictError:
    return ConflictError(
        "Operation-owned clone Media state conflicts with the requested snapshot.",
        entity="Media",
    )


def _json_safe(value: Any) -> Any:
    return _normalize_persisted_value(value)


def _json_column(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(
        _json_safe(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


class CloneSnapshotRepository:
    """Materialize active Media clone inputs through one repeatable transaction."""

    def __init__(self, session: MediaDbLike):
        self.session = session

    @classmethod
    def from_legacy_db(cls, db: MediaDbLike) -> CloneSnapshotRepository:
        return cls(
            session=require_media_database_like(
                db,
                error_message="db_instance must be a Media database object.",
            )
        )

    @staticmethod
    def _validate_media_ids(media_ids: Sequence[int]) -> tuple[int, ...]:
        from tldw_Server_API.app.core.Sharing.clone_models import CloneSnapshotUnavailable

        if isinstance(media_ids, (str, bytes, bytearray)) or not isinstance(
            media_ids,
            Sequence,
        ):
            raise CloneSnapshotUnavailable(cleanup_state="complete")
        normalized = tuple(media_ids)
        if any(
            isinstance(media_id, bool)
            or not isinstance(media_id, int)
            or media_id <= 0
            for media_id in normalized
        ):
            raise CloneSnapshotUnavailable(cleanup_state="complete")
        if len(set(normalized)) != len(normalized):
            raise CloneSnapshotUnavailable(cleanup_state="complete")
        return normalized

    def _run_snapshot(self, reader: Callable[[Any, Any], _SnapshotResult]) -> _SnapshotResult:
        from tldw_Server_API.app.core.Sharing.clone_models import CloneSnapshotUnavailable

        backend = self.session.backend  # type: ignore[attr-defined]
        connection: Any | None = None
        pool: Any | None = None
        committed = False
        primary_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        result: _SnapshotResult | None = None

        try:
            if backend.backend_type == BackendType.SQLITE:
                sqlite_path = str(getattr(backend.config, "sqlite_path", "") or "").strip()
                lowered_path = sqlite_path.lower()
                private_memory = sqlite_path == ":memory:" or (
                    "mode=memory" in lowered_path and "cache=shared" not in lowered_path
                )
                if not sqlite_path or private_memory:
                    raise CloneSnapshotUnavailable(cleanup_state="complete")
                connection = backend.connect()
                connection.execute("PRAGMA query_only = ON")
                query_only_row = connection.execute("PRAGMA query_only").fetchone()
                if query_only_row is None or int(query_only_row[0]) != 1:
                    raise RuntimeError("SQLite query-only mode unavailable")
                connection.execute("BEGIN")
                if not bool(getattr(connection, "in_transaction", False)):
                    raise RuntimeError("SQLite snapshot transaction unavailable")
            else:
                pool = backend.get_pool()
                connection = pool.get_connection()
                connection.rollback()
                backend.apply_and_verify_scope(
                    connection,
                    fallback_user_id=self.session.client_id,
                )
                with connection.cursor() as cursor:
                    cursor.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
                isolation_rows = backend.execute(
                    "SHOW transaction_isolation",
                    connection=connection,
                    log_errors=False,
                ).rows
                read_only_rows = backend.execute(
                    "SHOW transaction_read_only",
                    connection=connection,
                    log_errors=False,
                ).rows
                isolation = next(iter(isolation_rows[0].values()), None) if isolation_rows else None
                read_only = next(iter(read_only_rows[0].values()), None) if read_only_rows else None
                if str(isolation).lower() != "repeatable read" or str(read_only).lower() not in {
                    "on",
                    "true",
                }:
                    raise RuntimeError("PostgreSQL repeatable read unavailable")

            result = reader(backend, connection)
            connection.commit()
            committed = True
        except BaseException as exc:  # noqa: BLE001 - cleanup must run for every path
            primary_error = exc

        if connection is not None:
            if not committed:
                try:
                    connection.rollback()
                except BaseException as exc:  # noqa: BLE001 - preserve primary failure
                    cleanup_error = exc
            try:
                if backend.backend_type == BackendType.SQLITE:
                    backend.disconnect(connection)
                else:
                    (pool or backend.get_pool()).return_connection(connection)
            except BaseException as exc:  # noqa: BLE001 - convert cleanup failures below
                cleanup_error = cleanup_error or exc

        if primary_error is not None and not isinstance(primary_error, Exception):
            raise primary_error
        if primary_error is not None or cleanup_error is not None or result is None:
            failure = primary_error or cleanup_error
            logger.bind(
                backend=backend.backend_type.value,
                exception_type=type(failure).__name__ if failure is not None else "Unknown",
            ).warning("Media clone snapshot read failed")
            raise CloneSnapshotUnavailable(cleanup_state="complete") from None
        return result

    @staticmethod
    def _storage_url(operation_id: str, source_identity: str) -> str:
        operation_digest = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()
        source_digest = hashlib.sha256(source_identity.encode("utf-8")).hexdigest()
        return f"tldw-clone://workspace/{operation_digest}/{source_digest}"

    def _owned_candidates(
        self,
        connection: Any,
        *,
        storage_url: str,
        operation_id: str,
        source_identity: str,
    ) -> list[dict[str, Any]]:
        rows = self.session._fetchall_with_connection(
            connection,
            "SELECT id, url, uuid, deleted, is_trash, system_operation_id, "
            "system_operation_kind, system_source_identity, system_content_hash "
            "FROM Media WHERE url = ? OR system_operation_id = ?",
            (storage_url, operation_id),
        )
        return [dict(row) for row in rows]

    def _lock_owned_insert(
        self,
        connection: Any,
        *,
        operation_id: str,
        source_identity: str,
    ) -> None:
        if self.session.backend_type != BackendType.POSTGRESQL:  # type: ignore[attr-defined]
            return
        self.session._execute_with_connection(
            connection,
            "SELECT pg_advisory_xact_lock(hashtext(?), hashtext(?))",
            (f"media-clone:{operation_id}", source_identity),
        )

    def _verify_owned_candidate(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        storage_url: str,
        operation_id: str,
        source_identity: str,
        expected_content_hash: str,
        require_pending: bool,
    ) -> Mapping[str, Any] | None:
        relevant: list[Mapping[str, Any]] = []
        for candidate in candidates:
            candidate_operation = candidate.get("system_operation_id")
            candidate_source = candidate.get("system_source_identity")
            if candidate_operation == operation_id:
                if (
                    candidate.get("system_operation_kind") != _CLONE_OPERATION_KIND
                    or not isinstance(candidate_source, str)
                    or candidate.get("url")
                    != self._storage_url(operation_id, candidate_source)
                ):
                    raise _operation_conflict()
            if candidate.get("url") == storage_url or (
                candidate_operation == operation_id
                and candidate_source == source_identity
            ):
                relevant.append(candidate)
        if not relevant:
            return None
        if len(relevant) != 1:
            raise _operation_conflict()
        candidate = relevant[0]
        exact_marker = (
            candidate.get("url") == storage_url
            and candidate.get("system_operation_kind") == _CLONE_OPERATION_KIND
            and candidate.get("system_operation_id") == operation_id
            and candidate.get("system_source_identity") == source_identity
            and hmac.compare_digest(
                str(candidate.get("system_content_hash") or ""),
                expected_content_hash,
            )
        )
        pending = not bool(candidate.get("deleted")) and bool(candidate.get("is_trash"))
        if not exact_marker or (require_pending and not pending):
            raise _operation_conflict()
        return candidate

    def _active_value(self) -> bool | int:
        return False if self.session.backend_type == BackendType.POSTGRESQL else 0  # type: ignore[attr-defined]

    @staticmethod
    def _decode_json_value(value: Any) -> Any:
        if value is None or isinstance(value, Mapping):
            return value
        if not isinstance(value, str):
            raise _operation_conflict()
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            raise _operation_conflict() from None

    def _persisted_logical_copy_projection(
        self,
        connection: Any,
        *,
        media_id: int,
        operation_id: str,
        source_identity: str,
    ) -> Mapping[str, Any]:
        active_value = self._active_value()
        media_rows = self.session._fetchall_with_connection(
            connection,
            "SELECT title, type, content, author, ingestion_date, transcription_model, "
            "chunking_status, content_hash, source_hash, latest_transcription_run_id, "
            "next_transcription_run_id FROM Media WHERE id = ?",
            (media_id,),
        )
        document_rows = self.session._fetchall_with_connection(
            connection,
            "SELECT version_number, prompt, analysis_content, safe_metadata, content "
            "FROM DocumentVersions WHERE media_id = ? AND deleted = ?",
            (media_id, active_value),
        )
        if len(media_rows) != 1 or len(document_rows) != 1:
            raise _operation_conflict()

        document_row = dict(document_rows[0])
        provenance = self._decode_json_value(document_row.pop("safe_metadata"))
        if not isinstance(provenance, Mapping):
            raise _operation_conflict()
        clone_provenance = provenance.get("clone_provenance")
        if not isinstance(clone_provenance, Mapping) or "source_url" not in clone_provenance:
            raise _operation_conflict()
        document_row["source_url"] = clone_provenance["source_url"]

        keyword_rows = self.session._fetchall_with_connection(
            connection,
            "SELECT mk.keyword_id, k.keyword FROM MediaKeywords mk "
            "JOIN Keywords k ON k.id = mk.keyword_id "
            "WHERE mk.media_id = ? AND k.deleted = ?",
            (media_id, active_value),
        )
        hold_rows = self.session._fetchall_with_connection(
            connection,
            "SELECT keyword_id, operation_id, source_identity "
            "FROM OperationOwnedCloneKeywords WHERE media_id = ?",
            (media_id,),
        )
        keyword_ids = {int(row["keyword_id"]) for row in keyword_rows}
        if keyword_ids != {int(row["keyword_id"]) for row in hold_rows} or any(
            row["operation_id"] != operation_id
            or row["source_identity"] != source_identity
            for row in hold_rows
        ):
            raise _operation_conflict()
        chunk_rows = [
            dict(row)
            for row in self.session._fetchall_with_connection(
                connection,
                "SELECT chunk_text, chunk_index, start_char, end_char, chunk_type, "
                "creation_date, last_modified_orig, is_processed, metadata "
                "FROM UnvectorizedMediaChunks WHERE media_id = ? AND deleted = ?",
                (media_id, active_value),
            )
        ]
        for chunk in chunk_rows:
            chunk["metadata"] = self._decode_json_value(chunk.get("metadata"))
        transcript_rows = [
            dict(row)
            for row in self.session._fetchall_with_connection(
                connection,
                "SELECT whisper_model, transcription, created_at, transcription_run_id, "
                "supersedes_run_id, idempotency_key FROM Transcripts "
                "WHERE media_id = ? AND deleted = ?",
                (media_id, active_value),
            )
        ]
        try:
            return _logical_copy_projection(
                media=dict(media_rows[0]),
                keywords=tuple(str(row["keyword"]) for row in keyword_rows),
                chunks=chunk_rows,
                transcripts=transcript_rows,
                document=document_row,
                source_rows=False,
            )
        except (InputError, TypeError, ValueError):
            raise _operation_conflict() from None

    def _verify_persisted_logical_copy(
        self,
        connection: Any,
        *,
        media_id: int,
        operation_id: str,
        source_identity: str,
        expected_content_hash: str,
    ) -> None:
        persisted_hash = _hash_logical_copy_projection(
            self._persisted_logical_copy_projection(
                connection,
                media_id=media_id,
                operation_id=operation_id,
                source_identity=source_identity,
            )
        )
        if not hmac.compare_digest(persisted_hash, expected_content_hash):
            raise _operation_conflict()

    def _insert_media_row(
        self,
        connection: Any,
        *,
        snapshot: MediaCloneSnapshot,
        storage_url: str,
        operation_id: str,
        source_identity: str,
        expected_content_hash: str,
        now: str,
    ) -> tuple[int, str]:
        media = snapshot.media
        title = media.get("title")
        media_type = media.get("type")
        content = media.get("content")
        content_hash = media.get("content_hash")
        if not isinstance(title, str) or not title:
            raise InputError("snapshot Media title must be a non-empty string")
        if not isinstance(media_type, str) or not media_type:
            raise InputError("snapshot Media type must be a non-empty string")
        if content is not None and not isinstance(content, str):
            raise InputError("snapshot Media content must be a string or null")
        if not isinstance(content_hash, str) or not content_hash:
            raise InputError("snapshot Media content_hash must be a non-empty string")

        media_uuid = str(uuid4())
        active_value = self._active_value()
        sql = """
            INSERT INTO Media (
                url, title, type, content, author, ingestion_date,
                transcription_model, is_trash, trash_date, vector_embedding,
                chunking_status, vector_processing, content_hash, source_hash,
                uuid, last_modified, version, org_id, team_id, visibility,
                owner_user_id, latest_transcription_run_id, next_transcription_run_id,
                client_id, deleted, prev_version, merge_parent_uuid,
                system_operation_id, system_operation_kind, system_source_identity,
                system_content_hash
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """
        if self.session.backend_type == BackendType.POSTGRESQL:  # type: ignore[attr-defined]
            sql += " RETURNING id"
        cursor = self.session._execute_with_connection(
            connection,
            sql,
            (
                storage_url,
                title,
                media_type,
                content,
                media.get("author"),
                _normalize_persisted_value(media.get("ingestion_date")),
                media.get("transcription_model"),
                True if self.session.backend_type == BackendType.POSTGRESQL else 1,  # type: ignore[attr-defined]
                None,
                None,
                media.get("chunking_status") or "completed",
                0,
                content_hash,
                media.get("source_hash"),
                media_uuid,
                now,
                1,
                None,
                None,
                "personal",
                None,
                media.get("latest_transcription_run_id"),
                media.get("next_transcription_run_id") or 1,
                self.session.client_id,
                active_value,
                None,
                None,
                operation_id,
                _CLONE_OPERATION_KIND,
                source_identity,
                expected_content_hash,
            ),
        )
        if self.session.backend_type == BackendType.POSTGRESQL:  # type: ignore[attr-defined]
            inserted = cursor.fetchone()
            media_id = int(inserted["id"]) if inserted else 0
        else:
            media_id = int(cursor.lastrowid or 0)
        if media_id <= 0:
            raise DatabaseError("Operation-owned clone Media insert returned no identity.")
        return media_id, media_uuid

    def _insert_document_version(
        self,
        connection: Any,
        *,
        media_id: int,
        content: str,
        source_url: Any,
        now: str,
    ) -> None:
        bounded_source_url = _bounded_source_url(source_url)
        safe_metadata = json.dumps(
            {"clone_provenance": {"source_url": bounded_source_url}},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.session._execute_with_connection(
            connection,
            "INSERT INTO DocumentVersions "
            "(media_id, version_number, prompt, analysis_content, safe_metadata, "
            "content, created_at, uuid, last_modified, version, client_id, deleted) "
            "VALUES (?, 1, NULL, NULL, ?, ?, ?, ?, ?, 1, ?, ?)",
            (
                media_id,
                safe_metadata,
                content,
                now,
                str(uuid4()),
                now,
                self.session.client_id,
                self._active_value(),
            ),
        )

    def _insert_keywords(
        self,
        connection: Any,
        *,
        media_id: int,
        keywords: Any,
        operation_id: str,
        source_identity: str,
        now: str,
    ) -> None:
        for keyword in _normalize_keywords(keywords):
            rows = self.session._fetchall_with_connection(
                connection,
                "SELECT id, deleted FROM Keywords WHERE LOWER(keyword) = ?",
                (keyword,),
            )
            if len(rows) > 1 or (rows and bool(rows[0]["deleted"])):
                raise _operation_conflict()
            if rows:
                keyword_id = int(rows[0]["id"])
                existing_holds = self.session._fetchall_with_connection(
                    connection,
                    "SELECT 1 FROM OperationOwnedCloneKeywords "
                    "WHERE keyword_id = ? AND created_by_clone = ? LIMIT 1",
                    (
                        keyword_id,
                        True
                        if self.session.backend_type == BackendType.POSTGRESQL  # type: ignore[attr-defined]
                        else 1,
                    ),
                )
                created_by_clone = bool(existing_holds)
            else:
                sql = (
                    "INSERT INTO Keywords "
                    "(keyword, uuid, last_modified, version, client_id, deleted) "
                    "VALUES (?, ?, ?, 1, ?, ?)"
                )
                if self.session.backend_type == BackendType.POSTGRESQL:  # type: ignore[attr-defined]
                    sql += " RETURNING id"
                cursor = self.session._execute_with_connection(
                    connection,
                    sql,
                    (
                        keyword,
                        str(uuid4()),
                        now,
                        self.session.client_id,
                        self._active_value(),
                    ),
                )
                if self.session.backend_type == BackendType.POSTGRESQL:  # type: ignore[attr-defined]
                    inserted = cursor.fetchone()
                    keyword_id = int(inserted["id"]) if inserted else 0
                else:
                    keyword_id = int(cursor.lastrowid or 0)
                if keyword_id <= 0:
                    raise DatabaseError("Clone keyword insert returned no identity.")
                created_by_clone = True
            self.session._execute_with_connection(
                connection,
                "INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)",
                (media_id, keyword_id),
            )
            self.session._execute_with_connection(
                connection,
                "INSERT INTO OperationOwnedCloneKeywords "
                "(media_id, keyword_id, operation_id, source_identity, created_by_clone) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    media_id,
                    keyword_id,
                    operation_id,
                    source_identity,
                    created_by_clone
                    if self.session.backend_type == BackendType.POSTGRESQL  # type: ignore[attr-defined]
                    else int(created_by_clone),
                ),
            )

    def _insert_chunks(
        self,
        connection: Any,
        *,
        media_id: int,
        chunks: Sequence[Mapping[str, Any]],
        now: str,
    ) -> None:
        for chunk in chunks:
            chunk_text = chunk.get("chunk_text")
            chunk_index = chunk.get("chunk_index")
            if not isinstance(chunk_text, str):
                raise InputError("snapshot chunk_text must be a string")
            if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
                raise InputError("snapshot chunk_index must be an integer")
            self.session._execute_with_connection(
                connection,
                "INSERT INTO UnvectorizedMediaChunks "
                "(media_id, chunk_text, chunk_index, start_char, end_char, chunk_type, "
                "creation_date, last_modified_orig, is_processed, metadata, uuid, "
                "last_modified, version, client_id, deleted) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                (
                    media_id,
                    chunk_text,
                    chunk_index,
                    chunk.get("start_char"),
                    chunk.get("end_char"),
                    chunk.get("chunk_type"),
                    _normalize_persisted_value(chunk.get("creation_date")),
                    _normalize_persisted_value(chunk.get("last_modified_orig")),
                    self._active_value(),
                    _json_column(chunk.get("metadata")),
                    str(uuid4()),
                    now,
                    self.session.client_id,
                    self._active_value(),
                ),
            )

    def _insert_transcripts(
        self,
        connection: Any,
        *,
        media_id: int,
        transcripts: Sequence[Mapping[str, Any]],
        now: str,
    ) -> None:
        for transcript in transcripts:
            transcription = transcript.get("transcription")
            if transcription is not None and not isinstance(transcription, str):
                raise InputError("snapshot transcription must be a string or null")
            self.session._execute_with_connection(
                connection,
                "INSERT INTO Transcripts "
                "(media_id, whisper_model, transcription, created_at, "
                "transcription_run_id, supersedes_run_id, idempotency_key, uuid, "
                "last_modified, version, client_id, deleted) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                (
                    media_id,
                    transcript.get("whisper_model"),
                    transcription,
                    _normalize_persisted_value(transcript.get("created_at")),
                    transcript.get("transcription_run_id"),
                    transcript.get("supersedes_run_id"),
                    transcript.get("idempotency_key"),
                    str(uuid4()),
                    now,
                    self.session.client_id,
                    self._active_value(),
                ),
            )

    def insert_operation_owned_clone_media(
        self,
        *,
        snapshot: MediaCloneSnapshot,
        operation_id: str,
        source_identity: str,
        expected_content_hash: str,
    ) -> OperationOwnedMediaResult:
        """Insert or exactly replay one operation-owned immutable Media snapshot."""
        operation_id = _validate_identifier(operation_id, "operation_id")
        source_identity = _validate_identifier(source_identity, "source_identity")
        expected_content_hash = _validate_sha256(
            expected_content_hash,
            "expected_content_hash",
        )
        actual_content_hash = hash_media_clone_snapshot(snapshot)
        if not hmac.compare_digest(actual_content_hash, expected_content_hash):
            raise _operation_conflict()

        storage_url = self._storage_url(operation_id, source_identity)
        try:
            with self.session.transaction() as connection:
                self._lock_owned_insert(
                    connection,
                    operation_id=operation_id,
                    source_identity=source_identity,
                )
                candidate = self._verify_owned_candidate(
                    self._owned_candidates(
                        connection,
                        storage_url=storage_url,
                        operation_id=operation_id,
                        source_identity=source_identity,
                    ),
                    storage_url=storage_url,
                    operation_id=operation_id,
                    source_identity=source_identity,
                    expected_content_hash=expected_content_hash,
                    require_pending=True,
                )
                if candidate is not None:
                    self._verify_persisted_logical_copy(
                        connection,
                        media_id=int(candidate["id"]),
                        operation_id=operation_id,
                        source_identity=source_identity,
                        expected_content_hash=expected_content_hash,
                    )
                    return OperationOwnedMediaResult(
                        media_id=int(candidate["id"]),
                        media_uuid=str(candidate["uuid"]),
                        created=False,
                        replayed=True,
                    )

                now = self.session._get_current_utc_timestamp_str()
                media_id, media_uuid = self._insert_media_row(
                    connection,
                    snapshot=snapshot,
                    storage_url=storage_url,
                    operation_id=operation_id,
                    source_identity=source_identity,
                    expected_content_hash=expected_content_hash,
                    now=now,
                )
                self._insert_document_version(
                    connection,
                    media_id=media_id,
                    content=(
                        ""
                        if snapshot.media.get("content") is None
                        else str(snapshot.media["content"])
                    ),
                    source_url=snapshot.media.get("url"),
                    now=now,
                )
                self._insert_keywords(
                    connection,
                    media_id=media_id,
                    keywords=snapshot.media.get("keywords", ()),
                    operation_id=operation_id,
                    source_identity=source_identity,
                    now=now,
                )
                self._insert_chunks(
                    connection,
                    media_id=media_id,
                    chunks=snapshot.chunks,
                    now=now,
                )
                self._insert_transcripts(
                    connection,
                    media_id=media_id,
                    transcripts=snapshot.transcripts,
                    now=now,
                )
                return OperationOwnedMediaResult(
                    media_id=media_id,
                    media_uuid=media_uuid,
                    created=True,
                    replayed=False,
                )
        except (ConflictError, InputError):
            raise
        except MEDIA_NONCRITICAL_EXCEPTIONS as exc:
            logger.bind(exception_type=type(exc).__name__).warning(
                "Operation-owned clone Media insert failed"
            )
            raise DatabaseError("Operation-owned clone Media insert failed.") from None

    def delete_operation_owned_clone_media(
        self,
        *,
        operation_id: str,
        source_identity: str,
        expected_content_hash: str,
    ) -> int:
        """Hard-delete only the exact operation-owned Media graph."""
        operation_id = _validate_identifier(operation_id, "operation_id")
        source_identity = _validate_identifier(source_identity, "source_identity")
        expected_content_hash = _validate_sha256(
            expected_content_hash,
            "expected_content_hash",
        )
        storage_url = self._storage_url(operation_id, source_identity)
        try:
            with self.session.transaction() as connection:
                candidate = self._verify_owned_candidate(
                    self._owned_candidates(
                        connection,
                        storage_url=storage_url,
                        operation_id=operation_id,
                        source_identity=source_identity,
                    ),
                    storage_url=storage_url,
                    operation_id=operation_id,
                    source_identity=source_identity,
                    expected_content_hash=expected_content_hash,
                    require_pending=False,
                )
                if candidate is None:
                    return 0
                media_id = int(candidate["id"])
                keyword_holds = self.session._fetchall_with_connection(
                    connection,
                    "SELECT keyword_id, created_by_clone "
                    "FROM OperationOwnedCloneKeywords WHERE media_id = ? "
                    "AND operation_id = ? AND source_identity = ?",
                    (media_id, operation_id, source_identity),
                )
                self.session._execute_with_connection(
                    connection,
                    "DELETE FROM DocumentVersionIdentifiers WHERE dv_id IN "
                    "(SELECT id FROM DocumentVersions WHERE media_id = ?)",
                    (media_id,),
                )
                self.session._execute_with_connection(
                    connection,
                    "DELETE FROM OperationOwnedCloneKeywords WHERE media_id = ? "
                    "AND operation_id = ? AND source_identity = ?",
                    (media_id, operation_id, source_identity),
                )
                for table in (
                    "DocumentVersions",
                    "MediaKeywords",
                    "UnvectorizedMediaChunks",
                    "MediaChunks",
                    "Transcripts",
                ):
                    self.session._execute_with_connection(
                        connection,
                        f"DELETE FROM {table} WHERE media_id = ?",  # nosec B608
                        (media_id,),
                    )
                cursor = self.session._execute_with_connection(
                    connection,
                    "DELETE FROM Media WHERE id = ? AND system_operation_kind = ? "
                    "AND system_operation_id = ? AND system_source_identity = ? "
                    "AND system_content_hash = ?",
                    (
                        media_id,
                        _CLONE_OPERATION_KIND,
                        operation_id,
                        source_identity,
                        expected_content_hash,
                    ),
                )
                if cursor.rowcount != 1:
                    raise _operation_conflict()
                for hold in keyword_holds:
                    if not bool(hold["created_by_clone"]):
                        continue
                    self.session._execute_with_connection(
                        connection,
                        "DELETE FROM Keywords WHERE id = ? "
                        "AND NOT EXISTS (SELECT 1 FROM MediaKeywords "
                        "WHERE keyword_id = ?) "
                        "AND NOT EXISTS (SELECT 1 FROM OperationOwnedCloneKeywords "
                        "WHERE keyword_id = ?)",
                        (
                            int(hold["keyword_id"]),
                            int(hold["keyword_id"]),
                            int(hold["keyword_id"]),
                        ),
                    )
                return 1
        except (ConflictError, InputError):
            raise
        except MEDIA_NONCRITICAL_EXCEPTIONS as exc:
            logger.bind(exception_type=type(exc).__name__).warning(
                "Operation-owned clone Media cleanup failed"
            )
            raise DatabaseError("Operation-owned clone Media cleanup failed.") from None

    def list_operation_owned_clone_media(
        self,
        *,
        operation_id: str,
        limit: int = 100,
    ) -> list[OperationOwnedMediaReference]:
        """List bounded pending identities for one caller-supplied operation."""
        operation_id = _validate_identifier(operation_id, "operation_id")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise InputError("limit must be an integer between 1 and 100")
        with self.session.transaction() as connection:
            rows = self.session._fetchall_with_connection(
                connection,
                "SELECT id, uuid, system_source_identity, system_content_hash "
                "FROM Media WHERE system_operation_kind = ? AND system_operation_id = ? "
                "ORDER BY id ASC LIMIT ?",
                (_CLONE_OPERATION_KIND, operation_id, limit),
            )
        return [
            OperationOwnedMediaReference(
                media_id=int(row["id"]),
                media_uuid=str(row["uuid"]),
                source_identity=str(row["system_source_identity"]),
                expected_content_hash=str(row["system_content_hash"]),
            )
            for row in rows
        ]

    def confirm_operation_owned_clone_media(
        self,
        *,
        operation_id: str,
        source_identity: str,
        expected_content_hash: str,
    ) -> int:
        """Activate and clear markers from one exact pending clone Media row."""
        operation_id = _validate_identifier(operation_id, "operation_id")
        source_identity = _validate_identifier(source_identity, "source_identity")
        expected_content_hash = _validate_sha256(
            expected_content_hash,
            "expected_content_hash",
        )
        storage_url = self._storage_url(operation_id, source_identity)
        with self.session.transaction() as connection:
            candidate = self._verify_owned_candidate(
                self._owned_candidates(
                    connection,
                    storage_url=storage_url,
                    operation_id=operation_id,
                    source_identity=source_identity,
                ),
                storage_url=storage_url,
                operation_id=operation_id,
                source_identity=source_identity,
                expected_content_hash=expected_content_hash,
                require_pending=True,
            )
            if candidate is None:
                return 0
            self._verify_persisted_logical_copy(
                connection,
                media_id=int(candidate["id"]),
                operation_id=operation_id,
                source_identity=source_identity,
                expected_content_hash=expected_content_hash,
            )
            media_id = int(candidate["id"])
            released_value = (
                False
                if self.session.backend_type == BackendType.POSTGRESQL  # type: ignore[attr-defined]
                else 0
            )
            self.session._execute_with_connection(
                connection,
                "UPDATE OperationOwnedCloneKeywords SET created_by_clone = ? "
                "WHERE keyword_id IN (SELECT keyword_id "
                "FROM OperationOwnedCloneKeywords WHERE media_id = ? "
                "AND operation_id = ? AND source_identity = ?)",
                (released_value, media_id, operation_id, source_identity),
            )
            self.session._execute_with_connection(
                connection,
                "DELETE FROM OperationOwnedCloneKeywords WHERE media_id = ? "
                "AND operation_id = ? AND source_identity = ?",
                (media_id, operation_id, source_identity),
            )
            cursor = self.session._execute_with_connection(
                connection,
                "UPDATE Media SET is_trash = ?, trash_date = NULL, "
                "system_operation_id = NULL, system_operation_kind = NULL, "
                "system_source_identity = NULL, system_content_hash = NULL, "
                "last_modified = ?, version = version + 1, client_id = ? "
                "WHERE id = ? AND deleted = ? AND is_trash = ? "
                "AND system_operation_kind = ? AND system_operation_id = ? "
                "AND system_source_identity = ? AND system_content_hash = ?",
                (
                    self._active_value(),
                    self.session._get_current_utc_timestamp_str(),
                    self.session.client_id,
                    media_id,
                    self._active_value(),
                    True if self.session.backend_type == BackendType.POSTGRESQL else 1,  # type: ignore[attr-defined]
                    _CLONE_OPERATION_KIND,
                    operation_id,
                    source_identity,
                    expected_content_hash,
                ),
            )
            if cursor.rowcount != 1:
                raise _operation_conflict()
            return 1

    def read(self, media_ids: Sequence[int]) -> dict[int, MediaCloneSnapshot]:
        """Return active Media rows and child collections in requested ID order."""
        from tldw_Server_API.app.core.Sharing.clone_models import (
            CloneSnapshotUnavailable,
            MediaCloneSnapshot,
        )

        requested_ids = self._validate_media_ids(media_ids)
        if not requested_ids:
            return {}

        def materialize(backend: Any, connection: Any) -> dict[int, MediaCloneSnapshot]:
            placeholders = ", ".join("?" for _ in requested_ids)
            active_value = False if backend.backend_type == BackendType.POSTGRESQL else 0

            def read_rows(query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
                query_result = backend.execute(
                    query,
                    params,
                    connection=connection,
                    log_errors=False,
                )
                return [dict(row) for row in query_result.rows]

            media_rows = read_rows(
                f"SELECT * FROM Media WHERE id IN ({placeholders}) "  # nosec B608
                "AND deleted = ? AND is_trash = ?",
                (*requested_ids, active_value, active_value),
            )
            media_by_id = {int(row["id"]): row for row in media_rows}
            if len(media_by_id) != len(requested_ids) or any(
                media_id not in media_by_id for media_id in requested_ids
            ):
                raise CloneSnapshotUnavailable(cleanup_state="complete")

            chunk_rows = read_rows(
                f"SELECT * FROM UnvectorizedMediaChunks "  # nosec B608
                f"WHERE media_id IN ({placeholders}) AND deleted = ? "
                "ORDER BY media_id, chunk_index, id",
                (*requested_ids, active_value),
            )
            transcript_rows = read_rows(
                f"SELECT t.* FROM Transcripts t JOIN Media m ON m.id = t.media_id "  # nosec B608
                f"WHERE t.media_id IN ({placeholders}) AND t.deleted = ? "
                "AND m.deleted = ? AND m.is_trash = ? "
                "ORDER BY t.media_id, "
                "CASE WHEN m.latest_transcription_run_id IS NOT NULL "
                "AND t.transcription_run_id = m.latest_transcription_run_id THEN 0 ELSE 1 END, "
                "CASE WHEN t.transcription_run_id IS NULL THEN 1 ELSE 0 END, "
                "t.transcription_run_id DESC, t.created_at DESC, t.id DESC",
                (*requested_ids, active_value, active_value, active_value),
            )
            keyword_order = (
                "LOWER(k.keyword), k.keyword"
                if backend.backend_type == BackendType.POSTGRESQL
                else "k.keyword COLLATE NOCASE"
            )
            keyword_rows = read_rows(
                f"SELECT mk.media_id, k.keyword FROM MediaKeywords mk "  # nosec B608
                "JOIN Keywords k ON k.id = mk.keyword_id "
                "JOIN Media m ON m.id = mk.media_id "
                f"WHERE mk.media_id IN ({placeholders}) AND k.deleted = ? "
                "AND m.deleted = ? AND m.is_trash = ? "
                f"ORDER BY mk.media_id, {keyword_order}, k.id",  # nosec B608
                (*requested_ids, active_value, active_value, active_value),
            )

            chunks_by_media = {media_id: [] for media_id in requested_ids}
            transcripts_by_media = {media_id: [] for media_id in requested_ids}
            keywords_by_media = {media_id: [] for media_id in requested_ids}
            for row in chunk_rows:
                media_id = int(row["media_id"])
                if media_id not in chunks_by_media:
                    raise CloneSnapshotUnavailable(cleanup_state="complete")
                chunks_by_media[media_id].append(row)
            for row in transcript_rows:
                media_id = int(row["media_id"])
                if media_id not in transcripts_by_media:
                    raise CloneSnapshotUnavailable(cleanup_state="complete")
                transcripts_by_media[media_id].append(row)
            for row in keyword_rows:
                media_id = int(row["media_id"])
                if media_id not in keywords_by_media:
                    raise CloneSnapshotUnavailable(cleanup_state="complete")
                keywords_by_media[media_id].append(str(row["keyword"]))

            snapshots: dict[int, MediaCloneSnapshot] = {}
            for media_id in requested_ids:
                media_row = dict(media_by_id[media_id])
                media_row["keywords"] = tuple(keywords_by_media[media_id])
                snapshots[media_id] = MediaCloneSnapshot.from_rows(
                    media=media_row,
                    chunks=chunks_by_media[media_id],
                    transcripts=transcripts_by_media[media_id],
                )
            return snapshots

        return self._run_snapshot(materialize)


def read_media_clone_snapshots(
    self: MediaDbLike,
    media_ids: Sequence[int],
) -> dict[int, MediaCloneSnapshot]:
    """MediaDatabase binding for repeatable clone source reads."""
    return CloneSnapshotRepository.from_legacy_db(self).read(media_ids)


def insert_operation_owned_clone_media(
    self: MediaDbLike,
    *,
    snapshot: MediaCloneSnapshot,
    operation_id: str,
    source_identity: str,
    expected_content_hash: str,
) -> OperationOwnedMediaResult:
    """MediaDatabase binding for an isolated operation-owned clone insert."""
    return CloneSnapshotRepository.from_legacy_db(self).insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_content_hash,
    )


def delete_operation_owned_clone_media(
    self: MediaDbLike,
    *,
    operation_id: str,
    source_identity: str,
    expected_content_hash: str,
) -> int:
    """MediaDatabase binding for exact operation-owned clone cleanup."""
    return CloneSnapshotRepository.from_legacy_db(self).delete_operation_owned_clone_media(
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_content_hash,
    )


def list_operation_owned_clone_media(
    self: MediaDbLike,
    *,
    operation_id: str,
    limit: int = 100,
) -> list[OperationOwnedMediaReference]:
    """MediaDatabase binding for bounded pending clone Media enumeration."""
    return CloneSnapshotRepository.from_legacy_db(self).list_operation_owned_clone_media(
        operation_id=operation_id,
        limit=limit,
    )


def confirm_operation_owned_clone_media(
    self: MediaDbLike,
    *,
    operation_id: str,
    source_identity: str,
    expected_content_hash: str,
) -> int:
    """MediaDatabase binding for exact pending clone Media confirmation."""
    return CloneSnapshotRepository.from_legacy_db(self).confirm_operation_owned_clone_media(
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_content_hash,
    )


__all__ = [
    "CloneSnapshotRepository",
    "OperationOwnedMediaReference",
    "OperationOwnedMediaResult",
    "confirm_operation_owned_clone_media",
    "delete_operation_owned_clone_media",
    "hash_media_clone_snapshot",
    "insert_operation_owned_clone_media",
    "list_operation_owned_clone_media",
    "read_media_clone_snapshots",
]

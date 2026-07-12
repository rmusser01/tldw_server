"""Owner-scoped persistence for playlist inspection and ingest manifests."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Generic, TypeVar, overload
from uuid import uuid4

from tldw_Server_API.app.core.AuthNZ.crypto_utils import derive_hmac_key

_NOT_FOUND_MESSAGE = "playlist resource not found"
_CURSOR_ORDER = "ordinal_asc"
_MAX_CURSOR_LENGTH = 4096
_MAX_CURSOR_PAYLOAD = 2048
_MAX_PAGE_SIZE = 500
_COMPACT_DISPLAY_METADATA_FIELDS = frozenset(
    {
        "title",
        "channel_or_uploader",
        "duration_seconds",
        "published_at",
        "thumbnail_url",
        "playlist_id",
        "playlist_title",
    }
)


class PlaylistIngestNotFoundError(LookupError):
    """Raised for absent, unauthorized, expired, or invalid cursor resources."""


class PlaylistIngestConflictError(RuntimeError):
    """Raised when immutable or compare-and-set state has changed."""


@dataclass(frozen=True, slots=True)
class PlaylistPreflightRecord:
    preflight_id: str
    owner_user_id: str
    status: str
    source_url: str
    source_kind: str
    playlist_id: str | None
    job_id: int | None
    summary: dict[str, Any] | None
    error: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class PlaylistItemRecord:
    occurrence_id: str
    ordinal: int
    source_url: str | None
    normalized_source_id: str | None
    source_kind: str
    display_metadata: dict[str, Any]
    occurrence_index_for_source: int | None = None
    availability: str | None = None
    duplicate_status: str | None = None
    duplicate_of_occurrence_id: str | None = None
    selected_by_default: bool | None = None


@dataclass(frozen=True, slots=True)
class PlaylistMaterializationRecord:
    materialization_id: str
    preflight_id: str
    owner_user_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime

    @property
    def id(self) -> str:
        """Return the caller-facing resource identifier."""
        return self.materialization_id


@dataclass(frozen=True, slots=True)
class MediaIngestRunRecord:
    run_id: str
    owner_user_id: str
    status: str
    collection_id: int | None
    processing_options: dict[str, Any] | None
    playlist_summaries: list[dict[str, Any]] | None
    batch_ids: list[str] | None
    version: int
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class MediaIngestRunItemRecord:
    occurrence_id: str
    ordinal: int
    input_kind: str
    state: str
    outcome: str | None
    materialization_id: str | None
    source_url: str | None
    normalized_source_id: str | None
    source_kind: str | None
    display_metadata: dict[str, Any]
    duplicate_policy: str | None
    metadata_patch: dict[str, Any] | None
    job_id: int | None
    batch_id: str | None
    attempt: int
    progress_percent: float | None
    progress_message: str | None
    retryable: bool
    media_id: int | None


@dataclass(frozen=True, slots=True)
class MediaIngestRunEventRecord:
    event_id: int
    run_id: str
    occurrence_id: str | None
    job_id: int | None
    batch_id: str | None
    event_type: str
    state: str | None
    outcome: str | None
    progress_percent: float | None
    progress_message: str | None
    attrs: dict[str, Any]
    occurred_at: datetime


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PlaylistPage(Sequence[T], Generic[T]):
    """List-compatible immutable page plus its signed continuation cursor."""

    items: tuple[T, ...]
    next_cursor: str | None = None

    def __len__(self) -> int:
        return len(self.items)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[T, ...]: ...

    def __getitem__(self, index: int | slice) -> T | tuple[T, ...]:
        return self.items[index]

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)


class PlaylistIngestStore:
    """Persist owner-scoped playlist resources in the injected Jobs database."""

    def __init__(self, job_manager: Any) -> None:
        if not callable(getattr(job_manager, "_connect", None)):
            raise TypeError("job_manager must expose the Jobs connection contract")
        self._jobs = job_manager
        self._postgres = getattr(job_manager, "backend", "sqlite") == "postgres"
        self._cursor_key = derive_hmac_key()

    @staticmethod
    def _owner(owner_user_id: str) -> str:
        if type(owner_user_id) is not str:
            raise ValueError("owner_user_id is required")
        owner = owner_user_id.strip()
        if not owner:
            raise ValueError("owner_user_id is required")
        return owner

    def _now(self) -> datetime:
        value = self._jobs._clock.now_utc()
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _db_datetime(self, value: datetime) -> datetime | str:
        normalized = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        normalized = normalized.astimezone(timezone.utc)
        return normalized if self._postgres else normalized.isoformat()

    def _future_expiry(self, value: datetime, *, now: datetime) -> datetime:
        normalized = self._datetime(value)
        if normalized <= now:
            raise ValueError("expires_at must be in the future")
        return normalized

    def _json_value(self, value: Any) -> Any:
        if value is None:
            return None
        if self._postgres:
            from psycopg.types.json import Jsonb

            return Jsonb(value)
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def _query(self, db: Any, sql: str, params: Sequence[Any] = ()) -> Any:
        statement = sql.replace("?", "%s") if self._postgres else sql
        result = db.execute(statement, tuple(params))
        return db if result is None else result

    @contextmanager
    def _connection(self, *, write: bool) -> Iterator[Any]:
        conn = self._jobs._connect()
        try:
            if self._postgres:
                with self._jobs._pg_cursor(conn) as cursor:
                    yield cursor
            else:
                if write:
                    conn.execute("BEGIN IMMEDIATE")
                yield conn
            if write:
                conn.commit()
        except Exception:
            if write:
                conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _row_dict(row: Any) -> dict[str, Any]:
        return dict(row)

    @staticmethod
    def _json_dict(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        parsed = json.loads(value) if isinstance(value, str) else value
        return dict(parsed) if isinstance(parsed, Mapping) else None

    @staticmethod
    def _json_list(value: Any) -> list[Any] | None:
        if value is None:
            return None
        parsed = json.loads(value) if isinstance(value, str) else value
        return list(parsed) if isinstance(parsed, list) else None

    @staticmethod
    def _datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @classmethod
    def _preflight_record(cls, row: Any) -> PlaylistPreflightRecord:
        data = cls._row_dict(row)
        return PlaylistPreflightRecord(
            preflight_id=str(data["preflight_id"]),
            owner_user_id=str(data["owner_user_id"]),
            status=str(data["status"]),
            source_url=str(data["source_url"]),
            source_kind=str(data["source_kind"]),
            playlist_id=data.get("playlist_id"),
            job_id=int(data["job_id"]) if data.get("job_id") is not None else None,
            summary=cls._json_dict(data.get("summary_json")),
            error=cls._json_dict(data.get("error_json")),
            created_at=cls._datetime(data["created_at"]),
            updated_at=cls._datetime(data["updated_at"]),
            expires_at=cls._datetime(data["expires_at"]),
        )

    @classmethod
    def _materialization_record(cls, row: Any) -> PlaylistMaterializationRecord:
        data = cls._row_dict(row)
        return PlaylistMaterializationRecord(
            materialization_id=str(data["materialization_id"]),
            preflight_id=str(data["preflight_id"]),
            owner_user_id=str(data["owner_user_id"]),
            status=str(data["status"]),
            created_at=cls._datetime(data["created_at"]),
            updated_at=cls._datetime(data["updated_at"]),
            expires_at=cls._datetime(data["expires_at"]),
        )

    @classmethod
    def _playlist_item_record(cls, row: Any) -> PlaylistItemRecord:
        data = cls._row_dict(row)
        return PlaylistItemRecord(
            occurrence_id=str(data["occurrence_id"]),
            ordinal=int(data["ordinal"]),
            source_url=data.get("source_url"),
            normalized_source_id=data.get("normalized_source_id"),
            source_kind=str(data["source_kind"]),
            display_metadata=cls._json_dict(data.get("display_metadata_json")) or {},
            occurrence_index_for_source=(
                int(data["occurrence_index_for_source"])
                if data.get("occurrence_index_for_source") is not None
                else None
            ),
            availability=data.get("availability"),
            duplicate_status=data.get("duplicate_status"),
            duplicate_of_occurrence_id=data.get("duplicate_of_occurrence_id"),
            selected_by_default=(
                bool(data["selected_by_default"])
                if data.get("selected_by_default") is not None
                else None
            ),
        )

    @classmethod
    def _run_record(cls, row: Any) -> MediaIngestRunRecord:
        data = cls._row_dict(row)
        return MediaIngestRunRecord(
            run_id=str(data["run_id"]),
            owner_user_id=str(data["owner_user_id"]),
            status=str(data["status"]),
            collection_id=(int(data["collection_id"]) if data.get("collection_id") is not None else None),
            processing_options=cls._json_dict(data.get("processing_options_json")),
            playlist_summaries=cls._json_list(data.get("playlist_summaries_json")),
            batch_ids=cls._json_list(data.get("batch_ids_json")),
            version=int(data["version"]),
            created_at=cls._datetime(data["created_at"]),
            updated_at=cls._datetime(data["updated_at"]),
            expires_at=cls._datetime(data["expires_at"]),
        )

    @classmethod
    def _run_item_record(cls, row: Any) -> MediaIngestRunItemRecord:
        data = cls._row_dict(row)
        return MediaIngestRunItemRecord(
            occurrence_id=str(data["occurrence_id"]),
            ordinal=int(data["ordinal"]),
            input_kind=str(data["input_kind"]),
            state=str(data["state"]),
            outcome=data.get("outcome"),
            materialization_id=data.get("materialization_id"),
            source_url=data.get("source_url"),
            normalized_source_id=data.get("normalized_source_id"),
            source_kind=data.get("source_kind"),
            display_metadata=cls._json_dict(data.get("display_metadata_json")) or {},
            duplicate_policy=data.get("duplicate_policy"),
            metadata_patch=cls._json_dict(data.get("metadata_patch_json")),
            job_id=int(data["job_id"]) if data.get("job_id") is not None else None,
            batch_id=data.get("batch_id"),
            attempt=int(data["attempt"]),
            progress_percent=(
                float(data["progress_percent"]) if data.get("progress_percent") is not None else None
            ),
            progress_message=data.get("progress_message"),
            retryable=bool(data["retryable"]),
            media_id=int(data["media_id"]) if data.get("media_id") is not None else None,
        )

    @classmethod
    def _event_record(cls, row: Any) -> MediaIngestRunEventRecord:
        data = cls._row_dict(row)
        return MediaIngestRunEventRecord(
            event_id=int(data["event_id"]),
            run_id=str(data["run_id"]),
            occurrence_id=data.get("occurrence_id"),
            job_id=int(data["job_id"]) if data.get("job_id") is not None else None,
            batch_id=data.get("batch_id"),
            event_type=str(data["event_type"]),
            state=data.get("state"),
            outcome=data.get("outcome"),
            progress_percent=(
                float(data["progress_percent"]) if data.get("progress_percent") is not None else None
            ),
            progress_message=data.get("progress_message"),
            attrs=cls._json_dict(data.get("attrs_json")) or {},
            occurred_at=cls._datetime(data["occurred_at"]),
        )

    @staticmethod
    def _not_found() -> PlaylistIngestNotFoundError:
        return PlaylistIngestNotFoundError(_NOT_FOUND_MESSAGE)

    @staticmethod
    def _b64encode(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def _b64decode(segment: str) -> bytes:
        if not segment or "=" in segment:
            raise ValueError("noncanonical base64url segment")
        padding = "=" * (-len(segment) % 4)
        decoded = base64.b64decode(segment + padding, altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=") != segment:
            raise ValueError("noncanonical base64url segment")
        return decoded

    def _encode_cursor(self, *, owner: str, kind: str, resource_id: str, last_ordinal: int) -> str:
        payload = {
            "v": 1,
            "owner": owner,
            "kind": kind,
            "resource_id": resource_id,
            "order": _CURSOR_ORDER,
            "last_ordinal": last_ordinal,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        signature = hmac.new(self._cursor_key, raw, hashlib.sha256).digest()
        return f"{self._b64encode(raw)}.{self._b64encode(signature)}"

    def _decode_cursor(self, cursor: str, *, owner: str, kind: str, resource_id: str) -> int:
        try:
            if not isinstance(cursor, str) or not (1 <= len(cursor) <= _MAX_CURSOR_LENGTH):
                raise ValueError
            payload_segment, signature_segment = cursor.split(".")
            raw = self._b64decode(payload_segment)
            signature = self._b64decode(signature_segment)
            if len(raw) > _MAX_CURSOR_PAYLOAD or len(signature) != hashlib.sha256().digest_size:
                raise ValueError
            expected = hmac.new(self._cursor_key, raw, hashlib.sha256).digest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict) or set(payload) != {
                "v",
                "owner",
                "kind",
                "resource_id",
                "order",
                "last_ordinal",
            }:
                raise ValueError
            if (
                payload["v"] != 1
                or payload["owner"] != owner
                or payload["kind"] != kind
                or payload["resource_id"] != resource_id
                or payload["order"] != _CURSOR_ORDER
                or type(payload["last_ordinal"]) is not int
                or payload["last_ordinal"] < 1
            ):
                raise ValueError
            return int(payload["last_ordinal"])
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise self._not_found() from exc

    @staticmethod
    def _page_limit(limit: int) -> int:
        if type(limit) is not int or not 1 <= limit <= _MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {_MAX_PAGE_SIZE}")
        return limit

    def create_preflight(
        self,
        owner_user_id: str,
        *,
        source_url: str,
        source_kind: str,
        expires_at: datetime,
        playlist_id: str | None = None,
        job_id: int | None = None,
    ) -> PlaylistPreflightRecord:
        """Create a pending immutable-inspection resource."""
        owner = self._owner(owner_user_id)
        preflight_id = str(uuid4())
        now = self._now()
        expires = self._future_expiry(expires_at, now=now)
        with self._connection(write=True) as db:
            self._query(
                db,
                """
                INSERT INTO playlist_preflights (
                    preflight_id, owner_user_id, status, source_url, source_kind,
                    playlist_id, job_id, created_at, updated_at, expires_at
                ) VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    preflight_id,
                    owner,
                    str(source_url),
                    str(source_kind),
                    playlist_id,
                    job_id,
                    self._db_datetime(now),
                    self._db_datetime(now),
                    self._db_datetime(expires),
                ),
            )
        return self.get_preflight(owner, preflight_id)

    def get_preflight(self, owner_user_id: str, preflight_id: str) -> PlaylistPreflightRecord:
        """Return one owner-scoped preflight or the generic not-found error."""
        owner = self._owner(owner_user_id)
        with self._connection(write=False) as db:
            row = self._query(
                db,
                """
                SELECT * FROM playlist_preflights
                WHERE owner_user_id = ? AND preflight_id = ? AND expires_at > ?
                """,
                (owner, str(preflight_id), self._db_datetime(self._now())),
            ).fetchone()
        if row is None:
            raise self._not_found()
        return self._preflight_record(row)

    def replace_preflight_snapshot(
        self,
        owner_user_id: str,
        preflight_id: str,
        *,
        status: str,
        items: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> PlaylistPreflightRecord:
        """Replace inspection items and status atomically while the snapshot is mutable."""
        owner = self._owner(owner_user_id)
        occurrence_ids = [str(item.get("occurrence_id") or "").strip() for item in items]
        ordinals = [item.get("ordinal") for item in items]
        if any(not value for value in occurrence_ids) or len(set(occurrence_ids)) != len(occurrence_ids):
            raise ValueError("occurrence_id values must be non-empty and unique")
        if any(type(value) is not int or value < 1 for value in ordinals) or len(set(ordinals)) != len(ordinals):
            raise ValueError("ordinal values must be positive and unique")

        now = self._now()
        with self._connection(write=True) as db:
            if self._postgres:
                mutable_preflight_query = """
                    SELECT status, expires_at FROM playlist_preflights
                    WHERE owner_user_id = ? AND preflight_id = ? AND expires_at > ?
                    FOR UPDATE
                """
            else:
                mutable_preflight_query = """
                    SELECT status, expires_at FROM playlist_preflights
                    WHERE owner_user_id = ? AND preflight_id = ? AND expires_at > ?
                """
            row = self._query(
                db,
                mutable_preflight_query,
                (owner, str(preflight_id), self._db_datetime(now)),
            ).fetchone()
            if row is None:
                raise self._not_found()
            if str(self._row_dict(row)["status"]) not in {"pending", "running"}:
                raise PlaylistIngestConflictError("ready playlist snapshot ordering is immutable")

            self._query(
                db,
                "DELETE FROM playlist_preflight_items WHERE owner_user_id = ? AND preflight_id = ?",
                (owner, str(preflight_id)),
            )
            for item, occurrence_id, ordinal in zip(items, occurrence_ids, ordinals, strict=True):
                self._query(
                    db,
                    """
                    INSERT INTO playlist_preflight_items (
                        preflight_id, owner_user_id, occurrence_id, ordinal,
                        occurrence_index_for_source, source_url, normalized_source_id,
                        source_kind, availability, duplicate_status,
                        duplicate_of_occurrence_id, selected_by_default, display_metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(preflight_id),
                        owner,
                        occurrence_id,
                        ordinal,
                        int(item.get("occurrence_index_for_source", 1)),
                        item.get("source_url"),
                        item.get("normalized_source_id"),
                        str(item.get("source_kind") or "unknown"),
                        str(item.get("availability") or "unknown"),
                        str(item.get("duplicate_status") or "unknown"),
                        item.get("duplicate_of_occurrence_id"),
                        bool(item.get("selected_by_default", True)),
                        self._json_value(dict(item.get("display_metadata") or {})),
                    ),
                )
            self._query(
                db,
                """
                UPDATE playlist_preflights
                SET status = ?, summary_json = ?, error_json = ?, updated_at = ?
                WHERE owner_user_id = ? AND preflight_id = ?
                """,
                (
                    str(status),
                    self._json_value(dict(summary)) if summary is not None else None,
                    self._json_value(dict(error)) if error is not None else None,
                    self._db_datetime(now),
                    owner,
                    str(preflight_id),
                ),
            )
        return self.get_preflight(owner, preflight_id)

    def _list_ordinal_page(
        self,
        *,
        owner: str,
        resource_id: str,
        kind: str,
        parent_table: str,
        parent_id_column: str,
        item_table: str,
        limit: int,
        cursor: str | None,
        converter: Any,
    ) -> PlaylistPage[Any]:
        page_limit = self._page_limit(limit)
        last_ordinal = self._decode_cursor(
            cursor,
            owner=owner,
            kind=kind,
            resource_id=resource_id,
        ) if cursor else 0
        with self._connection(write=False) as db:
            exists = self._query(
                db,
                f"""
                SELECT 1 FROM {parent_table}
                WHERE owner_user_id = ? AND {parent_id_column} = ? AND expires_at > ?
                """,  # nosec B608
                (owner, resource_id, self._db_datetime(self._now())),
            ).fetchone()
            if exists is None:
                raise self._not_found()
            rows = self._query(
                db,
                f"""
                SELECT * FROM {item_table}
                WHERE owner_user_id = ? AND {parent_id_column} = ? AND ordinal > ?
                ORDER BY ordinal ASC
                LIMIT ?
                """,  # nosec B608
                (owner, resource_id, last_ordinal, page_limit + 1),
            ).fetchall()
        has_more = len(rows) > page_limit
        records = tuple(converter(row) for row in rows[:page_limit])
        next_cursor = None
        if has_more and records:
            next_cursor = self._encode_cursor(
                owner=owner,
                kind=kind,
                resource_id=resource_id,
                last_ordinal=records[-1].ordinal,
            )
        return PlaylistPage(records, next_cursor)

    def list_preflight_items(
        self,
        owner_user_id: str,
        preflight_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> PlaylistPage[PlaylistItemRecord]:
        """List immutable preflight occurrences in ordinal order."""
        owner = self._owner(owner_user_id)
        return self._list_ordinal_page(
            owner=owner,
            resource_id=str(preflight_id),
            kind="preflight_items",
            parent_table="playlist_preflights",
            parent_id_column="preflight_id",
            item_table="playlist_preflight_items",
            limit=limit,
            cursor=cursor,
            converter=self._playlist_item_record,
        )

    def create_materialization(
        self,
        owner_user_id: str,
        *,
        preflight_id: str,
        occurrence_ids: Sequence[str],
        expires_at: datetime | None = None,
    ) -> PlaylistMaterializationRecord:
        """Copy selected authoritative identities from a ready preflight atomically."""
        owner = self._owner(owner_user_id)
        selected = [str(value).strip() for value in occurrence_ids]
        if not selected or any(not value for value in selected):
            raise ValueError("at least one selected occurrence_id is required")
        if len(set(selected)) != len(selected):
            raise ValueError("duplicate occurrence_id selection is not allowed")

        materialization_id = str(uuid4())
        now = self._now()
        requested_expiry = self._future_expiry(expires_at, now=now) if expires_at is not None else None
        with self._connection(write=True) as db:
            preflight = self._query(
                db,
                """
                SELECT expires_at FROM playlist_preflights
                WHERE owner_user_id = ? AND preflight_id = ? AND status = 'ready' AND expires_at > ?
                """,
                (owner, str(preflight_id), self._db_datetime(now)),
            ).fetchone()
            if preflight is None:
                raise self._not_found()

            placeholders = ", ".join(["?"] * len(selected))
            rows = self._query(
                db,
                f"""
                SELECT * FROM playlist_preflight_items
                WHERE owner_user_id = ? AND preflight_id = ?
                  AND occurrence_id IN ({placeholders})
                ORDER BY ordinal ASC
                """,  # nosec B608
                (owner, str(preflight_id), *selected),
            ).fetchall()
            found = {str(self._row_dict(row)["occurrence_id"]) for row in rows}
            if found != set(selected):
                raise ValueError("selected occurrence_id does not exist in the source snapshot")
            if any(not self._row_dict(row).get("source_url") for row in rows):
                raise ValueError("selected occurrence_id does not have an authoritative source URL")

            preflight_expiry = self._datetime(self._row_dict(preflight)["expires_at"])
            materialization_expiry = requested_expiry or preflight_expiry
            self._query(
                db,
                """
                INSERT INTO playlist_materializations (
                    materialization_id, preflight_id, owner_user_id, status,
                    created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, 'ready', ?, ?, ?)
                """,
                (
                    materialization_id,
                    str(preflight_id),
                    owner,
                    self._db_datetime(now),
                    self._db_datetime(now),
                    self._db_datetime(materialization_expiry),
                ),
            )
            for row in rows:
                data = self._row_dict(row)
                source_display = self._json_dict(data.get("display_metadata_json")) or {}
                display = {
                    key: source_display[key]
                    for key in _COMPACT_DISPLAY_METADATA_FIELDS
                    if key in source_display
                }
                self._query(
                    db,
                    """
                    INSERT INTO playlist_materialization_items (
                        materialization_id, owner_user_id, occurrence_id, ordinal,
                        source_url, normalized_source_id, source_kind, display_metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        materialization_id,
                        owner,
                        data["occurrence_id"],
                        data["ordinal"],
                        data["source_url"],
                        data.get("normalized_source_id"),
                        data["source_kind"],
                        self._json_value(display),
                    ),
                )
        return self.get_materialization(owner, materialization_id)

    def get_materialization(
        self,
        owner_user_id: str,
        materialization_id: str,
    ) -> PlaylistMaterializationRecord:
        """Return one owner-scoped materialization."""
        owner = self._owner(owner_user_id)
        with self._connection(write=False) as db:
            row = self._query(
                db,
                """
                SELECT * FROM playlist_materializations
                WHERE owner_user_id = ? AND materialization_id = ? AND expires_at > ?
                """,
                (owner, str(materialization_id), self._db_datetime(self._now())),
            ).fetchone()
        if row is None:
            raise self._not_found()
        return self._materialization_record(row)

    def list_materialization_items(
        self,
        owner_user_id: str,
        materialization_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> PlaylistPage[PlaylistItemRecord]:
        """List copied identities in immutable source ordinal order."""
        owner = self._owner(owner_user_id)
        return self._list_ordinal_page(
            owner=owner,
            resource_id=str(materialization_id),
            kind="materialization_items",
            parent_table="playlist_materializations",
            parent_id_column="materialization_id",
            item_table="playlist_materialization_items",
            limit=limit,
            cursor=cursor,
            converter=self._playlist_item_record,
        )

    def create_run(
        self,
        owner_user_id: str,
        *,
        materialization_ids: Sequence[str],
        processing_options: Mapping[str, Any] | None = None,
        playlist_summaries: Sequence[Mapping[str, Any]] | None = None,
        collection_id: int | None = None,
        expires_at: datetime | None = None,
    ) -> MediaIngestRunRecord:
        """Create a staged run and copy all selected identities in one transaction."""
        owner = self._owner(owner_user_id)
        materializations = [str(value).strip() for value in materialization_ids]
        if not materializations or any(not value for value in materializations):
            raise ValueError("at least one materialization_id is required")
        if len(set(materializations)) != len(materializations):
            raise ValueError("materialization_ids must be unique")

        run_id = str(uuid4())
        now = self._now()
        expires = self._future_expiry(expires_at, now=now) if expires_at is not None else now + timedelta(days=7)
        copied: list[dict[str, Any]] = []
        with self._connection(write=True) as db:
            for materialization_id in materializations:
                parent = self._query(
                    db,
                    """
                    SELECT 1 FROM playlist_materializations
                    WHERE owner_user_id = ? AND materialization_id = ?
                      AND status = 'ready' AND expires_at > ?
                    """,
                    (owner, materialization_id, self._db_datetime(now)),
                ).fetchone()
                if parent is None:
                    raise self._not_found()
                rows = self._query(
                    db,
                    """
                    SELECT * FROM playlist_materialization_items
                    WHERE owner_user_id = ? AND materialization_id = ?
                    ORDER BY ordinal ASC
                    """,
                    (owner, materialization_id),
                ).fetchall()
                copied.extend(self._row_dict(row) for row in rows)

            occurrence_ids = [str(row["occurrence_id"]) for row in copied]
            if len(set(occurrence_ids)) != len(occurrence_ids):
                raise ValueError("materializations contain duplicate occurrence_id values")

            self._query(
                db,
                """
                INSERT INTO media_ingest_runs (
                    run_id, owner_user_id, status, collection_id,
                    processing_options_json, playlist_summaries_json, batch_ids_json,
                    version, created_at, updated_at, expires_at
                ) VALUES (?, ?, 'staged', ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    run_id,
                    owner,
                    collection_id,
                    self._json_value(dict(processing_options)) if processing_options is not None else None,
                    self._json_value([dict(item) for item in playlist_summaries])
                    if playlist_summaries is not None
                    else None,
                    self._json_value([]),
                    self._db_datetime(now),
                    self._db_datetime(now),
                    self._db_datetime(expires),
                ),
            )
            for ordinal, row in enumerate(copied, start=1):
                self._query(
                    db,
                    """
                    INSERT INTO media_ingest_run_items (
                        run_id, owner_user_id, occurrence_id, ordinal, input_kind,
                        materialization_id, source_url, normalized_source_id, source_kind,
                        display_metadata_json, state, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'playlist', ?, ?, ?, ?, ?, 'staged', ?, ?)
                    """,
                    (
                        run_id,
                        owner,
                        row["occurrence_id"],
                        ordinal,
                        row["materialization_id"],
                        row.get("source_url"),
                        row.get("normalized_source_id"),
                        row.get("source_kind"),
                        self._json_value(self._json_dict(row.get("display_metadata_json")) or {}),
                        self._db_datetime(now),
                        self._db_datetime(now),
                    ),
                )
        return self.get_run(owner, run_id)

    def get_run(self, owner_user_id: str, run_id: str) -> MediaIngestRunRecord:
        """Return one owner-scoped ingest run."""
        owner = self._owner(owner_user_id)
        with self._connection(write=False) as db:
            row = self._query(
                db,
                """
                SELECT * FROM media_ingest_runs
                WHERE owner_user_id = ? AND run_id = ? AND expires_at > ?
                """,
                (owner, str(run_id), self._db_datetime(self._now())),
            ).fetchone()
        if row is None:
            raise self._not_found()
        return self._run_record(row)

    def list_run_items(
        self,
        owner_user_id: str,
        run_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> PlaylistPage[MediaIngestRunItemRecord]:
        """List run items in immutable run ordinal order."""
        owner = self._owner(owner_user_id)
        return self._list_ordinal_page(
            owner=owner,
            resource_id=str(run_id),
            kind="run_items",
            parent_table="media_ingest_runs",
            parent_id_column="run_id",
            item_table="media_ingest_run_items",
            limit=limit,
            cursor=cursor,
            converter=self._run_item_record,
        )

    def append_run_event(
        self,
        owner_user_id: str,
        run_id: str,
        *,
        event_type: str,
        occurrence_id: str | None = None,
        job_id: int | None = None,
        batch_id: str | None = None,
        state: str | None = None,
        outcome: str | None = None,
        progress_percent: float | None = None,
        progress_message: str | None = None,
        attrs: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> MediaIngestRunEventRecord:
        """Append an event and bump the run version in the same transaction."""
        owner = self._owner(owner_user_id)
        now = self._now()
        with self._connection(write=True) as db:
            if self._postgres:
                run_version_query = """
                    SELECT version FROM media_ingest_runs
                    WHERE owner_user_id = ? AND run_id = ? AND expires_at > ?
                    FOR UPDATE
                """
            else:
                run_version_query = """
                    SELECT version FROM media_ingest_runs
                    WHERE owner_user_id = ? AND run_id = ? AND expires_at > ?
                """
            row = self._query(
                db,
                run_version_query,
                (owner, str(run_id), self._db_datetime(now)),
            ).fetchone()
            if row is None:
                raise self._not_found()
            current_version = int(self._row_dict(row)["version"])
            if expected_version is not None and current_version != expected_version:
                raise PlaylistIngestConflictError("run version no longer matches expected version")
            updated = self._query(
                db,
                """
                UPDATE media_ingest_runs SET version = version + 1, updated_at = ?
                WHERE owner_user_id = ? AND run_id = ? AND version = ?
                """,
                (self._db_datetime(now), owner, str(run_id), current_version),
            )
            if updated.rowcount != 1:
                raise PlaylistIngestConflictError("run version no longer matches expected version")

            insert_sql = """
                INSERT INTO media_ingest_run_events (
                    run_id, owner_user_id, occurrence_id, job_id, batch_id,
                    event_type, state, outcome, progress_percent, progress_message,
                    attrs_json, occurred_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            if self._postgres:
                insert_sql += " RETURNING event_id"
            inserted = self._query(
                db,
                insert_sql,
                (
                    str(run_id),
                    owner,
                    occurrence_id,
                    job_id,
                    batch_id,
                    str(event_type),
                    state,
                    outcome,
                    progress_percent,
                    progress_message,
                    self._json_value(dict(attrs or {})),
                    self._db_datetime(now),
                ),
            )
            event_id = (
                int(self._row_dict(inserted.fetchone())["event_id"])
                if self._postgres
                else int(inserted.lastrowid)
            )
            event_row = self._query(
                db,
                "SELECT * FROM media_ingest_run_events WHERE event_id = ? AND owner_user_id = ?",
                (event_id, owner),
            ).fetchone()
        return self._event_record(event_row)

    def list_run_events(
        self,
        owner_user_id: str,
        run_id: str,
        *,
        after_event_id: int = 0,
        limit: int = 500,
    ) -> PlaylistPage[MediaIngestRunEventRecord]:
        """Replay owner-scoped run events in stable append order."""
        owner = self._owner(owner_user_id)
        page_limit = self._page_limit(limit)
        with self._connection(write=False) as db:
            exists = self._query(
                db,
                """
                SELECT 1 FROM media_ingest_runs
                WHERE owner_user_id = ? AND run_id = ? AND expires_at > ?
                """,
                (owner, str(run_id), self._db_datetime(self._now())),
            ).fetchone()
            if exists is None:
                raise self._not_found()
            rows = self._query(
                db,
                """
                SELECT * FROM media_ingest_run_events
                WHERE owner_user_id = ? AND run_id = ? AND event_id > ?
                ORDER BY event_id ASC LIMIT ?
                """,
                (owner, str(run_id), int(after_event_id), page_limit),
            ).fetchall()
        return PlaylistPage(tuple(self._event_record(row) for row in rows))

    def compare_and_set_run_item_state(
        self,
        owner_user_id: str,
        run_id: str,
        occurrence_id: str,
        *,
        expected_state: str,
        new_state: str,
        outcome: str | None = None,
        expected_attempt: int = 1,
    ) -> bool:
        """Atomically transition an occurrence only from its expected state and attempt."""
        owner = self._owner(owner_user_id)
        if (new_state == "terminal") != (outcome is not None):
            raise ValueError("outcome is required exactly when new_state is terminal")
        with self._connection(write=True) as db:
            exists = self._query(
                db,
                """
                SELECT 1 FROM media_ingest_runs
                WHERE owner_user_id = ? AND run_id = ? AND expires_at > ?
                """,
                (owner, str(run_id), self._db_datetime(self._now())),
            ).fetchone()
            if exists is None:
                raise self._not_found()
            updated = self._query(
                db,
                """
                UPDATE media_ingest_run_items
                SET state = ?, outcome = ?, updated_at = ?
                WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
                  AND state = ? AND attempt = ?
                """,
                (
                    str(new_state),
                    outcome,
                    self._db_datetime(self._now()),
                    owner,
                    str(run_id),
                    str(occurrence_id),
                    str(expected_state),
                    int(expected_attempt),
                ),
            )
            return updated.rowcount == 1

    def cleanup_expired(
        self,
        owner_user_id: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, int]:
        """Delete only expired playlist resources owned by the caller."""
        owner = self._owner(owner_user_id)
        cutoff = self._db_datetime(now or self._now())
        with self._connection(write=True) as db:
            if self._postgres:
                preflight_rows = self._query(
                    db,
                    """
                    SELECT preflight_id FROM playlist_preflights
                    WHERE owner_user_id = ? AND expires_at <= ?
                    ORDER BY preflight_id FOR UPDATE
                    """,
                    (owner, cutoff),
                ).fetchall()
                materialization_rows = self._query(
                    db,
                    """
                    SELECT materialization_id FROM playlist_materializations
                    WHERE owner_user_id = ? AND expires_at <= ?
                    ORDER BY materialization_id FOR UPDATE
                    """,
                    (owner, cutoff),
                ).fetchall()
                run_rows = self._query(
                    db,
                    """
                    SELECT run_id FROM media_ingest_runs
                    WHERE owner_user_id = ? AND expires_at <= ?
                    ORDER BY run_id FOR UPDATE
                    """,
                    (owner, cutoff),
                ).fetchall()
                preflight_ids = [str(self._row_dict(row)["preflight_id"]) for row in preflight_rows]
                materialization_ids = [
                    str(self._row_dict(row)["materialization_id"])
                    for row in materialization_rows
                ]
                run_ids = [str(self._row_dict(row)["run_id"]) for row in run_rows]

                if run_ids:
                    self._query(
                        db,
                        """
                        DELETE FROM media_ingest_run_events
                        WHERE owner_user_id = ? AND run_id = ANY(?)
                        """,
                        (owner, run_ids),
                    )
                    self._query(
                        db,
                        """
                        DELETE FROM media_ingest_run_items
                        WHERE owner_user_id = ? AND run_id = ANY(?)
                        """,
                        (owner, run_ids),
                    )
                    runs = self._query(
                        db,
                        """
                        DELETE FROM media_ingest_runs
                        WHERE owner_user_id = ? AND run_id = ANY(?)
                        """,
                        (owner, run_ids),
                    ).rowcount
                else:
                    runs = 0

                if materialization_ids:
                    self._query(
                        db,
                        """
                        DELETE FROM playlist_materialization_items
                        WHERE owner_user_id = ? AND materialization_id = ANY(?)
                        """,
                        (owner, materialization_ids),
                    )
                    materializations = self._query(
                        db,
                        """
                        DELETE FROM playlist_materializations
                        WHERE owner_user_id = ? AND materialization_id = ANY(?)
                        """,
                        (owner, materialization_ids),
                    ).rowcount
                else:
                    materializations = 0

                if preflight_ids:
                    self._query(
                        db,
                        """
                        DELETE FROM playlist_preflight_items
                        WHERE owner_user_id = ? AND preflight_id = ANY(?)
                        """,
                        (owner, preflight_ids),
                    )
                    preflights = self._query(
                        db,
                        """
                        DELETE FROM playlist_preflights
                        WHERE owner_user_id = ? AND preflight_id = ANY(?)
                        """,
                        (owner, preflight_ids),
                    ).rowcount
                else:
                    preflights = 0
                return {
                    "preflights": int(preflights),
                    "materializations": int(materializations),
                    "runs": int(runs),
                }

            self._query(
                db,
                """
                DELETE FROM media_ingest_run_events WHERE owner_user_id = ? AND run_id IN (
                    SELECT run_id FROM media_ingest_runs WHERE owner_user_id = ? AND expires_at <= ?
                )
                """,
                (owner, owner, cutoff),
            )
            self._query(
                db,
                """
                DELETE FROM media_ingest_run_items WHERE owner_user_id = ? AND run_id IN (
                    SELECT run_id FROM media_ingest_runs WHERE owner_user_id = ? AND expires_at <= ?
                )
                """,
                (owner, owner, cutoff),
            )
            runs = self._query(
                db,
                "DELETE FROM media_ingest_runs WHERE owner_user_id = ? AND expires_at <= ?",
                (owner, cutoff),
            ).rowcount
            self._query(
                db,
                """
                DELETE FROM playlist_materialization_items
                WHERE owner_user_id = ? AND materialization_id IN (
                    SELECT materialization_id FROM playlist_materializations
                    WHERE owner_user_id = ? AND expires_at <= ?
                )
                """,
                (owner, owner, cutoff),
            )
            materializations = self._query(
                db,
                "DELETE FROM playlist_materializations WHERE owner_user_id = ? AND expires_at <= ?",
                (owner, cutoff),
            ).rowcount
            self._query(
                db,
                """
                DELETE FROM playlist_preflight_items
                WHERE owner_user_id = ? AND preflight_id IN (
                    SELECT preflight_id FROM playlist_preflights
                    WHERE owner_user_id = ? AND expires_at <= ?
                )
                """,
                (owner, owner, cutoff),
            )
            preflights = self._query(
                db,
                "DELETE FROM playlist_preflights WHERE owner_user_id = ? AND expires_at <= ?",
                (owner, cutoff),
            ).rowcount
        return {
            "preflights": int(preflights),
            "materializations": int(materializations),
            "runs": int(runs),
        }


__all__ = [
    "MediaIngestRunEventRecord",
    "MediaIngestRunItemRecord",
    "MediaIngestRunRecord",
    "PlaylistIngestConflictError",
    "PlaylistIngestNotFoundError",
    "PlaylistIngestStore",
    "PlaylistItemRecord",
    "PlaylistMaterializationRecord",
    "PlaylistPage",
    "PlaylistPreflightRecord",
]

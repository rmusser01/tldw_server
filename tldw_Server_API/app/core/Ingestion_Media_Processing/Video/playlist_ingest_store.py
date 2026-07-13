"""Owner-scoped persistence for playlist inspection and ingest manifests."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
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
_MAX_RUN_JSON_BYTES = 65536
_MAX_RUN_JSON_DEPTH = 8
_MAX_RUN_JSON_ITEMS = 2000
_MAX_RUN_IDENTITY_LENGTH = 255
_MAX_RUN_URL_LENGTH = 8192
_PREFLIGHT_JOB_DOMAIN = "media_ingest"
_PREFLIGHT_JOB_TYPE = "playlist_preflight"
_PREFLIGHT_JOB_SENTINEL = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
_PREFLIGHT_ORPHAN_CLAIM_SENTINEL = datetime(9999, 12, 31, 23, 59, 58, tzinfo=timezone.utc)
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


class PlaylistPreflightCapacityError(RuntimeError):
    """Raised when transactional preflight admission has no free slot."""


class PlaylistPreflightLeaseLostError(PlaylistIngestConflictError):
    """Raised before mutation when the claimed preflight job lease is not authoritative."""

    def __init__(self, *, cancelled: bool = False) -> None:
        self.cancelled = bool(cancelled)
        super().__init__("playlist_preflight_lease_lost")


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
class ResolvedMaterializationOccurrence:
    """One owner-authorized immutable occurrence resolved in request order."""

    materialization_id: str
    occurrence_id: str
    source_url: str | None
    normalized_source_id: str | None
    source_kind: str
    display_metadata: dict[str, Any]


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
    planned_collection_item_id: int | None

    @property
    def action(self) -> str:
        """Return the reviewed initial action without requiring a schema column."""
        return self.duplicate_policy or "ingest"


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

    def _job_datetime(self, value: datetime) -> datetime | str:
        """Match JobManager's backend-native schedule serialization."""
        normalized = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        normalized = normalized.astimezone(timezone.utc)
        return normalized if self._postgres else normalized.strftime("%Y-%m-%d %H:%M:%S")

    def _unexpired_sql(self) -> str:
        """Compare timestamps safely across SQLite's supported text formats."""
        return "expires_at > ?" if self._postgres else "julianday(expires_at) > julianday(?)"

    def _expired_sql(self) -> str:
        """Return the backend-safe complement used by cleanup predicates."""
        return "expires_at <= ?" if self._postgres else "julianday(expires_at) <= julianday(?)"

    def _future_expiry(self, value: datetime, *, now: datetime) -> datetime:
        normalized = self._datetime(value)
        if normalized <= now:
            raise ValueError("expires_at must be in the future")
        return normalized

    @staticmethod
    def _bounded_json(
        value: Any,
        *,
        max_bytes: int = _MAX_RUN_JSON_BYTES,
        max_depth: int = _MAX_RUN_JSON_DEPTH,
        max_items: int = _MAX_RUN_JSON_ITEMS,
    ) -> Any:
        """Return a detached JSON value after recursive safety and size checks."""
        item_count = 0

        def visit(candidate: Any, depth: int) -> None:
            nonlocal item_count
            if depth > max_depth:
                raise ValueError("JSON value is too deeply nested")
            if candidate is None or type(candidate) in {bool, str, int}:
                return
            if type(candidate) is float:
                if not math.isfinite(candidate):
                    raise ValueError("JSON numbers must be finite")
                return
            if type(candidate) is list:
                item_count += len(candidate)
                if item_count > max_items:
                    raise ValueError("JSON value contains too many items")
                for entry in candidate:
                    visit(entry, depth + 1)
                return
            if type(candidate) is dict:
                item_count += len(candidate)
                if item_count > max_items:
                    raise ValueError("JSON value contains too many items")
                for key, entry in candidate.items():
                    if type(key) is not str:
                        raise ValueError("JSON object keys must be strings")
                    visit(entry, depth + 1)
                return
            raise ValueError("value must contain only JSON types")

        visit(value, 0)
        try:
            encoded = json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("value must be JSON serializable") from exc
        if len(encoded.encode("utf-8")) > max_bytes:
            raise ValueError("JSON value is too large")
        return json.loads(encoded)

    @staticmethod
    def _run_text(value: Any, name: str, *, max_length: int) -> str:
        if type(value) is not str or not value or value.strip() != value or len(value) > max_length:
            raise ValueError(f"{name} must be a canonical non-empty string")
        return value

    @classmethod
    def _normalize_validated_run_item(cls, item: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and detach one trusted-boundary run item before any write lock."""
        if type(item) is not dict:
            raise ValueError("run items must be plain objects")
        allowed = {
            "occurrence_id",
            "input_kind",
            "materialization_id",
            "source_url",
            "normalized_source_id",
            "source_kind",
            "display_metadata",
            "state",
            "action",
            "metadata_patch",
            "attempt",
            "media_id",
        }
        if set(item) - allowed:
            raise ValueError("run item contains unsupported fields")
        occurrence_id = cls._run_text(
            item.get("occurrence_id"),
            "occurrence_id",
            max_length=_MAX_RUN_IDENTITY_LENGTH,
        )
        input_kind = item.get("input_kind")
        expected_state = {
            "materialized_playlist_item": "staged",
            "direct_url": "staged",
            "file_stub": "awaiting_upload",
        }.get(input_kind)
        if expected_state is None:
            raise ValueError("invalid run item input_kind")
        if item.get("state") != expected_state:
            raise ValueError("invalid initial run item state")
        action = item.get("action")
        if action not in {"ingest", "overwrite", "skip", "include_existing", "update_metadata_only"}:
            raise ValueError("invalid initial run item action")
        attempt = item.get("attempt", 1)
        if type(attempt) is not int or attempt != 1:
            raise ValueError("initial run item attempt must be the integer 1")

        materialization_id = item.get("materialization_id")
        source_url = item.get("source_url")
        normalized_source_id = item.get("normalized_source_id")
        source_kind = item.get("source_kind")
        if input_kind == "file_stub":
            if materialization_id is not None or source_url is not None or normalized_source_id is not None:
                raise ValueError("file stubs cannot contain materialized source identity")
            if source_kind != "file":
                raise ValueError("file stubs require file source_kind")
        else:
            source_url = cls._run_text(source_url, "source_url", max_length=_MAX_RUN_URL_LENGTH)
            normalized_source_id = cls._run_text(
                normalized_source_id,
                "normalized_source_id",
                max_length=_MAX_RUN_URL_LENGTH,
            )
            source_kind = cls._run_text(source_kind, "source_kind", max_length=64)
            if input_kind == "materialized_playlist_item":
                materialization_id = cls._run_text(
                    materialization_id,
                    "materialization_id",
                    max_length=_MAX_RUN_IDENTITY_LENGTH,
                )
            elif materialization_id is not None:
                raise ValueError("direct URLs cannot contain materialization_id")

        display = item.get("display_metadata", {})
        if type(display) is not dict:
            raise ValueError("display_metadata must be an object")
        display = cls._bounded_json(display)
        metadata_patch = item.get("metadata_patch")
        if metadata_patch is not None:
            if type(metadata_patch) is not dict:
                raise ValueError("metadata_patch must be an object")
            metadata_patch = cls._bounded_json(metadata_patch, max_bytes=8192, max_depth=2, max_items=110)
        if action in {"ingest", "skip", "include_existing"} and metadata_patch is not None:
            raise ValueError("initial run item action does not allow metadata_patch")
        if action == "update_metadata_only" and metadata_patch is None:
            raise ValueError("update_metadata_only requires metadata_patch")
        media_id = item.get("media_id")
        if media_id is not None and (
            type(media_id) is not int
            or media_id < 1
            or action not in {"skip", "include_existing", "update_metadata_only"}
        ):
            raise ValueError("media_id is valid only for a non-processing duplicate action")

        return {
            "occurrence_id": occurrence_id,
            "input_kind": input_kind,
            "materialization_id": materialization_id,
            "source_url": source_url,
            "normalized_source_id": normalized_source_id,
            "source_kind": source_kind,
            "display_metadata": display,
            "state": expected_state,
            "action": action,
            "metadata_patch": metadata_patch,
            "attempt": 1,
            "media_id": media_id,
        }

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
                bool(data["selected_by_default"]) if data.get("selected_by_default") is not None else None
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
            progress_percent=(float(data["progress_percent"]) if data.get("progress_percent") is not None else None),
            progress_message=data.get("progress_message"),
            retryable=bool(data["retryable"]),
            media_id=int(data["media_id"]) if data.get("media_id") is not None else None,
            planned_collection_item_id=(
                int(data["planned_collection_item_id"]) if data.get("planned_collection_item_id") is not None else None
            ),
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
            progress_percent=(float(data["progress_percent"]) if data.get("progress_percent") is not None else None),
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

    def reserve_preflight(
        self,
        owner_user_id: str,
        *,
        source_url: str,
        source_kind: str,
        ttl_seconds: int,
        global_capacity: int,
        owner_capacity: int,
        playlist_id: str | None = None,
    ) -> PlaylistPreflightRecord:
        """Atomically reserve bounded global and owner preflight capacity."""
        owner = self._owner(owner_user_id)
        if type(global_capacity) is not int or global_capacity < 1:
            raise ValueError("global_capacity must be positive")
        if type(owner_capacity) is not int or owner_capacity < 1:
            raise ValueError("owner_capacity must be positive")
        if type(ttl_seconds) is not int or ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")
        preflight_id = str(uuid4())
        with self._connection(write=True) as db:
            if self._postgres:
                self._query(
                    db,
                    "SELECT pg_advisory_xact_lock(?)",
                    (self._jobs._pg_advisory_key("playlist_preflight_admission"),),
                )
            global_row = self._query(
                db,
                (
                    """
                    SELECT COUNT(*) AS active_count FROM playlist_preflights
                    WHERE status IN ('pending', 'running') AND expires_at > NOW()
                    """
                    if self._postgres
                    else """
                    SELECT COUNT(*) AS active_count FROM playlist_preflights
                    WHERE status IN ('pending', 'running')
                      AND julianday(expires_at) > julianday('now')
                    """
                ),
            ).fetchone()
            owner_row = self._query(
                db,
                (
                    """
                    SELECT COUNT(*) AS active_count FROM playlist_preflights
                    WHERE owner_user_id = ? AND status IN ('pending', 'running')
                      AND expires_at > NOW()
                    """
                    if self._postgres
                    else """
                    SELECT COUNT(*) AS active_count FROM playlist_preflights
                    WHERE owner_user_id = ? AND status IN ('pending', 'running')
                      AND julianday(expires_at) > julianday('now')
                    """
                ),
                (owner,),
            ).fetchone()
            if (
                int(self._row_dict(global_row)["active_count"]) >= global_capacity
                or int(self._row_dict(owner_row)["active_count"]) >= owner_capacity
            ):
                raise PlaylistPreflightCapacityError("preflight_busy")
            if self._postgres:
                self._query(
                    db,
                    """
                    INSERT INTO playlist_preflights (
                        preflight_id, owner_user_id, status, source_url, source_kind,
                        playlist_id, job_id, created_at, updated_at, expires_at
                    ) VALUES (?, ?, 'pending', ?, ?, ?, NULL, NOW(), NOW(),
                              NOW() + (? * INTERVAL '1 second'))
                    """,
                    (preflight_id, owner, str(source_url), str(source_kind), playlist_id, ttl_seconds),
                )
            else:
                self._query(
                    db,
                    """
                    INSERT INTO playlist_preflights (
                        preflight_id, owner_user_id, status, source_url, source_kind,
                        playlist_id, job_id, created_at, updated_at, expires_at
                    ) VALUES (?, ?, 'pending', ?, ?, ?, NULL, DATETIME('now'), DATETIME('now'),
                              DATETIME('now', ?))
                    """,
                    (
                        preflight_id,
                        owner,
                        str(source_url),
                        str(source_kind),
                        playlist_id,
                        f"+{ttl_seconds} seconds",
                    ),
                )
        return self.get_preflight(owner, preflight_id)

    def bind_preflight_job(
        self,
        owner_user_id: str,
        preflight_id: str,
        job_id: int,
        *,
        expected_queue: str,
        expected_payload: Mapping[str, Any],
    ) -> PlaylistPreflightRecord:
        """Bind and publish one scheduled internal job in the same transaction."""
        owner = self._owner(owner_user_id)
        if type(job_id) is not int or job_id < 1:
            raise ValueError("job_id must be positive")
        queue = str(expected_queue or "").strip()
        if not queue:
            raise ValueError("expected_queue is required")
        expected_payload_json = json.dumps(dict(expected_payload), sort_keys=True, separators=(",", ":"))
        with self._connection(write=True) as db:
            preflight_query = (
                """
                SELECT status, job_id FROM playlist_preflights
                WHERE owner_user_id = ? AND preflight_id = ? AND expires_at > NOW()
                FOR UPDATE
                """
                if self._postgres
                else """
                SELECT status, job_id FROM playlist_preflights
                WHERE owner_user_id = ? AND preflight_id = ?
                  AND julianday(expires_at) > julianday('now')
                """
            )
            preflight = self._query(
                db,
                preflight_query,
                (owner, str(preflight_id)),
            ).fetchone()
            if preflight is None:
                raise self._not_found()
            preflight_data = self._row_dict(preflight)
            if str(preflight_data["status"]) != "pending" or preflight_data.get("job_id") is not None:
                raise PlaylistIngestConflictError("preflight job is already bound")

            job_query = (
                """
                SELECT id, owner_user_id, domain, queue, job_type, status, available_at, payload
                FROM jobs WHERE id = ? FOR UPDATE
                """
                if self._postgres
                else """
                SELECT id, owner_user_id, domain, queue, job_type, status, available_at, payload
                FROM jobs WHERE id = ?
                """
            )
            job = self._query(db, job_query, (job_id,)).fetchone()
            if job is None:
                raise PlaylistIngestConflictError("preflight job is unavailable")
            job_data = self._row_dict(job)
            raw_payload = self._jobs._parse_json_value(job_data.get("payload"))
            actual_payload = self._jobs._maybe_decrypt_json(raw_payload)
            actual_payload_json = (
                json.dumps(dict(actual_payload), sort_keys=True, separators=(",", ":"))
                if isinstance(actual_payload, Mapping)
                else ""
            )
            if (
                str(job_data.get("owner_user_id") or "") != owner
                or str(job_data.get("domain") or "") != _PREFLIGHT_JOB_DOMAIN
                or str(job_data.get("queue") or "") != queue
                or str(job_data.get("job_type") or "") != _PREFLIGHT_JOB_TYPE
                or str(job_data.get("status") or "") != "queued"
                or self._datetime(job_data["available_at"]) != _PREFLIGHT_JOB_SENTINEL
                or actual_payload_json != expected_payload_json
            ):
                raise PlaylistIngestConflictError("preflight job is unavailable")

            bound = self._query(
                db,
                (
                    """
                    UPDATE playlist_preflights SET job_id = ?, updated_at = NOW()
                    WHERE owner_user_id = ? AND preflight_id = ? AND job_id IS NULL
                    """
                    if self._postgres
                    else """
                    UPDATE playlist_preflights SET job_id = ?, updated_at = DATETIME('now')
                    WHERE owner_user_id = ? AND preflight_id = ? AND job_id IS NULL
                    """
                ),
                (job_id, owner, str(preflight_id)),
            )
            if bound.rowcount != 1:
                raise PlaylistIngestConflictError("preflight job is unavailable")
            published = self._query(
                db,
                (
                    """
                    UPDATE jobs SET available_at = NOW(), updated_at = NOW()
                    WHERE id = ? AND owner_user_id = ? AND domain = ? AND queue = ?
                      AND job_type = ? AND status = 'queued' AND available_at = ?
                    """
                    if self._postgres
                    else """
                    UPDATE jobs SET available_at = DATETIME('now'), updated_at = DATETIME('now')
                    WHERE id = ? AND owner_user_id = ? AND domain = ? AND queue = ?
                      AND job_type = ? AND status = 'queued' AND available_at = ?
                    """
                ),
                (
                    job_id,
                    owner,
                    _PREFLIGHT_JOB_DOMAIN,
                    queue,
                    _PREFLIGHT_JOB_TYPE,
                    self._job_datetime(_PREFLIGHT_JOB_SENTINEL),
                ),
            )
            if published.rowcount != 1:
                raise PlaylistIngestConflictError("preflight job is unavailable")
            if self._jobs._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")):
                if self._postgres:
                    self._query(
                        db,
                        """
                        INSERT INTO job_counters (
                            domain, queue, job_type, ready_count, scheduled_count,
                            processing_count, quarantined_count
                        ) VALUES (?, ?, ?, 1, 0, 0, 0)
                        ON CONFLICT (domain, queue, job_type) DO UPDATE
                        SET ready_count = job_counters.ready_count + 1,
                            scheduled_count = GREATEST(job_counters.scheduled_count - 1, 0),
                            updated_at = NOW()
                        """,
                        (_PREFLIGHT_JOB_DOMAIN, queue, _PREFLIGHT_JOB_TYPE),
                    )
                else:
                    self._query(
                        db,
                        """
                        INSERT INTO job_counters (
                            domain, queue, job_type, ready_count, scheduled_count,
                            processing_count, quarantined_count
                        ) VALUES (?, ?, ?, 1, 0, 0, 0)
                        ON CONFLICT (domain, queue, job_type) DO UPDATE
                        SET ready_count = ready_count + 1,
                            scheduled_count = CASE
                                WHEN scheduled_count > 0 THEN scheduled_count - 1 ELSE 0 END,
                            updated_at = DATETIME('now')
                        """,
                        (_PREFLIGHT_JOB_DOMAIN, queue, _PREFLIGHT_JOB_TYPE),
                    )
        return self.get_preflight(owner, preflight_id)

    def list_orphaned_preflight_jobs(
        self,
        owner_user_id: str,
        *,
        queue: str,
        grace_seconds: int,
        limit: int,
    ) -> list[tuple[int, str | None]]:
        """Return a bounded owner-scoped set of old never-published jobs."""
        owner = self._owner(owner_user_id)
        if type(grace_seconds) is not int or grace_seconds < 1:
            raise ValueError("grace_seconds must be positive")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        sentinel = self._job_datetime(_PREFLIGHT_JOB_SENTINEL)
        claimed = self._job_datetime(_PREFLIGHT_ORPHAN_CLAIM_SENTINEL)
        with self._connection(write=False) as db:
            rows = self._query(
                db,
                (
                    """
                    SELECT j.id, j.payload FROM jobs j
                    LEFT JOIN playlist_preflights p ON p.job_id = j.id
                    WHERE j.owner_user_id = ? AND j.domain = ? AND j.queue = ?
                      AND j.job_type = ? AND j.status = 'queued'
                      AND j.available_at IN (?, ?) AND p.preflight_id IS NULL
                      AND j.created_at <= NOW() - (? * INTERVAL '1 second')
                    ORDER BY j.id ASC LIMIT ?
                    """
                    if self._postgres
                    else """
                    SELECT j.id, j.payload FROM jobs j
                    LEFT JOIN playlist_preflights p ON p.job_id = j.id
                    WHERE j.owner_user_id = ? AND j.domain = ? AND j.queue = ?
                      AND j.job_type = ? AND j.status = 'queued'
                      AND j.available_at IN (?, ?) AND p.preflight_id IS NULL
                      AND julianday(j.created_at) <= julianday('now', ?)
                    ORDER BY j.id ASC LIMIT ?
                    """
                ),
                (
                    owner,
                    _PREFLIGHT_JOB_DOMAIN,
                    str(queue),
                    _PREFLIGHT_JOB_TYPE,
                    sentinel,
                    claimed,
                    grace_seconds if self._postgres else f"-{grace_seconds} seconds",
                    limit,
                ),
            ).fetchall()
        candidates: list[tuple[int, str | None]] = []
        for row in rows:
            data = self._row_dict(row)
            payload = self._jobs._maybe_decrypt_json(self._jobs._parse_json_value(data.get("payload")))
            preflight_id = payload.get("preflight_id") if isinstance(payload, Mapping) else None
            candidates.append((int(data["id"]), str(preflight_id) if preflight_id else None))
        return candidates

    def claim_orphaned_preflight_job(
        self,
        owner_user_id: str,
        *,
        preflight_id: str | None,
        job_id: int,
        queue: str,
        grace_seconds: int,
    ) -> bool:
        """Fence one old unbound job before cancellation through JobManager."""
        owner = self._owner(owner_user_id)
        sentinel = self._job_datetime(_PREFLIGHT_JOB_SENTINEL)
        claimed = self._job_datetime(_PREFLIGHT_ORPHAN_CLAIM_SENTINEL)
        with self._connection(write=True) as db:
            preflight_data: dict[str, Any] | None = None
            if preflight_id:
                preflight = self._query(
                    db,
                    (
                        """
                        SELECT status, job_id FROM playlist_preflights
                        WHERE owner_user_id = ? AND preflight_id = ? FOR UPDATE
                        """
                        if self._postgres
                        else """
                        SELECT status, job_id FROM playlist_preflights
                        WHERE owner_user_id = ? AND preflight_id = ?
                        """
                    ),
                    (owner, preflight_id),
                ).fetchone()
                preflight_data = self._row_dict(preflight) if preflight is not None else None
                if preflight_data is not None and preflight_data.get("job_id") is not None:
                    return False

            job = self._query(
                db,
                (
                    """
                    SELECT id, payload FROM jobs
                    WHERE id = ? AND owner_user_id = ? AND domain = ? AND queue = ?
                      AND job_type = ? AND status = 'queued' AND available_at IN (?, ?)
                      AND created_at <= NOW() - (? * INTERVAL '1 second')
                    FOR UPDATE
                    """
                    if self._postgres
                    else """
                    SELECT id, payload FROM jobs
                    WHERE id = ? AND owner_user_id = ? AND domain = ? AND queue = ?
                      AND job_type = ? AND status = 'queued' AND available_at IN (?, ?)
                      AND julianday(created_at) <= julianday('now', ?)
                    """
                ),
                (
                    job_id,
                    owner,
                    _PREFLIGHT_JOB_DOMAIN,
                    str(queue),
                    _PREFLIGHT_JOB_TYPE,
                    sentinel,
                    claimed,
                    grace_seconds if self._postgres else f"-{grace_seconds} seconds",
                ),
            ).fetchone()
            if job is None:
                return False
            payload = self._jobs._maybe_decrypt_json(self._jobs._parse_json_value(self._row_dict(job).get("payload")))
            if preflight_id and (not isinstance(payload, Mapping) or payload.get("preflight_id") != preflight_id):
                return False

            fenced = self._query(
                db,
                """
                UPDATE jobs SET available_at = ?
                WHERE id = ? AND owner_user_id = ? AND domain = ? AND queue = ?
                  AND job_type = ? AND status = 'queued' AND available_at IN (?, ?)
                """,
                (
                    claimed,
                    job_id,
                    owner,
                    _PREFLIGHT_JOB_DOMAIN,
                    str(queue),
                    _PREFLIGHT_JOB_TYPE,
                    sentinel,
                    claimed,
                ),
            )
            if fenced.rowcount != 1:
                return False
            if preflight_id and preflight_data is not None and preflight_data.get("job_id") is None:
                self._query(
                    db,
                    (
                        """
                        UPDATE playlist_preflights
                        SET status = 'blocked', updated_at = NOW(), expires_at = NOW()
                        WHERE owner_user_id = ? AND preflight_id = ? AND job_id IS NULL
                          AND status = 'pending'
                        """
                        if self._postgres
                        else """
                        UPDATE playlist_preflights
                        SET status = 'blocked', updated_at = DATETIME('now'), expires_at = DATETIME('now')
                        WHERE owner_user_id = ? AND preflight_id = ? AND job_id IS NULL
                          AND status = 'pending'
                        """
                    ),
                    (owner, preflight_id),
                )
        return True

    def expire_preflight(
        self,
        owner_user_id: str,
        preflight_id: str,
        *,
        status: str = "cancelled",
    ) -> int | None:
        """Expire one owner resource and return its linked job for cancellation."""
        owner = self._owner(owner_user_id)
        if status not in {"blocked", "cancelled", "expired"}:
            raise ValueError("invalid terminal preflight status")
        now = self._now()
        with self._connection(write=True) as db:
            query = (
                """
                SELECT job_id FROM playlist_preflights
                WHERE owner_user_id = ? AND preflight_id = ? FOR UPDATE
                """
                if self._postgres
                else """
                SELECT job_id FROM playlist_preflights
                WHERE owner_user_id = ? AND preflight_id = ?
                """
            )
            row = self._query(db, query, (owner, str(preflight_id))).fetchone()
            if row is None:
                raise self._not_found()
            self._query(
                db,
                """
                UPDATE playlist_preflights
                SET status = ?, updated_at = ?, expires_at = ?
                WHERE owner_user_id = ? AND preflight_id = ?
                """,
                (
                    status,
                    self._db_datetime(now),
                    self._db_datetime(now),
                    owner,
                    str(preflight_id),
                ),
            )
            job_id = self._row_dict(row).get("job_id")
        return int(job_id) if job_id is not None else None

    def get_preflight(self, owner_user_id: str, preflight_id: str) -> PlaylistPreflightRecord:
        """Return one owner-scoped preflight or the generic not-found error."""
        owner = self._owner(owner_user_id)
        with self._connection(write=False) as db:
            row = self._query(
                db,
                (
                    """
                    SELECT * FROM playlist_preflights
                    WHERE owner_user_id = ? AND preflight_id = ? AND expires_at > ?
                    """
                    if self._postgres
                    else """
                    SELECT * FROM playlist_preflights
                    WHERE owner_user_id = ? AND preflight_id = ?
                      AND julianday(expires_at) > julianday(?)
                    """
                ),
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
        expected_job_id: int | None = None,
        expected_lease_id: str | None = None,
        expected_worker_id: str | None = None,
    ) -> PlaylistPreflightRecord:
        """Replace a mutable snapshot, optionally guarded by its active processing job."""
        owner = self._owner(owner_user_id)
        guarded = any(value is not None for value in (expected_job_id, expected_lease_id, expected_worker_id))
        if guarded and (
            type(expected_job_id) is not int
            or not isinstance(expected_lease_id, str)
            or not expected_lease_id.strip()
            or not isinstance(expected_worker_id, str)
            or not expected_worker_id.strip()
        ):
            raise PlaylistPreflightLeaseLostError()
        occurrence_ids = [str(item.get("occurrence_id") or "").strip() for item in items]
        ordinals = [item.get("ordinal") for item in items]
        if any(not value for value in occurrence_ids) or len(set(occurrence_ids)) != len(occurrence_ids):
            raise ValueError("occurrence_id values must be non-empty and unique")
        if any(type(value) is not int or value < 1 for value in ordinals) or len(set(ordinals)) != len(ordinals):
            raise ValueError("ordinal values must be positive and unique")

        now = self._now()
        with self._connection(write=True) as db:
            mutable_preflight_query = f"""
                    SELECT status, expires_at, job_id FROM playlist_preflights
                    WHERE owner_user_id = ? AND preflight_id = ? AND {self._unexpired_sql()}
                """ + (  # nosec B608
                " FOR UPDATE" if self._postgres else ""
            )
            row = self._query(
                db,
                mutable_preflight_query,
                (owner, str(preflight_id), self._db_datetime(now)),
            ).fetchone()
            if row is None:
                raise self._not_found()
            preflight_row = self._row_dict(row)
            if str(preflight_row["status"]) not in {"pending", "running"}:
                raise PlaylistIngestConflictError("ready playlist snapshot ordering is immutable")
            if guarded:
                if preflight_row.get("job_id") != int(expected_job_id):
                    raise PlaylistPreflightLeaseLostError()
                active_job_query = (
                    """
                    SELECT owner_user_id, status, lease_id, worker_id,
                           (leased_until IS NOT NULL AND leased_until > NOW()) AS lease_active
                    FROM jobs WHERE id = ?
                    FOR UPDATE
                """
                    if self._postgres
                    else """
                    SELECT owner_user_id, status, lease_id, worker_id,
                           (leased_until IS NOT NULL AND leased_until > DATETIME('now')) AS lease_active
                    FROM jobs WHERE id = ?
                """
                )
                job_row = self._query(db, active_job_query, (int(expected_job_id),)).fetchone()
                if job_row is None:
                    raise PlaylistPreflightLeaseLostError()
                active_job = self._row_dict(job_row)
                cancelled = (
                    str(active_job.get("owner_user_id") or "") == owner
                    and str(active_job.get("status") or "") == "cancelled"
                )
                if (
                    str(active_job.get("owner_user_id") or "") != owner
                    or str(active_job.get("status") or "") != "processing"
                    or str(active_job.get("lease_id") or "") != expected_lease_id
                    or str(active_job.get("worker_id") or "") != expected_worker_id
                    or not bool(active_job.get("lease_active"))
                ):
                    raise PlaylistPreflightLeaseLostError(cancelled=cancelled)

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
        last_ordinal = (
            self._decode_cursor(
                cursor,
                owner=owner,
                kind=kind,
                resource_id=resource_id,
            )
            if cursor
            else 0
        )
        with self._connection(write=False) as db:
            exists = self._query(
                db,
                f"""
                SELECT 1 FROM {parent_table}
                WHERE owner_user_id = ? AND {parent_id_column} = ? AND {self._unexpired_sql()}
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
                f"""
                SELECT expires_at FROM playlist_preflights
                WHERE owner_user_id = ? AND preflight_id = ? AND status = 'ready'
                  AND {self._unexpired_sql()}
                """,  # nosec B608
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
                    key: source_display[key] for key in _COMPACT_DISPLAY_METADATA_FIELDS if key in source_display
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
                f"""
                SELECT * FROM playlist_materializations
                WHERE owner_user_id = ? AND materialization_id = ? AND {self._unexpired_sql()}
                """,  # nosec B608
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

    @classmethod
    def _normalize_materialization_pairs(
        cls,
        pairs: Sequence[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        if not 1 <= len(pairs) <= _MAX_PAGE_SIZE:
            raise ValueError("materialization occurrence pairs must contain between 1 and 500 entries")
        normalized: list[tuple[str, str]] = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("materialization occurrence pairs must contain two strings")
            materialization_id = cls._run_text(
                pair[0],
                "materialization_id",
                max_length=_MAX_RUN_IDENTITY_LENGTH,
            )
            occurrence_id = cls._run_text(
                pair[1],
                "occurrence_id",
                max_length=_MAX_RUN_IDENTITY_LENGTH,
            )
            normalized.append((materialization_id, occurrence_id))
        if len(set(normalized)) != len(normalized):
            raise ValueError("materialization occurrence pairs must be unique")
        return normalized

    def _resolve_materialization_occurrences_in_connection(
        self,
        db: Any,
        owner: str,
        pairs: Sequence[tuple[str, str]],
        *,
        now: datetime,
        lock: bool,
    ) -> list[ResolvedMaterializationOccurrence]:
        """Resolve an ordered authority set with one fixed-shape collection bind."""
        request_json = json.dumps(
            [
                {"materialization_id": materialization_id, "occurrence_id": occurrence_id}
                for materialization_id, occurrence_id in pairs
            ],
            separators=(",", ":"),
        )
        if self._postgres:
            requested_sql = """
                SELECT materialization_id, occurrence_id, request_ordinal
                FROM ROWS FROM (
                    jsonb_to_recordset(?::jsonb) AS (
                        materialization_id text,
                        occurrence_id text
                    )
                ) WITH ORDINALITY AS requested(
                    materialization_id,
                    occurrence_id,
                    request_ordinal
                )
            """
            lock_sql = "FOR SHARE OF m, mi" if lock else ""
        else:
            requested_sql = """
                SELECT
                    json_extract(value, '$.materialization_id') AS materialization_id,
                    json_extract(value, '$.occurrence_id') AS occurrence_id,
                    CAST(key AS INTEGER) + 1 AS request_ordinal
                FROM json_each(?)
            """
            lock_sql = ""
        rows = self._query(
            db,
            f"""
            WITH requested AS ({requested_sql})
            SELECT
                r.request_ordinal,
                r.materialization_id,
                r.occurrence_id,
                mi.source_url,
                mi.normalized_source_id,
                mi.source_kind,
                mi.display_metadata_json
            FROM requested AS r
            JOIN playlist_materializations AS m
              ON m.materialization_id = r.materialization_id
             AND m.owner_user_id = ?
             AND m.status = 'ready'
            JOIN playlist_materialization_items AS mi
              ON mi.materialization_id = m.materialization_id
             AND mi.owner_user_id = m.owner_user_id
             AND mi.occurrence_id = r.occurrence_id
            WHERE {('m.expires_at > ?' if self._postgres else 'julianday(m.expires_at) > julianday(?)')}
            ORDER BY r.request_ordinal
            {lock_sql}
            """,  # nosec B608 - backend-specific fixed SQL fragments only
            (request_json, owner, self._db_datetime(now)),
        ).fetchall()
        if len(rows) != len(pairs):
            raise self._not_found()
        resolved: list[ResolvedMaterializationOccurrence] = []
        for request_ordinal, (row, expected_pair) in enumerate(zip(rows, pairs, strict=True), start=1):
            data = self._row_dict(row)
            actual_pair = (str(data["materialization_id"]), str(data["occurrence_id"]))
            if int(data["request_ordinal"]) != request_ordinal or actual_pair != expected_pair:
                raise self._not_found()
            resolved.append(
                ResolvedMaterializationOccurrence(
                    materialization_id=actual_pair[0],
                    occurrence_id=actual_pair[1],
                    source_url=data.get("source_url"),
                    normalized_source_id=data.get("normalized_source_id"),
                    source_kind=str(data["source_kind"]),
                    display_metadata=self._json_dict(data.get("display_metadata_json")) or {},
                )
            )
        return resolved

    def resolve_materialization_occurrences(
        self,
        owner_user_id: str,
        pairs: Sequence[tuple[str, str]],
    ) -> list[ResolvedMaterializationOccurrence]:
        """Resolve up to 500 owner-authorized occurrences with one bulk query."""
        owner = self._owner(owner_user_id)
        normalized = self._normalize_materialization_pairs(pairs)
        with self._connection(write=False) as db:
            return self._resolve_materialization_occurrences_in_connection(
                db,
                owner,
                normalized,
                now=self._now(),
                lock=False,
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
                    f"""
                    SELECT 1 FROM playlist_materializations
                    WHERE owner_user_id = ? AND materialization_id = ?
                      AND status = 'ready' AND {self._unexpired_sql()}
                    """,  # nosec B608
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
                    (
                        self._json_value([dict(item) for item in playlist_summaries])
                        if playlist_summaries is not None
                        else None
                    ),
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

    def create_validated_run(
        self,
        owner_user_id: str,
        *,
        items: Sequence[Mapping[str, Any]],
        processing_options: Mapping[str, Any] | None = None,
        playlist_summaries: Sequence[Mapping[str, Any]] | None = None,
        collection_id: int | None = None,
        expires_at: datetime | None = None,
    ) -> MediaIngestRunRecord:
        """Persist one fully validated mixed manifest and its initial events atomically."""
        owner = self._owner(owner_user_id)
        if not 1 <= len(items) <= _MAX_PAGE_SIZE:
            raise ValueError("items must contain between 1 and 500 occurrences")
        records = [self._normalize_validated_run_item(item) for item in items]
        occurrence_ids = [item["occurrence_id"] for item in records]
        if len(set(occurrence_ids)) != len(occurrence_ids):
            raise ValueError("occurrence_id values must be unique")
        if collection_id is not None and (type(collection_id) is not int or collection_id < 1):
            raise ValueError("collection_id must be a positive integer")
        if processing_options is None:
            normalized_options = None
        elif type(processing_options) is dict:
            normalized_options = self._bounded_json(processing_options)
        else:
            raise ValueError("processing_options must be an object")
        if playlist_summaries is None:
            normalized_summaries = None
        elif type(playlist_summaries) is list and all(type(summary) is dict for summary in playlist_summaries):
            normalized_summaries = self._bounded_json(playlist_summaries)
        else:
            raise ValueError("playlist_summaries must be a list of objects")

        run_id = str(uuid4())
        now = self._now()
        expires = self._future_expiry(expires_at, now=now) if expires_at is not None else now + timedelta(days=7)
        materialized_records = [item for item in records if item["input_kind"] == "materialized_playlist_item"]
        materialized_pairs = [
            (str(item["materialization_id"]), str(item["occurrence_id"])) for item in materialized_records
        ]
        with self._connection(write=True) as db:
            if materialized_pairs:
                authoritative = self._resolve_materialization_occurrences_in_connection(
                    db,
                    owner,
                    materialized_pairs,
                    now=now,
                    lock=True,
                )
                for item, current in zip(materialized_records, authoritative, strict=True):
                    if (
                        item["source_url"] != current.source_url
                        or item["normalized_source_id"] != current.normalized_source_id
                        or item["source_kind"] != current.source_kind
                        or item["display_metadata"] != current.display_metadata
                    ):
                        raise self._not_found()
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
                    self._json_value(normalized_options) if normalized_options is not None else None,
                    (self._json_value(normalized_summaries) if normalized_summaries is not None else None),
                    self._json_value([]),
                    self._db_datetime(now),
                    self._db_datetime(now),
                    self._db_datetime(expires),
                ),
            )
            for ordinal, item in enumerate(records, start=1):
                action = str(item["action"])
                duplicate_policy = None if action == "ingest" else action
                self._query(
                    db,
                    """
                    INSERT INTO media_ingest_run_items (
                        run_id, owner_user_id, occurrence_id, ordinal, input_kind,
                        materialization_id, source_url, normalized_source_id, source_kind,
                        display_metadata_json, duplicate_policy, metadata_patch_json,
                        state, attempt, media_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                    """,
                    (
                        run_id,
                        owner,
                        item["occurrence_id"],
                        ordinal,
                        item["input_kind"],
                        item.get("materialization_id"),
                        item.get("source_url"),
                        item.get("normalized_source_id"),
                        item.get("source_kind"),
                        self._json_value(dict(item.get("display_metadata") or {})),
                        duplicate_policy,
                        (
                            self._json_value(dict(item["metadata_patch"]))
                            if item.get("metadata_patch") is not None
                            else None
                        ),
                        item["state"],
                        item.get("media_id"),
                        self._db_datetime(now),
                        self._db_datetime(now),
                    ),
                )
                self._query(
                    db,
                    """
                    INSERT INTO media_ingest_run_events (
                        run_id, owner_user_id, occurrence_id, event_type, state,
                        attrs_json, occurred_at
                    ) VALUES (?, ?, ?, 'manifest_item_staged', ?, ?, ?)
                    """,
                    (
                        run_id,
                        owner,
                        item["occurrence_id"],
                        item["state"],
                        self._json_value({"action": action, "input_kind": item["input_kind"]}),
                        self._db_datetime(now),
                    ),
                )
            self._query(
                db,
                """
                UPDATE media_ingest_runs SET version = version + 1, updated_at = ?
                WHERE owner_user_id = ? AND run_id = ? AND version = 1
                """,
                (self._db_datetime(now), owner, run_id),
            )
        return self.get_run(owner, run_id)

    def get_run(self, owner_user_id: str, run_id: str) -> MediaIngestRunRecord:
        """Return one owner-scoped ingest run."""
        owner = self._owner(owner_user_id)
        with self._connection(write=False) as db:
            row = self._query(
                db,
                f"""
                SELECT * FROM media_ingest_runs
                WHERE owner_user_id = ? AND run_id = ? AND {self._unexpired_sql()}
                """,  # nosec B608
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

    def get_run_item(
        self,
        owner_user_id: str,
        run_id: str,
        occurrence_id: str,
    ) -> MediaIngestRunItemRecord:
        """Return one owner-scoped run occurrence by its stable identity."""
        owner = self._owner(owner_user_id)
        expiry_sql = "run.expires_at > ?" if self._postgres else "julianday(run.expires_at) > julianday(?)"
        with self._connection(write=False) as db:
            row = self._query(
                db,
                f"""
                SELECT item.*
                FROM media_ingest_run_items AS item
                JOIN media_ingest_runs AS run
                  ON run.run_id = item.run_id AND run.owner_user_id = item.owner_user_id
                WHERE item.owner_user_id = ? AND item.run_id = ? AND item.occurrence_id = ?
                  AND {expiry_sql}
                """,  # nosec B608
                (
                    owner,
                    str(run_id),
                    str(occurrence_id),
                    self._db_datetime(self._now()),
                ),
            ).fetchone()
        if row is None:
            raise self._not_found()
        return self._run_item_record(row)

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
            run_version_query = f"""
                    SELECT version FROM media_ingest_runs
                    WHERE owner_user_id = ? AND run_id = ? AND {self._unexpired_sql()}
                """ + (  # nosec B608
                " FOR UPDATE" if self._postgres else ""
            )
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
                int(self._row_dict(inserted.fetchone())["event_id"]) if self._postgres else int(inserted.lastrowid)
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
                f"""
                SELECT 1 FROM media_ingest_runs
                WHERE owner_user_id = ? AND run_id = ? AND {self._unexpired_sql()}
                """,  # nosec B608
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
                f"""
                SELECT 1 FROM media_ingest_runs
                WHERE owner_user_id = ? AND run_id = ? AND {self._unexpired_sql()}
                """,  # nosec B608
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

    def prepare_run_item_job_submission(
        self,
        owner_user_id: str,
        run_id: str,
        occurrence_id: str,
        *,
        attempt: int,
        batch_id: str,
        idempotency_identity: str,
        source_kind: str,
        planned_item_id: int | None,
    ) -> MediaIngestRunItemRecord:
        """Reserve one processing occurrence before upload staging or job creation."""
        owner = self._owner(owner_user_id)
        run_identity = self._run_text(run_id, "run_id", max_length=_MAX_RUN_IDENTITY_LENGTH)
        occurrence = self._run_text(
            occurrence_id,
            "occurrence_id",
            max_length=_MAX_RUN_IDENTITY_LENGTH,
        )
        batch = self._run_text(batch_id, "batch_id", max_length=_MAX_RUN_IDENTITY_LENGTH)
        identity = self._run_text(
            idempotency_identity,
            "idempotency_identity",
            max_length=_MAX_RUN_IDENTITY_LENGTH,
        )
        if type(attempt) is not int or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if source_kind not in {"url", "file"}:
            raise ValueError("source_kind must be url or file")
        if planned_item_id is not None and (type(planned_item_id) is not int or planned_item_id < 1):
            raise ValueError("planned_item_id must be a positive integer")

        now = self._now()
        with self._connection(write=True) as db:
            lock = " FOR UPDATE OF run, item" if self._postgres else ""
            expiry_sql = "run.expires_at > ?" if self._postgres else "julianday(run.expires_at) > julianday(?)"
            row = self._query(
                db,
                f"""
                SELECT run.status AS run_status, run.version AS run_version,
                       item.*
                FROM media_ingest_runs AS run
                JOIN media_ingest_run_items AS item
                  ON item.run_id = run.run_id AND item.owner_user_id = run.owner_user_id
                WHERE run.owner_user_id = ? AND run.run_id = ? AND item.occurrence_id = ?
                  AND {expiry_sql}
                {lock}
                """,  # nosec B608
                (owner, run_identity, occurrence, self._db_datetime(now)),
            ).fetchone()
            if row is None:
                raise self._not_found()
            data = self._row_dict(row)
            if str(data["run_status"]) not in {"staged", "running"}:
                raise PlaylistIngestConflictError("run is not accepting jobs")
            if int(data["attempt"]) != attempt:
                raise PlaylistIngestConflictError("occurrence attempt no longer matches")
            stored_planned = (
                int(data["planned_collection_item_id"]) if data.get("planned_collection_item_id") is not None else None
            )
            if planned_item_id is not None and stored_planned != planned_item_id:
                raise PlaylistIngestConflictError("planned item no longer matches")
            action = str(data.get("duplicate_policy") or "ingest")
            if action not in {"ingest", "overwrite"}:
                raise PlaylistIngestConflictError("occurrence does not require processing")
            expected_state = "awaiting_upload" if source_kind == "file" else "staged"
            expected_input_kind = "file_stub" if source_kind == "file" else None
            if expected_input_kind and str(data.get("input_kind")) != expected_input_kind:
                raise PlaylistIngestConflictError("occurrence source kind no longer matches")
            if source_kind == "url" and (str(data.get("input_kind")) == "file_stub" or not data.get("source_url")):
                raise PlaylistIngestConflictError("occurrence source kind no longer matches")

            state = str(data["state"])
            if state == "queued":
                if data.get("idempotency_identity") != identity or data.get("job_id") is None:
                    raise PlaylistIngestConflictError("occurrence is already bound")
                return self._run_item_record(data)
            if state == "submit_pending":
                if data.get("idempotency_identity") != identity:
                    raise PlaylistIngestConflictError("occurrence submission is already pending")
                return self._run_item_record(data)
            if state != expected_state:
                raise PlaylistIngestConflictError("occurrence is not processable")

            updated = self._query(
                db,
                """
                UPDATE media_ingest_run_items
                SET state = 'submit_pending', batch_id = ?, idempotency_identity = ?,
                    updated_at = ?
                WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
                  AND state = ? AND attempt = ? AND job_id IS NULL
                """,
                (
                    batch,
                    identity,
                    self._db_datetime(now),
                    owner,
                    run_identity,
                    occurrence,
                    expected_state,
                    attempt,
                ),
            )
            if updated.rowcount != 1:
                raise PlaylistIngestConflictError("occurrence state no longer matches")
            version = int(data["run_version"])
            run_updated = self._query(
                db,
                """
                UPDATE media_ingest_runs SET version = version + 1, updated_at = ?
                WHERE owner_user_id = ? AND run_id = ? AND version = ?
                """,
                (self._db_datetime(now), owner, run_identity, version),
            )
            if run_updated.rowcount != 1:
                raise PlaylistIngestConflictError("run version no longer matches")
            self._query(
                db,
                """
                INSERT INTO media_ingest_run_events (
                    run_id, owner_user_id, occurrence_id, batch_id, event_type,
                    state, attrs_json, occurred_at
                ) VALUES (?, ?, ?, ?, 'occurrence_submit_pending',
                          'submit_pending', ?, ?)
                """,
                (
                    run_identity,
                    owner,
                    occurrence,
                    batch,
                    self._json_value({"attempt": attempt}),
                    self._db_datetime(now),
                ),
            )
            item_row = self._query(
                db,
                """
                SELECT * FROM media_ingest_run_items
                WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
                """,
                (owner, run_identity, occurrence),
            ).fetchone()
        return self._run_item_record(item_row)

    def bind_run_item_job(
        self,
        owner_user_id: str,
        run_id: str,
        occurrence_id: str,
        *,
        attempt: int,
        job_id: int,
        batch_id: str,
        idempotency_identity: str,
    ) -> MediaIngestRunItemRecord:
        """Bind the exact owner-scoped idempotent Jobs row to one reserved occurrence."""
        owner = self._owner(owner_user_id)
        run_identity = self._run_text(run_id, "run_id", max_length=_MAX_RUN_IDENTITY_LENGTH)
        occurrence = self._run_text(
            occurrence_id,
            "occurrence_id",
            max_length=_MAX_RUN_IDENTITY_LENGTH,
        )
        batch = self._run_text(batch_id, "batch_id", max_length=_MAX_RUN_IDENTITY_LENGTH)
        identity = self._run_text(
            idempotency_identity,
            "idempotency_identity",
            max_length=_MAX_RUN_IDENTITY_LENGTH,
        )
        if type(attempt) is not int or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if type(job_id) is not int or job_id < 1:
            raise ValueError("job_id must be a positive integer")

        now = self._now()
        with self._connection(write=True) as db:
            lock = " FOR UPDATE OF run, item" if self._postgres else ""
            expiry_sql = "run.expires_at > ?" if self._postgres else "julianday(run.expires_at) > julianday(?)"
            row = self._query(
                db,
                f"""
                SELECT run.version AS run_version, run.batch_ids_json, item.*
                FROM media_ingest_runs AS run
                JOIN media_ingest_run_items AS item
                  ON item.run_id = run.run_id AND item.owner_user_id = run.owner_user_id
                WHERE run.owner_user_id = ? AND run.run_id = ? AND item.occurrence_id = ?
                  AND {expiry_sql}
                {lock}
                """,  # nosec B608
                (owner, run_identity, occurrence, self._db_datetime(now)),
            ).fetchone()
            if row is None:
                raise self._not_found()
            data = self._row_dict(row)
            if int(data["attempt"]) != attempt:
                raise PlaylistIngestConflictError("occurrence attempt no longer matches")
            if data.get("idempotency_identity") != identity or data.get("batch_id") != batch:
                raise PlaylistIngestConflictError("occurrence submission identity no longer matches")
            if str(data["state"]) == "queued":
                if int(data.get("job_id") or 0) != job_id:
                    raise PlaylistIngestConflictError("occurrence is already bound")
                return self._run_item_record(data)
            if str(data["state"]) != "submit_pending" or data.get("job_id") is not None:
                raise PlaylistIngestConflictError("occurrence submission is not pending")

            job_lock = " FOR SHARE" if self._postgres else ""
            job_row = self._query(
                db,
                f"""
                SELECT id, owner_user_id, domain, job_type, batch_group,
                       idempotency_key, payload
                FROM jobs WHERE id = ?
                {job_lock}
                """,  # nosec B608
                (job_id,),
            ).fetchone()
            if job_row is None:
                raise PlaylistIngestConflictError("media job is unavailable")
            job = self._row_dict(job_row)
            payload = self._json_dict(job.get("payload")) or {}
            if (
                str(job.get("owner_user_id") or "") != owner
                or job.get("domain") != "media_ingest"
                or job.get("job_type") != "media_ingest_item"
                or job.get("batch_group") != batch
                or job.get("idempotency_key") != identity
                or payload.get("run_id") != run_identity
                or payload.get("occurrence_id") != occurrence
                or type(payload.get("attempt")) is not int
                or payload.get("attempt") != attempt
            ):
                raise PlaylistIngestConflictError("media job binding does not match occurrence")

            updated = self._query(
                db,
                """
                UPDATE media_ingest_run_items
                SET state = 'queued', job_id = ?, updated_at = ?
                WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
                  AND state = 'submit_pending' AND attempt = ? AND job_id IS NULL
                  AND batch_id = ? AND idempotency_identity = ?
                """,
                (
                    job_id,
                    self._db_datetime(now),
                    owner,
                    run_identity,
                    occurrence,
                    attempt,
                    batch,
                    identity,
                ),
            )
            if updated.rowcount != 1:
                raise PlaylistIngestConflictError("occurrence submission no longer matches")
            batch_ids = self._json_list(data.get("batch_ids_json")) or []
            if batch not in batch_ids:
                batch_ids.append(batch)
            version = int(data["run_version"])
            run_updated = self._query(
                db,
                """
                UPDATE media_ingest_runs
                SET batch_ids_json = ?, version = version + 1, updated_at = ?
                WHERE owner_user_id = ? AND run_id = ? AND version = ?
                """,
                (self._json_value(batch_ids), self._db_datetime(now), owner, run_identity, version),
            )
            if run_updated.rowcount != 1:
                raise PlaylistIngestConflictError("run version no longer matches")
            self._query(
                db,
                """
                INSERT INTO media_ingest_run_events (
                    run_id, owner_user_id, occurrence_id, job_id, batch_id,
                    event_type, state, attrs_json, occurred_at
                ) VALUES (?, ?, ?, ?, ?, 'occurrence_job_accepted', 'queued', ?, ?)
                """,
                (
                    run_identity,
                    owner,
                    occurrence,
                    job_id,
                    batch,
                    self._json_value({"attempt": attempt}),
                    self._db_datetime(now),
                ),
            )
            item_row = self._query(
                db,
                """
                SELECT * FROM media_ingest_run_items
                WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
                """,
                (owner, run_identity, occurrence),
            ).fetchone()
        return self._run_item_record(item_row)

    def reset_run_item_job_submission(
        self,
        owner_user_id: str,
        run_id: str,
        occurrence_id: str,
        *,
        attempt: int,
        idempotency_identity: str,
    ) -> MediaIngestRunItemRecord:
        """Release an exact reservation when no media job was accepted."""
        owner = self._owner(owner_user_id)
        if type(attempt) is not int or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        identity = self._run_text(
            idempotency_identity,
            "idempotency_identity",
            max_length=_MAX_RUN_IDENTITY_LENGTH,
        )
        now = self._now()
        with self._connection(write=True) as db:
            lock = " FOR UPDATE OF run, item" if self._postgres else ""
            expiry_sql = "run.expires_at > ?" if self._postgres else "julianday(run.expires_at) > julianday(?)"
            row = self._query(
                db,
                f"""
                SELECT run.version AS run_version, item.*
                FROM media_ingest_runs AS run
                JOIN media_ingest_run_items AS item
                  ON item.run_id = run.run_id AND item.owner_user_id = run.owner_user_id
                WHERE run.owner_user_id = ? AND run.run_id = ? AND item.occurrence_id = ?
                  AND {expiry_sql}
                {lock}
                """,  # nosec B608
                (owner, str(run_id), str(occurrence_id), self._db_datetime(now)),
            ).fetchone()
            if row is None:
                raise self._not_found()
            data = self._row_dict(row)
            reset_state = "awaiting_upload" if str(data.get("input_kind")) == "file_stub" else "staged"
            if str(data["state"]) == reset_state and data.get("job_id") is None:
                return self._run_item_record(data)
            if (
                str(data["state"]) != "submit_pending"
                or int(data["attempt"]) != attempt
                or data.get("idempotency_identity") != identity
                or data.get("job_id") is not None
            ):
                raise PlaylistIngestConflictError("occurrence submission no longer matches")
            self._query(
                db,
                """
                UPDATE media_ingest_run_items
                SET state = ?, batch_id = NULL, idempotency_identity = NULL,
                    updated_at = ?
                WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
                  AND state = 'submit_pending' AND attempt = ?
                  AND idempotency_identity = ? AND job_id IS NULL
                """,
                (
                    reset_state,
                    self._db_datetime(now),
                    owner,
                    str(run_id),
                    str(occurrence_id),
                    attempt,
                    identity,
                ),
            )
            self._query(
                db,
                """
                UPDATE media_ingest_runs SET version = version + 1, updated_at = ?
                WHERE owner_user_id = ? AND run_id = ? AND version = ?
                """,
                (self._db_datetime(now), owner, str(run_id), int(data["run_version"])),
            )
            item_row = self._query(
                db,
                """
                SELECT * FROM media_ingest_run_items
                WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
                """,
                (owner, str(run_id), str(occurrence_id)),
            ).fetchone()
        return self._run_item_record(item_row)

    def prepare_nonprocessing_run_item(
        self,
        owner_user_id: str,
        run_id: str,
        occurrence_id: str,
    ) -> MediaIngestRunItemRecord:
        """Durably record intent before a reviewed duplicate action has side effects."""
        owner = self._owner(owner_user_id)
        now = self._now()
        with self._connection(write=True) as db:
            lock = " FOR UPDATE" if self._postgres else ""
            run_row = self._query(
                db,
                f"""
                SELECT version FROM media_ingest_runs
                WHERE owner_user_id = ? AND run_id = ? AND {self._unexpired_sql()}
                {lock}
                """,  # nosec B608
                (owner, str(run_id), self._db_datetime(now)),
            ).fetchone()
            if run_row is None:
                raise self._not_found()
            version = int(self._row_dict(run_row)["version"])
            item_row = self._query(
                db,
                f"""
                SELECT * FROM media_ingest_run_items
                WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
                {lock}
                """,  # nosec B608
                (owner, str(run_id), str(occurrence_id)),
            ).fetchone()
            if item_row is None:
                raise self._not_found()
            item = self._row_dict(item_row)
            if item.get("duplicate_policy") not in {
                "skip",
                "include_existing",
                "update_metadata_only",
            }:
                raise ValueError("invalid non-processing action outcome")
            if item.get("state") in {"preparing", "terminal"}:
                return self._run_item_record(item_row)
            if item.get("state") != "staged" or int(item.get("attempt") or 0) != 1:
                raise PlaylistIngestConflictError("run item is no longer staged")
            updated = self._query(
                db,
                """
                UPDATE media_ingest_run_items
                SET state = 'preparing', updated_at = ?
                WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
                  AND state = 'staged' AND attempt = 1
                """,
                (
                    self._db_datetime(now),
                    owner,
                    str(run_id),
                    str(occurrence_id),
                ),
            )
            if updated.rowcount != 1:
                raise PlaylistIngestConflictError("run item is no longer staged")
            bumped = self._query(
                db,
                """
                UPDATE media_ingest_runs SET version = version + 1, updated_at = ?
                WHERE owner_user_id = ? AND run_id = ? AND version = ?
                """,
                (self._db_datetime(now), owner, str(run_id), version),
            )
            if bumped.rowcount != 1:
                raise PlaylistIngestConflictError("run version no longer matches expected version")
            self._query(
                db,
                """
                INSERT INTO media_ingest_run_events (
                    run_id, owner_user_id, occurrence_id, event_type, state,
                    attrs_json, occurred_at
                ) VALUES (?, ?, ?, 'duplicate_action_preparing', 'preparing', ?, ?)
                """,
                (
                    str(run_id),
                    owner,
                    str(occurrence_id),
                    self._json_value({"action": item["duplicate_policy"]}),
                    self._db_datetime(now),
                ),
            )
            prepared_row = self._query(
                db,
                """
                SELECT * FROM media_ingest_run_items
                WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
                """,
                (owner, str(run_id), str(occurrence_id)),
            ).fetchone()
            if prepared_row is None:
                raise PlaylistIngestConflictError("run item preparation changed")
            prepared = self._run_item_record(prepared_row)
        return prepared

    def resolve_nonprocessing_run_item(
        self,
        owner_user_id: str,
        run_id: str,
        occurrence_id: str,
        *,
        outcome: str,
        media_id: int | None,
    ) -> MediaIngestRunItemRecord:
        """Set one reviewed duplicate action terminal and append its event atomically."""
        if outcome not in {
            "skipped_existing",
            "included_existing",
            "metadata_updated",
            "metadata_update_failed",
        }:
            raise ValueError("invalid non-processing outcome")
        if media_id is not None and (type(media_id) is not int or media_id < 1):
            raise ValueError("media_id must be a positive integer")
        owner = self._owner(owner_user_id)
        now = self._now()
        with self._connection(write=True) as db:
            lock = " FOR UPDATE" if self._postgres else ""
            run_sql = f"""
                SELECT version, status FROM media_ingest_runs
                WHERE owner_user_id = ? AND run_id = ? AND {self._unexpired_sql()}
                {lock}
            """  # nosec B608
            run_row = self._query(
                db,
                run_sql,
                (owner, str(run_id), self._db_datetime(now)),
            ).fetchone()
            if run_row is None:
                raise self._not_found()
            run = self._row_dict(run_row)
            version = int(run["version"])
            item_row = self._query(
                db,
                f"""
                SELECT * FROM media_ingest_run_items
                WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
                {lock}
                """,  # nosec B608
                (owner, str(run_id), str(occurrence_id)),
            ).fetchone()
            if item_row is None:
                raise self._not_found()
            item = self._row_dict(item_row)
            action = item.get("duplicate_policy")
            allowed = {
                "skip": {"skipped_existing"},
                "include_existing": {"included_existing", "metadata_update_failed"},
                "update_metadata_only": {"metadata_updated", "metadata_update_failed"},
            }
            if action not in allowed or outcome not in allowed[action] or (action != "skip" and media_id is None):
                raise ValueError("invalid non-processing action outcome")
            if item.get("state") == "terminal":
                if item.get("outcome") == outcome and item.get("media_id") == media_id:
                    return self._run_item_record(item_row)
                raise PlaylistIngestConflictError("run item terminal result does not match")
            if item.get("state") != "preparing" or int(item.get("attempt") or 0) != 1:
                raise PlaylistIngestConflictError("run item is no longer preparing")
            updated = self._query(
                db,
                """
                UPDATE media_ingest_run_items
                SET state = 'terminal', outcome = ?, media_id = ?, retryable = ?, updated_at = ?
                WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
                  AND state = 'preparing' AND attempt = 1
                """,
                (
                    outcome,
                    media_id,
                    False,
                    self._db_datetime(now),
                    owner,
                    str(run_id),
                    str(occurrence_id),
                ),
            )
            if updated.rowcount != 1:
                raise PlaylistIngestConflictError("run item is no longer preparing")
            incomplete = self._query(
                db,
                """
                SELECT 1 FROM media_ingest_run_items
                WHERE owner_user_id = ? AND run_id = ? AND state <> 'terminal'
                LIMIT 1
                """,
                (owner, str(run_id)),
            ).fetchone()
            run_status = "completed" if incomplete is None else str(run["status"])
            bumped = self._query(
                db,
                """
                UPDATE media_ingest_runs
                SET status = ?, version = version + 1, updated_at = ?
                WHERE owner_user_id = ? AND run_id = ? AND version = ?
                """,
                (run_status, self._db_datetime(now), owner, str(run_id), version),
            )
            if bumped.rowcount != 1:
                raise PlaylistIngestConflictError("run version no longer matches expected version")
            self._query(
                db,
                """
                INSERT INTO media_ingest_run_events (
                    run_id, owner_user_id, occurrence_id, event_type, state,
                    outcome, attrs_json, occurred_at
                ) VALUES (?, ?, ?, 'duplicate_action_resolved', 'terminal', ?, ?, ?)
                """,
                (
                    str(run_id),
                    owner,
                    str(occurrence_id),
                    outcome,
                    self._json_value({"media_id": media_id, "run_status": run_status}),
                    self._db_datetime(now),
                ),
            )
            resolved_row = self._query(
                db,
                """
                SELECT * FROM media_ingest_run_items
                WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
                """,
                (owner, str(run_id), str(occurrence_id)),
            ).fetchone()
            if resolved_row is None:
                raise PlaylistIngestConflictError("run item finalization changed")
            resolved = self._run_item_record(resolved_row)
        return resolved

    def attach_collection_plan(
        self,
        owner_user_id: str,
        run_id: str,
        *,
        collection_id: int,
        planned_item_ids: Mapping[str, int],
    ) -> MediaIngestRunRecord:
        """Attach one complete non-skip collection plan to a staged run atomically."""
        if type(collection_id) is not int or collection_id < 1:
            raise ValueError("collection_id must be a positive integer")
        if type(planned_item_ids) is not dict or len(planned_item_ids) > _MAX_PAGE_SIZE:
            raise ValueError("planned_item_ids must be a bounded object")
        normalized: dict[str, int] = {}
        for occurrence_id, item_id in planned_item_ids.items():
            if type(occurrence_id) is not str or not occurrence_id or len(occurrence_id) > 255:
                raise ValueError("planned item occurrence IDs must be canonical strings")
            if type(item_id) is not int or item_id < 1:
                raise ValueError("planned collection item IDs must be positive integers")
            normalized[occurrence_id] = item_id
        if len(set(normalized.values())) != len(normalized):
            raise ValueError("planned collection item IDs must be unique")

        owner = self._owner(owner_user_id)
        now = self._now()
        with self._connection(write=True) as db:
            if self._postgres:
                run_sql = """
                    SELECT version FROM media_ingest_runs
                    WHERE owner_user_id = ? AND run_id = ? AND status = 'staged'
                      AND collection_id IS NULL AND expires_at > ?
                    FOR UPDATE
                """
                items_sql = """
                    SELECT occurrence_id, duplicate_policy
                    FROM media_ingest_run_items
                    WHERE owner_user_id = ? AND run_id = ?
                    ORDER BY ordinal
                    FOR UPDATE
                """
            else:
                run_sql = """
                    SELECT version FROM media_ingest_runs
                    WHERE owner_user_id = ? AND run_id = ? AND status = 'staged'
                      AND collection_id IS NULL
                      AND julianday(expires_at) > julianday(?)
                """
                items_sql = """
                    SELECT occurrence_id, duplicate_policy
                    FROM media_ingest_run_items
                    WHERE owner_user_id = ? AND run_id = ?
                    ORDER BY ordinal
                """
            run_row = self._query(
                db,
                run_sql,
                (owner, str(run_id), self._db_datetime(now)),
            ).fetchone()
            if run_row is None:
                raise PlaylistIngestConflictError("run is not available for collection planning")
            version = int(self._row_dict(run_row)["version"])
            item_rows = self._query(db, items_sql, (owner, str(run_id))).fetchall()
            expected = {
                str(self._row_dict(row)["occurrence_id"])
                for row in item_rows
                if self._row_dict(row).get("duplicate_policy") != "skip"
            }
            if set(normalized) != expected:
                raise ValueError("planned collection mapping must cover every non-skip occurrence")
            for occurrence_id, planned_item_id in normalized.items():
                updated = self._query(
                    db,
                    """
                    UPDATE media_ingest_run_items
                    SET planned_collection_item_id = ?, updated_at = ?
                    WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
                      AND planned_collection_item_id IS NULL
                    """,
                    (
                        planned_item_id,
                        self._db_datetime(now),
                        owner,
                        str(run_id),
                        occurrence_id,
                    ),
                )
                if updated.rowcount != 1:
                    raise PlaylistIngestConflictError("run item collection mapping changed")
            attached = self._query(
                db,
                """
                UPDATE media_ingest_runs
                SET collection_id = ?, version = version + 1, updated_at = ?
                WHERE owner_user_id = ? AND run_id = ? AND status = 'staged'
                  AND collection_id IS NULL AND version = ?
                """,
                (collection_id, self._db_datetime(now), owner, str(run_id), version),
            )
            if attached.rowcount != 1:
                raise PlaylistIngestConflictError("run collection plan changed")
            self._query(
                db,
                """
                INSERT INTO media_ingest_run_events (
                    run_id, owner_user_id, event_type, state, attrs_json, occurred_at
                ) VALUES (?, ?, 'collection_plan_attached', 'staged', ?, ?)
                """,
                (
                    str(run_id),
                    owner,
                    self._json_value({"collection_id": collection_id, "planned_item_count": len(normalized)}),
                    self._db_datetime(now),
                ),
            )
            attached_row = self._query(
                db,
                """
                SELECT * FROM media_ingest_runs
                WHERE owner_user_id = ? AND run_id = ? AND collection_id = ?
                """,
                (owner, str(run_id), collection_id),
            ).fetchone()
            if attached_row is None:
                raise PlaylistIngestConflictError("run collection plan changed")
            attached_run = self._run_record(attached_row)
        return attached_run

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
                materialization_ids = [str(self._row_dict(row)["materialization_id"]) for row in materialization_rows]
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
                f"""
                DELETE FROM media_ingest_run_events WHERE owner_user_id = ? AND run_id IN (
                    SELECT run_id FROM media_ingest_runs WHERE owner_user_id = ? AND {self._expired_sql()}
                )
                """,  # nosec B608
                (owner, owner, cutoff),
            )
            self._query(
                db,
                f"""
                DELETE FROM media_ingest_run_items WHERE owner_user_id = ? AND run_id IN (
                    SELECT run_id FROM media_ingest_runs WHERE owner_user_id = ? AND {self._expired_sql()}
                )
                """,  # nosec B608
                (owner, owner, cutoff),
            )
            runs = self._query(
                db,
                f"DELETE FROM media_ingest_runs WHERE owner_user_id = ? AND {self._expired_sql()}",  # nosec B608
                (owner, cutoff),
            ).rowcount
            self._query(
                db,
                f"""
                DELETE FROM playlist_materialization_items
                WHERE owner_user_id = ? AND materialization_id IN (
                    SELECT materialization_id FROM playlist_materializations
                    WHERE owner_user_id = ? AND {self._expired_sql()}
                )
                """,  # nosec B608
                (owner, owner, cutoff),
            )
            materializations = self._query(
                db,
                f"DELETE FROM playlist_materializations WHERE owner_user_id = ? AND {self._expired_sql()}",  # nosec B608
                (owner, cutoff),
            ).rowcount
            self._query(
                db,
                f"""
                DELETE FROM playlist_preflight_items
                WHERE owner_user_id = ? AND preflight_id IN (
                    SELECT preflight_id FROM playlist_preflights
                    WHERE owner_user_id = ? AND {self._expired_sql()}
                )
                """,  # nosec B608
                (owner, owner, cutoff),
            )
            preflights = self._query(
                db,
                f"DELETE FROM playlist_preflights WHERE owner_user_id = ? AND {self._expired_sql()}",  # nosec B608
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
    "PlaylistPreflightLeaseLostError",
    "PlaylistPreflightCapacityError",
    "PlaylistPreflightRecord",
    "ResolvedMaterializationOccurrence",
]

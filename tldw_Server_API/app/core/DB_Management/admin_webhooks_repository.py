"""Transactional SQLite/PostgreSQL repository for canonical admin webhooks."""

from __future__ import annotations

import hmac
import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

import asyncpg

from tldw_Server_API.app.core.Admin_Webhooks.crypto import ProtectedValue
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    IdempotencyScope,
    WebhookError,
    WebhookRegistration,
    redact_target,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    ConnectionPoolExhaustedError,
    DatabaseLockError,
)

_MAX_PAGE_SIZE = 500
_MIN_TOMBSTONE_RETENTION_DAYS = 30
DATABASE_PROTECTED_TABLE_ORDER = (
    "registration_targets",
    "registration_secrets",
    "event_bodies",
    "idempotency_replay_secrets",
)
_DATABASE_PROTECTED_TABLES = frozenset(DATABASE_PROTECTED_TABLE_ORDER)
_ACTIVITY_KINDS = frozenset({"registration_mutation", "event_capture", "delivery_attempt"})
_SAFE_METADATA_KEYS = frozenset(
    {
        "completed_after_config_change",
        "latency_ms",
        "reason_code",
        "redelivery_to_changed_config",
        "result_kind",
        "retry_after_seconds",
        "status_code",
    }
)
_SAFE_METADATA_STRING = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_LOOKUP_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REQUEST_FINGERPRINT = re.compile(r"^hmac-sha256:[0-9a-f]{64}$")
_MIGRATION_JSON_COLUMNS = {
    "source_mapping_json": dict,
    "source_rejections_json": list,
}
_MIGRATION_MUTABLE_COLUMNS = frozenset(
    {
        "phase",
        "import_operation_id",
        "import_operator_id",
        "import_started_at",
        "import_approved_at",
        "artifacts_ready_at",
        "database_committed_at",
        "fingerprint_key_id",
        "active_primary_key_id",
        "system_ops_webhook_fingerprint",
        "legacy_table_fingerprint",
        "source_mapping_json",
        "redacted_report_digest",
        "protected_backup_ciphertext_digest",
        "source_rejections_json",
        "completed_at",
        "active_report_path",
        "active_backup_path",
        "active_key_path",
        "staging_report_path",
        "staging_backup_path",
        "staging_key_path",
        "report_owner_id",
        "report_group_id",
        "report_mode",
        "report_file_identity",
        "backup_owner_id",
        "backup_group_id",
        "backup_mode",
        "backup_file_identity",
        "rollback_key_owner_id",
        "rollback_key_group_id",
        "rollback_key_mode",
        "rollback_key_file_identity",
        "rollback_expires_at",
        "rollback_retirement_phase",
        "rollback_retirement_operator_id",
        "rollback_retirement_started_at",
        "rollback_retirement_completed_at",
        "expected_ciphertext_digest",
        "rotation_operation_id",
        "rotation_source_key_id",
        "rotation_target_key_id",
        "rotation_phase",
        "rotation_table_cursor",
        "rotation_key_cursor",
        "rotation_processed_count",
        "rotation_verified_count",
        "rotation_started_at",
        "rotation_completed_at",
    }
)


class WebhookRepositoryErrorCode(str, Enum):
    """Stable failures owned by the canonical persistence boundary."""

    DATABASE_BUSY = "admin_webhook_database_busy"
    SEQUENCE_UNAVAILABLE = "admin_webhook_sequence_unavailable"
    NOT_FOUND = "admin_webhook_not_found"
    STALE_REVISION = "admin_webhook_stale_revision"
    REGISTRATION_LIMIT = "admin_webhook_registration_limit"
    ACTIVE_LIMIT = "admin_webhook_active_limit"
    IDEMPOTENCY_NOT_CLAIMED = "admin_webhook_idempotency_not_claimed"
    IDEMPOTENCY_COMPLETION_INVALID = "admin_webhook_idempotency_completion_invalid"
    MIGRATION_STATE_UNAVAILABLE = "admin_webhook_migration_state_unavailable"
    STALE_MIGRATION_STATE = "admin_webhook_migration_state_stale"


class WebhookRepositoryError(Exception):
    """Sanitized repository failure with no database or caller-controlled text."""

    def __init__(self, code: WebhookRepositoryErrorCode | str) -> None:
        self.code = WebhookRepositoryErrorCode(code)
        self.http_status = 503 if self.code is WebhookRepositoryErrorCode.DATABASE_BUSY else None
        super().__init__(self.code.value)


@dataclass(frozen=True)
class RegistrationTarget:
    """Protected destination plus the only redacted searchable metadata."""

    protected: ProtectedValue
    hostname: str
    display: str


@dataclass(frozen=True)
class RegistrationInsert:
    """Complete canonical registration insert with no plaintext secret or URL."""

    id: int
    description: str
    target: RegistrationTarget
    event_types: tuple[str, ...]
    active: bool
    timeout_seconds: int
    secret: ProtectedValue
    secret_rotation_required: bool
    actor_user_id: int
    now: datetime


@dataclass(frozen=True)
class LegacyWebhookRow:
    """One raw legacy database row kept inside the migration boundary."""

    source_identity: str
    values: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.source_identity, str) or not self.source_identity:
            raise ValueError("legacy source identity is invalid")
        if not isinstance(self.values, Mapping):
            raise TypeError("legacy row values must be a mapping")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True)
class LegacyImportDatabaseSnapshot:
    """Deterministic legacy and canonical allocator state for one import scan."""

    table_present: bool
    rows: tuple[LegacyWebhookRow, ...]
    canonical_registration_ids: tuple[int, ...]
    canonical_non_deleted_count: int
    next_registration_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.table_present, bool):
            raise TypeError("legacy table presence must be boolean")
        if self.next_registration_id < 1:
            raise ValueError("next registration ID must be positive")
        if tuple(sorted(set(self.canonical_registration_ids))) != (
            self.canonical_registration_ids
        ):
            raise ValueError("canonical registration IDs must be unique and sorted")
        if (
            isinstance(self.canonical_non_deleted_count, bool)
            or not isinstance(self.canonical_non_deleted_count, int)
            or not 0 <= self.canonical_non_deleted_count <= len(self.canonical_registration_ids)
        ):
            raise ValueError("canonical non-deleted count is invalid")


class _Unset:
    __slots__ = ()


UNSET = _Unset()


@dataclass(frozen=True)
class RegistrationPatch:
    """Partial registration mutation; omitted fields are distinct from values."""

    description: str | _Unset = UNSET
    target: RegistrationTarget | _Unset = UNSET
    event_types: tuple[str, ...] | _Unset = UNSET
    active: bool | _Unset = UNSET
    timeout_seconds: int | _Unset = UNSET
    secret: ProtectedValue | _Unset = UNSET
    secret_rotation_required: bool | _Unset = UNSET


@dataclass(frozen=True)
class RegistrationPatchResult:
    """Conditional mutation result including explicit effective-no-op state."""

    changed: bool
    registration: WebhookRegistration


@dataclass(frozen=True)
class StoredWebhookRegistration:
    """Internal registration row including protected values for trusted services."""

    registration: WebhookRegistration
    target: ProtectedValue
    secret: ProtectedValue


@dataclass(frozen=True)
class ProtectedRow:
    """One closed protected-value inventory entry with contextual identity."""

    table: str
    row_identity: str
    field: str
    protected: ProtectedValue
    purpose: str
    envelope_identity: Mapping[str, str | int]

    def __post_init__(self) -> None:
        if self.table not in _DATABASE_PROTECTED_TABLES:
            raise ValueError("protected row table is invalid")
        if not isinstance(self.row_identity, str) or not self.row_identity:
            raise ValueError("protected row identity is invalid")
        if not isinstance(self.protected, ProtectedValue):
            raise TypeError("protected row value is invalid")
        expected = {
            "registration_targets": ("target", "registration.target"),
            "registration_secrets": ("secret", "registration.secret"),
            "event_bodies": ("body", "event.body"),
            "idempotency_replay_secrets": (
                "replay_secret",
                "idempotency.secret_replay",
            ),
        }[self.table]
        if (self.field, self.purpose) != expected:
            raise ValueError("protected row field is invalid")
        if not isinstance(self.envelope_identity, Mapping) or not self.envelope_identity:
            raise ValueError("protected row envelope identity is invalid")
        object.__setattr__(
            self,
            "envelope_identity",
            MappingProxyType(dict(self.envelope_identity)),
        )


@dataclass(frozen=True)
class RegistrationLimitState:
    """Read-only admission state that remains valid while already over limit."""

    current: int
    limit: int
    at_limit: bool
    over_limit: bool


class IdempotencyLookupKind(str, Enum):
    """Outcome of one scoped idempotency lookup or claim."""

    NEW = "new"
    REPLAY = "replay"
    CONFLICT = "conflict"
    IN_PROGRESS = "in_progress"


@dataclass(frozen=True)
class IdempotencyLookup:
    """Protected replay state without a raw key or canonical request body."""

    kind: IdempotencyLookupKind
    resource_id: int | None = None
    resource_version: int | None = None
    secret_version: int | None = None
    replay_secret: ProtectedValue | None = None
    test_delivery_id: str | None = None
    test_attempt_id: str | None = None
    response_status: int | None = None
    response_metadata: Mapping[str, object] | None = None
    resource_superseded: bool = False
    expires_at: datetime | None = None


@dataclass(frozen=True)
class MigrationState:
    """Typed durable singleton shared by import, rotation, and status services."""

    singleton_id: int
    schema_version: int
    state_revision: int
    phase: str
    import_operation_id: str | None
    import_operator_id: int | None
    import_started_at: datetime | None
    import_approved_at: datetime | None
    artifacts_ready_at: datetime | None
    database_committed_at: datetime | None
    fingerprint_key_id: str | None
    active_primary_key_id: str | None
    system_ops_webhook_fingerprint: str | None
    legacy_table_fingerprint: str | None
    source_mapping: Mapping[str, object]
    redacted_report_digest: str | None
    protected_backup_ciphertext_digest: str | None
    source_rejections: tuple[object, ...]
    completed_at: datetime | None
    active_report_path: str | None
    active_backup_path: str | None
    active_key_path: str | None
    staging_report_path: str | None
    staging_backup_path: str | None
    staging_key_path: str | None
    report_owner_id: int | None
    report_group_id: int | None
    report_mode: int | None
    report_file_identity: str | None
    backup_owner_id: int | None
    backup_group_id: int | None
    backup_mode: int | None
    backup_file_identity: str | None
    rollback_key_owner_id: int | None
    rollback_key_group_id: int | None
    rollback_key_mode: int | None
    rollback_key_file_identity: str | None
    rollback_expires_at: datetime | None
    rollback_retirement_phase: str
    rollback_retirement_operator_id: int | None
    rollback_retirement_started_at: datetime | None
    rollback_retirement_completed_at: datetime | None
    expected_ciphertext_digest: str | None
    first_canonical_activity_at: datetime | None
    first_canonical_activity_kind: str | None
    rotation_operation_id: str | None
    rotation_source_key_id: str | None
    rotation_target_key_id: str | None
    rotation_phase: str | None
    rotation_table_cursor: str | None
    rotation_key_cursor: str | None
    rotation_processed_count: int
    rotation_verified_count: int
    rotation_started_at: datetime | None
    rotation_completed_at: datetime | None
    updated_at: datetime


def _utc_datetime(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    else:
        raise ValueError("invalid persisted datetime")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _validate_registration_target(target: RegistrationTarget) -> None:
    try:
        canonical_display = redact_target(target.display)
        parsed_display = urlsplit(target.display)
        display_hostname = parsed_display.hostname
    except (TypeError, ValueError, WebhookError) as exc:
        raise ValueError("registration target metadata is invalid") from exc
    if (
        canonical_display != target.display
        or parsed_display.scheme not in {"https", "http"}
        or display_hostname is None
        or display_hostname.rstrip(".").lower() != target.hostname.rstrip(".").lower()
    ):
        raise ValueError("registration target metadata is not a redacted origin")


def _strict_json_object(value: object) -> dict[str, object]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("persisted JSON must be an object")
    return parsed


def _strict_json_array(value: object) -> list[object]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("persisted JSON must be an array")
    return parsed


def _row_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    keys = getattr(row, "keys", None)
    if callable(keys):
        return {str(key): row[key] for key in keys()}
    return dict(row)


def _question_marks_to_postgres(query: str, count: int) -> str:
    if query.count("?") != count:
        raise ValueError("repository SQL placeholder mismatch")
    parts = query.split("?")
    return "".join(part + (f"${index + 1}" if index < count else "") for index, part in enumerate(parts))


def _registration_from_row(row: Mapping[str, Any]) -> WebhookRegistration:
    raw_events = _strict_json_array(row["event_types_json"])
    if any(not isinstance(item, str) for item in raw_events):
        raise ValueError("persisted event set is invalid")
    created_at = _parse_datetime(row["created_at"])
    updated_at = _parse_datetime(row["updated_at"])
    if created_at is None or updated_at is None:
        raise ValueError("persisted registration timestamps are invalid")
    return WebhookRegistration(
        id=int(row["id"]),
        description=str(row["description"]),
        target_display=str(row["target_display"]),
        target_hostname=str(row["target_hostname"]),
        event_types=tuple(str(item) for item in raw_events),
        active=bool(row["active"]),
        timeout_seconds=int(row["timeout_seconds"]),
        revision=int(row["revision"]),
        delivery_config_version=int(row["delivery_config_version"]),
        target_version=int(row["target_version"]),
        secret_version=int(row["secret_version"]),
        secret_rotation_required=bool(row["secret_rotation_required"]),
        created_by_user_id=int(row["created_by_user_id"]),
        updated_by_user_id=int(row["updated_by_user_id"]),
        created_at=created_at,
        updated_at=updated_at,
        deleted_at=_parse_datetime(row["deleted_at"]),
        deleted_by_user_id=(int(row["deleted_by_user_id"]) if row["deleted_by_user_id"] is not None else None),
    )


def _stored_registration_from_row(
    row: Mapping[str, Any],
) -> StoredWebhookRegistration:
    return StoredWebhookRegistration(
        registration=_registration_from_row(row),
        target=ProtectedValue(
            ciphertext_json=str(row["target_ciphertext_json"]),
            key_id=str(row["target_key_id"]),
        ),
        secret=ProtectedValue(
            ciphertext_json=str(row["secret_ciphertext_json"]),
            key_id=str(row["secret_key_id"]),
        ),
    )


def _migration_state_from_row(row: Mapping[str, Any]) -> MigrationState:
    source_mapping = _strict_json_object(row["source_mapping_json"])
    source_rejections = _strict_json_array(row["source_rejections_json"])
    updated_at = _parse_datetime(row["updated_at"])
    if updated_at is None:
        raise ValueError("persisted migration timestamp is invalid")

    def optional_int(name: str) -> int | None:
        value = row[name]
        return int(value) if value is not None else None

    def optional_text(name: str) -> str | None:
        value = row[name]
        return str(value) if value is not None else None

    return MigrationState(
        singleton_id=int(row["singleton_id"]),
        schema_version=int(row["schema_version"]),
        state_revision=int(row["state_revision"]),
        phase=str(row["phase"]),
        import_operation_id=optional_text("import_operation_id"),
        import_operator_id=optional_int("import_operator_id"),
        import_started_at=_parse_datetime(row["import_started_at"]),
        import_approved_at=_parse_datetime(row["import_approved_at"]),
        artifacts_ready_at=_parse_datetime(row["artifacts_ready_at"]),
        database_committed_at=_parse_datetime(row["database_committed_at"]),
        fingerprint_key_id=optional_text("fingerprint_key_id"),
        active_primary_key_id=optional_text("active_primary_key_id"),
        system_ops_webhook_fingerprint=optional_text("system_ops_webhook_fingerprint"),
        legacy_table_fingerprint=optional_text("legacy_table_fingerprint"),
        source_mapping=MappingProxyType(source_mapping),
        redacted_report_digest=optional_text("redacted_report_digest"),
        protected_backup_ciphertext_digest=optional_text("protected_backup_ciphertext_digest"),
        source_rejections=tuple(source_rejections),
        completed_at=_parse_datetime(row["completed_at"]),
        active_report_path=optional_text("active_report_path"),
        active_backup_path=optional_text("active_backup_path"),
        active_key_path=optional_text("active_key_path"),
        staging_report_path=optional_text("staging_report_path"),
        staging_backup_path=optional_text("staging_backup_path"),
        staging_key_path=optional_text("staging_key_path"),
        report_owner_id=optional_int("report_owner_id"),
        report_group_id=optional_int("report_group_id"),
        report_mode=optional_int("report_mode"),
        report_file_identity=optional_text("report_file_identity"),
        backup_owner_id=optional_int("backup_owner_id"),
        backup_group_id=optional_int("backup_group_id"),
        backup_mode=optional_int("backup_mode"),
        backup_file_identity=optional_text("backup_file_identity"),
        rollback_key_owner_id=optional_int("rollback_key_owner_id"),
        rollback_key_group_id=optional_int("rollback_key_group_id"),
        rollback_key_mode=optional_int("rollback_key_mode"),
        rollback_key_file_identity=optional_text("rollback_key_file_identity"),
        rollback_expires_at=_parse_datetime(row["rollback_expires_at"]),
        rollback_retirement_phase=str(row["rollback_retirement_phase"]),
        rollback_retirement_operator_id=optional_int("rollback_retirement_operator_id"),
        rollback_retirement_started_at=_parse_datetime(row["rollback_retirement_started_at"]),
        rollback_retirement_completed_at=_parse_datetime(row["rollback_retirement_completed_at"]),
        expected_ciphertext_digest=optional_text("expected_ciphertext_digest"),
        first_canonical_activity_at=_parse_datetime(row["first_canonical_activity_at"]),
        first_canonical_activity_kind=optional_text("first_canonical_activity_kind"),
        rotation_operation_id=optional_text("rotation_operation_id"),
        rotation_source_key_id=optional_text("rotation_source_key_id"),
        rotation_target_key_id=optional_text("rotation_target_key_id"),
        rotation_phase=optional_text("rotation_phase"),
        rotation_table_cursor=optional_text("rotation_table_cursor"),
        rotation_key_cursor=optional_text("rotation_key_cursor"),
        rotation_processed_count=int(row["rotation_processed_count"]),
        rotation_verified_count=int(row["rotation_verified_count"]),
        rotation_started_at=_parse_datetime(row["rotation_started_at"]),
        rotation_completed_at=_parse_datetime(row["rotation_completed_at"]),
        updated_at=updated_at,
    )


def _safe_response_metadata(
    value: Mapping[str, object] | None,
) -> tuple[str | None, Mapping[str, object] | None]:
    if value is None:
        return None, None
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if key not in _SAFE_METADATA_KEYS:
            raise ValueError("response metadata key is not allowed")
        if isinstance(item, str):
            if _SAFE_METADATA_STRING.fullmatch(item) is None:
                raise ValueError("response metadata string is invalid")
        elif isinstance(item, bool) or item is None:
            pass
        elif isinstance(item, int):
            if not -(2**31) <= item <= 2**31 - 1:
                raise ValueError("response metadata integer is out of range")
        else:
            raise ValueError("response metadata value is invalid")
        normalized[key] = item
    encoded = _canonical_json(normalized)
    if len(encoded.encode("utf-8")) > 16_384:
        raise ValueError("response metadata is too large")
    return encoded, MappingProxyType(normalized)


def _referenced_registration_ids(value: object) -> set[int]:
    referenced: set[int] = set()
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, bool):
            continue
        if isinstance(current, int) and current > 0:
            referenced.add(current)
        elif isinstance(current, Mapping):
            pending.extend(current.values())
        elif isinstance(current, Sequence) and not isinstance(
            current,
            (str, bytes, bytearray),
        ):
            pending.extend(current)
    return referenced


def _is_database_busy(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (
                ConnectionPoolExhaustedError,
                DatabaseLockError,
                asyncpg.LockNotAvailableError,
                asyncpg.QueryCanceledError,
                asyncpg.DeadlockDetectedError,
            ),
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


class _ConnectionAdapter:
    def __init__(self, connection: object, *, is_postgres: bool) -> None:
        self._connection = connection
        self._is_postgres = is_postgres

    def _query_and_params(
        self,
        query: str,
        params: Sequence[object],
    ) -> tuple[str, tuple[object, ...]]:
        normalized = tuple(params)
        if self._is_postgres:
            return _question_marks_to_postgres(query, len(normalized)), normalized
        sqlite_params = tuple(
            item.astimezone(timezone.utc).isoformat(sep=" ")
            if isinstance(item, datetime)
            else int(item)
            if isinstance(item, bool)
            else item
            for item in normalized
        )
        return query, sqlite_params

    async def _fetchrow(
        self,
        query: str,
        params: Sequence[object] = (),
    ) -> dict[str, Any] | None:
        normalized_query, normalized_params = self._query_and_params(query, params)
        if self._is_postgres:
            row = await self._connection.fetchrow(  # type: ignore[attr-defined]
                normalized_query,
                *normalized_params,
            )
        else:
            cursor = await self._connection.execute(  # type: ignore[attr-defined]
                normalized_query,
                normalized_params,
            )
            row = await cursor.fetchone()
        return _row_dict(row) if row is not None else None

    async def _fetch(
        self,
        query: str,
        params: Sequence[object] = (),
    ) -> list[dict[str, Any]]:
        normalized_query, normalized_params = self._query_and_params(query, params)
        if self._is_postgres:
            rows = await self._connection.fetch(  # type: ignore[attr-defined]
                normalized_query,
                *normalized_params,
            )
        else:
            cursor = await self._connection.execute(  # type: ignore[attr-defined]
                normalized_query,
                normalized_params,
            )
            rows = await cursor.fetchall()
        return [_row_dict(row) for row in rows]

    async def _execute(
        self,
        query: str,
        params: Sequence[object] = (),
    ) -> int:
        normalized_query, normalized_params = self._query_and_params(query, params)
        if self._is_postgres:
            status = await self._connection.execute(  # type: ignore[attr-defined]
                normalized_query,
                *normalized_params,
            )
            tail = str(status).rsplit(" ", 1)[-1]
            return int(tail) if tail.isdigit() else 0
        cursor = await self._connection.execute(  # type: ignore[attr-defined]
            normalized_query,
            normalized_params,
        )
        return max(int(cursor.rowcount), 0)


class AdminWebhookRepository:
    """Application-scoped repository that owns all canonical webhook SQL."""

    def __init__(
        self,
        pool: DatabasePool,
        *,
        postgres_lock_timeout_ms: int = 5_000,
        postgres_statement_timeout_ms: int = 15_000,
    ) -> None:
        if postgres_lock_timeout_ms < 1 or postgres_statement_timeout_ms < 1:
            raise ValueError("database timeouts must be positive")
        self._pool = pool
        self._postgres_lock_timeout_ms = postgres_lock_timeout_ms
        self._postgres_statement_timeout_ms = postgres_statement_timeout_ms

    @property
    def database_path(self) -> Path | None:
        """Return the SQLite filesystem path without inventing one for PostgreSQL."""
        value = getattr(self._pool, "_sqlite_fs_path", None)
        return Path(value) if value and value != ":memory:" else None

    @property
    def is_postgres(self) -> bool:
        return self._pool.pool is not None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AdminWebhookUnitOfWork]:
        """Open one atomic unit and map only bounded contention to a stable error."""
        try:
            async with self._pool.transaction() as connection:
                unit = AdminWebhookUnitOfWork(
                    connection,
                    is_postgres=self.is_postgres,
                )
                if self.is_postgres:
                    await unit._fetchrow(
                        "SELECT set_config('lock_timeout', ?, true) AS value",
                        (f"{self._postgres_lock_timeout_ms}ms",),
                    )
                    await unit._fetchrow(
                        "SELECT set_config('statement_timeout', ?, true) AS value",
                        (f"{self._postgres_statement_timeout_ms}ms",),
                    )
                yield unit
        except MandatoryAuditWriteError:
            raise
        except Exception as exc:
            if _is_database_busy(exc):
                raise WebhookRepositoryError(WebhookRepositoryErrorCode.DATABASE_BUSY) from None
            raise

    @asynccontextmanager
    async def _read_connection(self) -> AsyncIterator[object]:
        try:
            async with self._pool.acquire(timeout=5.0) as connection:
                yield connection
        except TimeoutError:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.DATABASE_BUSY) from None
        except Exception as exc:
            if _is_database_busy(exc):
                raise WebhookRepositoryError(WebhookRepositoryErrorCode.DATABASE_BUSY) from None
            raise

    async def get_registration(
        self,
        webhook_id: int,
        *,
        include_deleted: bool = False,
    ) -> WebhookRegistration | None:
        async with self._read_connection() as connection:
            unit = AdminWebhookUnitOfWork(connection, is_postgres=self.is_postgres)
            stored = await unit.get_protected_registration(
                webhook_id,
                include_deleted=include_deleted,
            )
            return stored.registration if stored is not None else None

    async def get_protected_registration(
        self,
        webhook_id: int,
        *,
        include_deleted: bool = False,
    ) -> StoredWebhookRegistration | None:
        async with self._read_connection() as connection:
            unit = AdminWebhookUnitOfWork(connection, is_postgres=self.is_postgres)
            return await unit.get_protected_registration(
                webhook_id,
                include_deleted=include_deleted,
            )

    async def list_registrations(
        self,
        *,
        limit: int,
        before_id: int | None = None,
        offset: int = 0,
        include_deleted: bool = False,
    ) -> list[WebhookRegistration]:
        async with self._read_connection() as connection:
            unit = AdminWebhookUnitOfWork(connection, is_postgres=self.is_postgres)
            return await unit.list_registrations(
                limit=limit,
                before_id=before_id,
                offset=offset,
                include_deleted=include_deleted,
            )

    async def count_registrations(self) -> int:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).count_registrations()

    async def count_active_registrations(self) -> int:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).count_active_registrations()

    async def count_secret_rotation_required(self) -> int:
        """Return non-deleted registrations awaiting canonical secret rotation."""
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).count_secret_rotation_required()

    async def registration_limit_state(self, *, limit: int) -> RegistrationLimitState:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).registration_limit_state(limit=limit)

    async def active_registration_limit_state(
        self,
        *,
        limit: int,
    ) -> RegistrationLimitState:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).active_registration_limit_state(limit=limit)

    async def get_migration_state(self) -> MigrationState:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).get_migration_state(lock=False)

    async def get_legacy_import_snapshot(self) -> LegacyImportDatabaseSnapshot:
        """Read legacy rows and canonical ID-allocation state without mutation."""
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).get_legacy_import_snapshot(lock=False)

    async def page_protected_rows(
        self,
        *,
        table: str,
        after: str | None,
        limit: int,
        inventory_at: datetime | None = None,
    ) -> list[ProtectedRow]:
        """Read a bounded page, optionally freezing replay expiry eligibility."""
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).page_protected_rows(
                table=table,
                after=after,
                limit=limit,
                inventory_at=inventory_at,
            )

    async def find_purge_eligible_registration_ids(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> list[int]:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).find_purge_eligible_registration_ids(now=now, limit=limit)


class AdminWebhookUnitOfWork(_ConnectionAdapter):
    """Transaction-bound operations shared by both supported database backends."""

    async def allocate_registration_id(self) -> int:
        row = await self._fetchrow(
            """
            UPDATE admin_webhook_sequences
            SET next_value = next_value + 1
            WHERE name = ?
            RETURNING next_value - 1 AS allocated_id
            """,
            ("registration",),
        )
        if row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.SEQUENCE_UNAVAILABLE)
        return int(row["allocated_id"])

    async def ensure_registration_sequence_above(self, webhook_id: int) -> int:
        """Advance the allocator past one imported ID without moving it backward."""
        if isinstance(webhook_id, bool) or not 1 <= webhook_id < 2**63 - 1:
            raise ValueError("imported webhook ID cannot be followed by a positive ID")
        next_value = webhook_id + 1
        row = await self._fetchrow(
            """
            UPDATE admin_webhook_sequences
            SET next_value = CASE
                WHEN next_value < ? THEN ?
                ELSE next_value
            END
            WHERE name = ?
            RETURNING next_value
            """,
            (next_value, next_value, "registration"),
        )
        if row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.SEQUENCE_UNAVAILABLE)
        return int(row["next_value"])

    async def get_legacy_import_snapshot(
        self,
        *,
        lock: bool,
    ) -> LegacyImportDatabaseSnapshot:
        """Read one fixed-table legacy snapshot for planning or import validation."""
        if self._is_postgres:
            table_row = await self._fetchrow(
                "SELECT to_regclass(?) AS table_name",
                ("public.admin_webhooks",),
            )
            table_present = bool(table_row and table_row["table_name"] is not None)
        else:
            table_row = await self._fetchrow(
                """
                SELECT name AS table_name
                FROM sqlite_master
                WHERE type = ? AND name = ?
                """,
                ("table", "admin_webhooks"),
            )
            table_present = table_row is not None

        raw_rows: list[dict[str, Any]] = []
        if table_present:
            legacy_query = "SELECT * FROM admin_webhooks ORDER BY id ASC"
            if lock and self._is_postgres:
                legacy_query += " FOR SHARE"
            raw_rows = await self._fetch(legacy_query)

        canonical_query = "SELECT id, deleted_at FROM admin_webhook_registrations ORDER BY id ASC"
        if lock and self._is_postgres:
            canonical_query += " FOR SHARE"
        canonical_rows = await self._fetch(canonical_query)

        sequence_query = (
            "SELECT next_value FROM admin_webhook_sequences WHERE name = ?"
        )
        if lock and self._is_postgres:
            sequence_query += " FOR UPDATE"
        sequence_row = await self._fetchrow(sequence_query, ("registration",))
        if sequence_row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.SEQUENCE_UNAVAILABLE)

        rows: list[LegacyWebhookRow] = []
        for position, raw_row in enumerate(raw_rows):
            raw_identity = raw_row.get("id")
            if isinstance(raw_identity, bool):
                source_identity = "true" if raw_identity else "false"
            elif raw_identity is None:
                source_identity = "null"
            elif isinstance(raw_identity, (int, str)):
                source_identity = str(raw_identity)
            else:
                source_identity = f"row-{position + 1}"
            rows.append(
                LegacyWebhookRow(
                    source_identity=source_identity,
                    values=raw_row,
                )
            )

        canonical_ids = tuple(int(row["id"]) for row in canonical_rows)
        return LegacyImportDatabaseSnapshot(
            table_present=table_present,
            rows=tuple(rows),
            canonical_registration_ids=canonical_ids,
            canonical_non_deleted_count=sum(row["deleted_at"] is None for row in canonical_rows),
            next_registration_id=int(sequence_row["next_value"]),
        )

    async def insert_registration(
        self,
        registration: RegistrationInsert,
    ) -> WebhookRegistration:
        _utc_datetime(registration.now, field="now")
        _validate_registration_target(registration.target)
        if registration.id < 1 or registration.actor_user_id < 1:
            raise ValueError("registration and actor IDs must be positive")
        row = await self._fetchrow(
            """
            INSERT INTO admin_webhook_registrations (
                id, description, target_ciphertext_json, target_key_id,
                target_hostname, target_display, event_types_json, active,
                timeout_seconds, secret_ciphertext_json, secret_key_id,
                secret_rotation_required, created_by_user_id,
                updated_by_user_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
            """,
            (
                registration.id,
                registration.description,
                registration.target.protected.ciphertext_json,
                registration.target.protected.key_id,
                registration.target.hostname,
                registration.target.display,
                _canonical_json(list(registration.event_types)),
                registration.active,
                registration.timeout_seconds,
                registration.secret.ciphertext_json,
                registration.secret.key_id,
                registration.secret_rotation_required,
                registration.actor_user_id,
                registration.actor_user_id,
                registration.now,
                registration.now,
            ),
        )
        if row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.NOT_FOUND)
        return _registration_from_row(row)

    async def _registration_row(
        self,
        webhook_id: int,
        *,
        include_deleted: bool,
        lock: bool = False,
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM admin_webhook_registrations WHERE id = ?"
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        if lock and self._is_postgres:
            query += " FOR UPDATE"
        return await self._fetchrow(query, (webhook_id,))

    async def get_protected_registration(
        self,
        webhook_id: int,
        *,
        include_deleted: bool = False,
        lock: bool = False,
    ) -> StoredWebhookRegistration | None:
        row = await self._registration_row(
            webhook_id,
            include_deleted=include_deleted,
            lock=lock,
        )
        return _stored_registration_from_row(row) if row is not None else None

    async def list_registrations(
        self,
        *,
        limit: int,
        before_id: int | None = None,
        offset: int = 0,
        include_deleted: bool = False,
    ) -> list[WebhookRegistration]:
        if not 1 <= limit <= _MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {_MAX_PAGE_SIZE}")
        if not 0 <= offset <= 1_000:
            raise ValueError("offset must be between 0 and 1000")
        clauses: list[str] = []
        params: list[object] = []
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        if before_id is not None:
            if before_id < 1:
                raise ValueError("before_id must be positive")
            clauses.append("id < ?")
            params.append(before_id)
        query = "SELECT * FROM admin_webhook_registrations"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend((limit, offset))
        return [_registration_from_row(row) for row in await self._fetch(query, params)]

    async def count_registrations(self) -> int:
        row = await self._fetchrow(
            """
            SELECT COUNT(*) AS count
            FROM admin_webhook_registrations
            WHERE deleted_at IS NULL
            """
        )
        return int(row["count"]) if row is not None else 0

    async def count_active_registrations(self) -> int:
        row = await self._fetchrow(
            """
            SELECT COUNT(*) AS count
            FROM admin_webhook_registrations
            WHERE deleted_at IS NULL AND active = ?
            """,
            (True,),
        )
        return int(row["count"]) if row is not None else 0

    async def count_secret_rotation_required(self) -> int:
        row = await self._fetchrow(
            """
            SELECT COUNT(*) AS count
            FROM admin_webhook_registrations
            WHERE deleted_at IS NULL AND secret_rotation_required = ?
            """,
            (True,),
        )
        return int(row["count"]) if row is not None else 0

    async def page_protected_rows(
        self,
        *,
        table: str,
        after: str | None,
        limit: int,
        inventory_at: datetime | None = None,
    ) -> list[ProtectedRow]:
        """Return a deterministic page from one explicitly known table/field."""
        if table not in _DATABASE_PROTECTED_TABLES:
            raise ValueError("protected row table is invalid")
        if not 1 <= limit <= _MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {_MAX_PAGE_SIZE}")
        if inventory_at is not None:
            inventory_at = _utc_datetime(inventory_at, field="inventory_at")

        if table in {
            "registration_targets",
            "registration_secrets",
            "idempotency_replay_secrets",
        }:
            if after is None:
                after_id = 0
            elif not isinstance(after, str) or not after.isdigit() or int(after) < 1:
                raise ValueError("protected row cursor is invalid")
            else:
                after_id = int(after)

        if table == "registration_targets":
            rows = await self._fetch(
                """
                SELECT id, target_ciphertext_json, target_key_id, target_version
                FROM admin_webhook_registrations
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (after_id, limit),
            )
            return [
                ProtectedRow(
                    table=table,
                    row_identity=str(row["id"]),
                    field="target",
                    protected=ProtectedValue(
                        ciphertext_json=str(row["target_ciphertext_json"]),
                        key_id=str(row["target_key_id"]),
                    ),
                    purpose="registration.target",
                    envelope_identity={
                        "registration_id": int(row["id"]),
                        "target_version": int(row["target_version"]),
                    },
                )
                for row in rows
            ]

        if table == "registration_secrets":
            rows = await self._fetch(
                """
                SELECT id, secret_ciphertext_json, secret_key_id, secret_version
                FROM admin_webhook_registrations
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (after_id, limit),
            )
            return [
                ProtectedRow(
                    table=table,
                    row_identity=str(row["id"]),
                    field="secret",
                    protected=ProtectedValue(
                        ciphertext_json=str(row["secret_ciphertext_json"]),
                        key_id=str(row["secret_key_id"]),
                    ),
                    purpose="registration.secret",
                    envelope_identity={
                        "registration_id": int(row["id"]),
                        "secret_version": int(row["secret_version"]),
                    },
                )
                for row in rows
            ]

        if table == "event_bodies":
            if after is not None and (not isinstance(after, str) or not after):
                raise ValueError("protected row cursor is invalid")
            rows = await self._fetch(
                """
                SELECT id, api_version, body_ciphertext_json, body_key_id
                FROM admin_webhook_events
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (after or "", limit),
            )
            return [
                ProtectedRow(
                    table=table,
                    row_identity=str(row["id"]),
                    field="body",
                    protected=ProtectedValue(
                        ciphertext_json=str(row["body_ciphertext_json"]),
                        key_id=str(row["body_key_id"]),
                    ),
                    purpose="event.body",
                    envelope_identity={
                        "event_id": str(row["id"]),
                        "api_version": str(row["api_version"]),
                    },
                )
                for row in rows
            ]

        if inventory_at is None:
            rows = await self._fetch(
                """
                SELECT id, lookup_digest, resource_id, secret_version,
                       replay_secret_ciphertext_json, replay_secret_key_id
                FROM admin_webhook_idempotency
                WHERE id > ?
                  AND state = 'completed'
                  AND replay_secret_ciphertext_json IS NOT NULL
                  AND replay_secret_key_id IS NOT NULL
                  AND resource_id IS NOT NULL
                  AND secret_version IS NOT NULL
                  AND expires_at > CURRENT_TIMESTAMP
                ORDER BY id ASC
                LIMIT ?
                """,
                (after_id, limit),
            )
        else:
            rows = await self._fetch(
                """
                SELECT id, lookup_digest, resource_id, secret_version,
                       replay_secret_ciphertext_json, replay_secret_key_id
                FROM admin_webhook_idempotency
                WHERE id > ?
                  AND state = 'completed'
                  AND replay_secret_ciphertext_json IS NOT NULL
                  AND replay_secret_key_id IS NOT NULL
                  AND resource_id IS NOT NULL
                  AND secret_version IS NOT NULL
                  AND expires_at > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (after_id, inventory_at, limit),
            )
        return [
            ProtectedRow(
                table=table,
                row_identity=str(row["id"]),
                field="replay_secret",
                protected=ProtectedValue(
                    ciphertext_json=str(row["replay_secret_ciphertext_json"]),
                    key_id=str(row["replay_secret_key_id"]),
                ),
                purpose="idempotency.secret_replay",
                envelope_identity={
                    "lookup_digest": str(row["lookup_digest"]),
                    "registration_id": int(row["resource_id"]),
                    "secret_version": int(row["secret_version"]),
                },
            )
            for row in rows
        ]

    async def replace_protected_value(
        self,
        row: ProtectedRow,
        *,
        expected_ciphertext: str,
        replacement: ProtectedValue,
    ) -> bool:
        """Replace one known envelope only if its full persisted identity matches."""
        if not isinstance(row, ProtectedRow) or not isinstance(replacement, ProtectedValue):
            raise TypeError("protected row and replacement are required")
        if not isinstance(expected_ciphertext, str) or not hmac.compare_digest(
            expected_ciphertext,
            row.protected.ciphertext_json,
        ):
            raise ValueError("expected protected ciphertext is invalid")

        identity = row.envelope_identity
        if row.table == "registration_targets":
            changed = await self._execute(
                """
                UPDATE admin_webhook_registrations
                SET target_ciphertext_json = ?, target_key_id = ?
                WHERE id = ?
                  AND target_version = ?
                  AND target_ciphertext_json = ?
                  AND target_key_id = ?
                """,
                (
                    replacement.ciphertext_json,
                    replacement.key_id,
                    int(row.row_identity),
                    identity["target_version"],
                    expected_ciphertext,
                    row.protected.key_id,
                ),
            )
        elif row.table == "registration_secrets":
            changed = await self._execute(
                """
                UPDATE admin_webhook_registrations
                SET secret_ciphertext_json = ?, secret_key_id = ?
                WHERE id = ?
                  AND secret_version = ?
                  AND secret_ciphertext_json = ?
                  AND secret_key_id = ?
                """,
                (
                    replacement.ciphertext_json,
                    replacement.key_id,
                    int(row.row_identity),
                    identity["secret_version"],
                    expected_ciphertext,
                    row.protected.key_id,
                ),
            )
        elif row.table == "event_bodies":
            changed = await self._execute(
                """
                UPDATE admin_webhook_events
                SET body_ciphertext_json = ?, body_key_id = ?
                WHERE id = ?
                  AND api_version = ?
                  AND body_ciphertext_json = ?
                  AND body_key_id = ?
                """,
                (
                    replacement.ciphertext_json,
                    replacement.key_id,
                    row.row_identity,
                    identity["api_version"],
                    expected_ciphertext,
                    row.protected.key_id,
                ),
            )
        else:
            changed = await self._execute(
                """
                UPDATE admin_webhook_idempotency
                SET replay_secret_ciphertext_json = ?, replay_secret_key_id = ?
                WHERE id = ?
                  AND lookup_digest = ?
                  AND resource_id = ?
                  AND secret_version = ?
                  AND replay_secret_ciphertext_json = ?
                  AND replay_secret_key_id = ?
                """,
                (
                    replacement.ciphertext_json,
                    replacement.key_id,
                    int(row.row_identity),
                    identity["lookup_digest"],
                    identity["registration_id"],
                    identity["secret_version"],
                    expected_ciphertext,
                    row.protected.key_id,
                ),
            )
        if changed not in {0, 1}:
            raise RuntimeError("protected value compare-and-set affected multiple rows")
        return changed == 1

    async def _lock_registration_admission(self) -> None:
        query = (
            "SELECT next_value FROM admin_webhook_sequences WHERE name = ? FOR UPDATE"
            if self._is_postgres
            else "SELECT next_value FROM admin_webhook_sequences WHERE name = ?"
        )
        if await self._fetchrow(query, ("registration",)) is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.SEQUENCE_UNAVAILABLE)

    async def registration_limit_state(
        self,
        *,
        limit: int,
    ) -> RegistrationLimitState:
        if limit < 1:
            raise ValueError("limit must be positive")
        current = await self.count_registrations()
        return RegistrationLimitState(
            current=current,
            limit=limit,
            at_limit=current >= limit,
            over_limit=current > limit,
        )

    async def active_registration_limit_state(
        self,
        *,
        limit: int,
    ) -> RegistrationLimitState:
        if limit < 1:
            raise ValueError("limit must be positive")
        current = await self.count_active_registrations()
        return RegistrationLimitState(
            current=current,
            limit=limit,
            at_limit=current >= limit,
            over_limit=current > limit,
        )

    async def enforce_registration_limit(self, *, limit: int) -> RegistrationLimitState:
        """Serialize admission and reject create when no registration slot remains."""
        await self._lock_registration_admission()
        state = await self.registration_limit_state(limit=limit)
        if state.at_limit:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.REGISTRATION_LIMIT)
        return state

    async def enforce_active_registration_limit(
        self,
        *,
        limit: int,
    ) -> RegistrationLimitState:
        """Serialize admission and reject activation when no active slot remains."""
        await self._lock_registration_admission()
        state = await self.active_registration_limit_state(limit=limit)
        if state.at_limit:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.ACTIVE_LIMIT)
        return state

    async def patch_registration(
        self,
        webhook_id: int,
        *,
        expected_revision: int,
        patch: RegistrationPatch,
        actor_user_id: int,
        at: datetime,
    ) -> RegistrationPatchResult:
        at = _utc_datetime(at, field="at")
        if webhook_id < 1 or expected_revision < 1 or actor_user_id < 1:
            raise ValueError("IDs and expected revision must be positive")
        current_row = await self._registration_row(
            webhook_id,
            include_deleted=False,
            lock=True,
        )
        if current_row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.NOT_FOUND)
        current = _stored_registration_from_row(current_row)
        if current.registration.revision != expected_revision:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.STALE_REVISION)

        assignments: list[str] = []
        params: list[object] = []
        changes_delivery_config = False
        changes_target = False
        changes_secret = False

        def assign(column: str, value: object) -> None:
            assignments.append(f"{column} = ?")
            params.append(value)

        if isinstance(patch.description, str):
            if patch.description != current.registration.description:
                assign("description", patch.description)
        elif patch.description is not UNSET:
            raise TypeError("description patch is invalid")

        if isinstance(patch.target, RegistrationTarget):
            _validate_registration_target(patch.target)
            target_changed = (
                patch.target.protected != current.target
                or patch.target.hostname != current.registration.target_hostname
                or patch.target.display != current.registration.target_display
            )
            if target_changed:
                assign("target_ciphertext_json", patch.target.protected.ciphertext_json)
                assign("target_key_id", patch.target.protected.key_id)
                assign("target_hostname", patch.target.hostname)
                assign("target_display", patch.target.display)
                changes_delivery_config = True
                changes_target = True
        elif patch.target is not UNSET:
            raise TypeError("target patch is invalid")

        if isinstance(patch.event_types, tuple):
            if any(not isinstance(item, str) for item in patch.event_types):
                raise TypeError("event types patch is invalid")
            if patch.event_types != current.registration.event_types:
                assign("event_types_json", _canonical_json(list(patch.event_types)))
                changes_delivery_config = True
        elif patch.event_types is not UNSET:
            raise TypeError("event types patch is invalid")

        if isinstance(patch.active, bool):
            if patch.active != current.registration.active:
                assign("active", patch.active)
                changes_delivery_config = True
        elif patch.active is not UNSET:
            raise TypeError("active patch is invalid")

        if isinstance(patch.timeout_seconds, int) and not isinstance(
            patch.timeout_seconds,
            bool,
        ):
            if patch.timeout_seconds != current.registration.timeout_seconds:
                assign("timeout_seconds", patch.timeout_seconds)
                changes_delivery_config = True
        elif patch.timeout_seconds is not UNSET:
            raise TypeError("timeout patch is invalid")

        if isinstance(patch.secret, ProtectedValue):
            if patch.secret != current.secret:
                assign("secret_ciphertext_json", patch.secret.ciphertext_json)
                assign("secret_key_id", patch.secret.key_id)
                changes_delivery_config = True
                changes_secret = True
        elif patch.secret is not UNSET:
            raise TypeError("secret patch is invalid")

        if isinstance(patch.secret_rotation_required, bool):
            if patch.secret_rotation_required != current.registration.secret_rotation_required:
                if patch.secret_rotation_required is False and not changes_secret:
                    raise ValueError("secret rotation requirement cannot be cleared without a new secret")
                assign("secret_rotation_required", patch.secret_rotation_required)
        elif patch.secret_rotation_required is not UNSET:
            raise TypeError("secret rotation patch is invalid")

        if not assignments:
            return RegistrationPatchResult(
                changed=False,
                registration=current.registration,
            )

        assignments.extend(["revision = revision + 1", "updated_by_user_id = ?", "updated_at = ?"])
        params.extend([actor_user_id, at])
        if changes_delivery_config:
            assignments.append("delivery_config_version = delivery_config_version + 1")
        if changes_target:
            assignments.append("target_version = target_version + 1")
        if changes_secret:
            assignments.append("secret_version = secret_version + 1")
        params.extend([webhook_id, expected_revision])
        # Identifiers are selected only by the closed patch fields above.
        update_query = " ".join(
            (
                "UPDATE admin_webhook_registrations SET",
                ", ".join(assignments),
                "WHERE id = ? AND revision = ? AND deleted_at IS NULL RETURNING *",
            )
        )
        row = await self._fetchrow(update_query, params)
        if row is None:
            latest = await self._registration_row(
                webhook_id,
                include_deleted=False,
                lock=False,
            )
            code = WebhookRepositoryErrorCode.NOT_FOUND if latest is None else WebhookRepositoryErrorCode.STALE_REVISION
            raise WebhookRepositoryError(code)
        return RegistrationPatchResult(
            changed=True,
            registration=_registration_from_row(row),
        )

    async def soft_delete_registration(
        self,
        webhook_id: int,
        *,
        expected_revision: int,
        actor_user_id: int,
        at: datetime,
    ) -> WebhookRegistration:
        at = _utc_datetime(at, field="at")
        if webhook_id < 1 or expected_revision < 1 or actor_user_id < 1:
            raise ValueError("IDs and expected revision must be positive")
        row = await self._fetchrow(
            """
            UPDATE admin_webhook_registrations
            SET active = ?, deleted_at = ?, deleted_by_user_id = ?,
                updated_by_user_id = ?, updated_at = ?, revision = revision + 1,
                delivery_config_version = delivery_config_version + 1
            WHERE id = ? AND revision = ? AND deleted_at IS NULL
            RETURNING *
            """,
            (
                False,
                at,
                actor_user_id,
                actor_user_id,
                at,
                webhook_id,
                expected_revision,
            ),
        )
        if row is None:
            latest = await self._registration_row(
                webhook_id,
                include_deleted=False,
                lock=False,
            )
            code = WebhookRepositoryErrorCode.NOT_FOUND if latest is None else WebhookRepositoryErrorCode.STALE_REVISION
            raise WebhookRepositoryError(code)
        return _registration_from_row(row)

    async def claim_idempotency(
        self,
        *,
        lookup_digest: str,
        scope: IdempotencyScope,
        request_fingerprint: str,
        now: datetime,
        expires_at: datetime,
    ) -> IdempotencyLookup:
        now = _utc_datetime(now, field="now")
        expires_at = _utc_datetime(expires_at, field="expires_at")
        if expires_at <= now:
            raise ValueError("expires_at must be after now")
        if _LOOKUP_DIGEST.fullmatch(lookup_digest) is None:
            raise ValueError("lookup digest is invalid")
        if _REQUEST_FINGERPRINT.fullmatch(request_fingerprint) is None:
            raise ValueError("request fingerprint is invalid")
        lookup_query = (
            "SELECT * FROM admin_webhook_idempotency WHERE lookup_digest = ? FOR UPDATE"
            if self._is_postgres
            else "SELECT * FROM admin_webhook_idempotency WHERE lookup_digest = ?"
        )
        row = await self._fetchrow(
            lookup_query,
            (lookup_digest,),
        )
        if row is not None:
            row_expires_at = _parse_datetime(row["expires_at"])
            if row_expires_at is None:
                raise ValueError("persisted idempotency expiry is invalid")
            if row_expires_at <= now:
                await self._execute(
                    "DELETE FROM admin_webhook_idempotency WHERE id = ?",
                    (row["id"],),
                )
                row = None
        if row is None:
            row = await self._fetchrow(
                """
                INSERT INTO admin_webhook_idempotency (
                    lookup_digest, actor_id, operation, route, webhook_id,
                    delivery_id, request_fingerprint, state, created_at,
                    updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'in_progress', ?, ?, ?)
                ON CONFLICT (lookup_digest) DO NOTHING
                RETURNING *
                """,
                (
                    lookup_digest,
                    scope.actor_id,
                    scope.operation,
                    scope.route,
                    scope.webhook_id,
                    scope.delivery_id,
                    request_fingerprint,
                    now,
                    now,
                    expires_at,
                ),
            )
            if row is not None:
                return IdempotencyLookup(
                    kind=IdempotencyLookupKind.NEW,
                    expires_at=expires_at,
                )
            row = await self._fetchrow(
                lookup_query,
                (lookup_digest,),
            )
        if row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.IDEMPOTENCY_NOT_CLAIMED)
        return await self._idempotency_lookup_from_row(
            row,
            scope=scope,
            request_fingerprint=request_fingerprint,
        )

    async def _idempotency_lookup_from_row(
        self,
        row: Mapping[str, Any],
        *,
        scope: IdempotencyScope,
        request_fingerprint: str,
    ) -> IdempotencyLookup:
        scope_matches = (
            hmac.compare_digest(str(row["actor_id"]), scope.actor_id)
            and hmac.compare_digest(str(row["operation"]), scope.operation)
            and hmac.compare_digest(str(row["route"]), scope.route)
            and row["webhook_id"] == scope.webhook_id
            and row["delivery_id"] == scope.delivery_id
        )
        if not scope_matches or not hmac.compare_digest(
            str(row["request_fingerprint"]),
            request_fingerprint,
        ):
            return IdempotencyLookup(kind=IdempotencyLookupKind.CONFLICT)
        if str(row["state"]) == "in_progress":
            return IdempotencyLookup(
                kind=IdempotencyLookupKind.IN_PROGRESS,
                expires_at=_parse_datetime(row["expires_at"]),
            )

        replay_secret = None
        if row["replay_secret_ciphertext_json"] is not None:
            replay_secret = ProtectedValue(
                ciphertext_json=str(row["replay_secret_ciphertext_json"]),
                key_id=str(row["replay_secret_key_id"]),
            )
        metadata: Mapping[str, object] | None = None
        if row["response_metadata_json"] is not None:
            metadata = MappingProxyType(_strict_json_object(row["response_metadata_json"]))
        resource_id = int(row["resource_id"]) if row["resource_id"] is not None else None
        resource_version = int(row["resource_version"]) if row["resource_version"] is not None else None
        secret_version = int(row["secret_version"]) if row["secret_version"] is not None else None
        superseded = False
        if replay_secret is not None and resource_id is not None:
            current = await self._registration_row(
                resource_id,
                include_deleted=True,
                lock=False,
            )
            superseded = (
                current is None
                or current["deleted_at"] is not None
                or (secret_version is not None and int(current["secret_version"]) != secret_version)
            )
        return IdempotencyLookup(
            kind=IdempotencyLookupKind.REPLAY,
            resource_id=resource_id,
            resource_version=resource_version,
            secret_version=secret_version,
            replay_secret=replay_secret,
            test_delivery_id=(str(row["test_delivery_id"]) if row["test_delivery_id"] is not None else None),
            test_attempt_id=(str(row["test_attempt_id"]) if row["test_attempt_id"] is not None else None),
            response_status=(int(row["response_status"]) if row["response_status"] is not None else None),
            response_metadata=metadata,
            resource_superseded=superseded,
            expires_at=_parse_datetime(row["expires_at"]),
        )

    async def complete_idempotency(
        self,
        *,
        lookup_digest: str,
        request_fingerprint: str,
        resource_id: int | None,
        resource_version: int | None,
        secret_version: int | None = None,
        replay_secret: ProtectedValue | None = None,
        test_delivery_id: str | None = None,
        test_attempt_id: str | None = None,
        response_status: int,
        response_metadata: Mapping[str, object] | None,
        at: datetime,
    ) -> IdempotencyLookup:
        at = _utc_datetime(at, field="at")
        metadata_json, _ = _safe_response_metadata(response_metadata)
        if _LOOKUP_DIGEST.fullmatch(lookup_digest) is None:
            raise ValueError("lookup digest is invalid")
        if _REQUEST_FINGERPRINT.fullmatch(request_fingerprint) is None:
            raise ValueError("request fingerprint is invalid")
        if (resource_id is None) != (resource_version is None):
            raise ValueError("resource ID and version must be paired")
        if resource_id is not None:
            if resource_version is None or resource_id < 1 or resource_version < 1:
                raise ValueError("resource ID and version must be positive")
        if not 100 <= response_status <= 599:
            raise ValueError("response status is invalid")
        if (replay_secret is None) != (secret_version is None):
            raise ValueError("replay secret and secret version must be paired")
        if (test_delivery_id is None) != (test_attempt_id is None):
            raise ValueError("test delivery and attempt IDs must be paired")
        row = await self._fetchrow(
            """
            UPDATE admin_webhook_idempotency
            SET state = 'completed', resource_id = ?, resource_version = ?,
                secret_version = ?, replay_secret_ciphertext_json = ?,
                replay_secret_key_id = ?, test_delivery_id = ?,
                test_attempt_id = ?, response_status = ?,
                response_metadata_json = ?, updated_at = ?
            WHERE lookup_digest = ? AND request_fingerprint = ?
              AND state = 'in_progress'
            RETURNING *
            """,
            (
                resource_id,
                resource_version,
                secret_version,
                replay_secret.ciphertext_json if replay_secret is not None else None,
                replay_secret.key_id if replay_secret is not None else None,
                test_delivery_id,
                test_attempt_id,
                response_status,
                metadata_json,
                at,
                lookup_digest,
                request_fingerprint,
            ),
        )
        if row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.IDEMPOTENCY_COMPLETION_INVALID)
        scope = IdempotencyScope(
            actor_id=str(row["actor_id"]),
            operation=str(row["operation"]),
            route=str(row["route"]),
            webhook_id=(int(row["webhook_id"]) if row["webhook_id"] is not None else None),
            delivery_id=(str(row["delivery_id"]) if row["delivery_id"] is not None else None),
        )
        return await self._idempotency_lookup_from_row(
            row,
            scope=scope,
            request_fingerprint=request_fingerprint,
        )

    async def get_migration_state(self, *, lock: bool = False) -> MigrationState:
        query = "SELECT * FROM admin_webhook_migration_state WHERE singleton_id = 1"
        if lock and self._is_postgres:
            query += " FOR UPDATE"
        row = await self._fetchrow(query)
        if row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.MIGRATION_STATE_UNAVAILABLE)
        return _migration_state_from_row(row)

    async def lock_migration_state(self) -> MigrationState:
        return await self.get_migration_state(lock=True)

    async def compare_and_set_migration_state(
        self,
        *,
        expected_revision: int,
        updates: Mapping[str, object],
        at: datetime,
    ) -> MigrationState:
        at = _utc_datetime(at, field="at")
        if expected_revision < 1 or not updates:
            raise ValueError("expected revision and updates are required")
        invalid = set(updates) - _MIGRATION_MUTABLE_COLUMNS
        if invalid:
            raise ValueError("migration state update contains invalid fields")
        assignments: list[str] = []
        params: list[object] = []
        for column in sorted(updates):
            value = updates[column]
            expected_type = _MIGRATION_JSON_COLUMNS.get(column)
            if expected_type is not None:
                if not isinstance(value, expected_type):
                    raise ValueError(f"{column} has an invalid JSON shape")
                value = _canonical_json(value)
            assignments.append(f"{column} = ?")
            params.append(value)
        assignments.extend(["state_revision = state_revision + 1", "updated_at = ?"])
        params.extend([at, expected_revision])
        # Identifiers are admitted only from _MIGRATION_MUTABLE_COLUMNS.
        update_query = " ".join(
            (
                "UPDATE admin_webhook_migration_state SET",
                ", ".join(assignments),
                "WHERE singleton_id = 1 AND state_revision = ? RETURNING *",
            )
        )
        row = await self._fetchrow(update_query, params)
        if row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.STALE_MIGRATION_STATE)
        return _migration_state_from_row(row)

    async def mark_first_canonical_activity(
        self,
        kind: str,
        at: datetime,
    ) -> MigrationState:
        if kind not in _ACTIVITY_KINDS:
            raise ValueError("activity kind is invalid")
        at = _utc_datetime(at, field="at")
        row = await self._fetchrow(
            """
            UPDATE admin_webhook_migration_state
            SET first_canonical_activity_at = ?,
                first_canonical_activity_kind = ?,
                state_revision = state_revision + 1,
                updated_at = ?
            WHERE singleton_id = 1
              AND first_canonical_activity_at IS NULL
              AND first_canonical_activity_kind IS NULL
            RETURNING *
            """,
            (at, kind, at),
        )
        if row is not None:
            return _migration_state_from_row(row)
        return await self.get_migration_state(lock=True)

    async def find_purge_eligible_registration_ids(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> list[int]:
        now = _utc_datetime(now, field="now")
        if not 1 <= limit <= _MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {_MAX_PAGE_SIZE}")
        state = await self.get_migration_state(lock=False)
        referenced = _referenced_registration_ids(state.source_mapping)
        cutoff = now - timedelta(days=_MIN_TOMBSTONE_RETENTION_DAYS)
        result: list[int] = []
        after_id = 0
        batch_size = min(_MAX_PAGE_SIZE, max(50, limit * 2))
        while len(result) < limit:
            rows = await self._fetch(
                """
                SELECT registration.id
                FROM admin_webhook_registrations AS registration
                WHERE registration.id > ?
                  AND registration.deleted_at IS NOT NULL
                  AND registration.deleted_at <= ?
                  AND NOT EXISTS (
                      SELECT 1 FROM admin_webhook_deliveries AS delivery
                      WHERE delivery.webhook_id = registration.id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM admin_webhook_idempotency AS command
                      WHERE command.expires_at > ?
                        AND (
                            command.webhook_id = registration.id
                            OR command.resource_id = registration.id
                        )
                  )
                ORDER BY registration.id ASC
                LIMIT ?
                """,
                (after_id, cutoff, now, batch_size),
            )
            if not rows:
                break
            for row in rows:
                webhook_id = int(row["id"])
                if webhook_id not in referenced:
                    result.append(webhook_id)
                    if len(result) == limit:
                        break
            after_id = int(rows[-1]["id"])
            if len(rows) < batch_size:
                break
        return result


__all__ = [
    "AdminWebhookRepository",
    "AdminWebhookUnitOfWork",
    "IdempotencyLookup",
    "IdempotencyLookupKind",
    "MigrationState",
    "RegistrationInsert",
    "RegistrationLimitState",
    "RegistrationPatch",
    "RegistrationPatchResult",
    "RegistrationTarget",
    "StoredWebhookRegistration",
    "WebhookRepositoryError",
    "WebhookRepositoryErrorCode",
]

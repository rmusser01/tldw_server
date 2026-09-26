"""SQLite persistence for the Calendar domain."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Generator, Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from tldw_Server_API.app.core.Calendar.constants import (
    CALENDAR_ROLE_OWNER,
    CALENDAR_SOURCE_OWNER_PROVIDER,
    CALENDAR_SOURCE_OWNER_TLDW,
    DEFAULT_SYNC_INTERVAL_MINUTES,
    DEFAULT_SYNC_LOOKAHEAD_DAYS,
    DEFAULT_SYNC_LOOKBACK_DAYS,
)
from tldw_Server_API.app.core.Calendar.errors import (
    CalendarItemNotFound,
    CalendarNotFound,
    CalendarReadOnlyError,
    CalendarSyncError,
    CalendarValidationError,
)

_UNSET = object()


def _utcnow_iso() -> str:
    """Return a timezone-aware UTC ISO timestamp for persisted lifecycle fields."""
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def _widened_date_bounds(window_start: str, window_end: str) -> tuple[str, str]:
    """Pad ISO window dates by one day for timezone-safe candidate selection.

    Raise CalendarValidationError when either boundary cannot be parsed.
    """
    try:
        start = datetime.fromisoformat(window_start.replace("Z", "+00:00")).date()
        end = datetime.fromisoformat(window_end.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise CalendarValidationError("Calendar expansion window boundaries must be ISO timestamps") from exc
    return (
        (start - timedelta(days=1)).isoformat(),
        (end + timedelta(days=1)).isoformat(),
    )


def _default_calendar_db_path() -> Path:
    """Resolve the repository-local Calendar SQLite path without creating the file."""
    return Path(__file__).resolve().parents[4] / "Databases" / "calendar.db"


def _json_or_none(value: Any) -> str | None:
    """Preserve null/JSON strings or serialize structured values with stable key order."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _coerce_patch(patch: dict[str, Any] | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a new mutation mapping with keyword fields taking precedence over the patch."""
    merged: dict[str, Any] = {}
    if patch:
        merged.update(patch)
    merged.update(kwargs)
    return merged


@dataclass(frozen=True)
class CalendarRow:
    """Immutable calendar ownership, display policy, and archive-state snapshot."""

    id: int
    tenant_id: str
    owner_user_id: int
    org_id: int | None
    name: str
    description: str | None
    color: str | None
    timezone: str
    visibility: str
    default_reminder_policy_json: str | None
    rbac_policy_ref: str | None
    archived_at: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CalendarMembershipRow:
    """Immutable role assignment for a calendar principal; principal IDs are stored as text."""

    id: int
    calendar_id: int
    principal_type: str
    principal_id: str
    role: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CalendarItemRow:
    """Immutable item content, provider provenance, raw JSON, and local/remote deletion state."""

    id: int
    calendar_id: int
    kind: str
    source_owner: str
    provider_owned: bool
    title: str
    description: str | None
    location: str | None
    start_at: str | None
    end_at: str | None
    due_at: str | None
    timezone: str | None
    all_day: bool
    status: str
    local_tags_json: str | None
    metadata_json: str | None
    external_binding_id: int | None
    source_uid: str | None
    source_etag: str | None
    source_ctag: str | None
    source_payload_json: str | None
    source_updated_at: str | None
    copied_from_item_id: int | None
    linked_projection_type: str | None
    linked_projection_id: str | None
    deleted_at: str | None
    remote_deleted_at: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CalendarRecurrenceRow:
    """Immutable recurrence rule and raw inclusion/exclusion date JSON for an item."""

    id: int
    calendar_item_id: int
    rrule: str | None
    rdate_json: str | None
    exdate_json: str | None
    timezone: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CalendarAnnotationRow:
    """Immutable authored annotation; empty bodies represent actor-local tag overlays."""

    id: int
    calendar_item_id: int
    author_user_id: int
    body: str
    tags_json: str | None
    deleted_at: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CalendarLinkRow:
    """Immutable item context target and optional display metadata stored as raw JSON."""

    id: int
    calendar_item_id: int
    target_type: str
    target_id: str
    label: str | None
    url: str | None
    metadata_json: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ExternalCalendarAccountRow:
    """Immutable tenant/user provider account state with an opaque ref, never a credential payload."""

    id: int
    tenant_id: str
    user_id: int
    provider: str
    display_name: str
    secret_ref: str | None
    account_metadata_json: str | None
    status: str
    revoked_at: str | None
    deleted_at: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ExternalCalendarBindingRow:
    """Immutable remote-calendar binding, polling policy, sync progress, and deletion state."""

    id: int
    account_id: int
    calendar_id: int
    remote_calendar_id: str
    remote_display_name: str | None
    sync_enabled: bool
    sync_interval_minutes: int | None
    lookback_days: int
    lookahead_days: int
    provider_capabilities_json: str | None
    sync_cursor: str | None
    last_sync_at: str | None
    next_scan_at: str | None
    last_error: str | None
    disabled_at: str | None
    deleted_at: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CalendarSyncEventRow:
    """Immutable sync-history entry with item counters, outcome, and raw diagnostic JSON."""

    id: int
    binding_id: int | None
    account_id: int | None
    event_type: str
    status: str
    started_at: str | None
    finished_at: str | None
    items_seen: int
    items_upserted: int
    items_tombstoned: int
    error_message: str | None
    metadata_json: str | None
    created_at: str


class CalendarSecretStore:
    """Stores encrypted external account secrets behind opaque references."""

    def __init__(self, database: CalendarDatabase) -> None:
        """Share the repository's connections and transactions for secret storage."""
        self._database = database

    def create_secret_ref(
        self,
        *,
        tenant_id: str,
        user_id: int,
        provider: str,
        encrypted_payload: str,
    ) -> str:
        """Persist an already encrypted payload atomically and return a new opaque reference."""
        with self._database.transaction() as conn:
            return self.create_secret_ref_in_connection(
                conn,
                tenant_id=tenant_id,
                user_id=user_id,
                provider=provider,
                encrypted_payload=encrypted_payload,
            )

    def create_secret_ref_in_connection(
        self,
        conn: sqlite3.Connection,
        *,
        tenant_id: str,
        user_id: int,
        provider: str,
        encrypted_payload: str,
    ) -> str:
        """Insert an encrypted payload in the caller's transaction and return its opaque reference."""
        secret_ref = f"calendar_secret_{uuid4().hex}"
        now = _utcnow_iso()
        conn.execute(
            """
            INSERT INTO calendar_external_account_secrets
                (secret_ref, tenant_id, user_id, provider, encrypted_payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (secret_ref, tenant_id, user_id, provider, encrypted_payload, now),
        )
        return secret_ref

    def resolve_secret_ref(self, secret_ref: str) -> str:
        """Return the live encrypted payload; raise CalendarValidationError if the ref is unavailable.

        This unscoped internal lookup requires the caller to establish ownership.
        """
        with self._database.connection() as conn:
            row = self._get_secret_row(conn, secret_ref)
        return str(row["encrypted_payload"])

    def resolve_secret_ref_scoped(
        self,
        secret_ref: str,
        *,
        tenant_id: str,
        user_id: int,
        provider: str,
    ) -> str:
        """Return a live payload matching tenant, user, and provider or raise CalendarValidationError."""
        with self._database.connection() as conn:
            row = self._get_secret_row(
                conn,
                secret_ref,
                tenant_id=tenant_id,
                user_id=user_id,
                provider=provider,
            )
        return str(row["encrypted_payload"])

    def resolve_secret_ref_for_user(
        self,
        secret_ref: str,
        *,
        tenant_id: str,
        user_id: int,
    ) -> str:
        """Return a live payload for the tenant/user or raise CalendarValidationError on a scope mismatch."""
        with self._database.connection() as conn:
            row = self._get_secret_row(
                conn,
                secret_ref,
                tenant_id=tenant_id,
                user_id=user_id,
            )
        return str(row["encrypted_payload"])

    def delete_secret_ref(self, secret_ref: str) -> None:
        """Wipe a live payload and tombstone its ref; missing refs raise CalendarValidationError.

        Ownership must be established before this unscoped internal operation.
        """
        with self._database.transaction() as conn:
            self.delete_secret_ref_in_connection(conn, secret_ref)

    def delete_secret_ref_scoped(
        self,
        secret_ref: str,
        *,
        tenant_id: str,
        user_id: int,
        provider: str,
    ) -> None:
        """Wipe the tenant/user/provider's live secret or raise CalendarValidationError on mismatch."""
        with self._database.transaction() as conn:
            self.delete_secret_ref_in_connection(
                conn,
                secret_ref,
                tenant_id=tenant_id,
                user_id=user_id,
                provider=provider,
            )

    def delete_secret_ref_in_connection(
        self,
        conn: sqlite3.Connection,
        secret_ref: str | None,
        *,
        tenant_id: str | None = None,
        user_id: int | None = None,
        provider: str | None = None,
    ) -> None:
        """Wipe and tombstone a scoped secret within the caller's transaction.

        Null/empty references are no-ops; missing or mismatched live refs raise
        CalendarValidationError before any payload is changed.
        """
        if not secret_ref:
            return
        self._get_secret_row(
            conn,
            secret_ref,
            tenant_id=tenant_id,
            user_id=user_id,
            provider=provider,
        )
        clauses = ["secret_ref = ?", "deleted_at IS NULL"]
        params: list[Any] = [secret_ref]
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if provider is not None:
            clauses.append("provider = ?")
            params.append(provider)
        params.insert(0, _utcnow_iso())
        conn.execute(
            f"""
            UPDATE calendar_external_account_secrets
            SET encrypted_payload = '',
                deleted_at = ?
            WHERE {" AND ".join(clauses)}
            """,  # nosec B608
            tuple(params),
        )

    def validate_secret_ref_in_connection(
        self,
        conn: sqlite3.Connection,
        *,
        secret_ref: str,
        tenant_id: str,
        user_id: int,
        provider: str,
    ) -> None:
        """Require a live tenant/user/provider reference or raise CalendarValidationError."""
        self._get_secret_row(
            conn,
            secret_ref,
            tenant_id=tenant_id,
            user_id=user_id,
            provider=provider,
        )

    def delete_secret_ref_for_user(
        self,
        secret_ref: str,
        *,
        tenant_id: str,
        user_id: int,
    ) -> None:
        """Wipe the tenant/user's live secret or raise CalendarValidationError on mismatch."""
        with self._database.transaction() as conn:
            self.delete_secret_ref_in_connection(
                conn,
                secret_ref,
                tenant_id=tenant_id,
                user_id=user_id,
            )

    def _get_secret_row(
        self,
        conn: sqlite3.Connection,
        secret_ref: str,
        *,
        tenant_id: str | None = None,
        user_id: int | None = None,
        provider: str | None = None,
    ) -> sqlite3.Row:
        """Return a live payload row matching optional scope constraints; fail with CalendarValidationError."""
        clauses = ["secret_ref = ?", "deleted_at IS NULL"]
        params: list[Any] = [secret_ref]
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if provider is not None:
            clauses.append("provider = ?")
            params.append(provider)
        row = conn.execute(
            f"""
            SELECT encrypted_payload
            FROM calendar_external_account_secrets
            WHERE {" AND ".join(clauses)}
            """,  # nosec B608
            tuple(params),
        ).fetchone()
        if row is None:
            raise CalendarValidationError(f"Calendar secret ref not found: {secret_ref}")
        return row


class CalendarDatabase:
    """Repository for calendars, calendar items, and external calendar sync state."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize SQLite schema and secret storage at the supplied or repository-default path."""
        self.db_path = Path(db_path) if db_path is not None else _default_calendar_db_path()
        self._transaction_connection: ContextVar[sqlite3.Connection | None] = ContextVar(
            f"calendar_transaction_{id(self)}", default=None
        )
        self.secret_store = CalendarSecretStore(self)
        self.ensure_schema()

    @contextlib.contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield the active transaction or a temporary row-mapped connection with foreign keys enabled."""
        active = self._transaction_connection.get()
        if active is not None:
            yield active
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    @contextlib.contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield an atomic write scope, using savepoints when nested.

        Commit on success; roll back the applicable scope and re-raise errors.
        The outer transaction owns and closes its connection.
        """
        active = self._transaction_connection.get()
        if active is not None:
            savepoint = f"calendar_{uuid4().hex}"
            active.execute(f"SAVEPOINT {savepoint}")  # nosec B608 - generated hexadecimal identifier
            try:
                yield active
                active.execute(f"RELEASE SAVEPOINT {savepoint}")  # nosec B608
            except Exception:
                active.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")  # nosec B608
                active.execute(f"RELEASE SAVEPOINT {savepoint}")  # nosec B608
                raise
            return
        with self.connection() as conn:
            conn.execute("BEGIN")
            token = self._transaction_connection.set(conn)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                self._transaction_connection.reset(token)

    def ensure_schema(self) -> None:
        """Create missing Calendar tables and indexes without replacing existing data."""
        with self.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS calendars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    owner_user_id INTEGER NOT NULL,
                    org_id INTEGER,
                    name TEXT NOT NULL,
                    description TEXT,
                    color TEXT,
                    timezone TEXT NOT NULL DEFAULT 'UTC',
                    visibility TEXT NOT NULL DEFAULT 'private',
                    default_reminder_policy_json TEXT,
                    rbac_policy_ref TEXT,
                    archived_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_calendars_scope
                    ON calendars(tenant_id, owner_user_id, org_id, archived_at);

                CREATE TABLE IF NOT EXISTS calendar_memberships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    calendar_id INTEGER NOT NULL REFERENCES calendars(id) ON DELETE CASCADE,
                    principal_type TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(calendar_id, principal_type, principal_id)
                );
                CREATE INDEX IF NOT EXISTS idx_calendar_memberships_principal
                    ON calendar_memberships(calendar_id, principal_type, principal_id);

                CREATE TABLE IF NOT EXISTS external_calendar_accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    secret_ref TEXT,
                    account_metadata_json TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    revoked_at TEXT,
                    deleted_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_external_calendar_accounts_user
                    ON external_calendar_accounts(tenant_id, user_id, provider, deleted_at);

                CREATE TABLE IF NOT EXISTS external_calendar_bindings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id INTEGER NOT NULL REFERENCES external_calendar_accounts(id),
                    calendar_id INTEGER NOT NULL REFERENCES calendars(id) ON DELETE CASCADE,
                    remote_calendar_id TEXT NOT NULL,
                    remote_display_name TEXT,
                    sync_enabled INTEGER NOT NULL DEFAULT 1,
                    sync_interval_minutes INTEGER,
                    lookback_days INTEGER NOT NULL,
                    lookahead_days INTEGER NOT NULL,
                    provider_capabilities_json TEXT,
                    sync_cursor TEXT,
                    last_sync_at TEXT,
                    next_scan_at TEXT,
                    last_error TEXT,
                    disabled_at TEXT,
                    deleted_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(account_id, remote_calendar_id)
                );
                CREATE INDEX IF NOT EXISTS idx_external_calendar_bindings_account_sync
                    ON external_calendar_bindings(account_id, sync_enabled);

                CREATE TABLE IF NOT EXISTS calendar_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    calendar_id INTEGER NOT NULL REFERENCES calendars(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    source_owner TEXT NOT NULL,
                    provider_owned INTEGER NOT NULL DEFAULT 0,
                    title TEXT NOT NULL,
                    description TEXT,
                    location TEXT,
                    start_at TEXT,
                    end_at TEXT,
                    due_at TEXT,
                    timezone TEXT,
                    all_day INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'confirmed',
                    local_tags_json TEXT,
                    metadata_json TEXT,
                    external_binding_id INTEGER REFERENCES external_calendar_bindings(id),
                    source_uid TEXT,
                    source_etag TEXT,
                    source_ctag TEXT,
                    source_payload_json TEXT,
                    source_updated_at TEXT,
                    copied_from_item_id INTEGER REFERENCES calendar_items(id) ON DELETE SET NULL,
                    linked_projection_type TEXT,
                    linked_projection_id TEXT,
                    deleted_at TEXT,
                    remote_deleted_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_calendar_items_window
                    ON calendar_items(calendar_id, start_at, end_at, due_at, deleted_at, remote_deleted_at);
                CREATE INDEX IF NOT EXISTS idx_calendar_items_external_source
                    ON calendar_items(external_binding_id, source_uid);
                CREATE UNIQUE INDEX IF NOT EXISTS ux_calendar_items_external_source
                    ON calendar_items(external_binding_id, source_uid);

                CREATE TABLE IF NOT EXISTS calendar_recurrences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    calendar_item_id INTEGER NOT NULL REFERENCES calendar_items(id) ON DELETE CASCADE,
                    rrule TEXT,
                    rdate_json TEXT,
                    exdate_json TEXT,
                    timezone TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_calendar_recurrences_item
                    ON calendar_recurrences(calendar_item_id);

                CREATE TABLE IF NOT EXISTS calendar_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    calendar_item_id INTEGER NOT NULL REFERENCES calendar_items(id) ON DELETE CASCADE,
                    author_user_id INTEGER NOT NULL,
                    body TEXT NOT NULL,
                    tags_json TEXT,
                    deleted_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_calendar_annotations_item
                    ON calendar_annotations(calendar_item_id, deleted_at);

                CREATE TABLE IF NOT EXISTS calendar_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    calendar_item_id INTEGER NOT NULL REFERENCES calendar_items(id) ON DELETE CASCADE,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    label TEXT,
                    url TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_calendar_links_target
                    ON calendar_links(calendar_item_id, target_type, target_id);

                CREATE TABLE IF NOT EXISTS calendar_sync_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    binding_id INTEGER REFERENCES external_calendar_bindings(id),
                    account_id INTEGER REFERENCES external_calendar_accounts(id),
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    items_seen INTEGER NOT NULL DEFAULT 0,
                    items_upserted INTEGER NOT NULL DEFAULT 0,
                    items_tombstoned INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_calendar_sync_events_binding_created
                    ON calendar_sync_events(binding_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_calendar_sync_events_account_created
                    ON calendar_sync_events(account_id, created_at);

                CREATE TABLE IF NOT EXISTS calendar_external_account_secrets (
                    secret_ref TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    encrypted_payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    deleted_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_calendar_account_secrets_user
                    ON calendar_external_account_secrets(tenant_id, user_id, provider, deleted_at);
                """
            )

    def create_calendar(
        self,
        *,
        tenant_id: str,
        owner_user_id: int,
        org_id: int | None,
        name: str,
        timezone: str = "UTC",
        color: str | None = None,
        description: str | None = None,
        visibility: str = "private",
        default_reminder_policy_json: str | dict[str, Any] | None = None,
        rbac_policy_ref: str | None = None,
    ) -> CalendarRow:
        """Create a calendar and its owner membership atomically, returning the persisted row."""
        now = _utcnow_iso()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO calendars (
                    tenant_id, owner_user_id, org_id, name, description, color,
                    timezone, visibility, default_reminder_policy_json, rbac_policy_ref,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    owner_user_id,
                    org_id,
                    name,
                    description,
                    color,
                    timezone,
                    visibility,
                    _json_or_none(default_reminder_policy_json),
                    rbac_policy_ref,
                    now,
                    now,
                ),
            )
            calendar_id = int(cursor.lastrowid)
            conn.execute(
                """
                INSERT INTO calendar_memberships (
                    calendar_id, principal_type, principal_id, role, created_at, updated_at
                )
                VALUES (?, 'user', ?, ?, ?, ?)
                """,
                (calendar_id, str(owner_user_id), CALENDAR_ROLE_OWNER, now, now),
            )
            return self._get_calendar_row(conn, calendar_id)

    def update_calendar(
        self,
        calendar_id: int,
        patch: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CalendarRow:
        """Apply allowed calendar fields and return even archived rows; missing IDs raise CalendarNotFound."""
        updates = _coerce_patch(patch, kwargs)
        with self.transaction() as conn:
            self._apply_update(
                conn,
                "calendars",
                "id",
                calendar_id,
                updates,
                {
                    "name",
                    "description",
                    "color",
                    "timezone",
                    "visibility",
                    "default_reminder_policy_json",
                    "rbac_policy_ref",
                    "archived_at",
                },
            )
            return self._get_calendar_row(conn, calendar_id, include_archived=True)

    def get_calendar(self, calendar_id: int, *, include_archived: bool = False) -> CalendarRow:
        """Return a calendar or raise CalendarNotFound; archived rows are excluded unless requested."""
        with self.connection() as conn:
            return self._get_calendar_row(conn, calendar_id, include_archived=include_archived)

    def list_calendars(
        self,
        *,
        tenant_id: str | None = None,
        include_archived: bool = False,
    ) -> list[CalendarRow]:
        """List calendars ordered by name/ID, optionally filtering tenant and including archived rows."""
        clauses: list[str] = []
        params: list[Any] = []
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if not include_archived:
            clauses.append("archived_at IS NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT * FROM calendars
            {where}
            ORDER BY name ASC, id ASC
            """  # nosec B608
        with self.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._calendar_from_row(row) for row in rows]

    def list_calendars_for_user(
        self,
        *,
        user_id: int,
        tenant_id: str | None = None,
        include_archived: bool = False,
        org_ids: Iterable[int] | None = None,
    ) -> list[CalendarRow]:
        """List owned or directly user-shared calendars ordered by name/ID.

        Optionally restrict tenant/archive state; a nonempty org_ids filter
        retains personal calendars and calendars in those organizations.
        This lookup does not evaluate organization-role grants.
        """
        params: list[Any] = [user_id, str(user_id)]
        clauses = [
            """
            (
                owner_user_id = ?
                OR EXISTS (
                    SELECT 1 FROM calendar_memberships cm
                    WHERE cm.calendar_id = calendars.id
                      AND cm.principal_type = 'user'
                      AND cm.principal_id = ?
                )
            )
            """
        ]
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if not include_archived:
            clauses.append("archived_at IS NULL")
        org_id_list = list(org_ids or [])
        if org_id_list:
            placeholders = ", ".join("?" for _ in org_id_list)
            clauses.append(f"(org_id IS NULL OR org_id IN ({placeholders}))")  # nosec B608
            params.extend(org_id_list)
        sql = f"SELECT * FROM calendars WHERE {' AND '.join(clauses)} ORDER BY name ASC, id ASC"  # nosec B608
        with self.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._calendar_from_row(row) for row in rows]

    def archive_calendar(self, calendar_id: int, *, archived_at: str | None = None) -> CalendarRow:
        """Timestamp and return the archived calendar; raise CalendarNotFound for a missing ID."""
        return self.update_calendar(calendar_id, archived_at=archived_at or _utcnow_iso())

    def create_membership(
        self,
        *,
        calendar_id: int,
        principal_type: str,
        principal_id: str | int,
        role: str,
    ) -> CalendarMembershipRow:
        """Create or replace a principal's role and return its assignment; failed readback raises CalendarValidationError."""
        now = _utcnow_iso()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO calendar_memberships (
                    calendar_id, principal_type, principal_id, role, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(calendar_id, principal_type, principal_id)
                DO UPDATE SET role = excluded.role, updated_at = excluded.updated_at
                """,
                (calendar_id, principal_type, str(principal_id), role, now, now),
            )
            row = conn.execute(
                """
                SELECT * FROM calendar_memberships
                WHERE calendar_id = ? AND principal_type = ? AND principal_id = ?
                """,
                (calendar_id, principal_type, str(principal_id)),
            ).fetchone()
            if row is None:
                raise CalendarValidationError("Failed to create calendar membership")
            return self._membership_from_row(row)

    def list_memberships(self, calendar_id: int) -> list[CalendarMembershipRow]:
        """Return all principal assignments in ID order, or an empty list when none exist."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM calendar_memberships
                WHERE calendar_id = ?
                ORDER BY id ASC
                """,
                (calendar_id,),
            ).fetchall()
        return [self._membership_from_row(row) for row in rows]

    def remove_membership(
        self,
        *,
        calendar_id: int,
        principal_type: str,
        principal_id: str | int,
    ) -> int:
        """Remove the exact principal assignment and return the number of rows deleted."""
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                DELETE FROM calendar_memberships
                WHERE calendar_id = ? AND principal_type = ? AND principal_id = ?
                """,
                (calendar_id, principal_type, str(principal_id)),
            )
            return int(cursor.rowcount or 0)

    def create_item(
        self,
        *,
        calendar_id: int,
        kind: str,
        title: str,
        source_owner: str = CALENDAR_SOURCE_OWNER_TLDW,
        provider_owned: bool = False,
        description: str | None = None,
        location: str | None = None,
        start_at: str | None = None,
        end_at: str | None = None,
        due_at: str | None = None,
        timezone: str | None = None,
        all_day: bool = False,
        status: str = "confirmed",
        local_tags_json: str | list[str] | None = None,
        metadata_json: str | dict[str, Any] | None = None,
        copied_from_item_id: int | None = None,
        linked_projection_type: str | None = None,
        linked_projection_id: str | None = None,
    ) -> CalendarItemRow:
        """Create and return a local item; reject provider ownership with CalendarReadOnlyError."""
        if provider_owned or source_owner == CALENDAR_SOURCE_OWNER_PROVIDER:
            raise CalendarReadOnlyError("Provider-owned items must be imported through provider upsert")
        now = _utcnow_iso()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO calendar_items (
                    calendar_id, kind, source_owner, provider_owned, title, description,
                    location, start_at, end_at, due_at, timezone, all_day, status,
                    local_tags_json, metadata_json, copied_from_item_id,
                    linked_projection_type, linked_projection_id, created_at, updated_at
                )
                VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    calendar_id,
                    kind,
                    source_owner,
                    title,
                    description,
                    location,
                    start_at,
                    end_at,
                    due_at,
                    timezone,
                    int(all_day),
                    status,
                    _json_or_none(local_tags_json),
                    _json_or_none(metadata_json),
                    copied_from_item_id,
                    linked_projection_type,
                    linked_projection_id,
                    now,
                    now,
                ),
            )
            return self._get_item_row(conn, int(cursor.lastrowid), include_deleted=True)

    def get_item(self, item_id: int, *, include_deleted: bool = False) -> CalendarItemRow:
        """Return an item or raise CalendarItemNotFound, hiding local/remote tombstones unless requested."""
        with self.connection() as conn:
            return self._get_item_row(conn, item_id, include_deleted=include_deleted)

    def update_item(
        self,
        item_id: int,
        patch: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CalendarItemRow:
        """Apply allowed local-item fields; missing IDs or provider-owned rows raise domain errors."""
        updates = _coerce_patch(patch, kwargs)
        with self.transaction() as conn:
            current = self._get_item_row(conn, item_id, include_deleted=True)
            if current.provider_owned:
                raise CalendarReadOnlyError("Provider-owned items are read-only")
            self._apply_update(
                conn,
                "calendar_items",
                "id",
                item_id,
                updates,
                {
                    "kind",
                    "title",
                    "description",
                    "location",
                    "start_at",
                    "end_at",
                    "due_at",
                    "timezone",
                    "all_day",
                    "status",
                    "local_tags_json",
                    "metadata_json",
                    "copied_from_item_id",
                    "linked_projection_type",
                    "linked_projection_id",
                },
            )
            return self._get_item_row(conn, item_id, include_deleted=True)

    def soft_delete_item(self, item_id: int, *, deleted_at: str | None = None) -> CalendarItemRow:
        """Tombstone and return a local item; missing/provider-owned rows raise not-found/read-only errors."""
        with self.transaction() as conn:
            current = self._get_item_row(conn, item_id, include_deleted=True)
            if current.provider_owned:
                raise CalendarReadOnlyError("Provider-owned items are read-only")
            self._apply_update(
                conn,
                "calendar_items",
                "id",
                item_id,
                {"deleted_at": deleted_at or _utcnow_iso()},
                {"deleted_at"},
            )
            return self._get_item_row(conn, item_id, include_deleted=True)

    def list_items_window(
        self,
        *,
        calendar_ids: Iterable[int],
        window_start: str,
        window_end: str,
        include_deleted: bool = False,
        include_remote_deleted: bool = False,
    ) -> list[CalendarItemRow]:
        """Return stored items overlapping inclusive timestamp bounds, ordered by time/ID.

        Skip empty calendar sets and hide local/remote tombstones by default.
        Recurrences are not expanded here.
        """
        ids = list(calendar_ids)
        if not ids:
            return []
        placeholders = ", ".join("?" for _ in ids)
        clauses = [
            f"calendar_id IN ({placeholders})",  # nosec B608
            """
            (
                (start_at IS NOT NULL AND COALESCE(end_at, start_at) >= ? AND start_at <= ?)
                OR (due_at IS NOT NULL AND due_at >= ? AND due_at <= ?)
            )
            """,
        ]
        params: list[Any] = [*ids, window_start, window_end, window_start, window_end]
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        if not include_remote_deleted:
            clauses.append("remote_deleted_at IS NULL")
        sql = f"""
            SELECT * FROM calendar_items
            WHERE {" AND ".join(clauses)}
            ORDER BY COALESCE(start_at, due_at) ASC, id ASC
            """  # nosec B608
        with self.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._item_from_row(row) for row in rows]

    def list_items_for_expansion(
        self,
        *,
        calendar_ids: Iterable[int],
        window_start: str,
        window_end: str,
        include_deleted: bool = False,
        include_remote_deleted: bool = False,
    ) -> list[CalendarItemRow]:
        """Return broad candidates for service-level timezone and recurrence expansion.

        Include recurrence masters and provider overrides, widen date bounds
        for zone offsets, and hide tombstones by default. Invalid nonempty
        window bounds raise CalendarValidationError; empty calendar sets yield [].
        """
        ids = list(calendar_ids)
        if not ids:
            return []
        widened_start, widened_end = _widened_date_bounds(window_start, window_end)
        placeholders = ", ".join("?" for _ in ids)
        clauses = [
            f"calendar_id IN ({placeholders})",  # nosec B608
            """
            COALESCE(start_at, due_at) IS NOT NULL
            """,
            """
            (
                (
                    NOT EXISTS (
                        SELECT 1 FROM calendar_recurrences r
                        WHERE r.calendar_item_id = calendar_items.id
                    )
                    AND substr(COALESCE(end_at, start_at, due_at), 1, 10) >= ?
                    AND substr(COALESCE(start_at, due_at), 1, 10) <= ?
                )
                OR
                (
                    EXISTS (
                        SELECT 1 FROM calendar_recurrences r
                        WHERE r.calendar_item_id = calendar_items.id
                    )
                )
                OR (
                    source_owner = 'provider'
                    AND json_extract(source_payload_json, '$.recurrence_id') IS NOT NULL
                )
            )
            """,
        ]
        params: list[Any] = [*ids, widened_start, widened_end]
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        if not include_remote_deleted:
            clauses.append("remote_deleted_at IS NULL")
        sql = f"""
            SELECT * FROM calendar_items
            WHERE {" AND ".join(clauses)}
            ORDER BY COALESCE(start_at, due_at) ASC, id ASC
            """  # nosec B608
        with self.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._item_from_row(row) for row in rows]

    def upsert_recurrence(
        self,
        *,
        calendar_item_id: int,
        rrule: str | None,
        rdate_json: str | list[str] | None = None,
        exdate_json: str | list[str] | None = None,
        timezone: str | None = None,
    ) -> CalendarRecurrenceRow:
        """Create or replace an item's first recurrence and return it; missing items raise CalendarItemNotFound."""
        now = _utcnow_iso()
        with self.transaction() as conn:
            self._get_item_row(conn, calendar_item_id, include_deleted=True)
            existing = conn.execute(
                """
                SELECT * FROM calendar_recurrences
                WHERE calendar_item_id = ?
                ORDER BY id ASC
                LIMIT 1
                """,
                (calendar_item_id,),
            ).fetchone()
            if existing is None:
                cursor = conn.execute(
                    """
                    INSERT INTO calendar_recurrences (
                        calendar_item_id, rrule, rdate_json, exdate_json,
                        timezone, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        calendar_item_id,
                        rrule,
                        _json_or_none(rdate_json),
                        _json_or_none(exdate_json),
                        timezone,
                        now,
                        now,
                    ),
                )
                return self._get_recurrence_row(conn, int(cursor.lastrowid))
            self._apply_update(
                conn,
                "calendar_recurrences",
                "id",
                int(existing["id"]),
                {
                    "rrule": rrule,
                    "rdate_json": rdate_json,
                    "exdate_json": exdate_json,
                    "timezone": timezone,
                },
                {"rrule", "rdate_json", "exdate_json", "timezone"},
            )
            return self._get_recurrence_row(conn, int(existing["id"]))

    def delete_recurrence(self, calendar_item_id: int) -> None:
        """Clear recurrence within the caller's item mutation transaction."""
        with self.transaction() as conn:
            conn.execute("DELETE FROM calendar_recurrences WHERE calendar_item_id = ?", (calendar_item_id,))

    def list_recurrences_for_items(
        self,
        calendar_item_ids: Iterable[int],
    ) -> dict[int, CalendarRecurrenceRow]:
        """Batch the first recurrence by ID for each requested item; omit items without a recurrence."""
        ids = list(calendar_item_ids)
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        sql = f"""
            SELECT * FROM calendar_recurrences
            WHERE calendar_item_id IN ({placeholders})
            ORDER BY id ASC
            """  # nosec B608
        with self.connection() as conn:
            rows = conn.execute(sql, tuple(ids)).fetchall()
        recurrences: dict[int, CalendarRecurrenceRow] = {}
        for row in rows:
            recurrence = self._recurrence_from_row(row)
            recurrences.setdefault(recurrence.calendar_item_id, recurrence)
        return recurrences

    def upsert_provider_item(
        self,
        *,
        calendar_id: int,
        external_binding_id: int,
        source_uid: str,
        title: str,
        start_at: str | None = None,
        end_at: str | None = None,
        due_at: str | None = None,
        kind: str = "event",
        description: str | None = None,
        location: str | None = None,
        timezone: str | None = None,
        all_day: bool = False,
        status: str = "confirmed",
        provider_payload_json: str | dict[str, Any] | None = None,
        source_etag: str | None = None,
        source_ctag: str | None = None,
        source_updated_at: str | None = None,
        metadata_json: str | dict[str, Any] | None = None,
    ) -> CalendarItemRow:
        """Import or refresh a provider-owned item keyed by binding and source UID.

        Require an active account and matching personal owner/tenant calendar;
        invalid bindings raise validation/not-found errors. A refresh clears
        remote deletion, and failed readback raises CalendarSyncError.
        """
        now = _utcnow_iso()
        payload_json = _json_or_none(provider_payload_json)
        metadata = _json_or_none(metadata_json)
        with self.transaction() as conn:
            binding = self._get_external_binding_row(conn, external_binding_id)
            account = self._get_external_account_row(conn, binding.account_id)
            calendar = self._get_calendar_row(conn, binding.calendar_id)
            self._validate_external_account_active(account)
            self._validate_external_binding_calendar(account, calendar)
            if binding.calendar_id != calendar_id:
                raise CalendarValidationError("Provider item calendar_id must match external binding")
            conn.execute(
                """
                INSERT INTO calendar_items (
                    calendar_id, kind, source_owner, provider_owned, title, description,
                    location, start_at, end_at, due_at, timezone, all_day, status,
                    metadata_json, external_binding_id, source_uid, source_etag, source_ctag,
                    source_payload_json, source_updated_at, created_at, updated_at
                )
                VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(external_binding_id, source_uid)
                DO UPDATE SET
                    calendar_id = excluded.calendar_id,
                    kind = excluded.kind,
                    title = excluded.title,
                    description = excluded.description,
                    location = excluded.location,
                    start_at = excluded.start_at,
                    end_at = excluded.end_at,
                    due_at = excluded.due_at,
                    timezone = excluded.timezone,
                    all_day = excluded.all_day,
                    status = excluded.status,
                    metadata_json = excluded.metadata_json,
                    source_etag = excluded.source_etag,
                    source_ctag = excluded.source_ctag,
                    source_payload_json = excluded.source_payload_json,
                    source_updated_at = excluded.source_updated_at,
                    remote_deleted_at = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    binding.calendar_id,
                    kind,
                    CALENDAR_SOURCE_OWNER_PROVIDER,
                    title,
                    description,
                    location,
                    start_at,
                    end_at,
                    due_at,
                    timezone,
                    int(all_day),
                    status,
                    metadata,
                    external_binding_id,
                    source_uid,
                    source_etag,
                    source_ctag,
                    payload_json,
                    source_updated_at,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM calendar_items
                WHERE external_binding_id = ? AND source_uid = ?
                """,
                (external_binding_id, source_uid),
            ).fetchone()
            if row is None:
                raise CalendarSyncError("Failed to upsert provider calendar item")
            return self._item_from_row(row)

    def mark_provider_item_remote_deleted(
        self,
        *,
        external_binding_id: int,
        source_uid: str,
        remote_deleted_at: str | None = None,
    ) -> CalendarItemRow:
        """Tombstone a provider item by binding/UID; raise CalendarItemNotFound when no import matches."""
        now = _utcnow_iso()
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE calendar_items
                SET remote_deleted_at = ?, updated_at = ?
                WHERE external_binding_id = ? AND source_uid = ? AND provider_owned = 1
                """,
                (remote_deleted_at or now, now, external_binding_id, source_uid),
            )
            row = conn.execute(
                """
                SELECT * FROM calendar_items
                WHERE external_binding_id = ? AND source_uid = ? AND provider_owned = 1
                """,
                (external_binding_id, source_uid),
            ).fetchone()
            if row is None:
                raise CalendarItemNotFound("Provider calendar item not found")
            return self._item_from_row(row)

    def list_provider_items_for_binding(
        self,
        external_binding_id: int,
        *,
        include_deleted: bool = False,
        include_remote_deleted: bool = False,
    ) -> list[CalendarItemRow]:
        """List one binding's imported rows ordered by time/ID, hiding local/remote tombstones by default."""
        clauses = ["external_binding_id = ?", "provider_owned = 1"]
        params: list[Any] = [external_binding_id]
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        if not include_remote_deleted:
            clauses.append("remote_deleted_at IS NULL")
        sql = f"""
            SELECT * FROM calendar_items
            WHERE {" AND ".join(clauses)}
            ORDER BY COALESCE(start_at, due_at, created_at) ASC, id ASC
            """  # nosec B608
        with self.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._item_from_row(row) for row in rows]

    def delete_remote_tombstones_eligible_for_cleanup(
        self,
        *,
        before_iso: str,
        limit: int = 500,
    ) -> int:
        """Delete up to limit old remote tombstones without live annotations or links.

        Detach local copies before deleting sources and return the deletion count.
        """
        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT id FROM calendar_items ci
                WHERE ci.provider_owned = 1
                  AND ci.remote_deleted_at IS NOT NULL
                  AND ci.remote_deleted_at <= ?
                  AND NOT EXISTS (
                    SELECT 1 FROM calendar_annotations ca
                    WHERE ca.calendar_item_id = ci.id AND ca.deleted_at IS NULL
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM calendar_links cl
                    WHERE cl.calendar_item_id = ci.id
                  )
                ORDER BY ci.remote_deleted_at ASC, ci.id ASC
                LIMIT ?
                """,
                (before_iso, limit),
            ).fetchall()
            ids = [int(row["id"]) for row in rows]
            if not ids:
                return 0
            self._detach_copied_items_from_provider_items(conn, ids)
            placeholders = ", ".join("?" for _ in ids)
            conn.execute(
                f"DELETE FROM calendar_items WHERE id IN ({placeholders})",  # nosec B608
                tuple(ids),
            )
            return len(ids)

    def create_annotation(
        self,
        *,
        calendar_item_id: int,
        author_user_id: int,
        body: str,
        tags_json: str | list[str] | None = None,
    ) -> CalendarAnnotationRow:
        """Create and return an authored annotation; an empty body stores a local tag overlay."""
        now = _utcnow_iso()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO calendar_annotations (
                    calendar_item_id, author_user_id, body, tags_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (calendar_item_id, author_user_id, body, _json_or_none(tags_json), now, now),
            )
            return self._get_annotation_row(conn, int(cursor.lastrowid), include_deleted=True)

    def update_annotation(
        self,
        annotation_id: int,
        patch: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CalendarAnnotationRow:
        """Apply allowed annotation fields and return even deleted rows; missing IDs raise CalendarItemNotFound."""
        updates = _coerce_patch(patch, kwargs)
        with self.transaction() as conn:
            self._apply_update(
                conn,
                "calendar_annotations",
                "id",
                annotation_id,
                updates,
                {"body", "tags_json", "deleted_at"},
            )
            return self._get_annotation_row(conn, annotation_id, include_deleted=True)

    def get_annotation(
        self,
        annotation_id: int,
        *,
        include_deleted: bool = False,
    ) -> CalendarAnnotationRow:
        """Return an annotation or raise CalendarItemNotFound, excluding tombstones unless requested."""
        with self.connection() as conn:
            return self._get_annotation_row(
                conn,
                annotation_id,
                include_deleted=include_deleted,
            )

    def delete_annotation(self, annotation_id: int, *, deleted_at: str | None = None) -> int:
        """Soft-delete a live annotation and return 0 for missing or already deleted rows, otherwise 1."""
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE calendar_annotations
                SET deleted_at = ?, updated_at = ?
                WHERE id = ? AND deleted_at IS NULL
                """,
                (deleted_at or _utcnow_iso(), _utcnow_iso(), annotation_id),
            )
            return int(cursor.rowcount or 0)

    def list_annotations(
        self,
        calendar_item_id: int,
        *,
        include_deleted: bool = False,
    ) -> list[CalendarAnnotationRow]:
        """List an item's annotations by creation time/ID, excluding tombstones unless requested."""
        clauses = ["calendar_item_id = ?"]
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        sql = f"""
            SELECT * FROM calendar_annotations
            WHERE {" AND ".join(clauses)}
            ORDER BY created_at ASC, id ASC
            """  # nosec B608
        with self.connection() as conn:
            rows = conn.execute(sql, (calendar_item_id,)).fetchall()
        return [self._annotation_from_row(row) for row in rows]

    def list_tag_overlays_for_items(
        self,
        calendar_item_ids: Iterable[int],
        *,
        author_user_id: int,
    ) -> dict[int, list[CalendarAnnotationRow]]:
        """Batch an actor's live empty-body tag overlays for already authorized items.

        Use one query ordered by annotation ID; omit items without overlays and
        return an empty mapping without querying for empty input. Callers must
        enforce item/tenant access before supplying IDs.
        """
        ids = list(calendar_item_ids)
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        sql = f"""
            SELECT * FROM calendar_annotations
            WHERE calendar_item_id IN ({placeholders})
              AND author_user_id = ?
              AND deleted_at IS NULL
              AND body = ''
            ORDER BY id ASC
            """  # nosec B608 - only parameter placeholders are interpolated
        with self.connection() as conn:
            rows = conn.execute(sql, (*ids, author_user_id)).fetchall()
        overlays: dict[int, list[CalendarAnnotationRow]] = {}
        for row in rows:
            annotation = self._annotation_from_row(row)
            overlays.setdefault(annotation.calendar_item_id, []).append(annotation)
        return overlays

    def create_link(
        self,
        *,
        calendar_item_id: int,
        target_type: str,
        target_id: str | int,
        label: str | None = None,
        url: str | None = None,
        metadata_json: str | dict[str, Any] | None = None,
    ) -> CalendarLinkRow:
        """Create and return an item context link, storing target IDs as text and metadata as JSON."""
        now = _utcnow_iso()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO calendar_links (
                    calendar_item_id, target_type, target_id, label, url, metadata_json,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    calendar_item_id,
                    target_type,
                    str(target_id),
                    label,
                    url,
                    _json_or_none(metadata_json),
                    now,
                    now,
                ),
            )
            return self._get_link_row(conn, int(cursor.lastrowid))

    def delete_link(self, link_id: int) -> int:
        """Physically remove a context link and return the affected-row count, including 0 for missing IDs."""
        with self.transaction() as conn:
            cursor = conn.execute("DELETE FROM calendar_links WHERE id = ?", (link_id,))
            return int(cursor.rowcount or 0)

    def get_link(self, link_id: int) -> CalendarLinkRow:
        """Return a context link or raise CalendarItemNotFound when its ID is absent."""
        with self.connection() as conn:
            return self._get_link_row(conn, link_id)

    def list_links_for_items(self, calendar_item_ids: Iterable[int]) -> dict[int, list[CalendarLinkRow]]:
        """Load persisted context links in one query for already authorized view items."""
        ids = list(calendar_item_ids)
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        sql = f"SELECT * FROM calendar_links WHERE calendar_item_id IN ({placeholders}) ORDER BY id"  # nosec B608
        with self.connection() as conn:
            rows = conn.execute(sql, tuple(ids)).fetchall()
        links: dict[int, list[CalendarLinkRow]] = {}
        for row in rows:
            link = self._link_from_row(row)
            links.setdefault(link.calendar_item_id, []).append(link)
        return links

    def list_links(self, calendar_item_id: int) -> list[CalendarLinkRow]:
        """Return an item's context links ordered by creation time and ID, or [] when none exist."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM calendar_links
                WHERE calendar_item_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (calendar_item_id,),
            ).fetchall()
        return [self._link_from_row(row) for row in rows]

    def create_secret_ref(
        self,
        *,
        tenant_id: str,
        user_id: int,
        provider: str,
        encrypted_payload: str,
    ) -> str:
        """Store an already encrypted provider payload and return a new opaque secret reference."""
        return self.secret_store.create_secret_ref(
            tenant_id=tenant_id,
            user_id=user_id,
            provider=provider,
            encrypted_payload=encrypted_payload,
        )

    def resolve_secret_ref(self, secret_ref: str) -> str:
        """Return a live encrypted payload after caller authorization; missing refs raise CalendarValidationError."""
        return self.secret_store.resolve_secret_ref(secret_ref)

    def resolve_secret_ref_scoped(
        self,
        secret_ref: str,
        *,
        tenant_id: str,
        user_id: int,
        provider: str,
    ) -> str:
        """Return the scoped encrypted payload or raise CalendarValidationError for unavailable/mismatched refs."""
        return self.secret_store.resolve_secret_ref_scoped(
            secret_ref,
            tenant_id=tenant_id,
            user_id=user_id,
            provider=provider,
        )

    def resolve_secret_ref_for_user(
        self,
        secret_ref: str,
        *,
        tenant_id: str,
        user_id: int,
    ) -> str:
        """Return a tenant/user's encrypted payload or raise CalendarValidationError on unavailable/mismatched refs."""
        return self.secret_store.resolve_secret_ref_for_user(
            secret_ref,
            tenant_id=tenant_id,
            user_id=user_id,
        )

    def delete_secret_ref(self, secret_ref: str) -> None:
        """Wipe and tombstone a caller-authorized live secret; missing refs raise CalendarValidationError."""
        self.secret_store.delete_secret_ref(secret_ref)

    def delete_secret_ref_scoped(
        self,
        secret_ref: str,
        *,
        tenant_id: str,
        user_id: int,
        provider: str,
    ) -> None:
        """Wipe a matching tenant/user/provider secret or raise CalendarValidationError before mutation."""
        self.secret_store.delete_secret_ref_scoped(
            secret_ref,
            tenant_id=tenant_id,
            user_id=user_id,
            provider=provider,
        )

    def delete_secret_ref_for_user(
        self,
        secret_ref: str,
        *,
        tenant_id: str,
        user_id: int,
    ) -> None:
        """Wipe a matching tenant/user secret or raise CalendarValidationError before mutation."""
        self.secret_store.delete_secret_ref_for_user(
            secret_ref,
            tenant_id=tenant_id,
            user_id=user_id,
        )

    def create_external_account(
        self,
        *,
        tenant_id: str,
        user_id: int,
        provider: str,
        display_name: str,
        secret_ref: str | None,
        account_metadata_json: str | dict[str, Any] | None = None,
        status: str = "active",
    ) -> ExternalCalendarAccountRow:
        """Create and return provider account metadata without storing credentials in the account row.

        A supplied secret ref must match tenant, user, and provider or creation
        raises CalendarValidationError.
        """
        now = _utcnow_iso()
        with self.transaction() as conn:
            if secret_ref is not None:
                self.secret_store.validate_secret_ref_in_connection(
                    conn,
                    secret_ref=secret_ref,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    provider=provider,
                )
            cursor = conn.execute(
                """
                INSERT INTO external_calendar_accounts (
                    tenant_id, user_id, provider, display_name, secret_ref,
                    account_metadata_json, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    user_id,
                    provider,
                    display_name,
                    secret_ref,
                    _json_or_none(account_metadata_json),
                    status,
                    now,
                    now,
                ),
            )
            return self._get_external_account_row(conn, int(cursor.lastrowid), include_deleted=True)

    def get_external_account(
        self,
        account_id: int,
        *,
        include_deleted: bool = False,
    ) -> ExternalCalendarAccountRow:
        """Return an account or raise CalendarNotFound, hiding soft-deleted accounts unless requested."""
        with self.connection() as conn:
            return self._get_external_account_row(conn, account_id, include_deleted=include_deleted)

    def list_external_accounts_for_user(
        self,
        *,
        user_id: int,
        tenant_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[ExternalCalendarAccountRow]:
        """List a user's accounts by provider/name/ID, optionally restricting tenant and including deleted rows."""
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        sql = f"""
            SELECT * FROM external_calendar_accounts
            WHERE {" AND ".join(clauses)}
            ORDER BY provider ASC, display_name ASC, id ASC
            """  # nosec B608
        with self.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._external_account_from_row(row) for row in rows]

    def delete_external_account(
        self,
        account_id: int,
        *,
        destructive_imported_record_cleanup: bool = False,
        deleted_at: str | None = None,
    ) -> ExternalCalendarAccountRow:
        """Delete an account atomically, wiping its secret and disabling its bindings.

        Tombstone imported items unless destructive cleanup is requested, which
        preserves detached local copies. Return deleted account state; missing
        accounts or unavailable referenced secrets raise domain errors.
        """
        now = deleted_at or _utcnow_iso()
        with self.transaction() as conn:
            account = self._get_external_account_row(conn, account_id, include_deleted=True)
            self.secret_store.delete_secret_ref_in_connection(
                conn,
                account.secret_ref,
                tenant_id=account.tenant_id,
                user_id=account.user_id,
                provider=account.provider,
            )
            bindings = self._list_external_bindings_for_account_rows(
                conn,
                account_id,
                include_deleted=True,
            )
            for binding in bindings:
                self._disable_binding_in_connection(conn, binding.id, now)
                self._cleanup_provider_items_for_binding(
                    conn,
                    binding.id,
                    remote_deleted_at=now,
                    destructive=destructive_imported_record_cleanup,
                )
            conn.execute(
                """
                UPDATE external_calendar_accounts
                SET status = 'deleted',
                    revoked_at = COALESCE(revoked_at, ?),
                    deleted_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, now, account_id),
            )
            return self._get_external_account_row(conn, account_id, include_deleted=True)

    def revoke_external_account(
        self,
        account_id: int,
        *,
        revoked_at: str | None = None,
    ) -> ExternalCalendarAccountRow:
        """Revoke an account, wipe its secret, disable bindings, and tombstone imports atomically.

        Return revoked account state; missing accounts or unavailable referenced
        secrets raise domain errors.
        """
        now = revoked_at or _utcnow_iso()
        with self.transaction() as conn:
            account = self._get_external_account_row(conn, account_id, include_deleted=True)
            self.secret_store.delete_secret_ref_in_connection(
                conn,
                account.secret_ref,
                tenant_id=account.tenant_id,
                user_id=account.user_id,
                provider=account.provider,
            )
            bindings = self._list_external_bindings_for_account_rows(
                conn,
                account_id,
                include_deleted=True,
            )
            for binding in bindings:
                self._disable_binding_in_connection(conn, binding.id, now)
                self._cleanup_provider_items_for_binding(
                    conn,
                    binding.id,
                    remote_deleted_at=now,
                    destructive=False,
                )
            conn.execute(
                """
                UPDATE external_calendar_accounts
                SET status = 'revoked', revoked_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, now, account_id),
            )
            return self._get_external_account_row(conn, account_id, include_deleted=True)

    def create_external_binding(
        self,
        *,
        account_id: int,
        calendar_id: int,
        remote_calendar_id: str,
        remote_display_name: str | None = None,
        sync_enabled: bool = True,
        sync_interval_minutes: int | None = DEFAULT_SYNC_INTERVAL_MINUTES,
        lookback_days: int = DEFAULT_SYNC_LOOKBACK_DAYS,
        lookahead_days: int = DEFAULT_SYNC_LOOKAHEAD_DAYS,
        provider_capabilities_json: str | dict[str, Any] | None = None,
        sync_cursor: str | None = None,
        next_scan_at: str | None = None,
    ) -> ExternalCalendarBindingRow:
        """Bind an active account to its owner's personal same-tenant calendar.

        Return a new binding or revive a deleted binding with reset sync state.
        An omitted interval defaults to hourly polling; explicit None is manual-only.
        Missing resources raise CalendarNotFound; invalid ownership/account state
        or a duplicate live remote calendar raises CalendarValidationError.
        """
        now = _utcnow_iso()
        with self.transaction() as conn:
            account = self._get_external_account_row(conn, account_id)
            calendar = self._get_calendar_row(conn, calendar_id)
            self._validate_external_account_active(account)
            self._validate_external_binding_calendar(account, calendar)
            existing = conn.execute(
                """
                SELECT * FROM external_calendar_bindings
                WHERE account_id = ? AND remote_calendar_id = ?
                """,
                (account_id, remote_calendar_id),
            ).fetchone()
            if existing is not None:
                existing_binding = self._external_binding_from_row(existing)
                if existing_binding.deleted_at is None:
                    raise CalendarValidationError(
                        "External calendar binding already exists for this account and remote calendar"
                    )
                conn.execute(
                    """
                    UPDATE external_calendar_bindings
                    SET calendar_id = ?,
                        remote_display_name = ?,
                        sync_enabled = ?,
                        sync_interval_minutes = ?,
                        lookback_days = ?,
                        lookahead_days = ?,
                        provider_capabilities_json = ?,
                        sync_cursor = ?,
                        last_sync_at = NULL,
                        next_scan_at = ?,
                        last_error = NULL,
                        disabled_at = NULL,
                        deleted_at = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        calendar_id,
                        remote_display_name,
                        int(sync_enabled),
                        sync_interval_minutes,
                        lookback_days,
                        lookahead_days,
                        _json_or_none(provider_capabilities_json),
                        sync_cursor,
                        next_scan_at,
                        now,
                        existing_binding.id,
                    ),
                )
                return self._get_external_binding_row(
                    conn,
                    existing_binding.id,
                    include_deleted=True,
                )
            cursor = conn.execute(
                """
                INSERT INTO external_calendar_bindings (
                    account_id, calendar_id, remote_calendar_id, remote_display_name,
                    sync_enabled, sync_interval_minutes, lookback_days, lookahead_days,
                    provider_capabilities_json, sync_cursor, next_scan_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    account_id,
                    calendar_id,
                    remote_calendar_id,
                    remote_display_name,
                    int(sync_enabled),
                    sync_interval_minutes,
                    lookback_days,
                    lookahead_days,
                    _json_or_none(provider_capabilities_json),
                    sync_cursor,
                    next_scan_at,
                    now,
                    now,
                ),
            )
            return self._get_external_binding_row(conn, int(cursor.lastrowid), include_deleted=True)

    def list_provider_binding_owners(
        self, binding_ids: Iterable[int], *, tenant_id: str,
    ) -> dict[int, int]:
        """Return binding/account ownership in one tenant-scoped lookup.

        Include deleted bindings/accounts to preserve imported-row visibility
        semantics; missing bindings have no owner. This exposes no credentials.
        """
        ids = sorted(set(binding_ids))
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        sql = f"""
            SELECT b.id AS binding_id, a.user_id
            FROM external_calendar_bindings b
            JOIN external_calendar_accounts a ON a.id = b.account_id
            WHERE b.id IN ({placeholders}) AND a.tenant_id = ?
            """  # nosec B608
        with self.connection() as conn:
            rows = conn.execute(sql, (*ids, tenant_id)).fetchall()
        return {int(row["binding_id"]): int(row["user_id"]) for row in rows}

    def get_external_binding(
        self,
        binding_id: int,
        *,
        include_deleted: bool = False,
    ) -> ExternalCalendarBindingRow:
        """Return a binding or raise CalendarNotFound, hiding tombstones unless requested."""
        with self.connection() as conn:
            return self._get_external_binding_row(conn, binding_id, include_deleted=include_deleted)

    def list_external_bindings_for_account(
        self,
        account_id: int,
        *,
        include_deleted: bool = False,
    ) -> list[ExternalCalendarBindingRow]:
        """List an account's bindings by remote name/ID, hiding tombstones unless requested."""
        with self.connection() as conn:
            return self._list_external_bindings_for_account_rows(
                conn,
                account_id,
                include_deleted=include_deleted,
            )

    def list_sync_enabled_bindings_due_for_scan(
        self,
        *,
        now_iso: str | None = None,
        limit: int = 100,
    ) -> list[ExternalCalendarBindingRow]:
        """Return due enabled bindings with positive intervals for live active accounts.

        Null intervals (including legacy rows) are manual-only even with no next
        scan time. Order by scan time/ID and return at most limit bindings.
        """
        now = now_iso or _utcnow_iso()
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT b.* FROM external_calendar_bindings b
                JOIN external_calendar_accounts a ON a.id = b.account_id
                WHERE b.sync_enabled = 1
                  AND b.sync_interval_minutes > 0
                  AND b.disabled_at IS NULL
                  AND b.deleted_at IS NULL
                  AND a.status = 'active'
                  AND a.revoked_at IS NULL
                  AND a.deleted_at IS NULL
                  AND (b.next_scan_at IS NULL OR b.next_scan_at <= ?)
                ORDER BY COALESCE(b.next_scan_at, b.created_at) ASC, b.id ASC
                LIMIT ?
                """,
                (now, limit),
            ).fetchall()
        return [self._external_binding_from_row(row) for row in rows]

    def update_external_binding(
        self,
        binding_id: int,
        patch: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ExternalCalendarBindingRow:
        """Apply allowed binding fields and return even deleted state; missing IDs raise CalendarNotFound."""
        updates = _coerce_patch(patch, kwargs)
        with self.transaction() as conn:
            self._apply_update(
                conn,
                "external_calendar_bindings",
                "id",
                binding_id,
                updates,
                {
                    "remote_display_name",
                    "sync_enabled",
                    "sync_interval_minutes",
                    "lookback_days",
                    "lookahead_days",
                    "provider_capabilities_json",
                    "sync_cursor",
                    "last_sync_at",
                    "next_scan_at",
                    "last_error",
                    "disabled_at",
                    "deleted_at",
                },
            )
            return self._get_external_binding_row(conn, binding_id, include_deleted=True)

    def disable_external_binding(
        self,
        binding_id: int,
        *,
        disabled_at: str | None = None,
    ) -> ExternalCalendarBindingRow:
        """Stop polling while preserving the first disabled timestamp; return state or raise CalendarNotFound."""
        now = disabled_at or _utcnow_iso()
        with self.transaction() as conn:
            self._disable_binding_in_connection(conn, binding_id, now)
            return self._get_external_binding_row(conn, binding_id, include_deleted=True)

    def delete_external_binding(
        self,
        binding_id: int,
        *,
        destructive_imported_record_cleanup: bool = False,
        deleted_at: str | None = None,
    ) -> ExternalCalendarBindingRow:
        """Disable and soft-delete a binding, tombstoning imports unless destructive cleanup is requested.

        Detach local copies before destructive cleanup; return deleted binding
        state or raise CalendarNotFound for a missing ID.
        """
        now = deleted_at or _utcnow_iso()
        with self.transaction() as conn:
            self._disable_binding_in_connection(conn, binding_id, now)
            self._cleanup_provider_items_for_binding(
                conn,
                binding_id,
                remote_deleted_at=now,
                destructive=destructive_imported_record_cleanup,
            )
            conn.execute(
                """
                UPDATE external_calendar_bindings
                SET deleted_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, now, binding_id),
            )
            return self._get_external_binding_row(conn, binding_id, include_deleted=True)

    def update_binding_sync_state(
        self,
        binding_id: int,
        *,
        sync_cursor: str | None | object = _UNSET,
        last_sync_at: str | None | object = _UNSET,
        next_scan_at: str | None | object = _UNSET,
        last_error: str | None | object = _UNSET,
    ) -> ExternalCalendarBindingRow:
        """Return updated sync state, preserving omitted fields while allowing explicit null clears.

        Missing binding IDs raise CalendarNotFound.
        """
        patch: dict[str, Any] = {}
        if sync_cursor is not _UNSET:
            patch["sync_cursor"] = sync_cursor
        if last_sync_at is not _UNSET:
            patch["last_sync_at"] = last_sync_at
        if next_scan_at is not _UNSET:
            patch["next_scan_at"] = next_scan_at
        if last_error is not _UNSET:
            patch["last_error"] = last_error
        return self.update_external_binding(binding_id, patch)

    def record_sync_event(
        self,
        *,
        event_type: str,
        status: str,
        binding_id: int | None = None,
        account_id: int | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        items_seen: int = 0,
        items_upserted: int = 0,
        items_tombstoned: int = 0,
        error_message: str | None = None,
        metadata_json: str | dict[str, Any] | None = None,
    ) -> CalendarSyncEventRow:
        """Record and return a sync-history event, deriving the account from a binding when omitted.

        Missing binding IDs raise CalendarNotFound; failed readback raises CalendarSyncError.
        """
        created_at = _utcnow_iso()
        with self.transaction() as conn:
            if account_id is None and binding_id is not None:
                binding = self._get_external_binding_row(conn, binding_id, include_deleted=True)
                account_id = binding.account_id
            cursor = conn.execute(
                """
                INSERT INTO calendar_sync_events (
                    binding_id, account_id, event_type, status, started_at, finished_at,
                    items_seen, items_upserted, items_tombstoned, error_message,
                    metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    binding_id,
                    account_id,
                    event_type,
                    status,
                    started_at,
                    finished_at,
                    items_seen,
                    items_upserted,
                    items_tombstoned,
                    error_message,
                    _json_or_none(metadata_json),
                    created_at,
                ),
            )
            return self._get_sync_event_row(conn, int(cursor.lastrowid))

    def list_sync_events(
        self,
        *,
        binding_id: int | None = None,
        account_id: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[CalendarSyncEventRow]:
        """Return newest-first sync history with optional binding/account filters and limit/offset paging."""
        clauses: list[str] = []
        params: list[Any] = []
        if binding_id is not None:
            clauses.append("binding_id = ?")
            params.append(binding_id)
        if account_id is not None:
            clauses.append("account_id = ?")
            params.append(account_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT * FROM calendar_sync_events
            {where}
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
            """  # nosec B608
        params.extend([limit, offset])
        with self.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._sync_event_from_row(row) for row in rows]

    def _apply_update(
        self,
        conn: sqlite3.Connection,
        table: str,
        id_column: str,
        row_id: int,
        patch: dict[str, Any],
        allowed_columns: set[str],
    ) -> None:
        """Apply whitelisted fields with parameterized values, JSON conversion, and an updated timestamp.

        Ignore unknown keys and empty patches; table and column identifiers
        must come from trusted repository call sites.
        """
        updates: dict[str, Any] = {}
        for key, value in patch.items():
            if key not in allowed_columns:
                continue
            updates[key] = _json_or_none(value) if key.endswith("_json") else value
        if not updates:
            return
        updates["updated_at"] = _utcnow_iso()
        assignments = ", ".join(f"{column} = ?" for column in updates)
        values = [*updates.values(), row_id]
        conn.execute(
            f"UPDATE {table} SET {assignments} WHERE {id_column} = ?",  # nosec B608
            tuple(values),
        )

    def _disable_binding_in_connection(
        self,
        conn: sqlite3.Connection,
        binding_id: int,
        disabled_at: str,
    ) -> None:
        """Disable polling in the caller's transaction, preserving any existing disabled timestamp."""
        conn.execute(
            """
            UPDATE external_calendar_bindings
            SET sync_enabled = 0,
                disabled_at = COALESCE(disabled_at, ?),
                updated_at = ?
            WHERE id = ?
            """,
            (disabled_at, disabled_at, binding_id),
        )

    @staticmethod
    def _validate_external_account_active(account: ExternalCalendarAccountRow) -> None:
        """Raise CalendarValidationError unless the account is active, unrevoked, and undeleted."""
        if account.status != "active" or account.revoked_at is not None or account.deleted_at is not None:
            raise CalendarValidationError("External calendar account is not active")

    @staticmethod
    def _validate_external_binding_calendar(
        account: ExternalCalendarAccountRow,
        calendar: CalendarRow,
    ) -> None:
        """Raise CalendarValidationError unless a binding targets its owner's personal same-tenant calendar."""
        if calendar.org_id is not None:
            raise CalendarValidationError("Personal external calendar accounts cannot bind to org calendars")
        if calendar.owner_user_id != account.user_id:
            raise CalendarValidationError("External calendar account can only bind to the owner's calendar")
        if calendar.tenant_id != account.tenant_id:
            raise CalendarValidationError("External calendar account and calendar tenant must match")

    def _cleanup_provider_items_for_binding(
        self,
        conn: sqlite3.Connection,
        binding_id: int,
        *,
        remote_deleted_at: str,
        destructive: bool,
    ) -> None:
        """Tombstone one binding's imports or delete them after detaching local copies, without committing."""
        if destructive:
            self._detach_copied_items_for_binding(conn, binding_id)
            conn.execute(
                """
                DELETE FROM calendar_items
                WHERE external_binding_id = ? AND provider_owned = 1
                """,
                (binding_id,),
            )
            return
        conn.execute(
            """
            UPDATE calendar_items
            SET remote_deleted_at = COALESCE(remote_deleted_at, ?),
                updated_at = ?
            WHERE external_binding_id = ? AND provider_owned = 1
            """,
            (remote_deleted_at, remote_deleted_at, binding_id),
        )

    def _detach_copied_items_for_binding(self, conn: sqlite3.Connection, binding_id: int) -> None:
        """Clear copy-source references before deleting a binding's imported source rows."""
        conn.execute(
            """
            UPDATE calendar_items
            SET copied_from_item_id = NULL,
                updated_at = ?
            WHERE copied_from_item_id IN (
                SELECT id FROM calendar_items
                WHERE external_binding_id = ? AND provider_owned = 1
            )
            """,
            (_utcnow_iso(), binding_id),
        )

    def _detach_copied_items_from_provider_items(
        self,
        conn: sqlite3.Connection,
        item_ids: Iterable[int],
    ) -> None:
        """Clear local copy-source references to a batch of imports; empty batches perform no SQL."""
        ids = list(item_ids)
        if not ids:
            return
        placeholders = ", ".join("?" for _ in ids)
        conn.execute(
            f"""
            UPDATE calendar_items
            SET copied_from_item_id = NULL,
                updated_at = ?
            WHERE copied_from_item_id IN ({placeholders})
            """,  # nosec B608
            (_utcnow_iso(), *ids),
        )

    def _get_calendar_row(
        self,
        conn: sqlite3.Connection,
        calendar_id: int,
        *,
        include_archived: bool = False,
    ) -> CalendarRow:
        """Read calendar state in the caller's connection or raise CalendarNotFound, honoring archive filtering."""
        clauses = ["id = ?"]
        if not include_archived:
            clauses.append("archived_at IS NULL")
        sql = f"SELECT * FROM calendars WHERE {' AND '.join(clauses)}"  # nosec B608
        row = conn.execute(sql, (calendar_id,)).fetchone()
        if row is None:
            raise CalendarNotFound(f"Calendar not found: {calendar_id}")
        return self._calendar_from_row(row)

    def _get_item_row(
        self,
        conn: sqlite3.Connection,
        item_id: int,
        *,
        include_deleted: bool = False,
    ) -> CalendarItemRow:
        """Read item state or raise CalendarItemNotFound, honoring local and remote tombstone filtering."""
        clauses = ["id = ?"]
        if not include_deleted:
            clauses.extend(["deleted_at IS NULL", "remote_deleted_at IS NULL"])
        sql = f"SELECT * FROM calendar_items WHERE {' AND '.join(clauses)}"  # nosec B608
        row = conn.execute(sql, (item_id,)).fetchone()
        if row is None:
            raise CalendarItemNotFound(f"Calendar item not found: {item_id}")
        return self._item_from_row(row)

    def _get_recurrence_row(
        self,
        conn: sqlite3.Connection,
        recurrence_id: int,
    ) -> CalendarRecurrenceRow:
        """Read a recurrence by ID in the caller's connection or raise CalendarItemNotFound."""
        row = conn.execute(
            "SELECT * FROM calendar_recurrences WHERE id = ?",
            (recurrence_id,),
        ).fetchone()
        if row is None:
            raise CalendarItemNotFound(f"Calendar recurrence not found: {recurrence_id}")
        return self._recurrence_from_row(row)

    def _get_annotation_row(
        self,
        conn: sqlite3.Connection,
        annotation_id: int,
        *,
        include_deleted: bool = False,
    ) -> CalendarAnnotationRow:
        """Read annotation state or raise CalendarItemNotFound, honoring soft-deletion filtering."""
        clauses = ["id = ?"]
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        sql = f"SELECT * FROM calendar_annotations WHERE {' AND '.join(clauses)}"  # nosec B608
        row = conn.execute(sql, (annotation_id,)).fetchone()
        if row is None:
            raise CalendarItemNotFound(f"Calendar annotation not found: {annotation_id}")
        return self._annotation_from_row(row)

    def _get_link_row(self, conn: sqlite3.Connection, link_id: int) -> CalendarLinkRow:
        """Read a context link in the caller's connection or raise CalendarItemNotFound."""
        row = conn.execute("SELECT * FROM calendar_links WHERE id = ?", (link_id,)).fetchone()
        if row is None:
            raise CalendarItemNotFound(f"Calendar link not found: {link_id}")
        return self._link_from_row(row)

    def _get_external_account_row(
        self,
        conn: sqlite3.Connection,
        account_id: int,
        *,
        include_deleted: bool = False,
    ) -> ExternalCalendarAccountRow:
        """Read provider account state or raise CalendarNotFound, honoring soft-deletion filtering."""
        clauses = ["id = ?"]
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        sql = f"SELECT * FROM external_calendar_accounts WHERE {' AND '.join(clauses)}"  # nosec B608
        row = conn.execute(sql, (account_id,)).fetchone()
        if row is None:
            raise CalendarNotFound(f"External calendar account not found: {account_id}")
        return self._external_account_from_row(row)

    def _get_external_binding_row(
        self,
        conn: sqlite3.Connection,
        binding_id: int,
        *,
        include_deleted: bool = False,
    ) -> ExternalCalendarBindingRow:
        """Read remote binding state or raise CalendarNotFound, honoring soft-deletion filtering."""
        clauses = ["id = ?"]
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        sql = f"SELECT * FROM external_calendar_bindings WHERE {' AND '.join(clauses)}"  # nosec B608
        row = conn.execute(sql, (binding_id,)).fetchone()
        if row is None:
            raise CalendarNotFound(f"External calendar binding not found: {binding_id}")
        return self._external_binding_from_row(row)

    def _list_external_bindings_for_account_rows(
        self,
        conn: sqlite3.Connection,
        account_id: int,
        *,
        include_deleted: bool = False,
    ) -> list[ExternalCalendarBindingRow]:
        """Read an account's bindings by remote name/ID in the supplied connection, with optional tombstones."""
        clauses = ["account_id = ?"]
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        sql = f"""
            SELECT * FROM external_calendar_bindings
            WHERE {" AND ".join(clauses)}
            ORDER BY remote_display_name ASC, id ASC
            """  # nosec B608
        rows = conn.execute(sql, (account_id,)).fetchall()
        return [self._external_binding_from_row(row) for row in rows]

    def _get_sync_event_row(
        self,
        conn: sqlite3.Connection,
        event_id: int,
    ) -> CalendarSyncEventRow:
        """Read a sync-history event in the supplied connection or raise CalendarSyncError."""
        row = conn.execute("SELECT * FROM calendar_sync_events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            raise CalendarSyncError(f"Calendar sync event not found: {event_id}")
        return self._sync_event_from_row(row)

    @staticmethod
    def _calendar_from_row(row: sqlite3.Row) -> CalendarRow:
        """Return an immutable calendar snapshot, retaining raw JSON storage fields."""
        return CalendarRow(**dict(row))

    @staticmethod
    def _membership_from_row(row: sqlite3.Row) -> CalendarMembershipRow:
        """Return an immutable principal-role assignment from a complete membership row."""
        return CalendarMembershipRow(**dict(row))

    @staticmethod
    def _item_from_row(row: sqlite3.Row) -> CalendarItemRow:
        """Return an immutable item snapshot, normalizing integer ownership/all-day flags to booleans."""
        data = dict(row)
        data["provider_owned"] = bool(data["provider_owned"])
        data["all_day"] = bool(data["all_day"])
        return CalendarItemRow(**data)

    @staticmethod
    def _recurrence_from_row(row: sqlite3.Row) -> CalendarRecurrenceRow:
        """Return an immutable recurrence snapshot without decoding stored date-list JSON."""
        return CalendarRecurrenceRow(**dict(row))

    @staticmethod
    def _annotation_from_row(row: sqlite3.Row) -> CalendarAnnotationRow:
        """Return an immutable authored annotation snapshot with raw tag JSON."""
        return CalendarAnnotationRow(**dict(row))

    @staticmethod
    def _link_from_row(row: sqlite3.Row) -> CalendarLinkRow:
        """Return an immutable context-link snapshot with raw metadata JSON."""
        return CalendarLinkRow(**dict(row))

    @staticmethod
    def _external_account_from_row(row: sqlite3.Row) -> ExternalCalendarAccountRow:
        """Return an immutable account snapshot containing only an opaque credential reference."""
        return ExternalCalendarAccountRow(**dict(row))

    @staticmethod
    def _external_binding_from_row(row: sqlite3.Row) -> ExternalCalendarBindingRow:
        """Return an immutable binding snapshot, normalizing the stored sync-enabled flag to bool."""
        data = dict(row)
        data["sync_enabled"] = bool(data["sync_enabled"])
        return ExternalCalendarBindingRow(**data)

    @staticmethod
    def _sync_event_from_row(row: sqlite3.Row) -> CalendarSyncEventRow:
        """Return an immutable sync-history snapshot with counters and raw diagnostic JSON."""
        return CalendarSyncEventRow(**dict(row))


__all__ = [
    "CalendarAnnotationRow",
    "CalendarDatabase",
    "CalendarItemRow",
    "CalendarLinkRow",
    "CalendarMembershipRow",
    "CalendarRecurrenceRow",
    "CalendarRow",
    "CalendarSecretStore",
    "CalendarSyncEventRow",
    "ExternalCalendarAccountRow",
    "ExternalCalendarBindingRow",
]

"""
Collections_DB - Persistence for Content Collections feature slices.

Scope:
- Output Templates (CRUD + lookup for preview)
- Reading Highlights (CRUD + maintenance helpers)

Notes:
- Uses DatabaseBackendFactory to support SQLite (default) and future PG.
- Stores data in the per-user Media database by default to colocate content artifacts.
- All methods are idempotent and should avoid raising on existing schema.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import re
import threading
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.core.Collections.utils import (
    build_highlight_context,
    find_highlight_span,
    hash_text_sha256,
)
from tldw_Server_API.app.core.config import load_comprehensive_config, settings
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.DB_Management.content_backend import (
    backend_target_key,
    get_content_backend,
    load_content_db_settings,
)
from tldw_Server_API.app.core.exceptions import (
    InvalidStorageUserIdError,
    StorageUnavailableError,
)

from .backends.base import BackendType, DatabaseBackend, DatabaseConfig, DatabaseError
from .backends.factory import (
    DatabaseBackendFactory,
    is_factory_managed_backend,
    release_managed_backend,
)
from .backends.query_utils import prepare_backend_statement
from .db_path_utils import DatabasePaths, normalize_output_storage_filename

_COLLECTIONS_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    json.JSONDecodeError,
    DatabaseError,
    StorageUnavailableError,
    InvalidStorageUserIdError,
)


def _utcnow_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def _extract_output_byte_size(metadata_json: str | None) -> int | None:
    if not metadata_json:
        return None
    try:
        payload = json.loads(metadata_json)
    except _COLLECTIONS_NONCRITICAL_EXCEPTIONS:
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("byte_size")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _is_audiobook_output_type(type_value: str | None) -> bool:
    if not type_value:
        return False
    return str(type_value).startswith("audiobook_")


def _resolve_output_size_bytes(user_id: str, storage_path: str | None) -> int | None:
    if not storage_path:
        return None
    try:
        user_int = int(user_id)
    except (TypeError, ValueError):
        logger.warning("audiobook_quota: invalid user id for output size: {}", user_id)
        return None
    try:
        outputs_dir = DatabasePaths.get_user_outputs_dir(user_int)
        return (outputs_dir / storage_path).stat().st_size
    except FileNotFoundError:
        logger.warning("audiobook_quota: missing output file {}", storage_path)
    except OSError as exc:
        logger.warning("audiobook_quota: failed to stat {}: {}", storage_path, exc)
    return None

_SQLITE_PRAGMA_TABLES = {
    "audiobook_artifacts",
    "audiobook_chapters",
    "audiobook_projects",
    "collection_tags",
    "content_item_note_links",
    "content_item_tags",
    "content_items",
    "file_artifacts",
    "notification_preferences",
    "notification_bridge_state",
    "output_templates",
    "outputs",
    "reminder_task_runs",
    "reminder_tasks",
    "saved_searches",
    "reading_digest_schedules",
    "reading_highlights",
    "user_notifications",
}


def _is_backfill_noop_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "duplicate column" in message or "already exists" in message


@dataclass
class OutputTemplateRow:
    id: int
    user_id: str
    name: str
    type: str
    format: str
    body: str
    description: str | None
    is_default: bool
    created_at: str
    updated_at: str
    metadata_json: str | None = None


@dataclass
class HighlightRow:
    id: int
    user_id: str
    item_id: int
    quote: str
    start_offset: int | None
    end_offset: int | None
    color: str | None
    note: str | None
    created_at: str
    anchor_strategy: str
    content_hash_ref: str | None
    context_before: str | None
    context_after: str | None
    state: str


@dataclass
class CollectionTagRow:
    id: int
    user_id: str
    name: str


@dataclass
class ContentItemRow:
    id: int
    user_id: str
    origin: str
    origin_type: str | None
    origin_id: int | None
    url: str | None
    canonical_url: str | None
    domain: str | None
    title: str | None
    summary: str | None
    notes: str | None
    content_hash: str | None
    word_count: int | None
    published_at: str | None
    status: str | None
    favorite: bool
    metadata_json: str | None
    media_id: int | None
    job_id: int | None
    run_id: int | None
    source_id: int | None
    read_at: str | None
    created_at: str
    updated_at: str
    tags: list[str]
    is_new: bool = False
    content_changed: bool = False


@dataclass
class SavedSearchRow:
    id: int
    user_id: str
    name: str
    query_json: str
    sort: str | None
    created_at: str
    updated_at: str


@dataclass
class ContentItemNoteLinkRow:
    user_id: str
    item_id: int
    note_id: str
    created_at: str


@dataclass
class ReadingDigestScheduleRow:
    """Row model for reading_digest_schedules entries."""
    id: str
    tenant_id: str
    user_id: str
    name: str | None
    cron: str
    timezone: str | None
    enabled: bool
    require_online: bool
    filters_json: str
    template_id: int | None
    template_name: str | None
    format: str
    retention_days: int | None
    last_run_at: str | None
    next_run_at: str | None
    last_status: str | None
    created_at: str
    updated_at: str


@dataclass
class ReminderTaskRow:
    id: str
    user_id: str
    tenant_id: str
    title: str
    body: str | None
    link_type: str | None
    link_id: str | None
    link_url: str | None
    schedule_kind: str
    run_at: str | None
    cron: str | None
    timezone: str | None
    enabled: bool
    last_run_at: str | None
    next_run_at: str | None
    last_status: str | None
    created_at: str
    updated_at: str


@dataclass
class ReminderTaskRunRow:
    id: int
    task_id: str
    user_id: str
    scheduled_for: str | None
    job_id: str | None
    run_slot_utc: str
    run_slot_key: str
    status: str
    error: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None


@dataclass
class UserNotificationRow:
    id: int
    user_id: str
    kind: str
    title: str
    message: str
    severity: str
    source_task_id: str | None
    source_task_run_id: int | None
    source_job_id: str | None
    source_domain: str | None
    source_job_type: str | None
    link_type: str | None
    link_id: str | None
    link_url: str | None
    dedupe_key: str | None
    retention_until: str | None
    archived_at: str | None
    created_at: str
    read_at: str | None
    dismissed_at: str | None
    snooze_task_id: str | None = None
    delivery_status: str = "pending"
    delivered_at: str | None = None


@dataclass
class NotificationPreferencesRow:
    user_id: str
    reminder_enabled: bool
    job_completed_enabled: bool
    job_failed_enabled: bool
    updated_at: str


@dataclass
class NotificationBridgeStateRow:
    consumer_name: str
    last_event_id: int
    lease_owner_id: str | None
    lease_expires_at: str | None
    updated_at: str


@dataclass
class VoiceProfileRow:
    profile_id: str
    user_id: str
    name: str
    default_voice: str
    default_speed: float
    chapter_overrides_json: str | None
    created_at: str
    updated_at: str


@dataclass
class AudiobookProjectRow:
    id: int
    user_id: str
    project_id: str | None
    title: str | None
    source_ref: str | None
    status: str | None
    settings_json: str | None
    created_at: str
    updated_at: str


@dataclass
class AudiobookChapterRow:
    id: int
    project_id: int
    chapter_index: int
    title: str | None
    start_offset: int | None
    end_offset: int | None
    voice_profile_id: str | None
    speed: float | None
    metadata_json: str | None


@dataclass
class AudiobookArtifactRow:
    id: int
    project_id: int
    artifact_type: str
    format: str
    output_id: int
    metadata_json: str | None


def _pin_collections_public_operation(method):
    @functools.wraps(method)
    def wrapper(self: "CollectionsDatabase", *args: Any, **kwargs: Any):
        with self._operation_backend_pin():
            return method(self, *args, **kwargs)

    return wrapper


def _decorate_collections_public_operations(cls: type["CollectionsDatabase"]) -> type["CollectionsDatabase"]:
    for name, attribute in list(vars(cls).items()):
        if name.startswith("_") or name in {"ensure_schema", "transaction"}:
            continue
        if isinstance(attribute, (classmethod, staticmethod, property)):
            continue
        if not callable(attribute):
            continue
        setattr(cls, name, _pin_collections_public_operation(attribute))
    return cls


@_decorate_collections_public_operations
class CollectionsDatabase:
    """Adapter for Collections tables stored in the per-user Media DB."""

    _bootstrapped_backend_targets: set[str] = set()

    def __init__(self, user_id: int | str, backend: DatabaseBackend | None = None):
        self.user_id = str(user_id)
        self._owns_backend = False
        self._local = threading.local()
        self._backend: DatabaseBackend
        self._uses_shared_content_backend = False
        self._backend_refresh_suspended = False
        if backend is None:
            backend, uses_shared_backend = self._resolve_backend()
            self._uses_shared_content_backend = uses_shared_backend
        else:
            self._owns_backend = False
        self._backend = backend
        self._fts_available = True
        self._fts_supports_direct_delete = True
        if self._backend.backend_type == BackendType.POSTGRESQL:
            self._ensure_bootstrap_for_backend(self._backend)
        else:
            self._run_backend_bootstrap()

    @classmethod
    def for_user(cls, user_id: int | str) -> CollectionsDatabase:
        return cls(user_id=user_id)

    @classmethod
    def from_backend(cls, user_id: int | str, backend: DatabaseBackend) -> CollectionsDatabase:
        """Construct using an existing backend (avoids path resolution and int casting)."""
        return cls(user_id=user_id, backend=backend)

    @property
    def backend(self) -> DatabaseBackend:
        pinned_backend = self._get_pinned_backend()
        if pinned_backend is not None:
            return pinned_backend
        return self._refresh_backend_if_needed()

    @property
    def backend_type(self) -> BackendType:
        return self.backend.backend_type

    def _refresh_backend_if_needed(self) -> DatabaseBackend:
        if not self._uses_shared_content_backend:
            return self._backend
        if self._backend_refresh_suspended:
            return self._backend

        previous_owns_backend = self._owns_backend
        current_key = self._backend_target_key(self._backend)
        refreshed_backend, uses_shared_backend = self._resolve_backend()
        refreshed_key = self._backend_target_key(refreshed_backend)

        # A transient refresh failure must not demote an active shared backend
        # to a new SQLite backend mid-operation.
        if not uses_shared_backend or refreshed_backend.backend_type != BackendType.POSTGRESQL:
            self._owns_backend = previous_owns_backend
            if refreshed_backend is not self._backend:
                with contextlib.suppress(_COLLECTIONS_NONCRITICAL_EXCEPTIONS):
                    refreshed_backend.get_pool().close_all()
            logger.warning(
                "CollectionsDatabase refresh fell back to a non-shared backend while {} remained active; keeping the current backend",
                current_key or "<unknown>",
            )
            return self._backend

        if (
            uses_shared_backend
            and refreshed_backend.backend_type == BackendType.POSTGRESQL
            and refreshed_key != current_key
        ):
            self._ensure_bootstrap_for_backend(refreshed_backend)

        self._uses_shared_content_backend = uses_shared_backend
        self._backend = refreshed_backend
        return self._backend

    def _backend_target_key(self, backend: DatabaseBackend | None) -> str | None:
        return backend_target_key(backend)

    def _get_pinned_backend(self) -> DatabaseBackend | None:
        return getattr(self._local, "backend_pin", None)

    def _run_backend_bootstrap(self) -> None:
        self.ensure_schema()
        self._seed_watchlists_output_templates()

    def _mark_backend_bootstrapped(self, backend: DatabaseBackend | None) -> None:
        key = self._backend_target_key(backend)
        if key:
            type(self)._bootstrapped_backend_targets.add(key)

    def _ensure_bootstrap_for_backend(self, backend: DatabaseBackend) -> None:
        if backend.backend_type != BackendType.POSTGRESQL:
            return
        key = self._backend_target_key(backend)
        if key in type(self)._bootstrapped_backend_targets:
            return
        previous_suspend = self._backend_refresh_suspended
        self._backend_refresh_suspended = True
        try:
            self._backend = backend
            self._run_backend_bootstrap()
        finally:
            self._backend_refresh_suspended = previous_suspend
        if key:
            type(self)._bootstrapped_backend_targets.add(key)

    def _resolve_backend(self) -> tuple[DatabaseBackend, bool]:
        backend_mode_env = (os.getenv("TLDW_CONTENT_DB_BACKEND") or "").strip().lower()
        if backend_mode_env in {"postgres", "postgresql"}:
            parser = load_comprehensive_config()
            resolved = get_content_backend(parser)
            if resolved is None:
                raise DatabaseError("PostgreSQL content backend requested but not initialized")
            self._owns_backend = False
            return resolved, True

        try:
            parser = load_comprehensive_config()
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS:
            parser = None

        if parser is not None:
            try:
                content_settings = load_content_db_settings(parser)
                if content_settings.backend_type == BackendType.POSTGRESQL:
                    resolved = get_content_backend(parser)
                    if resolved is None:
                        raise DatabaseError("PostgreSQL content backend requested but not initialized")
                    self._owns_backend = False
                    return resolved, True
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS:
                pass

        db_path = str(DatabasePaths.get_media_db_path(int(self.user_id)))
        cfg = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=db_path)
        self._owns_backend = True
        return DatabaseBackendFactory.create_backend(cfg), False

    def close(self) -> None:
        """Release backend connections if this instance owns the backend."""
        if not self._owns_backend:
            return
        self._owns_backend = False
        if (
            getattr(self._backend, "backend_type", None) == BackendType.SQLITE
            and is_factory_managed_backend(self._backend)
        ):
            release_managed_backend(self._backend)
            return
        try:
            self._backend.get_pool().close_all()
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("collections_db: failed to close backend for user {}: {}", self.user_id, exc)

    def __enter__(self) -> CollectionsDatabase:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @contextlib.contextmanager
    def _operation_backend_pin(self) -> Generator[DatabaseBackend, None, None]:
        previous_backend = self._get_pinned_backend()
        if previous_backend is None:
            self._local.backend_pin = self._refresh_backend_if_needed()
        try:
            yield self._local.backend_pin
        finally:
            if previous_backend is None:
                with contextlib.suppress(AttributeError):
                    del self._local.backend_pin
            else:
                self._local.backend_pin = previous_backend

    @contextlib.contextmanager
    def transaction(self) -> Generator[Any, None, None]:
        """Pin the active backend for the duration of a transactional operation."""
        backend = self.backend
        previous_backend = self._get_pinned_backend()
        self._local.backend_pin = backend
        try:
            with backend.transaction() as conn:
                yield conn
        finally:
            if previous_backend is None:
                with contextlib.suppress(AttributeError):
                    del self._local.backend_pin
            else:
                self._local.backend_pin = previous_backend

    def _execute_insert(self, query: str, params: tuple[Any, ...]) -> Any:
        if self.backend.backend_type == BackendType.POSTGRESQL:
            prepared_query, prepared_params = prepare_backend_statement(
                BackendType.POSTGRESQL,
                query,
                params,
                apply_default_transform=True,
                ensure_returning=True,
            )
            return self.backend.execute(prepared_query, prepared_params)
        return self.backend.execute(query, params)

    @staticmethod
    def _extract_lastrowid(result: Any) -> int | None:
        try:
            if result is None:
                return None
            lastrowid = getattr(result, "lastrowid", None)
            if lastrowid:
                return int(lastrowid)
            first = getattr(result, "first", None)
            if isinstance(first, dict) and first.get("id") is not None:
                return int(first.get("id"))
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS:
            return None
        return None

    @staticmethod
    def _coerce_bool_flag(value: Any, *, postgres: bool) -> Any:
        if postgres:
            return bool(value)
        return 1 if value else 0

    @staticmethod
    def _coerce_bool_setting(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        raw = str(value).strip().lower()
        if is_truthy(raw):
            return True
        if raw in {"0", "false", "no", "off", "n"}:
            return False
        return default

    @staticmethod
    def _seeded_template_hash(body: str, description: str | None, fmt: str, type_: str) -> str | None:
        payload = json.dumps(
            {
                "body": body or "",
                "description": description or "",
                "format": fmt or "",
                "type": type_ or "",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hash_text_sha256(payload)

    def _sqlite_columns(self, table: str) -> set[str]:
        if self.backend.backend_type != BackendType.SQLITE:
            return set()
        if table not in _SQLITE_PRAGMA_TABLES:
            return set()
        try:
            result = self.backend.execute(f"PRAGMA table_info({table})", ())
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.exception(
                "collections_db: failed to read sqlite columns for table {}: {}",
                table,
                exc,
            )
            return set()
        columns: set[str] = set()
        for row in result.rows:
            name = row.get("name")
            if name:
                columns.add(str(name))
        return columns

    def _table_columns(self, table: str) -> set[str]:
        if self.backend.backend_type == BackendType.SQLITE:
            return self._sqlite_columns(table)
        return {
            str(row["name"])
            for row in self.backend.get_table_info(table)
            if row.get("name")
        }

    def _backfill_audiobook_project_ids(self) -> None:
        try:
            rows = self.backend.execute(
                "SELECT id, settings_json FROM audiobook_projects WHERE project_id IS NULL OR project_id = ''",
                (),
            ).rows
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("collections backfill: audiobook_projects.project_id fetch failed: {}", exc)
            return
        updated = 0
        for row in rows:
            raw_settings = row.get("settings_json")
            if not raw_settings:
                continue
            try:
                settings = json.loads(raw_settings)
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS:
                continue
            project_id = settings.get("project_id") if isinstance(settings, dict) else None
            if not isinstance(project_id, str) or not project_id:
                continue
            try:
                self.backend.execute(
                    "UPDATE audiobook_projects SET project_id = ? WHERE id = ? AND user_id = ?",
                    (project_id, row.get("id"), self.user_id),
                )
                updated += 1
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("collections backfill: audiobook_projects.project_id update failed: {}", exc)
        if updated:
            logger.debug("collections backfill: audiobook_projects.project_id updated {} rows", updated)

    def ensure_schema(self) -> None:
        """Create tables if they do not already exist."""
        if self.backend.backend_type == BackendType.POSTGRESQL:
            ddl = """
            CREATE TABLE IF NOT EXISTS output_templates (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                format TEXT NOT NULL,
                body TEXT NOT NULL,
                description TEXT,
                is_default BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_output_templates_user ON output_templates(user_id);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_output_templates_user_name ON output_templates(user_id, name);

            CREATE TABLE IF NOT EXISTS reading_highlights (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                item_id BIGINT NOT NULL,
                quote TEXT NOT NULL,
                start_offset INTEGER,
                end_offset INTEGER,
                color TEXT,
                note TEXT,
                created_at TEXT NOT NULL,
                anchor_strategy TEXT NOT NULL DEFAULT 'fuzzy_quote',
                content_hash_ref TEXT,
                context_before TEXT,
                context_after TEXT,
                state TEXT NOT NULL DEFAULT 'active'
            );
            CREATE INDEX IF NOT EXISTS idx_highlights_user_item ON reading_highlights(user_id, item_id);

            CREATE TABLE IF NOT EXISTS outputs (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                job_id BIGINT,
                run_id BIGINT,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                format TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                metadata_json TEXT,
                workspace_tag TEXT,
                created_at TEXT NOT NULL,
                media_item_id BIGINT,
                chatbook_path TEXT,
                deleted BOOLEAN NOT NULL DEFAULT FALSE,
                deleted_at TEXT,
                retention_until TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_outputs_user ON outputs(user_id);
            CREATE INDEX IF NOT EXISTS idx_outputs_run ON outputs(run_id);
            -- NOTE: workspace_tag/deleted indexes are created after backfill to avoid schema init failures

            CREATE TABLE IF NOT EXISTS reading_digest_schedules (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                name TEXT,
                cron TEXT NOT NULL,
                timezone TEXT,
                enabled BOOLEAN NOT NULL DEFAULT TRUE,
                require_online BOOLEAN NOT NULL DEFAULT FALSE,
                filters_json TEXT NOT NULL DEFAULT '{}',
                template_id BIGINT,
                template_name TEXT,
                format TEXT NOT NULL DEFAULT 'md',
                retention_days INTEGER,
                last_run_at TEXT,
                next_run_at TEXT,
                last_status TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_reading_digest_user ON reading_digest_schedules(user_id);
            CREATE INDEX IF NOT EXISTS idx_reading_digest_tenant ON reading_digest_schedules(tenant_id);

            CREATE TABLE IF NOT EXISTS reminder_tasks (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT,
                link_type TEXT,
                link_id TEXT,
                link_url TEXT,
                schedule_kind TEXT NOT NULL,
                run_at TEXT,
                cron TEXT,
                timezone TEXT,
                enabled BOOLEAN NOT NULL DEFAULT TRUE,
                last_run_at TEXT,
                next_run_at TEXT,
                last_status TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_reminder_tasks_user ON reminder_tasks(user_id);
            CREATE INDEX IF NOT EXISTS idx_reminder_tasks_user_enabled ON reminder_tasks(user_id, enabled);

            CREATE TABLE IF NOT EXISTS reminder_task_runs (
                id BIGSERIAL PRIMARY KEY,
                task_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                scheduled_for TEXT,
                job_id TEXT,
                run_slot_utc TEXT NOT NULL,
                run_slot_key TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_reminder_task_runs_task_id ON reminder_task_runs(task_id);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_reminder_task_runs_slot ON reminder_task_runs(task_id, run_slot_key);

            CREATE TABLE IF NOT EXISTS user_notifications (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                source_task_id TEXT,
                source_task_run_id BIGINT,
                source_job_id TEXT,
                source_domain TEXT,
                source_job_type TEXT,
                link_type TEXT,
                link_id TEXT,
                link_url TEXT,
                dedupe_key TEXT,
                retention_until TEXT,
                archived_at TEXT,
                created_at TEXT NOT NULL,
                read_at TEXT,
                dismissed_at TEXT,
                snooze_task_id TEXT,
                delivery_status TEXT NOT NULL DEFAULT 'pending',
                delivered_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_user_notifications_user_created ON user_notifications(user_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_user_notifications_user_unread ON user_notifications(user_id, read_at);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_user_notifications_user_dedupe ON user_notifications(user_id, dedupe_key) WHERE dedupe_key IS NOT NULL;

            CREATE TABLE IF NOT EXISTS notification_preferences (
                user_id TEXT PRIMARY KEY,
                reminder_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                job_completed_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                job_failed_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS notification_bridge_state (
                consumer_name TEXT PRIMARY KEY,
                last_event_id BIGINT,
                lease_owner_id TEXT,
                lease_expires_at TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS file_artifacts (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                file_type TEXT NOT NULL,
                title TEXT NOT NULL,
                structured_json TEXT NOT NULL,
                validation_json TEXT NOT NULL,
                export_status TEXT NOT NULL DEFAULT 'none',
                export_format TEXT,
                export_storage_path TEXT,
                export_bytes BIGINT,
                export_content_type TEXT,
                export_job_id TEXT,
                export_expires_at TEXT,
                export_consumed_at TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted BOOLEAN NOT NULL DEFAULT FALSE,
                deleted_at TEXT,
                retention_until TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_file_artifacts_user ON file_artifacts(user_id);
            CREATE INDEX IF NOT EXISTS idx_file_artifacts_user_type ON file_artifacts(user_id, file_type);
            CREATE INDEX IF NOT EXISTS idx_file_artifacts_created ON file_artifacts(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_file_artifacts_export_status ON file_artifacts(export_status);
            CREATE INDEX IF NOT EXISTS idx_file_artifacts_retention_until ON file_artifacts(retention_until);
            CREATE INDEX IF NOT EXISTS idx_file_artifacts_deleted_at ON file_artifacts(deleted_at);
            CREATE INDEX IF NOT EXISTS idx_file_artifacts_export_expires_at ON file_artifacts(export_expires_at);

            CREATE TABLE IF NOT EXISTS audiobook_projects (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                project_id TEXT,
                title TEXT,
                source_ref TEXT,
                status TEXT,
                settings_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_audiobook_projects_user ON audiobook_projects(user_id);

            CREATE TABLE IF NOT EXISTS audiobook_chapters (
                id BIGSERIAL PRIMARY KEY,
                project_id BIGINT NOT NULL,
                chapter_index INTEGER NOT NULL,
                title TEXT,
                start_offset INTEGER,
                end_offset INTEGER,
                voice_profile_id TEXT,
                speed DOUBLE PRECISION,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_audiobook_chapters_project ON audiobook_chapters(project_id);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_audiobook_chapters_project_index ON audiobook_chapters(project_id, chapter_index);

            CREATE TABLE IF NOT EXISTS audiobook_artifacts (
                id BIGSERIAL PRIMARY KEY,
                project_id BIGINT NOT NULL,
                artifact_type TEXT NOT NULL,
                format TEXT NOT NULL,
                output_id BIGINT NOT NULL,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_audiobook_artifacts_project ON audiobook_artifacts(project_id);
            CREATE INDEX IF NOT EXISTS idx_audiobook_artifacts_output ON audiobook_artifacts(output_id);

            CREATE TABLE IF NOT EXISTS audiobook_voice_profiles (
                profile_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                default_voice TEXT NOT NULL,
                default_speed DOUBLE PRECISION NOT NULL,
                chapter_overrides_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_audiobook_voice_profiles_user ON audiobook_voice_profiles(user_id);

            CREATE TABLE IF NOT EXISTS audiobook_output_usage (
                user_id TEXT PRIMARY KEY,
                used_bytes BIGINT NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            """
        else:
            ddl = """
            CREATE TABLE IF NOT EXISTS output_templates (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                format TEXT NOT NULL,
                body TEXT NOT NULL,
                description TEXT,
                is_default INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_output_templates_user ON output_templates(user_id);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_output_templates_user_name ON output_templates(user_id, name);

            CREATE TABLE IF NOT EXISTS reading_highlights (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                item_id INTEGER NOT NULL,
                quote TEXT NOT NULL,
                start_offset INTEGER,
                end_offset INTEGER,
                color TEXT,
                note TEXT,
                created_at TEXT NOT NULL,
                anchor_strategy TEXT NOT NULL DEFAULT 'fuzzy_quote',
                content_hash_ref TEXT,
                context_before TEXT,
                context_after TEXT,
                state TEXT NOT NULL DEFAULT 'active'
            );
            CREATE INDEX IF NOT EXISTS idx_highlights_user_item ON reading_highlights(user_id, item_id);

            CREATE TABLE IF NOT EXISTS outputs (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                job_id INTEGER,
                run_id INTEGER,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                format TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                metadata_json TEXT,
                workspace_tag TEXT,
                created_at TEXT NOT NULL,
                media_item_id INTEGER,
                chatbook_path TEXT,
                deleted INTEGER NOT NULL DEFAULT 0,
                deleted_at TEXT,
                retention_until TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_outputs_user ON outputs(user_id);
            CREATE INDEX IF NOT EXISTS idx_outputs_run ON outputs(run_id);
            -- NOTE: workspace_tag/deleted indexes are created after backfill to avoid schema init failures

            CREATE TABLE IF NOT EXISTS reading_digest_schedules (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                name TEXT,
                cron TEXT NOT NULL,
                timezone TEXT,
                enabled INTEGER NOT NULL DEFAULT 1,
                require_online INTEGER NOT NULL DEFAULT 0,
                filters_json TEXT NOT NULL DEFAULT '{}',
                template_id INTEGER,
                template_name TEXT,
                format TEXT NOT NULL DEFAULT 'md',
                retention_days INTEGER,
                last_run_at TEXT,
                next_run_at TEXT,
                last_status TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_reading_digest_user ON reading_digest_schedules(user_id);
            CREATE INDEX IF NOT EXISTS idx_reading_digest_tenant ON reading_digest_schedules(tenant_id);

            CREATE TABLE IF NOT EXISTS reminder_tasks (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT,
                link_type TEXT,
                link_id TEXT,
                link_url TEXT,
                schedule_kind TEXT NOT NULL,
                run_at TEXT,
                cron TEXT,
                timezone TEXT,
                enabled INTEGER NOT NULL DEFAULT 1,
                last_run_at TEXT,
                next_run_at TEXT,
                last_status TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_reminder_tasks_user ON reminder_tasks(user_id);
            CREATE INDEX IF NOT EXISTS idx_reminder_tasks_user_enabled ON reminder_tasks(user_id, enabled);

            CREATE TABLE IF NOT EXISTS reminder_task_runs (
                id INTEGER PRIMARY KEY,
                task_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                scheduled_for TEXT,
                job_id TEXT,
                run_slot_utc TEXT NOT NULL,
                run_slot_key TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_reminder_task_runs_task_id ON reminder_task_runs(task_id);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_reminder_task_runs_slot ON reminder_task_runs(task_id, run_slot_key);

            CREATE TABLE IF NOT EXISTS user_notifications (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                source_task_id TEXT,
                source_task_run_id INTEGER,
                source_job_id TEXT,
                source_domain TEXT,
                source_job_type TEXT,
                link_type TEXT,
                link_id TEXT,
                link_url TEXT,
                dedupe_key TEXT,
                retention_until TEXT,
                archived_at TEXT,
                created_at TEXT NOT NULL,
                read_at TEXT,
                dismissed_at TEXT,
                snooze_task_id TEXT,
                delivery_status TEXT NOT NULL DEFAULT 'pending',
                delivered_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_user_notifications_user_created ON user_notifications(user_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_user_notifications_user_unread ON user_notifications(user_id, read_at);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_user_notifications_user_dedupe ON user_notifications(user_id, dedupe_key);

            CREATE TABLE IF NOT EXISTS notification_preferences (
                user_id TEXT PRIMARY KEY,
                reminder_enabled INTEGER NOT NULL DEFAULT 1,
                job_completed_enabled INTEGER NOT NULL DEFAULT 1,
                job_failed_enabled INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS notification_bridge_state (
                consumer_name TEXT PRIMARY KEY,
                last_event_id INTEGER,
                lease_owner_id TEXT,
                lease_expires_at TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS file_artifacts (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                file_type TEXT NOT NULL,
                title TEXT NOT NULL,
                structured_json TEXT NOT NULL,
                validation_json TEXT NOT NULL,
                export_status TEXT NOT NULL DEFAULT 'none',
                export_format TEXT,
                export_storage_path TEXT,
                export_bytes INTEGER,
                export_content_type TEXT,
                export_job_id TEXT,
                export_expires_at TEXT,
                export_consumed_at TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0,
                deleted_at TEXT,
                retention_until TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_file_artifacts_user ON file_artifacts(user_id);
            CREATE INDEX IF NOT EXISTS idx_file_artifacts_user_type ON file_artifacts(user_id, file_type);
            CREATE INDEX IF NOT EXISTS idx_file_artifacts_created ON file_artifacts(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_file_artifacts_export_status ON file_artifacts(export_status);
            CREATE INDEX IF NOT EXISTS idx_file_artifacts_retention_until ON file_artifacts(retention_until);
            CREATE INDEX IF NOT EXISTS idx_file_artifacts_deleted_at ON file_artifacts(deleted_at);
            CREATE INDEX IF NOT EXISTS idx_file_artifacts_export_expires_at ON file_artifacts(export_expires_at);

            CREATE TABLE IF NOT EXISTS audiobook_projects (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                project_id TEXT,
                title TEXT,
                source_ref TEXT,
                status TEXT,
                settings_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_audiobook_projects_user ON audiobook_projects(user_id);

            CREATE TABLE IF NOT EXISTS audiobook_chapters (
                id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL,
                chapter_index INTEGER NOT NULL,
                title TEXT,
                start_offset INTEGER,
                end_offset INTEGER,
                voice_profile_id TEXT,
                speed REAL,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_audiobook_chapters_project ON audiobook_chapters(project_id);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_audiobook_chapters_project_index ON audiobook_chapters(project_id, chapter_index);

            CREATE TABLE IF NOT EXISTS audiobook_artifacts (
                id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL,
                artifact_type TEXT NOT NULL,
                format TEXT NOT NULL,
                output_id INTEGER NOT NULL,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_audiobook_artifacts_project ON audiobook_artifacts(project_id);
            CREATE INDEX IF NOT EXISTS idx_audiobook_artifacts_output ON audiobook_artifacts(output_id);

            CREATE TABLE IF NOT EXISTS audiobook_voice_profiles (
                profile_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                default_voice TEXT NOT NULL,
                default_speed REAL NOT NULL,
                chapter_overrides_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_audiobook_voice_profiles_user ON audiobook_voice_profiles(user_id);

            CREATE TABLE IF NOT EXISTS audiobook_output_usage (
                user_id TEXT PRIMARY KEY,
                used_bytes INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            """
        try:
            self.backend.create_tables(ddl)
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Collections schema init failed: {e}")
            raise
        output_template_columns: set[str] = set()
        output_columns: set[str] = set()
        digest_columns: set[str] = set()
        file_artifact_columns: set[str] = set()
        content_columns: set[str] = set()
        audiobook_project_columns: set[str] = set()
        if self.backend.backend_type == BackendType.SQLITE:
            output_template_columns = self._sqlite_columns("output_templates")
            output_columns = self._sqlite_columns("outputs")
            digest_columns = self._sqlite_columns("reading_digest_schedules")
            file_artifact_columns = self._sqlite_columns("file_artifacts")
            content_columns = self._sqlite_columns("content_items")
            audiobook_project_columns = self._sqlite_columns("audiobook_projects")
        # Backfill columns for existing tables
        if "metadata_json" not in output_template_columns:
            try:
                # Attempt to add metadata_json if missing
                self.backend.execute("ALTER TABLE output_templates ADD COLUMN metadata_json TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: output_templates.metadata_json already exists or skipped")
                else:
                    raise
        # Outputs table backfills
        if "deleted" not in output_columns:
            try:
                deleted_type = "BOOLEAN" if self.backend.backend_type == BackendType.POSTGRESQL else "INTEGER"
                deleted_default = "FALSE" if self.backend.backend_type == BackendType.POSTGRESQL else "0"
                self.backend.execute(
                    f"ALTER TABLE outputs ADD COLUMN deleted {deleted_type} NOT NULL DEFAULT {deleted_default}",
                    (),
                )
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: outputs.deleted already exists or skipped")
                else:
                    raise
        if "deleted_at" not in output_columns:
            try:
                self.backend.execute("ALTER TABLE outputs ADD COLUMN deleted_at TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: outputs.deleted_at already exists or skipped")
                else:
                    raise
        if "retention_until" not in output_columns:
            try:
                self.backend.execute("ALTER TABLE outputs ADD COLUMN retention_until TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: outputs.retention_until already exists or skipped")
                else:
                    raise
        if "workspace_tag" not in output_columns:
            try:
                self.backend.execute("ALTER TABLE outputs ADD COLUMN workspace_tag TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: outputs.workspace_tag already exists or skipped")
                else:
                    raise
        try:
            self.backend.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_outputs_user_title_format ON outputs(user_id, title, format) WHERE deleted = 0", ())
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
            if _is_backfill_noop_error(exc):
                logger.debug("collections backfill: outputs unique index already exists or skipped")
            else:
                raise
        try:
            self.backend.execute("CREATE INDEX IF NOT EXISTS idx_outputs_workspace_tag ON outputs(workspace_tag)", ())
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
            if _is_backfill_noop_error(exc):
                logger.debug("collections backfill: outputs workspace_tag index already exists or skipped")
            else:
                raise
        # Reading digest schedule backfills
        if "enabled" not in digest_columns:
            try:
                enabled_type = "BOOLEAN" if self.backend.backend_type == BackendType.POSTGRESQL else "INTEGER"
                enabled_default = "TRUE" if self.backend.backend_type == BackendType.POSTGRESQL else "1"
                self.backend.execute(
                    f"ALTER TABLE reading_digest_schedules ADD COLUMN enabled {enabled_type} NOT NULL DEFAULT {enabled_default}",
                    (),
                )
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: reading_digest_schedules.enabled already exists or skipped")
                else:
                    raise
        if "require_online" not in digest_columns:
            try:
                online_type = "BOOLEAN" if self.backend.backend_type == BackendType.POSTGRESQL else "INTEGER"
                online_default = "FALSE" if self.backend.backend_type == BackendType.POSTGRESQL else "0"
                self.backend.execute(
                    f"ALTER TABLE reading_digest_schedules ADD COLUMN require_online {online_type} NOT NULL DEFAULT {online_default}",
                    (),
                )
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: reading_digest_schedules.require_online already exists or skipped")
                else:
                    raise
        if "filters_json" not in digest_columns:
            try:
                self.backend.execute("ALTER TABLE reading_digest_schedules ADD COLUMN filters_json TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: reading_digest_schedules.filters_json already exists or skipped")
                else:
                    raise
        if "template_id" not in digest_columns:
            try:
                template_id_type = "BIGINT" if self.backend.backend_type == BackendType.POSTGRESQL else "INTEGER"
                self.backend.execute(f"ALTER TABLE reading_digest_schedules ADD COLUMN template_id {template_id_type}", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: reading_digest_schedules.template_id already exists or skipped")
                else:
                    raise
        if "template_name" not in digest_columns:
            try:
                self.backend.execute("ALTER TABLE reading_digest_schedules ADD COLUMN template_name TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: reading_digest_schedules.template_name already exists or skipped")
                else:
                    raise
        if "format" not in digest_columns:
            try:
                self.backend.execute("ALTER TABLE reading_digest_schedules ADD COLUMN format TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: reading_digest_schedules.format already exists or skipped")
                else:
                    raise
        if "retention_days" not in digest_columns:
            try:
                self.backend.execute("ALTER TABLE reading_digest_schedules ADD COLUMN retention_days INTEGER", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: reading_digest_schedules.retention_days already exists or skipped")
                else:
                    raise
        if "last_run_at" not in digest_columns:
            try:
                self.backend.execute("ALTER TABLE reading_digest_schedules ADD COLUMN last_run_at TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: reading_digest_schedules.last_run_at already exists or skipped")
                else:
                    raise
        if "next_run_at" not in digest_columns:
            try:
                self.backend.execute("ALTER TABLE reading_digest_schedules ADD COLUMN next_run_at TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: reading_digest_schedules.next_run_at already exists or skipped")
                else:
                    raise
        if "last_status" not in digest_columns:
            try:
                self.backend.execute("ALTER TABLE reading_digest_schedules ADD COLUMN last_status TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: reading_digest_schedules.last_status already exists or skipped")
                else:
                    raise
        # File artifacts backfills
        if "deleted" not in file_artifact_columns:
            try:
                deleted_type = "BOOLEAN" if self.backend.backend_type == BackendType.POSTGRESQL else "INTEGER"
                deleted_default = "FALSE" if self.backend.backend_type == BackendType.POSTGRESQL else "0"
                self.backend.execute(
                    f"ALTER TABLE file_artifacts ADD COLUMN deleted {deleted_type} NOT NULL DEFAULT {deleted_default}",
                    (),
                )
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: file_artifacts.deleted already exists or skipped")
                else:
                    raise
        if "deleted_at" not in file_artifact_columns:
            try:
                self.backend.execute("ALTER TABLE file_artifacts ADD COLUMN deleted_at TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: file_artifacts.deleted_at already exists or skipped")
                else:
                    raise
        if "retention_until" not in file_artifact_columns:
            try:
                self.backend.execute("ALTER TABLE file_artifacts ADD COLUMN retention_until TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: file_artifacts.retention_until already exists or skipped")
                else:
                    raise
        if "export_status" not in file_artifact_columns:
            try:
                self.backend.execute("ALTER TABLE file_artifacts ADD COLUMN export_status TEXT NOT NULL DEFAULT 'none'", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: file_artifacts.export_status already exists or skipped")
                else:
                    raise
        if "export_format" not in file_artifact_columns:
            try:
                self.backend.execute("ALTER TABLE file_artifacts ADD COLUMN export_format TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: file_artifacts.export_format already exists or skipped")
                else:
                    raise
        if "export_storage_path" not in file_artifact_columns:
            try:
                self.backend.execute("ALTER TABLE file_artifacts ADD COLUMN export_storage_path TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: file_artifacts.export_storage_path already exists or skipped")
                else:
                    raise
        if "export_bytes" not in file_artifact_columns:
            try:
                export_bytes_type = "BIGINT" if self.backend.backend_type == BackendType.POSTGRESQL else "INTEGER"
                self.backend.execute(f"ALTER TABLE file_artifacts ADD COLUMN export_bytes {export_bytes_type}", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: file_artifacts.export_bytes already exists or skipped")
                else:
                    raise
        if "export_content_type" not in file_artifact_columns:
            try:
                self.backend.execute("ALTER TABLE file_artifacts ADD COLUMN export_content_type TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: file_artifacts.export_content_type already exists or skipped")
                else:
                    raise
        if "export_job_id" not in file_artifact_columns:
            try:
                self.backend.execute("ALTER TABLE file_artifacts ADD COLUMN export_job_id TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: file_artifacts.export_job_id already exists or skipped")
                else:
                    raise
        if "export_expires_at" not in file_artifact_columns:
            try:
                self.backend.execute("ALTER TABLE file_artifacts ADD COLUMN export_expires_at TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: file_artifacts.export_expires_at already exists or skipped")
                else:
                    raise
        if "export_consumed_at" not in file_artifact_columns:
            try:
                self.backend.execute("ALTER TABLE file_artifacts ADD COLUMN export_consumed_at TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: file_artifacts.export_consumed_at already exists or skipped")
                else:
                    raise
        # Audiobook projects backfills
        if self.backend.backend_type == BackendType.POSTGRESQL:
            try:
                self.backend.execute("ALTER TABLE audiobook_projects ADD COLUMN IF NOT EXISTS project_id TEXT", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: audiobook_projects.project_id already exists or skipped")
                else:
                    raise
            try:
                self.backend.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audiobook_projects_project_id ON audiobook_projects(user_id, project_id)",
                    (),
                )
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: audiobook_projects.project_id index already exists or skipped")
                else:
                    raise
        else:
            if "project_id" not in audiobook_project_columns:
                try:
                    self.backend.execute("ALTER TABLE audiobook_projects ADD COLUMN project_id TEXT", ())
                except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                    if _is_backfill_noop_error(exc):
                        logger.debug("collections backfill: audiobook_projects.project_id already exists or skipped")
                    else:
                        raise
            try:
                self.backend.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audiobook_projects_project_id ON audiobook_projects(user_id, project_id)",
                    (),
                )
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: audiobook_projects.project_id index already exists or skipped")
                else:
                    raise
        self._backfill_audiobook_project_ids()
        # Collections layer tables
        fts_available = self.backend.backend_type == BackendType.SQLITE
        if self.backend.backend_type == BackendType.POSTGRESQL:
            content_ddl = """
            CREATE TABLE IF NOT EXISTS collection_tags (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                UNIQUE (user_id, name)
            );

            CREATE TABLE IF NOT EXISTS content_items (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                origin TEXT NOT NULL,
                origin_type TEXT,
                origin_id BIGINT,
                url TEXT,
                canonical_url TEXT,
                domain TEXT,
                title TEXT,
                summary TEXT,
                notes TEXT,
                content_hash TEXT,
                word_count INTEGER,
                published_at TEXT,
                status TEXT,
                favorite INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT,
                media_id BIGINT,
                job_id BIGINT,
                run_id BIGINT,
                source_id BIGINT,
                read_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS ux_content_items_user_canonical ON content_items(user_id, canonical_url) WHERE canonical_url IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS ux_content_items_user_hash ON content_items(user_id, content_hash) WHERE content_hash IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_content_items_user_updated ON content_items(user_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_content_items_user_domain ON content_items(user_id, domain);
            CREATE INDEX IF NOT EXISTS idx_content_items_job ON content_items(job_id);
            CREATE INDEX IF NOT EXISTS idx_content_items_run ON content_items(run_id);

            CREATE TABLE IF NOT EXISTS content_item_tags (
                item_id BIGINT NOT NULL,
                tag_id BIGINT NOT NULL,
                UNIQUE (item_id, tag_id)
            );

            CREATE TABLE IF NOT EXISTS saved_searches (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                query_json TEXT NOT NULL,
                sort TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ux_saved_searches_user_name ON saved_searches(user_id, name);
            CREATE INDEX IF NOT EXISTS idx_saved_searches_user_updated ON saved_searches(user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS content_item_note_links (
                user_id TEXT NOT NULL,
                item_id BIGINT NOT NULL,
                note_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (user_id, item_id, note_id)
            );
            CREATE INDEX IF NOT EXISTS idx_content_item_note_links_user_item
                ON content_item_note_links(user_id, item_id, created_at DESC);
            """
        else:
            content_ddl = """
            CREATE TABLE IF NOT EXISTS collection_tags (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                UNIQUE (user_id, name)
            );

            CREATE TABLE IF NOT EXISTS content_items (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                origin TEXT NOT NULL,
                origin_type TEXT,
                origin_id INTEGER,
                url TEXT,
                canonical_url TEXT,
                domain TEXT,
                title TEXT,
                summary TEXT,
                notes TEXT,
                content_hash TEXT,
                word_count INTEGER,
                published_at TEXT,
                status TEXT,
                favorite INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT,
                media_id INTEGER,
                job_id INTEGER,
                run_id INTEGER,
                source_id INTEGER,
                read_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS ux_content_items_user_canonical ON content_items(user_id, canonical_url) WHERE canonical_url IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS ux_content_items_user_hash ON content_items(user_id, content_hash) WHERE content_hash IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_content_items_user_updated ON content_items(user_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_content_items_user_domain ON content_items(user_id, domain);
            CREATE INDEX IF NOT EXISTS idx_content_items_job ON content_items(job_id);
            CREATE INDEX IF NOT EXISTS idx_content_items_run ON content_items(run_id);

            CREATE TABLE IF NOT EXISTS content_item_tags (
                item_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                UNIQUE (item_id, tag_id)
            );

            CREATE TABLE IF NOT EXISTS saved_searches (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                query_json TEXT NOT NULL,
                sort TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (user_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_saved_searches_user_updated ON saved_searches(user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS content_item_note_links (
                user_id TEXT NOT NULL,
                item_id INTEGER NOT NULL,
                note_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (user_id, item_id, note_id)
            );
            CREATE INDEX IF NOT EXISTS idx_content_item_note_links_user_item
                ON content_item_note_links(user_id, item_id, created_at DESC);
            """
        try:
            self.backend.create_tables(content_ddl)
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Collections content_items schema init failed: {e}")
            raise
        if fts_available:
            try:
                self.backend.create_tables(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS content_items_fts USING fts5(
                        title,
                        summary,
                        metadata,
                        content=''
                    );
                    """
                )
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Collections FTS unavailable: {e}")
                fts_available = False
        # Backfill columns for prior installs if table existed
        _backfill_columns: dict[str, str] = {
            "origin_type": "TEXT",
            "origin_id": "INTEGER",
            "job_id": "INTEGER",
            "run_id": "INTEGER",
            "source_id": "INTEGER",
            "read_at": "TEXT",
            "notes": "TEXT",
        }
        for column, col_type in _backfill_columns.items():
            if column in content_columns:
                continue
            try:
                self.backend.execute(f"ALTER TABLE content_items ADD COLUMN {column} {col_type}", ())
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: content_items.{} already exists or skipped", column)
                else:
                    raise
        # Backfill user_notifications columns
        notif_columns: set[str] = set()
        if self.backend.table_exists("user_notifications"):
            with contextlib.suppress(*_COLLECTIONS_NONCRITICAL_EXCEPTIONS):
                notif_columns = self._table_columns("user_notifications")
        if notif_columns and "delivery_status" not in notif_columns:
            try:
                self.backend.execute(
                    "ALTER TABLE user_notifications ADD COLUMN delivery_status TEXT NOT NULL DEFAULT 'pending'",
                    (),
                )
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: user_notifications.delivery_status already exists or skipped")
                else:
                    raise
        if notif_columns and "delivered_at" not in notif_columns:
            try:
                self.backend.execute(
                    "ALTER TABLE user_notifications ADD COLUMN delivered_at TEXT",
                    (),
                )
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: user_notifications.delivered_at already exists or skipped")
                else:
                    raise
        if notif_columns and "snooze_task_id" not in notif_columns:
            try:
                self.backend.execute(
                    "ALTER TABLE user_notifications ADD COLUMN snooze_task_id TEXT",
                    (),
                )
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                if _is_backfill_noop_error(exc):
                    logger.debug("collections backfill: user_notifications.snooze_task_id already exists or skipped")
                else:
                    raise

        self._fts_available = fts_available
        self._refresh_fts_capabilities()

    def _refresh_fts_capabilities(self) -> None:
        """Refresh FTS behavior flags based on the concrete virtual table definition."""
        self._fts_supports_direct_delete = True
        if not self._fts_available:
            return
        if self.backend.backend_type != BackendType.SQLITE:
            return
        try:
            row = self.backend.execute(
                """
                SELECT sql
                FROM sqlite_master
                WHERE name = 'content_items_fts'
                LIMIT 1
                """,
                (),
            ).first
            ddl = str((row or {}).get("sql") or "").lower()
            # content='' tables are contentless unless explicitly created with contentless_delete=1.
            if "content=''" in ddl and "contentless_delete=1" not in ddl:
                self._fts_supports_direct_delete = False
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("collections: failed to inspect content_items_fts capabilities: {}", exc)

    @staticmethod
    def _build_content_fts_metadata_text(
        *,
        notes: str | None,
        tags: Iterable[str] | None,
        metadata_json: str | None,
    ) -> str:
        metadata_text = metadata_json or ""
        if notes:
            metadata_text = f"{metadata_text}\n{notes}" if metadata_text else notes
        if tags:
            tag_text = " ".join([str(tag).strip() for tag in tags if tag])
            if tag_text:
                metadata_text = f"{metadata_text}\n{tag_text}" if metadata_text else tag_text
        return metadata_text

    def _seed_watchlists_output_templates(self) -> None:
        seed_setting = settings.get("WATCHLISTS_SEED_OUTPUT_TEMPLATES")
        if seed_setting is None:
            seed_setting = os.getenv("WATCHLISTS_SEED_OUTPUT_TEMPLATES")
        if not self._coerce_bool_setting(seed_setting, True):
            return

        try:
            from tldw_Server_API.app.core.Watchlists import template_store
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("collections: watchlists template store unavailable: {}", exc)
            return

        try:
            summaries = template_store.list_templates()
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("collections: failed to list watchlists templates: {}", exc)
            return

        if not summaries:
            return

        existing = self._list_output_template_names()
        seen: set[str] = set()
        for summary in summaries:
            name = summary.name
            if name in seen:
                continue
            seen.add(name)
            try:
                record = template_store.load_template(name)
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("collections: failed to load watchlists template {}: {}", name, exc)
                continue
            fmt = record.format.lower()
            type_ = self._infer_output_template_type(record.name, fmt)
            seeded_hash = self._seeded_template_hash(record.content, record.description, fmt, type_)
            desired_meta = {
                "seeded_from": "watchlists_templates",
                "seeded_hash": seeded_hash,
                "seeded_mtime": record.updated_at,
            }
            metadata_json = json.dumps(desired_meta, ensure_ascii=False)

            if name not in existing:
                try:
                    self.create_output_template(
                        name=record.name,
                        type_=type_,
                        format_=fmt,
                        body=record.content,
                        description=record.description,
                        is_default=False,
                        metadata_json=metadata_json,
                    )
                    existing.add(record.name)
                except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug("collections: failed to seed watchlists template {}: {}", record.name, exc)
                continue

            try:
                current = self.get_output_template_by_name(name)
            except KeyError:
                continue
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("collections: failed to lookup watchlists template {}: {}", name, exc)
                continue

            current_meta: dict[str, Any] = {}
            if current.metadata_json:
                try:
                    current_meta = json.loads(current.metadata_json)
                except _COLLECTIONS_NONCRITICAL_EXCEPTIONS:
                    current_meta = {}

            if current_meta.get("seeded_from") != "watchlists_templates":
                continue
            if current_meta.get("seeded_locked"):
                continue

            current_seed_hash = current_meta.get("seeded_hash")
            if not current_seed_hash:
                continue

            current_row_hash = self._seeded_template_hash(
                current.body,
                current.description,
                current.format,
                current.type,
            )
            if current_row_hash and current_seed_hash != current_row_hash:
                continue

            if (
                current_seed_hash == seeded_hash
                and current_meta.get("seeded_mtime") == record.updated_at
            ):
                continue

            patch: dict[str, Any] = {}
            if current.body != record.content:
                patch["body"] = record.content
            if current.description != record.description:
                patch["description"] = record.description
            if current.format != fmt:
                patch["format"] = fmt
            if current.type != type_:
                patch["type"] = type_

            new_meta = dict(current_meta)
            new_meta.update(desired_meta)
            new_metadata_json = json.dumps(new_meta, ensure_ascii=False)
            if current.metadata_json != new_metadata_json:
                patch["metadata_json"] = new_metadata_json

            if patch:
                try:
                    self.update_output_template(current.id, patch)
                except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug("collections: failed to refresh watchlists template {}: {}", name, exc)

    # ------------------------
    # Collections Tags helpers
    # ------------------------
    @staticmethod
    def _normalize_collection_tag(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def _infer_output_template_type(name: str, fmt: str) -> str:
        if fmt == "html":
            return "newsletter_html"
        lowered = name.lower()
        if "mece" in lowered:
            return "mece_markdown"
        if "newsletter" in lowered:
            return "newsletter_markdown"
        return "briefing_markdown"

    def _list_output_template_names(self) -> set[str]:
        try:
            rows = self.backend.execute(
                "SELECT name FROM output_templates WHERE user_id = ?",
                (self.user_id,),
            ).rows
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS:
            return set()
        names: set[str] = set()
        for row in rows:
            if isinstance(row, dict) and row.get("name") is not None:
                names.add(str(row.get("name")))
        return names

    @staticmethod
    def _domain_from_url(url: str | None) -> str | None:
        if not url:
            return None
        try:
            parsed = urlparse(url)
            return parsed.hostname
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS:
            return None

    @staticmethod
    def _fts_query_string(query: str) -> str:
        """Build a simple FTS query string with prefix matches."""
        tokens = [tok.strip() for tok in query.replace('"', " ").split() if tok.strip()]
        if not tokens:
            return ""
        return " AND ".join(f"{token}*" for token in tokens)

    @staticmethod
    def _fts_query_candidates(query: str) -> list[str]:
        """Generate FTS query candidates, preferring safe variants when needed."""
        raw = (query or "").strip()
        if not raw:
            return []
        sanitized = CollectionsDatabase._fts_query_string(raw)
        upper = raw.upper()
        raw_ops = {"AND", "OR", "NOT", "NEAR"}
        has_operator = any(op in upper.split() for op in raw_ops)
        has_syntax = bool(re.search(r'[":*()]', raw))
        prefer_raw = has_operator or has_syntax

        candidates: list[str] = []
        if prefer_raw:
            candidates.append(raw)
            if sanitized and sanitized != raw:
                candidates.append(sanitized)
        else:
            if sanitized:
                candidates.append(sanitized)
            if raw and raw not in candidates:
                candidates.append(raw)
        return [cand for cand in candidates if cand]

    @staticmethod
    def _is_unique_violation(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "unique constraint failed" in msg or "duplicate key value violates unique constraint" in msg

    @staticmethod
    def _is_fts_query_error(exc: Exception) -> bool:
        """Detect common FTS query errors for fallback handling."""
        msg = str(exc).lower()
        return (
            "fts" in msg
            or "fts5" in msg
            or ("match" in msg and "content_items_fts" in msg)
            or "malformed match expression" in msg
        )

    def ensure_collection_tag_ids(self, names: Iterable[str]) -> list[int]:
        normed: list[str] = []
        seen: set[str] = set()
        for raw in names or []:
            if not raw:
                continue
            nm = self._normalize_collection_tag(str(raw))
            if not nm or nm in seen:
                continue
            seen.add(nm)
            normed.append(nm)
        if not normed:
            return []

        ids: list[int] = []
        select_sql = "SELECT id FROM collection_tags WHERE user_id = ? AND name = ?"
        insert_sql = "INSERT INTO collection_tags (user_id, name) VALUES (?, ?)"
        for nm in normed:
            select_params = (self.user_id, nm)
            row = self.backend.execute(select_sql, select_params).first
            if row:
                ids.append(int(row.get("id")))
                continue
            insert_exc: Exception | None = None
            tag_id: int | None = None
            try:
                res = self._execute_insert(insert_sql, (self.user_id, nm))
                tag_id = self._extract_lastrowid(res)
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                insert_exc = exc
            if tag_id is None:
                row = self.backend.execute(select_sql, select_params).first
                if row:
                    tag_id = int(row.get("id"))
            if tag_id is None:
                if insert_exc:
                    raise insert_exc
                raise DatabaseError("Failed to ensure collection tag id")
            ids.append(tag_id)
        return ids

    def _replace_item_tags(self, item_id: int, tag_ids: Iterable[int]) -> None:
        self.backend.execute("DELETE FROM content_item_tags WHERE item_id = ?", (item_id,))
        for tag_id in tag_ids or []:
            try:
                self.backend.execute(
                    "INSERT INTO content_item_tags (item_id, tag_id) VALUES (?, ?)",
                    (item_id, tag_id),
                )
            except DatabaseError:
                # Ignore unique violations (already linked)
                continue

    def _fetch_tags_for_item_ids(self, item_ids: Iterable[int]) -> dict[int, list[str]]:
        ids = [int(i) for i in set(item_ids or []) if i is not None]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.backend.execute(
            """
            SELECT cit.item_id AS item_id, ct.name AS name
            FROM content_item_tags cit
            JOIN collection_tags ct ON ct.id = cit.tag_id
            WHERE cit.item_id IN ({placeholders})
            """.format_map(locals()),  # nosec B608
            tuple(ids),
        ).rows
        mapping: dict[int, list[str]] = {item_id: [] for item_id in ids}
        for row in rows:
            item_id = int(row.get("item_id"))
            name = str(row.get("name"))
            mapping.setdefault(item_id, []).append(name)
        for tag_list in mapping.values():
            tag_list.sort()
        return mapping

    def _update_content_fts_entry(
        self,
        item_id: int,
        *,
        title: str | None,
        summary: str | None,
        notes: str | None,
        tags: list[str] | None,
        metadata_json: str | None,
        previous_title: str | None = None,
        previous_summary: str | None = None,
        previous_notes: str | None = None,
        previous_tags: list[str] | None = None,
        previous_metadata_json: str | None = None,
        has_previous_entry: bool = False,
    ) -> None:
        if not self._fts_available:
            return
        try:
            metadata_text = self._build_content_fts_metadata_text(
                notes=notes,
                tags=tags,
                metadata_json=metadata_json,
            )
            had_previous = has_previous_entry or any(
                value is not None for value in (previous_title, previous_summary, previous_notes, previous_metadata_json)
            ) or bool(previous_tags)

            if had_previous:
                previous_metadata_text = self._build_content_fts_metadata_text(
                    notes=previous_notes,
                    tags=previous_tags,
                    metadata_json=previous_metadata_json,
                )
                if self._fts_supports_direct_delete:
                    self.backend.execute(
                        "DELETE FROM content_items_fts WHERE rowid = ?",
                        (item_id,),
                    )
                else:
                    # Contentless FTS5 tables require the special delete command with the old indexed values.
                    self.backend.execute(
                        """
                        INSERT INTO content_items_fts(content_items_fts, rowid, title, summary, metadata)
                        VALUES('delete', ?, ?, ?, ?)
                        """,
                        (item_id, previous_title or "", previous_summary or "", previous_metadata_text),
                    )
            self.backend.execute(
                "INSERT INTO content_items_fts(rowid, title, summary, metadata) VALUES (?, ?, ?, ?)",
                (item_id, title or "", summary or "", metadata_text),
            )
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Collections FTS update failed for item {item_id}: {exc}")

    def _delete_content_fts_entry(
        self,
        item_id: int,
        *,
        title: str | None = None,
        summary: str | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
        metadata_json: str | None = None,
    ) -> None:
        if not self._fts_available:
            return
        try:
            if self._fts_supports_direct_delete:
                self.backend.execute(
                    "DELETE FROM content_items_fts WHERE rowid = ?",
                    (item_id,),
                )
                return

            # Contentless FTS5 delete needs the original indexed values.
            metadata_text = self._build_content_fts_metadata_text(
                notes=notes,
                tags=tags,
                metadata_json=metadata_json,
            )
            self.backend.execute(
                """
                INSERT INTO content_items_fts(content_items_fts, rowid, title, summary, metadata)
                VALUES('delete', ?, ?, ?, ?)
                """,
                (item_id, title or "", summary or "", metadata_text),
            )
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Collections FTS delete failed for item {item_id}: {exc}")

    def _row_to_content_item(
        self,
        row: dict[str, Any],
        tags: list[str] | None = None,
        *,
        is_new: bool = False,
        content_changed: bool = False,
    ) -> ContentItemRow:
        return ContentItemRow(
            id=int(row.get("id")),
            user_id=str(row.get("user_id")),
            origin=str(row.get("origin")),
            origin_type=row.get("origin_type"),
            origin_id=(int(row.get("origin_id")) if row.get("origin_id") is not None else None),
            url=row.get("url"),
            canonical_url=row.get("canonical_url"),
            domain=row.get("domain"),
            title=row.get("title"),
            summary=row.get("summary"),
            notes=row.get("notes"),
            content_hash=row.get("content_hash"),
            word_count=(int(row.get("word_count")) if row.get("word_count") is not None else None),
            published_at=row.get("published_at"),
            status=row.get("status"),
            favorite=bool(row.get("favorite", 0)),
            metadata_json=row.get("metadata_json"),
            media_id=(int(row.get("media_id")) if row.get("media_id") is not None else None),
            job_id=(int(row.get("job_id")) if row.get("job_id") is not None else None),
            run_id=(int(row.get("run_id")) if row.get("run_id") is not None else None),
            source_id=(int(row.get("source_id")) if row.get("source_id") is not None else None),
            read_at=row.get("read_at"),
            created_at=str(row.get("created_at")),
            updated_at=str(row.get("updated_at")),
            tags=tags or [],
            is_new=is_new,
            content_changed=content_changed,
        )

    def get_content_item(self, item_id: int) -> ContentItemRow:
        row = self.backend.execute(
            """
            SELECT id, user_id, origin, origin_type, origin_id, url, canonical_url, domain,
                   title, summary, notes, content_hash, word_count, published_at, status, favorite,
                   metadata_json, media_id, job_id, run_id, source_id, read_at, created_at, updated_at
            FROM content_items
            WHERE id = ? AND user_id = ?
            """,
            (item_id, self.user_id),
        ).first
        if not row:
            raise KeyError("content_item_not_found")
        tags_map = self._fetch_tags_for_item_ids([item_id])
        return self._row_to_content_item(row, tags_map.get(item_id, []))

    def get_content_item_by_media_id(self, media_id: int) -> ContentItemRow:
        row = self.backend.execute(
            """
            SELECT id, user_id, origin, origin_type, origin_id, url, canonical_url, domain,
                   title, summary, notes, content_hash, word_count, published_at, status, favorite,
                   metadata_json, media_id, job_id, run_id, source_id, read_at, created_at, updated_at
            FROM content_items
            WHERE media_id = ? AND user_id = ?
            """,
            (media_id, self.user_id),
        ).first
        if not row:
            raise KeyError("content_item_not_found")
        item_id = int(row.get("id"))
        tags_map = self._fetch_tags_for_item_ids([item_id])
        return self._row_to_content_item(row, tags_map.get(item_id, []))

    def get_content_item_by_url(self, url: str) -> ContentItemRow | None:
        if not url:
            return None
        row = self.backend.execute(
            """
            SELECT id, user_id, origin, origin_type, origin_id, url, canonical_url, domain,
                   title, summary, notes, content_hash, word_count, published_at, status, favorite,
                   metadata_json, media_id, job_id, run_id, source_id, read_at, created_at, updated_at
            FROM content_items
            WHERE user_id = ? AND (canonical_url = ? OR url = ?)
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            (self.user_id, url, url),
        ).first
        if not row:
            return None
        tags_map = self._fetch_tags_for_item_ids([int(row.get("id"))])
        return self._row_to_content_item(row, tags_map.get(int(row.get("id")), []))

    def upsert_content_item(
        self,
        *,
        origin: str,
        origin_type: str | None = None,
        origin_id: int | None = None,
        url: str | None,
        canonical_url: str | None,
        domain: str | None,
        title: str | None,
        summary: str | None,
        notes: str | None = None,
        content_hash: str | None,
        word_count: int | None,
        published_at: str | None,
        status: str | None = None,
        favorite: bool | None = None,
        metadata: dict[str, Any] | None = None,
        media_id: int | None = None,
        job_id: int | None = None,
        run_id: int | None = None,
        source_id: int | None = None,
        read_at: str | None = None,
        tags: Iterable[str] | None = None,
        merge_tags: bool = False,
        preserve_existing_on_null: bool = False,
    ) -> ContentItemRow:
        """Insert or update a content item record and attach tags."""
        now = _utcnow_iso()
        canonical = canonical_url or url

        selectors: list[tuple[str, Any]] = []
        if canonical:
            selectors.append(("canonical_url", canonical))
        if content_hash:
            selectors.append(("content_hash", content_hash))
        if url:
            selectors.append(("url", url))

        def _lookup_existing() -> tuple[dict[str, Any] | None, int | None]:
            for column, value in selectors:
                row = self.backend.execute(
                    """
                    SELECT id, user_id, origin, origin_type, origin_id, url, canonical_url, domain,
                           title, summary, notes, content_hash, word_count, published_at, status, favorite,
                           metadata_json, media_id, job_id, run_id, source_id, read_at, created_at, updated_at
                    FROM content_items
                    WHERE user_id = ? AND {column} = ?
                    """.format_map(locals()),  # nosec B608
                    (self.user_id, value),
                ).first
                if row:
                    return row, int(row.get("id"))
            return None, None

        def _apply_preserve(existing: dict[str, Any]) -> dict[str, Any]:
            return {
                "origin_type": origin_type if origin_type is not None else existing.get("origin_type"),
                "origin_id": origin_id if origin_id is not None else existing.get("origin_id"),
                "url": url if url is not None else existing.get("url"),
                "canonical_url": canonical_url if canonical_url is not None else existing.get("canonical_url"),
                "domain": domain if domain is not None else existing.get("domain"),
                "title": title if title is not None else existing.get("title"),
                "summary": summary if summary is not None else existing.get("summary"),
                "notes": notes if notes is not None else existing.get("notes"),
                "content_hash": content_hash if content_hash is not None else existing.get("content_hash"),
                "word_count": word_count if word_count is not None else existing.get("word_count"),
                "published_at": published_at if published_at is not None else existing.get("published_at"),
                "status": status if status is not None else existing.get("status"),
                "media_id": media_id if media_id is not None else existing.get("media_id"),
                "job_id": job_id if job_id is not None else existing.get("job_id"),
                "run_id": run_id if run_id is not None else existing.get("run_id"),
                "source_id": source_id if source_id is not None else existing.get("source_id"),
                "read_at": read_at if read_at is not None else existing.get("read_at"),
            }

        def _build_metadata_json(existing: dict[str, Any] | None) -> str | None:
            if not metadata:
                if preserve_existing_on_null and existing is not None:
                    return existing.get("metadata_json")
                return None
            if preserve_existing_on_null and existing is not None:
                current_meta: dict[str, Any] = {}
                existing_json = existing.get("metadata_json")
                if existing_json:
                    try:
                        current_meta = json.loads(existing_json)
                    except _COLLECTIONS_NONCRITICAL_EXCEPTIONS:
                        current_meta = {}
                current_meta.update(metadata)
                try:
                    return json.dumps(current_meta, ensure_ascii=False)
                except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(f"Failed to encode collections metadata: {exc}")
                    return None
            try:
                return json.dumps(metadata, ensure_ascii=False)
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to encode collections metadata: {exc}")
                return None

        def _refresh_derived() -> tuple[str | None, str | None, int, str]:
            current_canonical = canonical_url or url
            current_domain = domain or self._domain_from_url(current_canonical)
            if favorite is None and preserve_existing_on_null and existing_row is not None:
                current_favorite = int(existing_row.get("favorite") or 0)
            else:
                current_favorite = 1 if favorite else 0
            current_status = status or "new"
            return current_canonical, current_domain, current_favorite, current_status

        existing_row, item_id = _lookup_existing()
        existing_tags_for_fts: list[str] | None = None
        if existing_row and item_id is not None:
            existing_tags_for_fts = self._fetch_tags_for_item_ids([int(item_id)]).get(int(item_id), [])
        if existing_row and preserve_existing_on_null:
            preserved = _apply_preserve(existing_row)
            origin_type = preserved["origin_type"]
            origin_id = preserved["origin_id"]
            url = preserved["url"]
            canonical_url = preserved["canonical_url"]
            domain = preserved["domain"]
            title = preserved["title"]
            summary = preserved["summary"]
            notes = preserved["notes"]
            content_hash = preserved["content_hash"]
            word_count = preserved["word_count"]
            published_at = preserved["published_at"]
            status = preserved["status"]
            media_id = preserved["media_id"]
            job_id = preserved["job_id"]
            run_id = preserved["run_id"]
            source_id = preserved["source_id"]
            read_at = preserved["read_at"]

        canonical, domain_val, favorite_int, status_val = _refresh_derived()
        metadata_json = _build_metadata_json(existing_row)

        prev_hash = existing_row.get("content_hash") if existing_row else None
        created = False
        content_changed = False

        if item_id is None:
            try:
                res = self._execute_insert(
                    """
                    INSERT INTO content_items (
                        user_id, origin, origin_type, origin_id, url, canonical_url, domain, title, summary,
                        notes, content_hash, word_count, published_at, status, favorite, metadata_json, media_id,
                        job_id, run_id, source_id, read_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.user_id,
                        origin,
                        origin_type,
                        origin_id,
                        url,
                        canonical,
                        domain_val,
                        title,
                        summary,
                        notes,
                        content_hash,
                        word_count,
                        published_at,
                        status_val,
                        favorite_int,
                        metadata_json,
                        media_id,
                        job_id,
                        run_id,
                        source_id,
                        read_at,
                        now,
                        now,
                    ),
                )
                item_id = self._extract_lastrowid(res)
                if not item_id:
                    raise DatabaseError("Failed to insert content item")
                created = True
                content_changed = True
            except DatabaseError as exc:
                if not self._is_unique_violation(exc):
                    raise
                existing_row, item_id = _lookup_existing()
                if not existing_row or item_id is None:
                    raise
                existing_tags_for_fts = self._fetch_tags_for_item_ids([int(item_id)]).get(int(item_id), [])
                if preserve_existing_on_null:
                    preserved = _apply_preserve(existing_row)
                    origin_type = preserved["origin_type"]
                    origin_id = preserved["origin_id"]
                    url = preserved["url"]
                    canonical_url = preserved["canonical_url"]
                    domain = preserved["domain"]
                    title = preserved["title"]
                    summary = preserved["summary"]
                    notes = preserved["notes"]
                    content_hash = preserved["content_hash"]
                    word_count = preserved["word_count"]
                    published_at = preserved["published_at"]
                    status = preserved["status"]
                    media_id = preserved["media_id"]
                    job_id = preserved["job_id"]
                    run_id = preserved["run_id"]
                    source_id = preserved["source_id"]
                    read_at = preserved["read_at"]
                canonical, domain_val, favorite_int, status_val = _refresh_derived()
                metadata_json = _build_metadata_json(existing_row)
                prev_hash = existing_row.get("content_hash")

        if item_id is not None and not created:
            fields: list[str] = [
                "origin = ?",
                "origin_type = ?",
                "origin_id = ?",
                "url = ?",
                "canonical_url = ?",
                "domain = ?",
                "title = ?",
                "summary = ?",
                "notes = ?",
                "content_hash = ?",
                "word_count = ?",
                "published_at = ?",
                "status = ?",
                "favorite = ?",
                "metadata_json = ?",
                "media_id = ?",
                "job_id = ?",
                "run_id = ?",
                "source_id = ?",
                "read_at = ?",
                "updated_at = ?",
            ]
            params = (
                origin,
                origin_type,
                origin_id,
                url,
                canonical,
                domain_val,
                title,
                summary,
                notes,
                content_hash,
                word_count,
                published_at,
                status_val,
                favorite_int,
                metadata_json,
                media_id,
                job_id,
                run_id,
                source_id,
                read_at,
                now,
                item_id,
                self.user_id,
            )
            self.backend.execute(
                f"UPDATE content_items SET {', '.join(fields)} WHERE id = ? AND user_id = ?",  # nosec B608
                params,
            )
            content_changed = not (prev_hash == content_hash or prev_hash is None and content_hash is None)

        if tags is not None:
            tag_list = list(tags)
            if merge_tags and not created:
                existing_tags = self._fetch_tags_for_item_ids([int(item_id or 0)]).get(int(item_id or 0), [])
                if existing_tags:
                    tag_list = list(dict.fromkeys([*existing_tags, *tag_list]))
            tag_ids = self.ensure_collection_tag_ids(tag_list)
            self._replace_item_tags(item_id, tag_ids)

        row = self.get_content_item(item_id)
        with contextlib.suppress(_COLLECTIONS_NONCRITICAL_EXCEPTIONS):
            self._update_content_fts_entry(
                item_id,
                title=row.title,
                summary=row.summary,
                notes=row.notes,
                tags=row.tags,
                metadata_json=row.metadata_json,
                previous_title=(existing_row.get("title") if existing_row else None),
                previous_summary=(existing_row.get("summary") if existing_row else None),
                previous_notes=(existing_row.get("notes") if existing_row else None),
                previous_tags=existing_tags_for_fts,
                previous_metadata_json=(existing_row.get("metadata_json") if existing_row else None),
                has_previous_entry=bool(existing_row),
            )
        row.is_new = created
        row.content_changed = content_changed
        return row

    def list_content_items(
        self,
        *,
        ids: Iterable[int] | None = None,
        q: str | None = None,
        tags: Iterable[str] | None = None,
        domain: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        status: Iterable[str] | None = None,
        favorite: bool | None = None,
        job_id: int | None = None,
        run_id: int | None = None,
        origin: str | None = None,
        page: int = 1,
        size: int = 20,
        offset: int | None = None,
        limit: int | None = None,
        sort: str | None = None,
    ) -> tuple[list[ContentItemRow], int]:
        where: list[str] = ["ci.user_id = ?"]
        params: list[Any] = [self.user_id]
        joins: list[str] = []
        having = ""

        if ids:
            id_list = [int(i) for i in ids]
            if id_list:
                placeholders = ",".join("?" for _ in id_list)
                where.append(f"ci.id IN ({placeholders})")
                params.extend(id_list)

        if domain:
            where.append("ci.domain = ?")
            params.append(domain)

        if date_from:
            where.append("ci.created_at >= ?")
            params.append(date_from)

        if date_to:
            where.append("ci.created_at <= ?")
            params.append(date_to)

        if job_id is not None:
            where.append("ci.job_id = ?")
            params.append(int(job_id))

        if run_id is not None:
            where.append("ci.run_id = ?")
            params.append(int(run_id))

        if origin:
            where.append("ci.origin = ?")
            params.append(origin)

        status_filters: list[str] = []
        if status:
            status_filters = [status.lower()] if isinstance(status, str) else [str(s).lower() for s in status if s]
        if status_filters:
            placeholders = ",".join("?" for _ in status_filters)
            where.append(f"LOWER(ci.status) IN ({placeholders})")
            params.extend(status_filters)

        if favorite is not None:
            where.append("ci.favorite = ?")
            params.append(1 if favorite else 0)

        tag_filters: list[str] = []
        if tags:
            for t in tags:
                if t:
                    tag_filters.append(self._normalize_collection_tag(str(t)))
        if tag_filters:
            tag_filters = list(dict.fromkeys(tag_filters))
            joins.append("INNER JOIN content_item_tags cit ON cit.item_id = ci.id")
            joins.append("INNER JOIN collection_tags ct ON ct.id = cit.tag_id")
            placeholders = ",".join("?" for _ in tag_filters)
            where.append(f"ct.name IN ({placeholders})")
            params.extend(tag_filters)
            having = f"HAVING COUNT(DISTINCT ct.name) >= {len(tag_filters)}"

        def _apply_like_search(where_clause: list[str], clause_params: list[Any]) -> None:
            q_like = f"%{q.lower()}%"
            where_clause.append(
                "("
                "LOWER(COALESCE(ci.title, '')) LIKE ? OR "
                "LOWER(COALESCE(ci.summary, '')) LIKE ? OR "
                "LOWER(COALESCE(ci.notes, '')) LIKE ? OR "
                "EXISTS ("
                "SELECT 1 FROM content_item_tags cit_q "
                "JOIN collection_tags ct_q ON ct_q.id = cit_q.tag_id "
                "WHERE cit_q.item_id = ci.id AND ct_q.user_id = ? AND LOWER(ct_q.name) LIKE ?"
                ")"
                ")"
            )
            clause_params.extend([q_like, q_like, q_like, self.user_id, q_like])

        def _execute_query(
            where_clause: list[str],
            clause_params: list[Any],
            query_joins: list[str],
            fts_joined: bool,
        ) -> tuple[list[ContentItemRow], int]:
            where_sql = " AND ".join(where_clause) if where_clause else "1=1"
            group_by = "GROUP BY ci.id" if tag_filters else ""
            joins_sql = f" {' '.join(query_joins)}" if query_joins else ""
            base_from = f"FROM content_items ci{joins_sql}"
            subquery = f"SELECT ci.id {base_from} WHERE {where_sql} {group_by} {having}"
            count_sql = f"SELECT COUNT(*) AS cnt FROM ({subquery}) AS subq"  # nosec B608
            total = int(self.backend.execute(count_sql, tuple(clause_params)).scalar or 0)

            resolved_limit = limit if isinstance(limit, int) and limit > 0 else size
            resolved_offset = offset if isinstance(offset, int) and offset >= 0 else max(0, (page - 1) * size)

            sort_key = (sort or "").strip().lower()
            order_by = "ci.updated_at DESC, ci.id DESC"
            if sort_key == "updated_asc":
                order_by = "ci.updated_at ASC, ci.id ASC"
            elif sort_key == "created_desc":
                order_by = "ci.created_at DESC, ci.id DESC"
            elif sort_key == "created_asc":
                order_by = "ci.created_at ASC, ci.id ASC"
            elif sort_key == "title_asc":
                order_by = "ci.title ASC, ci.id ASC"
            elif sort_key == "title_desc":
                order_by = "ci.title DESC, ci.id DESC"
            elif sort_key == "relevance" and fts_joined and not tag_filters or not sort_key and fts_joined and not tag_filters:
                order_by = "bm25(content_items_fts) ASC, ci.updated_at DESC, ci.id DESC"
            rows_sql = f"""
                SELECT
                    ci.id, ci.user_id, ci.origin, ci.origin_type, ci.origin_id, ci.url, ci.canonical_url,
                    ci.domain, ci.title, ci.summary, ci.notes, ci.content_hash, ci.word_count, ci.published_at,
                    ci.status, ci.favorite, ci.metadata_json, ci.media_id, ci.job_id, ci.run_id,
                    ci.source_id, ci.read_at, ci.created_at, ci.updated_at
                {base_from}
                WHERE {where_sql}
                {group_by}
                {having}
                ORDER BY {order_by}
                LIMIT ? OFFSET ?
            """
            row_params = tuple(clause_params + [resolved_limit, resolved_offset])
            rows = self.backend.execute(rows_sql, row_params).rows
            item_ids = [int(r.get("id")) for r in rows]
            tags_map = self._fetch_tags_for_item_ids(item_ids)
            content_rows = [self._row_to_content_item(r, tags_map.get(int(r.get("id")), [])) for r in rows]
            return content_rows, total

        if q:
            q = q.strip()
        if q and self._fts_available:
            for fts_query in self._fts_query_candidates(q):
                if not fts_query:
                    continue
                fts_where = list(where)
                fts_params = list(params)
                fts_joins = list(joins)
                fts_joins.append("INNER JOIN content_items_fts ON content_items_fts.rowid = ci.id")
                fts_where.append("content_items_fts MATCH ?")
                fts_params.append(fts_query)
                try:
                    return _execute_query(fts_where, fts_params, fts_joins, True)
                except DatabaseError as exc:
                    if not self._is_fts_query_error(exc):
                        raise

        if q:
            like_where = list(where)
            like_params = list(params)
            like_joins = list(joins)
            _apply_like_search(like_where, like_params)
            return _execute_query(like_where, like_params, like_joins, False)

        return _execute_query(where, params, joins, False)

    def update_content_item(
        self,
        item_id: int,
        *,
        status: str | None = None,
        favorite: bool | None = None,
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
        title: str | None = None,
        summary: str | None = None,
        notes: str | None = None,
        read_at: str | None = None,
        clear_read_at: bool = False,
    ) -> ContentItemRow:
        """Update persisted content item fields and tags."""
        existing = self.get_content_item(item_id)
        updates: list[str] = []
        params: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if favorite is not None:
            updates.append("favorite = ?")
            params.append(1 if favorite else 0)
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if summary is not None:
            updates.append("summary = ?")
            params.append(summary)
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        if read_at is not None or clear_read_at:
            updates.append("read_at = ?")
            params.append(read_at)

        metadata_json = None
        if metadata is not None:
            current_meta: dict[str, Any] = {}
            if existing.metadata_json:
                try:
                    current_meta = json.loads(existing.metadata_json)
                except _COLLECTIONS_NONCRITICAL_EXCEPTIONS:
                    current_meta = {}
            current_meta.update(metadata)
            metadata_json = json.dumps(current_meta, ensure_ascii=False)
            updates.append("metadata_json = ?")
            params.append(metadata_json)

        if updates:
            updates.append("updated_at = ?")
            params.append(_utcnow_iso())
            params.extend([item_id, self.user_id])
            self.backend.execute(
                f"UPDATE content_items SET {', '.join(updates)} WHERE id = ? AND user_id = ?",  # nosec B608
                tuple(params),
            )

        if tags is not None:
            tag_ids = self.ensure_collection_tag_ids(tags)
            self._replace_item_tags(item_id, tag_ids)

        try:
            tgt = self.get_content_item(item_id)
            self._update_content_fts_entry(
                item_id,
                title=tgt.title,
                summary=tgt.summary,
                notes=tgt.notes,
                tags=tgt.tags,
                metadata_json=tgt.metadata_json,
                previous_title=existing.title,
                previous_summary=existing.summary,
                previous_notes=existing.notes,
                previous_tags=existing.tags,
                previous_metadata_json=existing.metadata_json,
                has_previous_entry=True,
            )
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS:
            pass

        row = self.get_content_item(item_id)
        row.is_new = False
        row.content_changed = False
        return row

    def delete_content_item(self, item_id: int) -> None:
        """Delete a content item and its tags/FTS entry."""
        row = self.backend.execute(
            """
            SELECT id, title, summary, notes, metadata_json
            FROM content_items
            WHERE id = ? AND user_id = ?
            """,
            (item_id, self.user_id),
        ).first
        if not row:
            raise KeyError("content_item_not_found")
        tags_map = self._fetch_tags_for_item_ids([item_id])
        existing_tags = tags_map.get(item_id, [])

        self.backend.execute(
            "DELETE FROM content_item_tags WHERE item_id = ?",
            (item_id,),
        )
        self.backend.execute(
            "DELETE FROM content_item_note_links WHERE user_id = ? AND item_id = ?",
            (self.user_id, item_id),
        )
        self.backend.execute(
            "DELETE FROM content_items WHERE id = ? AND user_id = ?",
            (item_id, self.user_id),
        )
        with contextlib.suppress(_COLLECTIONS_NONCRITICAL_EXCEPTIONS):
            self._delete_content_fts_entry(
                item_id,
                title=row.get("title"),
                summary=row.get("summary"),
                notes=row.get("notes"),
                tags=existing_tags,
                metadata_json=row.get("metadata_json"),
            )

    def prune_content_items_for_source(
        self,
        *,
        origin: str,
        origin_id: int,
        max_items: int | None = None,
        retention_days: int | None = None,
    ) -> int:
        """Delete oldest content items for a source exceeding retention limits.

        Returns count of deleted items.
        """
        deleted = 0
        if retention_days and retention_days > 0:
            cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
            old_ids = [
                int(r["id"])
                for r in self.backend.execute(
                    "SELECT id FROM content_items WHERE user_id = ? AND origin = ? AND origin_id = ? AND created_at < ?",
                    (self.user_id, origin, origin_id, cutoff),
                ).rows
            ]
            for item_id in old_ids:
                with contextlib.suppress(_COLLECTIONS_NONCRITICAL_EXCEPTIONS):
                    self.delete_content_item(item_id)
                    deleted += 1

        if max_items and max_items > 0:
            total = int(
                self.backend.execute(
                    "SELECT COUNT(*) AS cnt FROM content_items WHERE user_id = ? AND origin = ? AND origin_id = ?",
                    (self.user_id, origin, origin_id),
                ).scalar or 0
            )
            excess = total - max_items
            if excess > 0:
                excess_ids = [
                    int(r["id"])
                    for r in self.backend.execute(
                        "SELECT id FROM content_items WHERE user_id = ? AND origin = ? AND origin_id = ? ORDER BY created_at ASC LIMIT ?",
                        (self.user_id, origin, origin_id, excess),
                    ).rows
                ]
                for item_id in excess_ids:
                    with contextlib.suppress(_COLLECTIONS_NONCRITICAL_EXCEPTIONS):
                        self.delete_content_item(item_id)
                        deleted += 1
        return deleted

    # ------------------------
    # Reading Saved Searches API
    # ------------------------
    @staticmethod
    def _saved_search_row_from_db(row: dict[str, Any]) -> SavedSearchRow:
        return SavedSearchRow(
            id=int(row.get("id")),
            user_id=str(row.get("user_id") or ""),
            name=str(row.get("name") or ""),
            query_json=str(row.get("query_json") or "{}"),
            sort=row.get("sort"),
            created_at=str(row.get("created_at") or ""),
            updated_at=str(row.get("updated_at") or ""),
        )

    def create_saved_search(
        self,
        *,
        name: str,
        query_json: str,
        sort: str | None = None,
    ) -> SavedSearchRow:
        now = _utcnow_iso()
        q = (
            "INSERT INTO saved_searches (user_id, name, query_json, sort, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        params = (self.user_id, name, query_json, sort, now, now)
        try:
            res = self._execute_insert(q, params)
            new_id = self._extract_lastrowid(res)
            if not new_id:
                raise DatabaseError("Failed to create saved search")
            return self.get_saved_search(int(new_id))
        except DatabaseError as exc:
            if not self._is_unique_violation(exc):
                raise
            existing = self.get_saved_search_by_name(name)
            if existing is None:
                raise
            return existing

    def get_saved_search(self, search_id: int) -> SavedSearchRow:
        row = self.backend.execute(
            "SELECT id, user_id, name, query_json, sort, created_at, updated_at "
            "FROM saved_searches WHERE id = ? AND user_id = ?",
            (search_id, self.user_id),
        ).first
        if not row:
            raise KeyError("saved_search_not_found")
        return self._saved_search_row_from_db(row)

    def get_saved_search_by_name(self, name: str) -> SavedSearchRow | None:
        row = self.backend.execute(
            "SELECT id, user_id, name, query_json, sort, created_at, updated_at "
            "FROM saved_searches WHERE user_id = ? AND name = ?",
            (self.user_id, name),
        ).first
        if not row:
            return None
        return self._saved_search_row_from_db(row)

    def list_saved_searches(self, *, limit: int = 50, offset: int = 0) -> tuple[list[SavedSearchRow], int]:
        count = int(
            self.backend.execute(
                "SELECT COUNT(*) AS cnt FROM saved_searches WHERE user_id = ?",
                (self.user_id,),
            ).scalar or 0
        )
        rows = self.backend.execute(
            "SELECT id, user_id, name, query_json, sort, created_at, updated_at "
            "FROM saved_searches WHERE user_id = ? "
            "ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
            (self.user_id, limit, offset),
        ).rows
        return [self._saved_search_row_from_db(row) for row in rows], count

    def update_saved_search(self, search_id: int, patch: dict[str, Any]) -> SavedSearchRow:
        if not patch:
            return self.get_saved_search(search_id)
        fields: list[str] = []
        params: list[Any] = []
        for key in ("name", "query_json", "sort"):
            if key in patch:
                fields.append(f"{key} = ?")
                params.append(patch.get(key))
        if not fields:
            return self.get_saved_search(search_id)
        fields.append("updated_at = ?")
        params.append(_utcnow_iso())
        params.extend([search_id, self.user_id])
        q = f"UPDATE saved_searches SET {', '.join(fields)} WHERE id = ? AND user_id = ?"  # nosec B608
        res = self.backend.execute(q, tuple(params))
        if res.rowcount <= 0:
            raise KeyError("saved_search_not_found")
        return self.get_saved_search(search_id)

    def delete_saved_search(self, search_id: int) -> bool:
        res = self.backend.execute(
            "DELETE FROM saved_searches WHERE id = ? AND user_id = ?",
            (search_id, self.user_id),
        )
        return bool(res.rowcount and res.rowcount > 0)

    # ------------------------
    # Reading ↔ Notes Link API
    # ------------------------
    @staticmethod
    def _content_item_note_link_row_from_db(row: dict[str, Any]) -> ContentItemNoteLinkRow:
        return ContentItemNoteLinkRow(
            user_id=str(row.get("user_id") or ""),
            item_id=int(row.get("item_id")),
            note_id=str(row.get("note_id") or ""),
            created_at=str(row.get("created_at") or ""),
        )

    def link_note_to_content_item(self, *, item_id: int, note_id: str) -> ContentItemNoteLinkRow:
        self.get_content_item(item_id)
        now = _utcnow_iso()
        self.backend.execute(
            "INSERT INTO content_item_note_links (user_id, item_id, note_id, created_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(user_id, item_id, note_id) DO NOTHING",
            (self.user_id, item_id, note_id, now),
        )
        row = self.backend.execute(
            "SELECT user_id, item_id, note_id, created_at FROM content_item_note_links "
            "WHERE user_id = ? AND item_id = ? AND note_id = ?",
            (self.user_id, item_id, note_id),
        ).first
        if not row:
            raise DatabaseError("Failed to link note to content item")
        return self._content_item_note_link_row_from_db(row)

    def list_note_links_for_content_item(self, item_id: int) -> list[ContentItemNoteLinkRow]:
        self.get_content_item(item_id)
        rows = self.backend.execute(
            "SELECT user_id, item_id, note_id, created_at FROM content_item_note_links "
            "WHERE user_id = ? AND item_id = ? ORDER BY created_at DESC, note_id DESC",
            (self.user_id, item_id),
        ).rows
        return [self._content_item_note_link_row_from_db(row) for row in rows]

    def unlink_note_from_content_item(self, *, item_id: int, note_id: str) -> bool:
        res = self.backend.execute(
            "DELETE FROM content_item_note_links WHERE user_id = ? AND item_id = ? AND note_id = ?",
            (self.user_id, item_id, note_id),
        )
        return bool(res.rowcount and res.rowcount > 0)

    # ------------------------
    # Output Templates API
    # ------------------------
    def list_output_templates(self, q: str | None, limit: int, offset: int) -> tuple[list[OutputTemplateRow], int]:
        where = ["user_id = ?"]
        params: list[Any] = [self.user_id]
        if q:
            where.append("(LOWER(name) LIKE ? OR LOWER(COALESCE(description, '')) LIKE ?)")
            like = f"%{q.lower()}%"
            params.extend([like, like])
        where_sql = " AND ".join(where)

        count_q = f"SELECT COUNT(*) AS cnt FROM output_templates WHERE {where_sql}"  # nosec B608
        total = int(self.backend.execute(count_q, tuple(params)).scalar or 0)

        select_q = (
            f"SELECT id, user_id, name, type, format, body, description, is_default, created_at, updated_at, metadata_json "  # nosec B608
            f"FROM output_templates WHERE {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        rows = self.backend.execute(select_q, tuple(params + [limit, offset])).rows
        mapped = [OutputTemplateRow(**{**r, "is_default": bool(r.get("is_default", 0))}) for r in rows]
        return mapped, total

    def get_default_output_template_by_type(self, type_: str) -> OutputTemplateRow | None:
        q = (
            "SELECT id, user_id, name, type, format, body, description, is_default, created_at, updated_at, metadata_json "
            "FROM output_templates WHERE user_id = ? AND type = ? AND is_default = 1 ORDER BY updated_at DESC LIMIT 1"
        )
        row = self.backend.execute(q, (self.user_id, type_)).first
        if not row:
            return None
        row["is_default"] = bool(row.get("is_default", 0))
        return OutputTemplateRow(**row)

    def create_output_template(self, name: str, type_: str, format_: str, body: str, description: str | None, is_default: bool, metadata_json: str | None = None) -> OutputTemplateRow:
        now = _utcnow_iso()
        q = (
            "INSERT INTO output_templates (user_id, name, type, format, body, description, is_default, created_at, updated_at, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        is_default_val = self._coerce_bool_flag(is_default, postgres=self.backend.backend_type == BackendType.POSTGRESQL)
        params = (self.user_id, name, type_, format_, body, description, is_default_val, now, now, metadata_json)
        res = self._execute_insert(q, params)
        new_id = self._extract_lastrowid(res)
        if not new_id:
            raise DatabaseError("Failed to create output template")
        return self.get_output_template(new_id)

    def get_output_template(self, template_id: int) -> OutputTemplateRow:
        q = (
            "SELECT id, user_id, name, type, format, body, description, is_default, created_at, updated_at, metadata_json "
            "FROM output_templates WHERE id = ? AND user_id = ?"
        )
        row = self.backend.execute(q, (template_id, self.user_id)).first
        if not row:
            raise KeyError("template_not_found")
        row["is_default"] = bool(row.get("is_default", 0))
        return OutputTemplateRow(**row)

    def get_output_template_by_name(self, name: str) -> OutputTemplateRow:
        q = (
            "SELECT id, user_id, name, type, format, body, description, is_default, created_at, updated_at, metadata_json "
            "FROM output_templates WHERE user_id = ? AND name = ?"
        )
        row = self.backend.execute(q, (self.user_id, name)).first
        if not row:
            raise KeyError("template_not_found")
        row["is_default"] = bool(row.get("is_default", 0))
        return OutputTemplateRow(**row)

    def update_output_template(self, template_id: int, patch: dict[str, Any]) -> OutputTemplateRow:
        if not patch:
            return self.get_output_template(template_id)
        fields = []
        params: list[Any] = []
        for key in ("name", "type", "format", "body", "description", "is_default", "metadata_json"):
            if key in patch and patch[key] is not None:
                fields.append(f"{key} = ?")
                val = patch[key]
                if key == "is_default":
                    val = self._coerce_bool_flag(val, postgres=self.backend.backend_type == BackendType.POSTGRESQL)
                params.append(val)
        fields.append("updated_at = ?")
        params.append(_utcnow_iso())
        params.extend([template_id, self.user_id])
        q = f"UPDATE output_templates SET {', '.join(fields)} WHERE id = ? AND user_id = ?"  # nosec B608
        res = self.backend.execute(q, tuple(params))
        if res.rowcount <= 0:
            raise KeyError("template_not_found")
        return self.get_output_template(template_id)

    def delete_output_template(self, template_id: int) -> bool:
        q = "DELETE FROM output_templates WHERE id = ? AND user_id = ?"
        res = self.backend.execute(q, (template_id, self.user_id))
        return res.rowcount > 0

    # ------------------------
    # Highlights API
    # ------------------------
    def create_highlight(
        self,
        item_id: int,
        quote: str,
        start_offset: int | None,
        end_offset: int | None,
        color: str | None,
        note: str | None,
        anchor_strategy: str = "fuzzy_quote",
        content_hash_ref: str | None = None,
        context_before: str | None = None,
        context_after: str | None = None,
        state: str = "active",
    ) -> HighlightRow:
        now = _utcnow_iso()
        q = (
            "INSERT INTO reading_highlights (user_id, item_id, quote, start_offset, end_offset, color, note, created_at, "
            "anchor_strategy, content_hash_ref, context_before, context_after, state) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            self.user_id,
            item_id,
            quote,
            start_offset,
            end_offset,
            color,
            note,
            now,
            anchor_strategy,
            content_hash_ref,
            context_before,
            context_after,
            state,
        )
        res = self._execute_insert(q, params)
        new_id = self._extract_lastrowid(res)
        if not new_id:
            raise DatabaseError("Failed to create highlight")
        return self.get_highlight(new_id)

    def list_highlights_by_item(self, item_id: int) -> list[HighlightRow]:
        q = (
            "SELECT id, user_id, item_id, quote, start_offset, end_offset, color, note, created_at, "
            "anchor_strategy, content_hash_ref, context_before, context_after, state "
            "FROM reading_highlights WHERE user_id = ? AND item_id = ? ORDER BY created_at ASC"
        )
        rows = self.backend.execute(q, (self.user_id, item_id)).rows
        return [HighlightRow(**row) for row in rows]

    def get_highlight(self, highlight_id: int) -> HighlightRow:
        q = (
            "SELECT id, user_id, item_id, quote, start_offset, end_offset, color, note, created_at, "
            "anchor_strategy, content_hash_ref, context_before, context_after, state "
            "FROM reading_highlights WHERE id = ? AND user_id = ?"
        )
        row = self.backend.execute(q, (highlight_id, self.user_id)).first
        if not row:
            raise KeyError("highlight_not_found")
        return HighlightRow(**row)

    def update_highlight(self, highlight_id: int, patch: dict[str, Any]) -> HighlightRow:
        if not patch:
            return self.get_highlight(highlight_id)
        fields = []
        params: list[Any] = []
        for key in (
            "quote",
            "start_offset",
            "end_offset",
            "color",
            "note",
            "anchor_strategy",
            "content_hash_ref",
            "context_before",
            "context_after",
            "state",
        ):
            if key in patch and patch[key] is not None:
                fields.append(f"{key} = ?")
                params.append(patch[key])
        if not fields:
            return self.get_highlight(highlight_id)
        params.extend([highlight_id, self.user_id])
        q = f"UPDATE reading_highlights SET {', '.join(fields)} WHERE id = ? AND user_id = ?"  # nosec B608
        res = self.backend.execute(q, tuple(params))
        if res.rowcount <= 0:
            raise KeyError("highlight_not_found")
        return self.get_highlight(highlight_id)

    def delete_highlight(self, highlight_id: int) -> bool:
        q = "DELETE FROM reading_highlights WHERE id = ? AND user_id = ?"
        res = self.backend.execute(q, (highlight_id, self.user_id))
        return res.rowcount > 0

    # ------------------------
    # Maintenance hooks
    # ------------------------
    def mark_highlights_stale_if_content_changed(self, item_id: int, new_content_hash: str | None) -> int:
        """Mark highlights as stale if their stored content_hash_ref doesn't match new hash.

        Returns number of rows updated. Intended to be called by item update pipeline.
        """
        if not new_content_hash:
            return 0
        q = (
            "UPDATE reading_highlights SET state = 'stale' WHERE user_id = ? AND item_id = ? "
            "AND content_hash_ref IS NOT NULL AND content_hash_ref <> ?"
        )
        res = self.backend.execute(q, (self.user_id, item_id, new_content_hash))
        return int(res.rowcount or 0)

    def reanchor_highlights_for_item(
        self,
        item_id: int,
        *,
        content_text: str | None,
        content_hash: str | None,
    ) -> dict[str, int]:
        if not content_text:
            return {"updated": 0, "stale": 0, "skipped": 0}
        resolved_hash = content_hash or hash_text_sha256(content_text)
        if not resolved_hash:
            return {"updated": 0, "stale": 0, "skipped": 0}

        updated = 0
        stale = 0
        skipped = 0
        for highlight in self.list_highlights_by_item(item_id=item_id):
            if not highlight.quote:
                skipped += 1
                continue
            span = find_highlight_span(
                content_text,
                highlight.quote,
                start_offset=highlight.start_offset,
                end_offset=highlight.end_offset,
                context_before=highlight.context_before,
                context_after=highlight.context_after,
                anchor_strategy=highlight.anchor_strategy,
            )
            if span is None:
                if highlight.state != "stale":
                    self.update_highlight(highlight.id, {"state": "stale"})
                stale += 1
                continue

            start_offset, end_offset = span
            context_before, context_after = build_highlight_context(content_text, start_offset, end_offset)
            self.update_highlight(
                highlight.id,
                {
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "content_hash_ref": resolved_hash,
                    "context_before": context_before,
                    "context_after": context_after,
                    "state": "active",
                },
            )
            updated += 1

        return {"updated": updated, "stale": stale, "skipped": skipped}

    # ------------------------
    # Outputs artifacts API
    # ------------------------
    def resolve_output_storage_path(self, path_value: str | Path) -> str:
        """Resolve and validate output storage paths as relative filenames."""
        try:
            user_id = int(self.user_id)
        except (TypeError, ValueError) as exc:
            logger.error(f"outputs: invalid user id for storage path resolution: {self.user_id}")
            raise InvalidStorageUserIdError("invalid_user_id") from exc
        base_dir = DatabasePaths.get_user_base_directory(user_id) / "outputs"
        try:
            base_resolved = base_dir.resolve(strict=False)
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.error(f"outputs: failed to resolve outputs base dir for user {self.user_id}: {exc}")
            raise StorageUnavailableError("storage_unavailable") from exc
        return normalize_output_storage_filename(
            storage_path=path_value,
            allow_absolute=False,
            reject_relative_with_separators=True,
            base_resolved=base_resolved,
            check_relative_containment=True,
            log_message=logger.warning,
            log_prefix="outputs",
        )

    def resolve_temp_output_storage_path(self, path_value: str | Path) -> str:
        """Resolve and validate transient output storage paths as relative filenames."""
        try:
            user_id = int(self.user_id)
        except (TypeError, ValueError) as exc:
            logger.error(f"temp outputs: invalid user id for storage path resolution: {self.user_id}")
            raise InvalidStorageUserIdError("invalid_user_id") from exc
        base_dir = DatabasePaths.get_user_temp_outputs_dir(user_id)
        try:
            base_resolved = base_dir.resolve(strict=False)
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.error(f"temp outputs: failed to resolve outputs base dir for user {self.user_id}: {exc}")
            raise StorageUnavailableError("storage_unavailable") from exc
        return normalize_output_storage_filename(
            storage_path=path_value,
            allow_absolute=False,
            reject_relative_with_separators=True,
            base_resolved=base_resolved,
            check_relative_containment=True,
            log_message=logger.warning,
            log_prefix="temp outputs",
        )

    @dataclass
    class OutputArtifactRow:
        id: int
        user_id: str
        job_id: int | None
        run_id: int | None
        type: str
        title: str
        format: str
        storage_path: str
        metadata_json: str | None
        workspace_tag: str | None
        created_at: str
        media_item_id: int | None
        chatbook_path: str | None

    def create_output_artifact(
        self,
        *,
        type_: str,
        title: str,
        format_: str,
        storage_path: str,
        metadata_json: str | None = None,
        workspace_tag: str | None = None,
        job_id: int | None = None,
        run_id: int | None = None,
        media_item_id: int | None = None,
        chatbook_path: str | None = None,
        retention_until: str | None = None,
    ) -> CollectionsDatabase.OutputArtifactRow:
        now = _utcnow_iso()
        resolved_storage_path = self.resolve_output_storage_path(storage_path)
        q = (
            "INSERT INTO outputs (user_id, job_id, run_id, type, title, format, storage_path, metadata_json, workspace_tag, created_at, media_item_id, chatbook_path, deleted, deleted_at, retention_until) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?)"
        )
        params = (
            self.user_id,
            job_id,
            run_id,
            type_,
            title,
            format_,
            resolved_storage_path,
            metadata_json,
            workspace_tag,
            now,
            media_item_id,
            chatbook_path,
            retention_until,
        )
        res = self._execute_insert(q, params)
        new_id = self._extract_lastrowid(res)
        if not new_id:
            raise DatabaseError("Failed to create output artifact")
        return self.get_output_artifact(new_id)

    def update_output_media_item_id(self, output_id: int, media_item_id: int | None) -> CollectionsDatabase.OutputArtifactRow:
        q = "UPDATE outputs SET media_item_id = ? WHERE id = ? AND user_id = ?"
        res = self.backend.execute(q, (media_item_id, output_id, self.user_id))
        if res.rowcount <= 0:
            raise KeyError("output_not_found")
        return self.get_output_artifact(output_id)

    def update_output_artifact_metadata(
        self,
        output_id: int,
        *,
        metadata_json: str | None = None,
        chatbook_path: str | None = None,
    ) -> CollectionsDatabase.OutputArtifactRow:
        fields: list[str] = []
        params: list[Any] = []
        if metadata_json is not None:
            fields.append("metadata_json = ?")
            params.append(metadata_json)
        if chatbook_path is not None:
            fields.append("chatbook_path = ?")
            params.append(chatbook_path)
        if not fields:
            return self.get_output_artifact(output_id)
        params.extend([output_id, self.user_id])
        q = f"UPDATE outputs SET {', '.join(fields)} WHERE id = ? AND user_id = ? AND deleted = 0"  # nosec B608
        res = self.backend.execute(q, tuple(params))
        if res.rowcount <= 0:
            raise KeyError("output_not_found")
        return self.get_output_artifact(output_id)

    def get_output_artifact(self, output_id: int, include_deleted: bool = False) -> CollectionsDatabase.OutputArtifactRow:
        cond = "id = ? AND user_id = ?" + ("" if include_deleted else " AND deleted = 0")
        q = (
            "SELECT id, user_id, job_id, run_id, type, title, format, storage_path, metadata_json, workspace_tag, created_at, media_item_id, chatbook_path "  # nosec B608
            f"FROM outputs WHERE {cond}"
        )
        row = self.backend.execute(q, (output_id, self.user_id)).first
        if not row:
            raise KeyError("output_not_found")
        return CollectionsDatabase.OutputArtifactRow(**row)

    def delete_output_artifact(self, output_id: int, *, hard: bool = False) -> bool:
        row = self.backend.execute(
            "SELECT id, type, metadata_json, storage_path, deleted FROM outputs WHERE id = ? AND user_id = ?",
            (output_id, self.user_id),
        ).first
        if not row:
            return False
        deleted_flag = int(row["deleted"] if isinstance(row, dict) else row[4] or 0)
        output_type = row["type"] if isinstance(row, dict) else row[1]
        metadata_json = row["metadata_json"] if isinstance(row, dict) else row[2]
        storage_path = row["storage_path"] if isinstance(row, dict) else row[3]
        should_decrement = deleted_flag == 0 and _is_audiobook_output_type(output_type)
        size_bytes = None
        if should_decrement:
            size_bytes = _extract_output_byte_size(metadata_json)
            if size_bytes is None:
                size_bytes = _resolve_output_size_bytes(self.user_id, storage_path)

        if hard:
            q = "DELETE FROM outputs WHERE id = ? AND user_id = ?"
            res = self.backend.execute(q, (output_id, self.user_id))
        else:
            q = "UPDATE outputs SET deleted = 1, deleted_at = ? WHERE id = ? AND user_id = ? AND deleted = 0"
            res = self.backend.execute(q, (_utcnow_iso(), output_id, self.user_id))
        ok = res.rowcount > 0
        if ok and should_decrement and size_bytes:
            try:
                self.update_audiobook_output_usage(-size_bytes)
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning("audiobook_quota: failed to decrement usage: {}", exc)
        return ok

    def get_output_artifact_by_title(self, title: str, format_: str | None = None, include_deleted: bool = False) -> CollectionsDatabase.OutputArtifactRow:
        where = ["user_id = ?", "title = ?"]
        params: list[Any] = [self.user_id, title]
        if format_:
            where.append("format = ?")
            params.append(format_)
        if not include_deleted:
            where.append("deleted = 0")
        q = (
            "SELECT id, user_id, job_id, run_id, type, title, format, storage_path, metadata_json, workspace_tag, created_at, media_item_id, chatbook_path "  # nosec B608
            f"FROM outputs WHERE {' AND '.join(where)} ORDER BY created_at DESC LIMIT 1"
        )
        row = self.backend.execute(q, tuple(params)).first
        if not row:
            raise KeyError("output_not_found")
        return CollectionsDatabase.OutputArtifactRow(**row)

    def list_output_artifacts(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        job_id: int | None = None,
        run_id: int | None = None,
        type_: str | None = None,
        workspace_tag: str | None = None,
        metadata_origin: str | None = None,
        metadata_presentation_id: str | None = None,
        include_deleted: bool = False,
        only_deleted: bool = False,
    ) -> tuple[list[CollectionsDatabase.OutputArtifactRow], int]:
        where = ["user_id = ?"]
        params: list[Any] = [self.user_id]
        if only_deleted:
            where.append("deleted = 1")
        elif not include_deleted:
            where.append("deleted = 0")
        if job_id is not None:
            where.append("job_id = ?")
            params.append(job_id)
        if run_id is not None:
            where.append("run_id = ?")
            params.append(run_id)
        if type_:
            where.append("type = ?")
            params.append(type_)
        if workspace_tag:
            where.append("workspace_tag = ?")
            params.append(workspace_tag)
        if metadata_origin:
            origin = str(metadata_origin).strip()
            if origin:
                where.append("(metadata_json LIKE ? OR metadata_json LIKE ?)")
                params.extend([
                    f'%\"origin\":\"{origin}\"%',
                    f'%\"origin\": \"{origin}\"%',
                ])
        if metadata_presentation_id:
            presentation_id = str(metadata_presentation_id).strip()
            if presentation_id:
                where.append("(metadata_json LIKE ? OR metadata_json LIKE ?)")
                params.extend([
                    f'%\"presentation_id\":\"{presentation_id}\"%',
                    f'%\"presentation_id\": \"{presentation_id}\"%',
                ])
        where_sql = " AND ".join(where)

        cq = f"SELECT COUNT(*) AS cnt FROM outputs WHERE {where_sql}"  # nosec B608
        total = int(self.backend.execute(cq, tuple(params)).scalar or 0)
        sq = (
            "SELECT id, user_id, job_id, run_id, type, title, format, storage_path, metadata_json, workspace_tag, created_at, media_item_id, chatbook_path "  # nosec B608
            f"FROM outputs WHERE {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        rows = self.backend.execute(sq, tuple(params + [limit, offset])).rows
        return [CollectionsDatabase.OutputArtifactRow(**row) for row in rows], total

    def list_output_artifacts_by_types(
        self,
        *,
        types: list[str],
        limit: int = 200,
        offset: int = 0,
        include_deleted: bool = False,
        workspace_tag: str | None = None,
    ) -> tuple[list[CollectionsDatabase.OutputArtifactRow], int]:
        if not types:
            return [], 0
        where = ["user_id = ?"]
        params: list[Any] = [self.user_id]
        if not include_deleted:
            where.append("deleted = 0")
        placeholders = ", ".join(["?"] * len(types))
        where.append(f"type IN ({placeholders})")
        params.extend(types)
        if workspace_tag:
            where.append("workspace_tag = ?")
            params.append(workspace_tag)
        where_sql = " AND ".join(where)
        cq = f"SELECT COUNT(*) AS cnt FROM outputs WHERE {where_sql}"  # nosec B608
        total = int(self.backend.execute(cq, tuple(params)).scalar or 0)
        sq = (
            "SELECT id, user_id, job_id, run_id, type, title, format, storage_path, metadata_json, workspace_tag, created_at, media_item_id, chatbook_path "  # nosec B608
            f"FROM outputs WHERE {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        rows = self.backend.execute(sq, tuple(params + [limit, offset])).rows
        return [CollectionsDatabase.OutputArtifactRow(**row) for row in rows], total

    def get_audiobook_output_usage(self) -> int | None:
        row = self.backend.execute(
            "SELECT used_bytes FROM audiobook_output_usage WHERE user_id = ?",
            (self.user_id,),
        ).first
        if not row:
            return None
        value = row["used_bytes"] if isinstance(row, dict) else row[0]
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return None

    def set_audiobook_output_usage(self, used_bytes: int) -> int:
        value = max(0, int(used_bytes))
        now = _utcnow_iso()
        if self.backend.backend_type == BackendType.POSTGRESQL:
            row = self.backend.execute(
                """
                INSERT INTO audiobook_output_usage (user_id, used_bytes, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET used_bytes = EXCLUDED.used_bytes, updated_at = EXCLUDED.updated_at
                RETURNING used_bytes
                """,
                (self.user_id, value, now),
            ).first
            if row:
                return int(row["used_bytes"] if isinstance(row, dict) else row[0] or 0)
            return value
        self.backend.execute(
            """
            INSERT INTO audiobook_output_usage (user_id, used_bytes, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE
            SET used_bytes = excluded.used_bytes, updated_at = excluded.updated_at
            """,
            (self.user_id, value, now),
        )
        return value

    def update_audiobook_output_usage(self, delta_bytes: int) -> int:
        delta = int(delta_bytes or 0)
        now = _utcnow_iso()
        initial = max(0, delta)
        if self.backend.backend_type == BackendType.POSTGRESQL:
            row = self.backend.execute(
                """
                INSERT INTO audiobook_output_usage (user_id, used_bytes, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET used_bytes = GREATEST(0, audiobook_output_usage.used_bytes + %s),
                    updated_at = EXCLUDED.updated_at
                RETURNING used_bytes
                """,
                (self.user_id, initial, now, delta),
            ).first
            if row:
                return int(row["used_bytes"] if isinstance(row, dict) else row[0] or 0)
            return initial
        self.backend.execute(
            """
            INSERT INTO audiobook_output_usage (user_id, used_bytes, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE
            SET used_bytes = MAX(0, audiobook_output_usage.used_bytes + ?),
                updated_at = excluded.updated_at
            """,
            (self.user_id, initial, now, delta),
        )
        row = self.backend.execute(
            "SELECT used_bytes FROM audiobook_output_usage WHERE user_id = ?",
            (self.user_id,),
        ).first
        if not row:
            return initial
        return int(row["used_bytes"] if isinstance(row, dict) else row[0] or 0)

    def recompute_audiobook_output_usage(self) -> int:
        try:
            user_int = int(self.user_id)
        except (TypeError, ValueError):
            logger.warning("audiobook_quota: invalid user id for recompute: {}", self.user_id)
            return 0
        DatabasePaths.get_user_outputs_dir(user_int)
        total_bytes = 0
        offset = 0
        limit = 200
        while True:
            rows, total = self.list_output_artifacts_by_types(
                types=["audiobook_audio", "audiobook_subtitle", "audiobook_alignment", "audiobook_package"],
                limit=limit,
                offset=offset,
            )
            for row in rows:
                size = _extract_output_byte_size(row.metadata_json)
                if size is None:
                    size = _resolve_output_size_bytes(self.user_id, row.storage_path)
                if size:
                    total_bytes += size
            offset += len(rows)
            if offset >= total or not rows:
                break
        return self.set_audiobook_output_usage(total_bytes)

    def rename_output_artifact(self, output_id: int, new_title: str, new_storage_path: str | None = None) -> CollectionsDatabase.OutputArtifactRow:
        fields = ["title = ?"]
        params: list[Any] = [new_title]
        if new_storage_path is not None:
            new_storage_path = self.resolve_output_storage_path(new_storage_path)
            fields.append("storage_path = ?")
            params.append(new_storage_path)
        params.extend([output_id, self.user_id])
        q = f"UPDATE outputs SET {', '.join(fields)} WHERE id = ? AND user_id = ? AND deleted = 0"  # nosec B608
        res = self.backend.execute(q, tuple(params))
        if res.rowcount <= 0:
            raise KeyError("output_not_found")
        return self.get_output_artifact(output_id)

    # ------------------------
    # Audiobook voice profiles
    # ------------------------
    def create_audiobook_project(
        self,
        *,
        project_id: str | None,
        title: str | None,
        source_ref: str | None,
        status: str | None,
        settings_json: str | None,
    ) -> AudiobookProjectRow:
        now = _utcnow_iso()
        q = (
            "INSERT INTO audiobook_projects (user_id, project_id, title, source_ref, status, settings_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            self.user_id,
            project_id,
            title,
            source_ref,
            status,
            settings_json,
            now,
            now,
        )
        res = self._execute_insert(q, params)
        project_id = self._extract_lastrowid(res)
        if project_id is None:
            raise DatabaseError("audiobook_project_insert_failed")
        return self.get_audiobook_project(project_id)

    def get_audiobook_project(self, project_id: int) -> AudiobookProjectRow:
        q = (
            "SELECT id, user_id, project_id, title, source_ref, status, settings_json, created_at, updated_at "
            "FROM audiobook_projects WHERE id = ? AND user_id = ?"
        )
        row = self.backend.execute(q, (project_id, self.user_id)).first
        if not row:
            raise KeyError("audiobook_project_not_found")
        return AudiobookProjectRow(**row)

    def get_audiobook_project_by_project_id(self, project_id: str) -> AudiobookProjectRow:
        q = (
            "SELECT id, user_id, project_id, title, source_ref, status, settings_json, created_at, updated_at "
            "FROM audiobook_projects WHERE user_id = ? AND project_id = ?"
        )
        row = self.backend.execute(q, (self.user_id, project_id)).first
        if not row:
            raise KeyError("audiobook_project_not_found")
        return AudiobookProjectRow(**row)

    def list_audiobook_projects(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AudiobookProjectRow]:
        q = (
            "SELECT id, user_id, project_id, title, source_ref, status, settings_json, created_at, updated_at "
            "FROM audiobook_projects WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        rows = self.backend.execute(q, (self.user_id, limit, offset)).rows
        return [AudiobookProjectRow(**row) for row in rows]

    def update_audiobook_project_status(
        self,
        project_id: int,
        *,
        status: str,
        settings_json: str | None = None,
    ) -> AudiobookProjectRow:
        fields = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status, _utcnow_iso()]
        if settings_json is not None:
            fields.append("settings_json = ?")
            params.append(settings_json)
        params.extend([project_id, self.user_id])
        q = f"UPDATE audiobook_projects SET {', '.join(fields)} WHERE id = ? AND user_id = ?"  # nosec B608
        res = self.backend.execute(q, tuple(params))
        if res.rowcount <= 0:
            raise KeyError("audiobook_project_not_found")
        return self.get_audiobook_project(project_id)

    def create_audiobook_chapter(
        self,
        *,
        project_id: int,
        chapter_index: int,
        title: str | None,
        start_offset: int | None,
        end_offset: int | None,
        voice_profile_id: str | None,
        speed: float | None,
        metadata_json: str | None,
    ) -> AudiobookChapterRow:
        q = (
            "INSERT INTO audiobook_chapters (project_id, chapter_index, title, start_offset, end_offset, "
            "voice_profile_id, speed, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            project_id,
            chapter_index,
            title,
            start_offset,
            end_offset,
            voice_profile_id,
            speed,
            metadata_json,
        )
        res = self._execute_insert(q, params)
        chapter_id = self._extract_lastrowid(res)
        if chapter_id is None:
            raise DatabaseError("audiobook_chapter_insert_failed")
        return self.get_audiobook_chapter(chapter_id)

    def get_audiobook_chapter(self, chapter_id: int) -> AudiobookChapterRow:
        q = (
            "SELECT id, project_id, chapter_index, title, start_offset, end_offset, voice_profile_id, speed, metadata_json "
            "FROM audiobook_chapters WHERE id = ?"
        )
        row = self.backend.execute(q, (chapter_id,)).first
        if not row:
            raise KeyError("audiobook_chapter_not_found")
        return AudiobookChapterRow(**row)

    def list_audiobook_chapters(
        self,
        *,
        project_id: int,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[AudiobookChapterRow]:
        q = (
            "SELECT id, project_id, chapter_index, title, start_offset, end_offset, voice_profile_id, speed, metadata_json "
            "FROM audiobook_chapters WHERE project_id = ? ORDER BY chapter_index ASC LIMIT ? OFFSET ?"
        )
        rows = self.backend.execute(q, (project_id, limit, offset)).rows
        return [AudiobookChapterRow(**row) for row in rows]

    def create_audiobook_artifact(
        self,
        *,
        project_id: int,
        artifact_type: str,
        format_: str,
        output_id: int,
        metadata_json: str | None,
    ) -> AudiobookArtifactRow:
        q = (
            "INSERT INTO audiobook_artifacts (project_id, artifact_type, format, output_id, metadata_json) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        params = (
            project_id,
            artifact_type,
            format_,
            output_id,
            metadata_json,
        )
        res = self._execute_insert(q, params)
        artifact_id = self._extract_lastrowid(res)
        if artifact_id is None:
            raise DatabaseError("audiobook_artifact_insert_failed")
        return self.get_audiobook_artifact(artifact_id)

    def get_audiobook_artifact(self, artifact_id: int) -> AudiobookArtifactRow:
        q = (
            "SELECT id, project_id, artifact_type, format, output_id, metadata_json "
            "FROM audiobook_artifacts WHERE id = ?"
        )
        row = self.backend.execute(q, (artifact_id,)).first
        if not row:
            raise KeyError("audiobook_artifact_not_found")
        return AudiobookArtifactRow(**row)

    def list_audiobook_artifacts(
        self,
        *,
        project_id: int,
        limit: int = 2000,
        offset: int = 0,
    ) -> list[AudiobookArtifactRow]:
        q = (
            "SELECT id, project_id, artifact_type, format, output_id, metadata_json "
            "FROM audiobook_artifacts WHERE project_id = ? ORDER BY id ASC LIMIT ? OFFSET ?"
        )
        rows = self.backend.execute(q, (project_id, limit, offset)).rows
        return [AudiobookArtifactRow(**row) for row in rows]

    def create_voice_profile(
        self,
        *,
        profile_id: str,
        name: str,
        default_voice: str,
        default_speed: float,
        chapter_overrides_json: str | None,
    ) -> VoiceProfileRow:
        now = _utcnow_iso()
        q = (
            "INSERT INTO audiobook_voice_profiles (profile_id, user_id, name, default_voice, default_speed, chapter_overrides_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            profile_id,
            self.user_id,
            name,
            default_voice,
            default_speed,
            chapter_overrides_json,
            now,
            now,
        )
        self.backend.execute(q, params)
        return self.get_voice_profile(profile_id)

    def get_voice_profile(self, profile_id: str) -> VoiceProfileRow:
        q = (
            "SELECT profile_id, user_id, name, default_voice, default_speed, chapter_overrides_json, created_at, updated_at "
            "FROM audiobook_voice_profiles WHERE profile_id = ? AND user_id = ?"
        )
        row = self.backend.execute(q, (profile_id, self.user_id)).first
        if not row:
            raise KeyError("voice_profile_not_found")
        return VoiceProfileRow(**row)

    def list_voice_profiles(self, *, limit: int = 100, offset: int = 0) -> list[VoiceProfileRow]:
        q = (
            "SELECT profile_id, user_id, name, default_voice, default_speed, chapter_overrides_json, created_at, updated_at "
            "FROM audiobook_voice_profiles WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        rows = self.backend.execute(q, (self.user_id, limit, offset)).rows
        return [VoiceProfileRow(**row) for row in rows]

    def delete_voice_profile(self, profile_id: str) -> None:
        q = "DELETE FROM audiobook_voice_profiles WHERE profile_id = ? AND user_id = ?"
        res = self.backend.execute(q, (profile_id, self.user_id))
        if res.rowcount <= 0:
            raise KeyError("voice_profile_not_found")

    # ------------------------
    # Reading digest schedules API
    # ------------------------
    @staticmethod
    def _coerce_truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "t"}

    def _reading_digest_row_from_db(self, row: dict[str, Any]) -> ReadingDigestScheduleRow:
        return ReadingDigestScheduleRow(
            id=str(row.get("id")),
            tenant_id=str(row.get("tenant_id") or "default"),
            user_id=str(row.get("user_id")),
            name=row.get("name"),
            cron=str(row.get("cron")),
            timezone=row.get("timezone"),
            enabled=self._coerce_truthy(row.get("enabled")),
            require_online=self._coerce_truthy(row.get("require_online")),
            filters_json=row.get("filters_json") or "{}",
            template_id=row.get("template_id"),
            template_name=row.get("template_name"),
            format=str(row.get("format") or "md"),
            retention_days=row.get("retention_days"),
            last_run_at=row.get("last_run_at"),
            next_run_at=row.get("next_run_at"),
            last_status=row.get("last_status"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    def create_reading_digest_schedule(
        self,
        *,
        id: str,
        tenant_id: str,
        name: str | None,
        cron: str,
        timezone: str | None,
        enabled: bool,
        require_online: bool,
        filters: dict[str, Any],
        template_id: int | None,
        template_name: str | None,
        format: str,
        retention_days: int | None,
    ) -> ReadingDigestScheduleRow:
        """Create a reading digest schedule for the current user.

        Params:
            id: Schedule identifier to persist.
            tenant_id: Tenant identifier.
            name: Optional display name.
            cron: Cron expression.
            timezone: Optional IANA timezone.
            enabled: Whether the schedule is active.
            require_online: Whether runs require online status.
            filters: Filter payload stored as JSON.
            template_id: Optional output template id.
            template_name: Optional output template name.
            format: Output format ("md" or "html").
            retention_days: Optional retention window in days.

        Returns:
            Newly created schedule row.

        Raises:
            DatabaseError: Insert or lookup failure.
            KeyError: Schedule not found after insert.
        """
        now = _utcnow_iso()
        enabled_flag = self._coerce_bool_flag(enabled, postgres=self.backend.backend_type == BackendType.POSTGRESQL)
        online_flag = self._coerce_bool_flag(require_online, postgres=self.backend.backend_type == BackendType.POSTGRESQL)
        params = (
            id,
            tenant_id,
            self.user_id,
            name,
            cron,
            timezone,
            enabled_flag,
            online_flag,
            json.dumps(filters or {}),
            template_id,
            template_name,
            format or "md",
            retention_days,
            None,
            None,
            None,
            now,
            now,
        )
        q = (
            "INSERT INTO reading_digest_schedules ("
            "id, tenant_id, user_id, name, cron, timezone, enabled, require_online, filters_json, "
            "template_id, template_name, format, retention_days, last_run_at, next_run_at, last_status, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        self._execute_insert(q, params)
        return self.get_reading_digest_schedule(id)

    def update_reading_digest_schedule(self, schedule_id: str, patch: dict[str, Any]) -> ReadingDigestScheduleRow:
        """Update a reading digest schedule for the current user.

        Params:
            schedule_id: Schedule identifier to update.
            patch: Fields to update (filters, schedule fields, or history columns).

        Returns:
            Updated schedule row.

        Raises:
            KeyError: Schedule not found.
            DatabaseError: Update or lookup failure.
        """
        if not patch:
            return self.get_reading_digest_schedule(schedule_id)
        fields = []
        params: list[Any] = []
        if "filters" in patch:
            fields.append("filters_json = ?")
            params.append(json.dumps(patch.get("filters") or {}))
        if "enabled" in patch:
            fields.append("enabled = ?")
            params.append(self._coerce_bool_flag(patch.get("enabled"), postgres=self.backend.backend_type == BackendType.POSTGRESQL))
        if "require_online" in patch:
            fields.append("require_online = ?")
            params.append(self._coerce_bool_flag(patch.get("require_online"), postgres=self.backend.backend_type == BackendType.POSTGRESQL))
        for key in (
            "name",
            "cron",
            "timezone",
            "template_id",
            "template_name",
            "format",
            "retention_days",
            "last_run_at",
            "next_run_at",
            "last_status",
        ):
            if key in patch:
                fields.append(f"{key} = ?")
                params.append(patch.get(key))
        fields.append("updated_at = ?")
        params.append(_utcnow_iso())
        params.extend([schedule_id, self.user_id])
        q = f"UPDATE reading_digest_schedules SET {', '.join(fields)} WHERE id = ? AND user_id = ?"  # nosec B608
        res = self.backend.execute(q, tuple(params))
        if res.rowcount <= 0:
            raise KeyError("reading_digest_schedule_not_found")
        return self.get_reading_digest_schedule(schedule_id)

    def delete_reading_digest_schedule(self, schedule_id: str) -> bool:
        """Delete a reading digest schedule for the current user.

        Params:
            schedule_id: Schedule identifier to delete.

        Returns:
            True when a row was deleted, otherwise False.

        Raises:
            DatabaseError: Delete failure.
        """
        q = "DELETE FROM reading_digest_schedules WHERE id = ? AND user_id = ?"
        res = self.backend.execute(q, (schedule_id, self.user_id))
        return res.rowcount > 0

    def get_reading_digest_schedule(self, schedule_id: str) -> ReadingDigestScheduleRow:
        """Fetch a reading digest schedule for the current user.

        Params:
            schedule_id: Schedule identifier to retrieve.

        Returns:
            Schedule row.

        Raises:
            KeyError: Schedule not found.
            DatabaseError: Query failure.
        """
        q = "SELECT * FROM reading_digest_schedules WHERE id = ? AND user_id = ?"
        row = self.backend.execute(q, (schedule_id, self.user_id)).first
        if not row:
            raise KeyError("reading_digest_schedule_not_found")
        return self._reading_digest_row_from_db(row)

    def list_reading_digest_schedules(
        self,
        *,
        tenant_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ReadingDigestScheduleRow], int]:
        """List reading digest schedules for the current user and tenant.

        Params:
            tenant_id: Tenant identifier to filter schedules.
            limit: Maximum number of rows to return.
            offset: Starting offset for pagination.

        Returns:
            Tuple of (schedule rows, total count).

        Raises:
            DatabaseError: Query failure.
        """
        where = "user_id = ? AND tenant_id = ?"
        params: list[Any] = [self.user_id, tenant_id]
        count_q = f"SELECT COUNT(*) AS cnt FROM reading_digest_schedules WHERE {where}"  # nosec B608
        total = int(self.backend.execute(count_q, tuple(params)).scalar or 0)
        q = (
            "SELECT * FROM reading_digest_schedules "  # nosec B608
            f"WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        rows = self.backend.execute(q, (*params, limit, offset)).rows
        return [self._reading_digest_row_from_db(row) for row in rows], total

    def set_reading_digest_history(
        self,
        schedule_id: str,
        *,
        last_run_at: str | None = None,
        next_run_at: str | None = None,
        last_status: str | None = None,
    ) -> None:
        """Update reading digest schedule history fields for the current user.

        Params:
            schedule_id: Schedule identifier to update.
            last_run_at: ISO timestamp of last run (optional).
            next_run_at: ISO timestamp of next scheduled run (optional).
            last_status: Last run status string (optional).

        Returns:
            None.

        Raises:
            KeyError: Schedule not found.
            DatabaseError: Update failure.
        """
        update: dict[str, Any] = {}
        if last_run_at is not None:
            update["last_run_at"] = last_run_at
        if next_run_at is not None:
            update["next_run_at"] = next_run_at
        if last_status is not None:
            update["last_status"] = last_status
        if not update:
            return
        self.update_reading_digest_schedule(schedule_id, update)

    def try_claim_reading_digest_run(
        self,
        schedule_id: str,
        *,
        expected_next_run_at: str | None,
        next_run_at: str | None,
        last_run_at: str | None,
        last_status: str | None,
        disallow_statuses: tuple[str, ...] | None = None,
    ) -> bool:
        """Attempt to claim a schedule run by updating run timestamps/status.

        Params:
            schedule_id: Schedule identifier to claim.
            expected_next_run_at: Required current next_run_at for optimistic claim.
            next_run_at: New next_run_at timestamp to set.
            last_run_at: Last run timestamp to set.
            last_status: Last run status to set.
            disallow_statuses: Status values that block the claim.

        Returns:
            True if the schedule was updated (claim succeeded), otherwise False.

        Raises:
            DatabaseError: Update failure.
        """
        fields = ["next_run_at = ?", "last_run_at = ?", "last_status = ?"]
        params: list[Any] = [next_run_at, last_run_at, last_status, schedule_id, self.user_id]
        status_clause = ""
        status_params: list[Any] = []
        if disallow_statuses:
            placeholders = ", ".join(["?"] * len(disallow_statuses))
            status_clause = f" AND (last_status IS NULL OR last_status NOT IN ({placeholders}))"
            status_params.extend(disallow_statuses)
        if expected_next_run_at is None:
            q = (
                f"UPDATE reading_digest_schedules SET {', '.join(fields)} "  # nosec B608
                "WHERE id = ? AND user_id = ? AND next_run_at IS NULL"
                f"{status_clause}"
            )
        else:
            q = (
                f"UPDATE reading_digest_schedules SET {', '.join(fields)} "  # nosec B608
                "WHERE id = ? AND user_id = ? AND next_run_at = ?"
                f"{status_clause}"
            )
            params.append(expected_next_run_at)
        params.extend(status_params)
        res = self.backend.execute(q, tuple(params))
        return res.rowcount > 0

    # ------------------------
    # Reminder tasks API
    # ------------------------
    def _reminder_task_row_from_db(self, row: dict[str, Any]) -> ReminderTaskRow:
        return ReminderTaskRow(
            id=str(row.get("id")),
            user_id=str(row.get("user_id")),
            tenant_id=str(row.get("tenant_id") or "default"),
            title=str(row.get("title") or ""),
            body=row.get("body"),
            link_type=row.get("link_type"),
            link_id=row.get("link_id"),
            link_url=row.get("link_url"),
            schedule_kind=str(row.get("schedule_kind") or ""),
            run_at=row.get("run_at"),
            cron=row.get("cron"),
            timezone=row.get("timezone"),
            enabled=self._coerce_bool_setting(row.get("enabled"), True),
            last_run_at=row.get("last_run_at"),
            next_run_at=row.get("next_run_at"),
            last_status=row.get("last_status"),
            created_at=str(row.get("created_at") or ""),
            updated_at=str(row.get("updated_at") or ""),
        )

    def create_reminder_task(
        self,
        *,
        title: str,
        body: str | None,
        schedule_kind: str,
        run_at: str | None,
        cron: str | None,
        timezone: str | None,
        enabled: bool = True,
        link_type: str | None = None,
        link_id: str | None = None,
        link_url: str | None = None,
        tenant_id: str = "default",
    ) -> str:
        now = _utcnow_iso()
        task_id = uuid4().hex
        enabled_flag = self._coerce_bool_flag(enabled, postgres=self.backend.backend_type == BackendType.POSTGRESQL)
        q = (
            "INSERT INTO reminder_tasks ("
            "id, user_id, tenant_id, title, body, link_type, link_id, link_url, "
            "schedule_kind, run_at, cron, timezone, enabled, last_run_at, next_run_at, last_status, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            task_id,
            self.user_id,
            tenant_id,
            title,
            body,
            link_type,
            link_id,
            link_url,
            schedule_kind,
            run_at,
            cron,
            timezone,
            enabled_flag,
            None,
            None,
            None,
            now,
            now,
        )
        self._execute_insert(q, params)
        return task_id

    def get_reminder_task(self, task_id: str) -> ReminderTaskRow:
        row = self.backend.execute(
            "SELECT * FROM reminder_tasks WHERE id = ? AND user_id = ?",
            (task_id, self.user_id),
        ).first
        if not row:
            raise KeyError("reminder_task_not_found")
        return self._reminder_task_row_from_db(row)

    def list_reminder_tasks(self, *, include_disabled: bool = True) -> list[ReminderTaskRow]:
        params: list[Any] = [self.user_id]
        where = "user_id = ?"
        if not include_disabled:
            where += " AND enabled = ?"
            params.append(self._coerce_bool_flag(True, postgres=self.backend.backend_type == BackendType.POSTGRESQL))
        q = f"SELECT * FROM reminder_tasks WHERE {where} ORDER BY created_at DESC"  # nosec B608
        rows = self.backend.execute(q, tuple(params)).rows
        return [self._reminder_task_row_from_db(row) for row in rows]

    def update_reminder_task(self, task_id: str, patch: dict[str, Any]) -> ReminderTaskRow:
        if not patch:
            return self.get_reminder_task(task_id)
        fields: list[str] = []
        params: list[Any] = []
        for key in (
            "title",
            "body",
            "link_type",
            "link_id",
            "link_url",
            "schedule_kind",
            "run_at",
            "cron",
            "timezone",
            "last_run_at",
            "next_run_at",
            "last_status",
        ):
            if key in patch:
                fields.append(f"{key} = ?")
                params.append(patch.get(key))
        if "enabled" in patch:
            fields.append("enabled = ?")
            params.append(
                self._coerce_bool_flag(
                    patch.get("enabled"),
                    postgres=self.backend.backend_type == BackendType.POSTGRESQL,
                )
            )
        fields.append("updated_at = ?")
        params.append(_utcnow_iso())
        params.extend([task_id, self.user_id])
        q = f"UPDATE reminder_tasks SET {', '.join(fields)} WHERE id = ? AND user_id = ?"  # nosec B608
        res = self.backend.execute(q, tuple(params))
        if res.rowcount <= 0:
            raise KeyError("reminder_task_not_found")
        return self.get_reminder_task(task_id)

    def delete_reminder_task(self, task_id: str) -> bool:
        res = self.backend.execute(
            "DELETE FROM reminder_tasks WHERE id = ? AND user_id = ?",
            (task_id, self.user_id),
        )
        return bool(res.rowcount and res.rowcount > 0)

    def try_claim_reminder_task_slot(
        self,
        task_id: str,
        *,
        expected_next_run_at: str | None,
        next_run_at: str | None,
        last_run_at: str | None,
        last_status: str | None,
        disallow_statuses: tuple[str, ...] | None = None,
    ) -> bool:
        """Attempt to claim a reminder slot with optimistic next_run_at matching."""
        fields = ["next_run_at = ?", "last_run_at = ?", "last_status = ?", "updated_at = ?"]
        params: list[Any] = [next_run_at, last_run_at, last_status, _utcnow_iso(), task_id, self.user_id]
        status_clause = ""
        status_params: list[Any] = []
        if disallow_statuses:
            placeholders = ", ".join(["?"] * len(disallow_statuses))
            status_clause = f" AND (last_status IS NULL OR last_status NOT IN ({placeholders}))"
            status_params.extend(disallow_statuses)
        enabled_true = self._coerce_bool_flag(True, postgres=self.backend.backend_type == BackendType.POSTGRESQL)
        if expected_next_run_at is None:
            q = (
                f"UPDATE reminder_tasks SET {', '.join(fields)} "  # nosec B608
                "WHERE id = ? AND user_id = ? AND enabled = ? AND next_run_at IS NULL"
                f"{status_clause}"
            )
        else:
            q = (
                f"UPDATE reminder_tasks SET {', '.join(fields)} "  # nosec B608
                "WHERE id = ? AND user_id = ? AND enabled = ? AND next_run_at = ?"
                f"{status_clause}"
            )
            params.append(expected_next_run_at)
        params.insert(6, enabled_true)
        params.extend(status_params)
        res = self.backend.execute(q, tuple(params))
        return res.rowcount > 0

    def _reminder_task_run_row_from_db(self, row: dict[str, Any]) -> ReminderTaskRunRow:
        return ReminderTaskRunRow(
            id=int(row.get("id")),
            task_id=str(row.get("task_id") or ""),
            user_id=str(row.get("user_id") or ""),
            scheduled_for=row.get("scheduled_for"),
            job_id=row.get("job_id"),
            run_slot_utc=str(row.get("run_slot_utc") or ""),
            run_slot_key=str(row.get("run_slot_key") or ""),
            status=str(row.get("status") or ""),
            error=row.get("error"),
            created_at=str(row.get("created_at") or ""),
            started_at=row.get("started_at"),
            completed_at=row.get("completed_at"),
        )

    def create_reminder_task_run(
        self,
        *,
        task_id: str,
        scheduled_for: str | None,
        job_id: str | None,
        run_slot_utc: str,
        run_slot_key: str,
        status: str,
        error: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> ReminderTaskRunRow:
        now = _utcnow_iso()
        q = (
            "INSERT INTO reminder_task_runs ("
            "task_id, user_id, scheduled_for, job_id, run_slot_utc, run_slot_key, status, error, "
            "created_at, started_at, completed_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(task_id, run_slot_key) DO NOTHING"
        )
        params = (
            task_id,
            self.user_id,
            scheduled_for,
            job_id,
            run_slot_utc,
            run_slot_key,
            status,
            error,
            now,
            started_at,
            completed_at,
        )
        self.backend.execute(q, params)
        return self.get_reminder_task_run_by_slot(task_id=task_id, run_slot_key=run_slot_key)

    def get_reminder_task_run_by_slot(self, *, task_id: str, run_slot_key: str) -> ReminderTaskRunRow:
        row = self.backend.execute(
            "SELECT * FROM reminder_task_runs WHERE task_id = ? AND user_id = ? AND run_slot_key = ?",
            (task_id, self.user_id, run_slot_key),
        ).first
        if not row:
            raise KeyError("reminder_task_run_not_found")
        return self._reminder_task_run_row_from_db(row)

    def update_reminder_task_run_status(
        self,
        *,
        run_id: int,
        status: str,
        error: str | None = None,
        completed_at: str | None = None,
    ) -> ReminderTaskRunRow:
        q = (
            "UPDATE reminder_task_runs SET status = ?, error = ?, completed_at = ? "
            "WHERE id = ? AND user_id = ?"
        )
        res = self.backend.execute(q, (status, error, completed_at, run_id, self.user_id))
        if res.rowcount <= 0:
            raise KeyError("reminder_task_run_not_found")
        row = self.backend.execute(
            "SELECT * FROM reminder_task_runs WHERE id = ? AND user_id = ?",
            (run_id, self.user_id),
        ).first
        if not row:
            raise KeyError("reminder_task_run_not_found")
        return self._reminder_task_run_row_from_db(row)

    # ------------------------
    # Notifications API
    # ------------------------
    def _notification_row_from_db(self, row: dict[str, Any]) -> UserNotificationRow:
        return UserNotificationRow(
            id=int(row.get("id")),
            user_id=str(row.get("user_id")),
            kind=str(row.get("kind") or ""),
            title=str(row.get("title") or ""),
            message=str(row.get("message") or ""),
            severity=str(row.get("severity") or "info"),
            source_task_id=row.get("source_task_id"),
            source_task_run_id=(int(row["source_task_run_id"]) if row.get("source_task_run_id") is not None else None),
            source_job_id=row.get("source_job_id"),
            source_domain=row.get("source_domain"),
            source_job_type=row.get("source_job_type"),
            link_type=row.get("link_type"),
            link_id=row.get("link_id"),
            link_url=row.get("link_url"),
            dedupe_key=row.get("dedupe_key"),
            retention_until=row.get("retention_until"),
            archived_at=row.get("archived_at"),
            created_at=str(row.get("created_at") or ""),
            read_at=row.get("read_at"),
            dismissed_at=row.get("dismissed_at"),
            snooze_task_id=row.get("snooze_task_id"),
            delivery_status=str(row.get("delivery_status") or "pending"),
            delivered_at=row.get("delivered_at"),
        )

    def create_user_notification(
        self,
        *,
        kind: str,
        title: str,
        message: str,
        severity: str,
        source_task_id: str | None = None,
        source_task_run_id: int | None = None,
        source_job_id: str | None = None,
        source_domain: str | None = None,
        source_job_type: str | None = None,
        link_type: str | None = None,
        link_id: str | None = None,
        link_url: str | None = None,
        dedupe_key: str | None = None,
        retention_until: str | None = None,
    ) -> UserNotificationRow:
        now = _utcnow_iso()
        q = (
            "INSERT INTO user_notifications ("
            "user_id, kind, title, message, severity, source_task_id, source_task_run_id, source_job_id, "
            "source_domain, source_job_type, link_type, link_id, link_url, dedupe_key, retention_until, archived_at, "
            "created_at, read_at, dismissed_at, delivery_status, delivered_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            self.user_id,
            kind,
            title,
            message,
            severity,
            source_task_id,
            source_task_run_id,
            source_job_id,
            source_domain,
            source_job_type,
            link_type,
            link_id,
            link_url,
            dedupe_key,
            retention_until,
            None,
            now,
            None,
            None,
            "pending",
            None,
        )
        duplicate_dedupe_error = False
        try:
            res = self._execute_insert(q, params)
        except DatabaseError as exc:
            duplicate_dedupe_error = bool(
                dedupe_key
                and (
                    "unique constraint failed: user_notifications.user_id, user_notifications.dedupe_key"
                    in str(exc).lower()
                    or "duplicate key value violates unique constraint" in str(exc).lower()
                    or "ux_user_notifications_user_dedupe" in str(exc).lower()
                )
            )
            if not duplicate_dedupe_error:
                raise
            res = None
        new_id = self._extract_lastrowid(res) if res is not None else None
        if (duplicate_dedupe_error or not new_id) and dedupe_key:
            existing = self.backend.execute(
                "SELECT * FROM user_notifications WHERE user_id = ? AND dedupe_key = ? ORDER BY id DESC LIMIT 1",
                (self.user_id, dedupe_key),
            ).first
            if existing:
                return self._notification_row_from_db(existing)
        if not new_id:
            raise DatabaseError("Failed to create user notification")
        row = self.backend.execute(
            "SELECT * FROM user_notifications WHERE id = ? AND user_id = ?",
            (new_id, self.user_id),
        ).first
        if not row:
            raise KeyError("user_notification_not_found")
        return self._notification_row_from_db(row)

    def update_notification_delivery_status(
        self,
        notification_id: int,
        status: str,
        delivered_at: str | None = None,
    ) -> bool:
        """Update the delivery_status (and optionally delivered_at) of a notification."""
        if delivered_at is not None:
            cursor = self.backend.execute(
                "UPDATE user_notifications SET delivery_status = ?, delivered_at = ? WHERE id = ? AND user_id = ?",
                (status, delivered_at, notification_id, self.user_id),
            )
        else:
            cursor = self.backend.execute(
                "UPDATE user_notifications SET delivery_status = ?, delivered_at = NULL WHERE id = ? AND user_id = ?",
                (status, notification_id, self.user_id),
            )
        return bool(cursor.rowcount and cursor.rowcount > 0)

    def list_user_notifications(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[UserNotificationRow]:
        params: list[Any] = [self.user_id]
        where = "user_id = ?"
        if not include_archived:
            where += " AND archived_at IS NULL AND dismissed_at IS NULL"
        q = f"SELECT * FROM user_notifications WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"  # nosec B608
        params.extend([limit, offset])
        rows = self.backend.execute(q, tuple(params)).rows
        return [self._notification_row_from_db(row) for row in rows]

    def list_user_dismissed_notifications(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[UserNotificationRow]:
        rows = self.backend.execute(
            (
                "SELECT * FROM user_notifications "
                "WHERE user_id = ? AND dismissed_at IS NOT NULL "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?"
            ),
            (self.user_id, limit, offset),
        ).rows
        return [self._notification_row_from_db(row) for row in rows]

    def list_user_notifications_after_id(
        self,
        *,
        after_id: int,
        limit: int = 100,
        include_archived: bool = False,
    ) -> list[UserNotificationRow]:
        bounded_limit = max(1, min(1000, int(limit)))
        if include_archived:
            q = (
                "SELECT * FROM user_notifications "
                "WHERE user_id = ? AND id > ? "
                "ORDER BY id ASC LIMIT ?"
            )
            params = (self.user_id, int(after_id), bounded_limit)
        else:
            q = (
                "SELECT * FROM user_notifications "
                "WHERE user_id = ? AND archived_at IS NULL AND dismissed_at IS NULL AND id > ? "
                "ORDER BY id ASC LIMIT ?"
            )
            params = (self.user_id, int(after_id), bounded_limit)
        rows = self.backend.execute(q, params).rows
        return [self._notification_row_from_db(row) for row in rows]

    def get_user_notifications_latest_id(self, *, include_archived: bool = False) -> int:
        if include_archived:
            q = "SELECT MAX(id) AS max_id FROM user_notifications WHERE user_id = ?"
            row = self.backend.execute(q, (self.user_id,)).first
        else:
            q = (
                "SELECT MAX(id) AS max_id FROM user_notifications "
                "WHERE user_id = ? AND archived_at IS NULL AND dismissed_at IS NULL"
            )
            row = self.backend.execute(q, (self.user_id,)).first
        if not row:
            return 0
        max_id = row.get("max_id")
        return int(max_id or 0)

    def get_user_notifications_window_floor_id(
        self,
        *,
        window_size: int,
        include_archived: bool = False,
    ) -> int | None:
        bounded_window = max(1, int(window_size))
        offset = bounded_window - 1
        if include_archived:
            q = (
                "SELECT id FROM user_notifications "
                "WHERE user_id = ? "
                "ORDER BY id DESC LIMIT 1 OFFSET ?"
            )
            row = self.backend.execute(q, (self.user_id, offset)).first
        else:
            q = (
                "SELECT id FROM user_notifications "
                "WHERE user_id = ? AND archived_at IS NULL AND dismissed_at IS NULL "
                "ORDER BY id DESC LIMIT 1 OFFSET ?"
            )
            row = self.backend.execute(q, (self.user_id, offset)).first
        if not row:
            return None
        raw_id = row.get("id")
        if raw_id is None:
            return None
        return int(raw_id)

    def get_user_notification(self, notification_id: int) -> UserNotificationRow:
        row = self.backend.execute(
            "SELECT * FROM user_notifications WHERE id = ? AND user_id = ?",
            (notification_id, self.user_id),
        ).first
        if not row:
            raise KeyError("user_notification_not_found")
        return self._notification_row_from_db(row)

    def mark_user_notifications_read(self, ids: list[int]) -> int:
        if not ids:
            return 0
        placeholders = ",".join(["?"] * len(ids))
        q = (
            f"UPDATE user_notifications SET read_at = ? WHERE user_id = ? AND id IN ({placeholders}) "  # nosec B608
            "AND read_at IS NULL"
        )
        params: list[Any] = [_utcnow_iso(), self.user_id, *ids]
        res = self.backend.execute(q, tuple(params))
        return int(res.rowcount or 0)

    def dismiss_user_notification(self, notification_id: int) -> bool:
        q = (
            "UPDATE user_notifications SET dismissed_at = ? "
            "WHERE id = ? AND user_id = ? AND dismissed_at IS NULL"
        )
        res = self.backend.execute(q, (_utcnow_iso(), notification_id, self.user_id))
        return bool(res.rowcount and res.rowcount > 0)

    def set_user_notification_snooze_task(self, notification_id: int, task_id: str | None) -> bool:
        res = self.backend.execute(
            "UPDATE user_notifications SET snooze_task_id = ? WHERE id = ? AND user_id = ?",
            (task_id, notification_id, self.user_id),
        )
        return bool(res.rowcount and res.rowcount > 0)

    def delete_user_notifications_by_link(
        self,
        *,
        link_type: str,
        link_ids: list[str],
    ) -> int:
        """Hard-delete notifications matching one link type and a bounded list of link ids."""
        if not link_ids:
            return 0
        placeholders = ",".join(["?"] * len(link_ids))
        q = (
            f"DELETE FROM user_notifications WHERE user_id = ? AND link_type = ? AND link_id IN ({placeholders})"  # nosec B608
        )
        params: list[Any] = [self.user_id, str(link_type), *[str(link_id) for link_id in link_ids]]
        res = self.backend.execute(q, tuple(params))
        return int(res.rowcount or 0)

    def count_unread_user_notifications(self) -> int:
        q = (
            "SELECT COUNT(*) AS cnt FROM user_notifications "
            "WHERE user_id = ? AND read_at IS NULL AND dismissed_at IS NULL AND archived_at IS NULL"
        )
        return int(self.backend.execute(q, (self.user_id,)).scalar or 0)

    @staticmethod
    def _parse_iso_utc(value: Any) -> datetime | None:
        if value in (None, ""):
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def prune_user_notifications(
        self,
        *,
        now_iso: str | None = None,
        retention_days_by_kind: dict[str, int] | None = None,
        read_dismissed_grace_days: int = 30,
        archive_grace_days: int = 7,
    ) -> tuple[int, int]:
        """Archive expired notifications and hard-delete stale archived notifications.

        Returns:
            (archived_count, deleted_count)
        """
        now_dt = self._parse_iso_utc(now_iso) or datetime.now(timezone.utc)
        now_effective_iso = now_dt.isoformat()

        retention = dict(retention_days_by_kind or {})
        accel_days = max(0, int(read_dismissed_grace_days))
        grace_days = max(0, int(archive_grace_days))
        archived_cutoff_dt = now_dt - timedelta(days=grace_days)

        rows = self.backend.execute(
            "SELECT id, kind, created_at, read_at, dismissed_at, retention_until, archived_at "
            "FROM user_notifications WHERE user_id = ?",
            (self.user_id,),
        ).rows

        archive_ids: list[int] = []
        delete_ids: list[int] = []
        for row in rows:
            if isinstance(row, dict):
                row_id = int(row.get("id"))
                kind = str(row.get("kind") or "")
                created_at = self._parse_iso_utc(row.get("created_at"))
                read_at = self._parse_iso_utc(row.get("read_at"))
                dismissed_at = self._parse_iso_utc(row.get("dismissed_at"))
                retention_until = self._parse_iso_utc(row.get("retention_until"))
                archived_at = self._parse_iso_utc(row.get("archived_at"))
            else:
                row_id = int(row[0])
                kind = str(row[1] or "")
                created_at = self._parse_iso_utc(row[2])
                read_at = self._parse_iso_utc(row[3])
                dismissed_at = self._parse_iso_utc(row[4])
                retention_until = self._parse_iso_utc(row[5])
                archived_at = self._parse_iso_utc(row[6])

            if archived_at is not None:
                if archived_at <= archived_cutoff_dt:
                    delete_ids.append(row_id)
                continue

            expire_candidates: list[datetime] = []
            if retention_until is not None:
                expire_candidates.append(retention_until)

            default_days = retention.get(kind)
            if default_days is not None and default_days > 0 and created_at is not None:
                expire_candidates.append(created_at + timedelta(days=int(default_days)))

            if accel_days > 0:
                if read_at is not None:
                    expire_candidates.append(read_at + timedelta(days=accel_days))
                if dismissed_at is not None:
                    expire_candidates.append(dismissed_at + timedelta(days=accel_days))

            if expire_candidates and min(expire_candidates) <= now_dt:
                archive_ids.append(row_id)

        archived = 0
        if archive_ids:
            placeholders = ",".join(["?"] * len(archive_ids))
            q = (
                f"UPDATE user_notifications SET archived_at = ? "  # nosec B608
                f"WHERE user_id = ? AND archived_at IS NULL AND id IN ({placeholders})"
            )
            params: list[Any] = [now_effective_iso, self.user_id, *archive_ids]
            res = self.backend.execute(q, tuple(params))
            archived = int(res.rowcount or 0)

        deleted = 0
        if delete_ids:
            placeholders = ",".join(["?"] * len(delete_ids))
            q = f"DELETE FROM user_notifications WHERE user_id = ? AND id IN ({placeholders})"  # nosec B608
            params = tuple([self.user_id, *delete_ids])
            res = self.backend.execute(q, params)
            deleted = int(res.rowcount or 0)

        return archived, deleted

    def _notification_preferences_row_from_db(self, row: dict[str, Any]) -> NotificationPreferencesRow:
        return NotificationPreferencesRow(
            user_id=str(row.get("user_id")),
            reminder_enabled=self._coerce_bool_setting(row.get("reminder_enabled"), True),
            job_completed_enabled=self._coerce_bool_setting(row.get("job_completed_enabled"), True),
            job_failed_enabled=self._coerce_bool_setting(row.get("job_failed_enabled"), True),
            updated_at=str(row.get("updated_at") or ""),
        )

    def get_notification_preferences(self) -> NotificationPreferencesRow:
        row = self.backend.execute(
            "SELECT * FROM notification_preferences WHERE user_id = ?",
            (self.user_id,),
        ).first
        if row:
            return self._notification_preferences_row_from_db(row)
        now = _utcnow_iso()
        return NotificationPreferencesRow(
            user_id=self.user_id,
            reminder_enabled=True,
            job_completed_enabled=True,
            job_failed_enabled=True,
            updated_at=now,
        )

    def update_notification_preferences(
        self,
        *,
        reminder_enabled: bool | None = None,
        job_completed_enabled: bool | None = None,
        job_failed_enabled: bool | None = None,
    ) -> NotificationPreferencesRow:
        current = self.get_notification_preferences()
        reminder = current.reminder_enabled if reminder_enabled is None else bool(reminder_enabled)
        completed = current.job_completed_enabled if job_completed_enabled is None else bool(job_completed_enabled)
        failed = current.job_failed_enabled if job_failed_enabled is None else bool(job_failed_enabled)
        now = _utcnow_iso()

        reminder_flag = self._coerce_bool_flag(reminder, postgres=self.backend.backend_type == BackendType.POSTGRESQL)
        completed_flag = self._coerce_bool_flag(completed, postgres=self.backend.backend_type == BackendType.POSTGRESQL)
        failed_flag = self._coerce_bool_flag(failed, postgres=self.backend.backend_type == BackendType.POSTGRESQL)
        q = (
            "INSERT INTO notification_preferences (user_id, reminder_enabled, job_completed_enabled, job_failed_enabled, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET "
            "reminder_enabled = excluded.reminder_enabled, "
            "job_completed_enabled = excluded.job_completed_enabled, "
            "job_failed_enabled = excluded.job_failed_enabled, "
            "updated_at = excluded.updated_at"
        )
        self.backend.execute(q, (self.user_id, reminder_flag, completed_flag, failed_flag, now))
        return self.get_notification_preferences()

    def _notification_bridge_state_row_from_db(self, row: dict[str, Any]) -> NotificationBridgeStateRow:
        return NotificationBridgeStateRow(
            consumer_name=str(row.get("consumer_name") or ""),
            last_event_id=int(row.get("last_event_id") or 0),
            lease_owner_id=row.get("lease_owner_id"),
            lease_expires_at=row.get("lease_expires_at"),
            updated_at=str(row.get("updated_at") or ""),
        )

    def get_notification_bridge_state(self, *, consumer_name: str) -> NotificationBridgeStateRow:
        row = self.backend.execute(
            "SELECT * FROM notification_bridge_state WHERE consumer_name = ?",
            (consumer_name,),
        ).first
        if row:
            return self._notification_bridge_state_row_from_db(row)
        return NotificationBridgeStateRow(
            consumer_name=consumer_name,
            last_event_id=0,
            lease_owner_id=None,
            lease_expires_at=None,
            updated_at=_utcnow_iso(),
        )

    def update_notification_bridge_state(
        self,
        *,
        consumer_name: str,
        last_event_id: int | None = None,
        lease_owner_id: str | None = None,
        lease_expires_at: str | None = None,
    ) -> NotificationBridgeStateRow:
        current = self.get_notification_bridge_state(consumer_name=consumer_name)
        next_last_event_id = current.last_event_id if last_event_id is None else int(last_event_id)
        next_lease_owner_id = current.lease_owner_id if lease_owner_id is None else lease_owner_id
        next_lease_expires_at = current.lease_expires_at if lease_expires_at is None else lease_expires_at
        now = _utcnow_iso()

        q = (
            "INSERT INTO notification_bridge_state (consumer_name, last_event_id, lease_owner_id, lease_expires_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(consumer_name) DO UPDATE SET "
            "last_event_id = excluded.last_event_id, "
            "lease_owner_id = excluded.lease_owner_id, "
            "lease_expires_at = excluded.lease_expires_at, "
            "updated_at = excluded.updated_at"
        )
        self.backend.execute(
            q,
            (
                consumer_name,
                next_last_event_id,
                next_lease_owner_id,
                next_lease_expires_at,
                now,
            ),
        )
        return self.get_notification_bridge_state(consumer_name=consumer_name)

    def try_claim_notification_bridge_lease(
        self,
        *,
        consumer_name: str,
        lease_owner_id: str,
        lease_expires_at: str,
        now_iso: str | None = None,
    ) -> bool:
        now = now_iso or _utcnow_iso()
        self.backend.execute(
            (
                "INSERT INTO notification_bridge_state (consumer_name, last_event_id, lease_owner_id, lease_expires_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(consumer_name) DO NOTHING"
            ),
            (consumer_name, 0, None, None, now),
        )
        q = (
            "UPDATE notification_bridge_state "
            "SET lease_owner_id = ?, lease_expires_at = ?, updated_at = ? "
            "WHERE consumer_name = ? AND ("
            "lease_owner_id IS NULL OR lease_owner_id = ? OR lease_expires_at IS NULL OR lease_expires_at <= ?"
            ")"
        )
        res = self.backend.execute(
            q,
            (
                lease_owner_id,
                lease_expires_at,
                now,
                consumer_name,
                lease_owner_id,
                now,
            ),
        )
        return bool(res.rowcount and res.rowcount > 0)

    def release_notification_bridge_lease(self, *, consumer_name: str, lease_owner_id: str) -> bool:
        q = (
            "UPDATE notification_bridge_state "
            "SET lease_owner_id = NULL, lease_expires_at = NULL, updated_at = ? "
            "WHERE consumer_name = ? AND lease_owner_id = ?"
        )
        res = self.backend.execute(q, (_utcnow_iso(), consumer_name, lease_owner_id))
        return bool(res.rowcount and res.rowcount > 0)

    def purge_expired_outputs(self) -> int:
        """Hard delete expired/retained outputs. Returns number of rows removed."""
        now = _utcnow_iso()
        if self.backend.backend_type == BackendType.POSTGRESQL:
            r1 = self.backend.execute(
                "DELETE FROM outputs WHERE user_id = ? AND retention_until IS NOT NULL AND retention_until::timestamptz <= ?",
                (self.user_id, now),
            )
            try:
                r2 = self.backend.execute(
                    "DELETE FROM outputs WHERE user_id = ? AND deleted = 1 AND deleted_at IS NOT NULL AND (NOW() - deleted_at::timestamptz) >= INTERVAL '30 days'",
                    (self.user_id,),
                )
                return int((r1.rowcount or 0) + (r2.rowcount or 0))
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS:
                return int(r1.rowcount or 0)
        # Hard delete those with retention_until past
        r1 = self.backend.execute(
            "DELETE FROM outputs WHERE user_id = ? AND retention_until IS NOT NULL AND retention_until <= ?",
            (self.user_id, now),
        )
        # Soft-deleted older than 30 days
        try:
            r2 = self.backend.execute(
                "DELETE FROM outputs WHERE user_id = ? AND deleted = 1 AND deleted_at IS NOT NULL AND julianday(?) - julianday(deleted_at) >= 30",
                (self.user_id, now),
            )
            return int((r1.rowcount or 0) + (r2.rowcount or 0))
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS:
            return int(r1.rowcount or 0)

    # ------------------------
    # File artifacts API
    # ------------------------
    @dataclass
    class FileArtifactRow:
        id: int
        user_id: str
        file_type: str
        title: str
        structured_json: str
        validation_json: str
        export_status: str
        export_format: str | None
        export_storage_path: str | None
        export_bytes: int | None
        export_content_type: str | None
        export_job_id: str | None
        export_expires_at: str | None
        export_consumed_at: str | None
        metadata_json: str | None
        created_at: str
        updated_at: str
        retention_until: str | None = None

    def create_file_artifact(
        self,
        *,
        file_type: str,
        title: str,
        structured_json: str,
        validation_json: str,
        export_status: str = "none",
        export_format: str | None = None,
        export_storage_path: str | None = None,
        export_bytes: int | None = None,
        export_content_type: str | None = None,
        export_job_id: str | None = None,
        export_expires_at: str | None = None,
        export_consumed_at: str | None = None,
        metadata_json: str | None = None,
        retention_until: str | None = None,
    ) -> CollectionsDatabase.FileArtifactRow:
        """Create a file artifact record and return the stored row."""
        now = _utcnow_iso()
        resolved_storage_path = None
        if export_storage_path is not None:
            resolved_storage_path = self.resolve_temp_output_storage_path(export_storage_path)
        q = (
            "INSERT INTO file_artifacts (user_id, file_type, title, structured_json, validation_json, export_status, export_format, "
            "export_storage_path, export_bytes, export_content_type, export_job_id, export_expires_at, export_consumed_at, metadata_json, created_at, updated_at, "
            "deleted, deleted_at, retention_until) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?)"
        )
        params = (
            self.user_id,
            file_type,
            title,
            structured_json,
            validation_json,
            export_status,
            export_format,
            resolved_storage_path,
            export_bytes,
            export_content_type,
            export_job_id,
            export_expires_at,
            export_consumed_at,
            metadata_json,
            now,
            now,
            retention_until,
        )
        res = self._execute_insert(q, params)
        new_id = self._extract_lastrowid(res)
        if not new_id:
            raise DatabaseError("Failed to create file artifact")
        return self.get_file_artifact(new_id)

    def get_file_artifact(self, file_id: int, include_deleted: bool = False) -> CollectionsDatabase.FileArtifactRow:
        """Fetch a file artifact by id."""
        cond = "id = ? AND user_id = ?" + ("" if include_deleted else " AND deleted = 0")
        q = (
            "SELECT id, user_id, file_type, title, structured_json, validation_json, export_status, export_format, "  # nosec B608
            "export_storage_path, export_bytes, export_content_type, export_job_id, export_expires_at, export_consumed_at, metadata_json, created_at, updated_at, retention_until "
            f"FROM file_artifacts WHERE {cond}"
        )
        row = self.backend.execute(q, (file_id, self.user_id)).first
        if not row:
            raise KeyError("file_artifact_not_found")
        return CollectionsDatabase.FileArtifactRow(**row)

    def update_file_artifact_export(
        self,
        file_id: int,
        *,
        export_status: str,
        export_format: str | None = None,
        export_storage_path: str | None = None,
        export_bytes: int | None = None,
        export_content_type: str | None = None,
        export_job_id: str | None = None,
        export_expires_at: str | None = None,
        export_consumed_at: str | None = None,
    ) -> CollectionsDatabase.FileArtifactRow:
        """Update export fields for a file artifact."""
        updated_at = _utcnow_iso()
        resolved_storage_path = None
        if export_storage_path is not None:
            resolved_storage_path = self.resolve_temp_output_storage_path(export_storage_path)
        q = (
            "UPDATE file_artifacts SET export_status = ?, export_format = ?, export_storage_path = ?, export_bytes = ?, "
            "export_content_type = ?, export_job_id = ?, export_expires_at = ?, export_consumed_at = ?, updated_at = ? "
            "WHERE id = ? AND user_id = ? AND deleted = 0"
        )
        params = (
            export_status,
            export_format,
            resolved_storage_path,
            export_bytes,
            export_content_type,
            export_job_id,
            export_expires_at,
            export_consumed_at,
            updated_at,
            file_id,
            self.user_id,
        )
        res = self.backend.execute(q, params)
        if res.rowcount <= 0:
            raise KeyError("file_artifact_not_found")
        return self.get_file_artifact(file_id)

    def consume_file_artifact_export(self, file_id: int, *, consumed_at: str) -> bool:
        """Mark a ready export as consumed (one-time download guard)."""
        updated_at = _utcnow_iso()
        q = (
            "UPDATE file_artifacts SET export_consumed_at = ?, updated_at = ? "
            "WHERE id = ? AND user_id = ? AND deleted = 0 "
            "AND export_status = 'ready' AND export_storage_path IS NOT NULL "
            "AND export_consumed_at IS NULL"
        )
        params = (
            consumed_at,
            updated_at,
            file_id,
            self.user_id,
        )
        res = self.backend.execute(q, params)
        return res.rowcount > 0

    def delete_file_artifact(self, file_id: int, *, hard: bool = False) -> bool:
        """Delete a file artifact (soft delete by default)."""
        if hard:
            q = "DELETE FROM file_artifacts WHERE id = ? AND user_id = ?"
            res = self.backend.execute(q, (file_id, self.user_id))
            return res.rowcount > 0
        q = "UPDATE file_artifacts SET deleted = 1, deleted_at = ? WHERE id = ? AND user_id = ? AND deleted = 0"
        res = self.backend.execute(q, (_utcnow_iso(), file_id, self.user_id))
        return res.rowcount > 0

    def list_file_artifacts_for_purge(
        self,
        *,
        now_iso: str,
        soft_deleted_grace_days: int,
        include_retention: bool,
    ) -> dict[int, str | None]:
        """List artifact ids and paths eligible for retention or soft-delete purges."""
        paths: dict[int, str | None] = {}
        if include_retention:
            q = (
                "SELECT id, export_storage_path FROM file_artifacts "
                "WHERE user_id = ? AND retention_until IS NOT NULL AND retention_until <= ?"
            )
            try:
                cur = self.backend.execute(q, (self.user_id, now_iso))
                for row in cur.rows:
                    rid = int(row["id"]) if isinstance(row, dict) else int(row[0])
                    paths[rid] = row["export_storage_path"] if isinstance(row, dict) else row[1]
            except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"file_artifacts.purge: retention scan failed: {exc}")
        try:
            now_dt = datetime.fromisoformat(str(now_iso).replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            logger.warning("file_artifacts.purge: invalid now_iso '{}': {}", now_iso, exc)
            now_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        cutoff = (now_dt - timedelta(days=soft_deleted_grace_days)).isoformat()
        q2 = (
            "SELECT id, export_storage_path FROM file_artifacts "
            "WHERE user_id = ? AND deleted = 1 AND deleted_at IS NOT NULL AND deleted_at <= ?"
        )
        try:
            cur2 = self.backend.execute(q2, (self.user_id, cutoff))
            for row in cur2.rows:
                rid = int(row["id"]) if isinstance(row, dict) else int(row[0])
                paths[rid] = row["export_storage_path"] if isinstance(row, dict) else row[1]
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"file_artifacts.purge: soft-deleted scan failed: {exc}")
        return paths

    def list_file_artifacts_expired_exports(self, *, now_iso: str) -> list[dict[str, Any]]:
        """List ready exports that have expired for cleanup."""
        rows: list[dict[str, Any]] = []
        q = (
            "SELECT id, export_storage_path, export_format, export_bytes, export_content_type, export_job_id, export_expires_at "
            "FROM file_artifacts WHERE user_id = ? AND export_status = 'ready' AND export_storage_path IS NOT NULL "
            "AND export_expires_at IS NOT NULL AND export_expires_at <= ?"
        )
        try:
            cur = self.backend.execute(q, (self.user_id, now_iso))
        except _COLLECTIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"file_artifacts.export_gc: expired export scan failed: {exc}")
            return rows
        for row in cur.rows:
            if isinstance(row, dict):
                rows.append(row)
                continue
            rows.append(
                {
                    "id": row[0],
                    "export_storage_path": row[1],
                    "export_format": row[2],
                    "export_bytes": row[3],
                    "export_content_type": row[4],
                    "export_job_id": row[5],
                    "export_expires_at": row[6],
                }
            )
        return rows

    def delete_file_artifacts_by_ids(self, ids: list[int]) -> int:
        """Delete file artifacts by id list for the current user."""
        if not ids:
            return 0
        placeholders = ",".join(["?"] * len(ids))
        q = f"DELETE FROM file_artifacts WHERE user_id = ? AND id IN ({placeholders})"  # nosec B608
        res = self.backend.execute(q, tuple([self.user_id] + list(ids)))
        return int(res.rowcount or 0)

"""
Watchlists_DB - Persistence for Watchlists (sources, groups/tags, jobs, runs).

Tables (per-user, colocated with Media DB):
- sources(id, user_id, name, url, source_type, active, settings_json,
          last_scraped_at, etag, last_modified, status, created_at, updated_at)
- groups(id, user_id, name, description, parent_group_id)
- source_groups(source_id, group_id)
- tags(id, user_id, name) with UNIQUE(user_id, name)
- source_tags(source_id, tag_id)
- scrape_jobs(id, user_id, name, description, scope_json, schedule_expr,
              active, max_concurrency, per_host_delay_ms, retry_policy_json,
              output_prefs_json, schedule_timezone, created_at, updated_at,
              last_run_at, next_run_at)
- scrape_runs(id, job_id, status, started_at, finished_at, stats_json,
              error_msg, log_path)
- scrape_run_items(run_id, media_id, source_id)
- scraped_items(id, run_id, job_id, source_id, media_id, media_uuid, url, title,
                summary, content, published_at, tags_json, status, reviewed,
                queued_for_briefing, created_at)
- watchlist_ia_experiment_events(id, user_id, variant, session_id, previous_tab,
                current_tab, transitions, visited_tabs_json, first_seen_at,
                last_seen_at, elapsed_ms, reached_target, created_at)
- watchlist_onboarding_events(id, user_id, session_id, event_type, event_at,
                details_json, created_at)

Notes:
- Backed by DatabaseBackendFactory; default to per-user SQLite Media DB path.
- Provides minimal CRUD required by the API layer; scraping is implemented elsewhere.
- Watchlists outputs are stored in Collections outputs; legacy watchlist_outputs is retired.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import sqlite3
import threading
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.DB_Management.content_backend import (
    backend_target_key,
    get_content_backend,
    load_content_db_settings,
)

from .backends.base import BackendType, DatabaseBackend, DatabaseConfig, DatabaseError as _DatabaseError
from .backends.factory import DatabaseBackendFactory
from .backends.query_utils import prepare_backend_statement
from .db_path_utils import DatabasePaths

_WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    sqlite3.IntegrityError,
    json.JSONDecodeError,
)


def _utcnow_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


@dataclass
class SourceRow:
    id: int
    user_id: str
    name: str
    url: str
    source_type: str
    active: int
    settings_json: str | None
    last_scraped_at: str | None
    etag: str | None
    last_modified: str | None
    defer_until: str | None
    status: str | None
    consec_not_modified: int | None
    consec_errors: int | None
    created_at: str
    updated_at: str
    tags: list[str]
    was_created: bool = False


@dataclass
class GroupRow:
    id: int
    user_id: str
    name: str
    description: str | None
    parent_group_id: int | None


@dataclass
class TagRow:
    id: int
    user_id: str
    name: str


@dataclass
class JobRow:
    id: int
    user_id: str
    name: str
    description: str | None
    scope_json: str | None
    schedule_expr: str | None
    schedule_timezone: str | None
    active: int
    max_concurrency: int | None
    per_host_delay_ms: int | None
    retry_policy_json: str | None
    output_prefs_json: str | None
    job_filters_json: str | None
    created_at: str
    updated_at: str
    last_run_at: str | None
    next_run_at: str | None
    wf_schedule_id: str | None = None


@dataclass
class RunRow:
    id: int
    job_id: int
    status: str
    started_at: str | None
    finished_at: str | None
    stats_json: str | None
    error_msg: str | None
    log_path: str | None


@dataclass
class ScrapedItemRow:
    id: int
    run_id: int
    job_id: int
    source_id: int
    media_id: int | None
    media_uuid: str | None
    url: str | None
    title: str | None
    summary: str | None
    content: str | None
    published_at: str | None
    tags_json: str | None
    status: str
    reviewed: int
    queued_for_briefing: int
    created_at: str

    def tags(self) -> list[str]:
        if not self.tags_json:
            return []
        try:
            data = json.loads(self.tags_json)
            if isinstance(data, list):
                return [str(t) for t in data if isinstance(t, str)]
        except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
            return []
        return []


@dataclass
class WebSubRow:
    id: int
    user_id: str
    source_id: int
    hub_url: str
    topic_url: str
    callback_token: str
    secret: str
    state: str
    lease_seconds: int | None
    verified_at: str | None
    expires_at: str | None
    last_push_at: str | None
    created_at: str
    updated_at: str


@dataclass
class WatchlistIaExperimentEventRow:
    id: int
    user_id: str
    variant: str
    session_id: str
    previous_tab: str | None
    current_tab: str
    transitions: int
    visited_tabs_json: str
    first_seen_at: str | None
    last_seen_at: str | None
    elapsed_ms: int | None
    reached_target: int
    created_at: str


@dataclass
class WatchlistOnboardingEventRow:
    id: int
    user_id: str
    session_id: str
    event_type: str
    event_at: str
    details_json: str
    created_at: str


def _pin_watchlists_public_operation(method):
    @functools.wraps(method)
    def wrapper(self: "WatchlistsDatabase", *args: Any, **kwargs: Any):
        with self._operation_backend_pin():
            return method(self, *args, **kwargs)

    return wrapper


def _decorate_watchlists_public_operations(cls: type["WatchlistsDatabase"]) -> type["WatchlistsDatabase"]:
    for name, attribute in list(vars(cls).items()):
        if name.startswith("_") or name in {"ensure_schema", "transaction"}:
            continue
        if isinstance(attribute, (classmethod, staticmethod, property)):
            continue
        if not callable(attribute):
            continue
        setattr(cls, name, _pin_watchlists_public_operation(attribute))
    return cls


@_decorate_watchlists_public_operations
class WatchlistsDatabase:
    # Keep track of schema initialization per DB path to avoid redundant ALTER checks/noise
    _schema_init_keys: set[str] = set()

    def __init__(self, user_id: int | str, backend: DatabaseBackend | None = None):
        self.user_id = str(user_id)
        self._local = threading.local()
        self._backend_refresh_suspended = False
        db_key: str | None = None
        self._backend: DatabaseBackend
        self._uses_shared_content_backend = False
        if backend is None:
            backend, db_key, uses_shared_backend = self._resolve_backend()
            self._uses_shared_content_backend = uses_shared_backend
        else:
            # Fallback key when external backend is supplied (best-effort de-dupe)
            db_key = f"backend:{id(backend)}"
        self._backend = backend
        # De-duplicate schema ensures across startup/requests
        if db_key:
            self._ensure_schema_for_key(self._backend, db_key)
        else:
            # If we couldn't derive a key, fall back to ensuring schema once here
            self.ensure_schema()

    @classmethod
    def for_user(cls, user_id: int | str) -> WatchlistsDatabase:
        return cls(user_id=user_id)

    @property
    def backend(self) -> DatabaseBackend:
        pinned_backend = getattr(self._local, "backend_pin", None)
        if pinned_backend is not None:
            return pinned_backend
        return self._refresh_backend_if_needed()

    def _refresh_backend_if_needed(self) -> DatabaseBackend:
        if not self._uses_shared_content_backend:
            return self._backend
        if self._backend_refresh_suspended:
            return self._backend

        current_key = self._backend_target_key(self._backend)
        refreshed_backend, refreshed_key, uses_shared_backend = self._resolve_backend()
        self._uses_shared_content_backend = uses_shared_backend
        self._backend = refreshed_backend
        if (
            uses_shared_backend
            and refreshed_backend.backend_type == BackendType.POSTGRESQL
            and refreshed_key != current_key
        ):
            self._ensure_schema_for_key(refreshed_backend, refreshed_key)
        return self._backend

    def _backend_target_key(self, backend: DatabaseBackend | None) -> str | None:
        return backend_target_key(backend)

    def _ensure_schema_for_key(self, backend: DatabaseBackend, db_key: str | None) -> None:
        if not db_key or db_key in WatchlistsDatabase._schema_init_keys:
            return
        previous_suspend = self._backend_refresh_suspended
        self._backend_refresh_suspended = True
        try:
            self._backend = backend
            self.ensure_schema()
        finally:
            self._backend_refresh_suspended = previous_suspend
        WatchlistsDatabase._schema_init_keys.add(db_key)

    @contextlib.contextmanager
    def _operation_backend_pin(self) -> Generator[DatabaseBackend, None, None]:
        previous_backend = getattr(self._local, "backend_pin", None)
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
        previous_backend = getattr(self._local, "backend_pin", None)
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

    def _resolve_backend(self) -> tuple[DatabaseBackend, str | None, bool]:
        backend_mode_env = (os.getenv("TLDW_CONTENT_DB_BACKEND") or "").strip().lower()
        if backend_mode_env in {"postgres", "postgresql"}:
            parser = load_comprehensive_config()
            resolved = get_content_backend(parser)
            if resolved is None:
                raise RuntimeError("PostgreSQL content backend requested but not initialized")
            return (
                resolved,
                self._backend_target_key(resolved),
                True,
            )

        try:
            parser = load_comprehensive_config()
        except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
            parser = None

        if parser is not None:
            try:
                content_settings = load_content_db_settings(parser)
                if content_settings.backend_type == BackendType.POSTGRESQL:
                    resolved = get_content_backend(parser)
                    if resolved is None:
                        raise RuntimeError("PostgreSQL content backend requested but not initialized")
                    return (
                        resolved,
                        self._backend_target_key(resolved),
                        True,
                    )
            except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                pass

        db_path = str(DatabasePaths.get_media_db_path(int(self.user_id)))
        cfg = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=db_path)
        return DatabaseBackendFactory.create_backend(cfg), f"sqlite:{db_path}", False

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
        except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
            return None
        return None

    def ensure_schema(self) -> None:
        if self.backend.backend_type == BackendType.POSTGRESQL:
            ddl = """
            CREATE TABLE IF NOT EXISTS sources (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                source_type TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                settings_json TEXT,
                last_scraped_at TEXT,
                etag TEXT,
                last_modified TEXT,
                defer_until TEXT,
                status TEXT,
                consec_not_modified INTEGER NOT NULL DEFAULT 0,
                consec_errors INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sources_user ON sources(user_id);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_sources_user_url ON sources(user_id, url);

            CREATE TABLE IF NOT EXISTS groups (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                parent_group_id BIGINT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ux_groups_user_name ON groups(user_id, name);

            CREATE TABLE IF NOT EXISTS source_groups (
                source_id BIGINT NOT NULL,
                group_id BIGINT NOT NULL,
                UNIQUE (source_id, group_id)
            );

            CREATE TABLE IF NOT EXISTS tags (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                UNIQUE (user_id, name)
            );
            CREATE TABLE IF NOT EXISTS source_tags (
                source_id BIGINT NOT NULL,
                tag_id BIGINT NOT NULL,
                UNIQUE (source_id, tag_id)
            );

            CREATE TABLE IF NOT EXISTS scrape_jobs (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                scope_json TEXT,
                schedule_expr TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                max_concurrency INTEGER,
                per_host_delay_ms INTEGER,
                retry_policy_json TEXT,
                output_prefs_json TEXT,
                job_filters_json TEXT,
                schedule_timezone TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_run_at TEXT,
                next_run_at TEXT,
                wf_schedule_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_user ON scrape_jobs(user_id);

            CREATE TABLE IF NOT EXISTS scrape_runs (
                id BIGSERIAL PRIMARY KEY,
                job_id BIGINT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                stats_json TEXT,
                error_msg TEXT,
                log_path TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_runs_job ON scrape_runs(job_id);

            CREATE TABLE IF NOT EXISTS scrape_run_items (
                run_id BIGINT NOT NULL,
                media_id BIGINT NOT NULL,
                source_id BIGINT,
                UNIQUE (run_id, media_id)
            );

            CREATE TABLE IF NOT EXISTS source_seen_items (
                source_id BIGINT NOT NULL,
                item_key TEXT NOT NULL,
                etag TEXT,
                last_modified TEXT,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                UNIQUE (source_id, item_key)
            );

            CREATE TABLE IF NOT EXISTS scraped_items (
                id BIGSERIAL PRIMARY KEY,
                run_id BIGINT NOT NULL,
                job_id BIGINT NOT NULL,
                source_id BIGINT NOT NULL,
                media_id BIGINT,
                media_uuid TEXT,
                url TEXT,
                title TEXT,
                summary TEXT,
                content TEXT,
                published_at TEXT,
                tags_json TEXT,
                status TEXT NOT NULL,
                reviewed INTEGER NOT NULL DEFAULT 0,
                queued_for_briefing INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_scraped_items_run ON scraped_items(run_id);
            CREATE INDEX IF NOT EXISTS idx_scraped_items_job ON scraped_items(job_id);
            CREATE INDEX IF NOT EXISTS idx_scraped_items_source ON scraped_items(source_id);
            CREATE INDEX IF NOT EXISTS idx_scraped_items_status ON scraped_items(status);
            CREATE INDEX IF NOT EXISTS idx_scraped_items_reviewed ON scraped_items(reviewed);
            CREATE INDEX IF NOT EXISTS idx_scraped_items_created ON scraped_items(created_at);

            CREATE TABLE IF NOT EXISTS watchlist_clusters (
                job_id BIGINT NOT NULL,
                cluster_id BIGINT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (job_id, cluster_id)
            );
            CREATE INDEX IF NOT EXISTS idx_watchlist_clusters_job ON watchlist_clusters(job_id);
            CREATE INDEX IF NOT EXISTS idx_watchlist_clusters_cluster ON watchlist_clusters(cluster_id);

            CREATE TABLE IF NOT EXISTS watchlist_ia_experiment_events (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                variant TEXT NOT NULL,
                session_id TEXT NOT NULL,
                previous_tab TEXT,
                current_tab TEXT NOT NULL,
                transitions INTEGER NOT NULL DEFAULT 0,
                visited_tabs_json TEXT NOT NULL,
                first_seen_at TEXT,
                last_seen_at TEXT,
                elapsed_ms INTEGER,
                reached_target INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_watchlist_ia_events_user_variant_created
                ON watchlist_ia_experiment_events(user_id, variant, created_at);
            CREATE INDEX IF NOT EXISTS idx_watchlist_ia_events_user_session_created
                ON watchlist_ia_experiment_events(user_id, session_id, created_at);

            CREATE TABLE IF NOT EXISTS watchlist_onboarding_events (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_at TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_watchlist_onboarding_user_event_created
                ON watchlist_onboarding_events(user_id, event_type, created_at);
            CREATE INDEX IF NOT EXISTS idx_watchlist_onboarding_user_session_eventat
                ON watchlist_onboarding_events(user_id, session_id, event_at);

            CREATE TABLE IF NOT EXISTS feed_websub_subscriptions (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                source_id BIGINT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                hub_url TEXT NOT NULL,
                topic_url TEXT NOT NULL,
                callback_token TEXT NOT NULL UNIQUE,
                secret TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending',
                lease_seconds INTEGER,
                verified_at TEXT,
                expires_at TEXT,
                last_push_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_websub_source ON feed_websub_subscriptions(source_id);
            CREATE INDEX IF NOT EXISTS idx_websub_state ON feed_websub_subscriptions(state);
            CREATE INDEX IF NOT EXISTS idx_websub_expires ON feed_websub_subscriptions(expires_at);

            CREATE TABLE IF NOT EXISTS deleted_sources (
                user_id TEXT NOT NULL,
                source_id BIGINT NOT NULL,
                payload_json TEXT NOT NULL,
                deleted_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                PRIMARY KEY (user_id, source_id)
            );
            CREATE INDEX IF NOT EXISTS idx_deleted_sources_expires ON deleted_sources(expires_at);

            CREATE TABLE IF NOT EXISTS deleted_jobs (
                user_id TEXT NOT NULL,
                job_id BIGINT NOT NULL,
                payload_json TEXT NOT NULL,
                deleted_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                PRIMARY KEY (user_id, job_id)
            );
            CREATE INDEX IF NOT EXISTS idx_deleted_jobs_expires ON deleted_jobs(expires_at);
            """
        else:
            ddl = """
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                source_type TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                settings_json TEXT,
                last_scraped_at TEXT,
                etag TEXT,
                last_modified TEXT,
                defer_until TEXT,
                status TEXT,
                consec_not_modified INTEGER NOT NULL DEFAULT 0,
                consec_errors INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sources_user ON sources(user_id);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_sources_user_url ON sources(user_id, url);

            CREATE TABLE IF NOT EXISTS groups (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                parent_group_id INTEGER
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ux_groups_user_name ON groups(user_id, name);

            CREATE TABLE IF NOT EXISTS source_groups (
                source_id INTEGER NOT NULL,
                group_id INTEGER NOT NULL,
                UNIQUE (source_id, group_id)
            );

            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                UNIQUE (user_id, name)
            );
            CREATE TABLE IF NOT EXISTS source_tags (
                source_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                UNIQUE (source_id, tag_id)
            );

            CREATE TABLE IF NOT EXISTS scrape_jobs (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                scope_json TEXT,
                schedule_expr TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                max_concurrency INTEGER,
                per_host_delay_ms INTEGER,
                retry_policy_json TEXT,
                output_prefs_json TEXT,
                job_filters_json TEXT,
                schedule_timezone TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_run_at TEXT,
                next_run_at TEXT,
                wf_schedule_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_user ON scrape_jobs(user_id);

            CREATE TABLE IF NOT EXISTS scrape_runs (
                id INTEGER PRIMARY KEY,
                job_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                stats_json TEXT,
                error_msg TEXT,
                log_path TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_runs_job ON scrape_runs(job_id);

            CREATE TABLE IF NOT EXISTS scrape_run_items (
                run_id INTEGER NOT NULL,
                media_id INTEGER NOT NULL,
                source_id INTEGER,
                UNIQUE (run_id, media_id)
            );

            -- Track per-source seen RSS/Atom items for deduplication
            CREATE TABLE IF NOT EXISTS source_seen_items (
                source_id INTEGER NOT NULL,
                item_key TEXT NOT NULL,
                etag TEXT,
                last_modified TEXT,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                UNIQUE (source_id, item_key)
            );

            CREATE TABLE IF NOT EXISTS scraped_items (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL,
                job_id INTEGER NOT NULL,
                source_id INTEGER NOT NULL,
                media_id INTEGER,
                media_uuid TEXT,
                url TEXT,
                title TEXT,
                summary TEXT,
                content TEXT,
                published_at TEXT,
                tags_json TEXT,
                status TEXT NOT NULL,
                reviewed INTEGER NOT NULL DEFAULT 0,
                queued_for_briefing INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_scraped_items_run ON scraped_items(run_id);
            CREATE INDEX IF NOT EXISTS idx_scraped_items_job ON scraped_items(job_id);
            CREATE INDEX IF NOT EXISTS idx_scraped_items_source ON scraped_items(source_id);
            CREATE INDEX IF NOT EXISTS idx_scraped_items_status ON scraped_items(status);
            CREATE INDEX IF NOT EXISTS idx_scraped_items_reviewed ON scraped_items(reviewed);
            CREATE INDEX IF NOT EXISTS idx_scraped_items_created ON scraped_items(created_at);

            CREATE TABLE IF NOT EXISTS watchlist_clusters (
                job_id INTEGER NOT NULL,
                cluster_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (job_id, cluster_id)
            );
            CREATE INDEX IF NOT EXISTS idx_watchlist_clusters_job ON watchlist_clusters(job_id);
            CREATE INDEX IF NOT EXISTS idx_watchlist_clusters_cluster ON watchlist_clusters(cluster_id);

            CREATE TABLE IF NOT EXISTS watchlist_ia_experiment_events (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                variant TEXT NOT NULL,
                session_id TEXT NOT NULL,
                previous_tab TEXT,
                current_tab TEXT NOT NULL,
                transitions INTEGER NOT NULL DEFAULT 0,
                visited_tabs_json TEXT NOT NULL,
                first_seen_at TEXT,
                last_seen_at TEXT,
                elapsed_ms INTEGER,
                reached_target INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_watchlist_ia_events_user_variant_created
                ON watchlist_ia_experiment_events(user_id, variant, created_at);
            CREATE INDEX IF NOT EXISTS idx_watchlist_ia_events_user_session_created
                ON watchlist_ia_experiment_events(user_id, session_id, created_at);

            CREATE TABLE IF NOT EXISTS watchlist_onboarding_events (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_at TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_watchlist_onboarding_user_event_created
                ON watchlist_onboarding_events(user_id, event_type, created_at);
            CREATE INDEX IF NOT EXISTS idx_watchlist_onboarding_user_session_eventat
                ON watchlist_onboarding_events(user_id, session_id, event_at);

            CREATE TABLE IF NOT EXISTS feed_websub_subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                hub_url TEXT NOT NULL,
                topic_url TEXT NOT NULL,
                callback_token TEXT NOT NULL UNIQUE,
                secret TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending',
                lease_seconds INTEGER,
                verified_at TEXT,
                expires_at TEXT,
                last_push_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_websub_source ON feed_websub_subscriptions(source_id);
            CREATE INDEX IF NOT EXISTS idx_websub_state ON feed_websub_subscriptions(state);
            CREATE INDEX IF NOT EXISTS idx_websub_expires ON feed_websub_subscriptions(expires_at);

            CREATE TABLE IF NOT EXISTS deleted_sources (
                user_id TEXT NOT NULL,
                source_id INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                deleted_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                PRIMARY KEY (user_id, source_id)
            );
            CREATE INDEX IF NOT EXISTS idx_deleted_sources_expires ON deleted_sources(expires_at);

            CREATE TABLE IF NOT EXISTS deleted_jobs (
                user_id TEXT NOT NULL,
                job_id INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                deleted_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                PRIMARY KEY (user_id, job_id)
            );
            CREATE INDEX IF NOT EXISTS idx_deleted_jobs_expires ON deleted_jobs(expires_at);
            """
        self.backend.create_tables(ddl)
        # Backfill columns in case tables existed (guarded to avoid noisy duplicate-column errors)
        def _col_exists(table: str, col: str) -> bool:
            try:
                info = self.backend.get_table_info(table)
                names = {str(c.get("name")) for c in info if c.get("name") is not None}
                return col in names
            except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                return False

        if not _col_exists("scrape_jobs", "wf_schedule_id"):
            with contextlib.suppress(_WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS):
                self.backend.execute("ALTER TABLE scrape_jobs ADD COLUMN wf_schedule_id TEXT", ())
        if not _col_exists("scrape_jobs", "job_filters_json"):
            with contextlib.suppress(_WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS):
                self.backend.execute("ALTER TABLE scrape_jobs ADD COLUMN job_filters_json TEXT", ())
        if not _col_exists("sources", "defer_until"):
            with contextlib.suppress(_WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS):
                self.backend.execute("ALTER TABLE sources ADD COLUMN defer_until TEXT", ())
        if not _col_exists("sources", "consec_not_modified"):
            with contextlib.suppress(_WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS):
                self.backend.execute("ALTER TABLE sources ADD COLUMN consec_not_modified INTEGER DEFAULT 0", ())
        if not _col_exists("sources", "consec_errors"):
            with contextlib.suppress(_WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS):
                self.backend.execute("ALTER TABLE sources ADD COLUMN consec_errors INTEGER DEFAULT 0", ())
        if not _col_exists("scrape_run_items", "source_id"):
            with contextlib.suppress(_WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS):
                self.backend.execute("ALTER TABLE scrape_run_items ADD COLUMN source_id INTEGER", ())
        if not _col_exists("scraped_items", "content"):
            with contextlib.suppress(_WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS):
                self.backend.execute("ALTER TABLE scraped_items ADD COLUMN content TEXT", ())
        if not _col_exists("scraped_items", "queued_for_briefing"):
            with contextlib.suppress(_WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS):
                self.backend.execute(
                    "ALTER TABLE scraped_items ADD COLUMN queued_for_briefing INTEGER NOT NULL DEFAULT 0",
                    (),
                )
    # ------------------------
    # Tags helpers
    # ------------------------
    def _normalize_tag(self, name: str) -> str:
        return name.strip().lower()

    def _normalize_tag_names(self, names: Iterable[str] | None) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in names or []:
            if not raw:
                continue
            nm = self._normalize_tag(str(raw))
            if not nm or nm in seen:
                continue
            seen.add(nm)
            out.append(nm)
        return out

    def _restore_expires_at(self, undo_window_seconds: int) -> str:
        seconds = max(1, int(undo_window_seconds))
        return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()

    def _is_restore_expired(self, expires_at: str | None) -> bool:
        if not expires_at:
            return True
        try:
            parsed = datetime.fromisoformat(str(expires_at))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc) <= datetime.now(timezone.utc)
        except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
            return True

    def _purge_expired_deleted_records(self) -> None:
        now = _utcnow_iso()
        with contextlib.suppress(_WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS):
            self.backend.execute(
                "DELETE FROM deleted_sources WHERE user_id = ? AND expires_at <= ?",
                (self.user_id, now),
            )
        with contextlib.suppress(_WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS):
            self.backend.execute(
                "DELETE FROM deleted_jobs WHERE user_id = ? AND expires_at <= ?",
                (self.user_id, now),
            )

    def _ensure_tag_ids_with_connection(self, names: list[str], connection: Any) -> list[int]:
        normalized = self._normalize_tag_names(names)
        ids: list[int] = []
        for nm in normalized:
            row = self.backend.execute(
                "SELECT id FROM tags WHERE user_id = ? AND name = ?",
                (self.user_id, nm),
                connection=connection,
            ).first
            if row and row.get("id") is not None:
                ids.append(int(row.get("id")))
                continue

            insert_exc: Exception | None = None
            try:
                self.backend.execute(
                    """
                    INSERT INTO tags (user_id, name) VALUES (?, ?)
                    ON CONFLICT(user_id, name) DO NOTHING
                    """,
                    (self.user_id, nm),
                    connection=connection,
                )
            except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS as exc:
                insert_exc = exc

            row = self.backend.execute(
                "SELECT id FROM tags WHERE user_id = ? AND name = ?",
                (self.user_id, nm),
                connection=connection,
            ).first
            if row and row.get("id") is not None:
                ids.append(int(row.get("id")))
                continue
            if insert_exc:
                raise insert_exc
            raise RuntimeError("failed_to_ensure_tag_id")
        return ids

    def ensure_tag_ids(self, names: list[str]) -> list[int]:
        normed = [self._normalize_tag(n) for n in names if n and n.strip()]
        ids: list[int] = []
        select_sql = "SELECT id FROM tags WHERE user_id = ? AND name = ?"
        insert_sql = "INSERT INTO tags (user_id, name) VALUES (?, ?)"
        for nm in normed:
            params = (self.user_id, nm)
            row = self.backend.execute(select_sql, params).first
            if row:
                ids.append(int(row.get("id")))
                continue
            insert_exc: Exception | None = None
            tag_id: int | None = None
            try:
                res = self._execute_insert(insert_sql, params)
                tag_id = self._extract_lastrowid(res)
            except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS as exc:
                insert_exc = exc
            if tag_id is None:
                row = self.backend.execute(select_sql, params).first
                if row:
                    tag_id = int(row.get("id"))
            if tag_id is None:
                if insert_exc:
                    raise insert_exc
                raise RuntimeError("failed_to_ensure_tag_id")
            ids.append(tag_id)
        return ids

    def _lookup_tag_ids(self, names: Iterable[str]) -> dict[str, int]:
        normed: list[str] = []
        seen: set[str] = set()
        for raw in names or []:
            if not raw:
                continue
            nm = self._normalize_tag(str(raw))
            if not nm or nm in seen:
                continue
            seen.add(nm)
            normed.append(nm)
        if not normed:
            return {}

        placeholders = ",".join("?" for _ in normed)
        params = [self.user_id, *normed]
        rows = self.backend.execute(
            f"SELECT name, id FROM tags WHERE user_id = ? AND name IN ({placeholders})",  # nosec B608
            tuple(params),
        ).rows
        out: dict[str, int] = {}
        for row in rows:
            name_val = row.get("name")
            tag_id = row.get("id")
            if name_val is None or tag_id is None:
                continue
            try:
                out[str(name_val)] = int(tag_id)
            except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                continue
        return out

    def list_tags(self, q: str | None, limit: int, offset: int) -> tuple[list[TagRow], int]:
        where = ["user_id = ?"]
        params: list[Any] = [self.user_id]
        if q:
            where.append("name LIKE ?")
            params.append(f"%{q}%")
        where_sql = " AND ".join(where)
        total = int(self.backend.execute(f"SELECT COUNT(*) AS cnt FROM tags WHERE {where_sql}", tuple(params)).scalar or 0)  # nosec B608
        rows = self.backend.execute(
            f"SELECT id, user_id, name FROM tags WHERE {where_sql} ORDER BY name LIMIT ? OFFSET ?",  # nosec B608
            tuple(params + [limit, offset]),
        ).rows
        return [TagRow(**r) for r in rows], total

    # ------------------------
    # Sources
    # ------------------------
    def create_source(
        self,
        *,
        name: str,
        url: str,
        source_type: str,
        active: bool = True,
        settings_json: str | None = None,
        tags: list[str] | None = None,
        group_ids: list[int] | None = None,
    ) -> SourceRow:
        now = _utcnow_iso()
        # Try insert; if UNIQUE(user_id,url) violates, fetch existing id and proceed idempotently
        sid: int | None = None
        created_new = False
        try:
            res = self._execute_insert(
                "INSERT INTO sources (user_id, name, url, source_type, active, settings_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (self.user_id, name, url, source_type, 1 if active else 0, settings_json, now, now),
            )
            sid = self._extract_lastrowid(res)
            created_new = True
        except (*_WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS, _DatabaseError):
            # Look up existing source for idempotency
            try:
                row = self.backend.execute(
                    "SELECT id FROM sources WHERE user_id = ? AND url = ?",
                    (self.user_id, url),
                ).first
                if row and row.get("id") is not None:
                    sid = int(row.get("id"))
                else:
                    raise
            except (*_WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS, _DatabaseError):
                raise
        if sid is None:
                raise RuntimeError("failed_to_create_or_lookup_source")
        if tags:
            tag_ids = self.ensure_tag_ids(tags)
            for tid in tag_ids:
                try:
                    self.backend.execute(
                        "INSERT INTO source_tags (source_id, tag_id) VALUES (?, ?) ON CONFLICT(source_id, tag_id) DO NOTHING",
                        (sid, tid),
                    )
                except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                    # Fallback for engines without ON CONFLICT (should rarely trigger)
                    self.backend.execute(
                        "INSERT INTO source_tags (source_id, tag_id) SELECT ?, ? WHERE NOT EXISTS (SELECT 1 FROM source_tags WHERE source_id = ? AND tag_id = ?)",
                        (sid, tid, sid, tid),
                    )
        if group_ids:
            for gid in group_ids:
                try:
                    self.backend.execute(
                        "INSERT INTO source_groups (source_id, group_id) VALUES (?, ?) ON CONFLICT(source_id, group_id) DO NOTHING",
                        (sid, gid),
                    )
                except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                    self.backend.execute(
                        "INSERT INTO source_groups (source_id, group_id) SELECT ?, ? WHERE NOT EXISTS (SELECT 1 FROM source_groups WHERE source_id = ? AND group_id = ?)",
                        (sid, gid, sid, gid),
                    )
        row = self.get_source(sid)
        row.was_created = created_new
        return row

    def get_source_by_url(self, url: str) -> SourceRow | None:
        row = self.backend.execute(
            """
            SELECT id, user_id, name, url, source_type, active, settings_json,
                   last_scraped_at, etag, last_modified, status,
                   created_at, updated_at, defer_until, consec_not_modified, consec_errors
            FROM sources
            WHERE user_id = ? AND url = ?
            """,
            (self.user_id, url),
        ).first
        if not row:
            return None
        sid = int(row.get("id"))
        trows = self.backend.execute(
            "SELECT t.name FROM source_tags st JOIN tags t ON st.tag_id = t.id WHERE st.source_id = ?",
            (sid,),
        ).rows
        tags = [r.get("name") for r in trows if r.get("name")]
        return SourceRow(tags=tags, **row)  # type: ignore[arg-type]

    def get_source(self, source_id: int) -> SourceRow:
        row = self.backend.execute(
            "SELECT id, user_id, name, url, source_type, active, settings_json, last_scraped_at, etag, last_modified, defer_until, status, consec_not_modified, consec_errors, created_at, updated_at FROM sources WHERE id = ? AND user_id = ?",
            (source_id, self.user_id),
        ).first
        if not row:
            raise KeyError("source_not_found")
        tags_rows = self.backend.execute(
            "SELECT t.name FROM source_tags st JOIN tags t ON st.tag_id = t.id WHERE st.source_id = ?",
            (source_id,),
        ).rows
        tags = [r.get("name") for r in tags_rows if r.get("name")]
        return SourceRow(tags=tags, **row)  # type: ignore[arg-type]

    def list_sources(self, q: str | None, tag_names: list[str] | None, limit: int, offset: int, group_ids: list[int] | None = None) -> tuple[list[SourceRow], int]:
        where = ["user_id = ?"]
        params: list[Any] = [self.user_id]
        if q:
            where.append("(name LIKE ? OR url LIKE ?)")
            like = f"%{q}%"
            params.extend([like, like])
        # Group filter: OR semantics (source belongs to any of the given groups)
        if group_ids:
            placeholders = ",".join(["?"] * len(group_ids))
            where.append(
                f"EXISTS (SELECT 1 FROM source_groups sg WHERE sg.source_id = sources.id AND sg.group_id IN ({placeholders}))"  # nosec B608
            )
            params.extend(group_ids)
        # Tag filter: require all tags
        if tag_names:
            normalized_tags = []
            for tag in tag_names:
                if not tag:
                    continue
                nm = self._normalize_tag(str(tag))
                if nm and nm not in normalized_tags:
                    normalized_tags.append(nm)
            if normalized_tags:
                tag_map = self._lookup_tag_ids(normalized_tags)
                if len(tag_map) < len(normalized_tags):
                    return [], 0
                for nm in normalized_tags:
                    tid = tag_map[nm]
                    where.append(
                        "EXISTS (SELECT 1 FROM source_tags st WHERE st.source_id = sources.id AND st.tag_id = ?)"
                    )
                    params.append(tid)
        where_sql = " AND ".join(where)
        total = int(self.backend.execute(f"SELECT COUNT(*) AS cnt FROM sources WHERE {where_sql}", tuple(params)).scalar or 0)  # nosec B608
        rows = self.backend.execute(
            f"SELECT id, user_id, name, url, source_type, active, settings_json, last_scraped_at, etag, last_modified, defer_until, status, consec_not_modified, consec_errors, created_at, updated_at FROM sources WHERE {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?",  # nosec B608
            tuple(params + [limit, offset]),
        ).rows
        out: list[SourceRow] = []
        for r in rows:
            sid = int(r.get("id"))
            trows = self.backend.execute(
                "SELECT t.name FROM source_tags st JOIN tags t ON st.tag_id = t.id WHERE st.source_id = ?",
                (sid,),
            ).rows
            tags = [tr.get("name") for tr in trows if tr.get("name")]
            out.append(SourceRow(tags=tags, **r))  # type: ignore[arg-type]
        return out, total

    def update_source(self, source_id: int, patch: dict[str, Any]) -> SourceRow:
        if not patch:
            return self.get_source(source_id)
        fields = []
        params: list[Any] = []
        for key in ("name", "url", "source_type", "active", "settings_json", "status"):
            if key in patch and patch[key] is not None:
                fields.append(f"{key} = ?")
                val = patch[key]
                if key == "active":
                    val = 1 if bool(val) else 0
                params.append(val)
        if fields:
            fields.append("updated_at = ?")
            params.append(_utcnow_iso())
            params.extend([source_id, self.user_id])
            self.backend.execute(
                f"UPDATE sources SET {', '.join(fields)} WHERE id = ? AND user_id = ?",  # nosec B608
                tuple(params),
            )
        # tags/group updates handled via separate helpers
        return self.get_source(source_id)

    def update_source_scrape_meta(
        self,
        source_id: int,
        *,
        last_scraped_at: str | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
        defer_until: str | None = None,
        status: str | None = None,
        consec_not_modified: int | None = None,
        consec_errors: int | None = None,
        active: int | None = None,
    ) -> None:
        """Update scrape metadata fields for a source (idempotent, partial)."""
        fields: list[str] = []
        params: list[Any] = []
        if last_scraped_at is not None:
            fields.append("last_scraped_at = ?")
            params.append(last_scraped_at)
        if etag is not None:
            fields.append("etag = ?")
            params.append(etag)
        if last_modified is not None:
            fields.append("last_modified = ?")
            params.append(last_modified)
        if defer_until is not None:
            fields.append("defer_until = ?")
            params.append(defer_until)
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if consec_not_modified is not None:
            fields.append("consec_not_modified = ?")
            params.append(int(consec_not_modified))
        if consec_errors is not None:
            fields.append("consec_errors = ?")
            params.append(int(consec_errors))
        if active is not None:
            fields.append("active = ?")
            params.append(int(active))
        if not fields:
            return
        params.extend([source_id, self.user_id])
        self.backend.execute(
            f"UPDATE sources SET {', '.join(fields)} WHERE id = ? AND user_id = ?",  # nosec B608
            tuple(params),
        )

    def clear_source_defer_until(self, source_id: int) -> None:
        self.backend.execute(
            "UPDATE sources SET defer_until = NULL WHERE id = ? AND user_id = ?",
            (source_id, self.user_id),
        )

    def set_source_tags(self, source_id: int, tag_names: list[str]) -> list[str]:
        tag_ids = self.ensure_tag_ids(tag_names)
        self.backend.execute("DELETE FROM source_tags WHERE source_id = ?", (source_id,))
        for tid in tag_ids:
            try:
                self.backend.execute(
                    "INSERT INTO source_tags (source_id, tag_id) VALUES (?, ?) ON CONFLICT(source_id, tag_id) DO NOTHING",
                    (source_id, tid),
                )
            except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                self.backend.execute(
                    "INSERT INTO source_tags (source_id, tag_id) SELECT ?, ? WHERE NOT EXISTS (SELECT 1 FROM source_tags WHERE source_id = ? AND tag_id = ?)",
                    (source_id, tid, source_id, tid),
                )
        rows = self.backend.execute(
            "SELECT t.name FROM source_tags st JOIN tags t ON st.tag_id = t.id WHERE st.source_id = ?",
            (source_id,),
        ).rows
        return [r.get("name") for r in rows if r.get("name")]

    def set_source_groups(self, source_id: int, group_ids: list[int]) -> list[int]:
        clean_ids: list[int] = []
        seen: set[int] = set()
        for gid in group_ids or []:
            try:
                val = int(gid)
            except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                continue
            if val in seen:
                continue
            seen.add(val)
            clean_ids.append(val)
        self.backend.execute("DELETE FROM source_groups WHERE source_id = ?", (source_id,))
        for gid in clean_ids:
            try:
                self.backend.execute(
                    "INSERT INTO source_groups (source_id, group_id) VALUES (?, ?) ON CONFLICT(source_id, group_id) DO NOTHING",
                    (source_id, gid),
                )
            except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                self.backend.execute(
                    "INSERT INTO source_groups (source_id, group_id) SELECT ?, ? WHERE NOT EXISTS (SELECT 1 FROM source_groups WHERE source_id = ? AND group_id = ?)",
                    (source_id, gid, source_id, gid),
                )
        return clean_ids

    def get_source_group_ids(self, source_id: int) -> list[int]:
        """Return group IDs for a single source, ordered."""
        rows = self.backend.execute(
            "SELECT group_id FROM source_groups WHERE source_id = ? ORDER BY group_id",
            (source_id,),
        ).rows
        return [int(r.get("group_id")) for r in rows if r.get("group_id") is not None]

    def get_source_group_ids_batch(self, source_ids: list[int]) -> dict[int, list[int]]:
        """Return {source_id: [group_ids]} for multiple sources in one query."""
        if not source_ids:
            return {}
        placeholders = ",".join(["?"] * len(source_ids))
        rows = self.backend.execute(
            f"SELECT source_id, group_id FROM source_groups WHERE source_id IN ({placeholders}) ORDER BY source_id, group_id",  # nosec B608
            tuple(source_ids),
        ).rows
        result: dict[int, list[int]] = {sid: [] for sid in source_ids}
        for r in rows:
            sid = int(r.get("source_id"))
            gid = int(r.get("group_id"))
            if sid in result:
                result[sid].append(gid)
        return result

    def delete_source(self, source_id: int) -> bool:
        with self.transaction() as conn:
            self.backend.execute("DELETE FROM source_groups WHERE source_id = ?", (source_id,), connection=conn)
            self.backend.execute("DELETE FROM source_tags WHERE source_id = ?", (source_id,), connection=conn)
            self.backend.execute(
                "DELETE FROM feed_websub_subscriptions WHERE source_id = ? AND user_id = ?",
                (source_id, self.user_id),
                connection=conn,
            )
            res = self.backend.execute(
                "DELETE FROM sources WHERE id = ? AND user_id = ?",
                (source_id, self.user_id),
                connection=conn,
            )
        return bool(res.rowcount > 0)

    def delete_source_reversible(
        self,
        source_id: int,
        *,
        undo_window_seconds: int,
    ) -> tuple[bool, str | None]:
        self._purge_expired_deleted_records()
        try:
            src = self.get_source(source_id)
        except KeyError:
            return False, None

        payload = {
            "id": int(src.id),
            "name": src.name,
            "url": src.url,
            "source_type": src.source_type,
            "active": bool(src.active),
            "settings_json": src.settings_json,
            "last_scraped_at": src.last_scraped_at,
            "etag": src.etag,
            "last_modified": src.last_modified,
            "defer_until": src.defer_until,
            "status": src.status,
            "consec_not_modified": src.consec_not_modified,
            "consec_errors": src.consec_errors,
            "created_at": src.created_at,
            "updated_at": src.updated_at,
            "tags": list(src.tags or []),
            "group_ids": self.get_source_group_ids(source_id),
        }
        deleted_at = _utcnow_iso()
        expires_at = self._restore_expires_at(undo_window_seconds)

        with self.transaction() as conn:
            self.backend.execute(
                """
                INSERT INTO deleted_sources (user_id, source_id, payload_json, deleted_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, source_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    deleted_at = excluded.deleted_at,
                    expires_at = excluded.expires_at
                """,
                (
                    self.user_id,
                    source_id,
                    json.dumps(payload, ensure_ascii=False),
                    deleted_at,
                    expires_at,
                ),
                connection=conn,
            )
            self.backend.execute("DELETE FROM source_groups WHERE source_id = ?", (source_id,), connection=conn)
            self.backend.execute("DELETE FROM source_tags WHERE source_id = ?", (source_id,), connection=conn)
            self.backend.execute(
                "DELETE FROM feed_websub_subscriptions WHERE source_id = ? AND user_id = ?",
                (source_id, self.user_id),
                connection=conn,
            )
            deleted = self.backend.execute(
                "DELETE FROM sources WHERE id = ? AND user_id = ?",
                (source_id, self.user_id),
                connection=conn,
            )
            if deleted.rowcount <= 0:
                raise KeyError("source_not_found")
        return True, expires_at

    def restore_source(self, source_id: int) -> SourceRow:
        row = self.backend.execute(
            "SELECT payload_json, expires_at FROM deleted_sources WHERE user_id = ? AND source_id = ?",
            (self.user_id, source_id),
        ).first
        if not row:
            raise KeyError("source_restore_not_found")

        expires_at = row.get("expires_at")
        if self._is_restore_expired(str(expires_at) if expires_at is not None else None):
            self.backend.execute(
                "DELETE FROM deleted_sources WHERE user_id = ? AND source_id = ?",
                (self.user_id, source_id),
            )
            raise KeyError("source_restore_expired")

        try:
            payload = json.loads(row.get("payload_json") or "{}")
        except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
            payload = None
        if not isinstance(payload, dict):
            self.backend.execute(
                "DELETE FROM deleted_sources WHERE user_id = ? AND source_id = ?",
                (self.user_id, source_id),
            )
            raise KeyError("source_restore_invalid_payload")

        url = str(payload.get("url") or "").strip()
        source_type = str(payload.get("source_type") or "").strip()
        name = str(payload.get("name") or "").strip() or url
        if not url or not source_type:
            self.backend.execute(
                "DELETE FROM deleted_sources WHERE user_id = ? AND source_id = ?",
                (self.user_id, source_id),
            )
            raise KeyError("source_restore_invalid_payload")

        created_at = str(payload.get("created_at") or _utcnow_iso())
        updated_at = str(payload.get("updated_at") or created_at)
        tags = self._normalize_tag_names(payload.get("tags") if isinstance(payload.get("tags"), list) else [])
        group_ids: list[int] = []
        for raw_gid in payload.get("group_ids") if isinstance(payload.get("group_ids"), list) else []:
            try:
                group_ids.append(int(raw_gid))
            except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                continue

        with self.transaction() as conn:
            conflict = self.backend.execute(
                "SELECT id FROM sources WHERE user_id = ? AND url = ?",
                (self.user_id, url),
                connection=conn,
            ).first
            if conflict and int(conflict.get("id") or 0) != int(source_id):
                raise ValueError("source_restore_url_conflict")

            existing = self.backend.execute(
                "SELECT id FROM sources WHERE id = ? AND user_id = ?",
                (source_id, self.user_id),
                connection=conn,
            ).first
            if existing:
                self.backend.execute(
                    """
                    UPDATE sources
                    SET name = ?, url = ?, source_type = ?, active = ?, settings_json = ?,
                        last_scraped_at = ?, etag = ?, last_modified = ?, defer_until = ?,
                        status = ?, consec_not_modified = ?, consec_errors = ?,
                        created_at = ?, updated_at = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    (
                        name,
                        url,
                        source_type,
                        1 if bool(payload.get("active", True)) else 0,
                        payload.get("settings_json"),
                        payload.get("last_scraped_at"),
                        payload.get("etag"),
                        payload.get("last_modified"),
                        payload.get("defer_until"),
                        payload.get("status"),
                        int(payload.get("consec_not_modified") or 0),
                        int(payload.get("consec_errors") or 0),
                        created_at,
                        updated_at,
                        source_id,
                        self.user_id,
                    ),
                    connection=conn,
                )
            else:
                self.backend.execute(
                    """
                    INSERT INTO sources (
                        id, user_id, name, url, source_type, active, settings_json,
                        last_scraped_at, etag, last_modified, defer_until, status,
                        consec_not_modified, consec_errors, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_id,
                        self.user_id,
                        name,
                        url,
                        source_type,
                        1 if bool(payload.get("active", True)) else 0,
                        payload.get("settings_json"),
                        payload.get("last_scraped_at"),
                        payload.get("etag"),
                        payload.get("last_modified"),
                        payload.get("defer_until"),
                        payload.get("status"),
                        int(payload.get("consec_not_modified") or 0),
                        int(payload.get("consec_errors") or 0),
                        created_at,
                        updated_at,
                    ),
                    connection=conn,
                )

            tag_ids = self._ensure_tag_ids_with_connection(tags, connection=conn)
            self.backend.execute("DELETE FROM source_tags WHERE source_id = ?", (source_id,), connection=conn)
            for tag_id in tag_ids:
                self.backend.execute(
                    """
                    INSERT INTO source_tags (source_id, tag_id) VALUES (?, ?)
                    ON CONFLICT(source_id, tag_id) DO NOTHING
                    """,
                    (source_id, tag_id),
                    connection=conn,
                )

            self.backend.execute("DELETE FROM source_groups WHERE source_id = ?", (source_id,), connection=conn)
            for group_id in group_ids:
                self.backend.execute(
                    """
                    INSERT INTO source_groups (source_id, group_id) VALUES (?, ?)
                    ON CONFLICT(source_id, group_id) DO NOTHING
                    """,
                    (source_id, group_id),
                    connection=conn,
                )

            self.backend.execute(
                "DELETE FROM deleted_sources WHERE user_id = ? AND source_id = ?",
                (self.user_id, source_id),
                connection=conn,
            )

        return self.get_source(source_id)

    def list_sources_by_group_ids(self, group_ids: list[int], *, active_only: bool = True) -> list[SourceRow]:
        """Return sources that belong to any of the provided group IDs (OR semantics)."""
        if not group_ids:
            return []
        placeholders = ",".join(["?"] * len(group_ids))
        active_clause = "AND s.active = 1" if active_only else ""
        rows = self.backend.execute(
            """
            SELECT s.id, s.user_id, s.name, s.url, s.source_type, s.active, s.settings_json,
                   s.last_scraped_at, s.etag, s.last_modified, s.defer_until, s.status,
                   s.consec_not_modified, s.consec_errors, s.created_at, s.updated_at
            FROM sources s
            WHERE s.user_id = ? {active_clause}
              AND EXISTS (
                SELECT 1 FROM source_groups sg
                WHERE sg.source_id = s.id AND sg.group_id IN ({placeholders})
              )
            ORDER BY s.created_at DESC
            """.format_map(locals()),  # nosec B608
            tuple([self.user_id] + group_ids),
        ).rows
        out: list[SourceRow] = []
        for r in rows:
            sid = int(r.get("id"))
            trows = self.backend.execute(
                "SELECT t.name FROM source_tags st JOIN tags t ON st.tag_id = t.id WHERE st.source_id = ?",
                (sid,),
            ).rows
            tags = [tr.get("name") for tr in trows if tr.get("name")]
            out.append(SourceRow(tags=tags, **r))  # type: ignore[arg-type]
        return out

    # ------------------------
    # Groups
    # ------------------------
    def create_group(self, name: str, description: str | None, parent_group_id: int | None) -> GroupRow:
        # Idempotent by (user_id, name)
        try:
            res = self._execute_insert(
                "INSERT INTO groups (user_id, name, description, parent_group_id) VALUES (?, ?, ?, ?)",
                (self.user_id, name, description, parent_group_id),
            )
            new_id = self._extract_lastrowid(res)
            if not new_id:
                raise RuntimeError("failed_to_create_group")
            return self.get_group(new_id)
        except (*_WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS, _DatabaseError):
            # On UNIQUE violation, fetch existing
            row = self.backend.execute(
                "SELECT id FROM groups WHERE user_id = ? AND name = ?",
                (self.user_id, name),
            ).first
            if not row:
                # Re-raise original path if not found
                raise
            return self.get_group(int(row.get("id")))

    def get_group(self, group_id: int) -> GroupRow:
        row = self.backend.execute(
            "SELECT id, user_id, name, description, parent_group_id FROM groups WHERE id = ? AND user_id = ?",
            (group_id, self.user_id),
        ).first
        if not row:
            raise KeyError("group_not_found")
        return GroupRow(**row)

    def list_groups(self, q: str | None, limit: int, offset: int) -> tuple[list[GroupRow], int]:
        where = ["user_id = ?"]
        params: list[Any] = [self.user_id]
        if q:
            where.append("name LIKE ?")
            params.append(f"%{q}%")
        total = int(self.backend.execute(f"SELECT COUNT(*) AS cnt FROM groups WHERE {' AND '.join(where)}", tuple(params)).scalar or 0)  # nosec B608
        rows = self.backend.execute(
            f"SELECT id, user_id, name, description, parent_group_id FROM groups WHERE {' AND '.join(where)} ORDER BY name LIMIT ? OFFSET ?",  # nosec B608
            tuple(params + [limit, offset]),
        ).rows
        return [GroupRow(**r) for r in rows], total

    def update_group(self, group_id: int, patch: dict[str, Any]) -> GroupRow:
        fields = []
        params: list[Any] = []
        for key in ("name", "description", "parent_group_id"):
            if key in patch and patch[key] is not None:
                fields.append(f"{key} = ?")
                params.append(patch[key])
        if fields:
            params.extend([group_id, self.user_id])
            self.backend.execute(
                f"UPDATE groups SET {', '.join(fields)} WHERE id = ? AND user_id = ?",  # nosec B608
                tuple(params),
            )
        return self.get_group(group_id)

    def delete_group(self, group_id: int) -> bool:
        res = self.backend.execute("DELETE FROM groups WHERE id = ? AND user_id = ?", (group_id, self.user_id))
        return res.rowcount > 0

    # ------------------------
    # Jobs & Runs
    # ------------------------
    def create_job(
        self,
        *,
        name: str,
        description: str | None,
        scope_json: str | None,
        schedule_expr: str | None,
        schedule_timezone: str | None,
        active: bool,
        max_concurrency: int | None,
        per_host_delay_ms: int | None,
        retry_policy_json: str | None,
        output_prefs_json: str | None,
        job_filters_json: str | None = None,
    ) -> JobRow:
        now = _utcnow_iso()
        res = self._execute_insert(
            """
            INSERT INTO scrape_jobs (user_id, name, description, scope_json, schedule_expr, active, max_concurrency,
            per_host_delay_ms, retry_policy_json, output_prefs_json, job_filters_json, schedule_timezone, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.user_id,
                name,
                description,
                scope_json,
                schedule_expr,
                1 if active else 0,
                max_concurrency,
                per_host_delay_ms,
                retry_policy_json,
                output_prefs_json,
                job_filters_json,
                schedule_timezone,
                now,
                now,
            ),
        )
        new_id = self._extract_lastrowid(res)
        if not new_id:
            raise RuntimeError("failed_to_create_job")
        return self.get_job(new_id)

    def get_job(self, job_id: int) -> JobRow:
        row = self.backend.execute(
            """
            SELECT id, user_id, name, description, scope_json, schedule_expr, schedule_timezone, active,
                   max_concurrency, per_host_delay_ms, retry_policy_json, output_prefs_json, job_filters_json,
                   created_at, updated_at, last_run_at, next_run_at, wf_schedule_id
            FROM scrape_jobs WHERE id = ? AND user_id = ?
            """,
            (job_id, self.user_id),
        ).first
        if not row:
            raise KeyError("job_not_found")
        return JobRow(**row)

    def list_jobs(self, q: str | None, limit: int, offset: int) -> tuple[list[JobRow], int]:
        where = ["user_id = ?"]
        params: list[Any] = [self.user_id]
        if q:
            where.append("(name LIKE ? OR description LIKE ?)")
            like = f"%{q}%"
            params.extend([like, like])
        total = int(self.backend.execute(f"SELECT COUNT(*) AS cnt FROM scrape_jobs WHERE {' AND '.join(where)}", tuple(params)).scalar or 0)  # nosec B608
        rows = self.backend.execute(
            f"SELECT id, user_id, name, description, scope_json, schedule_expr, schedule_timezone, active, max_concurrency, per_host_delay_ms, retry_policy_json, output_prefs_json, job_filters_json, created_at, updated_at, last_run_at, next_run_at, wf_schedule_id FROM scrape_jobs WHERE {' AND '.join(where)} ORDER BY created_at DESC LIMIT ? OFFSET ?",  # nosec B608
            tuple(params + [limit, offset]),
        ).rows
        return [JobRow(**r) for r in rows], total

    def set_job_filters(self, job_id: int, filters: Any | None) -> JobRow:
        """Replace the job's filters payload.

        Accepts either a dict (e.g., {"filters": [...]}) or a raw list of filter objects.
        Passing None clears the filters (sets NULL).
        Returns the updated JobRow.
        """
        if filters is None:
            payload_json = None
        else:
            try:
                if isinstance(filters, dict):
                    payload = filters
                elif isinstance(filters, list):
                    payload = {"filters": filters}
                else:
                    # Fallback to best-effort JSON-serializable form
                    payload = {"filters": filters}
                payload_json = json.dumps(payload, ensure_ascii=False)
            except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                payload_json = None
        # Use update_job to ensure updated_at is maintained
        return self.update_job(job_id, {"job_filters_json": payload_json})

    def get_job_filters(self, job_id: int) -> dict[str, Any]:
        """Fetch and parse the job's filters payload.

        Returns a dict in the normalized shape {"filters": [...]}. Invalid or missing
        payloads yield {"filters": []}.
        """
        row = self.get_job(job_id)
        raw = getattr(row, "job_filters_json", None)
        if not raw:
            return {"filters": []}
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and isinstance(data.get("filters"), list):
                return {"filters": list(data.get("filters") or [])}
            if isinstance(data, list):
                return {"filters": data}
        except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
            pass
        return {"filters": []}

    def update_job(self, job_id: int, patch: dict[str, Any]) -> JobRow:
        if not patch:
            return self.get_job(job_id)
        fields = []
        params: list[Any] = []
        for key in (
            "name",
            "description",
            "scope_json",
            "schedule_expr",
            "schedule_timezone",
            "active",
            "max_concurrency",
            "per_host_delay_ms",
            "retry_policy_json",
            "output_prefs_json",
            "job_filters_json",
        ):
            if key in patch and patch[key] is not None:
                fields.append(f"{key} = ?")
                val = patch[key]
                if key == "active":
                    val = 1 if bool(val) else 0
                params.append(val)
        if fields:
            fields.append("updated_at = ?")
            params.append(_utcnow_iso())
            params.extend([job_id, self.user_id])
            self.backend.execute(
                f"UPDATE scrape_jobs SET {', '.join(fields)} WHERE id = ? AND user_id = ?",  # nosec B608
                tuple(params),
            )
        return self.get_job(job_id)

    def delete_job(self, job_id: int) -> bool:
        res = self.backend.execute(
            "DELETE FROM scrape_jobs WHERE id = ? AND user_id = ?",
            (job_id, self.user_id),
        )
        return bool(res.rowcount > 0)

    def delete_job_reversible(
        self,
        job_id: int,
        *,
        undo_window_seconds: int,
    ) -> tuple[bool, str | None]:
        self._purge_expired_deleted_records()
        try:
            job = self.get_job(job_id)
        except KeyError:
            return False, None

        payload = {
            "id": int(job.id),
            "name": job.name,
            "description": job.description,
            "scope_json": job.scope_json,
            "schedule_expr": job.schedule_expr,
            "schedule_timezone": job.schedule_timezone,
            "active": bool(job.active),
            "max_concurrency": job.max_concurrency,
            "per_host_delay_ms": job.per_host_delay_ms,
            "retry_policy_json": job.retry_policy_json,
            "output_prefs_json": job.output_prefs_json,
            "job_filters_json": job.job_filters_json,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "last_run_at": job.last_run_at,
            "next_run_at": job.next_run_at,
            "wf_schedule_id": job.wf_schedule_id,
        }
        deleted_at = _utcnow_iso()
        expires_at = self._restore_expires_at(undo_window_seconds)

        with self.transaction() as conn:
            self.backend.execute(
                """
                INSERT INTO deleted_jobs (user_id, job_id, payload_json, deleted_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, job_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    deleted_at = excluded.deleted_at,
                    expires_at = excluded.expires_at
                """,
                (
                    self.user_id,
                    job_id,
                    json.dumps(payload, ensure_ascii=False),
                    deleted_at,
                    expires_at,
                ),
                connection=conn,
            )
            deleted = self.backend.execute(
                "DELETE FROM scrape_jobs WHERE id = ? AND user_id = ?",
                (job_id, self.user_id),
                connection=conn,
            )
            if deleted.rowcount <= 0:
                raise KeyError("job_not_found")
        return True, expires_at

    def restore_job(self, job_id: int) -> JobRow:
        row = self.backend.execute(
            "SELECT payload_json, expires_at FROM deleted_jobs WHERE user_id = ? AND job_id = ?",
            (self.user_id, job_id),
        ).first
        if not row:
            raise KeyError("job_restore_not_found")

        expires_at = row.get("expires_at")
        if self._is_restore_expired(str(expires_at) if expires_at is not None else None):
            self.backend.execute(
                "DELETE FROM deleted_jobs WHERE user_id = ? AND job_id = ?",
                (self.user_id, job_id),
            )
            raise KeyError("job_restore_expired")

        try:
            payload = json.loads(row.get("payload_json") or "{}")
        except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
            payload = None
        if not isinstance(payload, dict):
            self.backend.execute(
                "DELETE FROM deleted_jobs WHERE user_id = ? AND job_id = ?",
                (self.user_id, job_id),
            )
            raise KeyError("job_restore_invalid_payload")

        name = str(payload.get("name") or "").strip()
        if not name:
            self.backend.execute(
                "DELETE FROM deleted_jobs WHERE user_id = ? AND job_id = ?",
                (self.user_id, job_id),
            )
            raise KeyError("job_restore_invalid_payload")

        created_at = str(payload.get("created_at") or _utcnow_iso())
        updated_at = str(payload.get("updated_at") or created_at)

        with self.transaction() as conn:
            existing = self.backend.execute(
                "SELECT id FROM scrape_jobs WHERE id = ? AND user_id = ?",
                (job_id, self.user_id),
                connection=conn,
            ).first

            values = (
                name,
                payload.get("description"),
                payload.get("scope_json"),
                payload.get("schedule_expr"),
                payload.get("schedule_timezone"),
                1 if bool(payload.get("active", True)) else 0,
                payload.get("max_concurrency"),
                payload.get("per_host_delay_ms"),
                payload.get("retry_policy_json"),
                payload.get("output_prefs_json"),
                payload.get("job_filters_json"),
                created_at,
                updated_at,
                payload.get("last_run_at"),
                payload.get("next_run_at"),
                payload.get("wf_schedule_id"),
            )

            if existing:
                self.backend.execute(
                    """
                    UPDATE scrape_jobs
                    SET name = ?, description = ?, scope_json = ?, schedule_expr = ?, schedule_timezone = ?,
                        active = ?, max_concurrency = ?, per_host_delay_ms = ?, retry_policy_json = ?,
                        output_prefs_json = ?, job_filters_json = ?, created_at = ?, updated_at = ?,
                        last_run_at = ?, next_run_at = ?, wf_schedule_id = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    (*values, job_id, self.user_id),
                    connection=conn,
                )
            else:
                self.backend.execute(
                    """
                    INSERT INTO scrape_jobs (
                        id, user_id, name, description, scope_json, schedule_expr, schedule_timezone,
                        active, max_concurrency, per_host_delay_ms, retry_policy_json, output_prefs_json,
                        job_filters_json, created_at, updated_at, last_run_at, next_run_at, wf_schedule_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, self.user_id, *values),
                    connection=conn,
                )

            self.backend.execute(
                "DELETE FROM deleted_jobs WHERE user_id = ? AND job_id = ?",
                (self.user_id, job_id),
                connection=conn,
            )

        return self.get_job(job_id)

    # ---- Schedule/history helpers ----
    def set_job_schedule_id(self, job_id: int, schedule_id: str | None) -> None:
        self.backend.execute(
            "UPDATE scrape_jobs SET wf_schedule_id = ? WHERE id = ? AND user_id = ?",
            (schedule_id, job_id, self.user_id),
        )

    def set_job_history(self, job_id: int, *, last_run_at: str | None = None, next_run_at: str | None = None) -> None:
        sets = []
        params: list[Any] = []
        if last_run_at is not None:
            sets.append("last_run_at = ?")
            params.append(last_run_at)
        if next_run_at is not None:
            sets.append("next_run_at = ?")
            params.append(next_run_at)
        if not sets:
            return
        params.extend([job_id, self.user_id])
        self.backend.execute(
            f"UPDATE scrape_jobs SET {', '.join(sets)} WHERE id = ? AND user_id = ?",  # nosec B608
            tuple(params),
        )

    def create_run(self, job_id: int, status: str = "queued") -> RunRow:
        # Enforce ownership even in shared-backend deployments.
        self.get_job(job_id)
        res = self._execute_insert(
            "INSERT INTO scrape_runs (job_id, status, started_at) VALUES (?, ?, ?)",
            (job_id, status, _utcnow_iso()),
        )
        new_id = self._extract_lastrowid(res)
        if not new_id:
            raise RuntimeError("failed_to_create_run")
        return self.get_run(new_id)

    def get_run(self, run_id: int) -> RunRow:
        row = self.backend.execute(
            """
            SELECT sr.id, sr.job_id, sr.status, sr.started_at, sr.finished_at, sr.stats_json, sr.error_msg, sr.log_path
            FROM scrape_runs sr
            JOIN scrape_jobs sj ON sj.id = sr.job_id
            WHERE sr.id = ? AND sj.user_id = ?
            """,
            (run_id, self.user_id),
        ).first
        if not row:
            raise KeyError("run_not_found")
        return RunRow(**row)

    def list_runs_for_job(self, job_id: int, limit: int, offset: int) -> tuple[list[RunRow], int]:
        total = int(
            self.backend.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM scrape_runs sr
                JOIN scrape_jobs sj ON sj.id = sr.job_id
                WHERE sr.job_id = ? AND sj.user_id = ?
                """,
                (job_id, self.user_id),
            ).scalar
            or 0
        )
        rows = self.backend.execute(
            """
            SELECT sr.id, sr.job_id, sr.status, sr.started_at, sr.finished_at, sr.stats_json, sr.error_msg, sr.log_path
            FROM scrape_runs sr
            JOIN scrape_jobs sj ON sj.id = sr.job_id
            WHERE sr.job_id = ? AND sj.user_id = ?
            ORDER BY sr.id DESC
            LIMIT ? OFFSET ?
            """,
            (job_id, self.user_id, limit, offset),
        ).rows
        return [RunRow(**r) for r in rows], total

    def list_runs(self, q: str | None, limit: int, offset: int) -> tuple[list[RunRow], int]:
        """List runs across all jobs for this user.

        Optional q filters by job name/description, run status, or run id (text match).
        """
        where = ["j.user_id = ?"]
        params: list[Any] = [self.user_id]
        if q:
            like = f"%{q}%"
            # Match job name/description or run status/id (text)
            where.append("(j.name LIKE ? OR j.description LIKE ? OR sr.status LIKE ? OR CAST(sr.id AS TEXT) LIKE ?)")
            params.extend([like, like, like, like])
        where_sql = " AND ".join(where)
        total = int(
            self.backend.execute(
                f"SELECT COUNT(*) AS cnt FROM scrape_runs sr JOIN scrape_jobs j ON j.id = sr.job_id WHERE {where_sql}",  # nosec B608
                tuple(params),
            ).scalar or 0
        )
        rows = self.backend.execute(
            """
            SELECT sr.id, sr.job_id, sr.status, sr.started_at, sr.finished_at, sr.stats_json, sr.error_msg, sr.log_path
            FROM scrape_runs sr JOIN scrape_jobs j ON j.id = sr.job_id
            WHERE {where_sql}
            ORDER BY sr.id DESC
            LIMIT ? OFFSET ?
            """.format_map(locals()),  # nosec B608
            tuple(params + [limit, offset]),
        ).rows
        return [RunRow(**r) for r in rows], total

    def append_run_item(self, run_id: int, media_id: int, source_id: int | None = None) -> None:
        # Ensure the run belongs to this user in shared-backend deployments.
        self.get_run(run_id)
        stmt = (
            "INSERT INTO scrape_run_items (run_id, media_id, source_id) "
            "VALUES (?, ?, ?) ON CONFLICT(run_id, media_id) DO NOTHING"
        )
        try:
            self.backend.execute(stmt, (run_id, media_id, source_id))
        except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
            self.backend.execute(
                "INSERT INTO scrape_run_items (run_id, media_id, source_id) "
                "SELECT ?, ?, ? WHERE NOT EXISTS (SELECT 1 FROM scrape_run_items WHERE run_id = ? AND media_id = ?)",
                (run_id, media_id, source_id, run_id, media_id),
            )

    def list_run_media_ids(self, run_id: int, limit: int = 1000) -> list[int]:
        # Ensure run ownership before returning media links.
        self.get_run(run_id)
        rows = self.backend.execute(
            "SELECT media_id FROM scrape_run_items WHERE run_id = ? ORDER BY media_id LIMIT ?",
            (run_id, limit),
        ).rows
        mids: list[int] = []
        for r in rows:
            v = r.get("media_id")
            try:
                mids.append(int(v))
            except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                continue
        return mids

    # ------------------------
    # Scraped items
    # ------------------------
    def record_scraped_item(
        self,
        *,
        run_id: int,
        job_id: int,
        source_id: int,
        media_id: int | None,
        media_uuid: str | None,
        url: str | None,
        title: str | None,
        summary: str | None,
        published_at: str | None,
        tags: list[str] | None,
        status: str,
        content: str | None = None,
    ) -> ScrapedItemRow:
        created_at = _utcnow_iso()
        tags_json = json.dumps(tags or [])
        res = self._execute_insert(
            """
            INSERT INTO scraped_items (
                run_id, job_id, source_id, media_id, media_uuid, url, title,
                summary, content, published_at, tags_json, status, reviewed, queued_for_briefing, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?)
            """,
            (
                run_id,
                job_id,
                source_id,
                media_id,
                media_uuid,
                url,
                title,
                summary,
                content,
                published_at,
                tags_json,
                status,
                created_at,
            ),
        )
        new_id = self._extract_lastrowid(res)
        if not new_id:
            raise RuntimeError("failed_to_record_scraped_item")
        return self.get_item(new_id)

    def get_item(self, item_id: int) -> ScrapedItemRow:
        row = self.backend.execute(
            """
            SELECT si.id, si.run_id, si.job_id, si.source_id, si.media_id, si.media_uuid, si.url, si.title,
                   si.summary, si.content, si.published_at, si.tags_json, si.status, si.reviewed, si.queued_for_briefing, si.created_at
            FROM scraped_items si
            JOIN scrape_jobs sj ON sj.id = si.job_id
            WHERE si.id = ? AND sj.user_id = ?
            """,
            (item_id, self.user_id),
        ).first
        if not row:
            raise KeyError("item_not_found")
        return ScrapedItemRow(**row)

    def list_items(
        self,
        *,
        run_id: int | None = None,
        job_id: int | None = None,
        source_id: int | None = None,
        status: str | None = None,
        reviewed: bool | None = None,
        queued_for_briefing: bool | None = None,
        search: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[ScrapedItemRow], int]:
        where = ["sj.user_id = ?"]
        params: list[Any] = [self.user_id]
        if run_id is not None:
            where.append("si.run_id = ?")
            params.append(run_id)
        if job_id is not None:
            where.append("si.job_id = ?")
            params.append(job_id)
        if source_id is not None:
            where.append("si.source_id = ?")
            params.append(source_id)
        if status:
            where.append("si.status = ?")
            params.append(status)
        if reviewed is not None:
            where.append("si.reviewed = ?")
            params.append(1 if reviewed else 0)
        if queued_for_briefing is not None:
            where.append("si.queued_for_briefing = ?")
            params.append(1 if queued_for_briefing else 0)
        if since:
            where.append("si.created_at >= ?")
            params.append(since)
        if until:
            where.append("si.created_at <= ?")
            params.append(until)
        if search:
            like = f"%{search}%"
            where.append("(si.title LIKE ? OR si.summary LIKE ? OR si.content LIKE ?)")
            params.extend([like, like, like])
        where_sql = " AND ".join(where)
        total = int(
            self.backend.execute(
                f"SELECT COUNT(*) AS cnt FROM scraped_items si JOIN scrape_jobs sj ON sj.id = si.job_id WHERE {where_sql}",  # nosec B608
                tuple(params),
            ).scalar
            or 0
        )
        rows = self.backend.execute(
            """
            SELECT si.id, si.run_id, si.job_id, si.source_id, si.media_id, si.media_uuid, si.url, si.title,
                   si.summary, si.content, si.published_at, si.tags_json, si.status, si.reviewed, si.queued_for_briefing, si.created_at
            FROM scraped_items si
            JOIN scrape_jobs sj ON sj.id = si.job_id
            WHERE {where_sql}
            ORDER BY si.created_at DESC
            LIMIT ? OFFSET ?
            """.format_map(locals()),  # nosec B608
            tuple(params + [limit, offset]),
        ).rows
        return [ScrapedItemRow(**r) for r in rows], total

    def get_item_smart_counts(
        self,
        *,
        run_id: int | None = None,
        job_id: int | None = None,
        source_id: int | None = None,
        status: str | None = None,
        search: str | None = None,
        since: str | None = None,
        until: str | None = None,
        queue_run_id: int | None = None,
        today_since: str | None = None,
    ) -> dict[str, int]:
        where = ["sj.user_id = ?"]
        params: list[Any] = [self.user_id]
        if run_id is not None:
            where.append("si.run_id = ?")
            params.append(run_id)
        if job_id is not None:
            where.append("si.job_id = ?")
            params.append(job_id)
        if source_id is not None:
            where.append("si.source_id = ?")
            params.append(source_id)
        if status:
            where.append("si.status = ?")
            params.append(status)
        if since:
            where.append("si.created_at >= ?")
            params.append(since)
        if until:
            where.append("si.created_at <= ?")
            params.append(until)
        if search:
            like = f"%{search}%"
            where.append("(si.title LIKE ? OR si.summary LIKE ? OR si.content LIKE ?)")
            params.extend([like, like, like])

        today_cutoff = today_since
        if not today_cutoff:
            utc_now = datetime.now(timezone.utc)
            today_cutoff = utc_now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        where_sql = " AND ".join(where)
        row = self.backend.execute(
            f"""
            SELECT
              COUNT(*) AS all_count,
              COALESCE(SUM(CASE WHEN si.created_at >= ? THEN 1 ELSE 0 END), 0) AS today_count,
              COALESCE(SUM(CASE WHEN si.created_at >= ? AND si.reviewed = 0 THEN 1 ELSE 0 END), 0) AS today_unread_count,
              COALESCE(SUM(CASE WHEN si.reviewed = 0 THEN 1 ELSE 0 END), 0) AS unread_count,
              COALESCE(SUM(CASE WHEN si.reviewed = 1 THEN 1 ELSE 0 END), 0) AS reviewed_count,
              COALESCE(SUM(CASE WHEN si.queued_for_briefing = 1 AND (? IS NULL OR si.run_id = ?) THEN 1 ELSE 0 END), 0) AS queued_count
            FROM scraped_items si
            JOIN scrape_jobs sj ON sj.id = si.job_id
            WHERE {where_sql}
            """,  # nosec B608
            tuple([today_cutoff, today_cutoff, queue_run_id, queue_run_id, *params]),
        ).first
        return {
            "all": int((row or {}).get("all_count") or 0),
            "today": int((row or {}).get("today_count") or 0),
            "today_unread": int((row or {}).get("today_unread_count") or 0),
            "unread": int((row or {}).get("unread_count") or 0),
            "reviewed": int((row or {}).get("reviewed_count") or 0),
            "queued": int((row or {}).get("queued_count") or 0),
        }

    def get_items_by_ids(self, item_ids: list[int]) -> list[ScrapedItemRow]:
        if not item_ids:
            return []
        placeholders = ",".join("?" for _ in item_ids)
        rows = self.backend.execute(
            """
            SELECT si.id, si.run_id, si.job_id, si.source_id, si.media_id, si.media_uuid, si.url, si.title,
                   si.summary, si.content, si.published_at, si.tags_json, si.status, si.reviewed, si.queued_for_briefing, si.created_at
            FROM scraped_items si
            JOIN scrape_jobs sj ON sj.id = si.job_id
            WHERE si.id IN ({placeholders}) AND sj.user_id = ?
            """.format_map(locals()),  # nosec B608
            tuple(item_ids + [self.user_id]),
        ).rows
        return [ScrapedItemRow(**r) for r in rows]

    def update_item_flags(
        self,
        item_id: int,
        *,
        reviewed: bool | None = None,
        status: str | None = None,
        queued_for_briefing: bool | None = None,
    ) -> ScrapedItemRow:
        fields: list[str] = []
        params: list[Any] = []
        if reviewed is not None:
            fields.append("reviewed = ?")
            params.append(1 if reviewed else 0)
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if queued_for_briefing is not None:
            fields.append("queued_for_briefing = ?")
            params.append(1 if queued_for_briefing else 0)
        if not fields:
            return self.get_item(item_id)
        params.extend([item_id, self.user_id])
        self.backend.execute(
            (
                f"UPDATE scraped_items SET {', '.join(fields)} "  # nosec B608
                "WHERE id = ? AND job_id IN (SELECT id FROM scrape_jobs WHERE user_id = ?)"
            ),
            tuple(params),
        )
        return self.get_item(item_id)

    def update_run(
        self,
        run_id: int,
        *,
        status: str | None = None,
        finished_at: str | None = None,
        stats_json: str | None = None,
        error_msg: str | None = None,
        log_path: str | None = None,
    ) -> RunRow:
        fields: list[str] = []
        params: list[Any] = []
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if finished_at is not None:
            fields.append("finished_at = ?")
            params.append(finished_at)
        if stats_json is not None:
            fields.append("stats_json = ?")
            params.append(stats_json)
        if error_msg is not None:
            fields.append("error_msg = ?")
            params.append(error_msg)
        if log_path is not None:
            fields.append("log_path = ?")
            params.append(log_path)
        if not fields:
            return self.get_run(run_id)
        self.backend.execute(
            (
                f"UPDATE scrape_runs SET {', '.join(fields)} "  # nosec B608
                "WHERE id = ? AND job_id IN (SELECT id FROM scrape_jobs WHERE user_id = ?)"
            ),
            tuple(params + [run_id, self.user_id]),
        )
        return self.get_run(run_id)

    # ------------------------
    # RSS item-level dedup helpers
    # ------------------------
    def has_seen_item(self, source_id: int, item_key: str) -> bool:
        row = self.backend.execute(
            "SELECT 1 FROM source_seen_items WHERE source_id = ? AND item_key = ?",
            (source_id, item_key),
        ).first
        return bool(row)

    def mark_seen_item(
        self,
        source_id: int,
        item_key: str,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
        seen_at: str | None = None,
    ) -> None:
        ts = seen_at or _utcnow_iso()
        # SQLite upsert pattern; backend handles SQL transparently in SQLite/Postgres where available
        try:
            self.backend.execute(
                """
                INSERT INTO source_seen_items (source_id, item_key, etag, last_modified, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, item_key) DO UPDATE SET
                    etag = COALESCE(excluded.etag, source_seen_items.etag),
                    last_modified = COALESCE(excluded.last_modified, source_seen_items.last_modified),
                    last_seen_at = excluded.last_seen_at
                """,
                (source_id, item_key, etag, last_modified, ts, ts),
            )
        except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
            # Fallback for engines without ON CONFLICT support
            row = self.backend.execute(
                "SELECT 1 FROM source_seen_items WHERE source_id = ? AND item_key = ?",
                (source_id, item_key),
            ).first
            if row:
                self.backend.execute(
                    "UPDATE source_seen_items SET etag = COALESCE(?, etag), last_modified = COALESCE(?, last_modified), last_seen_at = ? WHERE source_id = ? AND item_key = ?",
                    (etag, last_modified, ts, source_id, item_key),
                )
            else:
                self.backend.execute(
                    "INSERT INTO source_seen_items (source_id, item_key, etag, last_modified, first_seen_at, last_seen_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (source_id, item_key, etag, last_modified, ts, ts),
                )

    def list_seen_item_keys(self, source_id: int, *, limit: int | None = None) -> list[str]:
        sql = "SELECT item_key FROM source_seen_items WHERE source_id = ? ORDER BY last_seen_at DESC"
        params: tuple[Any, ...] = (source_id,)
        if isinstance(limit, int) and limit > 0:
            sql = sql + " LIMIT ?"
            params = (source_id, limit)
        rows = self.backend.execute(sql, params).rows
        keys: list[str] = []
        for r in rows:
            k = r.get("item_key")
            if isinstance(k, str) and k:
                keys.append(k)
        return keys

    def get_seen_item_stats(self, source_id: int) -> dict[str, Any]:
        row = self.backend.execute(
            "SELECT COUNT(*) AS seen_count, MAX(last_seen_at) AS latest_seen_at "
            "FROM source_seen_items WHERE source_id = ?",
            (source_id,),
        ).first
        if not row:
            return {"seen_count": 0, "latest_seen_at": None}
        try:
            count_val = int(row.get("seen_count") or 0)
        except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
            count_val = 0
        latest_seen_at = None
        try:
            latest_seen_at = row.get("latest_seen_at")
        except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
            latest_seen_at = None
        return {"seen_count": count_val, "latest_seen_at": latest_seen_at}

    def clear_seen_items(self, source_id: int) -> int:
        row = self.backend.execute(
            "SELECT COUNT(*) AS cnt FROM source_seen_items WHERE source_id = ?",
            (source_id,),
        ).first
        try:
            before_count = int((row or {}).get("cnt") or 0)
        except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
            before_count = 0
        self.backend.execute(
            "DELETE FROM source_seen_items WHERE source_id = ?",
            (source_id,),
        )
        return before_count

    def reset_source_backoff_state(self, source_id: int) -> bool:
        result = self.backend.execute(
            "UPDATE sources SET defer_until = NULL, consec_not_modified = 0 WHERE id = ? AND user_id = ?",
            (source_id, self.user_id),
        )
        try:
            return int(getattr(result, "rowcount", 0) or 0) > 0
        except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
            return False

    # ------------------------
    # Claim cluster subscriptions
    # ------------------------
    def add_watchlist_cluster(self, job_id: int, cluster_id: int) -> None:
        # Guard against cross-user subscription writes in shared-backend modes.
        self.get_job(int(job_id))
        ts = _utcnow_iso()
        self.backend.execute(
            "INSERT INTO watchlist_clusters (job_id, cluster_id, created_at) "
            "VALUES (?, ?, ?) ON CONFLICT(job_id, cluster_id) DO NOTHING",
            (int(job_id), int(cluster_id), ts),
        )

    def remove_watchlist_cluster(self, job_id: int, cluster_id: int) -> bool:
        result = self.backend.execute(
            (
                "DELETE FROM watchlist_clusters "
                "WHERE job_id = ? AND cluster_id = ? "
                "AND job_id IN (SELECT id FROM scrape_jobs WHERE user_id = ?)"
            ),
            (int(job_id), int(cluster_id), self.user_id),
        )
        try:
            return int(getattr(result, "rowcount", 0) or 0) > 0
        except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
            return False

    def list_watchlist_clusters(self, job_id: int) -> list[dict[str, Any]]:
        rows = self.backend.execute(
            """
            SELECT wc.cluster_id, wc.created_at
            FROM watchlist_clusters wc
            JOIN scrape_jobs sj ON sj.id = wc.job_id
            WHERE wc.job_id = ? AND sj.user_id = ?
            ORDER BY wc.created_at DESC
            """,
            (int(job_id), self.user_id),
        ).rows
        return [dict(r) for r in rows or []]

    def list_watchlist_cluster_subscriptions(
        self,
        *,
        cluster_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT wc.job_id, wc.cluster_id, wc.created_at "
            "FROM watchlist_clusters wc "
            "JOIN scrape_jobs sj ON sj.id = wc.job_id "
            "WHERE sj.user_id = ?"
        )
        params: list[Any] = [self.user_id]
        if cluster_ids:
            placeholders = ",".join("?" * len(cluster_ids))
            sql += f" AND wc.cluster_id IN ({placeholders})"
            params.extend(int(cid) for cid in cluster_ids)
        sql += " ORDER BY wc.created_at DESC"
        rows = self.backend.execute(sql, tuple(params)).rows
        return [dict(r) for r in rows or []]

    def list_watchlist_cluster_counts(
        self,
        *,
        cluster_ids: list[int] | None = None,
    ) -> dict[int, int]:
        sql = (
            "SELECT wc.cluster_id, COUNT(*) AS cnt "
            "FROM watchlist_clusters wc "
            "JOIN scrape_jobs sj ON sj.id = wc.job_id "
            "WHERE sj.user_id = ?"
        )
        params: list[Any] = [self.user_id]
        if cluster_ids:
            placeholders = ",".join("?" * len(cluster_ids))
            sql += f" AND wc.cluster_id IN ({placeholders})"
            params.extend(int(cid) for cid in cluster_ids)
        sql += " GROUP BY wc.cluster_id"
        rows = self.backend.execute(sql, tuple(params)).rows
        counts: dict[int, int] = {}
        for row in rows or []:
            try:
                cluster_id = int(row.get("cluster_id"))
                counts[cluster_id] = int(row.get("cnt") or 0)
            except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                continue
        return counts

    # ------------------------
    # Onboarding telemetry
    # ------------------------
    _ONBOARDING_EVENT_TYPES: set[str] = {
        "quick_setup_opened",
        "quick_setup_step_completed",
        "quick_setup_cancelled",
        "quick_setup_completed",
        "quick_setup_failed",
        "quick_setup_preview_loaded",
        "quick_setup_preview_failed",
        "quick_setup_test_run_triggered",
        "quick_setup_test_run_failed",
        "quick_setup_first_run_succeeded",
        "quick_setup_first_output_succeeded",
        "guided_tour_started",
        "guided_tour_step_viewed",
        "guided_tour_completed",
        "guided_tour_dismissed",
        "guided_tour_resumed",
    }

    @staticmethod
    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        mid = len(sorted_values) // 2
        if len(sorted_values) % 2 == 1:
            return float(sorted_values[mid])
        return (float(sorted_values[mid - 1]) + float(sorted_values[mid])) / 2.0

    def record_onboarding_event(
        self,
        *,
        session_id: str,
        event_type: str,
        event_at: str | None,
        details: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate and persist a single onboarding telemetry event for the current user."""
        normalized_session = str(session_id or "").strip()
        if not normalized_session:
            return {"accepted": False, "code": "session_id_required"}

        normalized_event_type = str(event_type or "").strip().lower()
        if not normalized_event_type:
            return {"accepted": False, "code": "event_type_required"}
        if normalized_event_type not in self._ONBOARDING_EVENT_TYPES:
            return {"accepted": False, "code": "event_type_invalid"}

        parsed_event_at = self._parse_iso_utc(event_at)
        if event_at is not None and parsed_event_at is None:
            return {"accepted": False, "code": "event_at_invalid"}
        if parsed_event_at is None:
            parsed_event_at = datetime.now(timezone.utc)

        details_payload: dict[str, Any] = {}
        if details is not None:
            if not isinstance(details, dict):
                return {"accepted": False, "code": "details_invalid"}
            details_payload = details

        now_iso = _utcnow_iso()
        self.backend.execute(
            """
            INSERT INTO watchlist_onboarding_events (
                user_id,
                session_id,
                event_type,
                event_at,
                details_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self.user_id,
                normalized_session[:128],
                normalized_event_type,
                parsed_event_at.isoformat(),
                json.dumps(details_payload, ensure_ascii=False),
                now_iso,
            ),
        )
        return {"accepted": True, "code": None}

    def summarize_onboarding_events(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate onboarding telemetry counters, rates, and timing metrics for a window."""
        parsed_since = self._parse_iso_utc(since)
        parsed_until = self._parse_iso_utc(until)
        since_iso = parsed_since.isoformat() if parsed_since else None
        until_iso = parsed_until.isoformat() if parsed_until else None

        sql = (
            "SELECT session_id, event_type, event_at "
            "FROM watchlist_onboarding_events WHERE user_id = ?"
        )
        params: list[Any] = [self.user_id]
        if since_iso:
            sql += " AND event_at >= ?"
            params.append(since_iso)
        if until_iso:
            sql += " AND event_at <= ?"
            params.append(until_iso)
        sql += " ORDER BY event_at ASC"
        rows = self.backend.execute(sql, tuple(params)).rows

        counters: dict[str, int] = {event_type: 0 for event_type in self._ONBOARDING_EVENT_TYPES}
        sessions: set[str] = set()
        session_events: dict[str, dict[str, datetime | None]] = {}

        for row in rows or []:
            session_id = str(row.get("session_id") or "").strip()
            if not session_id:
                continue
            event_type = str(row.get("event_type") or "").strip().lower()
            parsed_event_at = self._parse_iso_utc(row.get("event_at"))
            if not parsed_event_at:
                continue

            sessions.add(session_id)
            counters[event_type] = counters.get(event_type, 0) + 1

            event_record = session_events.setdefault(
                session_id,
                {
                    "opened_at": None,
                    "completed_at": None,
                    "first_run_at": None,
                    "first_output_at": None,
                },
            )
            if event_type == "quick_setup_opened" and event_record["opened_at"] is None:
                event_record["opened_at"] = parsed_event_at
            elif event_type == "quick_setup_completed" and event_record["completed_at"] is None:
                event_record["completed_at"] = parsed_event_at
            elif event_type == "quick_setup_first_run_succeeded" and event_record["first_run_at"] is None:
                event_record["first_run_at"] = parsed_event_at
            elif (
                event_type == "quick_setup_first_output_succeeded"
                and event_record["first_output_at"] is None
            ):
                event_record["first_output_at"] = parsed_event_at

        opened_sessions = 0
        completed_sessions = 0
        first_run_sessions = 0
        first_output_sessions = 0
        setup_durations: list[float] = []
        first_run_durations: list[float] = []
        first_output_durations: list[float] = []

        for event_record in session_events.values():
            opened_at = event_record.get("opened_at")
            completed_at = event_record.get("completed_at")
            first_run_at = event_record.get("first_run_at")
            first_output_at = event_record.get("first_output_at")

            if opened_at is not None:
                opened_sessions += 1
            if completed_at is not None:
                completed_sessions += 1
            if first_run_at is not None:
                first_run_sessions += 1
            if first_output_at is not None:
                first_output_sessions += 1

            if opened_at is not None and completed_at is not None and completed_at >= opened_at:
                setup_durations.append((completed_at - opened_at).total_seconds())
            if opened_at is not None and first_run_at is not None and first_run_at >= opened_at:
                first_run_durations.append((first_run_at - opened_at).total_seconds())
            if opened_at is not None and first_output_at is not None and first_output_at >= opened_at:
                first_output_durations.append((first_output_at - opened_at).total_seconds())

        counters["sessions"] = len(sessions)
        counters["users"] = 1 if sessions else 0
        counters["setup_completed_sessions"] = completed_sessions
        counters["first_run_success_sessions"] = first_run_sessions
        counters["first_output_success_sessions"] = first_output_sessions

        rates = {
            "setup_completion_rate": (
                float(completed_sessions) / float(opened_sessions) if opened_sessions > 0 else 0.0
            ),
            "first_run_success_rate": (
                float(first_run_sessions) / float(completed_sessions)
                if completed_sessions > 0
                else 0.0
            ),
            "first_output_success_rate": (
                float(first_output_sessions) / float(completed_sessions)
                if completed_sessions > 0
                else 0.0
            ),
        }

        timings = {
            "median_seconds_to_setup_completion": self._median(setup_durations),
            "median_seconds_to_first_run_success": self._median(first_run_durations),
            "median_seconds_to_first_output_success": self._median(first_output_durations),
        }

        return {
            "counters": counters,
            "rates": rates,
            "timings": timings,
            "since": since,
            "until": until,
        }

    def list_completed_run_ids(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> list[int]:
        parsed_since = self._parse_iso_utc(since)
        parsed_until = self._parse_iso_utc(until)
        since_iso = parsed_since.isoformat() if parsed_since else None
        until_iso = parsed_until.isoformat() if parsed_until else None

        sql = (
            "SELECT sr.id AS run_id "
            "FROM scrape_runs sr "
            "JOIN scrape_jobs sj ON sj.id = sr.job_id "
            "WHERE sj.user_id = ? AND LOWER(sr.status) = 'completed'"
        )
        params: list[Any] = [self.user_id]
        if since_iso:
            sql += " AND COALESCE(sr.finished_at, sr.started_at) >= ?"
            params.append(since_iso)
        if until_iso:
            sql += " AND COALESCE(sr.finished_at, sr.started_at) <= ?"
            params.append(until_iso)
        rows = self.backend.execute(sql, tuple(params)).rows

        run_ids: list[int] = []
        for row in rows or []:
            try:
                run_ids.append(int(row.get("run_id")))
            except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                continue
        return run_ids

    # ------------------------
    # IA experiment telemetry
    # ------------------------
    @staticmethod
    def _parse_iso_utc(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            raw = str(value).strip()
            if not raw:
                return None
            if raw.endswith("Z"):
                raw = f"{raw[:-1]}+00:00"
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
            return None

    def record_ia_experiment_event(
        self,
        *,
        variant: str,
        session_id: str,
        previous_tab: str | None,
        current_tab: str,
        transitions: int,
        visited_tabs: list[str] | None,
        first_seen_at: str | None,
        last_seen_at: str | None,
    ) -> bool:
        normalized_session = str(session_id or "").strip()
        if not normalized_session:
            return False

        normalized_variant = "experimental" if str(variant or "").strip().lower() == "experimental" else "baseline"
        normalized_current_tab = str(current_tab or "").strip().lower()[:64] or "unknown"
        normalized_previous_tab = str(previous_tab or "").strip().lower()[:64] or None
        now_iso = _utcnow_iso()

        tab_set: set[str] = set()
        normalized_visited_tabs: list[str] = []
        for raw_tab in visited_tabs or []:
            tab = str(raw_tab or "").strip().lower()
            if not tab or tab in tab_set:
                continue
            tab_set.add(tab)
            normalized_visited_tabs.append(tab[:64])
            if len(normalized_visited_tabs) >= 64:
                break
        if normalized_current_tab not in tab_set and len(normalized_visited_tabs) < 64:
            tab_set.add(normalized_current_tab)
            normalized_visited_tabs.append(normalized_current_tab)
        if (
            normalized_previous_tab
            and normalized_previous_tab not in tab_set
            and len(normalized_visited_tabs) < 64
        ):
            tab_set.add(normalized_previous_tab)
            normalized_visited_tabs.append(normalized_previous_tab)

        parsed_first = self._parse_iso_utc(first_seen_at)
        parsed_last = self._parse_iso_utc(last_seen_at)
        elapsed_ms: int | None = None
        if parsed_first and parsed_last:
            delta_ms = int((parsed_last - parsed_first).total_seconds() * 1000)
            if 0 <= delta_ms <= 7 * 24 * 60 * 60 * 1000:
                elapsed_ms = delta_ms

        try:
            normalized_transitions = max(0, int(transitions))
        except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
            normalized_transitions = 0

        reached_target = 1 if ("runs" in tab_set or "outputs" in tab_set) else 0

        self.backend.execute(
            """
            INSERT INTO watchlist_ia_experiment_events
            (
                user_id,
                variant,
                session_id,
                previous_tab,
                current_tab,
                transitions,
                visited_tabs_json,
                first_seen_at,
                last_seen_at,
                elapsed_ms,
                reached_target,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.user_id,
                normalized_variant,
                normalized_session[:128],
                normalized_previous_tab,
                normalized_current_tab,
                normalized_transitions,
                json.dumps(normalized_visited_tabs, ensure_ascii=False),
                parsed_first.isoformat() if parsed_first else None,
                parsed_last.isoformat() if parsed_last else None,
                elapsed_ms,
                reached_target,
                now_iso,
            ),
        )
        return True

    def summarize_ia_experiment_events(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        since_iso = None
        until_iso = None
        parsed_since = self._parse_iso_utc(since)
        parsed_until = self._parse_iso_utc(until)
        if parsed_since:
            since_iso = parsed_since.isoformat()
        if parsed_until:
            until_iso = parsed_until.isoformat()

        sql = (
            "SELECT variant, session_id, transitions, visited_tabs_json, elapsed_ms, "
            "reached_target, created_at "
            "FROM watchlist_ia_experiment_events WHERE user_id = ?"
        )
        params: list[Any] = [self.user_id]
        if since_iso:
            sql += " AND created_at >= ?"
            params.append(since_iso)
        if until_iso:
            sql += " AND created_at <= ?"
            params.append(until_iso)
        sql += " ORDER BY created_at ASC"
        rows = self.backend.execute(sql, tuple(params)).rows

        events_by_variant: dict[str, int] = {"baseline": 0, "experimental": 0}
        latest_by_session: dict[tuple[str, str], dict[str, Any]] = {}

        for row in rows or []:
            variant = "experimental" if str(row.get("variant") or "").strip().lower() == "experimental" else "baseline"
            events_by_variant[variant] += 1

            session_id = str(row.get("session_id") or "").strip()
            if not session_id:
                continue
            latest_by_session[(variant, session_id)] = row

        session_rows_by_variant: dict[str, list[dict[str, Any]]] = {
            "baseline": [],
            "experimental": [],
        }
        for (variant, _session_id), row in latest_by_session.items():
            session_rows_by_variant[variant].append(row)

        summaries: list[dict[str, Any]] = []
        for variant in ("baseline", "experimental"):
            session_rows = session_rows_by_variant[variant]
            sessions = len(session_rows)
            reached_target_sessions = 0
            transitions_sum = 0.0
            visited_tabs_sum = 0.0
            session_seconds_sum = 0.0
            session_seconds_count = 0

            for row in session_rows:
                try:
                    transitions_sum += float(int(row.get("transitions") or 0))
                except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                    transitions_sum += 0.0

                visited_count = 0
                raw_tabs = row.get("visited_tabs_json")
                if isinstance(raw_tabs, str) and raw_tabs.strip():
                    try:
                        parsed_tabs = json.loads(raw_tabs)
                        if isinstance(parsed_tabs, list):
                            visited_count = len(
                                [tab for tab in parsed_tabs if isinstance(tab, str) and tab.strip()]
                            )
                    except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                        visited_count = 0
                visited_tabs_sum += float(visited_count)

                try:
                    reached_target_sessions += 1 if int(row.get("reached_target") or 0) > 0 else 0
                except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                    reached_target_sessions += 0

                try:
                    elapsed_ms = row.get("elapsed_ms")
                    if elapsed_ms is not None:
                        elapsed_ms_int = int(elapsed_ms)
                        if elapsed_ms_int >= 0:
                            session_seconds_sum += float(elapsed_ms_int) / 1000.0
                            session_seconds_count += 1
                except _WATCHLISTS_DB_NONCRITICAL_EXCEPTIONS:
                    pass

            summaries.append(
                {
                    "variant": variant,
                    "events": int(events_by_variant.get(variant, 0)),
                    "sessions": sessions,
                    "reached_target_sessions": reached_target_sessions,
                    "avg_transitions": (transitions_sum / sessions) if sessions > 0 else 0.0,
                    "avg_visited_tabs": (visited_tabs_sum / sessions) if sessions > 0 else 0.0,
                    "avg_session_seconds": (
                        session_seconds_sum / session_seconds_count
                        if session_seconds_count > 0
                        else 0.0
                    ),
                }
            )

        return summaries

    # ------------------------
    # WebSub subscriptions
    # ------------------------
    def _row_to_websub(self, row: dict[str, Any]) -> WebSubRow:
        return WebSubRow(
            id=int(row["id"]),
            user_id=str(row["user_id"]),
            source_id=int(row["source_id"]),
            hub_url=str(row["hub_url"]),
            topic_url=str(row["topic_url"]),
            callback_token=str(row["callback_token"]),
            secret=str(row["secret"]),
            state=str(row.get("state") or "pending"),
            lease_seconds=int(row["lease_seconds"]) if row.get("lease_seconds") is not None else None,
            verified_at=row.get("verified_at"),
            expires_at=row.get("expires_at"),
            last_push_at=row.get("last_push_at"),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def create_websub_subscription(
        self,
        *,
        source_id: int,
        hub_url: str,
        topic_url: str,
        callback_token: str,
        secret: str,
        lease_seconds: int | None = None,
    ) -> WebSubRow:
        now = _utcnow_iso()
        res = self._execute_insert(
            "INSERT INTO feed_websub_subscriptions "
            "(user_id, source_id, hub_url, topic_url, callback_token, secret, state, lease_seconds, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self.user_id, source_id, hub_url, topic_url, callback_token, secret, "pending", lease_seconds, now, now),
        )
        sid = self._extract_lastrowid(res)
        if sid is None:
            raise RuntimeError("failed_to_create_websub_subscription")
        return self.get_websub_subscription(sid)

    def get_websub_subscription(self, sub_id: int) -> WebSubRow:
        row = self.backend.execute(
            "SELECT * FROM feed_websub_subscriptions WHERE id = ? AND user_id = ?",
            (sub_id, self.user_id),
        ).first
        if not row:
            raise KeyError("websub_subscription_not_found")
        return self._row_to_websub(row)

    def get_websub_subscription_by_token(self, callback_token: str) -> WebSubRow | None:
        row = self.backend.execute(
            "SELECT * FROM feed_websub_subscriptions WHERE callback_token = ?",
            (callback_token,),
        ).first
        if not row:
            return None
        return self._row_to_websub(row)

    def get_websub_subscription_for_source(self, source_id: int) -> WebSubRow | None:
        row = self.backend.execute(
            "SELECT * FROM feed_websub_subscriptions WHERE source_id = ? AND user_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (source_id, self.user_id),
        ).first
        if not row:
            return None
        return self._row_to_websub(row)

    def update_websub_subscription(self, sub_id: int, patch: dict[str, Any]) -> WebSubRow:
        allowed = {"state", "lease_seconds", "verified_at", "expires_at", "last_push_at"}
        fields: list[str] = []
        params: list[Any] = []
        for key in allowed:
            if key in patch:
                fields.append(f"{key} = ?")
                params.append(patch[key])
        if not fields:
            return self.get_websub_subscription(sub_id)
        fields.append("updated_at = ?")
        params.append(_utcnow_iso())
        params.extend([sub_id, self.user_id])
        self.backend.execute(
            f"UPDATE feed_websub_subscriptions SET {', '.join(fields)} WHERE id = ? AND user_id = ?",  # nosec B608
            tuple(params),
        )
        return self.get_websub_subscription(sub_id)

    def delete_websub_subscription(self, sub_id: int) -> None:
        self.backend.execute(
            "DELETE FROM feed_websub_subscriptions WHERE id = ? AND user_id = ?",
            (sub_id, self.user_id),
        )

    def list_expiring_websub_subscriptions(self, before_iso: str) -> list[WebSubRow]:
        rows = self.backend.execute(
            "SELECT * FROM feed_websub_subscriptions "
            "WHERE user_id = ? AND state = 'verified' AND expires_at IS NOT NULL AND expires_at <= ?",
            (self.user_id, before_iso),
        ).rows
        return [self._row_to_websub(r) for r in (rows or [])]

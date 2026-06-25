"""SlidesDatabase: per-user SQLite storage for presentations."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection,
)


class SlidesDatabaseError(Exception):
    """Base exception for SlidesDatabase."""


class SchemaError(SlidesDatabaseError):
    """Raised when schema migrations fail."""


class ConflictError(SlidesDatabaseError):
    """Raised when optimistic locking fails or duplicates exist."""

    def __init__(self, message: str, *, entity: str | None = None, identifier: str | None = None):
        super().__init__(message)
        self.entity = entity
        self.identifier = identifier


class InputError(ValueError):
    """Raised for invalid inputs."""


@dataclass
class PresentationRow:
    id: str
    title: str
    description: str | None
    theme: str
    marp_theme: str | None
    template_id: str | None
    visual_style_id: str | None
    visual_style_scope: str | None
    visual_style_name: str | None
    visual_style_version: int | None
    visual_style_snapshot: str | None
    settings: str | None
    studio_data: str | None
    slides: str
    slides_text: str
    source_type: str | None
    source_ref: str | None
    source_query: str | None
    custom_css: str | None
    created_at: str
    last_modified: str
    deleted: int
    client_id: str
    version: int


@dataclass
class PresentationVersionRow:
    presentation_id: str
    version: int
    payload_json: str
    created_at: str
    client_id: str


@dataclass
class VisualStyleRow:
    id: str
    name: str
    scope: str
    style_payload: str
    created_at: str
    updated_at: str


class SlidesDatabase:
    _SCHEMA_VERSION = 1
    _schema_init_paths: ClassVar[set[str]] = set()
    _schema_init_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, db_path: str | Path, client_id: str) -> None:
        if not client_id:
            raise ValueError("client_id is required")
        self.client_id = str(client_id)
        if isinstance(db_path, Path):
            self.db_path = db_path.resolve()
            self._db_path_str = str(self.db_path)
        else:
            self._db_path_str = str(db_path)
            self.db_path = Path(self._db_path_str).resolve() if self._db_path_str != ":memory:" else Path(":memory:")
        self._local = threading.local()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cache_schema_path = self._db_path_str != ":memory:"
        if cache_schema_path and self._db_path_str in self._schema_init_paths:
            return
        with self._schema_init_lock:
            if cache_schema_path and self._db_path_str in self._schema_init_paths:
                return
            conn = self.get_connection()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY NOT NULL
                    );
                    INSERT OR IGNORE INTO schema_version (version) VALUES (0);

                    CREATE TABLE IF NOT EXISTS presentations (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        description TEXT,
                        theme TEXT DEFAULT 'black',
                        marp_theme TEXT,
                        template_id TEXT,
                        visual_style_id TEXT,
                        visual_style_scope TEXT,
                        visual_style_name TEXT,
                        visual_style_version INTEGER,
                        visual_style_snapshot TEXT,
                        settings TEXT,
                        studio_data TEXT,
                        slides TEXT NOT NULL,
                        slides_text TEXT NOT NULL,
                        source_type TEXT,
                        source_ref TEXT,
                        source_query TEXT,
                        custom_css TEXT,
                        created_at DATETIME NOT NULL,
                        last_modified DATETIME NOT NULL,
                        deleted INTEGER DEFAULT 0,
                        client_id TEXT NOT NULL,
                        version INTEGER DEFAULT 1
                    );

                    CREATE INDEX IF NOT EXISTS idx_presentations_deleted ON presentations(deleted);
                    CREATE INDEX IF NOT EXISTS idx_presentations_created ON presentations(created_at);

                    CREATE TABLE IF NOT EXISTS presentations_versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        presentation_id TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        client_id TEXT NOT NULL
                    );

                    CREATE UNIQUE INDEX IF NOT EXISTS idx_presentations_versions_unique
                        ON presentations_versions(presentation_id, version);
                    CREATE INDEX IF NOT EXISTS idx_presentations_versions_pid
                        ON presentations_versions(presentation_id);
                    CREATE INDEX IF NOT EXISTS idx_presentations_versions_created
                        ON presentations_versions(created_at);

                    CREATE VIRTUAL TABLE IF NOT EXISTS presentations_fts USING fts5(
                        title,
                        slides_text,
                        content=presentations,
                        content_rowid=rowid
                    );

                    CREATE TRIGGER IF NOT EXISTS presentations_ai AFTER INSERT ON presentations BEGIN
                      INSERT INTO presentations_fts(rowid, title, slides_text)
                      VALUES (new.rowid, new.title, new.slides_text);
                    END;

                    CREATE TRIGGER IF NOT EXISTS presentations_ad AFTER DELETE ON presentations BEGIN
                      INSERT INTO presentations_fts(presentations_fts, rowid, title, slides_text)
                      VALUES ('delete', old.rowid, old.title, old.slides_text);
                    END;

                    CREATE TRIGGER IF NOT EXISTS presentations_au AFTER UPDATE ON presentations BEGIN
                      INSERT INTO presentations_fts(presentations_fts, rowid, title, slides_text)
                      VALUES ('delete', old.rowid, old.title, old.slides_text);
                      INSERT INTO presentations_fts(rowid, title, slides_text)
                      VALUES (new.rowid, new.title, new.slides_text);
                    END;

                    CREATE TABLE IF NOT EXISTS sync_log (
                        change_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entity TEXT NOT NULL,
                        entity_uuid TEXT NOT NULL,
                        operation TEXT NOT NULL CHECK(operation IN ('create','update','delete','restore')),
                        timestamp DATETIME NOT NULL,
                        client_id TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        payload TEXT
                    );

                    CREATE INDEX IF NOT EXISTS idx_sync_log_ts ON sync_log(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_sync_log_entity_uuid ON sync_log(entity_uuid);
                    CREATE INDEX IF NOT EXISTS idx_sync_log_client_id ON sync_log(client_id);

                    CREATE TABLE IF NOT EXISTS visual_styles (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        style_payload TEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_visual_styles_scope ON visual_styles(scope);
                    CREATE INDEX IF NOT EXISTS idx_visual_styles_name ON visual_styles(name);
                    """
                )
                self._ensure_marp_theme_column(conn)
                self._ensure_template_id_column(conn)
                self._ensure_presentation_visual_style_columns(conn)
                self._ensure_studio_data_column(conn)
                self._ensure_visual_styles_table(conn)
                conn.commit()
                if cache_schema_path:
                    self._schema_init_paths.add(self._db_path_str)
            except sqlite3.Error as exc:
                conn.rollback()
                raise SchemaError(f"Failed to initialize Slides DB schema: {exc}") from exc

    def get_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "connection", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path_str, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            configure_sqlite_connection(conn)
            self._local.connection = conn
        return conn

    def close_connection(self) -> None:
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            conn.close()
            self._local.connection = None

    @contextmanager
    def transaction(self) -> Iterable[sqlite3.Connection]:
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def _utcnow_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _insert_sync_log(
        self,
        conn: sqlite3.Connection,
        /,
        *,
        entity_uuid: str,
        operation: str,
        version: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        payload_json = json.dumps(payload) if payload is not None else None
        conn.execute(
            """
            INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "presentations",
                entity_uuid,
                operation,
                self._utcnow_iso(),
                self.client_id,
                version,
                payload_json,
            )
        )

    @staticmethod
    def _ensure_marp_theme_column(conn: sqlite3.Connection) -> None:
        columns = conn.execute("PRAGMA table_info(presentations)").fetchall()
        if any(col["name"] == "marp_theme" for col in columns):
            return
        conn.execute("ALTER TABLE presentations ADD COLUMN marp_theme TEXT")

    @staticmethod
    def _ensure_template_id_column(conn: sqlite3.Connection) -> None:
        columns = conn.execute("PRAGMA table_info(presentations)").fetchall()
        if any(col["name"] == "template_id" for col in columns):
            return
        conn.execute("ALTER TABLE presentations ADD COLUMN template_id TEXT")

    @staticmethod
    def _ensure_studio_data_column(conn: sqlite3.Connection) -> None:
        columns = conn.execute("PRAGMA table_info(presentations)").fetchall()
        if any(col["name"] == "studio_data" for col in columns):
            return
        conn.execute("ALTER TABLE presentations ADD COLUMN studio_data TEXT")

    @staticmethod
    def _ensure_presentation_visual_style_columns(conn: sqlite3.Connection) -> None:
        columns = {col["name"] for col in conn.execute("PRAGMA table_info(presentations)").fetchall()}
        if "visual_style_id" not in columns:
            conn.execute("ALTER TABLE presentations ADD COLUMN visual_style_id TEXT")
        if "visual_style_scope" not in columns:
            conn.execute("ALTER TABLE presentations ADD COLUMN visual_style_scope TEXT")
        if "visual_style_name" not in columns:
            conn.execute("ALTER TABLE presentations ADD COLUMN visual_style_name TEXT")
        if "visual_style_version" not in columns:
            conn.execute("ALTER TABLE presentations ADD COLUMN visual_style_version INTEGER")
        if "visual_style_snapshot" not in columns:
            conn.execute("ALTER TABLE presentations ADD COLUMN visual_style_snapshot TEXT")

    @staticmethod
    def _ensure_visual_styles_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS visual_styles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                scope TEXT NOT NULL,
                style_payload TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_visual_styles_scope ON visual_styles(scope)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_visual_styles_name ON visual_styles(name)"
        )

    @staticmethod
    def _fetch_presentation_by_id(
        conn: sqlite3.Connection, presentation_id: str, include_deleted: bool
    ) -> PresentationRow:
        query = "SELECT * FROM presentations WHERE id = ?"
        params: list[Any] = [presentation_id]
        if not include_deleted:
            query += " AND deleted = 0"
        row = conn.execute(query, tuple(params)).fetchone()
        if not row:
            raise KeyError("presentation_not_found")
        return PresentationRow(**dict(row))

    @staticmethod
    def _is_fts_query_error(exc: sqlite3.OperationalError) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "fts5: syntax error",
                "fts5 syntax error",
                "malformed match",
                "no such column:",
                "unterminated string",
                "syntax error near",
            )
        )

    @staticmethod
    def _build_version_payload(row: PresentationRow) -> dict[str, Any]:
        return {
            "id": row.id,
            "title": row.title,
            "description": row.description,
            "theme": row.theme,
            "marp_theme": row.marp_theme,
            "template_id": row.template_id,
            "visual_style_id": row.visual_style_id,
            "visual_style_scope": row.visual_style_scope,
            "visual_style_name": row.visual_style_name,
            "visual_style_version": row.visual_style_version,
            "visual_style_snapshot": row.visual_style_snapshot,
            "settings": row.settings,
            "studio_data": row.studio_data,
            "slides": row.slides,
            "custom_css": row.custom_css,
            "source_type": row.source_type,
            "source_ref": row.source_ref,
            "source_query": row.source_query,
            "created_at": row.created_at,
            "last_modified": row.last_modified,
            "deleted": int(row.deleted or 0),
            "client_id": row.client_id,
            "version": int(row.version),
        }

    def _insert_version_snapshot(self, conn: sqlite3.Connection, row: PresentationRow) -> None:
        payload_json = json.dumps(self._build_version_payload(row), ensure_ascii=True)
        conn.execute(
            """
            INSERT INTO presentations_versions (
                presentation_id, version, payload_json, created_at, client_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                row.id,
                int(row.version),
                payload_json,
                row.last_modified,
                row.client_id,
            ),
        )

    @staticmethod
    def _normalize_visual_style_payload(style_payload: str) -> str:
        if not style_payload:
            raise InputError("style_payload is required")
        try:
            parsed = json.loads(style_payload)
        except json.JSONDecodeError as exc:
            raise InputError("style_payload must be valid JSON") from exc
        return json.dumps(parsed, ensure_ascii=True, sort_keys=True)

    @staticmethod
    def _validate_visual_style_scope(scope: str) -> str:
        if scope not in {"builtin", "user"}:
            raise InputError("scope must be one of: builtin, user")
        return scope

    @staticmethod
    def _fetch_visual_style_by_id(conn: sqlite3.Connection, style_id: str) -> VisualStyleRow:
        row = conn.execute("SELECT * FROM visual_styles WHERE id = ?", (style_id,)).fetchone()
        if not row:
            raise KeyError("visual_style_not_found")
        return VisualStyleRow(**dict(row))

    def create_visual_style(
        self,
        *,
        name: str,
        scope: str,
        style_payload: str,
        style_id: str | None = None,
    ) -> VisualStyleRow:
        """Create and persist a visual style for the current user."""
        if not name:
            raise InputError("name is required")
        resolved_scope = self._validate_visual_style_scope(scope)
        normalized_payload = self._normalize_visual_style_payload(style_payload)
        resolved_style_id = style_id or str(uuid.uuid4())
        now = self._utcnow_iso()
        try:
            with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO visual_styles (id, name, scope, style_payload, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resolved_style_id,
                        name,
                        resolved_scope,
                        normalized_payload,
                        now,
                        now,
                    ),
                )
                return self._fetch_visual_style_by_id(conn, resolved_style_id)
        except sqlite3.IntegrityError as exc:
            if "UNIQUE" in str(exc).upper() or "PRIMARY" in str(exc).upper():
                raise ConflictError(
                    "visual style already exists",
                    entity="visual_styles",
                    identifier=resolved_style_id,
                ) from exc
            raise SlidesDatabaseError(f"Failed to create visual style: {exc}") from exc

    def get_visual_style_by_id(self, style_id: str) -> VisualStyleRow:
        """Fetch a single visual style by identifier."""
        conn = self.get_connection()
        return self._fetch_visual_style_by_id(conn, style_id)

    def count_visual_styles(self) -> int:
        """Return the number of persisted user visual styles."""

        conn = self.get_connection()
        count_row = conn.execute("SELECT COUNT(*) AS cnt FROM visual_styles").fetchone()
        return int(count_row["cnt"]) if count_row else 0

    def list_visual_styles(self, *, limit: int, offset: int) -> tuple[list[VisualStyleRow], int]:
        """List persisted user visual styles with pagination metadata."""
        if limit < 1:
            raise InputError("limit must be >= 1")
        if offset < 0:
            raise InputError("offset must be >= 0")
        conn = self.get_connection()
        rows = conn.execute(
            """
            SELECT * FROM visual_styles
            ORDER BY updated_at DESC, created_at DESC, name ASC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        count_row = conn.execute("SELECT COUNT(*) AS cnt FROM visual_styles").fetchone()
        total = int(count_row["cnt"]) if count_row else 0
        return [VisualStyleRow(**dict(row)) for row in rows], total

    def update_visual_style(
        self,
        *,
        style_id: str,
        name: str,
        style_payload: str,
        expected_updated_at: str,
    ) -> VisualStyleRow:
        """Update a stored visual style and return the refreshed row."""
        if not name:
            raise InputError("name is required")
        normalized_payload = self._normalize_visual_style_payload(style_payload)
        if not expected_updated_at:
            raise InputError("expected_updated_at is required")
        with self.transaction() as conn:
            cur = conn.execute(
                """
                UPDATE visual_styles
                SET name = ?, style_payload = ?, updated_at = ?
                WHERE id = ? AND updated_at = ?
                """,
                (name, normalized_payload, self._utcnow_iso(), style_id, expected_updated_at),
            )
            if cur.rowcount == 0:
                current_row = conn.execute(
                    "SELECT updated_at FROM visual_styles WHERE id = ?",
                    (style_id,),
                ).fetchone()
                if not current_row:
                    raise KeyError("visual_style_not_found")
                raise ConflictError(
                    "visual style update conflicted with a newer revision",
                    entity="visual_styles",
                    identifier=style_id,
                )
            return self._fetch_visual_style_by_id(conn, style_id)

    def delete_visual_style(self, style_id: str) -> bool:
        """Delete a stored visual style by identifier."""
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT 1 FROM visual_styles WHERE id = ?",
                (style_id,),
            ).fetchone()
            if not existing:
                return False
            in_use = conn.execute(
                """
                SELECT 1
                FROM presentations
                WHERE visual_style_id = ?
                  AND deleted = 0
                LIMIT 1
                """,
                (style_id,),
            ).fetchone()
            if in_use:
                raise ConflictError(
                    "visual style is still referenced by presentations",
                    entity="visual_styles",
                    identifier=style_id,
                )
            cur = conn.execute("DELETE FROM visual_styles WHERE id = ?", (style_id,))
            return cur.rowcount > 0

    def create_presentation(
        self,
        *,
        presentation_id: str | None,
        title: str,
        description: str | None,
        theme: str,
        marp_theme: str | None,
        settings: str | None,
        studio_data: str | None,
        template_id: str | None = None,
        visual_style_id: str | None = None,
        visual_style_scope: str | None = None,
        visual_style_name: str | None = None,
        visual_style_version: int | None = None,
        visual_style_snapshot: str | None = None,
        slides: str,
        slides_text: str,
        source_type: str | None,
        source_ref: str | None,
        source_query: str | None,
        custom_css: str | None,
    ) -> PresentationRow:
        if not title:
            raise InputError("title is required")
        pres_id = presentation_id or str(uuid.uuid4())
        now = self._utcnow_iso()
        try:
            with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO presentations (
                        id, title, description, theme, marp_theme, template_id,
                        visual_style_id, visual_style_scope, visual_style_name, visual_style_version, visual_style_snapshot,
                        settings, studio_data, slides, slides_text,
                        source_type, source_ref, source_query, custom_css,
                        created_at, last_modified, deleted, client_id, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 1)
                    """,
                    (
                        pres_id,
                        title,
                        description,
                        theme,
                        marp_theme,
                        template_id,
                        visual_style_id,
                        visual_style_scope,
                        visual_style_name,
                        visual_style_version,
                        visual_style_snapshot,
                        settings,
                        studio_data,
                        slides,
                        slides_text,
                        source_type,
                        source_ref,
                        source_query,
                        custom_css,
                        now,
                        now,
                        self.client_id,
                    ),
                )
                row = self._fetch_presentation_by_id(conn, pres_id, include_deleted=True)
                self._insert_version_snapshot(conn, row)
                self._insert_sync_log(
                    conn,
                    entity_uuid=pres_id,
                    operation="create",
                    version=1,
                    payload={"title": title, "theme": theme},
                )
            return row
        except sqlite3.IntegrityError as exc:
            if "UNIQUE" in str(exc).upper() or "PRIMARY" in str(exc).upper():
                raise ConflictError("presentation already exists", entity="presentations", identifier=pres_id) from exc
            raise SlidesDatabaseError(f"Failed to create presentation: {exc}") from exc

    def get_presentation_by_id(self, presentation_id: str, *, include_deleted: bool = False) -> PresentationRow:
        conn = self.get_connection()
        return self._fetch_presentation_by_id(conn, presentation_id, include_deleted)

    def list_presentations(
        self,
        *,
        limit: int,
        offset: int,
        include_deleted: bool,
        sort_column: str,
        sort_direction: str,
    ) -> tuple[list[PresentationRow], int]:
        if limit < 1:
            raise InputError("limit must be >= 1")
        allowed_columns = {
            "created_at": "created_at",
            "last_modified": "last_modified",
            "title": "title",
        }
        safe_column = allowed_columns.get(sort_column, "created_at")
        safe_direction = "DESC" if sort_direction.upper() == "DESC" else "ASC"
        where = "" if include_deleted else "WHERE deleted = 0"
        query_template = "SELECT * FROM presentations {where} ORDER BY {safe_column} {safe_direction} LIMIT ? OFFSET ?"
        query = query_template.format_map(locals())  # nosec B608
        count_query_template = "SELECT COUNT(*) AS cnt FROM presentations {where}"
        count_query = count_query_template.format_map(locals())  # nosec B608
        conn = self.get_connection()
        rows = conn.execute(query, (limit, offset)).fetchall()
        count_row = conn.execute(count_query).fetchone()
        total = int(count_row["cnt"]) if count_row else 0
        return [PresentationRow(**dict(row)) for row in rows], total

    def search_presentations(
        self,
        *,
        query: str,
        limit: int,
        offset: int,
        include_deleted: bool,
    ) -> tuple[list[PresentationRow], int]:
        if not query:
            raise InputError("query is required")
        if limit < 1:
            raise InputError("limit must be >= 1")
        where = "" if include_deleted else "AND p.deleted = 0"
        search_sql_template = (
            "SELECT p.* FROM presentations p "
            "JOIN presentations_fts fts ON p.rowid = fts.rowid "
            "WHERE presentations_fts MATCH ? "
            "{where} "
            "ORDER BY p.last_modified DESC LIMIT ? OFFSET ?"
        )
        sql = search_sql_template.format_map(locals())  # nosec B608
        count_sql_template = (
            "SELECT COUNT(*) AS cnt FROM presentations p "
            "JOIN presentations_fts fts ON p.rowid = fts.rowid "
            "WHERE presentations_fts MATCH ? "
            "{where}"
        )
        count_sql = count_sql_template.format_map(locals())  # nosec B608
        conn = self.get_connection()
        try:
            rows = conn.execute(sql, (query, limit, offset)).fetchall()
            count_row = conn.execute(count_sql, (query,)).fetchone()
        except sqlite3.OperationalError as exc:
            if not self._is_fts_query_error(exc):
                raise
            raise InputError("search query is invalid") from exc
        total = int(count_row["cnt"]) if count_row else 0
        return [PresentationRow(**dict(row)) for row in rows], total

    def update_presentation(
        self,
        *,
        presentation_id: str,
        update_fields: dict[str, Any],
        expected_version: int,
        operation: str = "update",
    ) -> PresentationRow:
        if not update_fields:
            raise InputError("update_fields is required")
        allowed = {
            "title",
            "description",
            "theme",
            "marp_theme",
            "template_id",
            "visual_style_id",
            "visual_style_scope",
            "visual_style_name",
            "visual_style_version",
            "visual_style_snapshot",
            "settings",
            "studio_data",
            "slides",
            "slides_text",
            "source_type",
            "source_ref",
            "source_query",
            "custom_css",
            "deleted",
        }
        sets: list[str] = []
        params: list[Any] = []
        for key, value in update_fields.items():
            if key not in allowed:
                continue
            sets.append(f"{key} = ?")
            params.append(value)
        if not sets:
            raise InputError("no valid fields to update")
        next_version = expected_version + 1
        sets.extend(["last_modified = ?", "version = ?", "client_id = ?"])
        params.extend([self._utcnow_iso(), next_version, self.client_id, presentation_id, expected_version])
        # Column names come from the allowlist above; values are always bound params.
        set_clause_sql = ", ".join(sets)
        update_sql_template = "UPDATE presentations SET {set_clause_sql} WHERE id = ? AND version = ?"
        sql = update_sql_template.format_map(locals())  # nosec B608
        with self.transaction() as conn:
            cur = conn.execute(sql, tuple(params))
            if cur.rowcount == 0:
                existing = conn.execute("SELECT version FROM presentations WHERE id = ?", (presentation_id,)).fetchone()
                if not existing:
                    raise KeyError("presentation_not_found")
                raise ConflictError("version_conflict", entity="presentations", identifier=presentation_id)
            row = self._fetch_presentation_by_id(conn, presentation_id, include_deleted=True)
            self._insert_version_snapshot(conn, row)
            self._insert_sync_log(
                conn,
                entity_uuid=presentation_id,
                operation=operation,
                version=next_version,
                payload={"fields": list(update_fields.keys())},
            )
        return row

    def list_presentation_versions(
        self,
        *,
        presentation_id: str,
        limit: int,
        offset: int,
    ) -> tuple[list[PresentationVersionRow], int]:
        if limit < 1:
            raise InputError("limit must be >= 1")
        conn = self.get_connection()
        rows = conn.execute(
            """
            SELECT presentation_id, version, payload_json, created_at, client_id
            FROM presentations_versions
            WHERE presentation_id = ?
            ORDER BY version DESC
            LIMIT ? OFFSET ?
            """,
            (presentation_id, limit, offset),
        ).fetchall()
        count_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM presentations_versions WHERE presentation_id = ?",
            (presentation_id,),
        ).fetchone()
        total = int(count_row["cnt"]) if count_row else 0
        return [PresentationVersionRow(**dict(row)) for row in rows], total

    def get_presentation_version(
        self,
        *,
        presentation_id: str,
        version: int,
    ) -> PresentationVersionRow:
        conn = self.get_connection()
        row = conn.execute(
            """
            SELECT presentation_id, version, payload_json, created_at, client_id
            FROM presentations_versions
            WHERE presentation_id = ? AND version = ?
            """,
            (presentation_id, version),
        ).fetchone()
        if not row:
            raise KeyError("presentation_version_not_found")
        return PresentationVersionRow(**dict(row))

    def soft_delete_presentation(self, presentation_id: str, expected_version: int) -> PresentationRow:
        return self.update_presentation(
            presentation_id=presentation_id,
            update_fields={"deleted": 1},
            expected_version=expected_version,
            operation="delete",
        )

    def restore_presentation(self, presentation_id: str, expected_version: int) -> PresentationRow:
        return self.update_presentation(
            presentation_id=presentation_id,
            update_fields={"deleted": 0},
            expected_version=expected_version,
            operation="restore",
        )

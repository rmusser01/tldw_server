"""Visual identity expression pack metadata storage for ChaChaNotes."""

from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

IDEMPOTENCY_IN_PROGRESS_STALE_AFTER_SECONDS = 15 * 60

VISUAL_IDENTITY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS visual_identity_packs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'archived', 'deleted')),
    active_version_id INTEGER REFERENCES visual_identity_pack_versions(id),
    default_expression_key TEXT NOT NULL DEFAULT 'neutral',
    source_kind TEXT NOT NULL DEFAULT 'manual',
    source_context_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS visual_identity_pack_drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    pack_id INTEGER REFERENCES visual_identity_packs(id),
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'importing'
        CHECK(status IN ('importing', 'ready_for_review', 'failed', 'abandoned', 'activated')),
    source_kind TEXT NOT NULL,
    source_filename TEXT NOT NULL DEFAULT '',
    import_job_id TEXT,
    validation_summary_json TEXT NOT NULL DEFAULT '{}',
    slot_map_json TEXT NOT NULL DEFAULT '{}',
    default_expression_key TEXT NOT NULL DEFAULT 'neutral',
    error_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS visual_identity_pack_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pack_id INTEGER NOT NULL REFERENCES visual_identity_packs(id),
    owner_user_id INTEGER NOT NULL,
    version_number INTEGER NOT NULL,
    default_expression_key TEXT NOT NULL DEFAULT 'neutral',
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pack_id, version_number)
);

CREATE TABLE IF NOT EXISTS visual_identity_assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    pack_id INTEGER REFERENCES visual_identity_packs(id),
    draft_id INTEGER REFERENCES visual_identity_pack_drafts(id),
    pack_version_id INTEGER REFERENCES visual_identity_pack_versions(id),
    expression_key TEXT NOT NULL,
    original_expression_key TEXT NOT NULL DEFAULT '',
    display_label TEXT NOT NULL DEFAULT '',
    source_filename TEXT NOT NULL,
    storage_relpath TEXT NOT NULL,
    content_type TEXT NOT NULL,
    bytes INTEGER NOT NULL CHECK(bytes > 0),
    sha256 TEXT NOT NULL,
    width INTEGER NOT NULL CHECK(width > 0),
    height INTEGER NOT NULL CHECK(height > 0),
    source_context_json TEXT NOT NULL DEFAULT '{}',
    is_animated INTEGER NOT NULL DEFAULT 0 CHECK(is_animated IN (0, 1)),
    frame_count INTEGER,
    duration_ms INTEGER,
    preview_relpath TEXT,
    deleted INTEGER NOT NULL DEFAULT 0 CHECK(deleted IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(draft_id IS NOT NULL OR pack_version_id IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS visual_identity_bindings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    actor_kind TEXT NOT NULL CHECK(actor_kind IN ('character', 'persona')),
    actor_id TEXT NOT NULL,
    pack_id INTEGER NOT NULL REFERENCES visual_identity_packs(id),
    active_version_id INTEGER NOT NULL REFERENCES visual_identity_pack_versions(id),
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'deleted')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS visual_identity_idempotency (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    scope TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    claim_token TEXT,
    status TEXT NOT NULL,
    response_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(owner_user_id, scope, resource_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_visual_identity_packs_owner_status
    ON visual_identity_packs(owner_user_id, status);
CREATE INDEX IF NOT EXISTS idx_visual_identity_drafts_owner_status
    ON visual_identity_pack_drafts(owner_user_id, status);
CREATE INDEX IF NOT EXISTS idx_visual_identity_assets_pack_expression
    ON visual_identity_assets(pack_id, pack_version_id, expression_key, deleted);
CREATE INDEX IF NOT EXISTS idx_visual_identity_assets_draft_expression
    ON visual_identity_assets(draft_id, expression_key, deleted);
CREATE UNIQUE INDEX IF NOT EXISTS idx_visual_identity_bindings_actor_active
    ON visual_identity_bindings(owner_user_id, actor_kind, actor_id)
    WHERE status = 'active';
"""

VISUAL_IDENTITY_SCHEMA_STATEMENTS = tuple(
    statement.strip()
    for statement in VISUAL_IDENTITY_SCHEMA_SQL.split(";")
    if statement.strip()
)


def ensure_visual_identity_tables(db: CharactersRAGDB) -> None:
    """Create visual identity tables in a SQLite ChaChaNotes database."""
    _require_sqlite_chacha_db(db)
    conn = db.get_connection()
    if not conn.in_transaction:
        conn.execute("PRAGMA foreign_keys = ON")
    with db.transaction() as conn:
        for statement in VISUAL_IDENTITY_SCHEMA_STATEMENTS:
            conn.execute(statement)
        _ensure_visual_identity_asset_columns(conn)
        _ensure_visual_identity_idempotency_columns(conn)


def _ensure_visual_identity_asset_columns(conn: sqlite3.Connection) -> None:
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(visual_identity_assets)").fetchall()
    }
    if "source_context_json" not in columns:
        conn.execute(
            "ALTER TABLE visual_identity_assets "
            "ADD COLUMN source_context_json TEXT NOT NULL DEFAULT '{}'"
        )


def _ensure_visual_identity_idempotency_columns(conn: sqlite3.Connection) -> None:
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(visual_identity_idempotency)").fetchall()
    }
    if "claim_token" not in columns:
        conn.execute("ALTER TABLE visual_identity_idempotency ADD COLUMN claim_token TEXT")


class VisualIdentityRepository:
    """Repository for visual identity expression pack metadata."""

    def __init__(self, db: CharactersRAGDB):
        _require_sqlite_chacha_db(db)
        self.db = db
        self._schema_initialized = False

    @classmethod
    def initialized(cls, db: CharactersRAGDB) -> VisualIdentityRepository:
        repo = cls(db)
        repo.initialize_schema()
        return repo

    def initialize_schema(self) -> None:
        ensure_visual_identity_tables(self.db)
        self._schema_initialized = True

    def create_pack(
        self,
        *,
        owner_user_id: int,
        title: str,
        description: str = "",
        status: str = "active",
        active_version_id: int | None = None,
        default_expression_key: str = "neutral",
        source_kind: str = "manual",
        source_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        if active_version_id is not None:
            raise ValueError("visual_identity_pack_version_not_found")
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO visual_identity_packs (
                    owner_user_id,
                    title,
                    description,
                    status,
                    active_version_id,
                    default_expression_key,
                    source_kind,
                    source_context_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    title,
                    description,
                    status,
                    active_version_id,
                    default_expression_key,
                    source_kind,
                    _json_dump(dict(source_context or {})),
                ),
            )
            pack_id = int(cursor.lastrowid)
        pack = self.get_pack(pack_id, owner_user_id=owner_user_id)
        if pack is None:
            raise RuntimeError("created_visual_identity_pack_not_found")
        return pack

    def get_pack(
        self,
        pack_id: int,
        *,
        owner_user_id: int | None = None,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT * FROM visual_identity_packs
            WHERE id = ?
              AND (? IS NULL OR owner_user_id = ?)
              AND (? = 1 OR status != 'deleted')
            """,
            (pack_id, owner_user_id, owner_user_id, int(include_deleted)),
        )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def list_packs(
        self,
        *,
        owner_user_id: int,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT * FROM visual_identity_packs
            WHERE owner_user_id = ?
              AND status != 'deleted'
              AND (? IS NULL OR status = ?)
            ORDER BY id ASC
            """,
            (owner_user_id, status, status),
        )
        return [dict(row) for row in cursor.fetchall()]

    def update_pack(
        self,
        *,
        pack_id: int,
        owner_user_id: int,
        fields: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        update_values = _update_values(fields, _PACK_UPDATE_STATEMENTS, {"source_context"})
        if not update_values:
            return self.get_pack(pack_id, owner_user_id=owner_user_id)

        with self.db.transaction() as conn:
            for statement, value in update_values:
                conn.execute(statement, (value, pack_id, owner_user_id))
            conn.execute(
                """
                UPDATE visual_identity_packs
                SET updated_at = CURRENT_TIMESTAMP,
                    version = version + 1
                WHERE id = ?
                  AND owner_user_id = ?
                  AND status != 'deleted'
                """,
                (pack_id, owner_user_id),
            )
        return self.get_pack(pack_id, owner_user_id=owner_user_id)

    def archive_pack(self, *, pack_id: int, owner_user_id: int) -> dict[str, Any]:
        pack = self.update_pack(
            pack_id=pack_id,
            owner_user_id=owner_user_id,
            fields={"status": "archived"},
        )
        if pack is None:
            raise ValueError("visual_identity_pack_not_found")
        return pack

    def mark_pack_deleted(self, *, pack_id: int, owner_user_id: int) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE visual_identity_packs
                SET status = 'deleted',
                    updated_at = CURRENT_TIMESTAMP,
                    version = version + 1
                WHERE id = ? AND owner_user_id = ?
                """,
                (pack_id, owner_user_id),
            )
        pack = self.get_pack(pack_id, owner_user_id=owner_user_id, include_deleted=True)
        if pack is None:
            raise ValueError("visual_identity_pack_not_found")
        return pack

    def create_draft(
        self,
        *,
        owner_user_id: int,
        title: str,
        source_kind: str,
        pack_id: int | None = None,
        status: str = "importing",
        source_filename: str = "",
        import_job_id: str | None = None,
        validation_summary: Mapping[str, Any] | None = None,
        slot_map: Mapping[str, Any] | None = None,
        default_expression_key: str = "neutral",
        error: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        if pack_id is not None and self.get_pack(pack_id, owner_user_id=owner_user_id) is None:
            raise ValueError("visual_identity_pack_not_found")
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO visual_identity_pack_drafts (
                    owner_user_id,
                    pack_id,
                    title,
                    status,
                    source_kind,
                    source_filename,
                    import_job_id,
                    validation_summary_json,
                    slot_map_json,
                    default_expression_key,
                    error_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    pack_id,
                    title,
                    status,
                    source_kind,
                    source_filename,
                    import_job_id,
                    _json_dump(dict(validation_summary or {})),
                    _json_dump(dict(slot_map or {})),
                    default_expression_key,
                    _json_dump(dict(error or {})),
                ),
            )
            draft_id = int(cursor.lastrowid)
        draft = self.get_draft(draft_id, owner_user_id=owner_user_id)
        if draft is None:
            raise RuntimeError("created_visual_identity_draft_not_found")
        return draft

    def get_draft(self, draft_id: int, *, owner_user_id: int | None = None) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        if owner_user_id is None:
            cursor = self.db.execute_query(
                "SELECT * FROM visual_identity_pack_drafts WHERE id = ?",
                (draft_id,),
            )
        else:
            cursor = self.db.execute_query(
                """
                SELECT * FROM visual_identity_pack_drafts
                WHERE id = ? AND owner_user_id = ?
                """,
                (draft_id, owner_user_id),
            )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def update_draft_slot_map(
        self,
        *,
        draft_id: int,
        owner_user_id: int,
        slot_map: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE visual_identity_pack_drafts
                SET slot_map_json = ?,
                    updated_at = CURRENT_TIMESTAMP,
                    version = version + 1
                WHERE id = ? AND owner_user_id = ?
                """,
                (_json_dump(dict(slot_map)), draft_id, owner_user_id),
            )
        draft = self.get_draft(draft_id, owner_user_id=owner_user_id)
        if draft is None:
            raise ValueError("visual_identity_draft_not_found")
        return draft

    def set_draft_import_job_id(
        self,
        *,
        draft_id: int,
        owner_user_id: int,
        import_job_id: str,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE visual_identity_pack_drafts
                SET import_job_id = ?,
                    updated_at = CURRENT_TIMESTAMP,
                    version = version + 1
                WHERE id = ? AND owner_user_id = ?
                """,
                (import_job_id, draft_id, owner_user_id),
            )
        draft = self.get_draft(draft_id, owner_user_id=owner_user_id)
        if draft is None:
            raise ValueError("visual_identity_draft_not_found")
        return draft

    def update_draft_validation_summary(
        self,
        *,
        draft_id: int,
        owner_user_id: int,
        validation_summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE visual_identity_pack_drafts
                SET validation_summary_json = ?,
                    updated_at = CURRENT_TIMESTAMP,
                    version = version + 1
                WHERE id = ? AND owner_user_id = ?
                """,
                (_json_dump(dict(validation_summary)), draft_id, owner_user_id),
            )
        draft = self.get_draft(draft_id, owner_user_id=owner_user_id)
        if draft is None:
            raise ValueError("visual_identity_draft_not_found")
        return draft

    def set_draft_status(
        self,
        *,
        draft_id: int,
        owner_user_id: int,
        status: str,
        error: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE visual_identity_pack_drafts
                SET status = ?,
                    error_json = COALESCE(?, error_json),
                    updated_at = CURRENT_TIMESTAMP,
                    version = version + 1
                WHERE id = ? AND owner_user_id = ?
                """,
                (
                    status,
                    None if error is None else _json_dump(dict(error)),
                    draft_id,
                    owner_user_id,
                ),
            )
        draft = self.get_draft(draft_id, owner_user_id=owner_user_id)
        if draft is None:
            raise ValueError("visual_identity_draft_not_found")
        return draft

    def list_draft_assets(self, draft_id: int, *, owner_user_id: int) -> list[dict[str, Any]]:
        return self.list_assets_for_draft(draft_id, owner_user_id=owner_user_id)

    def create_pack_version(
        self,
        *,
        pack_id: int,
        owner_user_id: int,
        version_number: int,
        manifest: Mapping[str, Any],
        default_expression_key: str = "neutral",
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        if self.get_pack(pack_id, owner_user_id=owner_user_id) is None:
            raise ValueError("visual_identity_pack_not_found")
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO visual_identity_pack_versions (
                    pack_id,
                    owner_user_id,
                    version_number,
                    default_expression_key,
                    manifest_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    pack_id,
                    owner_user_id,
                    version_number,
                    default_expression_key,
                    _json_dump(dict(manifest)),
                ),
            )
            pack_version_id = int(cursor.lastrowid)
        version = self.get_pack_version(pack_version_id, owner_user_id=owner_user_id)
        if version is None:
            raise RuntimeError("created_visual_identity_pack_version_not_found")
        return version

    def get_pack_version(
        self,
        pack_version_id: int,
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        if owner_user_id is None:
            cursor = self.db.execute_query(
                "SELECT * FROM visual_identity_pack_versions WHERE id = ?",
                (pack_version_id,),
            )
        else:
            cursor = self.db.execute_query(
                """
                SELECT * FROM visual_identity_pack_versions
                WHERE id = ? AND owner_user_id = ?
                """,
                (pack_version_id, owner_user_id),
            )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def set_active_version(
        self,
        *,
        pack_id: int,
        owner_user_id: int,
        pack_version_id: int,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        version = self.get_pack_version(pack_version_id, owner_user_id=owner_user_id)
        if version is None or int(version["pack_id"]) != int(pack_id):
            raise ValueError("visual_identity_pack_version_not_found")
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE visual_identity_packs
                SET active_version_id = ?,
                    default_expression_key = ?,
                    updated_at = CURRENT_TIMESTAMP,
                    version = version + 1
                WHERE id = ?
                  AND owner_user_id = ?
                  AND status != 'deleted'
                """,
                (pack_version_id, version["default_expression_key"], pack_id, owner_user_id),
            )
        pack = self.get_pack(pack_id, owner_user_id=owner_user_id)
        if pack is None:
            raise ValueError("visual_identity_pack_not_found")
        return pack

    def create_asset(
        self,
        *,
        owner_user_id: int,
        expression_key: str,
        source_filename: str,
        storage_relpath: str,
        content_type: str,
        bytes: int,
        sha256: str,
        width: int,
        height: int,
        pack_id: int | None = None,
        draft_id: int | None = None,
        pack_version_id: int | None = None,
        original_expression_key: str = "",
        display_label: str = "",
        is_animated: bool = False,
        frame_count: int | None = None,
        duration_ms: int | None = None,
        preview_relpath: str | None = None,
        source_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        pack_id = self._normalize_asset_attachment(
            owner_user_id=owner_user_id,
            pack_id=pack_id,
            draft_id=draft_id,
            pack_version_id=pack_version_id,
            bytes=bytes,
            width=width,
            height=height,
        )
        source_context_json = _json_dump(_canonicalize_source_context(source_context or {}))
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO visual_identity_assets (
                    owner_user_id,
                    pack_id,
                    draft_id,
                    pack_version_id,
                    expression_key,
                    original_expression_key,
                    display_label,
                    source_filename,
                    storage_relpath,
                    content_type,
                    bytes,
                    sha256,
                    width,
                    height,
                    source_context_json,
                    is_animated,
                    frame_count,
                    duration_ms,
                    preview_relpath
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    pack_id,
                    draft_id,
                    pack_version_id,
                    expression_key,
                    original_expression_key,
                    display_label,
                    source_filename,
                    storage_relpath,
                    content_type,
                    bytes,
                    sha256,
                    width,
                    height,
                    source_context_json,
                    int(is_animated),
                    frame_count,
                    duration_ms,
                    preview_relpath,
                ),
            )
            asset_id = int(cursor.lastrowid)
        asset = self.get_asset(asset_id, owner_user_id=owner_user_id)
        if asset is None:
            raise RuntimeError("created_visual_identity_asset_not_found")
        return asset

    def get_asset(
        self,
        asset_id: int,
        *,
        owner_user_id: int | None = None,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT * FROM visual_identity_assets
            WHERE id = ?
              AND (? IS NULL OR owner_user_id = ?)
              AND (? = 1 OR deleted = 0)
            """,
            (asset_id, owner_user_id, owner_user_id, int(include_deleted)),
        )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def list_assets_for_version(
        self,
        pack_version_id: int,
        *,
        owner_user_id: int,
    ) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT * FROM visual_identity_assets
            WHERE pack_version_id = ?
              AND owner_user_id = ?
              AND deleted = 0
            ORDER BY id ASC
            """,
            (pack_version_id, owner_user_id),
        )
        return [dict(row) for row in cursor.fetchall()]

    def list_assets_for_draft(
        self,
        draft_id: int,
        *,
        owner_user_id: int,
    ) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT * FROM visual_identity_assets
            WHERE draft_id = ?
              AND owner_user_id = ?
              AND deleted = 0
            ORDER BY id ASC
            """,
            (draft_id, owner_user_id),
        )
        return [dict(row) for row in cursor.fetchall()]

    def mark_draft_assets_deleted(self, *, draft_id: int, owner_user_id: int) -> int:
        self._ensure_schema_initialized()
        if self.get_draft(draft_id, owner_user_id=owner_user_id) is None:
            raise ValueError("visual_identity_draft_not_found")
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE visual_identity_assets
                SET deleted = 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE draft_id = ?
                  AND owner_user_id = ?
                  AND deleted = 0
                """,
                (draft_id, owner_user_id),
            )
            return int(cursor.rowcount or 0)

    def mark_asset_deleted(self, asset_id: int, *, owner_user_id: int) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE visual_identity_assets
                SET deleted = 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND owner_user_id = ?
                """,
                (asset_id, owner_user_id),
            )
        asset = self.get_asset(asset_id, owner_user_id=owner_user_id, include_deleted=True)
        if asset is None:
            raise ValueError("visual_identity_asset_not_found")
        return asset

    def upsert_binding(
        self,
        *,
        owner_user_id: int,
        actor_kind: str,
        actor_id: int | str,
        pack_id: int,
        active_version_id: int,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        self._require_owned_pack_version(
            owner_user_id=owner_user_id,
            pack_id=pack_id,
            pack_version_id=active_version_id,
        )
        normalized_actor_id = _normalize_actor_id(actor_id)
        with self.db.transaction() as conn:
            existing = conn.execute(
                """
                SELECT * FROM visual_identity_bindings
                WHERE owner_user_id = ?
                  AND actor_kind = ?
                  AND actor_id = ?
                  AND status = 'active'
                """,
                (owner_user_id, actor_kind, normalized_actor_id),
            ).fetchone()
            if existing is None:
                cursor = conn.execute(
                    """
                    INSERT INTO visual_identity_bindings (
                        owner_user_id,
                        actor_kind,
                        actor_id,
                        pack_id,
                        active_version_id,
                        status
                    )
                    VALUES (?, ?, ?, ?, ?, 'active')
                    """,
                    (owner_user_id, actor_kind, normalized_actor_id, pack_id, active_version_id),
                )
                binding_id = int(cursor.lastrowid)
            else:
                binding_id = int(existing["id"])
                conn.execute(
                    """
                    UPDATE visual_identity_bindings
                    SET pack_id = ?,
                        active_version_id = ?,
                        updated_at = CURRENT_TIMESTAMP,
                        version = version + 1
                    WHERE id = ? AND owner_user_id = ?
                    """,
                    (pack_id, active_version_id, binding_id, owner_user_id),
                )
        binding = self.get_binding(binding_id, owner_user_id=owner_user_id, include_deleted=True)
        if binding is None:
            raise RuntimeError("visual_identity_binding_not_found")
        return binding

    def delete_binding(self, binding_id: int, *, owner_user_id: int) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE visual_identity_bindings
                SET status = 'deleted',
                    updated_at = CURRENT_TIMESTAMP,
                    version = version + 1
                WHERE id = ? AND owner_user_id = ?
                """,
                (binding_id, owner_user_id),
            )
        return self.get_binding(binding_id, owner_user_id=owner_user_id, include_deleted=True)

    def get_binding(
        self,
        binding_id: int,
        *,
        owner_user_id: int | None = None,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT * FROM visual_identity_bindings
            WHERE id = ?
              AND (? IS NULL OR owner_user_id = ?)
              AND (? = 1 OR status = 'active')
            """,
            (binding_id, owner_user_id, owner_user_id, int(include_deleted)),
        )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def get_binding_for_actor(
        self,
        *,
        owner_user_id: int,
        actor_kind: str,
        actor_id: int | str,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        normalized_actor_id = _normalize_actor_id(actor_id)
        cursor = self.db.execute_query(
            """
            SELECT * FROM visual_identity_bindings
            WHERE owner_user_id = ?
              AND actor_kind = ?
              AND actor_id = ?
              AND status = 'active'
            """,
            (owner_user_id, actor_kind, normalized_actor_id),
        )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def resolve_active_binding(
        self,
        *,
        owner_user_id: int,
        actor_kind: str,
        actor_id: int | str,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        normalized_actor_id = _normalize_actor_id(actor_id)
        cursor = self.db.execute_query(
            """
            SELECT
                b.*,
                p.status AS pack_status,
                p.active_version_id AS pack_active_version_id,
                p.default_expression_key AS pack_default_expression_key,
                v.manifest_json AS active_manifest_json
            FROM visual_identity_bindings b
            JOIN visual_identity_packs p
              ON p.id = b.pack_id
             AND p.owner_user_id = b.owner_user_id
            JOIN visual_identity_pack_versions v
              ON v.id = b.active_version_id
             AND v.owner_user_id = b.owner_user_id
             AND v.pack_id = b.pack_id
            WHERE b.owner_user_id = ?
              AND b.actor_kind = ?
              AND b.actor_id = ?
              AND b.status = 'active'
              AND p.status = 'active'
            """,
            (owner_user_id, actor_kind, normalized_actor_id),
        )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def activate_draft_as_version(
        self,
        *,
        owner_user_id: int,
        draft_id: int,
        actor_kind: str | None = None,
        actor_id: int | str | None = None,
        title: str | None = None,
        description: str | None = None,
        source_kind: str | None = None,
        source_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically activate a draft as a new immutable pack version."""
        self._ensure_schema_initialized()
        if (actor_kind is None) != (actor_id is None):
            raise ValueError("visual_identity_binding_actor_required")
        if actor_kind is not None and actor_kind not in {"character", "persona"}:
            raise ValueError("visual_identity_actor_kind_invalid")
        normalized_actor_id = _normalize_actor_id(actor_id) if actor_id is not None else None

        with self.db.transaction() as conn:
            draft_row = conn.execute(
                """
                SELECT * FROM visual_identity_pack_drafts
                WHERE id = ? AND owner_user_id = ?
                """,
                (draft_id, owner_user_id),
            ).fetchone()
            if draft_row is None:
                raise ValueError("visual_identity_draft_not_found")
            draft = dict(draft_row)
            if draft["status"] != "ready_for_review":
                raise ValueError("visual_identity_draft_not_ready")

            pack_id = draft.get("pack_id")
            default_expression_key = str(draft.get("default_expression_key") or "neutral")

            if pack_id is None:
                resolved_title = title if title is not None else str(draft["title"])
                resolved_description = description if description is not None else ""
                resolved_source_kind = (
                    source_kind if source_kind is not None else draft["source_kind"]
                )
                resolved_source_context = {
                    "draft_id": int(draft["id"]),
                    "source_filename": draft.get("source_filename") or "",
                    **dict(source_context or {}),
                }
                pack_cursor = conn.execute(
                    """
                    INSERT INTO visual_identity_packs (
                        owner_user_id,
                        title,
                        description,
                        status,
                        active_version_id,
                        default_expression_key,
                        source_kind,
                        source_context_json
                    )
                    VALUES (?, ?, ?, 'active', NULL, ?, ?, ?)
                    """,
                    (
                        owner_user_id,
                        resolved_title,
                        resolved_description,
                        default_expression_key,
                        resolved_source_kind,
                        _json_dump(resolved_source_context),
                    ),
                )
                pack_id = int(pack_cursor.lastrowid)
            else:
                pack_id = int(pack_id)
                existing_pack_row = conn.execute(
                    """
                    SELECT * FROM visual_identity_packs
                    WHERE id = ?
                      AND owner_user_id = ?
                      AND status != 'deleted'
                    """,
                    (pack_id, owner_user_id),
                ).fetchone()
                if existing_pack_row is None:
                    raise ValueError("visual_identity_pack_not_found")
                existing_pack = dict(existing_pack_row)
                resolved_title = title if title is not None else str(existing_pack["title"])
                resolved_description = (
                    description
                    if description is not None
                    else str(existing_pack.get("description") or "")
                )
                resolved_source_kind = (
                    source_kind
                    if source_kind is not None
                    else str(existing_pack.get("source_kind") or draft["source_kind"])
                )
                resolved_source_context_json = (
                    _json_dump(dict(source_context))
                    if source_context is not None
                    else str(existing_pack.get("source_context_json") or "{}")
                )
                update_cursor = conn.execute(
                    """
                    UPDATE visual_identity_packs
                    SET title = ?,
                        description = ?,
                        status = 'active',
                        default_expression_key = ?,
                        source_kind = ?,
                        source_context_json = ?,
                        updated_at = CURRENT_TIMESTAMP,
                        version = version + 1
                    WHERE id = ?
                      AND owner_user_id = ?
                      AND status != 'deleted'
                    """,
                    (
                        resolved_title,
                        resolved_description,
                        default_expression_key,
                        resolved_source_kind,
                        resolved_source_context_json,
                        pack_id,
                        owner_user_id,
                    ),
                )
                if update_cursor.rowcount == 0:
                    raise ValueError("visual_identity_pack_not_found")

            version_number = int(
                conn.execute(
                    """
                    SELECT COALESCE(MAX(version_number), 0)
                    FROM visual_identity_pack_versions
                    WHERE pack_id = ? AND owner_user_id = ?
                    """,
                    (pack_id, owner_user_id),
                ).fetchone()[0]
            ) + 1
            version_cursor = conn.execute(
                """
                INSERT INTO visual_identity_pack_versions (
                    pack_id,
                    owner_user_id,
                    version_number,
                    default_expression_key,
                    manifest_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (pack_id, owner_user_id, version_number, default_expression_key, "{}"),
            )
            pack_version_id = int(version_cursor.lastrowid)

            slot_selections = _draft_slot_asset_selections(draft)
            if slot_selections is None:
                draft_assets = [
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT * FROM visual_identity_assets
                        WHERE draft_id = ?
                          AND owner_user_id = ?
                          AND deleted = 0
                        ORDER BY expression_key ASC, id ASC
                        """,
                        (draft_id, owner_user_id),
                    ).fetchall()
                ]
            else:
                if not slot_selections:
                    raise ValueError("visual_identity_draft_has_no_selected_assets")
                selected_ids = [selection["asset_id"] for selection in slot_selections]
                selected_rows = [
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT * FROM visual_identity_assets
                        WHERE draft_id = ?
                          AND owner_user_id = ?
                          AND deleted = 0
                        """,
                        (draft_id, owner_user_id),
                    ).fetchall()
                    if int(row["id"]) in selected_ids
                ]
                assets_by_id = {int(asset["id"]): asset for asset in selected_rows}
                missing_ids = sorted(set(selected_ids) - set(assets_by_id))
                if missing_ids:
                    raise ValueError("visual_identity_draft_slot_asset_not_found")
                draft_assets = [
                    {
                        **assets_by_id[int(selection["asset_id"])],
                        "expression_key": selection["expression_key"],
                        "display_label": selection["display_label"],
                    }
                    for selection in sorted(
                        slot_selections,
                        key=lambda item: (str(item["expression_key"]), int(item["asset_id"])),
                    )
                ]
            copied_assets: list[dict[str, Any]] = []
            for asset in draft_assets:
                source_context_json = str(asset.get("source_context_json") or "{}")
                asset_cursor = conn.execute(
                    """
                    INSERT INTO visual_identity_assets (
                        owner_user_id,
                        pack_id,
                        draft_id,
                        pack_version_id,
                        expression_key,
                        original_expression_key,
                        display_label,
                        source_filename,
                        storage_relpath,
                        content_type,
                        bytes,
                        sha256,
                        width,
                        height,
                        source_context_json,
                        is_animated,
                        frame_count,
                        duration_ms,
                        preview_relpath,
                        deleted
                    )
                    VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        owner_user_id,
                        pack_id,
                        pack_version_id,
                        asset["expression_key"],
                        asset["original_expression_key"],
                        asset["display_label"],
                        asset["source_filename"],
                        asset["storage_relpath"],
                        asset["content_type"],
                        asset["bytes"],
                        asset["sha256"],
                        asset["width"],
                        asset["height"],
                        source_context_json,
                        asset["is_animated"],
                        asset["frame_count"],
                        asset["duration_ms"],
                        asset["preview_relpath"],
                    ),
                )
                copied_assets.append(
                    {
                        **asset,
                        "id": int(asset_cursor.lastrowid),
                        "pack_id": pack_id,
                        "draft_id": None,
                        "pack_version_id": pack_version_id,
                        "source_context_json": source_context_json,
                        "deleted": 0,
                    }
                )
            copied_assets = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT * FROM visual_identity_assets
                    WHERE pack_version_id = ?
                      AND owner_user_id = ?
                      AND deleted = 0
                    ORDER BY expression_key ASC, id ASC
                    """,
                    (pack_version_id, owner_user_id),
                ).fetchall()
            ]

            manifest = _build_activation_manifest(
                draft=draft,
                version_number=version_number,
                assets=copied_assets,
                default_expression_key=default_expression_key,
            )
            conn.execute(
                """
                UPDATE visual_identity_pack_versions
                SET manifest_json = ?
                WHERE id = ? AND owner_user_id = ?
                """,
                (_json_dump(manifest), pack_version_id, owner_user_id),
            )
            conn.execute(
                """
                UPDATE visual_identity_packs
                SET active_version_id = ?,
                    default_expression_key = ?,
                    updated_at = CURRENT_TIMESTAMP,
                    version = version + 1
                WHERE id = ?
                  AND owner_user_id = ?
                  AND status != 'deleted'
                """,
                (pack_version_id, default_expression_key, pack_id, owner_user_id),
            )
            conn.execute(
                """
                UPDATE visual_identity_pack_drafts
                SET pack_id = ?,
                    status = 'activated',
                    updated_at = CURRENT_TIMESTAMP,
                    version = version + 1
                WHERE id = ? AND owner_user_id = ?
                """,
                (pack_id, draft_id, owner_user_id),
            )

            binding_id: int | None = None
            if actor_kind is not None and normalized_actor_id is not None:
                existing = conn.execute(
                    """
                    SELECT * FROM visual_identity_bindings
                    WHERE owner_user_id = ?
                      AND actor_kind = ?
                      AND actor_id = ?
                      AND status = 'active'
                    """,
                    (owner_user_id, actor_kind, normalized_actor_id),
                ).fetchone()
                if existing is None:
                    binding_cursor = conn.execute(
                        """
                        INSERT INTO visual_identity_bindings (
                            owner_user_id,
                            actor_kind,
                            actor_id,
                            pack_id,
                            active_version_id,
                            status
                        )
                        VALUES (?, ?, ?, ?, ?, 'active')
                        """,
                        (
                            owner_user_id,
                            actor_kind,
                            normalized_actor_id,
                            pack_id,
                            pack_version_id,
                        ),
                    )
                    binding_id = int(binding_cursor.lastrowid)
                else:
                    binding_id = int(existing["id"])
                    conn.execute(
                        """
                        UPDATE visual_identity_bindings
                        SET pack_id = ?,
                            active_version_id = ?,
                            updated_at = CURRENT_TIMESTAMP,
                            version = version + 1
                        WHERE id = ? AND owner_user_id = ?
                        """,
                        (pack_id, pack_version_id, binding_id, owner_user_id),
                    )

            pack = dict(
                conn.execute(
                    """
                    SELECT * FROM visual_identity_packs
                    WHERE id = ? AND owner_user_id = ?
                    """,
                    (pack_id, owner_user_id),
                ).fetchone()
            )
            pack_version = dict(
                conn.execute(
                    """
                    SELECT * FROM visual_identity_pack_versions
                    WHERE id = ? AND owner_user_id = ?
                    """,
                    (pack_version_id, owner_user_id),
                ).fetchone()
            )
            binding = (
                dict(
                    conn.execute(
                        """
                        SELECT * FROM visual_identity_bindings
                        WHERE id = ? AND owner_user_id = ?
                        """,
                        (binding_id, owner_user_id),
                    ).fetchone()
                )
                if binding_id is not None
                else None
            )

        return {
            "pack": pack,
            "pack_version": pack_version,
            "assets": copied_assets,
            "binding": binding,
        }

    def get_idempotency_record(
        self,
        *,
        owner_user_id: int,
        scope: str,
        resource_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT * FROM visual_identity_idempotency
            WHERE owner_user_id = ?
              AND scope = ?
              AND resource_id = ?
              AND idempotency_key = ?
            """,
            (owner_user_id, scope, resource_id, idempotency_key),
        )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def claim_idempotency_record(
        self,
        *,
        owner_user_id: int,
        scope: str,
        resource_id: str,
        idempotency_key: str,
        payload_hash: str,
        stale_after_seconds: int | None = IDEMPOTENCY_IN_PROGRESS_STALE_AFTER_SECONDS,
    ) -> tuple[dict[str, Any], bool]:
        self._ensure_schema_initialized()
        claim_token = secrets.token_urlsafe(24)
        with self.db.transaction() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO visual_identity_idempotency (
                        owner_user_id,
                        scope,
                        resource_id,
                        idempotency_key,
                        payload_hash,
                        claim_token,
                        status,
                        response_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'in_progress', '{}')
                    """,
                    (
                        owner_user_id,
                        scope,
                        resource_id,
                        idempotency_key,
                        payload_hash,
                        claim_token,
                    ),
                )
                claimed = True
            except sqlite3.IntegrityError:
                claimed = False
            row = conn.execute(
                """
                SELECT * FROM visual_identity_idempotency
                WHERE owner_user_id = ?
                  AND scope = ?
                  AND resource_id = ?
                  AND idempotency_key = ?
                """,
                (owner_user_id, scope, resource_id, idempotency_key),
            ).fetchone()
            if row is None:
                raise RuntimeError("visual_identity_idempotency_record_not_found")
            record = dict(row)
            if str(record["payload_hash"]) != payload_hash:
                raise ValueError("idempotency_key_conflict")
            if (
                not claimed
                and stale_after_seconds is not None
                and str(record.get("status") or "") == "in_progress"
                and int(stale_after_seconds) >= 0
            ):
                cursor = conn.execute(
                    """
                    UPDATE visual_identity_idempotency
                    SET updated_at = CURRENT_TIMESTAMP,
                        claim_token = ?,
                        response_json = '{}'
                    WHERE owner_user_id = ?
                      AND scope = ?
                      AND resource_id = ?
                      AND idempotency_key = ?
                      AND payload_hash = ?
                      AND status = 'in_progress'
                      AND datetime(updated_at) <= datetime('now', ?)
                    """,
                    (
                        claim_token,
                        owner_user_id,
                        scope,
                        resource_id,
                        idempotency_key,
                        payload_hash,
                        f"-{int(stale_after_seconds)} seconds",
                    ),
                )
                if cursor.rowcount:
                    row = conn.execute(
                        """
                        SELECT * FROM visual_identity_idempotency
                        WHERE owner_user_id = ?
                          AND scope = ?
                          AND resource_id = ?
                          AND idempotency_key = ?
                        """,
                        (owner_user_id, scope, resource_id, idempotency_key),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError("visual_identity_idempotency_record_not_found")
                    return dict(row), True
            return record, claimed

    def complete_idempotency_record(
        self,
        *,
        owner_user_id: int,
        scope: str,
        resource_id: str,
        idempotency_key: str,
        payload_hash: str,
        response: Mapping[str, Any],
        claim_token: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            existing = conn.execute(
                """
                SELECT * FROM visual_identity_idempotency
                WHERE owner_user_id = ?
                  AND scope = ?
                  AND resource_id = ?
                  AND idempotency_key = ?
                """,
                (owner_user_id, scope, resource_id, idempotency_key),
            ).fetchone()
            if existing is not None and str(existing["payload_hash"]) != payload_hash:
                raise ValueError("idempotency_key_conflict")
            cursor = conn.execute(
                """
                UPDATE visual_identity_idempotency
                SET status = 'completed',
                    response_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE owner_user_id = ?
                  AND scope = ?
                  AND resource_id = ?
                  AND idempotency_key = ?
                  AND (? IS NULL OR claim_token = ?)
                """,
                (
                    _json_dump(dict(response)),
                    owner_user_id,
                    scope,
                    resource_id,
                    idempotency_key,
                    claim_token,
                    claim_token,
                ),
            )
            if cursor.rowcount == 0:
                if claim_token is not None and existing is not None:
                    return dict(existing)
                conn.execute(
                    """
                    INSERT INTO visual_identity_idempotency (
                        owner_user_id,
                        scope,
                        resource_id,
                        idempotency_key,
                        payload_hash,
                        claim_token,
                        status,
                        response_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'completed', ?)
                    """,
                    (
                        owner_user_id,
                        scope,
                        resource_id,
                        idempotency_key,
                        payload_hash,
                        claim_token,
                        _json_dump(dict(response)),
                    ),
                )
        record = self.get_idempotency_record(
            owner_user_id=owner_user_id,
            scope=scope,
            resource_id=resource_id,
            idempotency_key=idempotency_key,
        )
        if record is None:
            raise RuntimeError("completed_visual_identity_idempotency_record_not_found")
        return record

    def release_idempotency_claim(
        self,
        *,
        owner_user_id: int,
        scope: str,
        resource_id: str,
        idempotency_key: str,
        claim_token: str | None = None,
    ) -> None:
        """Remove an in-progress idempotency claim after a failed side effect."""
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                DELETE FROM visual_identity_idempotency
                WHERE owner_user_id = ?
                  AND scope = ?
                  AND resource_id = ?
                  AND idempotency_key = ?
                  AND status = 'in_progress'
                  AND (? IS NULL OR claim_token = ?)
                """,
                (
                    owner_user_id,
                    scope,
                    resource_id,
                    idempotency_key,
                    claim_token,
                    claim_token,
                ),
            )

    def create_idempotency_record(
        self,
        *,
        owner_user_id: int,
        scope: str,
        resource_id: str,
        idempotency_key: str,
        payload_hash: str,
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            existing = conn.execute(
                """
                SELECT * FROM visual_identity_idempotency
                WHERE owner_user_id = ?
                  AND scope = ?
                  AND resource_id = ?
                  AND idempotency_key = ?
                """,
                (owner_user_id, scope, resource_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                if str(existing["payload_hash"]) != payload_hash:
                    raise ValueError("idempotency_key_conflict")
                conn.execute(
                    """
                    UPDATE visual_identity_idempotency
                    SET status = 'completed',
                        response_json = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE owner_user_id = ?
                      AND scope = ?
                      AND resource_id = ?
                      AND idempotency_key = ?
                    """,
                    (
                        _json_dump(dict(response)),
                        owner_user_id,
                        scope,
                        resource_id,
                        idempotency_key,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO visual_identity_idempotency (
                        owner_user_id,
                        scope,
                        resource_id,
                        idempotency_key,
                        payload_hash,
                        status,
                        response_json
                    )
                    VALUES (?, ?, ?, ?, ?, 'completed', ?)
                    """,
                    (
                        owner_user_id,
                        scope,
                        resource_id,
                        idempotency_key,
                        payload_hash,
                        _json_dump(dict(response)),
                    ),
                )
        record = self.get_idempotency_record(
            owner_user_id=owner_user_id,
            scope=scope,
            resource_id=resource_id,
            idempotency_key=idempotency_key,
        )
        if record is None:
            raise RuntimeError("created_visual_identity_idempotency_record_not_found")
        return record

    def _ensure_schema_initialized(self) -> None:
        if self._schema_initialized:
            return
        self.initialize_schema()

    def _require_owned_pack_version(
        self,
        *,
        owner_user_id: int,
        pack_id: int,
        pack_version_id: int,
    ) -> dict[str, Any]:
        version = self.get_pack_version(pack_version_id, owner_user_id=owner_user_id)
        if version is None or int(version["pack_id"]) != int(pack_id):
            raise ValueError("visual_identity_pack_version_not_found")
        if self.get_pack(pack_id, owner_user_id=owner_user_id) is None:
            raise ValueError("visual_identity_pack_not_found")
        return version

    def _normalize_asset_attachment(
        self,
        *,
        owner_user_id: int,
        pack_id: int | None,
        draft_id: int | None,
        pack_version_id: int | None,
        bytes: int,
        width: int,
        height: int,
    ) -> int | None:
        if bytes <= 0 or width <= 0 or height <= 0:
            raise ValueError("visual_identity_asset_dimensions_invalid")
        if draft_id is None and pack_version_id is None:
            raise ValueError("visual_identity_asset_attachment_required")

        normalized_pack_id = pack_id
        if normalized_pack_id is not None and self.get_pack(
            normalized_pack_id,
            owner_user_id=owner_user_id,
        ) is None:
            raise ValueError("visual_identity_pack_not_found")

        if draft_id is not None:
            draft = self.get_draft(draft_id, owner_user_id=owner_user_id)
            if draft is None:
                raise ValueError("visual_identity_draft_not_found")
            draft_pack_id = draft.get("pack_id")
            if draft_pack_id is not None:
                draft_pack_id = int(draft_pack_id)
                if normalized_pack_id is None:
                    normalized_pack_id = draft_pack_id
                elif int(normalized_pack_id) != draft_pack_id:
                    raise ValueError("visual_identity_draft_not_found")

        if pack_version_id is not None:
            version = self.get_pack_version(pack_version_id, owner_user_id=owner_user_id)
            if version is None:
                raise ValueError("visual_identity_pack_version_not_found")
            version_pack_id = int(version["pack_id"])
            if normalized_pack_id is None:
                normalized_pack_id = version_pack_id
            elif int(normalized_pack_id) != version_pack_id:
                raise ValueError("visual_identity_pack_version_not_found")

        return normalized_pack_id


def _update_values(
    fields: Mapping[str, Any],
    statements: Mapping[str, str],
    json_fields: set[str],
) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    for field_name, raw_value in fields.items():
        statement = statements.get(field_name)
        if statement is None:
            raise ValueError(f"unsupported_pack_update_field:{field_name}")
        value = _json_dump(dict(raw_value or {})) if field_name in json_fields else raw_value
        values.append((statement, value))
    return values


def _json_dump(value: Any) -> str:
    return json.dumps(value)


def _canonicalize_source_context(value: object) -> dict[str, Any]:
    from tldw_Server_API.app.core.Visual_Identities.source_context import (
        canonicalize_source_context,
    )

    return canonicalize_source_context(value)


def _json_loads_dict(value: Any) -> dict[str, Any]:
    try:
        loaded = json.loads(str(value or "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError("visual_identity_source_context_invalid") from exc
    if not isinstance(loaded, dict):
        raise ValueError("visual_identity_source_context_invalid")
    return loaded


def _draft_slot_asset_selections(
    draft: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    """Return selected draft assets from slot_map_json, or None for legacy copy-all drafts."""
    try:
        slot_map = json.loads(str(draft.get("slot_map_json") or "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError("visual_identity_draft_slot_map_invalid") from exc
    if not isinstance(slot_map, dict) or not slot_map:
        return None

    selections: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for slot_key, raw_entry in slot_map.items():
        if not isinstance(raw_entry, Mapping):
            continue
        asset_id = raw_entry.get("asset_id")
        if asset_id is None:
            continue
        try:
            normalized_asset_id = int(asset_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("visual_identity_draft_slot_asset_invalid") from exc
        if normalized_asset_id <= 0:
            raise ValueError("visual_identity_draft_slot_asset_invalid")
        expression_key = str(raw_entry.get("expression_key") or slot_key)
        display_label = str(raw_entry.get("display_label") or expression_key)
        selection_key = (normalized_asset_id, expression_key)
        if selection_key in seen:
            continue
        seen.add(selection_key)
        selections.append(
            {
                "asset_id": normalized_asset_id,
                "expression_key": expression_key,
                "display_label": display_label,
            }
        )
    return selections


def _build_activation_manifest(
    *,
    draft: Mapping[str, Any],
    version_number: int,
    assets: list[dict[str, Any]],
    default_expression_key: str,
) -> dict[str, Any]:
    """Build a deterministic manifest for an activated draft version."""
    return {
        "manifest_version": 1,
        "renderer": "expression_pack",
        "source": {
            "kind": draft.get("source_kind") or "",
            "draft_id": int(draft["id"]),
            "source_filename": draft.get("source_filename") or "",
        },
        "version_number": int(version_number),
        "default_expression_key": default_expression_key,
        "assets": [
            {
                "asset_id": int(asset["id"]),
                "expression_key": str(asset["expression_key"]),
                "source_context": _json_loads_dict(asset.get("source_context_json")),
            }
            for asset in sorted(
                assets,
                key=lambda item: (str(item["expression_key"]), int(item["id"])),
            )
        ],
    }


def _normalize_actor_id(actor_id: int | str | None) -> str:
    """Normalize integer character IDs and string persona IDs for binding storage."""
    normalized = "" if actor_id is None else str(actor_id).strip()
    if not normalized:
        raise ValueError("visual_identity_actor_id_required")
    return normalized


def _require_sqlite_chacha_db(db: CharactersRAGDB) -> None:
    if getattr(db, "backend_type", None) != BackendType.SQLITE:
        raise NotImplementedError(
            "Visual identity metadata currently supports SQLite ChaChaNotes databases only."
        )


_PACK_UPDATE_STATEMENTS = {
    "title": (
        "UPDATE visual_identity_packs "
        "SET title = ? WHERE id = ? AND owner_user_id = ? AND status != 'deleted'"
    ),
    "description": (
        "UPDATE visual_identity_packs "
        "SET description = ? WHERE id = ? AND owner_user_id = ? AND status != 'deleted'"
    ),
    "status": (
        "UPDATE visual_identity_packs "
        "SET status = ? WHERE id = ? AND owner_user_id = ? AND status != 'deleted'"
    ),
    "default_expression_key": (
        "UPDATE visual_identity_packs "
        "SET default_expression_key = ? WHERE id = ? AND owner_user_id = ? AND status != 'deleted'"
    ),
    "source_kind": (
        "UPDATE visual_identity_packs "
        "SET source_kind = ? WHERE id = ? AND owner_user_id = ? AND status != 'deleted'"
    ),
    "source_context": (
        "UPDATE visual_identity_packs "
        "SET source_context_json = ? WHERE id = ? AND owner_user_id = ? AND status != 'deleted'"
    ),
}

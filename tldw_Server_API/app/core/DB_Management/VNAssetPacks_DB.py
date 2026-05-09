"""VN asset pack metadata storage for per-user ChaChaNotes databases."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

VN_ASSET_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS vn_asset_packs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    content_rating TEXT NOT NULL DEFAULT 'general',
    primary_character_id INTEGER NOT NULL REFERENCES character_cards(id),
    source_world_book_ids_json TEXT NOT NULL DEFAULT '[]',
    scenario_notes TEXT,
    style_prompt TEXT,
    negative_prompt TEXT,
    default_backend TEXT,
    default_model TEXT,
    default_dimensions_json TEXT,
    style_lock_json TEXT,
    generation_budget_json TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    deleted BOOLEAN NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS vn_asset_slots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pack_id INTEGER NOT NULL REFERENCES vn_asset_packs(id) ON DELETE CASCADE,
    asset_type TEXT NOT NULL,
    slot_key TEXT NOT NULL,
    labels_json TEXT NOT NULL DEFAULT '{}',
    prompt_template TEXT,
    negative_prompt_template TEXT,
    variant_count INTEGER NOT NULL DEFAULT 1,
    width INTEGER,
    height INTEGER,
    backend_override TEXT,
    model_override TEXT,
    seed_policy_json TEXT,
    requires_review BOOLEAN NOT NULL DEFAULT 1,
    required_for_runtime BOOLEAN NOT NULL DEFAULT 1,
    depends_on_slot_id INTEGER REFERENCES vn_asset_slots(id),
    status TEXT NOT NULL DEFAULT 'planned',
    last_error TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pack_id, slot_key)
);

CREATE TABLE IF NOT EXISTS vn_asset_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pack_id INTEGER NOT NULL REFERENCES vn_asset_packs(id) ON DELETE CASCADE,
    slot_id INTEGER NOT NULL REFERENCES vn_asset_slots(id) ON DELETE CASCADE,
    variant_index INTEGER NOT NULL DEFAULT 0,
    file_artifact_id TEXT,
    generated_file_id INTEGER,
    storage_ref TEXT,
    mime_type TEXT,
    width INTEGER,
    height INTEGER,
    bytes INTEGER,
    review_status TEXT NOT NULL DEFAULT 'draft',
    preferred BOOLEAN NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'generated',
    generation_job_id TEXT,
    source_prompt_snapshot_json TEXT,
    source_context_snapshot_json TEXT,
    backend_metadata_json TEXT,
    depth_kind TEXT,
    parent_item_id INTEGER REFERENCES vn_asset_items(id),
    has_alpha BOOLEAN,
    crop_box_json TEXT,
    anchor_json TEXT,
    scale_hint REAL,
    trim_status TEXT NOT NULL DEFAULT 'unknown',
    quality_flags_json TEXT NOT NULL DEFAULT '[]',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_asset_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pack_id INTEGER NOT NULL REFERENCES vn_asset_packs(id) ON DELETE CASCADE,
    job_batch_id TEXT,
    requested_by_user_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'planned',
    total_slots INTEGER NOT NULL DEFAULT 0,
    total_variants INTEGER NOT NULL DEFAULT 0,
    planned_count INTEGER NOT NULL DEFAULT 0,
    enqueued_count INTEGER NOT NULL DEFAULT 0,
    enqueue_error TEXT,
    completed_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    cancelled_count INTEGER NOT NULL DEFAULT 0,
    started_at DATETIME,
    completed_at DATETIME,
    options_json TEXT NOT NULL DEFAULT '{}',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_pack_portability_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE,
    owner_user_id INTEGER NOT NULL,
    operation TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    pack_id INTEGER REFERENCES vn_asset_packs(id),
    preview_id INTEGER,
    import_id INTEGER,
    archive_path TEXT,
    archive_sha256 TEXT,
    canonical_payload_fingerprint TEXT,
    progress_json TEXT NOT NULL DEFAULT '{}',
    warnings_json TEXT NOT NULL DEFAULT '[]',
    error_code TEXT,
    error_message TEXT,
    download_url TEXT,
    expires_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_pack_import_previews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    job_id TEXT NOT NULL,
    status TEXT NOT NULL,
    archive_path TEXT NOT NULL,
    archive_sha256 TEXT,
    canonical_payload_fingerprint TEXT,
    schema_version TEXT,
    bundle_summary_json TEXT NOT NULL DEFAULT '{}',
    validation_warnings_json TEXT NOT NULL DEFAULT '[]',
    conflicts_json TEXT NOT NULL DEFAULT '[]',
    proposed_plan_json TEXT NOT NULL DEFAULT '{}',
    quota_estimate_json TEXT NOT NULL DEFAULT '{}',
    required_choices_json TEXT NOT NULL DEFAULT '[]',
    expires_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_pack_import_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    preview_id INTEGER NOT NULL REFERENCES vn_pack_import_previews(id),
    job_id TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    trust_mode TEXT NOT NULL,
    target_mode TEXT NOT NULL,
    target_pack_id INTEGER,
    archive_path TEXT,
    archive_sha256 TEXT,
    canonical_payload_fingerprint TEXT,
    id_maps_json TEXT NOT NULL DEFAULT '{}',
    created_records_json TEXT NOT NULL DEFAULT '{}',
    cleanup_status_json TEXT NOT NULL DEFAULT '{}',
    warnings_json TEXT NOT NULL DEFAULT '[]',
    error_code TEXT,
    error_message TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME
);

CREATE INDEX IF NOT EXISTS idx_vn_asset_packs_primary_character_id
    ON vn_asset_packs(primary_character_id);
CREATE INDEX IF NOT EXISTS idx_vn_asset_packs_deleted
    ON vn_asset_packs(deleted);
CREATE INDEX IF NOT EXISTS idx_vn_asset_slots_pack_id
    ON vn_asset_slots(pack_id);
CREATE INDEX IF NOT EXISTS idx_vn_asset_slots_depends_on_slot_id
    ON vn_asset_slots(depends_on_slot_id);
CREATE INDEX IF NOT EXISTS idx_vn_asset_items_pack_id
    ON vn_asset_items(pack_id);
CREATE INDEX IF NOT EXISTS idx_vn_asset_items_slot_id
    ON vn_asset_items(slot_id);
CREATE INDEX IF NOT EXISTS idx_vn_asset_items_generated_file_id
    ON vn_asset_items(generated_file_id);
CREATE INDEX IF NOT EXISTS idx_vn_asset_items_parent_item_id
    ON vn_asset_items(parent_item_id);
CREATE INDEX IF NOT EXISTS idx_vn_asset_batches_pack_id
    ON vn_asset_batches(pack_id);
CREATE INDEX IF NOT EXISTS idx_vn_asset_batches_job_batch_id
    ON vn_asset_batches(job_batch_id);
CREATE INDEX IF NOT EXISTS idx_vn_pack_portability_jobs_owner_user_id
    ON vn_pack_portability_jobs(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_vn_pack_portability_jobs_job_id
    ON vn_pack_portability_jobs(job_id);
CREATE INDEX IF NOT EXISTS idx_vn_pack_portability_jobs_status
    ON vn_pack_portability_jobs(status);
CREATE INDEX IF NOT EXISTS idx_vn_pack_portability_jobs_expires_at
    ON vn_pack_portability_jobs(expires_at);
CREATE INDEX IF NOT EXISTS idx_vn_pack_portability_jobs_fingerprint
    ON vn_pack_portability_jobs(canonical_payload_fingerprint);
CREATE INDEX IF NOT EXISTS idx_vn_pack_import_previews_owner_user_id
    ON vn_pack_import_previews(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_vn_pack_import_previews_job_id
    ON vn_pack_import_previews(job_id);
CREATE INDEX IF NOT EXISTS idx_vn_pack_import_previews_status
    ON vn_pack_import_previews(status);
CREATE INDEX IF NOT EXISTS idx_vn_pack_import_previews_expires_at
    ON vn_pack_import_previews(expires_at);
CREATE INDEX IF NOT EXISTS idx_vn_pack_import_previews_fingerprint
    ON vn_pack_import_previews(canonical_payload_fingerprint);
CREATE INDEX IF NOT EXISTS idx_vn_pack_import_journal_owner_user_id
    ON vn_pack_import_journal(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_vn_pack_import_journal_job_id
    ON vn_pack_import_journal(job_id);
CREATE INDEX IF NOT EXISTS idx_vn_pack_import_journal_status
    ON vn_pack_import_journal(status);
CREATE INDEX IF NOT EXISTS idx_vn_pack_import_journal_target_pack_id
    ON vn_pack_import_journal(target_pack_id);
CREATE INDEX IF NOT EXISTS idx_vn_pack_import_journal_fingerprint
    ON vn_pack_import_journal(canonical_payload_fingerprint);
"""

VN_ASSET_SCHEMA_STATEMENTS = tuple(
    statement.strip()
    for statement in VN_ASSET_SCHEMA_SQL.split(";")
    if statement.strip()
)


def ensure_vn_asset_tables(db: CharactersRAGDB) -> None:
    """Create VN asset metadata tables in the provided ChaChaNotes database."""
    _require_sqlite_chacha_db(db)
    with db.transaction() as conn:
        for statement in VN_ASSET_SCHEMA_STATEMENTS:
            conn.execute(statement)
        _ensure_batch_fanout_columns(conn)


class VNAssetPacksRepository:
    """Repository for VN asset pack metadata in a user's ChaChaNotes DB."""

    def __init__(self, db: CharactersRAGDB):
        _require_sqlite_chacha_db(db)
        self.db = db
        self._schema_initialized = False

    @classmethod
    def initialized(cls, db: CharactersRAGDB) -> VNAssetPacksRepository:
        repo = cls(db)
        repo.initialize_schema()
        return repo

    def initialize_schema(self) -> None:
        ensure_vn_asset_tables(self.db)
        self._schema_initialized = True

    def create_portability_job(
        self,
        *,
        owner_user_id: int,
        job_id: str,
        operation: str,
        status: str,
        stage: str,
        pack_id: int | None = None,
        preview_id: int | None = None,
        import_id: int | None = None,
        archive_path: str | None = None,
        archive_sha256: str | None = None,
        canonical_payload_fingerprint: str | None = None,
        progress: Mapping[str, Any] | None = None,
        warnings: list[Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        download_url: str | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO vn_pack_portability_jobs (
                    job_id,
                    owner_user_id,
                    operation,
                    status,
                    stage,
                    pack_id,
                    preview_id,
                    import_id,
                    archive_path,
                    archive_sha256,
                    canonical_payload_fingerprint,
                    progress_json,
                    warnings_json,
                    error_code,
                    error_message,
                    download_url,
                    expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    owner_user_id,
                    operation,
                    status,
                    stage,
                    pack_id,
                    preview_id,
                    import_id,
                    archive_path,
                    archive_sha256,
                    canonical_payload_fingerprint,
                    _json_dump(progress or {}),
                    _json_dump(warnings or []),
                    error_code,
                    error_message,
                    download_url,
                    expires_at,
                ),
            )
            portability_job_id = cursor.lastrowid

        job = self.get_portability_job(portability_job_id, owner_user_id=owner_user_id)
        if job is None:
            raise RuntimeError("created_portability_job_not_found")
        return job

    def get_portability_job(
        self,
        portability_job_id: int,
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        if owner_user_id is None:
            cursor = self.db.execute_query(
                "SELECT * FROM vn_pack_portability_jobs WHERE id = ?",
                (portability_job_id,),
            )
        else:
            cursor = self.db.execute_query(
                """
                SELECT * FROM vn_pack_portability_jobs
                WHERE id = ? AND owner_user_id = ?
                """,
                (portability_job_id, owner_user_id),
            )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def get_portability_job_by_job_id(
        self,
        job_id: str,
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        if owner_user_id is None:
            cursor = self.db.execute_query(
                "SELECT * FROM vn_pack_portability_jobs WHERE job_id = ?",
                (job_id,),
            )
        else:
            cursor = self.db.execute_query(
                """
                SELECT * FROM vn_pack_portability_jobs
                WHERE job_id = ? AND owner_user_id = ?
                """,
                (job_id, owner_user_id),
            )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def update_portability_job(
        self,
        job_id: str,
        fields: Mapping[str, Any],
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        current = self.get_portability_job_by_job_id(job_id, owner_user_id=owner_user_id)
        if current is None:
            return None

        update_values = _portability_update_values(
            fields,
            _PORTABILITY_JOB_UPDATE_COLUMNS,
            _PORTABILITY_JOB_JSON_DEFAULTS,
        )
        if not update_values:
            return current

        with self.db.transaction() as conn:
            where_clause = "job_id = ?"
            where_params: list[Any] = [job_id]
            if owner_user_id is not None:
                where_clause += " AND owner_user_id = ?"
                where_params.append(owner_user_id)
            _execute_portability_update(
                conn,
                table_name="vn_pack_portability_jobs",
                update_values=update_values,
                where_clause=where_clause,
                where_params=where_params,
            )
        return self.get_portability_job_by_job_id(job_id, owner_user_id=owner_user_id)

    def create_import_preview(
        self,
        *,
        owner_user_id: int,
        job_id: str,
        status: str,
        archive_path: str,
        archive_sha256: str | None = None,
        canonical_payload_fingerprint: str | None = None,
        schema_version: str | None = None,
        bundle_summary: Mapping[str, Any] | None = None,
        validation_warnings: list[Any] | None = None,
        conflicts: list[Any] | None = None,
        proposed_plan: Mapping[str, Any] | None = None,
        quota_estimate: Mapping[str, Any] | None = None,
        required_choices: list[Any] | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO vn_pack_import_previews (
                    owner_user_id,
                    job_id,
                    status,
                    archive_path,
                    archive_sha256,
                    canonical_payload_fingerprint,
                    schema_version,
                    bundle_summary_json,
                    validation_warnings_json,
                    conflicts_json,
                    proposed_plan_json,
                    quota_estimate_json,
                    required_choices_json,
                    expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    job_id,
                    status,
                    archive_path,
                    archive_sha256,
                    canonical_payload_fingerprint,
                    schema_version,
                    _json_dump(bundle_summary or {}),
                    _json_dump(validation_warnings or []),
                    _json_dump(conflicts or []),
                    _json_dump(proposed_plan or {}),
                    _json_dump(quota_estimate or {}),
                    _json_dump(required_choices or []),
                    expires_at,
                ),
            )
            preview_id = cursor.lastrowid

        preview = self.get_import_preview(preview_id, owner_user_id=owner_user_id)
        if preview is None:
            raise RuntimeError("created_import_preview_not_found")
        return preview

    def get_import_preview(
        self,
        preview_id: int,
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        if owner_user_id is None:
            cursor = self.db.execute_query(
                "SELECT * FROM vn_pack_import_previews WHERE id = ?",
                (preview_id,),
            )
        else:
            cursor = self.db.execute_query(
                """
                SELECT * FROM vn_pack_import_previews
                WHERE id = ? AND owner_user_id = ?
                """,
                (preview_id, owner_user_id),
            )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def update_import_preview(
        self,
        preview_id: int,
        fields: Mapping[str, Any],
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        current = self.get_import_preview(preview_id, owner_user_id=owner_user_id)
        if current is None:
            return None

        update_values = _portability_update_values(
            fields,
            _IMPORT_PREVIEW_UPDATE_COLUMNS,
            _IMPORT_PREVIEW_JSON_DEFAULTS,
        )
        if not update_values:
            return current

        with self.db.transaction() as conn:
            where_clause = "id = ?"
            where_params: list[Any] = [preview_id]
            if owner_user_id is not None:
                where_clause += " AND owner_user_id = ?"
                where_params.append(owner_user_id)
            _execute_portability_update(
                conn,
                table_name="vn_pack_import_previews",
                update_values=update_values,
                where_clause=where_clause,
                where_params=where_params,
            )
        return self.get_import_preview(preview_id, owner_user_id=owner_user_id)

    def create_import_journal(
        self,
        *,
        owner_user_id: int,
        preview_id: int,
        job_id: str,
        status: str,
        stage: str,
        trust_mode: str,
        target_mode: str,
        target_pack_id: int | None = None,
        archive_path: str | None = None,
        archive_sha256: str | None = None,
        canonical_payload_fingerprint: str | None = None,
        id_maps: Mapping[str, Any] | None = None,
        created_records: Mapping[str, Any] | None = None,
        cleanup_status: Mapping[str, Any] | None = None,
        warnings: list[Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        completed_at: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO vn_pack_import_journal (
                    owner_user_id,
                    preview_id,
                    job_id,
                    status,
                    stage,
                    trust_mode,
                    target_mode,
                    target_pack_id,
                    archive_path,
                    archive_sha256,
                    canonical_payload_fingerprint,
                    id_maps_json,
                    created_records_json,
                    cleanup_status_json,
                    warnings_json,
                    error_code,
                    error_message,
                    completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    preview_id,
                    job_id,
                    status,
                    stage,
                    trust_mode,
                    target_mode,
                    target_pack_id,
                    archive_path,
                    archive_sha256,
                    canonical_payload_fingerprint,
                    _json_dump(id_maps or {}),
                    _json_dump(created_records or {}),
                    _json_dump(cleanup_status or {}),
                    _json_dump(warnings or []),
                    error_code,
                    error_message,
                    completed_at,
                ),
            )
            import_id = cursor.lastrowid

        journal = self.get_import_journal(import_id, owner_user_id=owner_user_id)
        if journal is None:
            raise RuntimeError("created_import_journal_not_found")
        return journal

    def get_import_journal(
        self,
        import_id: int,
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        if owner_user_id is None:
            cursor = self.db.execute_query(
                "SELECT * FROM vn_pack_import_journal WHERE id = ?",
                (import_id,),
            )
        else:
            cursor = self.db.execute_query(
                """
                SELECT * FROM vn_pack_import_journal
                WHERE id = ? AND owner_user_id = ?
                """,
                (import_id, owner_user_id),
            )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def update_import_journal(
        self,
        import_id: int,
        fields: Mapping[str, Any],
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        current = self.get_import_journal(import_id, owner_user_id=owner_user_id)
        if current is None:
            return None

        update_values = _portability_update_values(
            fields,
            _IMPORT_JOURNAL_UPDATE_COLUMNS,
            _IMPORT_JOURNAL_JSON_DEFAULTS,
        )
        if not update_values:
            return current

        with self.db.transaction() as conn:
            where_clause = "id = ?"
            where_params: list[Any] = [import_id]
            if owner_user_id is not None:
                where_clause += " AND owner_user_id = ?"
                where_params.append(owner_user_id)
            _execute_portability_update(
                conn,
                table_name="vn_pack_import_journal",
                update_values=update_values,
                where_clause=where_clause,
                where_params=where_params,
            )
        return self.get_import_journal(import_id, owner_user_id=owner_user_id)

    def create_pack(
        self,
        *,
        owner_user_id: int,
        primary_character_id: int,
        title: str,
        description: str | None = None,
        content_rating: str = "general",
        source_world_book_ids: list[int] | None = None,
        scenario_notes: str | None = None,
        style_prompt: str | None = None,
        negative_prompt: str | None = None,
        default_backend: str | None = None,
        default_model: str | None = None,
        default_dimensions: Mapping[str, Any] | None = None,
        style_lock: Mapping[str, Any] | None = None,
        generation_budget: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()

        if not self._primary_character_exists(primary_character_id):
            raise ValueError("primary_character_not_found")

        source_world_book_ids_json = json.dumps(source_world_book_ids or [])
        default_dimensions_json = _json_or_none(default_dimensions)
        style_lock_json = _json_or_none(style_lock)
        generation_budget_json = _json_or_none(generation_budget)

        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO vn_asset_packs (
                    owner_user_id,
                    title,
                    description,
                    content_rating,
                    primary_character_id,
                    source_world_book_ids_json,
                    scenario_notes,
                    style_prompt,
                    negative_prompt,
                    default_backend,
                    default_model,
                    default_dimensions_json,
                    style_lock_json,
                    generation_budget_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    title,
                    description,
                    content_rating,
                    primary_character_id,
                    source_world_book_ids_json,
                    scenario_notes,
                    style_prompt,
                    negative_prompt,
                    default_backend,
                    default_model,
                    default_dimensions_json,
                    style_lock_json,
                    generation_budget_json,
                ),
            )
            pack_id = cursor.lastrowid

        pack = self.get_pack(pack_id)
        if pack is None:
            raise RuntimeError("created_pack_not_found")
        return pack

    def get_pack(self, pack_id: int) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query("SELECT * FROM vn_asset_packs WHERE id = ?", (pack_id,))
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def list_packs(self, *, owner_user_id: int | None = None) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        if owner_user_id is None:
            cursor = self.db.execute_query(
                "SELECT * FROM vn_asset_packs WHERE deleted = 0 ORDER BY id ASC"
            )
        else:
            cursor = self.db.execute_query(
                """
                SELECT * FROM vn_asset_packs
                WHERE deleted = 0 AND owner_user_id = ?
                ORDER BY id ASC
                """,
                (owner_user_id,),
            )
        return [dict(row) for row in cursor.fetchall()]

    def list_packs_for_setup(
        self,
        *,
        owner_user_id: int,
        query: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return one bounded setup page before readiness fanout."""
        self._ensure_schema_initialized()
        normalized_limit = max(1, int(limit))
        normalized_offset = max(0, int(offset))

        filters = ["deleted = 0", "owner_user_id = ?"]
        params: list[Any] = [owner_user_id]
        normalized_query = (query or "").strip().lower()
        if normalized_query:
            like_value = f"%{normalized_query}%"
            filters.append(
                "("
                "LOWER(COALESCE(title, '')) LIKE ? OR "
                "LOWER(COALESCE(description, '')) LIKE ?"
                ")"
            )
            params.extend([like_value, like_value])

        params.extend([normalized_limit + 1, normalized_offset])
        where_clause = " AND ".join(filters)
        cursor = self.db.execute_query(
            f"""
            SELECT * FROM vn_asset_packs
            WHERE {where_clause}
            ORDER BY id ASC
            LIMIT ? OFFSET ?
            """,  # nosec B608
            tuple(params),
        )
        rows = [dict(row) for row in cursor.fetchall()]
        return rows[:normalized_limit], len(rows) > normalized_limit

    def latest_completed_import_provenance_by_pack_ids(
        self,
        *,
        owner_user_id: int,
        pack_ids: list[int],
    ) -> dict[int, dict[str, Any]]:
        """Return latest completed import journal rows for the requested packs."""
        self._ensure_schema_initialized()
        normalized_pack_ids = sorted({int(pack_id) for pack_id in pack_ids})
        if not normalized_pack_ids:
            return {}

        placeholders = ",".join("?" for _ in normalized_pack_ids)
        cursor = self.db.execute_query(
            f"""
            SELECT * FROM vn_pack_import_journal
            WHERE owner_user_id = ?
              AND status = 'completed'
              AND target_pack_id IN ({placeholders})
            ORDER BY
              target_pack_id ASC,
              COALESCE(completed_at, updated_at, created_at) DESC,
              id DESC
            """,  # nosec B608
            (owner_user_id, *normalized_pack_ids),
        )
        provenance: dict[int, dict[str, Any]] = {}
        for row in cursor.fetchall():
            journal = dict(row)
            target_pack_id = journal.get("target_pack_id")
            if target_pack_id is None:
                continue
            pack_id = int(target_pack_id)
            provenance.setdefault(pack_id, journal)
        return provenance

    def update_pack(self, pack_id: int, fields: Mapping[str, Any]) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        json_fields = {
            "source_world_book_ids",
            "default_dimensions",
            "style_lock",
            "generation_budget",
        }

        update_values = [
            (
                field_name,
                json.dumps(value) if field_name in json_fields else value,
            )
            for field_name, value in fields.items()
            if _pack_update_statement(field_name) is not None
        ]
        if not update_values:
            return self.get_pack(pack_id)

        with self.db.transaction() as conn:
            for field_name, value in update_values:
                statement = _pack_update_statement(field_name)
                if statement is None:
                    continue
                conn.execute(statement, (value, pack_id))
            conn.execute(
                """
                UPDATE vn_asset_packs
                SET updated_at = CURRENT_TIMESTAMP, version = version + 1
                WHERE id = ?
                """,
                (pack_id,),
            )
        return self.get_pack(pack_id)

    def soft_delete_pack(self, pack_id: int) -> None:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE vn_asset_packs
                SET deleted = 1, updated_at = CURRENT_TIMESTAMP, version = version + 1
                WHERE id = ?
                """,
                (pack_id,),
            )

    def create_slot(
        self,
        *,
        pack_id: int,
        asset_type: str,
        slot_key: str,
        labels: Mapping[str, Any] | None = None,
        prompt_template: str | None = None,
        negative_prompt_template: str | None = None,
        variant_count: int = 1,
        width: int | None = None,
        height: int | None = None,
        backend_override: str | None = None,
        model_override: str | None = None,
        seed_policy: Mapping[str, Any] | None = None,
        requires_review: bool = True,
        required_for_runtime: bool = True,
        depends_on_slot_id: int | None = None,
        status: str = "planned",
        last_error: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            slot_id = self._insert_slot_row(
                conn,
                pack_id=pack_id,
                asset_type=asset_type,
                slot_key=slot_key,
                labels=labels,
                prompt_template=prompt_template,
                negative_prompt_template=negative_prompt_template,
                variant_count=variant_count,
                width=width,
                height=height,
                backend_override=backend_override,
                model_override=model_override,
                seed_policy=seed_policy,
                requires_review=requires_review,
                required_for_runtime=required_for_runtime,
                depends_on_slot_id=depends_on_slot_id,
                status=status,
                last_error=last_error,
            )

        slot = self.get_slot(slot_id)
        if slot is None:
            raise RuntimeError("created_slot_not_found")
        return slot

    def create_slots_for_matrix(
        self,
        *,
        pack_id: int,
        slot_specs: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        if not slot_specs:
            return []

        created_slot_ids: list[int] = []
        with self.db.transaction() as conn:
            slot_keys = [str(spec["slot_key"]) for spec in slot_specs]
            if len(set(slot_keys)) != len(slot_keys):
                raise ValueError("slot_already_exists")
            existing_slot_keys = self._existing_slot_keys(conn, pack_id, slot_keys)
            if existing_slot_keys:
                raise ValueError("slot_already_exists")

            slot_ids_by_key: dict[str, int] = {}
            pending_dependent_specs: list[Mapping[str, Any]] = []
            for spec in slot_specs:
                if spec.get("depends_on_slot_key"):
                    pending_dependent_specs.append(spec)
                    continue
                slot_id = self._insert_slot_row(conn, pack_id=pack_id, **_slot_insert_kwargs(spec))
                slot_ids_by_key[str(spec["slot_key"])] = slot_id
                created_slot_ids.append(slot_id)

            while pending_dependent_specs:
                unresolved_specs: list[Mapping[str, Any]] = []
                resolved_count = 0
                for spec in pending_dependent_specs:
                    parent_slot_key = str(spec["depends_on_slot_key"])
                    depends_on_slot_id = slot_ids_by_key.get(parent_slot_key)
                    if depends_on_slot_id is None:
                        unresolved_specs.append(spec)
                        continue
                    slot_kwargs = _slot_insert_kwargs(spec)
                    slot_kwargs["depends_on_slot_id"] = depends_on_slot_id
                    slot_id = self._insert_slot_row(
                        conn,
                        pack_id=pack_id,
                        **slot_kwargs,
                    )
                    slot_ids_by_key[str(spec["slot_key"])] = slot_id
                    created_slot_ids.append(slot_id)
                    resolved_count += 1

                if resolved_count == 0:
                    raise ValueError("dependent_slot_not_found")
                pending_dependent_specs = unresolved_specs

        return [
            slot
            for slot_id in created_slot_ids
            if (slot := self.get_slot(slot_id)) is not None
        ]

    def _insert_slot_row(
        self,
        conn: Any,
        *,
        pack_id: int,
        asset_type: str,
        slot_key: str,
        labels: Mapping[str, Any] | None = None,
        prompt_template: str | None = None,
        negative_prompt_template: str | None = None,
        variant_count: int = 1,
        width: int | None = None,
        height: int | None = None,
        backend_override: str | None = None,
        model_override: str | None = None,
        seed_policy: Mapping[str, Any] | None = None,
        requires_review: bool = True,
        required_for_runtime: bool = True,
        depends_on_slot_id: int | None = None,
        status: str = "planned",
        last_error: str | None = None,
    ) -> int:
        cursor = conn.execute(
            """
            INSERT INTO vn_asset_slots (
                pack_id,
                asset_type,
                slot_key,
                labels_json,
                prompt_template,
                negative_prompt_template,
                variant_count,
                width,
                height,
                backend_override,
                model_override,
                seed_policy_json,
                requires_review,
                required_for_runtime,
                depends_on_slot_id,
                status,
                last_error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pack_id,
                asset_type,
                slot_key,
                json.dumps(dict(labels or {})),
                prompt_template,
                negative_prompt_template,
                variant_count,
                width,
                height,
                backend_override,
                model_override,
                _json_or_none(seed_policy),
                int(requires_review),
                int(required_for_runtime),
                depends_on_slot_id,
                status,
                last_error,
            ),
        )
        return int(cursor.lastrowid)

    def get_slot(self, slot_id: int) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query("SELECT * FROM vn_asset_slots WHERE id = ?", (slot_id,))
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def list_slots(self, pack_id: int) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            "SELECT * FROM vn_asset_slots WHERE pack_id = ? ORDER BY id ASC",
            (pack_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def existing_slot_keys(self, pack_id: int, slot_keys: list[str]) -> set[str]:
        self._ensure_schema_initialized()
        return self._existing_slot_keys(self.db.get_connection(), pack_id, slot_keys)

    def _existing_slot_keys(self, conn: Any, pack_id: int, slot_keys: list[str]) -> set[str]:
        existing: set[str] = set()
        for slot_key in slot_keys:
            cursor = conn.execute(
                "SELECT slot_key FROM vn_asset_slots WHERE pack_id = ? AND slot_key = ?",
                (pack_id, slot_key),
            )
            row = cursor.fetchone()
            if row is not None:
                existing.add(str(row["slot_key"]))
        return existing

    def update_slot(self, slot_id: int, fields: Mapping[str, Any]) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        json_fields = {"labels", "seed_policy"}
        bool_fields = {"requires_review", "required_for_runtime"}
        update_values: list[tuple[str, Any]] = []
        for field_name, raw_value in fields.items():
            if _slot_update_statement(field_name) is None:
                continue
            if field_name in json_fields:
                value = json.dumps(raw_value) if raw_value is not None else None
            elif field_name in bool_fields:
                value = int(raw_value)
            else:
                value = raw_value
            update_values.append((field_name, value))

        if not update_values:
            return self.get_slot(slot_id)

        with self.db.transaction() as conn:
            for field_name, value in update_values:
                statement = _slot_update_statement(field_name)
                if statement is None:
                    continue
                conn.execute(statement, (value, slot_id))
            conn.execute(
                "UPDATE vn_asset_slots SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (slot_id,),
            )
        return self.get_slot(slot_id)

    def delete_slot(self, slot_id: int) -> None:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM vn_asset_slots WHERE id = ?", (slot_id,))

    def create_item(
        self,
        *,
        pack_id: int,
        slot_id: int,
        variant_index: int = 0,
        file_artifact_id: str | None = None,
        generated_file_id: int | None = None,
        storage_ref: str | None = None,
        mime_type: str | None = None,
        width: int | None = None,
        height: int | None = None,
        bytes: int | None = None,
        review_status: str = "draft",
        preferred: bool = False,
        source: str = "generated",
        generation_job_id: str | None = None,
        source_prompt_snapshot: Mapping[str, Any] | None = None,
        source_context_snapshot: Mapping[str, Any] | None = None,
        backend_metadata: Mapping[str, Any] | None = None,
        depth_kind: str | None = None,
        parent_item_id: int | None = None,
        has_alpha: bool | None = None,
        crop_box: Mapping[str, Any] | None = None,
        anchor: Mapping[str, float] | None = None,
        scale_hint: float | None = None,
        trim_status: str = "unknown",
        quality_flags: list[str] | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        if not self._slot_belongs_to_pack(pack_id, slot_id):
            raise ValueError("slot_not_in_pack")
        if preferred and review_status != "approved":
            raise ValueError("preferred_item_must_be_approved")

        with self.db.transaction() as conn:
            if preferred:
                conn.execute(
                    """
                    UPDATE vn_asset_items
                    SET preferred = 0, updated_at = CURRENT_TIMESTAMP
                    WHERE pack_id = ? AND slot_id = ?
                    """,
                    (pack_id, slot_id),
                )
            cursor = conn.execute(
                """
                INSERT INTO vn_asset_items (
                    pack_id,
                    slot_id,
                    variant_index,
                    file_artifact_id,
                    generated_file_id,
                    storage_ref,
                    mime_type,
                    width,
                    height,
                    bytes,
                    review_status,
                    preferred,
                    source,
                    generation_job_id,
                    source_prompt_snapshot_json,
                    source_context_snapshot_json,
                    backend_metadata_json,
                    depth_kind,
                    parent_item_id,
                    has_alpha,
                    crop_box_json,
                    anchor_json,
                    scale_hint,
                    trim_status,
                    quality_flags_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pack_id,
                    slot_id,
                    variant_index,
                    file_artifact_id,
                    generated_file_id,
                    storage_ref,
                    mime_type,
                    width,
                    height,
                    bytes,
                    review_status,
                    int(preferred),
                    source,
                    generation_job_id,
                    _json_or_none(source_prompt_snapshot),
                    _json_or_none(source_context_snapshot),
                    _json_or_none(backend_metadata),
                    depth_kind,
                    parent_item_id,
                    None if has_alpha is None else int(has_alpha),
                    _json_or_none(crop_box),
                    _json_or_none(anchor),
                    scale_hint,
                    trim_status,
                    json.dumps(quality_flags or []),
                ),
            )
            item_id = cursor.lastrowid

        item = self.get_item(item_id)
        if item is None:
            raise RuntimeError("created_item_not_found")
        return item

    def get_item(self, item_id: int) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query("SELECT * FROM vn_asset_items WHERE id = ?", (item_id,))
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def delete_item(self, item_id: int) -> bool:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            cursor = conn.execute("DELETE FROM vn_asset_items WHERE id = ?", (item_id,))
            return cursor.rowcount > 0

    def update_item_storage(
        self,
        item_id: int,
        *,
        generated_file_id: int | None,
        storage_ref: str | None,
        mime_type: str | None,
        width: int | None,
        height: int | None,
        bytes: int | None,
        backend_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE vn_asset_items
                SET generated_file_id = ?,
                    storage_ref = ?,
                    mime_type = ?,
                    width = ?,
                    height = ?,
                    bytes = ?,
                    backend_metadata_json = COALESCE(?, backend_metadata_json),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    generated_file_id,
                    storage_ref,
                    mime_type,
                    width,
                    height,
                    bytes,
                    _json_or_none(backend_metadata),
                    item_id,
                ),
            )
        return self.get_item(item_id)

    def list_items(self, pack_id: int) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            "SELECT * FROM vn_asset_items WHERE pack_id = ? ORDER BY id ASC",
            (pack_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def count_items_referencing_generated_file(
        self,
        generated_file_id: int,
        *,
        exclude_item_id: int | None = None,
    ) -> int:
        self._ensure_schema_initialized()
        if exclude_item_id is None:
            cursor = self.db.execute_query(
                "SELECT COUNT(*) AS count FROM vn_asset_items WHERE generated_file_id = ?",
                (generated_file_id,),
            )
        else:
            cursor = self.db.execute_query(
                """
                SELECT COUNT(*) AS count
                FROM vn_asset_items
                WHERE generated_file_id = ? AND id != ?
                """,
                (generated_file_id, exclude_item_id),
            )
        row = cursor.fetchone()
        return int(row["count"] if row is not None else 0)

    def update_item_review(
        self,
        item_id: int,
        *,
        review_status: str,
        preferred: bool | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        current_item = self.get_item(item_id)
        if current_item is None:
            return None

        next_preferred = preferred
        if review_status != "approved":
            next_preferred = False

        with self.db.transaction() as conn:
            if next_preferred is True:
                conn.execute(
                    """
                    UPDATE vn_asset_items
                    SET preferred = 0, updated_at = CURRENT_TIMESTAMP
                    WHERE pack_id = ? AND slot_id = ? AND id != ?
                    """,
                    (current_item["pack_id"], current_item["slot_id"], item_id),
                )
            if next_preferred is None:
                conn.execute(
                    """
                    UPDATE vn_asset_items
                    SET review_status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (review_status, item_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE vn_asset_items
                    SET review_status = ?, preferred = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (review_status, int(next_preferred), item_id),
                )
        return self.get_item(item_id)

    def bulk_update_item_review(
        self,
        item_ids: list[int],
        *,
        review_status: str,
    ) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        if not item_ids:
            return []

        with self.db.transaction() as conn:
            for item_id in item_ids:
                if review_status == "approved":
                    conn.execute(
                        """
                        UPDATE vn_asset_items
                        SET review_status = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (review_status, item_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE vn_asset_items
                        SET review_status = ?, preferred = 0, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (review_status, item_id),
                    )

        return [
            item
            for item_id in item_ids
            if (item := self.get_item(item_id)) is not None
        ]

    def create_batch(
        self,
        *,
        pack_id: int,
        requested_by_user_id: int,
        status: str = "planned",
        total_slots: int = 0,
        total_variants: int = 0,
        planned_count: int | None = None,
        job_batch_id: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO vn_asset_batches (
                    pack_id,
                    job_batch_id,
                    requested_by_user_id,
                    status,
                    total_slots,
                    total_variants,
                    planned_count,
                    options_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pack_id,
                    job_batch_id,
                    requested_by_user_id,
                    status,
                    total_slots,
                    total_variants,
                    total_variants if planned_count is None else planned_count,
                    json.dumps(dict(options or {})),
                ),
            )
            batch_id = cursor.lastrowid
        batch = self.get_batch(batch_id)
        if batch is None:
            raise RuntimeError("created_batch_not_found")
        return batch

    def get_batch(self, batch_id: int) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query("SELECT * FROM vn_asset_batches WHERE id = ?", (batch_id,))
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def list_batches(self, pack_id: int) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            "SELECT * FROM vn_asset_batches WHERE pack_id = ? ORDER BY id DESC",
            (pack_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def update_batch(self, batch_id: int, fields: Mapping[str, Any]) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        update_values: list[tuple[str, Any]] = []
        for field_name, raw_value in fields.items():
            statement = _batch_update_statement(field_name)
            if statement is None:
                continue
            value = json.dumps(dict(raw_value)) if field_name == "options" else raw_value
            update_values.append((field_name, value))

        if not update_values:
            return self.get_batch(batch_id)

        with self.db.transaction() as conn:
            for field_name, value in update_values:
                statement = _batch_update_statement(field_name)
                if statement is None:
                    continue
                conn.execute(statement, (value, batch_id))
            conn.execute(
                "UPDATE vn_asset_batches SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (batch_id,),
            )
        return self.get_batch(batch_id)

    def get_character(self, character_id: int) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            "SELECT * FROM character_cards WHERE id = ? AND deleted = 0",
            (character_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def _ensure_schema_initialized(self) -> None:
        if self._schema_initialized:
            return
        self.initialize_schema()

    def _primary_character_exists(self, primary_character_id: int) -> bool:
        cursor = self.db.execute_query(
            "SELECT 1 FROM character_cards WHERE id = ? AND deleted = 0",
            (primary_character_id,),
        )
        return cursor.fetchone() is not None

    def _slot_belongs_to_pack(self, pack_id: int, slot_id: int) -> bool:
        cursor = self.db.execute_query(
            "SELECT 1 FROM vn_asset_slots WHERE id = ? AND pack_id = ?",
            (slot_id, pack_id),
        )
        return cursor.fetchone() is not None


def _json_or_none(value: Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    return _json_dump(dict(value))


def _json_dump(value: Any) -> str:
    return json.dumps(value)


_PORTABILITY_JOB_UPDATE_COLUMNS = {
    "operation": "operation",
    "status": "status",
    "stage": "stage",
    "pack_id": "pack_id",
    "preview_id": "preview_id",
    "import_id": "import_id",
    "archive_path": "archive_path",
    "archive_sha256": "archive_sha256",
    "canonical_payload_fingerprint": "canonical_payload_fingerprint",
    "progress": "progress_json",
    "warnings": "warnings_json",
    "error_code": "error_code",
    "error_message": "error_message",
    "download_url": "download_url",
    "expires_at": "expires_at",
}

_PORTABILITY_JOB_JSON_DEFAULTS = {
    "progress": {},
    "warnings": [],
}

_IMPORT_PREVIEW_UPDATE_COLUMNS = {
    "job_id": "job_id",
    "status": "status",
    "archive_path": "archive_path",
    "archive_sha256": "archive_sha256",
    "canonical_payload_fingerprint": "canonical_payload_fingerprint",
    "schema_version": "schema_version",
    "bundle_summary": "bundle_summary_json",
    "validation_warnings": "validation_warnings_json",
    "conflicts": "conflicts_json",
    "proposed_plan": "proposed_plan_json",
    "quota_estimate": "quota_estimate_json",
    "required_choices": "required_choices_json",
    "expires_at": "expires_at",
}

_IMPORT_PREVIEW_JSON_DEFAULTS = {
    "bundle_summary": {},
    "validation_warnings": [],
    "conflicts": [],
    "proposed_plan": {},
    "quota_estimate": {},
    "required_choices": [],
}

_IMPORT_JOURNAL_UPDATE_COLUMNS = {
    "job_id": "job_id",
    "status": "status",
    "stage": "stage",
    "trust_mode": "trust_mode",
    "target_mode": "target_mode",
    "target_pack_id": "target_pack_id",
    "archive_path": "archive_path",
    "archive_sha256": "archive_sha256",
    "canonical_payload_fingerprint": "canonical_payload_fingerprint",
    "id_maps": "id_maps_json",
    "created_records": "created_records_json",
    "cleanup_status": "cleanup_status_json",
    "warnings": "warnings_json",
    "error_code": "error_code",
    "error_message": "error_message",
    "completed_at": "completed_at",
}

_IMPORT_JOURNAL_JSON_DEFAULTS = {
    "id_maps": {},
    "created_records": {},
    "cleanup_status": {},
    "warnings": [],
}

_PORTABILITY_UPDATE_TABLE_COLUMNS = {
    "vn_pack_portability_jobs": frozenset(_PORTABILITY_JOB_UPDATE_COLUMNS.values()),
    "vn_pack_import_previews": frozenset(_IMPORT_PREVIEW_UPDATE_COLUMNS.values()),
    "vn_pack_import_journal": frozenset(_IMPORT_JOURNAL_UPDATE_COLUMNS.values()),
}


def _portability_update_values(
    fields: Mapping[str, Any],
    update_columns: Mapping[str, str],
    json_defaults: Mapping[str, Any],
) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    for field_name, raw_value in fields.items():
        column_name = update_columns.get(field_name)
        if column_name is None:
            continue
        if field_name in json_defaults:
            default_value = json_defaults[field_name]
            value = _json_dump(default_value if raw_value is None else raw_value)
        else:
            value = raw_value
        values.append((column_name, value))

    return values


def _execute_portability_update(
    conn: Any,
    *,
    table_name: str,
    update_values: list[tuple[str, Any]],
    where_clause: str,
    where_params: list[Any],
) -> None:
    allowed_columns = _PORTABILITY_UPDATE_TABLE_COLUMNS.get(table_name)
    if allowed_columns is None:
        raise ValueError("unsupported_portability_update_table")
    if any(column_name not in allowed_columns for column_name, _ in update_values):
        raise ValueError("unsupported_portability_update_column")

    assignments = ", ".join(f"{column_name} = ?" for column_name, _ in update_values)
    # Identifiers are internal allowlists; user values stay parameterized.
    statement = (
        f"UPDATE {table_name} "  # nosec B608
        f"SET {assignments}, updated_at = CURRENT_TIMESTAMP "
        f"WHERE {where_clause}"
    )
    conn.execute(
        statement,
        tuple(value for _, value in update_values) + tuple(where_params),
    )


def _slot_insert_kwargs(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "asset_type": spec["asset_type"],
        "slot_key": spec["slot_key"],
        "labels": spec.get("labels"),
        "prompt_template": spec.get("prompt_template"),
        "negative_prompt_template": spec.get("negative_prompt_template"),
        "variant_count": spec.get("variant_count", 1),
        "width": spec.get("width"),
        "height": spec.get("height"),
        "backend_override": spec.get("backend_override"),
        "model_override": spec.get("model_override"),
        "seed_policy": spec.get("seed_policy"),
        "requires_review": spec.get("requires_review", True),
        "required_for_runtime": spec.get("required_for_runtime", True),
        "depends_on_slot_id": spec.get("depends_on_slot_id"),
        "status": spec.get("status", "planned"),
        "last_error": spec.get("last_error"),
    }


def _pack_update_statement(field_name: str) -> str | None:
    statements = {
        "title": "UPDATE vn_asset_packs SET title = ? WHERE id = ?",
        "description": "UPDATE vn_asset_packs SET description = ? WHERE id = ?",
        "status": "UPDATE vn_asset_packs SET status = ? WHERE id = ?",
        "content_rating": "UPDATE vn_asset_packs SET content_rating = ? WHERE id = ?",
        "source_world_book_ids": (
            "UPDATE vn_asset_packs SET source_world_book_ids_json = ? WHERE id = ?"
        ),
        "scenario_notes": "UPDATE vn_asset_packs SET scenario_notes = ? WHERE id = ?",
        "style_prompt": "UPDATE vn_asset_packs SET style_prompt = ? WHERE id = ?",
        "negative_prompt": "UPDATE vn_asset_packs SET negative_prompt = ? WHERE id = ?",
        "default_backend": "UPDATE vn_asset_packs SET default_backend = ? WHERE id = ?",
        "default_model": "UPDATE vn_asset_packs SET default_model = ? WHERE id = ?",
        "default_dimensions": (
            "UPDATE vn_asset_packs SET default_dimensions_json = ? WHERE id = ?"
        ),
        "style_lock": "UPDATE vn_asset_packs SET style_lock_json = ? WHERE id = ?",
        "generation_budget": (
            "UPDATE vn_asset_packs SET generation_budget_json = ? WHERE id = ?"
        ),
    }
    return statements.get(field_name)


def _batch_update_statement(field_name: str) -> str | None:
    statements = {
        "job_batch_id": "UPDATE vn_asset_batches SET job_batch_id = ? WHERE id = ?",
        "status": "UPDATE vn_asset_batches SET status = ? WHERE id = ?",
        "total_slots": "UPDATE vn_asset_batches SET total_slots = ? WHERE id = ?",
        "total_variants": "UPDATE vn_asset_batches SET total_variants = ? WHERE id = ?",
        "planned_count": "UPDATE vn_asset_batches SET planned_count = ? WHERE id = ?",
        "enqueued_count": "UPDATE vn_asset_batches SET enqueued_count = ? WHERE id = ?",
        "enqueue_error": "UPDATE vn_asset_batches SET enqueue_error = ? WHERE id = ?",
        "completed_count": "UPDATE vn_asset_batches SET completed_count = ? WHERE id = ?",
        "failed_count": "UPDATE vn_asset_batches SET failed_count = ? WHERE id = ?",
        "cancelled_count": "UPDATE vn_asset_batches SET cancelled_count = ? WHERE id = ?",
        "started_at": "UPDATE vn_asset_batches SET started_at = ? WHERE id = ?",
        "completed_at": "UPDATE vn_asset_batches SET completed_at = ? WHERE id = ?",
        "options": "UPDATE vn_asset_batches SET options_json = ? WHERE id = ?",
    }
    return statements.get(field_name)


def _slot_update_statement(field_name: str) -> str | None:
    statements = {
        "asset_type": "UPDATE vn_asset_slots SET asset_type = ? WHERE id = ?",
        "slot_key": "UPDATE vn_asset_slots SET slot_key = ? WHERE id = ?",
        "labels": "UPDATE vn_asset_slots SET labels_json = ? WHERE id = ?",
        "prompt_template": "UPDATE vn_asset_slots SET prompt_template = ? WHERE id = ?",
        "negative_prompt_template": (
            "UPDATE vn_asset_slots SET negative_prompt_template = ? WHERE id = ?"
        ),
        "variant_count": "UPDATE vn_asset_slots SET variant_count = ? WHERE id = ?",
        "width": "UPDATE vn_asset_slots SET width = ? WHERE id = ?",
        "height": "UPDATE vn_asset_slots SET height = ? WHERE id = ?",
        "backend_override": "UPDATE vn_asset_slots SET backend_override = ? WHERE id = ?",
        "model_override": "UPDATE vn_asset_slots SET model_override = ? WHERE id = ?",
        "seed_policy": "UPDATE vn_asset_slots SET seed_policy_json = ? WHERE id = ?",
        "requires_review": "UPDATE vn_asset_slots SET requires_review = ? WHERE id = ?",
        "required_for_runtime": (
            "UPDATE vn_asset_slots SET required_for_runtime = ? WHERE id = ?"
        ),
        "depends_on_slot_id": (
            "UPDATE vn_asset_slots SET depends_on_slot_id = ? WHERE id = ?"
        ),
        "status": "UPDATE vn_asset_slots SET status = ? WHERE id = ?",
        "last_error": "UPDATE vn_asset_slots SET last_error = ? WHERE id = ?",
    }
    return statements.get(field_name)


def _require_sqlite_chacha_db(db: CharactersRAGDB) -> None:
    if getattr(db, "backend_type", None) != BackendType.SQLITE:
        raise NotImplementedError(
            "VN asset pack metadata currently supports SQLite ChaChaNotes databases only."
        )


def _ensure_batch_fanout_columns(conn: Any) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(vn_asset_batches)").fetchall()}
    additions = {
        "planned_count": "ALTER TABLE vn_asset_batches ADD COLUMN planned_count INTEGER NOT NULL DEFAULT 0",
        "enqueued_count": "ALTER TABLE vn_asset_batches ADD COLUMN enqueued_count INTEGER NOT NULL DEFAULT 0",
        "enqueue_error": "ALTER TABLE vn_asset_batches ADD COLUMN enqueue_error TEXT",
    }
    for column_name, statement in additions.items():
        if column_name not in columns:
            conn.execute(statement)

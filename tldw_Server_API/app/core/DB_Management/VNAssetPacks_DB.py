"""VN asset pack metadata storage for per-user ChaChaNotes databases."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.exceptions import LegacyDisplayReconciliationError, VNAssetGenerationError
from tldw_Server_API.app.core.VN_Assets.state import derive_slot_status

LegacyActivityReader = Callable[
    [int, int, int, Mapping[int, str], set[tuple[int, int, str]], tuple[int, str] | None], tuple[bool, bool]
]

_VARIANT_OUTCOME_QUERY = """
    SELECT outcome_status, item_id, claim_token, claim_lease_id
    FROM vn_asset_generation_recipes
    WHERE batch_id = ? AND slot_id = ? AND variant_index = ?
"""


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
    recipe_version INTEGER NOT NULL DEFAULT 0,
    started_at DATETIME,
    completed_at DATETIME,
    options_json TEXT NOT NULL DEFAULT '{}',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_asset_generation_recipes (
    batch_id INTEGER NOT NULL REFERENCES vn_asset_batches(id) ON DELETE CASCADE,
    slot_id INTEGER NOT NULL REFERENCES vn_asset_slots(id) ON DELETE CASCADE,
    variant_index INTEGER NOT NULL,
    recipe_json TEXT NOT NULL,
    outcome_status TEXT NOT NULL DEFAULT 'planned',
    item_id INTEGER REFERENCES vn_asset_items(id),
    claim_lease_id TEXT,
    claim_token TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (batch_id, slot_id, variant_index)
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

CREATE TABLE IF NOT EXISTS vn_asset_idempotency_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    scope TEXT NOT NULL,
    resource_id TEXT NOT NULL DEFAULT '',
    idempotency_key TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'completed',
    batch_id INTEGER REFERENCES vn_asset_batches(id),
    response_json TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(owner_user_id, scope, resource_id, idempotency_key)
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
CREATE INDEX IF NOT EXISTS idx_vn_asset_idempotency_records_lookup
    ON vn_asset_idempotency_records(owner_user_id, scope, resource_id, idempotency_key);
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
        _ensure_recipe_outcome_columns(conn)


class VNAssetPacksRepository:
    """Repository for VN asset pack metadata in a user's ChaChaNotes DB."""

    def __init__(self, db: CharactersRAGDB):
        _require_sqlite_chacha_db(db)
        self.db = db
        self._schema_initialized = False
        self.legacy_activity_reader: LegacyActivityReader | None = None
        # Inline execution display is instance-local, never queue/lease authority.
        self._inline_legacy_activity: dict[tuple[int, int], int] = {}

    @classmethod
    def initialized(cls, db: CharactersRAGDB) -> VNAssetPacksRepository:
        repo = cls(db)
        repo.initialize_schema()
        return repo

    def initialize_schema(self) -> None:
        ensure_vn_asset_tables(self.db)
        self._schema_initialized = True

    def get_idempotency_record(
        self,
        *,
        owner_user_id: int,
        scope: str,
        resource_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        """Return a previously completed idempotent VN asset API response."""
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT * FROM vn_asset_idempotency_records
            WHERE owner_user_id = ?
              AND scope = ?
              AND resource_id = ?
              AND idempotency_key = ?
            """,
            (owner_user_id, scope, resource_id, idempotency_key),
        )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

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
        """Persist a completed idempotent VN asset API response."""
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            existing = conn.execute(
                """
                SELECT * FROM vn_asset_idempotency_records
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
                    UPDATE vn_asset_idempotency_records
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
                    INSERT INTO vn_asset_idempotency_records (
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
            row = conn.execute(
                """
                SELECT * FROM vn_asset_idempotency_records
                WHERE owner_user_id = ?
                  AND scope = ?
                  AND resource_id = ?
                  AND idempotency_key = ?
                """,
                (owner_user_id, scope, resource_id, idempotency_key),
            ).fetchone()
        if row is None:
            raise RuntimeError("created_idempotency_record_not_found")
        return dict(row)

    def claim_idempotency_record(
        self,
        *,
        owner_user_id: int,
        scope: str,
        resource_id: str,
        idempotency_key: str,
        payload_hash: str,
    ) -> tuple[dict[str, Any], bool]:
        """Atomically claim an idempotency key before side effects run.

        Returns the record and True when this caller created the in-progress
        claim. Returns the existing record and False for replay/in-progress
        cases. Raises ValueError when the same key was used for a different
        payload.
        """
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO vn_asset_idempotency_records (
                        owner_user_id,
                        scope,
                        resource_id,
                        idempotency_key,
                        payload_hash,
                        status,
                        response_json
                    )
                    VALUES (?, ?, ?, ?, ?, 'in_progress', '{}')
                    """,
                    (
                        owner_user_id,
                        scope,
                        resource_id,
                        idempotency_key,
                        payload_hash,
                    ),
                )
                claimed = True
            except sqlite3.IntegrityError:
                claimed = False
            cursor = conn.execute(
                """
                SELECT * FROM vn_asset_idempotency_records
                WHERE owner_user_id = ?
                  AND scope = ?
                  AND resource_id = ?
                  AND idempotency_key = ?
                """,
                (owner_user_id, scope, resource_id, idempotency_key),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("idempotency_record_not_found")
            record = dict(row)
            if str(record["payload_hash"]) != payload_hash:
                raise ValueError("idempotency_key_conflict")
            if (
                not claimed
                and scope in {
                    "vn_asset_generate", "vn_asset_slot_retry", "vn_asset_item_regenerate"
                }
                and record["status"] == "in_progress"
                and record["batch_id"] is None
            ):
                reclaimed = conn.execute(
                    """
                    UPDATE vn_asset_idempotency_records
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND status = 'in_progress' AND batch_id IS NULL
                      AND updated_at <= datetime('now', '-2 minutes')
                    """,
                    (record["id"],),
                )
                claimed = reclaimed.rowcount == 1
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
    ) -> dict[str, Any]:
        """Complete once for the exact scoped payload, preserving the first JSON.

        An unclaimed key retains insert-on-completion compatibility. Concurrent
        or delayed completions replay the first terminal record; a different
        payload hash raises the same conflict as claim without changing it.
        """
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO vn_asset_idempotency_records (
                    owner_user_id, scope, resource_id, idempotency_key,
                    payload_hash, status, response_json
                ) VALUES (?, ?, ?, ?, ?, 'completed', ?)
                ON CONFLICT(owner_user_id, scope, resource_id, idempotency_key)
                DO UPDATE SET status = 'completed',
                    response_json = excluded.response_json,
                    updated_at = CURRENT_TIMESTAMP
                WHERE vn_asset_idempotency_records.status = 'in_progress'
                  AND vn_asset_idempotency_records.payload_hash = excluded.payload_hash
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
                raise RuntimeError("completed_idempotency_record_not_found")
            if record["payload_hash"] != payload_hash:
                raise ValueError("idempotency_key_conflict")
            return record

    def release_idempotency_claim(
        self,
        *,
        owner_user_id: int,
        scope: str,
        resource_id: str,
        idempotency_key: str,
    ) -> None:
        """Remove an in-progress idempotency claim after a failed side effect."""
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                DELETE FROM vn_asset_idempotency_records
                WHERE owner_user_id = ?
                  AND scope = ?
                  AND resource_id = ?
                  AND idempotency_key = ?
                  AND status = 'in_progress'
                  AND batch_id IS NULL
                """,
                (owner_user_id, scope, resource_id, idempotency_key),
            )

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
            active_recipe = conn.execute(
                """
                SELECT 1 FROM vn_asset_generation_recipes AS recipe
                JOIN vn_asset_batches AS batch ON batch.id = recipe.batch_id
                WHERE recipe.slot_id = ?
                  AND batch.status IN ('planned', 'queued', 'enqueued', 'processing')
                LIMIT 1
                """,
                (slot_id,),
            ).fetchone()
            if active_recipe is not None:
                raise ValueError("slot_has_active_generation")
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

    def item_is_unpublished(self, item_id: int) -> bool:
        """Return whether item_id is linked to any non-completed recipe.

        Planned, failed and cancelled reservations are not reviewable, even if
        file metadata is attached. False does not establish item existence or
        approval; unlinked items and completed recipes are published. Database
        and schema initialization errors propagate.
        """
        self._ensure_schema_initialized()
        row = self.db.execute_query(
            """
            SELECT 1 FROM vn_asset_generation_recipes
            WHERE item_id = ? AND outcome_status != 'completed' LIMIT 1
            """,
            (item_id,),
        ).fetchone()
        return row is not None

    def delete_item(self, item_id: int) -> bool:
        """Delete an inactive item while retaining terminal recipes and depth children.

        Clear reference links in the same transaction for existing NO ACTION
        schemas. Active variant reservations remain protected from deletion.
        """
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            active = conn.execute(
                """
                SELECT 1 FROM vn_asset_generation_recipes AS recipe
                JOIN vn_asset_batches AS batch ON batch.id = recipe.batch_id
                WHERE recipe.item_id = ? AND recipe.outcome_status = 'planned'
                  AND batch.status NOT IN ('completed', 'failed', 'cancelled')
                LIMIT 1
                """,
                (item_id,),
            ).fetchone()
            if active is not None:
                raise VNAssetGenerationError(
                    "vn_asset_variant_in_progress", retryable=True, item_id=item_id,
                )
            conn.execute(
                "UPDATE vn_asset_generation_recipes SET item_id = NULL WHERE item_id = ?",
                (item_id,),
            )
            conn.execute(
                "UPDATE vn_asset_items SET parent_item_id = NULL WHERE parent_item_id = ?",
                (item_id,),
            )
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
        batch_id: int | None = None,
        slot_id: int | None = None,
        variant_index: int | None = None,
        attempt_token: str | None = None,
        validate_authority: Callable[[], None] | None = None,
    ) -> dict[str, Any] | None:
        """Attach file metadata to item_id and return the updated item, or None.

        Variant IDs and attempt_token fence a worker attachment; validate_authority
        is called after the VN write lock. Unclaimed legacy items need no token.
        Raises VNAssetGenerationError for a lost claim and propagates admission
        callback or database failures without translating their error codes.
        """
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            if attempt_token is not None:
                _lock_variant(conn, batch_id, slot_id, variant_index)
                if validate_authority is not None:
                    validate_authority()
                claim = conn.execute(
                    """
                    SELECT 1 FROM vn_asset_generation_recipes AS recipe
                    JOIN vn_asset_batches AS batch ON batch.id = recipe.batch_id
                    WHERE recipe.batch_id = ? AND recipe.slot_id = ?
                      AND recipe.variant_index = ? AND recipe.item_id = ?
                      AND recipe.claim_token = ? AND recipe.outcome_status = 'planned'
                      AND batch.status NOT IN ('cancelled', 'failed', 'completed')
                    """,
                    (batch_id, slot_id, variant_index, item_id, attempt_token),
                ).fetchone()
                if claim is None:
                    raise VNAssetGenerationError(
                        "vn_asset_variant_claim_lost", retryable=True,
                        batch_id=batch_id, item_id=item_id,
                    )
            else:
                conn.execute("UPDATE vn_asset_items SET id = id WHERE id = ?", (item_id,))
                claim = conn.execute(
                    """
                    SELECT 1 FROM vn_asset_generation_recipes
                    WHERE item_id = ? AND outcome_status = 'planned' AND claim_token IS NOT NULL
                    """,
                    (item_id,),
                ).fetchone()
                if claim is not None:
                    raise VNAssetGenerationError(
                        "vn_asset_variant_claim_lost", retryable=True, item_id=item_id,
                    )
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
            """
            SELECT * FROM vn_asset_items AS item
            WHERE item.pack_id = ? AND NOT EXISTS (
                SELECT 1 FROM vn_asset_generation_recipes AS recipe
                WHERE recipe.item_id = item.id AND recipe.outcome_status != 'completed'
            ) ORDER BY item.id ASC
            """,
            (pack_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def count_items_for_generation(self, pack_id: int) -> int:
        """Count visible items and active reservations, excluding terminal reservations."""
        self._ensure_schema_initialized()
        row = self.db.execute_query(
            """
            SELECT COUNT(*) AS item_count FROM vn_asset_items AS item
            WHERE item.pack_id = ? AND NOT EXISTS (
                SELECT 1 FROM vn_asset_generation_recipes AS recipe
                WHERE recipe.item_id = item.id
                  AND recipe.outcome_status IN ('failed', 'cancelled')
            )
            """,
            (pack_id,),
        ).fetchone()
        return int(row["item_count"])

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
        recipes: list[Mapping[str, Any]] | None = None,
        idempotency_receipt: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        if recipes is not None:
            expected_count = total_variants if planned_count is None else planned_count
            if not recipes or len(recipes) != expected_count:
                raise VNAssetGenerationError("vn_asset_recipe_count_mismatch", pack_id=pack_id)
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
                    options_json,
                    recipe_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    1 if recipes is not None else 0,
                ),
            )
            batch_id = cursor.lastrowid
            if idempotency_receipt is not None:
                linked = conn.execute(
                    """
                    UPDATE vn_asset_idempotency_records
                    SET batch_id = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE owner_user_id = ? AND scope = ? AND resource_id = ?
                      AND idempotency_key = ? AND payload_hash = ?
                      AND status = 'in_progress' AND batch_id IS NULL
                    """,
                    (
                        batch_id,
                        requested_by_user_id,
                        idempotency_receipt["scope"],
                        idempotency_receipt["resource_id"],
                        idempotency_receipt["idempotency_key"],
                        idempotency_receipt["payload_hash"],
                    ),
                )
                if linked.rowcount != 1:
                    raise VNAssetGenerationError("vn_asset_generation_receipt_not_claimed", pack_id=pack_id)
            if recipes is not None:
                for entry in recipes:
                    conn.execute(
                        """
                        INSERT INTO vn_asset_generation_recipes (
                            batch_id, slot_id, variant_index, recipe_json
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            batch_id,
                            int(entry["slot_id"]),
                            int(entry["variant_index"]),
                            json.dumps(dict(entry["recipe"])),
                        ),
                    )
        batch = self.get_batch(batch_id)
        if batch is None:
            raise RuntimeError("created_batch_not_found")
        return batch

    def get_batch(self, batch_id: int) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query("SELECT * FROM vn_asset_batches WHERE id = ?", (batch_id,))
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def begin_inline_legacy_display(self, batch_id: int, slot_id: int) -> None:
        """Track one inline call's exact slot locally, without admitting execution.

        This display-only count is not shared across repository instances or
        processes. Jobs-backed work uses the authoritative reader instead.
        No transaction/connection is held across the caller's await.
        """
        with self.db.transaction() as conn:
            _lock_variant(conn, batch_id, slot_id, None)
            key = (batch_id, slot_id)
            self._inline_legacy_activity[key] = self._inline_legacy_activity.get(key, 0) + 1

    def finish_legacy_display(
        self, batch_id: int, slot_id: int, *, inline: bool, fallback_status: str | None,
        finishing_delivery: tuple[int, str] | None = None,
    ) -> None:
        """Clear local execution display and reconcile mixed-version work.

        Legacy outcome/counter writes remain with the worker. Without V1
        history keep its terminal display; on coroutine cancellation derive a
        non-sticky display. Body errors raise LegacyDisplayReconciliationError
        with a safe rollback message and internal original type/traceback.
        Transaction entry/exit errors propagate unchanged.
        Local cleanup occurs even if reconciliation fails; it creates no
        persisted claim or fence.
        Exclude only this exact finishing Jobs ID/lease during SDK handoff;
        replacement leases and siblings still contribute to display.
        """
        if inline:
            key = (batch_id, slot_id)
            remaining = self._inline_legacy_activity.get(key, 0) - 1
            if remaining > 0:
                self._inline_legacy_activity[key] = remaining
            else:
                self._inline_legacy_activity.pop(key, None)
        with self.db.transaction() as conn:
            try:
                _lock_variant(conn, batch_id, slot_id, None)
                mixed = conn.execute(
                    "SELECT 1 FROM vn_asset_generation_recipes WHERE slot_id = ? LIMIT 1", (slot_id,),
                ).fetchone()
                if mixed is not None or fallback_status is None:
                    self._refresh_slot_generation_status(
                        conn, slot_id, fallback_status=fallback_status, finishing_delivery=finishing_delivery,
                    )
            except Exception as exc:  # noqa: BLE001 - display rollback must not log provider/reader messages
                raise LegacyDisplayReconciliationError(exc) from None

    def _refresh_slot_generation_status(
        self, conn: Any, slot_id: int, *, fallback_status: str | None = None,
        finishing_delivery: tuple[int, str] | None = None,
    ) -> None:
        """Read exact legacy activity after write admission, then reconcile V1.

        Failed V0 batches may still have executing children; cancelled batches
        never contribute. Published provenance settles only its exact Jobs
        delivery while completing. Aggregate counters are not liveness.
        """
        slot = conn.execute(
            """
            SELECT slot.pack_id, pack.owner_user_id FROM vn_asset_slots AS slot
            JOIN vn_asset_packs AS pack ON pack.id = slot.pack_id WHERE slot.id = ?
            """, (slot_id,),
        ).fetchone()
        if slot is None:
            return
        batches = {int(row["id"]): str(row["status"]) for row in conn.execute(
            "SELECT id, status FROM vn_asset_batches WHERE pack_id = ? AND recipe_version = 0 AND status != 'cancelled'",
            (slot["pack_id"],),
        ).fetchall()}
        active = any(self._inline_legacy_activity.get((batch_id, slot_id), 0) for batch_id in batches)
        queued = False
        if batches and self.legacy_activity_reader is not None:
            settled: set[tuple[int, int, str]] = set()
            for row in conn.execute(
                """
                SELECT source_context_snapshot_json FROM vn_asset_items AS item
                WHERE slot_id = ? AND generated_file_id IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM vn_asset_generation_recipes WHERE item_id = item.id
                )
                """, (slot_id,),
            ).fetchall():
                context = json.loads(row["source_context_snapshot_json"] or "{}")
                if isinstance(context, dict) and all(type(context.get(key)) is int for key in ("batch_id", "variant_index")):
                    fingerprint = context.get("legacy_delivery_fingerprint")
                    if isinstance(fingerprint, str) and fingerprint:
                        settled.add((context["batch_id"], context["variant_index"], fingerprint))
            jobs_active, queued = self.legacy_activity_reader(
                int(slot["pack_id"]), slot_id, int(slot["owner_user_id"]), batches, settled, finishing_delivery,
            )
            active = active or jobs_active
        _refresh_slot_generation_status(conn, slot_id, legacy_activity=(active, queued), fallback_status=fallback_status)

    def cancel_batch(self, batch_id: int) -> dict[str, Any] | None:
        """Cancel batch_id and return its row, or None if it does not exist.

        Reconcile already-cancelled V1 batches, preserving completed/failed
        recipes and counters. V0 keeps unconditional cancellation. Database
        failures propagate and roll back the transition.
        """
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            # Acquire write admission before observing the batch or its recipes.
            conn.execute("UPDATE vn_asset_batches SET status = status WHERE id = ?", (batch_id,))
            batch = conn.execute(
                "SELECT status, recipe_version FROM vn_asset_batches WHERE id = ?", (batch_id,)
            ).fetchone()
            if batch is None:
                return None
            if int(batch["recipe_version"] or 0) == 0:
                conn.execute(
                    "UPDATE vn_asset_batches SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (batch_id,),
                )
            elif batch["status"] not in {"completed", "failed"}:
                conn.execute(
                    """
                    UPDATE vn_asset_generation_recipes SET outcome_status = 'cancelled'
                    WHERE batch_id = ? AND outcome_status NOT IN ('completed', 'failed', 'cancelled')
                    """,
                    (batch_id,),
                )
                conn.execute(
                    """
                    UPDATE vn_asset_batches
                    SET status = 'cancelled',
                        cancelled_count = (
                            SELECT COUNT(*) FROM vn_asset_generation_recipes
                            WHERE batch_id = ? AND outcome_status = 'cancelled'
                        ), updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (batch_id, batch_id),
                )
                for row in conn.execute(
                    "SELECT DISTINCT slot_id FROM vn_asset_generation_recipes WHERE batch_id = ?", (batch_id,)
                ).fetchall():
                    self._refresh_slot_generation_status(conn, int(row["slot_id"]))
        return self.get_batch(batch_id)

    def list_batches(self, pack_id: int) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            "SELECT * FROM vn_asset_batches WHERE pack_id = ? ORDER BY id DESC",
            (pack_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def fail_batch_integrity(self, batch_id: int, *, error: str) -> dict[str, Any] | None:
        """Atomically fail surviving unfinished V1 recipes and release capacity.

        Increment failed_count only for this transition, preserving historical
        terminal counts even when ledger rows are missing. Reconcile every
        surviving slot without changing approved items or bytes. Cancellation
        takes precedence: leftover recipes become cancelled, adding only new
        cancellations to historical counts. Completed batches are unchanged;
        a repeated failure is idempotent.
        Missing rows do not imply new outcomes or permission to delete assets.
        Database/reconciliation failures propagate and roll back all changes.
        """
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute("UPDATE vn_asset_batches SET status=status WHERE id=?", (batch_id,))
            batch = conn.execute(
                "SELECT status, recipe_version FROM vn_asset_batches WHERE id=?", (batch_id,),
            ).fetchone()
            if batch is None:
                return None
            if int(batch["recipe_version"] or 0) != 1:
                raise VNAssetGenerationError("vn_asset_recipe_version_unsupported", batch_id=batch_id)
            if batch["status"] == "completed":
                return self.get_batch(batch_id)
            cancelled = batch["status"] == "cancelled"
            updated = conn.execute(
                """
                UPDATE vn_asset_generation_recipes SET outcome_status=?
                WHERE batch_id=? AND outcome_status NOT IN ('completed', 'failed', 'cancelled')
                """,
                ("cancelled" if cancelled else "failed", batch_id),
            ).rowcount
            if updated or batch["status"] not in {"failed", "cancelled"}:
                conn.execute(
                    """
                    UPDATE vn_asset_batches
                    SET status=CASE WHEN status='cancelled' THEN status ELSE 'failed' END,
                        enqueue_error=CASE WHEN status='cancelled' THEN enqueue_error ELSE ? END,
                        failed_count=failed_count+?, cancelled_count=cancelled_count+?,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (error, 0 if cancelled else updated, updated if cancelled else 0, batch_id),
                )
                slots = conn.execute(
                    "SELECT DISTINCT slot_id FROM vn_asset_generation_recipes WHERE batch_id=? ORDER BY slot_id",
                    (batch_id,),
                ).fetchall()
                for row in slots:
                    self._refresh_slot_generation_status(conn, int(row["slot_id"]))
        return self.get_batch(batch_id)

    def list_batch_recipes(self, batch_id: int) -> list[dict[str, Any]]:
        """Return decoded recipes for batch_id in slot/variant order.

        Returns an empty list for an unknown batch; database/JSON errors propagate.
        """
        self._ensure_schema_initialized()
        rows = self.db.execute_query(
            """
            SELECT slot_id, variant_index, recipe_json
            FROM vn_asset_generation_recipes
            WHERE batch_id = ? ORDER BY slot_id, variant_index
            """,
            (batch_id,),
        ).fetchall()
        return [
            {
                "slot_id": int(row["slot_id"]),
                "variant_index": int(row["variant_index"]),
                "recipe": json.loads(row["recipe_json"]),
            }
            for row in rows
        ]

    def get_batch_recipe(
        self, batch_id: int, slot_id: int, variant_index: int
    ) -> dict[str, Any] | None:
        """Return the decoded recipe for the supplied variant IDs, or None.

        Database and malformed persisted JSON errors propagate to the caller.
        """
        self._ensure_schema_initialized()
        row = self.db.execute_query(
            """
            SELECT recipe_json FROM vn_asset_generation_recipes
            WHERE batch_id = ? AND slot_id = ? AND variant_index = ?
            """,
            (batch_id, slot_id, variant_index),
        ).fetchone()
        return json.loads(row["recipe_json"]) if row is not None else None

    def get_variant_outcome(
        self, batch_id: int, slot_id: int, variant_index: int
    ) -> dict[str, Any] | None:
        """Return the outcome and fence for the supplied variant IDs, or None.

        This observation is not authority to mutate; database errors propagate.
        """
        self._ensure_schema_initialized()
        with closing(self.db.execute_query(_VARIANT_OUTCOME_QUERY, (batch_id, slot_id, variant_index))) as cursor:
            row = cursor.fetchone()
        return dict(row) if row is not None else None

    async def get_variant_outcome_async(
        self, batch_id: int, slot_id: int, variant_index: int,
    ) -> dict[str, Any] | None:
        """Observe normal file-backed outcomes off-thread using an owned reader.

        Only a plain dict/None crosses threads. Private memory and active caller
        transactions retain owner-thread reads to preserve connection-local
        state, so this is not a universal nonblocking I/O guarantee. Schema
        setup remains caller-owned. No cursor/transaction/pooled handle moves
        between threads or is closed by the reader; read failures propagate.
        """
        self._ensure_schema_initialized()
        if self.db.is_memory_db or self.db.get_connection().in_transaction:
            return self.get_variant_outcome(batch_id, slot_id, variant_index)
        return await asyncio.to_thread(
            _read_variant_outcome_file, self.db.db_path.as_uri(), batch_id, slot_id, variant_index,
        )

    def claim_variant(
        self,
        *,
        batch_id: int,
        slot_id: int,
        variant_index: int,
        lease_id: str,
        attempt_token: str,
        item_fields: Mapping[str, Any],
        allow_takeover: bool = False,
        expected_claim_token: str | None = None,
        validate_authority: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Admit a claim under the VN write lock with Jobs validation and token CAS.

        Variant IDs select the recipe; item_fields initialize its stable hidden
        item. lease_id/attempt_token identify this delivery. expected_claim_token
        must match the observed fence, and allow_takeover requires a fresh
        validate_authority callback. Returns the reserved item. Raises
        VNAssetGenerationError on a terminal variant, duplicate lease, changed
        fence, or unauthorized takeover; callback/database errors propagate.

        Jobs can move after admission; each later VN transition validates again.
        This deliberately does not hold a transaction across the two databases.
        """
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            _lock_variant(conn, batch_id, slot_id, variant_index)
            if validate_authority is not None:
                validate_authority()
            row = conn.execute(
                """
                SELECT recipe.item_id, recipe.outcome_status, recipe.claim_lease_id, recipe.claim_token,
                       batch.status AS batch_status
                FROM vn_asset_generation_recipes AS recipe
                JOIN vn_asset_batches AS batch ON batch.id = recipe.batch_id
                WHERE recipe.batch_id = ? AND recipe.slot_id = ? AND recipe.variant_index = ?
                """,
                (batch_id, slot_id, variant_index),
            ).fetchone()
            if row is None:
                raise VNAssetGenerationError("vn_asset_recipe_not_found", batch_id=batch_id, slot_id=slot_id)
            if row["batch_status"] in {"cancelled", "failed", "completed"}:
                raise VNAssetGenerationError("vn_asset_batch_terminal", batch_id=batch_id)
            if row["outcome_status"] != "planned":
                raise VNAssetGenerationError("vn_asset_variant_terminal", batch_id=batch_id, slot_id=slot_id)
            if row["claim_lease_id"] == lease_id or (row["claim_lease_id"] is not None and not allow_takeover):
                raise VNAssetGenerationError(
                    "vn_asset_variant_in_progress", retryable=True,
                    batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                )
            if row["claim_token"] != expected_claim_token:
                raise VNAssetGenerationError(
                    "vn_asset_variant_claim_changed", retryable=True,
                    batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                )
            if allow_takeover and validate_authority is None:
                raise VNAssetGenerationError(
                    "vn_asset_job_lease_lost", retryable=True, batch_id=batch_id,
                )
            if row["item_id"] is None:
                item = self.create_item(
                    slot_id=slot_id, variant_index=variant_index,
                    review_status="hidden", **dict(item_fields),
                )
                item_id = int(item["id"])
            else:
                item_id = int(row["item_id"])
                item = self.get_item(item_id)
                if item is None:
                    raise VNAssetGenerationError("vn_asset_recipe_item_missing", batch_id=batch_id, item_id=item_id)
            conn.execute(
                """
                UPDATE vn_asset_generation_recipes
                SET item_id = ?, claim_lease_id = ?, claim_token = ?
                WHERE batch_id = ? AND slot_id = ? AND variant_index = ?
                """,
                (item_id, lease_id, attempt_token, batch_id, slot_id, variant_index),
            )
        return item

    def start_variant_generation(
        self, *, batch_id: int, slot_id: int, variant_index: int, attempt_token: str,
        validate_authority: Callable[[], None] | None = None,
    ) -> None:
        """Admit V1 generating state for a current claim under the VN write lock.

        Variant IDs and attempt_token must identify a planned recipe in a
        nonterminal batch. Jobs claims require validate_authority; inline claims
        need only their current token. The callback runs after write admission
        and before any visible slot change. Reconcile all work sharing the slot
        and clear its error. Raises retryable VNAssetGenerationError on lost
        authority, fence or terminal batch; callback/database errors propagate
        and roll back. Jobs changes after admission are not retroactive.
        """
        self._ensure_schema_initialized()
        context = {"batch_id": batch_id, "slot_id": slot_id, "variant_index": variant_index}
        with self.db.transaction() as conn:
            _lock_variant(conn, batch_id, slot_id, variant_index)
            if validate_authority is not None:
                validate_authority()
            row = conn.execute(
                """
                SELECT recipe.outcome_status, recipe.claim_token, recipe.claim_lease_id,
                       batch.status AS batch_status
                FROM vn_asset_generation_recipes AS recipe
                JOIN vn_asset_batches AS batch ON batch.id = recipe.batch_id
                WHERE recipe.batch_id = ? AND recipe.slot_id = ? AND recipe.variant_index = ?
                """,
                (batch_id, slot_id, variant_index),
            ).fetchone()
            if row is not None and row["batch_status"] in {"completed", "failed", "cancelled"}:
                raise VNAssetGenerationError("vn_asset_batch_terminal", retryable=True, **context)
            if row is None or row["outcome_status"] != "planned" or row["claim_token"] != attempt_token:
                raise VNAssetGenerationError("vn_asset_variant_claim_lost", retryable=True, **context)
            if row["claim_lease_id"] != "inline" and validate_authority is None:
                raise VNAssetGenerationError("vn_asset_job_lease_lost", retryable=True, **context)
            self._refresh_slot_generation_status(conn, slot_id)
            conn.execute(
                "UPDATE vn_asset_slots SET last_error = NULL WHERE id = ?", (slot_id,),
            )

    def reserve_variant_item(
        self,
        *,
        batch_id: int,
        slot_id: int,
        variant_index: int,
        item_fields: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return a stable hidden item for the supplied legacy variant IDs.

        item_fields initializes the first reservation. Raises VNAssetGenerationError
        for missing recipes/items or terminal state; database errors propagate.
        """
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            row = conn.execute(
                """
                SELECT item_id, outcome_status FROM vn_asset_generation_recipes
                WHERE batch_id = ? AND slot_id = ? AND variant_index = ?
                """,
                (batch_id, slot_id, variant_index),
            ).fetchone()
            if row is None:
                raise VNAssetGenerationError("vn_asset_recipe_not_found", batch_id=batch_id, slot_id=slot_id)
            batch = conn.execute(
                "SELECT status FROM vn_asset_batches WHERE id = ?", (batch_id,)
            ).fetchone()
            if batch is None or batch["status"] in {"cancelled", "failed", "completed"}:
                raise VNAssetGenerationError("vn_asset_batch_terminal", batch_id=batch_id)
            if row["outcome_status"] in {"cancelled", "failed"}:
                raise VNAssetGenerationError("vn_asset_variant_terminal", batch_id=batch_id, slot_id=slot_id)
            if row["item_id"] is None:
                item = self.create_item(
                    slot_id=slot_id,
                    variant_index=variant_index,
                    review_status="hidden",
                    **dict(item_fields),
                )
                item_id = int(item["id"])
                conn.execute(
                    """
                    UPDATE vn_asset_generation_recipes SET item_id = ?
                    WHERE batch_id = ? AND slot_id = ? AND variant_index = ?
                    """,
                    (item_id, batch_id, slot_id, variant_index),
                )
            else:
                item_id = int(row["item_id"])
                item = self.get_item(item_id)
                if item is None:
                    raise VNAssetGenerationError("vn_asset_recipe_item_missing", batch_id=batch_id, item_id=item_id)
        return item

    def release_variant_claim(
        self, *, batch_id: int, slot_id: int, variant_index: int, attempt_token: str,
    ) -> None:
        """Release attempt_token for the supplied variant IDs, returning None.

        Only a planned inline claim can be released; stale tokens are a no-op.
        Database failures propagate.
        """
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            released = conn.execute(
                """
                UPDATE vn_asset_generation_recipes SET claim_lease_id = NULL, claim_token = NULL
                WHERE batch_id = ? AND slot_id = ? AND variant_index = ?
                  AND claim_token = ? AND claim_lease_id = 'inline' AND outcome_status = 'planned'
                """,
                (batch_id, slot_id, variant_index, attempt_token),
            )
            if released.rowcount:
                self._refresh_slot_generation_status(conn, slot_id)

    def complete_variant(
        self, *, batch_id: int, slot_id: int, variant_index: int, item_id: int,
        attempt_token: str | None = None,
        validate_authority: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Publish item_id for the supplied variant IDs and return the item.

        attempt_token must match the fence (both None for legacy reservations).
        validate_authority admits publication after the write lock. Completed
        outcomes replay without mutation. Raises VNAssetGenerationError for
        identity, state, fence, or storage failures; callback/DB errors propagate.
        """
        self._ensure_schema_initialized()
        context = {
            "batch_id": batch_id, "slot_id": slot_id, "variant_index": variant_index,
            "item_id": item_id, "operation": "complete_variant",
        }
        with self.db.transaction() as conn:
            _lock_variant(conn, batch_id, slot_id, variant_index)
            row = conn.execute(
                """
                SELECT item_id, outcome_status, claim_token FROM vn_asset_generation_recipes
                WHERE batch_id = ? AND slot_id = ? AND variant_index = ?
                """,
                (batch_id, slot_id, variant_index),
            ).fetchone()
            if row is None or int(row["item_id"] or 0) != item_id:
                raise VNAssetGenerationError("vn_asset_recipe_item_mismatch", **context)
            if row["outcome_status"] in {"failed", "cancelled"}:
                raise VNAssetGenerationError("vn_asset_variant_failed", **context)
            if row["outcome_status"] == "completed":
                result = self.get_item(item_id)
                if result is None:
                    raise VNAssetGenerationError("vn_asset_recipe_item_missing", **context)
                return result
            if validate_authority is not None:
                validate_authority()
            if row["claim_token"] != attempt_token:
                raise VNAssetGenerationError(
                    "vn_asset_variant_claim_lost", retryable=True,
                    **context,
                )
            batch = conn.execute(
                "SELECT status FROM vn_asset_batches WHERE id = ?", (batch_id,)
            ).fetchone()
            if batch is None or batch["status"] in {"cancelled", "failed"}:
                raise VNAssetGenerationError("vn_asset_batch_terminal", **context)
            item = conn.execute(
                "SELECT generated_file_id FROM vn_asset_items WHERE id = ?",
                (item_id,),
            ).fetchone()
            if item is None or item["generated_file_id"] is None:
                raise VNAssetGenerationError("vn_asset_item_storage_missing", **context)
            conn.execute(
                "UPDATE vn_asset_items SET review_status = 'draft' WHERE id = ?",
                (item_id,),
            )
            conn.execute(
                """
                UPDATE vn_asset_generation_recipes SET outcome_status = 'completed'
                WHERE batch_id = ? AND slot_id = ? AND variant_index = ?
                """,
                (batch_id, slot_id, variant_index),
            )
            _refresh_batch_outcome_counts(conn, batch_id)
            self._refresh_slot_generation_status(conn, slot_id)
            conn.execute("UPDATE vn_asset_slots SET last_error = NULL WHERE id = ?", (slot_id,))
        result = self.get_item(item_id)
        if result is None:
            raise VNAssetGenerationError("completed_item_not_found", retryable=True, **context)
        return result

    def fail_variant(
        self, *, batch_id: int, slot_id: int, variant_index: int, error: str,
        attempt_token: str | None = None,
        validate_authority: Callable[[], None] | None = None,
    ) -> None:
        """Fail the supplied variant IDs with error and return None.

        attempt_token fences claimed work; None only admits unclaimed legacy
        reservations. validate_authority runs under the write lock. Terminal or
        stale outcomes are a no-op; callback/database failures propagate.
        """
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            _lock_variant(conn, batch_id, slot_id, variant_index)
            if validate_authority is not None:
                validate_authority()
            updated = conn.execute(
                """
                UPDATE vn_asset_generation_recipes SET outcome_status = 'failed'
                WHERE batch_id = ? AND slot_id = ? AND variant_index = ?
                  AND outcome_status NOT IN ('completed', 'failed', 'cancelled')
                  AND ((? IS NULL AND claim_token IS NULL) OR claim_token = ?)
                  AND EXISTS (
                      SELECT 1 FROM vn_asset_batches
                      WHERE id = ? AND status NOT IN ('cancelled', 'failed', 'completed')
                  )
                """,
                (batch_id, slot_id, variant_index, attempt_token, attempt_token, batch_id),
            )
            if not updated.rowcount:
                return
            _refresh_batch_outcome_counts(conn, batch_id)
            self._refresh_slot_generation_status(conn, slot_id)
            conn.execute("UPDATE vn_asset_slots SET last_error = ? WHERE id = ?", (error, slot_id))

    def mark_batch_enqueued(
        self,
        batch_id: int,
        *,
        planned_count: int,
        enqueued_count: int,
        total_slots: int,
    ) -> None:
        """Store batch_id's planned/enqueued/slot counts and return None.

        Child progress and terminal state are preserved; database errors propagate.
        """
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE vn_asset_batches
                SET status = CASE WHEN status IN ('queued', 'enqueued')
                                  THEN 'enqueued' ELSE status END,
                    planned_count = ?, enqueued_count = ?, enqueue_error = NULL,
                    total_slots = ?, total_variants = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (planned_count, enqueued_count, total_slots, planned_count, batch_id),
            )

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


def _read_variant_outcome_file(
    database_uri: str, batch_id: int, slot_id: int, variant_index: int,
) -> dict[str, Any] | None:
    """Open, read and close an independent read-only SQLite handle on this thread.

    This repository rejects non-SQLite backends before reaching this boundary.
    Match SQLiteBackend's 10-second busy timeout, without changing journal mode.
    No pool/checkpoint/global lifecycle effects; caller connections stay open.
    Return only detached row data; missing files and native read errors propagate.
    """
    with closing(sqlite3.connect(f"{database_uri}?mode=ro", uri=True, timeout=10.0)) as conn:
        conn.row_factory = sqlite3.Row
        with closing(conn.execute(_VARIANT_OUTCOME_QUERY, (batch_id, slot_id, variant_index))) as cursor:
            row = cursor.fetchone()
        return dict(row) if row is not None else None


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
    """Add missing fanout/receipt columns on conn; return None, propagating SQL errors."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(vn_asset_batches)").fetchall()}
    additions = {
        "planned_count": "ALTER TABLE vn_asset_batches ADD COLUMN planned_count INTEGER NOT NULL DEFAULT 0",
        "enqueued_count": "ALTER TABLE vn_asset_batches ADD COLUMN enqueued_count INTEGER NOT NULL DEFAULT 0",
        "enqueue_error": "ALTER TABLE vn_asset_batches ADD COLUMN enqueue_error TEXT",
        "recipe_version": "ALTER TABLE vn_asset_batches ADD COLUMN recipe_version INTEGER NOT NULL DEFAULT 0",
    }
    for column_name, statement in additions.items():
        if column_name not in columns:
            conn.execute(statement)

    idempotency_columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(vn_asset_idempotency_records)").fetchall()
    }
    if "status" not in idempotency_columns:
        conn.execute(
            "ALTER TABLE vn_asset_idempotency_records "
            "ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'"
        )
    if "batch_id" not in idempotency_columns:
        conn.execute(
            "ALTER TABLE vn_asset_idempotency_records "
            "ADD COLUMN batch_id INTEGER REFERENCES vn_asset_batches(id)"
        )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_vn_asset_idempotency_batch_id "
        "ON vn_asset_idempotency_records(batch_id) WHERE batch_id IS NOT NULL"
    )


def _ensure_recipe_outcome_columns(conn: Any) -> None:
    """Add missing recipe outcome/fence columns on conn; SQL failures propagate."""
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(vn_asset_generation_recipes)").fetchall()
    }
    if "outcome_status" not in columns:
        conn.execute(
            "ALTER TABLE vn_asset_generation_recipes "
            "ADD COLUMN outcome_status TEXT NOT NULL DEFAULT 'planned'"
        )
    if "item_id" not in columns:
        conn.execute("ALTER TABLE vn_asset_generation_recipes ADD COLUMN item_id INTEGER")
    if "claim_lease_id" not in columns:
        conn.execute("ALTER TABLE vn_asset_generation_recipes ADD COLUMN claim_lease_id TEXT")
    if "claim_token" not in columns:
        conn.execute("ALTER TABLE vn_asset_generation_recipes ADD COLUMN claim_token TEXT")


def _lock_variant(conn: Any, batch_id: int | None, slot_id: int | None, variant_index: int | None) -> None:
    """Lock the variant selected by its IDs on conn; return None.

    The SQLite write lock serializes VN transitions, including cancellation.
    Missing IDs update no recipe; SQLite execution/locking failures propagate.
    """
    conn.execute(
        """
        UPDATE vn_asset_generation_recipes SET claim_token = claim_token
        WHERE batch_id = ? AND slot_id = ? AND variant_index = ?
        """,
        (batch_id, slot_id, variant_index),
    )


def _refresh_slot_generation_status(
    conn: Any, slot_id: int, *, legacy_activity: tuple[bool, bool] = (False, False),
    fallback_status: str | None = None,
) -> None:
    """Reconcile slot_id on the caller's write-locked SQLite transaction.

    Legacy activity comes from the write-admitted caller's Jobs reader/local
    execution display, not aggregate batch counters. V1 nonterminal batches
    contribute planned work: claimed recipes generate,
    unclaimed recipes queue. Published items use the same visibility predicate
    as list_items, so terminal reservations never become review candidates.
    Existing review precedence applies after work ends; failures with no
    published candidates are failed, even alongside cancellations. Historical
    completed recipes without items do not manufacture review readiness.
    A legacy terminal fallback fills only planned/cancelled display, never
    overriding review, skipped, active/queued work or a derived failure.
    Missing slots are a no-op; SQL errors propagate and roll back the caller's
    outcome transition. This repository does not support row-lock backends.
    """
    slot = conn.execute(
        "SELECT status, required_for_runtime FROM vn_asset_slots WHERE id = ?", (slot_id,),
    ).fetchone()
    if slot is None:
        return
    counts = conn.execute(
        """
        SELECT SUM(CASE WHEN recipe.outcome_status = 'planned'
                       AND batch.status NOT IN ('completed', 'failed', 'cancelled')
                       AND recipe.claim_token IS NOT NULL THEN 1 ELSE 0 END) AS active,
               SUM(CASE WHEN recipe.outcome_status = 'planned'
                       AND batch.status NOT IN ('completed', 'failed', 'cancelled')
                       AND recipe.claim_token IS NULL THEN 1 ELSE 0 END) AS queued,
               SUM(CASE WHEN recipe.outcome_status = 'failed' THEN 1 ELSE 0 END) AS failed,
               SUM(CASE WHEN recipe.outcome_status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled
        FROM vn_asset_generation_recipes AS recipe
        JOIN vn_asset_batches AS batch ON batch.id = recipe.batch_id
        WHERE recipe.slot_id = ?
        """,
        (slot_id,),
    ).fetchone()
    statuses = [row["review_status"] for row in conn.execute(
        """
        SELECT item.review_status FROM vn_asset_items AS item
        WHERE item.slot_id = ? AND NOT EXISTS (
            SELECT 1 FROM vn_asset_generation_recipes AS recipe
            WHERE recipe.item_id = item.id AND recipe.outcome_status != 'completed'
        )
        """,
        (slot_id,),
    ).fetchall()]
    failed = int(counts["failed"] or 0)
    active = bool(counts["active"]) or legacy_activity[0]
    queued = bool(counts["queued"]) or legacy_activity[1]
    status = derive_slot_status(
        has_active_job=active,
        has_queued_job=queued,
        is_skipped=slot["status"] == "skipped",
        is_cancelled=bool(counts["cancelled"]),
        requested_variants=failed + len(statuses),
        failed_variants=failed,
        review_statuses=statuses,
        required_for_runtime=bool(slot["required_for_runtime"]),
    )
    if fallback_status is not None and status in {"planned", "cancelled"}:
        status = fallback_status
    conn.execute(
        "UPDATE vn_asset_slots SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (status, slot_id),
    )


def _refresh_batch_outcome_counts(conn: Any, batch_id: int) -> None:
    """Recount batch_id outcomes on conn; return None, propagating SQL errors.

    Terminal batches are never reopened by derived progress.
    """
    counts = conn.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN outcome_status = 'completed' THEN 1 ELSE 0 END) AS completed,
               SUM(CASE WHEN outcome_status = 'failed' THEN 1 ELSE 0 END) AS failed
        FROM vn_asset_generation_recipes WHERE batch_id = ?
        """,
        (batch_id,),
    ).fetchone()
    completed = int(counts["completed"] or 0)
    failed = int(counts["failed"] or 0)
    total = int(counts["total"] or 0)
    conn.execute(
        """
        UPDATE vn_asset_batches
        SET completed_count = ?, failed_count = ?,
            status = CASE
                WHEN status IN ('cancelled', 'failed') THEN status
                WHEN ? > 0 AND ? + ? >= ? THEN 'failed'
                WHEN ? > 0 AND ? >= ? THEN 'completed'
                ELSE 'processing'
            END,
            completed_at = CASE WHEN status NOT IN ('cancelled', 'failed')
                AND ? > 0 AND ? >= ? AND ? = 0
                THEN CURRENT_TIMESTAMP ELSE completed_at END,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            completed, failed,
            failed, completed, failed, total,
            total, completed, total,
            total, completed, total, failed,
            batch_id,
        ),
    )

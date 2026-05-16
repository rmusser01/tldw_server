"""Persona visual pack portability metadata storage."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType


PERSONA_VISUAL_PORTABILITY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS persona_visual_pack_portability_jobs (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL UNIQUE,
    owner_user_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    persona_id TEXT,
    pack_id TEXT REFERENCES persona_visual_packs(id) ON DELETE SET NULL,
    preview_id TEXT,
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

CREATE TABLE IF NOT EXISTS persona_visual_pack_import_previews (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'queued',
    target_persona_id TEXT,
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
    target_warnings_json TEXT NOT NULL DEFAULT '[]',
    error_code TEXT,
    error_message TEXT,
    expires_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_persona_visual_portability_jobs_owner_user_id
    ON persona_visual_pack_portability_jobs(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_persona_visual_portability_jobs_job_id
    ON persona_visual_pack_portability_jobs(job_id);
CREATE INDEX IF NOT EXISTS idx_persona_visual_portability_jobs_status
    ON persona_visual_pack_portability_jobs(status);
CREATE INDEX IF NOT EXISTS idx_persona_visual_portability_jobs_pack
    ON persona_visual_pack_portability_jobs(owner_user_id, persona_id, pack_id);
CREATE INDEX IF NOT EXISTS idx_persona_visual_portability_jobs_expires_at
    ON persona_visual_pack_portability_jobs(expires_at);
CREATE INDEX IF NOT EXISTS idx_persona_visual_portability_jobs_fingerprint
    ON persona_visual_pack_portability_jobs(canonical_payload_fingerprint);

CREATE INDEX IF NOT EXISTS idx_persona_visual_import_previews_owner_user_id
    ON persona_visual_pack_import_previews(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_persona_visual_import_previews_job_id
    ON persona_visual_pack_import_previews(job_id);
CREATE INDEX IF NOT EXISTS idx_persona_visual_import_previews_status
    ON persona_visual_pack_import_previews(status);
CREATE INDEX IF NOT EXISTS idx_persona_visual_import_previews_target_persona
    ON persona_visual_pack_import_previews(owner_user_id, target_persona_id);
CREATE INDEX IF NOT EXISTS idx_persona_visual_import_previews_expires_at
    ON persona_visual_pack_import_previews(expires_at);
CREATE INDEX IF NOT EXISTS idx_persona_visual_import_previews_fingerprint
    ON persona_visual_pack_import_previews(canonical_payload_fingerprint);
"""


PERSONA_VISUAL_PORTABILITY_SCHEMA_STATEMENTS = tuple(
    statement.strip()
    for statement in PERSONA_VISUAL_PORTABILITY_SCHEMA_SQL.split(";")
    if statement.strip()
)


def ensure_persona_visual_portability_tables(db: CharactersRAGDB) -> None:
    """Create persona visual portability tables in a ChaChaNotes database."""
    _require_sqlite_chacha_db(db)
    with db.transaction() as conn:
        for statement in PERSONA_VISUAL_PORTABILITY_SCHEMA_STATEMENTS:
            conn.execute(statement)


class PersonaVisualPortabilityRepository:
    """Repository for persona visual export and import-preview bookkeeping."""

    def __init__(self, db: CharactersRAGDB):
        _require_sqlite_chacha_db(db)
        self.db = db
        self._schema_initialized = False

    @classmethod
    def initialized(cls, db: CharactersRAGDB) -> PersonaVisualPortabilityRepository:
        repo = cls(db)
        repo.initialize_schema()
        return repo

    def initialize_schema(self) -> None:
        ensure_persona_visual_portability_tables(self.db)
        self._schema_initialized = True

    def create_portability_job(
        self,
        *,
        owner_user_id: str,
        job_id: str,
        operation: str,
        status: str,
        stage: str,
        persona_id: str | None = None,
        pack_id: str | None = None,
        preview_id: str | None = None,
        archive_path: str | None = None,
        archive_sha256: str | None = None,
        canonical_payload_fingerprint: str | None = None,
        progress: Mapping[str, Any] | None = None,
        warnings: list[Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        download_url: str | None = None,
        expires_at: str | None = None,
        portability_job_id: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        row_id = str(portability_job_id or uuid.uuid4())
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO persona_visual_pack_portability_jobs (
                    id,
                    job_id,
                    owner_user_id,
                    operation,
                    status,
                    stage,
                    persona_id,
                    pack_id,
                    preview_id,
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
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    str(job_id),
                    str(owner_user_id),
                    str(operation),
                    str(status),
                    str(stage),
                    _text_or_none(persona_id),
                    _text_or_none(pack_id),
                    _text_or_none(preview_id),
                    _text_or_none(archive_path),
                    _text_or_none(archive_sha256),
                    _text_or_none(canonical_payload_fingerprint),
                    _json_dump(progress or {}),
                    _json_dump(warnings or []),
                    _text_or_none(error_code),
                    _text_or_none(error_message),
                    _text_or_none(download_url),
                    _text_or_none(expires_at),
                ),
            )

        job = self.get_portability_job(row_id, owner_user_id=owner_user_id)
        if job is None:
            raise RuntimeError("created_persona_visual_portability_job_not_found")
        return job

    def get_portability_job(
        self,
        portability_job_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        if owner_user_id is None:
            cursor = self.db.execute_query(
                "SELECT * FROM persona_visual_pack_portability_jobs WHERE id = ?",
                (str(portability_job_id),),
            )
        else:
            cursor = self.db.execute_query(
                """
                SELECT * FROM persona_visual_pack_portability_jobs
                WHERE id = ? AND owner_user_id = ?
                """,
                (str(portability_job_id), str(owner_user_id)),
            )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def get_portability_job_by_job_id(
        self,
        job_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        if owner_user_id is None:
            cursor = self.db.execute_query(
                "SELECT * FROM persona_visual_pack_portability_jobs WHERE job_id = ?",
                (str(job_id),),
            )
        else:
            cursor = self.db.execute_query(
                """
                SELECT * FROM persona_visual_pack_portability_jobs
                WHERE job_id = ? AND owner_user_id = ?
                """,
                (str(job_id), str(owner_user_id)),
            )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def update_portability_job(
        self,
        job_id: str,
        fields: Mapping[str, Any],
        *,
        owner_user_id: str | None = None,
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
            where_params: list[Any] = [str(job_id)]
            if owner_user_id is not None:
                where_clause += " AND owner_user_id = ?"
                where_params.append(str(owner_user_id))
            _execute_portability_update(
                conn,
                table_name="persona_visual_pack_portability_jobs",
                update_values=update_values,
                where_clause=where_clause,
                where_params=where_params,
            )
        return self.get_portability_job_by_job_id(job_id, owner_user_id=owner_user_id)

    def create_import_preview(
        self,
        *,
        owner_user_id: str,
        job_id: str,
        status: str,
        archive_path: str,
        stage: str = "queued",
        target_persona_id: str | None = None,
        archive_sha256: str | None = None,
        canonical_payload_fingerprint: str | None = None,
        schema_version: str | None = None,
        bundle_summary: Mapping[str, Any] | None = None,
        validation_warnings: list[Any] | None = None,
        conflicts: list[Any] | None = None,
        proposed_plan: Mapping[str, Any] | None = None,
        quota_estimate: Mapping[str, Any] | None = None,
        required_choices: list[Any] | None = None,
        target_warnings: list[Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        expires_at: str | None = None,
        preview_id: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        row_id = str(preview_id or uuid.uuid4())
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO persona_visual_pack_import_previews (
                    id,
                    owner_user_id,
                    job_id,
                    status,
                    stage,
                    target_persona_id,
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
                    target_warnings_json,
                    error_code,
                    error_message,
                    expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    str(owner_user_id),
                    str(job_id),
                    str(status),
                    str(stage),
                    _text_or_none(target_persona_id),
                    str(archive_path),
                    _text_or_none(archive_sha256),
                    _text_or_none(canonical_payload_fingerprint),
                    _text_or_none(schema_version),
                    _json_dump(bundle_summary or {}),
                    _json_dump(validation_warnings or []),
                    _json_dump(conflicts or []),
                    _json_dump(proposed_plan or {}),
                    _json_dump(quota_estimate or {}),
                    _json_dump(required_choices or []),
                    _json_dump(target_warnings or []),
                    _text_or_none(error_code),
                    _text_or_none(error_message),
                    _text_or_none(expires_at),
                ),
            )

        preview = self.get_import_preview(row_id, owner_user_id=owner_user_id)
        if preview is None:
            raise RuntimeError("created_persona_visual_import_preview_not_found")
        return preview

    def get_import_preview(
        self,
        preview_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        if owner_user_id is None:
            cursor = self.db.execute_query(
                "SELECT * FROM persona_visual_pack_import_previews WHERE id = ?",
                (str(preview_id),),
            )
        else:
            cursor = self.db.execute_query(
                """
                SELECT * FROM persona_visual_pack_import_previews
                WHERE id = ? AND owner_user_id = ?
                """,
                (str(preview_id), str(owner_user_id)),
            )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def update_import_preview(
        self,
        preview_id: str,
        fields: Mapping[str, Any],
        *,
        owner_user_id: str | None = None,
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
            where_params: list[Any] = [str(preview_id)]
            if owner_user_id is not None:
                where_clause += " AND owner_user_id = ?"
                where_params.append(str(owner_user_id))
            _execute_portability_update(
                conn,
                table_name="persona_visual_pack_import_previews",
                update_values=update_values,
                where_clause=where_clause,
                where_params=where_params,
            )
        return self.get_import_preview(preview_id, owner_user_id=owner_user_id)

    def replace_import_preview_proposed_plan_json(
        self,
        preview_id: str,
        proposed_plan_json: str | bytes,
        *,
        owner_user_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Overwrite stored preview plan JSON text through the repository layer."""
        self._ensure_schema_initialized()
        current = self.get_import_preview(preview_id, owner_user_id=owner_user_id)
        if current is None:
            return None

        with self.db.transaction() as conn:
            where_clause = "id = ?"
            where_params: list[Any] = [str(preview_id)]
            if owner_user_id is not None:
                where_clause += " AND owner_user_id = ?"
                where_params.append(str(owner_user_id))
            _execute_portability_update(
                conn,
                table_name="persona_visual_pack_import_previews",
                update_values=[("proposed_plan_json", proposed_plan_json)],
                where_clause=where_clause,
                where_params=where_params,
            )
        return self.get_import_preview(preview_id, owner_user_id=owner_user_id)

    def _ensure_schema_initialized(self) -> None:
        if not self._schema_initialized:
            self.initialize_schema()


_PORTABILITY_JOB_UPDATE_COLUMNS = {
    "operation": "operation",
    "status": "status",
    "stage": "stage",
    "persona_id": "persona_id",
    "pack_id": "pack_id",
    "preview_id": "preview_id",
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
    "stage": "stage",
    "target_persona_id": "target_persona_id",
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
    "target_warnings": "target_warnings_json",
    "error_code": "error_code",
    "error_message": "error_message",
    "expires_at": "expires_at",
}

_IMPORT_PREVIEW_JSON_DEFAULTS = {
    "bundle_summary": {},
    "validation_warnings": [],
    "conflicts": [],
    "proposed_plan": {},
    "quota_estimate": {},
    "required_choices": [],
    "target_warnings": [],
}

_PORTABILITY_UPDATE_TABLE_COLUMNS = {
    "persona_visual_pack_portability_jobs": frozenset(_PORTABILITY_JOB_UPDATE_COLUMNS.values()),
    "persona_visual_pack_import_previews": frozenset(_IMPORT_PREVIEW_UPDATE_COLUMNS.values()),
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
            value = _text_or_none(raw_value)
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
        raise ValueError("unsupported_persona_visual_portability_update_table")
    if any(column_name not in allowed_columns for column_name, _ in update_values):
        raise ValueError("unsupported_persona_visual_portability_update_column")

    assignments = ", ".join(f"{column_name} = ?" for column_name, _ in update_values)
    statement = (
        f"UPDATE {table_name} "  # nosec B608
        f"SET {assignments}, updated_at = CURRENT_TIMESTAMP "
        f"WHERE {where_clause}"
    )
    conn.execute(
        statement,
        tuple(value for _, value in update_values) + tuple(where_params),
    )


def _json_dump(value: Any) -> str:
    return json.dumps(value)


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _require_sqlite_chacha_db(db: CharactersRAGDB) -> None:
    if getattr(db, "backend_type", None) != BackendType.SQLITE:
        raise NotImplementedError(
            "Persona visual portability metadata currently supports SQLite ChaChaNotes databases only."
        )


__all__ = [
    "PERSONA_VISUAL_PORTABILITY_SCHEMA_SQL",
    "PersonaVisualPortabilityRepository",
    "ensure_persona_visual_portability_tables",
]

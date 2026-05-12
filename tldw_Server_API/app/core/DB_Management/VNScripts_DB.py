"""VN script metadata storage for per-user ChaChaNotes databases."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNPolicy_DB import ensure_vn_profile_snapshot_tables
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

_GENERATION_PROFILE_KEY_RE = re.compile(r"^[a-z0-9_.-]{1,64}$")
_MAX_GENERATION_PROFILE_MAP_SIZE = 16
_MAX_GENERATION_PROFILE_ID_LENGTH = 80


VN_SCRIPTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS vn_scripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    primary_asset_pack_id INTEGER NOT NULL,
    policy_profile_id TEXT NOT NULL DEFAULT 'local_default',
    generation_profile_id TEXT NOT NULL DEFAULT 'story_default',
    generation_profile_ids_json TEXT NOT NULL DEFAULT '{}',
    content_rating TEXT NOT NULL DEFAULT 'general',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS vn_script_drafts (
    script_id INTEGER PRIMARY KEY REFERENCES vn_scripts(id) ON DELETE CASCADE,
    owner_user_id INTEGER NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    draft_json TEXT NOT NULL DEFAULT '{}',
    diagnostics_json TEXT NOT NULL DEFAULT '{"valid": true, "errors": [], "warnings": []}',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_script_manifest_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    script_id INTEGER NOT NULL REFERENCES vn_scripts(id) ON DELETE CASCADE,
    version_id INTEGER,
    asset_pack_id INTEGER NOT NULL,
    manifest_json TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_script_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    script_id INTEGER NOT NULL REFERENCES vn_scripts(id) ON DELETE CASCADE,
    owner_user_id INTEGER NOT NULL,
    version_number INTEGER NOT NULL,
    label TEXT,
    draft_revision INTEGER NOT NULL,
    program_json TEXT NOT NULL,
    asset_pack_id INTEGER NOT NULL,
    manifest_snapshot_id INTEGER NOT NULL REFERENCES vn_script_manifest_snapshots(id),
    policy_snapshot_id INTEGER NOT NULL,
    generation_profile_snapshot_id INTEGER NOT NULL,
    generation_profile_snapshots_json TEXT NOT NULL DEFAULT '{}',
    script_defaults_json TEXT NOT NULL DEFAULT '{}',
    validation_json TEXT NOT NULL DEFAULT '{}',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(script_id, version_number)
);

CREATE TABLE IF NOT EXISTS vn_script_publish_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    script_id INTEGER NOT NULL REFERENCES vn_scripts(id) ON DELETE CASCADE,
    idempotency_key TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    request_payload_json TEXT,
    response_json TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(owner_user_id, script_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_vn_scripts_owner_user_id
    ON vn_scripts(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_vn_scripts_owner_status
    ON vn_scripts(owner_user_id, status);
CREATE INDEX IF NOT EXISTS idx_vn_script_versions_script_id
    ON vn_script_versions(script_id);
CREATE INDEX IF NOT EXISTS idx_vn_script_manifest_snapshots_script_id
    ON vn_script_manifest_snapshots(script_id);
CREATE INDEX IF NOT EXISTS idx_vn_script_publish_requests_lookup
    ON vn_script_publish_requests(owner_user_id, script_id, idempotency_key);
"""

VN_SCRIPTS_SCHEMA_STATEMENTS = tuple(
    statement.strip()
    for statement in VN_SCRIPTS_SCHEMA_SQL.split(";")
    if statement.strip()
)


def ensure_vn_scripts_tables(db: CharactersRAGDB) -> None:
    """Create VN script tables and profile snapshot tables in a user's DB."""
    _require_sqlite_chacha_db(db)
    ensure_vn_profile_snapshot_tables(db)
    with db.transaction() as conn:
        for statement in VN_SCRIPTS_SCHEMA_STATEMENTS:
            conn.execute(statement)
        _ensure_column(conn, "vn_scripts", "generation_profile_ids_json", "TEXT NOT NULL DEFAULT '{}'")
        _ensure_column(conn, "vn_script_versions", "generation_profile_snapshots_json", "TEXT NOT NULL DEFAULT '{}'")
        _ensure_column(conn, "vn_script_publish_requests", "request_payload_json", "TEXT")


class VNScriptsRepository:
    """Repository for authored VN scripts in a user's ChaChaNotes DB."""

    def __init__(self, db: CharactersRAGDB):
        _require_sqlite_chacha_db(db)
        self.db = db
        self._schema_initialized = False

    @classmethod
    def initialized(cls, db: CharactersRAGDB) -> "VNScriptsRepository":
        repo = cls(db)
        repo.initialize_schema()
        return repo

    def initialize_schema(self) -> None:
        ensure_vn_scripts_tables(self.db)
        self._schema_initialized = True

    def create_script(
        self,
        *,
        owner_user_id: int,
        title: str,
        primary_asset_pack_id: int,
        policy_profile_id: str,
        generation_profile_id: str,
        generation_profiles: Mapping[str, str] | None = None,
        description: str | None = None,
        content_rating: str = "general",
        status: str = "draft",
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO vn_scripts (
                    owner_user_id,
                    title,
                    description,
                    status,
                    primary_asset_pack_id,
                    policy_profile_id,
                    generation_profile_id,
                    generation_profile_ids_json,
                    content_rating
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    title,
                    description,
                    status,
                    primary_asset_pack_id,
                    policy_profile_id,
                    generation_profile_id,
                    _json_dump(_normalize_generation_profile_ids(generation_profile_id, generation_profiles)),
                    content_rating,
                ),
            )
            script_id = int(cursor.lastrowid)
            conn.execute(
                """
                INSERT INTO vn_script_drafts (
                    script_id,
                    owner_user_id,
                    revision,
                    draft_json,
                    diagnostics_json
                )
                VALUES (?, ?, 0, '{}', ?)
                """,
                (script_id, owner_user_id, _json_dump({"valid": True, "errors": [], "warnings": []})),
            )

        script = self.get_script(script_id, owner_user_id=owner_user_id)
        if script is None:
            raise RuntimeError("created_script_not_found")
        return script

    def get_script(
        self,
        script_id: int,
        *,
        owner_user_id: int | None = None,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        conditions = ["id = ?"]
        params: list[Any] = [script_id]
        if owner_user_id is not None:
            conditions.append("owner_user_id = ?")
            params.append(owner_user_id)
        if not include_deleted:
            conditions.append("deleted = 0")
        cursor = self.db.execute_query(
            f"SELECT * FROM vn_scripts WHERE {' AND '.join(conditions)}",  # nosec B608
            tuple(params),
        )
        row = cursor.fetchone()
        return self._script_with_draft(row) if row is not None else None

    def list_scripts(
        self,
        *,
        owner_user_id: int,
        limit: int = 50,
        offset: int = 0,
        include_deleted: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        self._ensure_schema_initialized()
        where_clause = "owner_user_id = ?" if include_deleted else "owner_user_id = ? AND deleted = 0"
        total_cursor = self.db.execute_query(
            f"SELECT COUNT(*) AS total FROM vn_scripts WHERE {where_clause}",  # nosec B608
            (owner_user_id,),
        )
        total_row = total_cursor.fetchone()
        total = int(total_row["total"] if total_row is not None else 0)
        cursor = self.db.execute_query(
            f"""
            SELECT *
            FROM vn_scripts
            WHERE {where_clause}
            ORDER BY updated_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,  # nosec B608
            (owner_user_id, max(1, min(int(limit), 100)), max(0, int(offset))),
        )
        return [self._script_with_draft(row) for row in cursor.fetchall()], total

    def update_script(
        self,
        script_id: int,
        fields: Mapping[str, Any],
        *,
        owner_user_id: int,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        allowed = {
            "title",
            "description",
            "status",
            "primary_asset_pack_id",
            "policy_profile_id",
            "generation_profile_id",
            "generation_profiles",
            "content_rating",
        }
        current = self.get_script(script_id, owner_user_id=owner_user_id)
        if current is None:
            return None
        normalized_fields = dict(fields)
        if "generation_profile_id" in normalized_fields or "generation_profiles" in normalized_fields:
            default_profile_id = str(normalized_fields.get("generation_profile_id") or current["generation_profile_id"])
            raw_profiles = normalized_fields.get(
                "generation_profiles",
                {
                    key: value
                    for key, value in dict(current.get("generation_profiles") or {}).items()
                    if key != "default"
                },
            )
            normalized_fields["generation_profile_id"] = default_profile_id
            normalized_fields["generation_profile_ids_json"] = _json_dump(
                _normalize_generation_profile_ids(default_profile_id, raw_profiles)
            )
            normalized_fields.pop("generation_profiles", None)
        allowed.add("generation_profile_ids_json")
        updates = [(key, value) for key, value in normalized_fields.items() if key in allowed]
        if not updates:
            return self.get_script(script_id, owner_user_id=owner_user_id)
        assignments = ", ".join(f"{key} = ?" for key, _ in updates)
        params = [value for _, value in updates]
        params.extend([script_id, owner_user_id])
        with self.db.transaction() as conn:
            conn.execute(
                f"""
                UPDATE vn_scripts
                SET {assignments}, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND owner_user_id = ? AND deleted = 0
                """,  # nosec B608
                tuple(params),
            )
        return self.get_script(script_id, owner_user_id=owner_user_id)

    def soft_delete_script(self, script_id: int, *, owner_user_id: int) -> None:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE vn_scripts
                SET deleted = 1, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND owner_user_id = ?
                """,
                (script_id, owner_user_id),
            )

    def get_draft(self, script_id: int, *, owner_user_id: int) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT draft.*
            FROM vn_script_drafts AS draft
            JOIN vn_scripts AS script ON script.id = draft.script_id
            WHERE draft.script_id = ?
              AND draft.owner_user_id = ?
              AND script.owner_user_id = draft.owner_user_id
              AND script.deleted = 0
            """,
            (script_id, owner_user_id),
        )
        row = cursor.fetchone()
        return _decode_draft(row) if row is not None else None

    def replace_draft(
        self,
        script_id: int,
        *,
        owner_user_id: int,
        if_revision: int,
        draft: Mapping[str, Any],
        diagnostics: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE vn_script_drafts
                SET revision = revision + 1,
                    draft_json = ?,
                    diagnostics_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE script_id = ?
                  AND owner_user_id = ?
                  AND revision = ?
                  AND EXISTS (
                      SELECT 1
                      FROM vn_scripts
                      WHERE id = vn_script_drafts.script_id
                        AND owner_user_id = vn_script_drafts.owner_user_id
                        AND deleted = 0
                  )
                """,
                (
                    _json_dump(dict(draft)),
                    _json_dump(dict(diagnostics)),
                    script_id,
                    owner_user_id,
                    if_revision,
                ),
            )
            if cursor.rowcount != 1:
                current = self.get_draft(script_id, owner_user_id=owner_user_id)
                if current is None:
                    raise ValueError("script_not_found")
                raise ValueError("draft_revision_conflict")
            conn.execute(
                """
                UPDATE vn_scripts
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND owner_user_id = ?
                """,
                (script_id, owner_user_id),
            )
        draft_row = self.get_draft(script_id, owner_user_id=owner_user_id)
        if draft_row is None:
            raise RuntimeError("updated_draft_not_found")
        return draft_row

    def store_diagnostics(
        self,
        script_id: int,
        *,
        owner_user_id: int,
        diagnostics: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE vn_script_drafts
                SET diagnostics_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE script_id = ? AND owner_user_id = ?
                """,
                (_json_dump(dict(diagnostics)), script_id, owner_user_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("script_not_found")
        draft = self.get_draft(script_id, owner_user_id=owner_user_id)
        if draft is None:
            raise RuntimeError("updated_diagnostics_not_found")
        return draft

    def create_manifest_snapshot(
        self,
        *,
        owner_user_id: int,
        script_id: int,
        asset_pack_id: int,
        manifest: Mapping[str, Any],
        manifest_hash: str,
        version_id: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO vn_script_manifest_snapshots (
                    owner_user_id,
                    script_id,
                    version_id,
                    asset_pack_id,
                    manifest_json,
                    manifest_hash
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    script_id,
                    version_id,
                    asset_pack_id,
                    _json_dump(dict(manifest)),
                    manifest_hash,
                ),
            )
            snapshot_id = int(cursor.lastrowid)
        snapshot = self.get_manifest_snapshot(snapshot_id, owner_user_id=owner_user_id)
        if snapshot is None:
            raise RuntimeError("created_manifest_snapshot_not_found")
        return snapshot

    def get_manifest_snapshot(self, snapshot_id: int, *, owner_user_id: int) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_script_manifest_snapshots
            WHERE id = ? AND owner_user_id = ?
            """,
            (snapshot_id, owner_user_id),
        )
        row = cursor.fetchone()
        return _decode_manifest_snapshot(row) if row is not None else None

    def get_manifest_snapshot_for_version(
        self,
        *,
        script_id: int,
        version_id: int,
        owner_user_id: int,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT snapshot.*
            FROM vn_script_manifest_snapshots AS snapshot
            JOIN vn_script_versions AS version ON version.manifest_snapshot_id = snapshot.id
            WHERE version.script_id = ?
              AND version.id = ?
              AND version.owner_user_id = ?
              AND snapshot.owner_user_id = version.owner_user_id
            """,
            (script_id, version_id, owner_user_id),
        )
        row = cursor.fetchone()
        return _decode_manifest_snapshot(row) if row is not None else None

    def find_asset_cleanup_blockers(
        self,
        *,
        owner_user_id: int,
        asset_pack_id: int,
        generated_file_ids: set[int],
    ) -> dict[int, list[dict[str, str]]]:
        """Find generated files referenced by published script manifest snapshots."""
        self._ensure_schema_initialized()
        if not generated_file_ids:
            return {}
        cursor = self.db.execute_query(
            """
            SELECT id, manifest_json
            FROM vn_script_manifest_snapshots
            WHERE owner_user_id = ?
              AND asset_pack_id = ?
              AND version_id IS NOT NULL
            """,
            (owner_user_id, asset_pack_id),
        )
        blockers: dict[int, list[dict[str, str]]] = {}
        for row in cursor.fetchall():
            manifest = _json_load(row["manifest_json"], {})
            referenced_file_ids = _collect_int_values(
                manifest,
                {"generated_file_id", "file_id"},
            )
            for file_id in generated_file_ids.intersection(referenced_file_ids):
                blockers.setdefault(file_id, []).append(
                    {
                        "code": "published_script_manifest",
                        "message": (
                            "File is referenced by a published VN script manifest "
                            f"snapshot {int(row['id'])}."
                        ),
                    }
                )
        return blockers

    def create_version(
        self,
        *,
        script_id: int,
        owner_user_id: int,
        label: str | None,
        draft_revision: int,
        program: Mapping[str, Any],
        asset_pack_id: int,
        manifest: Mapping[str, Any],
        manifest_hash: str,
        policy_snapshot_id: int,
        generation_profile_snapshot_id: int,
        generation_profile_snapshots: Mapping[str, int] | None = None,
        script_defaults: Mapping[str, Any],
        validation: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        normalized_generation_snapshots = _normalize_generation_profile_snapshots(
            generation_profile_snapshot_id,
            generation_profile_snapshots,
        )
        with self.db.transaction() as conn:
            number_row = conn.execute(
                """
                SELECT COALESCE(MAX(version_number), 0) + 1 AS version_number
                FROM vn_script_versions
                WHERE script_id = ? AND owner_user_id = ?
                """,
                (script_id, owner_user_id),
            ).fetchone()
            version_number = int(number_row["version_number"] if number_row is not None else 1)
            snapshot_cursor = conn.execute(
                """
                INSERT INTO vn_script_manifest_snapshots (
                    owner_user_id,
                    script_id,
                    asset_pack_id,
                    manifest_json,
                    manifest_hash
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    script_id,
                    asset_pack_id,
                    _json_dump(dict(manifest)),
                    manifest_hash,
                ),
            )
            manifest_snapshot_id = int(snapshot_cursor.lastrowid)
            version_cursor = conn.execute(
                """
                INSERT INTO vn_script_versions (
                    script_id,
                    owner_user_id,
                    version_number,
                    label,
                    draft_revision,
                    program_json,
                    asset_pack_id,
                    manifest_snapshot_id,
                    policy_snapshot_id,
                    generation_profile_snapshot_id,
                    generation_profile_snapshots_json,
                    script_defaults_json,
                    validation_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    script_id,
                    owner_user_id,
                    version_number,
                    label,
                    draft_revision,
                    _json_dump(dict(program)),
                    asset_pack_id,
                    manifest_snapshot_id,
                    policy_snapshot_id,
                    generation_profile_snapshot_id,
                    _json_dump(normalized_generation_snapshots),
                    _json_dump(dict(script_defaults)),
                    _json_dump(dict(validation)),
                ),
            )
            version_id = int(version_cursor.lastrowid)
            conn.execute(
                """
                UPDATE vn_script_manifest_snapshots
                SET version_id = ?
                WHERE id = ?
                """,
                (version_id, manifest_snapshot_id),
            )
            snapshot_ids = list(
                dict.fromkeys(
                    [
                        int(policy_snapshot_id),
                        *[
                            int(snapshot_id)
                            for snapshot_id in normalized_generation_snapshots.values()
                        ],
                    ]
                )
            )
            placeholders = ", ".join("?" for _ in snapshot_ids)
            conn.execute(
                f"""
                UPDATE vn_profile_snapshots
                SET resource_id = ?
                WHERE owner_user_id = ?
                  AND resource_type = 'script_version'
                  AND id IN ({placeholders})
                """,  # nosec B608 - placeholders are generated for bound parameters only.
                (
                    version_id,
                    owner_user_id,
                    *snapshot_ids,
                ),
            )
        version = self.get_version(script_id, version_id, owner_user_id=owner_user_id)
        if version is None:
            raise RuntimeError("created_script_version_not_found")
        return version

    def publish_version_with_request(
        self,
        *,
        owner_user_id: int,
        script_id: int,
        idempotency_key: str,
        payload_hash: str,
        request_payload: Mapping[str, Any] | None = None,
        label: str | None,
        draft_revision: int,
        program: Mapping[str, Any],
        asset_pack_id: int,
        manifest: Mapping[str, Any],
        manifest_hash: str,
        policy_profile: Mapping[str, Any],
        generation_profile: Mapping[str, Any],
        generation_profiles: Mapping[str, Mapping[str, Any]] | None = None,
        script_defaults: Mapping[str, Any],
        validation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically publish a script version and persist its idempotency replay record."""
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            existing_row = conn.execute(
                """
                SELECT *
                FROM vn_script_publish_requests
                WHERE owner_user_id = ? AND script_id = ? AND idempotency_key = ?
                """,
                (owner_user_id, script_id, idempotency_key),
            ).fetchone()
            if existing_row is not None:
                existing = _decode_publish_request(existing_row)
                if not _publish_request_matches(existing, request_payload=request_payload, legacy_payload_hash=payload_hash):
                    raise ValueError("idempotency_key_conflict")
                return {"replayed": True, "version": None, "response": dict(existing["response"])}

            number_row = conn.execute(
                """
                SELECT COALESCE(MAX(version_number), 0) + 1 AS version_number
                FROM vn_script_versions
                WHERE script_id = ? AND owner_user_id = ?
                """,
                (script_id, owner_user_id),
            ).fetchone()
            version_number = int(number_row["version_number"] if number_row is not None else 1)
            policy_snapshot_id = _insert_profile_snapshot(
                conn,
                owner_user_id=owner_user_id,
                snapshot_type="policy",
                profile=policy_profile,
                resource_type="script_version",
            )
            generation_snapshot_id = _insert_profile_snapshot(
                conn,
                owner_user_id=owner_user_id,
                snapshot_type="generation",
                profile=generation_profile,
                resource_type="script_version",
            )
            generation_snapshot_map = {"default": generation_snapshot_id}
            for profile_key, profile in (generation_profiles or {}).items():
                if profile_key == "default":
                    continue
                generation_snapshot_map[str(profile_key)] = _insert_profile_snapshot(
                    conn,
                    owner_user_id=owner_user_id,
                    snapshot_type="generation",
                    profile=profile,
                    resource_type="script_version",
                )
            snapshot_cursor = conn.execute(
                """
                INSERT INTO vn_script_manifest_snapshots (
                    owner_user_id,
                    script_id,
                    asset_pack_id,
                    manifest_json,
                    manifest_hash
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    script_id,
                    asset_pack_id,
                    _json_dump(dict(manifest)),
                    manifest_hash,
                ),
            )
            manifest_snapshot_id = int(snapshot_cursor.lastrowid)
            version_cursor = conn.execute(
                """
                INSERT INTO vn_script_versions (
                    script_id,
                    owner_user_id,
                    version_number,
                    label,
                    draft_revision,
                    program_json,
                    asset_pack_id,
                    manifest_snapshot_id,
                    policy_snapshot_id,
                    generation_profile_snapshot_id,
                    generation_profile_snapshots_json,
                    script_defaults_json,
                    validation_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    script_id,
                    owner_user_id,
                    version_number,
                    label,
                    draft_revision,
                    _json_dump(dict(program)),
                    asset_pack_id,
                    manifest_snapshot_id,
                    policy_snapshot_id,
                    generation_snapshot_id,
                    _json_dump(generation_snapshot_map),
                    _json_dump(dict(script_defaults)),
                    _json_dump(dict(validation)),
                ),
            )
            version_id = int(version_cursor.lastrowid)
            conn.execute(
                """
                UPDATE vn_script_manifest_snapshots
                SET version_id = ?
                WHERE id = ?
                """,
                (version_id, manifest_snapshot_id),
            )
            snapshot_ids = [policy_snapshot_id, *generation_snapshot_map.values()]
            snapshot_placeholders = ", ".join("?" for _ in snapshot_ids)
            conn.execute(
                f"""
                UPDATE vn_profile_snapshots
                SET resource_id = ?
                WHERE owner_user_id = ?
                  AND resource_type = 'script_version'
                  AND id IN ({snapshot_placeholders})
                """,  # nosec B608
                (version_id, owner_user_id, *snapshot_ids),
            )
            version_row = conn.execute(
                "SELECT * FROM vn_script_versions WHERE id = ? AND owner_user_id = ?",
                (version_id, owner_user_id),
            ).fetchone()
            if version_row is None:
                raise RuntimeError("created_script_version_not_found")
            version = _decode_version(version_row)
            response = _publish_response_payload(version, validation)
            conn.execute(
                """
                INSERT INTO vn_script_publish_requests (
                    owner_user_id,
                    script_id,
                    idempotency_key,
                    payload_hash,
                    request_payload_json,
                    response_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    script_id,
                    idempotency_key,
                    payload_hash,
                    _json_dump(dict(request_payload or {})) if request_payload is not None else None,
                    _json_dump(response),
                ),
            )
        return {"replayed": False, "version": version, "response": response}

    def list_versions(
        self,
        script_id: int,
        *,
        owner_user_id: int,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        self._ensure_schema_initialized()
        total_cursor = self.db.execute_query(
            """
            SELECT COUNT(*) AS total
            FROM vn_script_versions AS version
            JOIN vn_scripts AS script ON script.id = version.script_id
            WHERE version.script_id = ?
              AND version.owner_user_id = ?
              AND script.owner_user_id = version.owner_user_id
              AND script.deleted = 0
            """,
            (script_id, owner_user_id),
        )
        total_row = total_cursor.fetchone()
        total = int(total_row["total"] if total_row is not None else 0)
        cursor = self.db.execute_query(
            """
            SELECT version.*
            FROM vn_script_versions AS version
            JOIN vn_scripts AS script ON script.id = version.script_id
            WHERE version.script_id = ?
              AND version.owner_user_id = ?
              AND script.owner_user_id = version.owner_user_id
              AND script.deleted = 0
            ORDER BY version.version_number DESC
            LIMIT ? OFFSET ?
            """,
            (script_id, owner_user_id, max(1, min(int(limit), 100)), max(0, int(offset))),
        )
        return [_decode_version(row) for row in cursor.fetchall()], total

    def list_latest_versions_for_setup(
        self,
        *,
        owner_user_id: int,
        limit: int = 25,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return the latest published version for each active script."""
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT
                version.*,
                script.title AS script_title,
                script.policy_profile_id AS policy_profile_id,
                script.generation_profile_id AS generation_profile_id,
                script.content_rating AS script_content_rating
            FROM vn_script_versions AS version
            JOIN vn_scripts AS script ON script.id = version.script_id
            JOIN (
                SELECT script_id, MAX(version_number) AS latest_version_number
                FROM vn_script_versions
                WHERE owner_user_id = ?
                GROUP BY script_id
            ) AS latest
              ON latest.script_id = version.script_id
             AND latest.latest_version_number = version.version_number
            WHERE version.owner_user_id = ?
              AND script.owner_user_id = version.owner_user_id
              AND script.deleted = 0
            ORDER BY version.created_at DESC, version.id DESC
            LIMIT ? OFFSET ?
            """,
            (
                owner_user_id,
                owner_user_id,
                max(1, min(int(limit), 100)),
                max(0, int(offset)),
            ),
        )
        versions: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            version = _decode_version(row)
            version["title"] = row["script_title"]
            version["policy_profile_id"] = row["policy_profile_id"]
            version["generation_profile_id"] = row["generation_profile_id"]
            version["content_rating"] = row["script_content_rating"]
            versions.append(version)
        return versions

    def get_version(
        self,
        script_id: int,
        version_id: int,
        *,
        owner_user_id: int,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT version.*
            FROM vn_script_versions AS version
            JOIN vn_scripts AS script ON script.id = version.script_id
            WHERE version.script_id = ?
              AND version.id = ?
              AND version.owner_user_id = ?
              AND script.owner_user_id = version.owner_user_id
              AND script.deleted = 0
            """,
            (script_id, version_id, owner_user_id),
        )
        row = cursor.fetchone()
        return _decode_version(row) if row is not None else None

    def get_publish_request_by_key(
        self,
        *,
        owner_user_id: int,
        script_id: int,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_script_publish_requests
            WHERE owner_user_id = ? AND script_id = ? AND idempotency_key = ?
            """,
            (owner_user_id, script_id, idempotency_key),
        )
        row = cursor.fetchone()
        return _decode_publish_request(row) if row is not None else None

    def create_publish_request(
        self,
        *,
        owner_user_id: int,
        script_id: int,
        idempotency_key: str,
        payload_hash: str,
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        existing = self.get_publish_request_by_key(
            owner_user_id=owner_user_id,
            script_id=script_id,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            if existing["payload_hash"] != payload_hash:
                raise ValueError("idempotency_key_conflict")
            return existing
        try:
            with self.db.transaction() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO vn_script_publish_requests (
                        owner_user_id,
                        script_id,
                        idempotency_key,
                        payload_hash,
                        response_json
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (owner_user_id, script_id, idempotency_key, payload_hash, _json_dump(dict(response))),
                )
                request_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError:
            existing = self.get_publish_request_by_key(
                owner_user_id=owner_user_id,
                script_id=script_id,
                idempotency_key=idempotency_key,
            )
            if existing is not None and existing["payload_hash"] == payload_hash:
                return existing
            raise ValueError("idempotency_key_conflict") from None

        cursor = self.db.execute_query(
            "SELECT * FROM vn_script_publish_requests WHERE id = ?",
            (request_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("created_publish_request_not_found")
        return _decode_publish_request(row)

    def _script_with_draft(self, row: Any) -> dict[str, Any]:
        script = _decode_script(row)
        draft = self.get_draft(script["id"], owner_user_id=script["owner_user_id"])
        script["draft"] = draft
        return script

    def _ensure_schema_initialized(self) -> None:
        if not self._schema_initialized:
            self.initialize_schema()


def _require_sqlite_chacha_db(db: CharactersRAGDB) -> None:
    backend = getattr(db, "backend_type", None)
    if backend is not None and backend != BackendType.SQLITE:
        raise ValueError("VN scripts storage currently requires a SQLite ChaChaNotes DB")


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, definition: str) -> None:
    columns = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}  # nosec B608
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")  # nosec B608


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _insert_profile_snapshot(
    conn: Any,
    *,
    owner_user_id: int,
    snapshot_type: str,
    profile: Mapping[str, Any],
    resource_type: str,
) -> int:
    if snapshot_type not in {"policy", "generation"}:
        raise ValueError("invalid_snapshot_type")
    cursor = conn.execute(
        """
        INSERT INTO vn_profile_snapshots (
            owner_user_id,
            snapshot_type,
            profile_id,
            profile_version,
            resource_type,
            resource_id,
            definition_json
        )
        VALUES (?, ?, ?, ?, ?, NULL, ?)
        """,
        (
            owner_user_id,
            snapshot_type,
            str(profile["profile_id"]),
            int(profile["version"]),
            resource_type,
            _json_dump(dict(profile["definition"])),
        ),
    )
    return int(cursor.lastrowid)


def _normalize_generation_profile_ids(
    default_profile_id: str,
    generation_profiles: Mapping[str, Any] | None,
) -> dict[str, str]:
    normalized_default_profile_id = str(default_profile_id)
    if not normalized_default_profile_id or len(normalized_default_profile_id) > _MAX_GENERATION_PROFILE_ID_LENGTH:
        raise ValueError("generation_profile_id_invalid")
    profile_ids: dict[str, str] = {"default": normalized_default_profile_id}
    if isinstance(generation_profiles, Mapping):
        if len(generation_profiles) > _MAX_GENERATION_PROFILE_MAP_SIZE:
            raise ValueError("generation_profile_map_too_large")
        for key, value in generation_profiles.items():
            profile_key = str(key)
            if profile_key == "default":
                if str(value) != normalized_default_profile_id:
                    raise ValueError("generation_profile_default_reserved")
                continue
            if not _GENERATION_PROFILE_KEY_RE.fullmatch(profile_key):
                raise ValueError("generation_profile_key_invalid")
            profile_id = str(value)
            if not profile_id or len(profile_id) > _MAX_GENERATION_PROFILE_ID_LENGTH:
                raise ValueError("generation_profile_id_invalid")
            profile_ids[profile_key] = profile_id
    profile_ids["default"] = normalized_default_profile_id
    return profile_ids


def _normalize_generation_profile_snapshots(
    default_snapshot_id: int,
    generation_profile_snapshots: Mapping[str, Any] | None,
) -> dict[str, int]:
    snapshots: dict[str, int] = {"default": int(default_snapshot_id)}
    if isinstance(generation_profile_snapshots, Mapping):
        for key, value in generation_profile_snapshots.items():
            snapshots[str(key)] = int(value)
    snapshots["default"] = int(default_snapshot_id)
    return snapshots


def _publish_response_payload(version: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "script_id": int(version["script_id"]),
        "version_id": int(version["id"]),
        "version_number": int(version["version_number"]),
        "status": "published",
        "asset_pack_id": int(version["asset_pack_id"]),
        "manifest_snapshot_id": int(version["manifest_snapshot_id"]),
        "policy_snapshot_id": int(version["policy_snapshot_id"]),
        "generation_profile_snapshot_id": int(version["generation_profile_snapshot_id"]),
        "generation_profile_snapshots": dict(version["generation_profile_snapshots"]),
        "validation": dict(validation),
        "created_at": version["created_at"],
    }


def _decode_script(row: Any) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "owner_user_id": int(row["owner_user_id"]),
        "title": row["title"],
        "description": row["description"],
        "status": row["status"],
        "primary_asset_pack_id": int(row["primary_asset_pack_id"]),
        "policy_profile_id": row["policy_profile_id"],
        "generation_profile_id": row["generation_profile_id"],
        "generation_profiles": _normalize_generation_profile_ids(
            str(row["generation_profile_id"]),
            _json_load(row["generation_profile_ids_json"], {}),
        ),
        "content_rating": row["content_rating"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "deleted": bool(row["deleted"]),
    }


def _decode_draft(row: Any) -> dict[str, Any]:
    return {
        "script_id": int(row["script_id"]),
        "owner_user_id": int(row["owner_user_id"]),
        "revision": int(row["revision"]),
        "draft": _json_load(row["draft_json"], {}),
        "diagnostics": _json_load(row["diagnostics_json"], {"valid": True, "errors": [], "warnings": []}),
        "updated_at": row["updated_at"],
    }


def _decode_manifest_snapshot(row: Any) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "owner_user_id": int(row["owner_user_id"]),
        "script_id": int(row["script_id"]),
        "version_id": int(row["version_id"]) if row["version_id"] is not None else None,
        "asset_pack_id": int(row["asset_pack_id"]),
        "manifest": _json_load(row["manifest_json"], {}),
        "manifest_hash": row["manifest_hash"],
        "created_at": row["created_at"],
    }


def _collect_int_values(value: Any, keys: set[str]) -> set[int]:
    found: set[int] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in keys:
                try:
                    found.add(int(nested))
                except (TypeError, ValueError):
                    pass
            found.update(_collect_int_values(nested, keys))
    elif isinstance(value, list):
        for nested in value:
            found.update(_collect_int_values(nested, keys))
    return found


def _decode_version(row: Any) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "script_id": int(row["script_id"]),
        "owner_user_id": int(row["owner_user_id"]),
        "version_number": int(row["version_number"]),
        "label": row["label"],
        "draft_revision": int(row["draft_revision"]),
        "program": _json_load(row["program_json"], {}),
        "asset_pack_id": int(row["asset_pack_id"]),
        "manifest_snapshot_id": int(row["manifest_snapshot_id"]),
        "policy_snapshot_id": int(row["policy_snapshot_id"]),
        "generation_profile_snapshot_id": int(row["generation_profile_snapshot_id"]),
        "generation_profile_snapshots": _normalize_generation_profile_snapshots(
            int(row["generation_profile_snapshot_id"]),
            _json_load(row["generation_profile_snapshots_json"], {}),
        ),
        "script_defaults": _json_load(row["script_defaults_json"], {}),
        "validation": _json_load(row["validation_json"], {}),
        "created_at": row["created_at"],
    }


def _decode_publish_request(row: Any) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "owner_user_id": int(row["owner_user_id"]),
        "script_id": int(row["script_id"]),
        "idempotency_key": row["idempotency_key"],
        "payload_hash": row["payload_hash"],
        "request_payload": _json_load(row["request_payload_json"], None),
        "response": _json_load(row["response_json"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _publish_request_matches(
    existing: Mapping[str, Any],
    *,
    request_payload: Mapping[str, Any] | None,
    legacy_payload_hash: str,
) -> bool:
    existing_payload = existing.get("request_payload")
    if isinstance(existing_payload, Mapping):
        if request_payload is None:
            return False
        return dict(existing_payload) == dict(request_payload)
    return existing.get("payload_hash") == legacy_payload_hash

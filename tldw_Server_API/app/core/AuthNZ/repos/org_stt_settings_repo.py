from __future__ import annotations

import json
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool


class AuthnzOrgSttSettingsRepo:
    """Repository for org-scoped STT policy settings."""

    def __init__(self, db: Any) -> None:
        self.db = db

    def _is_db_pool(self) -> bool:
        return isinstance(self.db, DatabasePool)

    def _is_postgres(self) -> bool:
        if self._is_db_pool():
            return getattr(self.db, "pool", None) is not None

        sqlite_hint = getattr(self.db, "_is_sqlite", None)
        if isinstance(sqlite_hint, bool):
            return not sqlite_hint

        if getattr(self.db, "_c", None) is not None:
            return False

        module_name = getattr(type(self.db), "__module__", "")
        if isinstance(module_name, str) and module_name.startswith("asyncpg"):
            return True

        return callable(getattr(self.db, "fetchrow", None))

    async def ensure_tables(self) -> None:
        """Ensure the org STT settings table exists when a pool is available."""
        try:
            if self._is_postgres():
                from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                    ensure_org_stt_settings_pg,
                )

                ok = await ensure_org_stt_settings_pg(self.db)
                if not ok:
                    raise RuntimeError("PostgreSQL org_stt_settings schema ensure failed")
                return

            if self._is_db_pool():
                row = await self.db.fetchone(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='org_stt_settings'"
                )
                if not row:
                    raise RuntimeError(
                        "SQLite org_stt_settings table is missing. "
                        "Run the AuthNZ migrations/bootstrap."
                    )
            else:
                # Direct SQLite connection handle
                cursor = await self.db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='org_stt_settings'"
                )
                row = await cursor.fetchone()
                if not row:
                    raise RuntimeError(
                        "SQLite org_stt_settings table is missing. "
                        "Run the AuthNZ migrations/bootstrap."
                    )
        except Exception as exc:
            logger.error(f"AuthnzOrgSttSettingsRepo.ensure_tables failed: {exc}")
            raise

    @staticmethod
    def _normalize_categories(categories: list[str] | None) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in categories or []:
            value = str(raw).strip().lower()
            if not value or value in seen:
                continue
            normalized.append(value)
            seen.add(value)
        return normalized

    @staticmethod
    def _load_categories(raw: Any) -> list[str]:
        if isinstance(raw, list):
            return AuthnzOrgSttSettingsRepo._normalize_categories([str(item) for item in raw])
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                return []
            if isinstance(parsed, list):
                return AuthnzOrgSttSettingsRepo._normalize_categories([str(item) for item in parsed])
        return []

    @classmethod
    def _row_to_dict(cls, row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            raw = dict(row)
        else:
            try:
                keys = row.keys()
                raw = {key: row[key] for key in keys}
            except Exception:
                raw = dict(row)
        return {
            "org_id": int(raw["org_id"]),
            "delete_audio_after_success": bool(raw["delete_audio_after_success"]),
            "audio_retention_hours": float(raw["audio_retention_hours"]),
            "redact_pii": bool(raw["redact_pii"]),
            "allow_unredacted_partials": bool(raw["allow_unredacted_partials"]),
            "redact_categories": cls._load_categories(raw.get("redact_categories_json")),
        }

    async def get_settings(self, org_id: int) -> dict[str, Any] | None:
        try:
            if self._is_postgres():
                row = await self.db.fetchrow(
                    """
                    SELECT org_id, delete_audio_after_success, audio_retention_hours,
                           redact_pii, allow_unredacted_partials, redact_categories_json
                    FROM org_stt_settings
                    WHERE org_id = $1
                    """,
                    int(org_id),
                )
            else:
                if self._is_db_pool():
                    row = await self.db.fetchone(
                        """
                        SELECT org_id, delete_audio_after_success, audio_retention_hours,
                               redact_pii, allow_unredacted_partials, redact_categories_json
                        FROM org_stt_settings
                        WHERE org_id = ?
                        """,
                        (int(org_id),),
                    )
                else:
                    cursor = await self.db.execute(
                        """
                        SELECT org_id, delete_audio_after_success, audio_retention_hours,
                               redact_pii, allow_unredacted_partials, redact_categories_json
                        FROM org_stt_settings
                        WHERE org_id = ?
                        """,
                        (int(org_id),),
                    )
                    row = await cursor.fetchone()
            return self._row_to_dict(row) if row else None
        except Exception as exc:
            logger.error(f"AuthnzOrgSttSettingsRepo.get_settings failed: {exc}")
            raise

    async def upsert_settings(
        self,
        *,
        org_id: int,
        delete_audio_after_success: bool,
        audio_retention_hours: float,
        redact_pii: bool,
        allow_unredacted_partials: bool,
        redact_categories: list[str] | None,
        updated_by: int | None = None,
    ) -> dict[str, Any]:
        categories = self._normalize_categories(redact_categories)
        categories_json = json.dumps(categories)
        try:
            if self._is_postgres():
                row = await self.db.fetchrow(
                    """
                    INSERT INTO org_stt_settings (
                        org_id,
                        delete_audio_after_success,
                        audio_retention_hours,
                        redact_pii,
                        allow_unredacted_partials,
                        redact_categories_json,
                        updated_by,
                        updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, CURRENT_TIMESTAMP)
                    ON CONFLICT (org_id) DO UPDATE SET
                        delete_audio_after_success = EXCLUDED.delete_audio_after_success,
                        audio_retention_hours = EXCLUDED.audio_retention_hours,
                        redact_pii = EXCLUDED.redact_pii,
                        allow_unredacted_partials = EXCLUDED.allow_unredacted_partials,
                        redact_categories_json = EXCLUDED.redact_categories_json,
                        updated_by = EXCLUDED.updated_by,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING org_id, delete_audio_after_success, audio_retention_hours,
                              redact_pii, allow_unredacted_partials, redact_categories_json
                    """,
                    int(org_id),
                    bool(delete_audio_after_success),
                    float(audio_retention_hours),
                    bool(redact_pii),
                    bool(allow_unredacted_partials),
                    categories_json,
                    updated_by,
                )
                return self._row_to_dict(row) if row else {}

            params = (
                int(org_id),
                int(bool(delete_audio_after_success)),
                float(audio_retention_hours),
                int(bool(redact_pii)),
                int(bool(allow_unredacted_partials)),
                categories_json,
                updated_by,
            )
            await self.db.execute(
                """
                INSERT INTO org_stt_settings (
                    org_id,
                    delete_audio_after_success,
                    audio_retention_hours,
                    redact_pii,
                    allow_unredacted_partials,
                    redact_categories_json,
                    updated_by,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(org_id) DO UPDATE SET
                    delete_audio_after_success = excluded.delete_audio_after_success,
                    audio_retention_hours = excluded.audio_retention_hours,
                    redact_pii = excluded.redact_pii,
                    allow_unredacted_partials = excluded.allow_unredacted_partials,
                    redact_categories_json = excluded.redact_categories_json,
                    updated_by = excluded.updated_by,
                    updated_at = CURRENT_TIMESTAMP
                """,
                params,
            )
            row = await self.get_settings(int(org_id))
            return row or {}
        except Exception as exc:
            logger.error(f"AuthnzOrgSttSettingsRepo.upsert_settings failed: {exc}")
            raise

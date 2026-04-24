from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import normalize_provider_name


@dataclass
class AuthnzLLMProviderOverridesRepo:
    """Repository for runtime LLM provider overrides."""

    db_pool: DatabasePool

    async def _initialize_db_pool_if_needed(self) -> None:
        """Ensure lazy DatabasePool instances decide their backend before schema checks."""
        initialize = getattr(self.db_pool, "initialize", None)
        if not callable(initialize):
            return

        if getattr(self.db_pool, "pool", None) is not None:
            return

        if getattr(self.db_pool, "db_path", None) is not None and getattr(
            self.db_pool, "_initialized", None
        ) is not False:
            return

        await initialize()

    async def ensure_tables(self) -> None:
        """Ensure llm_provider_overrides schema exists."""
        try:
            await self._initialize_db_pool_if_needed()

            if getattr(self.db_pool, "pool", None) is not None:
                from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                    ensure_llm_provider_overrides_pg,
                )

                ok = await ensure_llm_provider_overrides_pg(self.db_pool)
                if not ok:
                    raise RuntimeError("PostgreSQL llm_provider_overrides schema ensure failed")
                return

            row = await self.db_pool.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='llm_provider_overrides'"
            )
            if not row:
                raise RuntimeError(
                    "SQLite llm_provider_overrides table is missing. "
                    "Run the AuthNZ migrations/bootstrap."
                )
        except Exception as exc:
            logger.error(f"AuthnzLLMProviderOverridesRepo.ensure_tables failed: {exc}")
            raise

    @staticmethod
    def _normalize_datetime_for_postgres(dt: datetime) -> datetime:
        return dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        if row is None:
            return {}
        if isinstance(row, dict):
            return dict(row)
        try:
            return dict(row)
        except Exception as row_cast_error:
            logger.debug("LLM provider override row cast failed; trying keys()/mapping fallback", exc_info=row_cast_error)
        try:
            keys = row.keys()
            return {key: row[key] for key in keys}
        except Exception:
            return {}

    async def list_overrides(self, provider: str | None = None) -> list[dict[str, Any]]:
        provider_norm = normalize_provider_name(provider) if provider else None
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                if provider_norm:
                    rows = await self.db_pool.fetchall(
                        """
                        SELECT provider, is_enabled, allowed_models, config_json, secret_blob,
                               api_key_hint, created_at, updated_at
                        FROM llm_provider_overrides
                        WHERE provider = $1
                        ORDER BY provider
                        """,
                        provider_norm,
                    )
                else:
                    rows = await self.db_pool.fetchall(
                        """
                        SELECT provider, is_enabled, allowed_models, config_json, secret_blob,
                               api_key_hint, created_at, updated_at
                        FROM llm_provider_overrides
                        ORDER BY provider
                        """
                    )
            else:
                if provider_norm:
                    rows = await self.db_pool.fetchall(
                        """
                        SELECT provider, is_enabled, allowed_models, config_json, secret_blob,
                               api_key_hint, created_at, updated_at
                        FROM llm_provider_overrides
                        WHERE provider = ?
                        ORDER BY provider
                        """,
                        (provider_norm,),
                    )
                else:
                    rows = await self.db_pool.fetchall(
                        """
                        SELECT provider, is_enabled, allowed_models, config_json, secret_blob,
                               api_key_hint, created_at, updated_at
                        FROM llm_provider_overrides
                        ORDER BY provider
                        """
                    )
            return [self._row_to_dict(row) for row in rows]
        except Exception as exc:
            logger.error(f"AuthnzLLMProviderOverridesRepo.list_overrides failed: {exc}")
            raise

    async def fetch_override(self, provider: str) -> dict[str, Any] | None:
        provider_norm = normalize_provider_name(provider)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    """
                    SELECT provider, is_enabled, allowed_models, config_json, secret_blob,
                           api_key_hint, created_at, updated_at
                    FROM llm_provider_overrides
                    WHERE provider = $1
                    """,
                    provider_norm,
                )
            else:
                row = await self.db_pool.fetchone(
                    """
                    SELECT provider, is_enabled, allowed_models, config_json, secret_blob,
                           api_key_hint, created_at, updated_at
                    FROM llm_provider_overrides
                    WHERE provider = ?
                    """,
                    (provider_norm,),
                )
            return self._row_to_dict(row) if row else None
        except Exception as exc:
            logger.error(f"AuthnzLLMProviderOverridesRepo.fetch_override failed: {exc}")
            raise

    async def upsert_override(
        self,
        *,
        provider: str,
        is_enabled: bool | None,
        allowed_models: str | None,
        config_json: str | None,
        secret_blob: str | None,
        api_key_hint: str | None,
        updated_at: datetime,
    ) -> dict[str, Any]:
        provider_norm = normalize_provider_name(provider)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                ts = self._normalize_datetime_for_postgres(updated_at)
                row = await self.db_pool.fetchone(
                    """
                    INSERT INTO llm_provider_overrides (
                        provider, is_enabled, allowed_models, config_json,
                        secret_blob, api_key_hint, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $7)
                    ON CONFLICT (provider) DO UPDATE SET
                        is_enabled = EXCLUDED.is_enabled,
                        allowed_models = EXCLUDED.allowed_models,
                        config_json = EXCLUDED.config_json,
                        secret_blob = EXCLUDED.secret_blob,
                        api_key_hint = EXCLUDED.api_key_hint,
                        updated_at = EXCLUDED.updated_at
                    RETURNING provider, is_enabled, allowed_models, config_json, secret_blob,
                              api_key_hint, created_at, updated_at
                    """,
                    provider_norm,
                    is_enabled,
                    allowed_models,
                    config_json,
                    secret_blob,
                    api_key_hint,
                    ts,
                )
                return self._row_to_dict(row) if row else {}

            await self.db_pool.execute(
                """
                INSERT INTO llm_provider_overrides (
                    provider, is_enabled, allowed_models, config_json,
                    secret_blob, api_key_hint, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider) DO UPDATE SET
                    is_enabled = excluded.is_enabled,
                    allowed_models = excluded.allowed_models,
                    config_json = excluded.config_json,
                    secret_blob = excluded.secret_blob,
                    api_key_hint = excluded.api_key_hint,
                    updated_at = excluded.updated_at
                """,
                (
                    provider_norm,
                    int(is_enabled) if is_enabled is not None else None,
                    allowed_models,
                    config_json,
                    secret_blob,
                    api_key_hint,
                    updated_at.isoformat(),
                    updated_at.isoformat(),
                ),
            )
            row = await self.db_pool.fetchone(
                """
                SELECT provider, is_enabled, allowed_models, config_json, secret_blob,
                       api_key_hint, created_at, updated_at
                FROM llm_provider_overrides
                WHERE provider = ?
                """,
                (provider_norm,),
            )
            return self._row_to_dict(row) if row else {}
        except Exception as exc:
            logger.error(f"AuthnzLLMProviderOverridesRepo.upsert_override failed: {exc}")
            raise

    async def delete_override(self, provider: str) -> bool:
        provider_norm = normalize_provider_name(provider)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                result = await self.db_pool.execute(
                    "DELETE FROM llm_provider_overrides WHERE provider = $1",
                    provider_norm,
                )
                if isinstance(result, str):
                    parts = result.split()
                    if parts and parts[-1].isdigit():
                        return int(parts[-1]) > 0
                return True

            cursor = await self.db_pool.execute(
                "DELETE FROM llm_provider_overrides WHERE provider = ?",
                (provider_norm,),
            )
            rowcount = getattr(cursor, "rowcount", 0)
            return rowcount > 0
        except Exception as exc:
            logger.error(f"AuthnzLLMProviderOverridesRepo.delete_override failed: {exc}")
            raise

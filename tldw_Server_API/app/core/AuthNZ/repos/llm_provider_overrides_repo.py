from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    fold_provider_credential_rows,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import (
    canonical_builtin_llm_provider_name,
)
from tldw_Server_API.app.core.LLM_Calls.provider_identity import provider_lookup_names

_PATCHABLE_OVERRIDE_COLUMNS = (
    "is_enabled",
    "allowed_models",
    "config_json",
    "secret_blob",
    "api_key_hint",
)
_OVERRIDE_RESULT_COLUMNS = (
    "provider, is_enabled, allowed_models, config_json, secret_blob, "
    "api_key_hint, created_at, updated_at"
)


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
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzLLMProviderOverridesRepo.ensure_tables failed"
            )
            raise

    @staticmethod
    def _normalize_datetime_for_storage(dt: datetime) -> datetime:
        """Return one aware UTC instant for both database backends."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        if row is None:
            return {}
        if isinstance(row, dict):
            return dict(row)
        try:
            return dict(row)
        except Exception as row_cast_error:
            logger.bind(error_type=type(row_cast_error).__name__).debug(
                "LLM provider override row cast failed; trying keys()/mapping fallback"
            )
        try:
            keys = row.keys()
            return {key: row[key] for key in keys}
        except Exception:
            return {}

    async def _canonicalize_legacy_rows_for_write(
        self,
        conn: Any,
        provider: str,
        *,
        postgres: bool,
    ) -> None:
        """Lock and migrate one authoritative legacy alias before mutation."""
        lookup_names = provider_lookup_names(provider)
        if postgres:
            rows = await conn.fetch(
                f"""
                SELECT {_OVERRIDE_RESULT_COLUMNS}
                FROM llm_provider_overrides
                WHERE provider = ANY($1::text[])
                ORDER BY provider
                FOR UPDATE
                """,  # nosec B608
                list(lookup_names),
            )
        else:
            placeholders = ", ".join("?" for _ in lookup_names)
            cursor = await conn.execute(
                f"""
                SELECT {_OVERRIDE_RESULT_COLUMNS}
                FROM llm_provider_overrides
                WHERE provider IN ({placeholders})
                ORDER BY provider
                """,  # nosec B608
                lookup_names,
            )
            rows = await cursor.fetchall()

        raw_rows = [self._row_to_dict(row) for row in rows]
        if not raw_rows:
            return
        fold_provider_credential_rows(raw_rows)

        if any(row.get("provider") == provider for row in raw_rows):
            if postgres:
                await conn.execute(
                    """
                    DELETE FROM llm_provider_overrides
                    WHERE provider = ANY($1::text[]) AND provider <> $2
                    """,
                    list(lookup_names),
                    provider,
                )
            else:
                placeholders = ", ".join("?" for _ in lookup_names)
                await conn.execute(
                    f"""
                    DELETE FROM llm_provider_overrides
                    WHERE provider IN ({placeholders}) AND provider <> ?
                    """,  # nosec B608
                    (*lookup_names, provider),
                )
            return

        legacy_provider = str(raw_rows[0].get("provider") or "")
        if postgres:
            await conn.execute(
                """
                UPDATE llm_provider_overrides
                SET provider = $1
                WHERE provider = $2
                """,
                provider,
                legacy_provider,
            )
        else:
            await conn.execute(
                """
                UPDATE llm_provider_overrides
                SET provider = ?
                WHERE provider = ?
                """,
                (provider, legacy_provider),
            )

    async def list_overrides(self, provider: str | None = None) -> list[dict[str, Any]]:
        provider_norm = (
            canonical_builtin_llm_provider_name(provider)
            if provider is not None
            else None
        )
        try:
            rows = await self.db_pool.fetchall(
                """
                SELECT provider, is_enabled, allowed_models, config_json, secret_blob,
                       api_key_hint, created_at, updated_at
                FROM llm_provider_overrides
                ORDER BY provider
                """
            )
            folded = fold_provider_credential_rows(
                [self._row_to_dict(row) for row in rows]
            )
            for row in folded:
                canonical_builtin_llm_provider_name(row.get("provider"))
            if provider_norm is not None:
                folded = [row for row in folded if row.get("provider") == provider_norm]
            return folded
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzLLMProviderOverridesRepo.list_overrides failed"
            )
            raise

    async def fetch_override(self, provider: str) -> dict[str, Any] | None:
        try:
            rows = await self.list_overrides(provider)
            return rows[0] if rows else None
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzLLMProviderOverridesRepo.fetch_override failed"
            )
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
        provider_norm = canonical_builtin_llm_provider_name(provider)
        ts = self._normalize_datetime_for_storage(updated_at)
        try:
            await self._initialize_db_pool_if_needed()
            postgres = getattr(self.db_pool, "pool", None) is not None
            async with self.db_pool.transaction() as conn:
                await self._canonicalize_legacy_rows_for_write(
                    conn,
                    provider_norm,
                    postgres=postgres,
                )
                if postgres:
                    row = await conn.fetchrow(
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
                        RETURNING provider, is_enabled, allowed_models, config_json,
                                  secret_blob, api_key_hint, created_at, updated_at
                        """,
                        provider_norm,
                        is_enabled,
                        allowed_models,
                        config_json,
                        secret_blob,
                        api_key_hint,
                        ts,
                    )
                else:
                    await conn.execute(
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
                            ts.isoformat(),
                            ts.isoformat(),
                        ),
                    )
                    cursor = await conn.execute(
                        """
                        SELECT provider, is_enabled, allowed_models, config_json,
                               secret_blob, api_key_hint, created_at, updated_at
                        FROM llm_provider_overrides
                        WHERE provider = ?
                        """,
                        (provider_norm,),
                    )
                    row = await cursor.fetchone()
            return self._row_to_dict(row) if row else {}
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzLLMProviderOverridesRepo.upsert_override failed"
            )
            raise

    async def patch_override(
        self,
        *,
        provider: str,
        fields: dict[str, Any],
        updated_at: datetime,
        compare_secret_blob: bool = False,
        expected_secret_blob: str | None = None,
    ) -> dict[str, Any] | None:
        """Atomically patch supplied columns, optionally CAS-guarding the secret."""
        provider_norm = canonical_builtin_llm_provider_name(provider)
        unknown = set(fields) - set(_PATCHABLE_OVERRIDE_COLUMNS)
        if not fields or unknown or (compare_secret_blob and "secret_blob" not in fields):
            raise ValueError("Provider override patch fields are invalid")

        columns = [name for name in _PATCHABLE_OVERRIDE_COLUMNS if name in fields]
        ts = self._normalize_datetime_for_storage(updated_at)
        try:
            await self._initialize_db_pool_if_needed()
            postgres = getattr(self.db_pool, "pool", None) is not None
            async with self.db_pool.transaction() as conn:
                await self._canonicalize_legacy_rows_for_write(
                    conn,
                    provider_norm,
                    postgres=postgres,
                )
                if postgres:
                    if compare_secret_blob:
                        values = [
                            *(fields[name] for name in columns),
                            ts,
                            provider_norm,
                            expected_secret_blob,
                        ]
                        assignments = ", ".join(
                            [
                                *(
                                    f"{name} = ${index}"
                                    for index, name in enumerate(columns, 1)
                                ),
                                f"updated_at = ${len(columns) + 1}",
                            ]
                        )
                        provider_index = len(columns) + 2
                        expected_index = len(columns) + 3
                        # Identifiers are selected only from the fixed module allowlist.
                        query = f"""
                            UPDATE llm_provider_overrides
                            SET {assignments}
                            WHERE provider = ${provider_index}
                              AND secret_blob IS NOT DISTINCT FROM ${expected_index}
                            RETURNING {_OVERRIDE_RESULT_COLUMNS}
                        """  # nosec
                        row = await conn.fetchrow(query, *values)
                        return self._row_to_dict(row) if row else None

                    insert_columns = ["provider", *columns, "created_at", "updated_at"]
                    values = [provider_norm, *(fields[name] for name in columns), ts, ts]
                    placeholders = ", ".join(
                        f"${index}" for index in range(1, len(values) + 1)
                    )
                    updates = ", ".join(
                        [
                            *(f"{name} = EXCLUDED.{name}" for name in columns),
                            "updated_at = EXCLUDED.updated_at",
                        ]
                    )
                    # Identifiers are selected only from the fixed module allowlist.
                    query = f"""
                        INSERT INTO llm_provider_overrides ({", ".join(insert_columns)})
                        VALUES ({placeholders})
                        ON CONFLICT (provider) DO UPDATE SET {updates}
                        RETURNING {_OVERRIDE_RESULT_COLUMNS}
                    """  # nosec
                    row = await conn.fetchrow(query, *values)
                    return self._row_to_dict(row) if row else {}

                sqlite_values = [
                    int(fields[name])
                    if name == "is_enabled" and fields[name] is not None
                    else fields[name]
                    for name in columns
                ]
                if compare_secret_blob:
                    assignments = ", ".join(
                        [*(f"{name} = ?" for name in columns), "updated_at = ?"]
                    )
                    values = [
                        *sqlite_values,
                        ts.isoformat(),
                        provider_norm,
                        expected_secret_blob,
                    ]
                    # Identifiers are selected only from the fixed module allowlist.
                    query = f"""
                        UPDATE llm_provider_overrides
                        SET {assignments}
                        WHERE provider = ? AND secret_blob IS ?
                    """  # nosec
                    cursor = await conn.execute(query, tuple(values))
                    if getattr(cursor, "rowcount", 0) <= 0:
                        return None
                    cursor = await conn.execute(
                        """
                        SELECT provider, is_enabled, allowed_models, config_json,
                               secret_blob, api_key_hint, created_at, updated_at
                        FROM llm_provider_overrides WHERE provider = ?
                        """,
                        (provider_norm,),
                    )
                    row = await cursor.fetchone()
                    return self._row_to_dict(row) if row else None

                insert_columns = ["provider", *columns, "created_at", "updated_at"]
                values = [
                    provider_norm,
                    *sqlite_values,
                    ts.isoformat(),
                    ts.isoformat(),
                ]
                placeholders = ", ".join("?" for _ in values)
                updates = ", ".join(
                    [
                        *(f"{name} = excluded.{name}" for name in columns),
                        "updated_at = excluded.updated_at",
                    ]
                )
                # Identifiers are selected only from the fixed module allowlist.
                query = f"""
                    INSERT INTO llm_provider_overrides ({", ".join(insert_columns)})
                    VALUES ({placeholders})
                    ON CONFLICT(provider) DO UPDATE SET {updates}
                """  # nosec
                await conn.execute(query, tuple(values))
                cursor = await conn.execute(
                    """
                    SELECT provider, is_enabled, allowed_models, config_json,
                           secret_blob, api_key_hint, created_at, updated_at
                    FROM llm_provider_overrides WHERE provider = ?
                    """,
                    (provider_norm,),
                )
                row = await cursor.fetchone()
            return self._row_to_dict(row) if row else {}
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzLLMProviderOverridesRepo.patch_override failed"
            )
            raise

    async def delete_override(self, provider: str) -> bool:
        provider_norm = canonical_builtin_llm_provider_name(provider)
        lookup_names = provider_lookup_names(provider_norm)
        try:
            await self._initialize_db_pool_if_needed()

            if getattr(self.db_pool, "pool", None) is not None:
                placeholders = ", ".join(
                    f"${index}" for index in range(1, len(lookup_names) + 1)
                )
                result = await self.db_pool.execute(
                    f"DELETE FROM llm_provider_overrides WHERE provider IN ({placeholders})",  # nosec B608
                    *lookup_names,
                )
                if isinstance(result, str):
                    parts = result.split()
                    if parts and parts[-1].isdigit():
                        return int(parts[-1]) > 0
                return True

            placeholders = ", ".join("?" for _ in lookup_names)
            cursor = await self.db_pool.execute(
                f"DELETE FROM llm_provider_overrides WHERE provider IN ({placeholders})",  # nosec B608
                lookup_names,
            )
            rowcount = getattr(cursor, "rowcount", 0)
            return rowcount > 0
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzLLMProviderOverridesRepo.delete_override failed"
            )
            raise

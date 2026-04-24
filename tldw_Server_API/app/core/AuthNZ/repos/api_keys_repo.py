from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import (
    DatabasePool,
    build_sqlite_in_clause,
    should_enforce_sqlite_schema_strictness,
    validate_required_sqlite_api_key_schema,
)


def _affected_row_count(result: Any) -> int:
    """Normalize SQLite cursor/asyncpg status values to an affected-row count."""
    rowcount = getattr(result, "rowcount", None)
    if isinstance(rowcount, int):
        return rowcount
    if isinstance(result, str):
        parts = result.strip().split()
        if parts and parts[-1].isdigit():
            return int(parts[-1])
    return 0


@dataclass
class AuthnzApiKeysRepo:
    """
    Repository for AuthNZ API key storage.

    This class centralizes the core queries for the ``api_keys`` table
    so callers do not need to worry about PostgreSQL vs SQLite dialect
    differences. It is intentionally small; additional helpers can be
    added incrementally as more modules adopt the repository layer.
    """

    db_pool: DatabasePool

    def _is_postgres_backend(self) -> bool:
        """
        Return True when the underlying DatabasePool is using PostgreSQL.

        Backend routing should rely on pool state rather than probing
        connection method presence at runtime.
        """
        return bool(getattr(self.db_pool, "pool", None))

    async def ensure_tables(self) -> None:
        """
        Ensure api_keys + api_key_audit_log schema exists.

        - SQLite: schema is defined/extended by AuthNZ migrations (migrations.py).
        - PostgreSQL: schema is ensured by pg_migrations_extra.ensure_api_keys_tables_pg().
        """
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                    ensure_api_keys_tables_pg,
                )

                ok = await ensure_api_keys_tables_pg(self.db_pool)
                if not ok:
                    raise RuntimeError("PostgreSQL api_keys schema ensure failed")
                return

            # SQLite path: migrations are responsible for schema creation and upgrades.
            # Fail early with a clear error if a deployment skipped the bootstrap.
            api_keys_row = await self.db_pool.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='api_keys'"
            )
            audit_row = await self.db_pool.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='api_key_audit_log'"
            )
            if not api_keys_row or not audit_row:
                raise RuntimeError(
                    "SQLite api_keys tables are missing. "
                    "Run the AuthNZ migrations/bootstrap (see "
                    "'python -m tldw_Server_API.app.core.AuthNZ.initialize')."
                )

            sqlite_path = getattr(self.db_pool, "_sqlite_fs_path", None)
            if should_enforce_sqlite_schema_strictness(sqlite_path):
                await asyncio.to_thread(validate_required_sqlite_api_key_schema, sqlite_path)
        except Exception as exc:
            logger.error(f"AuthnzApiKeysRepo.ensure_tables failed: {exc}")
            raise

    async def fetch_active_by_hash_candidates(
        self,
        hash_candidates: list[str],
    ) -> dict[str, Any] | None:
        """
        Fetch the most recent active API key whose hash matches any of the
        provided candidates.

        The query mirrors the lookup used by APIKeyManager.validate_api_key,
        but is encapsulated here to avoid duplicating dialect-specific SQL.
        """
        if not hash_candidates:
            return None

        try:
            # PostgreSQL path: use ANY($1::text[]) for hash candidate list
            if getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    """
                    SELECT id, user_id, name, scope, status, expires_at,
                           rate_limit, allowed_ips, usage_count, key_hash,
                           COALESCE(is_virtual, FALSE) AS is_virtual,
                           parent_key_id, org_id, team_id,
                           llm_budget_day_tokens, llm_budget_month_tokens,
                           llm_budget_day_usd, llm_budget_month_usd,
                           llm_allowed_endpoints, llm_allowed_providers, llm_allowed_models,
                           metadata
                    FROM api_keys
                    WHERE key_hash = ANY($1::text[]) AND status = $2
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    hash_candidates,
                    "active",
                )
            else:
                # SQLite path: emulate ANY with an IN (...) clause
                # SECURITY: Using build_sqlite_in_clause helper to safely generate
                # parameterized placeholders - see database.py for implementation details
                placeholders, hash_params = build_sqlite_in_clause(hash_candidates)
                hash_candidates_clause = f"({placeholders})"
                query_template = """
                    SELECT id, user_id, name, scope, status, expires_at,
                           rate_limit, allowed_ips, usage_count, key_hash,
                           COALESCE(is_virtual, 0) AS is_virtual,
                           parent_key_id, org_id, team_id,
                           llm_budget_day_tokens, llm_budget_month_tokens,
                           llm_budget_day_usd, llm_budget_month_usd,
                           llm_allowed_endpoints, llm_allowed_providers, llm_allowed_models,
                           metadata
                    FROM api_keys
                    WHERE key_hash IN {hash_candidates_clause} AND status = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                query = query_template.format_map(locals())  # nosec B608
                params = (*hash_params, "active")
                row = await self.db_pool.fetchone(query, params)

            if not row:
                return None

            return dict(row)
        except Exception as exc:  # pragma: no cover - surfaced via higher layers
            logger.error(f"AuthnzApiKeysRepo.fetch_active_by_hash_candidates failed: {exc}")
            raise

    async def fetch_active_by_key_id(
        self,
        key_id: str,
    ) -> dict[str, Any] | None:
        """Fetch the most recent active API key by key_id."""
        if not key_id:
            return None

        try:
            if getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    """
                    SELECT id, user_id, name, scope, status, expires_at,
                           rate_limit, allowed_ips, usage_count, key_hash,
                           COALESCE(is_virtual, FALSE) AS is_virtual,
                           parent_key_id, org_id, team_id,
                           llm_budget_day_tokens, llm_budget_month_tokens,
                           llm_budget_day_usd, llm_budget_month_usd,
                           llm_allowed_endpoints, llm_allowed_providers, llm_allowed_models,
                           metadata
                    FROM api_keys
                    WHERE key_id = $1 AND status = $2
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    key_id,
                    "active",
                )
            else:
                row = await self.db_pool.fetchone(
                    """
                    SELECT id, user_id, name, scope, status, expires_at,
                           rate_limit, allowed_ips, usage_count, key_hash,
                           COALESCE(is_virtual, 0) AS is_virtual,
                           parent_key_id, org_id, team_id,
                           llm_budget_day_tokens, llm_budget_month_tokens,
                           llm_budget_day_usd, llm_budget_month_usd,
                           llm_allowed_endpoints, llm_allowed_providers, llm_allowed_models,
                           metadata
                    FROM api_keys
                    WHERE key_id = ? AND status = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (key_id, "active"),
                )

            if not row:
                return None
            return dict(row)
        except Exception as exc:  # pragma: no cover - surfaced via higher layers
            logger.error(f"AuthnzApiKeysRepo.fetch_active_by_key_id failed: {exc}")
            raise

    async def fetch_key_for_user(
        self,
        key_id: int,
        user_id: int,
        *,
        conn: Any | None = None,
    ) -> dict[str, Any] | None:
        """Fetch a specific key row for a user (id + user_id match)."""
        try:
            if conn is not None and self._is_postgres_backend():
                row = await conn.fetchrow(
                    "SELECT * FROM api_keys WHERE id = $1 AND user_id = $2",
                    key_id,
                    user_id,
                )
            elif conn is not None:
                cursor = await conn.execute(
                    "SELECT * FROM api_keys WHERE id = ? AND user_id = ?",
                    (key_id, user_id),
                )
                row = await cursor.fetchone()
            elif getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    "SELECT * FROM api_keys WHERE id = $1 AND user_id = $2",
                    key_id,
                    user_id,
                )
            else:
                row = await self.db_pool.fetchone(
                    "SELECT * FROM api_keys WHERE id = ? AND user_id = ?",
                    key_id,
                    user_id,
                )
            return dict(row) if row else None
        except Exception as exc:
            logger.error(f"AuthnzApiKeysRepo.fetch_key_for_user failed: {exc}")
            raise

    async def update_key_hash(self, key_id: int, key_hash: str) -> None:
        """Normalize the stored key_hash for a key id."""
        if not key_hash:
            return
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                await self.db_pool.execute(
                    "UPDATE api_keys SET key_hash = $1 WHERE id = $2",
                    key_hash,
                    key_id,
                )
            else:
                await self.db_pool.execute(
                    "UPDATE api_keys SET key_hash = ? WHERE id = ?",
                    key_hash,
                    key_id,
                )
        except Exception as exc:
            logger.error(f"AuthnzApiKeysRepo.update_key_hash failed: {exc}")
            raise

    async def fetch_key_limits(
        self,
        key_id: int,
    ) -> dict[str, Any] | None:
        """
        Fetch LLM budget and org/team limit fields for a specific API key.

        This mirrors the select used by the virtual-keys budget helpers while
        encapsulating PostgreSQL vs SQLite differences in one place.
        """
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                # PostgreSQL: use BOOL-aware COALESCE for is_virtual
                row = await self.db_pool.fetchone(
                    """
                    SELECT id,
                           COALESCE(is_virtual, FALSE) AS is_virtual,
                           org_id, team_id,
                           llm_budget_day_tokens, llm_budget_month_tokens,
                           llm_budget_day_usd, llm_budget_month_usd,
                           llm_allowed_endpoints, llm_allowed_providers, llm_allowed_models
                    FROM api_keys
                    WHERE id = $1
                    """,
                    key_id,
                )
            else:
                # SQLite: fall back to INTEGER-based COALESCE
                row = await self.db_pool.fetchone(
                    """
                    SELECT id,
                           COALESCE(is_virtual,0) AS is_virtual,
                           org_id, team_id,
                           llm_budget_day_tokens, llm_budget_month_tokens,
                           llm_budget_day_usd, llm_budget_month_usd,
                           llm_allowed_endpoints, llm_allowed_providers, llm_allowed_models
                    FROM api_keys
                    WHERE id = ?
                    """,
                    key_id,
                )

            if not row:
                return None
            return dict(row)
        except Exception as exc:  # pragma: no cover - surfaced via higher layers
            logger.error(f"AuthnzApiKeysRepo.fetch_key_limits failed: {exc}")
            raise

    async def upsert_primary_key(
        self,
        *,
        user_id: int,
        key_hash: str,
        key_identifier: str | None,
        key_prefix: str,
        name: str,
        description: str,
        scope: str,
        is_virtual: bool = False,
    ) -> None:
        """
        Upsert a primary API key row for the given user and hash.

        This mirrors the bootstrap logic for the single-user primary key,
        but is backend-agnostic and safe to call repeatedly.
        """
        async with self.db_pool.transaction() as conn:
            try:
                if self._is_postgres_backend():
                    # PostgreSQL: upsert on key_hash
                    await conn.execute(
                        """
                        INSERT INTO api_keys (
                            user_id, key_hash, key_id, key_prefix, name, description,
                            scope, status, is_virtual
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, 'active', $8)
                        ON CONFLICT (key_hash) DO UPDATE SET
                            user_id = EXCLUDED.user_id,
                            key_id = EXCLUDED.key_id,
                            key_prefix = EXCLUDED.key_prefix,
                            scope = EXCLUDED.scope,
                            status = EXCLUDED.status,
                            is_virtual = EXCLUDED.is_virtual
                        """,
                        user_id,
                        key_hash,
                        key_identifier,
                        key_prefix,
                        name,
                        description,
                        scope,
                        bool(is_virtual),
                    )
                else:
                    # SQLite: emulate upsert by key_hash
                    await conn.execute(
                        """
                        INSERT OR REPLACE INTO api_keys (
                            id, user_id, key_hash, key_id, key_prefix, name, description,
                            scope, status, is_virtual
                        )
                        VALUES (
                            COALESCE(
                                (SELECT id FROM api_keys WHERE key_hash = ?),
                                COALESCE((SELECT MAX(id) FROM api_keys), 0) + 1
                            ),
                            ?, ?, ?, ?, ?, ?, ?, 'active', ?
                        )
                        """,
                        (
                            key_hash,
                            user_id,
                            key_hash,
                            key_identifier,
                            key_prefix,
                            name,
                            description,
                            scope,
                            1 if is_virtual else 0,
                        ),
                    )
            except Exception as exc:  # pragma: no cover - surfaced via higher layers
                logger.error(f"AuthnzApiKeysRepo.upsert_primary_key failed: {exc}")
                raise

    async def list_user_keys(
        self,
        *,
        user_id: int,
        include_revoked: bool = False,
    ) -> list[dict[str, Any]]:
        """
        List API keys for a given user.

        This returns raw rows (including ``key_hash``); callers decide which
        fields to expose externally.
        """
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                # PostgreSQL backend: use $-style placeholders
                if include_revoked:
                    query = """
                        SELECT * FROM api_keys
                        WHERE user_id = $1
                        ORDER BY created_at DESC
                    """
                    rows = await self.db_pool.fetchall(query, user_id)
                else:
                    query = """
                        SELECT * FROM api_keys
                        WHERE user_id = $1 AND status = $2
                        ORDER BY created_at DESC
                    """
                    rows = await self.db_pool.fetchall(query, user_id, "active")
            else:
                # SQLite backend: retain '?' placeholders
                if include_revoked:
                    query = """
                        SELECT * FROM api_keys
                        WHERE user_id = ?
                        ORDER BY created_at DESC
                    """
                    rows = await self.db_pool.fetchall(query, user_id)
                else:
                    query = """
                        SELECT * FROM api_keys
                        WHERE user_id = ? AND status = ?
                        ORDER BY created_at DESC
                    """
                    rows = await self.db_pool.fetchall(query, user_id, "active")

            return [dict(row) for row in rows]
        except Exception as exc:  # pragma: no cover - surfaced via higher layers
            logger.error(f"AuthnzApiKeysRepo.list_user_keys failed: {exc}")
            raise

    async def create_api_key_row(
        self,
        *,
        user_id: int,
        key_hash: str,
        key_identifier: str | None,
        key_prefix: str,
        name: str | None,
        description: str | None,
        scope: str,
        expires_at: datetime | None,
        rate_limit: int | None,
        allowed_ips: list[str] | None,
        metadata: dict[str, Any] | None,
        conn: Any | None = None,
    ) -> int:
        """
        Insert a new API key row and return its id.

        This mirrors the INSERT logic used by APIKeyManager.create_api_key
        while centralizing dialect-specific SQL in the repository layer.
        """
        try:
            if conn is None:
                async with self.db_pool.transaction() as txn:
                    return await self.create_api_key_row(
                        user_id=user_id,
                        key_hash=key_hash,
                        key_identifier=key_identifier,
                        key_prefix=key_prefix,
                        name=name,
                        description=description,
                        scope=scope,
                        expires_at=expires_at,
                        rate_limit=rate_limit,
                        allowed_ips=allowed_ips,
                        metadata=metadata,
                        conn=txn,
                    )

            if self._is_postgres_backend():
                expires_at_param = expires_at
                if (
                    isinstance(expires_at_param, datetime)
                    and expires_at_param.tzinfo is not None
                ):
                    expires_at_param = expires_at_param.astimezone(timezone.utc).replace(tzinfo=None)
                key_id = await conn.fetchval(
                    """
                    INSERT INTO api_keys (
                        user_id, key_hash, key_id, key_prefix, name, description,
                        scope, expires_at, rate_limit, allowed_ips, metadata
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                    """,
                    user_id,
                    key_hash,
                    key_identifier,
                    key_prefix,
                    name,
                    description,
                    scope,
                    expires_at_param,
                    rate_limit,
                    json.dumps(allowed_ips) if allowed_ips else None,
                    json.dumps(metadata) if metadata else None,
                )
            else:
                cursor = await conn.execute(
                    """
                    INSERT INTO api_keys (
                        user_id, key_hash, key_id, key_prefix, name, description,
                        scope, expires_at, rate_limit, allowed_ips, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        key_hash,
                        key_identifier,
                        key_prefix,
                        name,
                        description,
                        scope,
                        expires_at.isoformat() if expires_at else None,
                        rate_limit,
                        json.dumps(allowed_ips) if allowed_ips else None,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
                key_id = getattr(cursor, "lastrowid", None)

            return int(key_id)
        except Exception as exc:  # pragma: no cover - surfaced via higher layers
            logger.error(f"AuthnzApiKeysRepo.create_api_key_row failed: {exc}")
            raise

    async def create_virtual_key_row(
        self,
        *,
        user_id: int,
        key_hash: str,
        key_identifier: str | None,
        key_prefix: str,
        name: str | None,
        description: str | None,
        expires_at: datetime | None,
        org_id: int | None,
        team_id: int | None,
        scope: str | None,
        allowed_endpoints: list[str] | None,
        allowed_providers: list[str] | None,
        allowed_models: list[str] | None,
        budget_day_tokens: int | None,
        budget_month_tokens: int | None,
        budget_day_usd: float | None,
        budget_month_usd: float | None,
        parent_key_id: int | None,
        allowed_methods: list[str] | None,
        allowed_paths: list[str] | None,
        max_calls: int | None,
        max_runs: int | None,
        conn: Any | None = None,
    ) -> int:
        """
        Insert a new virtual API key row and return its id.

        This mirrors the INSERT logic used by APIKeyManager.create_virtual_key
        while centralizing dialect-specific SQL in the repository layer.
        """
        try:
            meta_dict: dict[str, Any] = {}
            if allowed_methods:
                meta_dict["allowed_methods"] = [
                    str(x).upper() for x in allowed_methods
                ]
            if allowed_paths:
                meta_dict["allowed_paths"] = [str(x) for x in allowed_paths]
            if max_calls is not None:
                meta_dict["max_calls"] = int(max_calls)
            if max_runs is not None:
                meta_dict["max_runs"] = int(max_runs)

            scope_value = str(scope).strip().lower() if scope else "read"

            if conn is None:
                async with self.db_pool.transaction() as txn:
                    return await self.create_virtual_key_row(
                        user_id=user_id,
                        key_hash=key_hash,
                        key_identifier=key_identifier,
                        key_prefix=key_prefix,
                        name=name,
                        description=description,
                        expires_at=expires_at,
                        org_id=org_id,
                        team_id=team_id,
                        scope=scope,
                        allowed_endpoints=allowed_endpoints,
                        allowed_providers=allowed_providers,
                        allowed_models=allowed_models,
                        budget_day_tokens=budget_day_tokens,
                        budget_month_tokens=budget_month_tokens,
                        budget_day_usd=budget_day_usd,
                        budget_month_usd=budget_month_usd,
                        parent_key_id=parent_key_id,
                        allowed_methods=allowed_methods,
                        allowed_paths=allowed_paths,
                        max_calls=max_calls,
                        max_runs=max_runs,
                        conn=txn,
                    )

            if self._is_postgres_backend():
                _endpoints = (
                    json.dumps(allowed_endpoints)
                    if allowed_endpoints is not None
                    else None
                )
                _providers = (
                    json.dumps(allowed_providers)
                    if allowed_providers is not None
                    else None
                )
                _models = (
                    json.dumps(allowed_models)
                    if allowed_models is not None
                    else None
                )
                _metadata = json.dumps(meta_dict) if meta_dict else None
                expires_at_param = expires_at
                if (
                    isinstance(expires_at_param, datetime)
                    and expires_at_param.tzinfo is not None
                ):
                    expires_at_param = expires_at_param.astimezone(timezone.utc).replace(tzinfo=None)

                try:
                    col_type = await conn.fetchval(
                        """
                        SELECT data_type FROM information_schema.columns
                        WHERE table_name = 'api_keys' AND column_name = 'llm_allowed_endpoints'
                        """
                    )
                except (AttributeError, LookupError) as exc:
                    logger.debug(
                        "AuthnzApiKeysRepo.create_virtual_key_row: column type "
                        "detection for api_keys.llm_allowed_endpoints failed; "
                        "treating as non-JSONB: {}",
                        exc,
                    )
                    col_type = None
                is_jsonb = isinstance(col_type, str) and ("json" in col_type.lower())

                if is_jsonb:
                    key_id = await conn.fetchval(
                        """
                        INSERT INTO api_keys (
                            user_id, key_hash, key_id, key_prefix, name, description, scope, status, expires_at,
                            is_virtual, parent_key_id, org_id, team_id,
                            llm_budget_day_tokens, llm_budget_month_tokens,
                            llm_budget_day_usd, llm_budget_month_usd,
                            llm_allowed_endpoints, llm_allowed_providers, llm_allowed_models,
                            metadata
                        ) VALUES (
                            $1,$2,$3,$4,$5,$6,$7,'active',$8,
                            TRUE,$9,$10,$11,
                            $12,$13,$14,$15,$16::jsonb,$17::jsonb,$18::jsonb,
                            ($19)::jsonb
                        ) RETURNING id
                        """,
                        user_id,
                        key_hash,
                        key_identifier,
                        key_prefix,
                        name,
                        description,
                        scope_value,
                        expires_at_param,
                        parent_key_id,
                        org_id,
                        team_id,
                        budget_day_tokens,
                        budget_month_tokens,
                        budget_day_usd,
                        budget_month_usd,
                        _endpoints,
                        _providers,
                        _models,
                        _metadata,
                    )
                else:
                    key_id = await conn.fetchval(
                        """
                        INSERT INTO api_keys (
                            user_id, key_hash, key_id, key_prefix, name, description, scope, status, expires_at,
                            is_virtual, parent_key_id, org_id, team_id,
                            llm_budget_day_tokens, llm_budget_month_tokens,
                            llm_budget_day_usd, llm_budget_month_usd,
                            llm_allowed_endpoints, llm_allowed_providers, llm_allowed_models,
                            metadata
                        ) VALUES (
                            $1,$2,$3,$4,$5,$6,$7,'active',$8,
                            TRUE,$9,$10,$11,
                            $12,$13,$14,$15,$16,$17,$18,
                            $19
                        ) RETURNING id
                        """,
                        user_id,
                        key_hash,
                        key_identifier,
                        key_prefix,
                        name,
                        description,
                        scope_value,
                        expires_at_param,
                        parent_key_id,
                        org_id,
                        team_id,
                        budget_day_tokens,
                        budget_month_tokens,
                        budget_day_usd,
                        budget_month_usd,
                        _endpoints,
                        _providers,
                        _models,
                        _metadata,
                    )
            else:
                _metadata = json.dumps(meta_dict) if meta_dict else None

                cursor = await conn.execute(
                    """
                    INSERT INTO api_keys (
                        user_id, key_hash, key_id, key_prefix, name, description, scope, status, expires_at,
                        is_virtual, parent_key_id, org_id, team_id,
                        llm_budget_day_tokens, llm_budget_month_tokens,
                        llm_budget_day_usd, llm_budget_month_usd,
                        llm_allowed_endpoints, llm_allowed_providers, llm_allowed_models,
                        metadata
                    ) VALUES (?,?,?,?,?,?,?,'active',?,
                        1,?,?,?,?,?,?,?,?,?,?,?
                    )
                    """,
                    (
                        user_id,
                        key_hash,
                        key_identifier,
                        key_prefix,
                        name,
                        description,
                        scope_value,
                        expires_at.isoformat() if expires_at else None,
                        parent_key_id,
                        org_id,
                        team_id,
                        budget_day_tokens,
                        budget_month_tokens,
                        budget_day_usd,
                        budget_month_usd,
                        (json.dumps(allowed_endpoints) if allowed_endpoints else None),
                        (json.dumps(allowed_providers) if allowed_providers else None),
                        (json.dumps(allowed_models) if allowed_models else None),
                        _metadata,
                    ),
                )
                key_id = getattr(cursor, "lastrowid", None)

            return int(key_id)
        except Exception as exc:  # pragma: no cover - surfaced via higher layers
            logger.error(f"AuthnzApiKeysRepo.create_virtual_key_row failed: {exc}")
            raise

    async def delete_api_key_for_user(
        self,
        *,
        key_id: int,
        user_id: int,
    ) -> bool:
        """Delete a user-owned API key row, used for compensating rollback."""
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    result = await conn.execute(
                        "DELETE FROM api_keys WHERE id = $1 AND user_id = $2",
                        key_id,
                        user_id,
                    )
                    return _affected_row_count(result) > 0

                cursor = await conn.execute(
                    "DELETE FROM api_keys WHERE id = ? AND user_id = ?",
                    (key_id, user_id),
                )
                return _affected_row_count(cursor) > 0
        except Exception as exc:  # pragma: no cover - surfaced via higher layers
            logger.error(f"AuthnzApiKeysRepo.delete_api_key_for_user failed: {exc}")
            raise

    async def rollback_key_rotation(
        self,
        *,
        user_id: int,
        old_key_id: int,
        new_key_id: int,
        active_status: str,
        rotated_status: str,
    ) -> bool:
        """Restore the old key to active state and delete the new rotated key."""
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    restore_result = await conn.execute(
                        """
                        UPDATE api_keys
                        SET status = $1,
                            rotated_to = NULL,
                            revoked_at = NULL,
                            revoked_by = NULL,
                            revoke_reason = NULL
                        WHERE id = $2 AND user_id = $3 AND status = $4 AND rotated_to = $5
                        """,
                        active_status,
                        old_key_id,
                        user_id,
                        rotated_status,
                        new_key_id,
                    )
                    delete_result = await conn.execute(
                        """
                        DELETE FROM api_keys
                        WHERE id = $1 AND user_id = $2 AND rotated_from = $3
                        """,
                        new_key_id,
                        user_id,
                        old_key_id,
                    )
                    return (
                        _affected_row_count(restore_result) > 0
                        and _affected_row_count(delete_result) > 0
                    )

                restore_cursor = await conn.execute(
                    """
                    UPDATE api_keys
                    SET status = ?,
                        rotated_to = NULL,
                        revoked_at = NULL,
                        revoked_by = NULL,
                        revoke_reason = NULL
                    WHERE id = ? AND user_id = ? AND status = ? AND rotated_to = ?
                    """,
                    (
                        active_status,
                        old_key_id,
                        user_id,
                        rotated_status,
                        new_key_id,
                    ),
                )
                delete_cursor = await conn.execute(
                    """
                    DELETE FROM api_keys
                    WHERE id = ? AND user_id = ? AND rotated_from = ?
                    """,
                    (new_key_id, user_id, old_key_id),
                )
                return (
                    _affected_row_count(restore_cursor) > 0
                    and _affected_row_count(delete_cursor) > 0
                )
        except Exception as exc:  # pragma: no cover - surfaced via higher layers
            logger.error(f"AuthnzApiKeysRepo.rollback_key_rotation failed: {exc}")
            raise

    async def restore_revoked_api_key(
        self,
        *,
        key_id: int,
        user_id: int,
        active_status: str,
        revoked_status: str,
    ) -> bool:
        """Restore a revoked key to active state for compensating rollback."""
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    result = await conn.execute(
                        """
                        UPDATE api_keys
                        SET status = $1,
                            revoked_at = NULL,
                            revoked_by = NULL,
                            revoke_reason = NULL
                        WHERE id = $2 AND user_id = $3 AND status = $4
                        """,
                        active_status,
                        key_id,
                        user_id,
                        revoked_status,
                    )
                    return _affected_row_count(result) > 0

                cursor = await conn.execute(
                    """
                    UPDATE api_keys
                    SET status = ?,
                        revoked_at = NULL,
                        revoked_by = NULL,
                        revoke_reason = NULL
                    WHERE id = ? AND user_id = ? AND status = ?
                    """,
                    (active_status, key_id, user_id, revoked_status),
                )
                return _affected_row_count(cursor) > 0
        except Exception as exc:  # pragma: no cover - surfaced via higher layers
            logger.error(f"AuthnzApiKeysRepo.restore_revoked_api_key failed: {exc}")
            raise

    async def mark_rotated(
        self,
        *,
        old_key_id: int,
        new_key_id: int,
        rotated_status: str,
        reason: str,
        revoked_at: datetime,
    ) -> None:
        """
        Mark an existing key as rotated and link it to a new key.

        This updates:
        - old key: ``status``, ``rotated_to``, ``revoked_at``, ``revoke_reason``
        - new key: ``rotated_from``
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    norm_revoked_at = revoked_at
                    if isinstance(norm_revoked_at, datetime) and norm_revoked_at.tzinfo is not None:
                        # Store as naive UTC for TIMESTAMP columns
                        norm_revoked_at = norm_revoked_at.astimezone(timezone.utc).replace(tzinfo=None)
                    await conn.execute(
                        """
                        UPDATE api_keys
                        SET status = $1, rotated_to = $2, revoked_at = $3,
                            revoke_reason = $4
                        WHERE id = $5
                        """,
                        rotated_status,
                        new_key_id,
                        norm_revoked_at,
                        reason,
                        old_key_id,
                    )
                    await conn.execute(
                        "UPDATE api_keys SET rotated_from = $1 WHERE id = $2",
                        old_key_id,
                        new_key_id,
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE api_keys
                        SET status = ?, rotated_to = ?, revoked_at = ?,
                            revoke_reason = ?
                        WHERE id = ?
                        """,
                        (
                            rotated_status,
                            new_key_id,
                            revoked_at.isoformat(),
                            reason,
                            old_key_id,
                        ),
                    )
                    await conn.execute(
                        "UPDATE api_keys SET rotated_from = ? WHERE id = ?",
                        (old_key_id, new_key_id),
                    )
        except Exception as exc:  # pragma: no cover - surfaced via higher layers
            logger.error(f"AuthnzApiKeysRepo.mark_rotated failed: {exc}")
            raise

    async def rotate_key_atomic(
        self,
        *,
        user_id: int,
        old_key_id: int,
        new_key_hash: str,
        new_key_identifier: str | None,
        new_key_prefix: str,
        new_name: str | None,
        new_description: str | None,
        new_scope: str,
        new_expires_at: datetime | None,
        new_rate_limit: int | None,
        new_allowed_ips: list[str] | None,
        new_metadata: dict[str, Any] | None,
        active_status: str,
        rotated_status: str,
        reason: str,
        revoked_at: datetime,
        conn: Any | None = None,
    ) -> int:
        """
        Atomically create a new API key and mark the old one as rotated.

        This ensures that either both operations succeed or both fail,
        preventing the security issue where a new key is created but the
        old key remains active.

        Returns:
            The ID of the newly created key.

        Raises:
            Exception: If any part of the rotation fails (entire transaction rolls back).
        """
        try:
            if conn is None:
                async with self.db_pool.transaction() as txn:
                    return await self.rotate_key_atomic(
                        user_id=user_id,
                        old_key_id=old_key_id,
                        new_key_hash=new_key_hash,
                        new_key_identifier=new_key_identifier,
                        new_key_prefix=new_key_prefix,
                        new_name=new_name,
                        new_description=new_description,
                        new_scope=new_scope,
                        new_expires_at=new_expires_at,
                        new_rate_limit=new_rate_limit,
                        new_allowed_ips=new_allowed_ips,
                        new_metadata=new_metadata,
                        active_status=active_status,
                        rotated_status=rotated_status,
                        reason=reason,
                        revoked_at=revoked_at,
                        conn=txn,
                    )

            if self._is_postgres_backend():
                expires_at_param = new_expires_at
                if (
                    isinstance(expires_at_param, datetime)
                    and expires_at_param.tzinfo is not None
                ):
                    expires_at_param = expires_at_param.astimezone(timezone.utc).replace(tzinfo=None)

                new_key_id = await conn.fetchval(
                    """
                    INSERT INTO api_keys (
                        user_id, key_hash, key_id, key_prefix, name, description,
                        scope, expires_at, rate_limit, allowed_ips, metadata
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                    """,
                    user_id,
                    new_key_hash,
                    new_key_identifier,
                    new_key_prefix,
                    new_name,
                    new_description,
                    new_scope,
                    expires_at_param,
                    new_rate_limit,
                    json.dumps(new_allowed_ips) if new_allowed_ips else None,
                    json.dumps(new_metadata) if new_metadata else None,
                )

                norm_revoked_at = revoked_at
                if isinstance(norm_revoked_at, datetime) and norm_revoked_at.tzinfo is not None:
                    norm_revoked_at = norm_revoked_at.astimezone(timezone.utc).replace(tzinfo=None)

                old_update_result = await conn.execute(
                    """
                    UPDATE api_keys
                    SET status = $1, rotated_to = $2, revoked_at = $3,
                        revoke_reason = $4
                    WHERE id = $5 AND user_id = $6 AND status = $7
                    """,
                    rotated_status,
                    new_key_id,
                    norm_revoked_at,
                    reason,
                    old_key_id,
                    user_id,
                    active_status,
                )
                if _affected_row_count(old_update_result) == 0:
                    raise ValueError("API key not found or inactive")
                await conn.execute(
                    "UPDATE api_keys SET rotated_from = $1 WHERE id = $2",
                    old_key_id,
                    new_key_id,
                )
            else:
                cursor = await conn.execute(
                    """
                    INSERT INTO api_keys (
                        user_id, key_hash, key_id, key_prefix, name, description,
                        scope, expires_at, rate_limit, allowed_ips, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        new_key_hash,
                        new_key_identifier,
                        new_key_prefix,
                        new_name,
                        new_description,
                        new_scope,
                        new_expires_at.isoformat() if new_expires_at else None,
                        new_rate_limit,
                        json.dumps(new_allowed_ips) if new_allowed_ips else None,
                        json.dumps(new_metadata) if new_metadata else None,
                    ),
                )
                new_key_id = getattr(cursor, "lastrowid", None)

                old_update_cursor = await conn.execute(
                    """
                    UPDATE api_keys
                    SET status = ?, rotated_to = ?, revoked_at = ?,
                        revoke_reason = ?
                    WHERE id = ? AND user_id = ? AND status = ?
                    """,
                    (
                        rotated_status,
                        new_key_id,
                        revoked_at.isoformat(),
                        reason,
                        old_key_id,
                        user_id,
                        active_status,
                    ),
                )
                if _affected_row_count(old_update_cursor) == 0:
                    raise ValueError("API key not found or inactive")
                await conn.execute(
                    "UPDATE api_keys SET rotated_from = ? WHERE id = ?",
                    (old_key_id, new_key_id),
                )

            return int(new_key_id)
        except Exception as exc:
            logger.error(f"AuthnzApiKeysRepo.rotate_key_atomic failed: {exc}")
            raise

    async def revoke_api_key_for_user(
        self,
        *,
        key_id: int,
        user_id: int,
        revoked_status: str,
        active_status: str,
        reason: str,
        revoked_at: datetime,
        conn: Any | None = None,
    ) -> bool:
        """
        Revoke an API key owned by the given user.

        Returns True when a row was updated (key existed and was active), False otherwise.
        """
        try:
            if conn is None:
                async with self.db_pool.transaction() as txn:
                    return await self.revoke_api_key_for_user(
                        key_id=key_id,
                        user_id=user_id,
                        revoked_status=revoked_status,
                        active_status=active_status,
                        reason=reason,
                        revoked_at=revoked_at,
                        conn=txn,
                    )

            if self._is_postgres_backend():
                norm_revoked_at = revoked_at
                if isinstance(norm_revoked_at, datetime) and norm_revoked_at.tzinfo is not None:
                    norm_revoked_at = norm_revoked_at.astimezone(timezone.utc).replace(tzinfo=None)
                result = await conn.execute(
                    """
                    UPDATE api_keys
                    SET status = $1, revoked_at = $2, revoked_by = $3,
                        revoke_reason = $4
                    WHERE id = $5 AND user_id = $6 AND status = $7
                    """,
                    revoked_status,
                    norm_revoked_at,
                    user_id,
                    reason,
                    key_id,
                    user_id,
                    active_status,
                )
                normalized = str(result).strip().upper()
                return normalized != "UPDATE 0"

            cursor = await conn.execute(
                """
                UPDATE api_keys
                SET status = ?, revoked_at = ?, revoked_by = ?,
                    revoke_reason = ?
                WHERE id = ? AND user_id = ? AND status = ?
                """,
                (
                    revoked_status,
                    revoked_at.isoformat(),
                    user_id,
                    reason,
                    key_id,
                    user_id,
                    active_status,
                ),
            )
            return getattr(cursor, "rowcount", 0) > 0
        except Exception as exc:  # pragma: no cover - surfaced via higher layers
            logger.error(f"AuthnzApiKeysRepo.revoke_api_key_for_user failed: {exc}")
            raise

    async def increment_usage(
        self,
        *,
        key_id: int,
        ip_address: str | None,
    ) -> None:
        """
        Increment usage_count and update last_used_at/last_used_ip for a key.

        Mirrors the behavior of APIKeyManager._update_usage while centralizing
        SQL and backend differences.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                    await conn.execute(
                        """
                        UPDATE api_keys
                        SET usage_count = COALESCE(usage_count, 0) + 1,
                            last_used_at = $1,
                            last_used_ip = $2
                        WHERE id = $3
                        """,
                        now_utc,
                        ip_address,
                        key_id,
                    )
                else:
                    now_utc = datetime.now(timezone.utc)
                    await conn.execute(
                        """
                        UPDATE api_keys
                        SET usage_count = COALESCE(usage_count, 0) + 1,
                            last_used_at = ?,
                            last_used_ip = ?
                        WHERE id = ?
                        """,
                        (now_utc.isoformat(), ip_address, key_id),
                    )
        except Exception as exc:  # pragma: no cover - surfaced via higher layers
            logger.error(f"AuthnzApiKeysRepo.increment_usage failed: {exc}")
            raise

    async def expire_keys_before(
        self,
        *,
        now: datetime,
        expired_status: str,
        active_status: str,
    ) -> int:
        """
        Mark keys as expired when their expires_at is before ``now``.

        This mirrors the behavior of APIKeyManager.cleanup_expired_keys while
        centralizing SQL and backend differences.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    norm_now = now
                    if isinstance(norm_now, datetime) and norm_now.tzinfo is not None:
                        norm_now = norm_now.astimezone(timezone.utc).replace(tzinfo=None)
                    result = await conn.execute(
                        """
                        UPDATE api_keys
                        SET status = $1
                        WHERE status = $2 AND expires_at < $3
                        """,
                        expired_status,
                        active_status,
                        norm_now,
                    )
                    # asyncpg returns a status string like "UPDATE <n>"; surface
                    # unexpected formats via the outer exception handler.
                    try:
                        parts = str(result).split()
                        if not parts:
                            logger.error(
                                'AuthnzApiKeysRepo.expire_keys_before: empty status string from driver: {}',
                                result,
                            )
                            raise ValueError("Empty status string from driver")
                        affected = int(parts[-1])
                    except (ValueError, IndexError) as parse_exc:
                        logger.error(
                            'AuthnzApiKeysRepo.expire_keys_before: failed to parse driver status {}: {}',
                            result,
                            parse_exc,
                        )
                        raise
                    return affected
                cursor = await conn.execute(
                    """
                    UPDATE api_keys
                    SET status = ?
                    WHERE status = ? AND expires_at < ?
                    """,
                    (expired_status, active_status, now.isoformat()),
                )
                return int(getattr(cursor, "rowcount", 0) or 0)
        except Exception as exc:  # pragma: no cover - surfaced via higher layers
            logger.error(f"AuthnzApiKeysRepo.expire_keys_before failed: {exc}")
            raise

    async def mark_key_expired(
        self,
        *,
        key_id: int,
        expired_status: str,
    ) -> None:
        """
        Mark a key as expired by updating its status.

        Mirrors the behavior of APIKeyManager._mark_expired.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    await conn.execute(
                        "UPDATE api_keys SET status = $1 WHERE id = $2",
                        expired_status,
                        key_id,
                    )
                else:
                    await conn.execute(
                        "UPDATE api_keys SET status = ? WHERE id = ?",
                        (expired_status, key_id),
                    )
        except Exception as exc:  # pragma: no cover - surfaced via higher layers
            logger.error(f"AuthnzApiKeysRepo.mark_key_expired failed: {exc}")
            raise

    async def insert_audit_log(
        self,
        *,
        key_id: int,
        action: str,
        user_id: int | None,
        details: dict[str, Any] | None,
    ) -> None:
        """
        Insert a row into api_key_audit_log for the given key/action.

        Centralizes audit-log SQL previously embedded in APIKeyManager._log_action.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    _details = json.dumps(details) if details is not None else None
                    await conn.execute(
                        """
                        INSERT INTO api_key_audit_log (api_key_id, action, user_id, details)
                        VALUES ($1, $2, $3, $4::jsonb)
                        """,
                        key_id,
                        action,
                        user_id,
                        _details,
                    )
                else:
                    await conn.execute(
                        """
                        INSERT INTO api_key_audit_log (api_key_id, action, user_id, details)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            key_id,
                            action,
                            user_id,
                            json.dumps(details) if details else None,
                        ),
                    )
        except Exception as exc:  # pragma: no cover - surfaced via higher layers
            logger.error(f"AuthnzApiKeysRepo.insert_audit_log failed: {exc}")
            raise

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool


def _normalize_scope_type(scope_type: str) -> str:
    normalized = (scope_type or "").strip().lower()
    if normalized in {"global", "system"}:
        return "global"
    if normalized in {"org", "organization", "organizations"}:
        return "org"
    raise ValueError(f"Invalid owner_scope_type: {scope_type}")


def _normalize_scope_id(scope_type: str, scope_id: int | None) -> int | None:
    if scope_type == "global":
        return None
    if scope_id is None:
        raise ValueError("owner_scope_id is required for org-scoped identity providers")
    return int(scope_id)


def _normalize_provider_type(provider_type: str) -> str:
    normalized = (provider_type or "").strip().lower()
    if normalized == "oidc":
        return normalized
    raise ValueError(f"Unsupported provider_type: {provider_type}")


@dataclass
class IdentityProviderRepo:
    """Repository for trusted identity provider configuration."""

    db_pool: DatabasePool

    async def ensure_tables(self) -> None:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                    ensure_identity_federation_tables_pg,
                )

                ok = await ensure_identity_federation_tables_pg(self.db_pool)
                if not ok:
                    raise RuntimeError("PostgreSQL identity federation schema ensure failed")
                return

            provider_row = await self.db_pool.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='identity_providers'"
            )
            if not provider_row:
                raise RuntimeError(
                    "SQLite identity_providers table is missing. "
                    "Run the AuthNZ migrations/bootstrap (see "
                    "'python -m tldw_Server_API.app.core.AuthNZ.initialize')."
                )
        except Exception as exc:
            logger.error(f"IdentityProviderRepo.ensure_tables failed: {exc}")
            raise

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return dict(row)
        try:
            keys = row.keys()
            return {key: row[key] for key in keys}
        except Exception as row_keys_error:
            logger.bind(error_type=type(row_keys_error).__name__).debug(
                "Identity provider row key materialization failed; falling back to dict(row)",
            )
        return dict(row)

    @staticmethod
    def _normalize_datetime_for_postgres(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=None) if getattr(value, "tzinfo", None) else value

    @classmethod
    def _normalize_row(cls, row: Any) -> dict[str, Any]:
        data = cls._row_to_dict(row)
        for key in ("created_at", "updated_at"):
            value = data.get(key)
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        if "enabled" in data:
            data["enabled"] = bool(data["enabled"])
        for raw_key, clean_key in (
            ("claim_mapping_json", "claim_mapping"),
            ("provisioning_policy_json", "provisioning_policy"),
        ):
            raw_value = data.pop(raw_key, None)
            if isinstance(raw_value, str) and raw_value.strip():
                try:
                    data[clean_key] = json.loads(raw_value)
                except json.JSONDecodeError:
                    data[clean_key] = {}
            else:
                data[clean_key] = raw_value if isinstance(raw_value, dict) else {}
        return data

    async def create_provider(
        self,
        *,
        slug: str,
        provider_type: str,
        owner_scope_type: str,
        owner_scope_id: int | None,
        enabled: bool,
        issuer: str,
        claim_mapping: dict[str, Any] | None,
        provisioning_policy: dict[str, Any] | None,
        display_name: str | None = None,
        discovery_url: str | None = None,
        authorization_url: str | None = None,
        token_url: str | None = None,
        jwks_url: str | None = None,
        client_id: str | None = None,
        client_secret_ref: str | None = None,
        created_by: int | None = None,
        updated_by: int | None = None,
    ) -> dict[str, Any]:
        scope_type = _normalize_scope_type(owner_scope_type)
        scope_id = _normalize_scope_id(scope_type, owner_scope_id)
        provider_kind = _normalize_provider_type(provider_type)
        claim_mapping_json = json.dumps(claim_mapping or {})
        provisioning_policy_json = json.dumps(provisioning_policy or {})

        try:
            if getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    """
                    INSERT INTO identity_providers (
                        slug, provider_type, owner_scope_type, owner_scope_id, enabled,
                        display_name, issuer, discovery_url, authorization_url, token_url,
                        jwks_url, client_id, client_secret_ref, claim_mapping_json,
                        provisioning_policy_json, created_by, updated_by
                    )
                    VALUES (
                        $1, $2, $3, $4, $5,
                        $6, $7, $8, $9, $10,
                        $11, $12, $13, $14,
                        $15, $16, $17
                    )
                    RETURNING id, slug, provider_type, owner_scope_type, owner_scope_id, enabled,
                              display_name, issuer, discovery_url, authorization_url, token_url,
                              jwks_url, client_id, client_secret_ref, claim_mapping_json,
                              provisioning_policy_json, created_by, updated_by, created_at, updated_at
                    """,
                    slug.strip(),
                    provider_kind,
                    scope_type,
                    scope_id,
                    enabled,
                    display_name,
                    issuer.strip(),
                    discovery_url,
                    authorization_url,
                    token_url,
                    jwks_url,
                    client_id,
                    client_secret_ref,
                    claim_mapping_json,
                    provisioning_policy_json,
                    created_by,
                    updated_by,
                )
                return self._normalize_row(row) if row else {}

            cursor = await self.db_pool.execute(
                """
                INSERT INTO identity_providers (
                    slug, provider_type, owner_scope_type, owner_scope_id, enabled,
                    display_name, issuer, discovery_url, authorization_url, token_url,
                    jwks_url, client_id, client_secret_ref, claim_mapping_json,
                    provisioning_policy_json, created_by, updated_by
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    slug.strip(),
                    provider_kind,
                    scope_type,
                    scope_id,
                    int(enabled),
                    display_name,
                    issuer.strip(),
                    discovery_url,
                    authorization_url,
                    token_url,
                    jwks_url,
                    client_id,
                    client_secret_ref,
                    claim_mapping_json,
                    provisioning_policy_json,
                    created_by,
                    updated_by,
                ),
            )
            row = await self.db_pool.fetchone(
                """
                SELECT id, slug, provider_type, owner_scope_type, owner_scope_id, enabled,
                       display_name, issuer, discovery_url, authorization_url, token_url,
                       jwks_url, client_id, client_secret_ref, claim_mapping_json,
                       provisioning_policy_json, created_by, updated_by, created_at, updated_at
                FROM identity_providers
                WHERE id = ?
                """,
                (cursor.lastrowid,),
            )
            return self._normalize_row(row) if row else {}
        except Exception as exc:
            logger.error(f"IdentityProviderRepo.create_provider failed: {exc}")
            raise

    async def get_provider(self, provider_id: int) -> dict[str, Any] | None:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    """
                    SELECT id, slug, provider_type, owner_scope_type, owner_scope_id, enabled,
                           display_name, issuer, discovery_url, authorization_url, token_url,
                           jwks_url, client_id, client_secret_ref, claim_mapping_json,
                           provisioning_policy_json, created_by, updated_by, created_at, updated_at
                    FROM identity_providers
                    WHERE id = $1
                    """,
                    int(provider_id),
                )
            else:
                row = await self.db_pool.fetchone(
                    """
                    SELECT id, slug, provider_type, owner_scope_type, owner_scope_id, enabled,
                           display_name, issuer, discovery_url, authorization_url, token_url,
                           jwks_url, client_id, client_secret_ref, claim_mapping_json,
                           provisioning_policy_json, created_by, updated_by, created_at, updated_at
                    FROM identity_providers
                    WHERE id = ?
                    """,
                    (int(provider_id),),
                )
            return self._normalize_row(row) if row else None
        except Exception as exc:
            logger.error(f"IdentityProviderRepo.get_provider failed: {exc}")
            raise

    async def get_provider_by_slug(
        self,
        *,
        slug: str,
        owner_scope_type: str,
        owner_scope_id: int | None,
    ) -> dict[str, Any] | None:
        scope_type = _normalize_scope_type(owner_scope_type)
        scope_id = _normalize_scope_id(scope_type, owner_scope_id)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    """
                    SELECT id, slug, provider_type, owner_scope_type, owner_scope_id, enabled,
                           display_name, issuer, discovery_url, authorization_url, token_url,
                           jwks_url, client_id, client_secret_ref, claim_mapping_json,
                           provisioning_policy_json, created_by, updated_by, created_at, updated_at
                    FROM identity_providers
                    WHERE slug = $1 AND owner_scope_type = $2
                      AND COALESCE(owner_scope_id, 0) = COALESCE($3, 0)
                    """,
                    slug.strip(),
                    scope_type,
                    scope_id,
                )
            else:
                row = await self.db_pool.fetchone(
                    """
                    SELECT id, slug, provider_type, owner_scope_type, owner_scope_id, enabled,
                           display_name, issuer, discovery_url, authorization_url, token_url,
                           jwks_url, client_id, client_secret_ref, claim_mapping_json,
                           provisioning_policy_json, created_by, updated_by, created_at, updated_at
                    FROM identity_providers
                    WHERE slug = ? AND owner_scope_type = ?
                      AND COALESCE(owner_scope_id, 0) = COALESCE(?, 0)
                    """,
                    (slug.strip(), scope_type, scope_id),
                )
            return self._normalize_row(row) if row else None
        except Exception as exc:
            logger.error(f"IdentityProviderRepo.get_provider_by_slug failed: {exc}")
            raise

    async def list_providers(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
        enabled: bool | None = None,
    ) -> list[dict[str, Any]]:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                clauses: list[str] = []
                params: list[Any] = []
                index = 1
                if owner_scope_type is not None:
                    scope_type = _normalize_scope_type(owner_scope_type)
                    scope_id = _normalize_scope_id(scope_type, owner_scope_id)
                    clauses.append(f"owner_scope_type = ${index}")
                    params.append(scope_type)
                    index += 1
                    clauses.append(f"COALESCE(owner_scope_id, 0) = COALESCE(${index}, 0)")
                    params.append(scope_id)
                    index += 1
                if enabled is not None:
                    clauses.append(f"enabled = ${index}")
                    params.append(enabled)
                    index += 1
                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                list_sql_template = """
                    SELECT id, slug, provider_type, owner_scope_type, owner_scope_id, enabled,
                           display_name, issuer, discovery_url, authorization_url, token_url,
                           jwks_url, client_id, client_secret_ref, claim_mapping_json,
                           provisioning_policy_json, created_by, updated_by, created_at, updated_at
                    FROM identity_providers
                    {where}
                    ORDER BY owner_scope_type, slug
                """
                list_sql = list_sql_template.format_map(locals())  # nosec B608
                rows = await self.db_pool.fetchall(list_sql, *params)
            else:
                clauses = []
                params = []
                if owner_scope_type is not None:
                    scope_type = _normalize_scope_type(owner_scope_type)
                    scope_id = _normalize_scope_id(scope_type, owner_scope_id)
                    clauses.append("owner_scope_type = ?")
                    params.append(scope_type)
                    clauses.append("COALESCE(owner_scope_id, 0) = COALESCE(?, 0)")
                    params.append(scope_id)
                if enabled is not None:
                    clauses.append("enabled = ?")
                    params.append(int(enabled))
                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                list_sql_template = """
                    SELECT id, slug, provider_type, owner_scope_type, owner_scope_id, enabled,
                           display_name, issuer, discovery_url, authorization_url, token_url,
                           jwks_url, client_id, client_secret_ref, claim_mapping_json,
                           provisioning_policy_json, created_by, updated_by, created_at, updated_at
                    FROM identity_providers
                    {where}
                    ORDER BY owner_scope_type, slug
                """
                list_sql = list_sql_template.format_map(locals())  # nosec B608
                rows = await self.db_pool.fetchall(list_sql, tuple(params))
            return [self._normalize_row(row) for row in rows]
        except Exception as exc:
            logger.error(f"IdentityProviderRepo.list_providers failed: {exc}")
            raise

    async def update_provider(
        self,
        provider_id: int,
        *,
        slug: str,
        provider_type: str,
        owner_scope_type: str,
        owner_scope_id: int | None,
        enabled: bool,
        issuer: str,
        claim_mapping: dict[str, Any] | None,
        provisioning_policy: dict[str, Any] | None,
        display_name: str | None = None,
        discovery_url: str | None = None,
        authorization_url: str | None = None,
        token_url: str | None = None,
        jwks_url: str | None = None,
        client_id: str | None = None,
        client_secret_ref: str | None = None,
        updated_by: int | None = None,
    ) -> dict[str, Any] | None:
        scope_type = _normalize_scope_type(owner_scope_type)
        scope_id = _normalize_scope_id(scope_type, owner_scope_id)
        provider_kind = _normalize_provider_type(provider_type)
        claim_mapping_json = json.dumps(claim_mapping or {})
        provisioning_policy_json = json.dumps(provisioning_policy or {})

        try:
            if getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    """
                    UPDATE identity_providers
                    SET slug = $2,
                        provider_type = $3,
                        owner_scope_type = $4,
                        owner_scope_id = $5,
                        enabled = $6,
                        display_name = $7,
                        issuer = $8,
                        discovery_url = $9,
                        authorization_url = $10,
                        token_url = $11,
                        jwks_url = $12,
                        client_id = $13,
                        client_secret_ref = $14,
                        claim_mapping_json = $15,
                        provisioning_policy_json = $16,
                        updated_by = $17,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = $1
                    RETURNING id, slug, provider_type, owner_scope_type, owner_scope_id, enabled,
                              display_name, issuer, discovery_url, authorization_url, token_url,
                              jwks_url, client_id, client_secret_ref, claim_mapping_json,
                              provisioning_policy_json, created_by, updated_by, created_at, updated_at
                    """,
                    int(provider_id),
                    slug.strip(),
                    provider_kind,
                    scope_type,
                    scope_id,
                    enabled,
                    display_name,
                    issuer.strip(),
                    discovery_url,
                    authorization_url,
                    token_url,
                    jwks_url,
                    client_id,
                    client_secret_ref,
                    claim_mapping_json,
                    provisioning_policy_json,
                    updated_by,
                )
                return self._normalize_row(row) if row else None

            await self.db_pool.execute(
                """
                UPDATE identity_providers
                SET slug = ?,
                    provider_type = ?,
                    owner_scope_type = ?,
                    owner_scope_id = ?,
                    enabled = ?,
                    display_name = ?,
                    issuer = ?,
                    discovery_url = ?,
                    authorization_url = ?,
                    token_url = ?,
                    jwks_url = ?,
                    client_id = ?,
                    client_secret_ref = ?,
                    claim_mapping_json = ?,
                    provisioning_policy_json = ?,
                    updated_by = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    slug.strip(),
                    provider_kind,
                    scope_type,
                    scope_id,
                    int(enabled),
                    display_name,
                    issuer.strip(),
                    discovery_url,
                    authorization_url,
                    token_url,
                    jwks_url,
                    client_id,
                    client_secret_ref,
                    claim_mapping_json,
                    provisioning_policy_json,
                    updated_by,
                    int(provider_id),
                ),
            )
            return await self.get_provider(int(provider_id))
        except Exception as exc:
            logger.error(f"IdentityProviderRepo.update_provider failed: {exc}")
            raise

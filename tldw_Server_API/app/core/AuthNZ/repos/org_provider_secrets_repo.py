from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    ActorMembershipWriteContext,
    MembershipAuthority,
    MembershipAuthorizationError,
    MembershipScopeNotFound,
    MembershipWriter,
    validate_membership_write_context,
)
from tldw_Server_API.app.core.AuthNZ.transaction_policy import (
    get_authnz_transaction_policy,
)
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    ProviderCredentialAliasConflictError,
    fold_provider_credential_rows,
)
from tldw_Server_API.app.core.LLM_Calls.provider_identity import (
    canonical_provider_name,
    provider_lookup_names,
)


def _normalize_scope_type(scope_type: str) -> str:
    st = (scope_type or "").strip().lower()
    if st in {"org", "organization", "orgs"}:
        return "org"
    if st in {"team", "teams"}:
        return "team"
    raise ValueError(f"Invalid scope_type: {scope_type}")


def _is_active_value(value: Any) -> bool:
    if type(value) is bool:
        return value
    if type(value) is int:
        return value == 1
    if type(value) is str:
        return value.strip().lower() in {"1", "true", "active"}
    return False


@dataclass
class AuthnzOrgProviderSecretsRepo:
    """Repository for org/team shared provider secrets (BYOK)."""

    db_pool: DatabasePool

    async def ensure_tables(self) -> None:
        """Ensure org_provider_secrets schema exists."""
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                    ensure_org_provider_secrets_pg,
                )

                ok = await ensure_org_provider_secrets_pg(self.db_pool)
                if not ok:
                    raise RuntimeError("PostgreSQL org_provider_secrets schema ensure failed")
                return

            row = await self.db_pool.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='org_provider_secrets'"
            )
            if not row:
                raise RuntimeError(
                    "SQLite org_provider_secrets table is missing. "
                    "Run the AuthNZ migrations/bootstrap (see "
                    "'python -m tldw_Server_API.app.core.AuthNZ.initialize')."
                )
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzOrgProviderSecretsRepo.ensure_tables failed"
            )
            raise

    @staticmethod
    def _normalize_datetime_for_postgres(dt: datetime) -> datetime:
        return dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return dict(row)
        try:
            keys = row.keys()
            return {key: row[key] for key in keys}
        except Exception as row_keys_error:
            logger.bind(error_type=type(row_keys_error).__name__).debug(
                "Org provider secret row key materialization failed; falling back to dict(row)"
            )
        return dict(row)

    @staticmethod
    def _select_authoritative_row(
        rows: list[dict[str, Any]],
        canonical: str,
    ) -> dict[str, Any] | None:
        """Choose the authoritative stored row without rewriting its spelling."""
        canonical_rows = [
            row
            for row in rows
            if str(row.get("provider") or "").strip().lower() == canonical
        ]
        if len(canonical_rows) > 1:
            raise ProviderCredentialAliasConflictError(
                "conflicting canonical provider credentials"
            )
        if canonical_rows:
            return canonical_rows[0]
        if len(rows) > 1:
            raise ProviderCredentialAliasConflictError(
                "conflicting legacy provider credentials"
            )
        return rows[0] if rows else None

    async def _lock_active_parent_scope(
        self,
        conn: Any,
        *,
        scope_type: str,
        scope_id: int,
        postgres: bool,
    ) -> int:
        if scope_type == "org":
            if postgres:
                row = await conn.fetchrow(
                    "SELECT id, is_active FROM public.organizations "
                    "WHERE id = $1 FOR UPDATE",
                    scope_id,
                )
            else:
                cursor = await conn.execute(
                    "SELECT id, is_active FROM main.organizations WHERE id = ?",
                    (scope_id,),
                )
                row = await cursor.fetchone()
            if row is None or not bool(self._row_to_dict(row).get("is_active")):
                raise MembershipScopeNotFound()
            return scope_id

        if postgres:
            parent = await conn.fetchrow(
                "SELECT org_id FROM public.teams WHERE id = $1",
                scope_id,
            )
        else:
            cursor = await conn.execute(
                "SELECT org_id FROM main.teams WHERE id = ?",
                (scope_id,),
            )
            parent = await cursor.fetchone()
        if parent is None:
            raise MembershipScopeNotFound()
        organization_id = int(self._row_to_dict(parent)["org_id"])

        if postgres:
            organization = await conn.fetchrow(
                "SELECT id, is_active FROM public.organizations "
                "WHERE id = $1 FOR UPDATE",
                organization_id,
            )
            team = await conn.fetchrow(
                "SELECT id, org_id, is_active FROM public.teams "
                "WHERE id = $1 FOR UPDATE",
                scope_id,
            )
        else:
            cursor = await conn.execute(
                "SELECT id, is_active FROM main.organizations WHERE id = ?",
                (organization_id,),
            )
            organization = await cursor.fetchone()
            cursor = await conn.execute(
                "SELECT id, org_id, is_active FROM main.teams WHERE id = ?",
                (scope_id,),
            )
            team = await cursor.fetchone()
        if organization is None or team is None:
            raise MembershipScopeNotFound()
        organization_row = self._row_to_dict(organization)
        team_row = self._row_to_dict(team)
        if (
            not bool(organization_row.get("is_active"))
            or not bool(team_row.get("is_active"))
            or int(team_row.get("org_id") or 0) != organization_id
        ):
            raise MembershipScopeNotFound()
        return organization_id

    async def _lock_authorized_active_parent_scope(
        self,
        conn: Any,
        *,
        scope_type: str,
        scope_id: int,
        postgres: bool,
        authorization_context: ActorMembershipWriteContext | None,
    ) -> None:
        if authorization_context is None:
            await self._lock_active_parent_scope(
                conn,
                scope_type=scope_type,
                scope_id=scope_id,
                postgres=postgres,
            )
            return

        validate_membership_write_context(authorization_context, serving=True)
        actor_user_id = authorization_context.actor_user_id
        if postgres:
            actor = await conn.fetchrow(
                "SELECT id, is_active, is_superuser, role FROM public.users "
                "WHERE id = $1 FOR UPDATE",
                actor_user_id,
            )
        else:
            cursor = await conn.execute(
                "SELECT id, is_active, is_superuser, role FROM main.users "
                "WHERE id = ?",
                (actor_user_id,),
            )
            actor = await cursor.fetchone()
        if actor is None:
            raise MembershipAuthorizationError()
        actor_row = self._row_to_dict(actor)
        if not _is_active_value(actor_row.get("is_active")):
            raise MembershipAuthorizationError()

        organization_id = await self._lock_active_parent_scope(
            conn,
            scope_type=scope_type,
            scope_id=scope_id,
            postgres=postgres,
        )
        if authorization_context.required_authority is MembershipAuthority.PLATFORM_ADMIN:
            membership_writer = MembershipWriter(self.db_pool)
            await membership_writer.lock_platform_admin_authority_rows(
                conn=conn,
                context=authorization_context,
            )
            role = str(actor_row.get("role") or "").strip().lower()
            if bool(actor_row.get("is_superuser")) or role in {
                "owner",
                "super_admin",
                "admin",
            }:
                return
            if await membership_writer.has_persisted_platform_admin(
                conn,
                actor_user_id,
            ):
                return
            raise MembershipAuthorizationError()

        membership_scopes = [("org_members", "org_id", organization_id, False)]
        if scope_type == "team":
            membership_scopes.append(("team_members", "team_id", scope_id, True))
        else:
            membership_scopes[0] = (
                "org_members",
                "org_id",
                organization_id,
                True,
            )

        for table, scope_column, membership_scope_id, require_manager in membership_scopes:
            if postgres:
                membership = await conn.fetchrow(
                    f"SELECT role, status FROM public.{table} "  # nosec B608
                    f"WHERE {scope_column} = $1 AND user_id = $2 FOR UPDATE",
                    membership_scope_id,
                    actor_user_id,
                )
            else:
                cursor = await conn.execute(
                    f"SELECT role, status FROM main.{table} "  # nosec B608
                    f"WHERE {scope_column} = ? AND user_id = ?",
                    (membership_scope_id, actor_user_id),
                )
                membership = await cursor.fetchone()
            if membership is None:
                raise MembershipAuthorizationError()
            membership_row = self._row_to_dict(membership)
            if str(membership_row.get("status") or "").strip().lower() != "active":
                raise MembershipAuthorizationError()
            if require_manager and (
                str(membership_row.get("role") or "").strip().lower()
                not in {"owner", "admin", "lead"}
            ):
                raise MembershipAuthorizationError()

    async def _canonicalize_provider_identity(
        self,
        conn: Any,
        scope_type: str,
        scope_id: int,
        provider: str,
        *,
        postgres: bool,
    ) -> None:
        """Lock and collapse every stored spelling for one provider identity."""
        scope_norm = _normalize_scope_type(scope_type)
        provider_norm = canonical_provider_name(provider)
        lookup_names = provider_lookup_names(provider_norm)

        if postgres:
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext($1), hashtext($2))",
                "org_provider_secrets",
                f"{scope_norm}:{int(scope_id)}:{provider_norm}",
            )
            rows = await conn.fetch(
                """
                SELECT provider, revoked_at
                FROM public.org_provider_secrets
                WHERE scope_type = $1 AND scope_id = $2
                  AND provider = ANY($3::text[])
                ORDER BY provider
                FOR UPDATE
                """,
                scope_norm,
                int(scope_id),
                list(lookup_names),
            )
        else:
            placeholders = ", ".join("?" for _ in lookup_names)
            cursor = await conn.execute(
                f"""
                SELECT provider, revoked_at
                FROM org_provider_secrets
                WHERE scope_type = ? AND scope_id = ?
                  AND provider IN ({placeholders})
                ORDER BY provider
                """,  # nosec B608
                (scope_norm, int(scope_id), *lookup_names),
            )
            rows = await cursor.fetchall()

        materialized_rows = [self._row_to_dict(row) for row in rows]
        fold_provider_credential_rows(
            materialized_rows,
            identity_fields=("scope_type", "scope_id"),
            include_revoked=True,
        )
        stored_names = {
            str(row.get("provider") or "") for row in materialized_rows
        }

        if provider_norm in stored_names:
            legacy_names = tuple(name for name in lookup_names if name != provider_norm)
            if not legacy_names:
                return
            if postgres:
                await conn.execute(
                    """
                    DELETE FROM public.org_provider_secrets
                    WHERE scope_type = $1 AND scope_id = $2
                      AND provider = ANY($3::text[])
                    """,
                    scope_norm,
                    int(scope_id),
                    list(legacy_names),
                )
            else:
                legacy_placeholders = ", ".join("?" for _ in legacy_names)
                await conn.execute(
                    f"""
                    DELETE FROM org_provider_secrets
                    WHERE scope_type = ? AND scope_id = ?
                      AND provider IN ({legacy_placeholders})
                    """,  # nosec B608
                    (scope_norm, int(scope_id), *legacy_names),
                )
            return

        if not materialized_rows:
            return

        stored_provider = str(materialized_rows[0].get("provider") or "")
        if postgres:
            await conn.execute(
                """
                UPDATE public.org_provider_secrets
                SET provider = $1
                WHERE scope_type = $2 AND scope_id = $3 AND provider = $4
                """,
                provider_norm,
                scope_norm,
                int(scope_id),
                stored_provider,
            )
        else:
            await conn.execute(
                """
                UPDATE org_provider_secrets
                SET provider = ?
                WHERE scope_type = ? AND scope_id = ? AND provider = ?
                """,
                (provider_norm, scope_norm, int(scope_id), stored_provider),
            )

    async def upsert_secret(
        self,
        *,
        scope_type: str,
        scope_id: int,
        provider: str,
        encrypted_blob: str,
        key_hint: str | None,
        metadata: dict[str, Any] | None,
        updated_at: datetime,
        created_by: int | None = None,
        updated_by: int | None = None,
        authorization_context: ActorMembershipWriteContext | None = None,
    ) -> dict[str, Any]:
        scope_norm = _normalize_scope_type(scope_type)
        provider_norm = canonical_provider_name(provider)
        metadata_json = json.dumps(metadata) if metadata is not None else None
        try:
            postgres = getattr(self.db_pool, "pool", None) is not None
            async with self.db_pool.transaction(
                acquire_timeout_seconds=(
                    get_authnz_transaction_policy().db_pool_acquire_timeout_seconds
                ),
            ) as conn:
                await self._lock_authorized_active_parent_scope(
                    conn,
                    scope_type=scope_norm,
                    scope_id=int(scope_id),
                    postgres=postgres,
                    authorization_context=authorization_context,
                )
                await self._canonicalize_provider_identity(
                    conn,
                    scope_norm,
                    scope_id,
                    provider_norm,
                    postgres=postgres,
                )

                if postgres:
                    ts = self._normalize_datetime_for_postgres(updated_at)
                    row = await conn.fetchrow(
                        """
                        INSERT INTO public.org_provider_secrets (
                            scope_type, scope_id, provider, encrypted_blob, key_hint, metadata,
                            created_by, updated_by, created_at, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $9)
                        ON CONFLICT (scope_type, scope_id, provider) DO UPDATE SET
                            encrypted_blob = EXCLUDED.encrypted_blob,
                            key_hint = EXCLUDED.key_hint,
                            metadata = EXCLUDED.metadata,
                            updated_at = EXCLUDED.updated_at,
                            updated_by = EXCLUDED.updated_by,
                            revoked_at = NULL,
                            revoked_by = NULL
                        RETURNING id, scope_type, scope_id, provider, key_hint, metadata, created_at, updated_at,
                                  last_used_at, created_by, updated_by, revoked_by, revoked_at
                        """,
                        scope_norm,
                        int(scope_id),
                        provider_norm,
                        encrypted_blob,
                        key_hint,
                        metadata_json,
                        created_by,
                        updated_by,
                        ts,
                    )
                    return self._row_to_dict(row) if row else {}

                await conn.execute(
                    """
                    INSERT INTO org_provider_secrets (
                        scope_type, scope_id, provider, encrypted_blob, key_hint, metadata,
                        created_by, updated_by, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(scope_type, scope_id, provider) DO UPDATE SET
                        encrypted_blob = excluded.encrypted_blob,
                        key_hint = excluded.key_hint,
                        metadata = excluded.metadata,
                        updated_at = excluded.updated_at,
                        updated_by = excluded.updated_by,
                        revoked_at = NULL,
                        revoked_by = NULL
                    """,
                    (
                        scope_norm,
                        int(scope_id),
                        provider_norm,
                        encrypted_blob,
                        key_hint,
                        metadata_json,
                        created_by,
                        updated_by,
                        updated_at.isoformat(),
                        updated_at.isoformat(),
                    ),
                )
                cursor = await conn.execute(
                    """
                    SELECT id, scope_type, scope_id, provider, key_hint, metadata, created_at, updated_at,
                           last_used_at, created_by, updated_by, revoked_by, revoked_at
                    FROM org_provider_secrets
                    WHERE scope_type = ? AND scope_id = ? AND provider = ?
                    """,
                    (scope_norm, int(scope_id), provider_norm),
                )
                row = await cursor.fetchone()
                return self._row_to_dict(row) if row else {}
        except TransactionError as exc:
            if isinstance(exc.__cause__, ProviderCredentialAliasConflictError):
                raise exc.__cause__ from None
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzOrgProviderSecretsRepo.upsert_secret failed"
            )
            raise
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzOrgProviderSecretsRepo.upsert_secret failed"
            )
            raise

    async def fetch_secret(
        self,
        scope_type: str,
        scope_id: int,
        provider: str,
        *,
        include_revoked: bool = False,
    ) -> dict[str, Any] | None:
        scope_norm = _normalize_scope_type(scope_type)
        try:
            canonical, *legacy_names = provider_lookup_names(provider)
            rows = await self._fetch_secrets_for_providers(
                scope_norm,
                scope_id,
                (canonical, *legacy_names),
            )
            row = self._select_authoritative_row(rows, canonical)
            return row if row is not None and (include_revoked or row.get("revoked_at") is None) else None
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzOrgProviderSecretsRepo.fetch_secret failed"
            )
            raise

    async def fetch_secret_for_manager(
        self,
        *,
        scope_type: str,
        scope_id: int,
        provider: str,
        authorization_context: ActorMembershipWriteContext,
        include_revoked: bool = False,
    ) -> dict[str, Any] | None:
        """Read one secret after manager authority is locked and revalidated."""

        scope_norm = _normalize_scope_type(scope_type)
        canonical, *legacy_names = provider_lookup_names(provider)
        providers = (canonical, *legacy_names)
        postgres = getattr(self.db_pool, "pool", None) is not None
        try:
            async with self.db_pool.transaction(
                acquire_timeout_seconds=(
                    get_authnz_transaction_policy().db_pool_acquire_timeout_seconds
                ),
            ) as conn:
                await self._lock_authorized_active_parent_scope(
                    conn,
                    scope_type=scope_norm,
                    scope_id=int(scope_id),
                    postgres=postgres,
                    authorization_context=authorization_context,
                )
                if postgres:
                    rows = await conn.fetch(
                        """
                        SELECT id, scope_type, scope_id, provider, encrypted_blob,
                               key_hint, metadata, created_at, updated_at,
                               last_used_at, created_by, updated_by, revoked_by,
                               revoked_at
                        FROM public.org_provider_secrets
                        WHERE scope_type = $1 AND scope_id = $2
                          AND provider = ANY($3::text[])
                        """,
                        scope_norm,
                        int(scope_id),
                        list(providers),
                    )
                else:
                    placeholders = ", ".join("?" for _provider in providers)
                    cursor = await conn.execute(
                        f"""
                        SELECT id, scope_type, scope_id, provider, encrypted_blob,
                               key_hint, metadata, created_at, updated_at,
                               last_used_at, created_by, updated_by, revoked_by,
                               revoked_at
                        FROM org_provider_secrets
                        WHERE scope_type = ? AND scope_id = ?
                          AND provider IN ({placeholders})
                        """,  # nosec B608
                        (scope_norm, int(scope_id), *providers),
                    )
                    rows = await cursor.fetchall()
                row = self._select_authoritative_row(
                    [self._row_to_dict(item) for item in rows],
                    canonical,
                )
                if row is None or (not include_revoked and row.get("revoked_at") is not None):
                    return None
                return row
        except TransactionError as exc:
            if isinstance(exc.__cause__, ProviderCredentialAliasConflictError):
                raise exc.__cause__ from None
            raise

    async def list_secrets_for_manager(
        self,
        *,
        scope_type: str,
        scope_id: int,
        authorization_context: ActorMembershipWriteContext,
        include_revoked: bool = False,
    ) -> list[dict[str, Any]]:
        """List scope metadata after manager authority is locked and revalidated."""

        scope_norm = _normalize_scope_type(scope_type)
        postgres = getattr(self.db_pool, "pool", None) is not None
        try:
            async with self.db_pool.transaction(
                acquire_timeout_seconds=(
                    get_authnz_transaction_policy().db_pool_acquire_timeout_seconds
                ),
            ) as conn:
                await self._lock_authorized_active_parent_scope(
                    conn,
                    scope_type=scope_norm,
                    scope_id=int(scope_id),
                    postgres=postgres,
                    authorization_context=authorization_context,
                )
                if postgres:
                    rows = await conn.fetch(
                        """
                        SELECT id, scope_type, scope_id, provider, key_hint,
                               metadata, created_at, updated_at, last_used_at,
                               created_by, updated_by, revoked_by, revoked_at
                        FROM public.org_provider_secrets
                        WHERE scope_type = $1 AND scope_id = $2
                        ORDER BY provider
                        """,
                        scope_norm,
                        int(scope_id),
                    )
                else:
                    cursor = await conn.execute(
                        """
                        SELECT id, scope_type, scope_id, provider, key_hint,
                               metadata, created_at, updated_at, last_used_at,
                               created_by, updated_by, revoked_by, revoked_at
                        FROM org_provider_secrets
                        WHERE scope_type = ? AND scope_id = ?
                        ORDER BY provider
                        """,
                        (scope_norm, int(scope_id)),
                    )
                    rows = await cursor.fetchall()
                return fold_provider_credential_rows(
                    [self._row_to_dict(row) for row in rows],
                    identity_fields=("scope_type", "scope_id"),
                    include_revoked=include_revoked,
                )
        except TransactionError as exc:
            if isinstance(exc.__cause__, ProviderCredentialAliasConflictError):
                raise exc.__cause__ from None
            raise

    async def fetch_authorized_secret_for_user(
        self,
        scope_type: str,
        scope_id: int,
        user_id: int,
        provider: str,
    ) -> dict[str, Any] | None:
        """Atomically fetch the authoritative shared row for an active user.

        Revoked rows are returned deliberately so a revoked canonical provider
        remains authoritative over any stale active legacy alias.  The runtime
        rejects that row before decrypting it.
        """
        scope_norm = _normalize_scope_type(scope_type)
        try:
            canonical, *legacy_names = provider_lookup_names(provider)
            rows = await self._fetch_authorized_secrets_for_providers(
                scope_norm,
                scope_id,
                user_id,
                (canonical, *legacy_names),
            )
            return self._select_authoritative_row(rows, canonical)
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzOrgProviderSecretsRepo.fetch_authorized_secret_for_user failed"
            )
            raise

    async def _fetch_authorized_secrets_for_providers(
        self,
        scope_type: str,
        scope_id: int,
        user_id: int,
        providers: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        """Read every provider alias through one authorized database snapshot."""
        is_postgres = getattr(self.db_pool, "pool", None) is not None
        if is_postgres:
            select_fields = """
                SELECT s.id, s.scope_type, s.scope_id, s.provider, s.encrypted_blob,
                       s.key_hint, s.metadata, s.created_at, s.updated_at,
                       s.last_used_at, s.created_by, s.updated_by, s.revoked_by,
                       s.revoked_at
                FROM public.org_provider_secrets s
            """
            provider_placeholders = ", ".join(f"${index}" for index in range(3, 3 + len(providers)))
            user_placeholder = f"${3 + len(providers)}"
        else:
            select_fields = """
                SELECT s.id, s.scope_type, s.scope_id, s.provider, s.encrypted_blob,
                       s.key_hint, s.metadata, s.created_at, s.updated_at,
                       s.last_used_at, s.created_by, s.updated_by, s.revoked_by,
                       s.revoked_at
                FROM org_provider_secrets s
            """
            provider_placeholders = ", ".join("?" for _provider in providers)
            user_placeholder = "?"

        # Provider and user values remain bound parameters; only placeholder tokens are interpolated.
        if scope_type == "team":
            if is_postgres:
                sql = select_fields + (
                    f"""
                    JOIN public.team_members tm
                      ON tm.team_id = s.scope_id
                     AND tm.user_id = {user_placeholder}
                     AND tm.status = 'active'
                    JOIN public.teams t
                      ON t.id = s.scope_id
                     AND t.is_active = TRUE
                    JOIN public.org_members om
                      ON om.org_id = t.org_id
                     AND om.user_id = {user_placeholder}
                     AND om.status = 'active'
                    JOIN public.organizations o
                      ON o.id = t.org_id
                     AND o.is_active = TRUE
                    JOIN public.users u
                      ON u.id = {user_placeholder}
                     AND u.is_active = TRUE
                    WHERE s.scope_type = $1
                      AND s.scope_id = $2
                      AND s.provider IN ({provider_placeholders})
                """
                )
            else:
                sql = select_fields + (
                    f"""
                    JOIN team_members tm
                      ON tm.team_id = s.scope_id
                     AND tm.user_id = {user_placeholder}
                     AND tm.status = 'active'
                    JOIN teams t
                      ON t.id = s.scope_id
                     AND t.is_active = 1
                    JOIN org_members om
                      ON om.org_id = t.org_id
                     AND om.user_id = {user_placeholder}
                     AND om.status = 'active'
                    JOIN organizations o
                      ON o.id = t.org_id
                     AND o.is_active = 1
                    JOIN users u
                      ON u.id = {user_placeholder}
                     AND u.is_active = 1
                    WHERE s.scope_type = ?
                      AND s.scope_id = ?
                      AND s.provider IN ({provider_placeholders})
                """
                )
        else:
            if is_postgres:
                sql = select_fields + (
                    f"""
                    JOIN public.org_members om
                      ON om.org_id = s.scope_id
                     AND om.user_id = {user_placeholder}
                     AND om.status = 'active'
                    JOIN public.organizations o
                      ON o.id = s.scope_id
                     AND o.is_active = TRUE
                    JOIN public.users u
                      ON u.id = {user_placeholder}
                     AND u.is_active = TRUE
                    WHERE s.scope_type = $1
                      AND s.scope_id = $2
                      AND s.provider IN ({provider_placeholders})
                """
                )
            else:
                sql = select_fields + (
                    f"""
                    JOIN org_members om
                      ON om.org_id = s.scope_id
                     AND om.user_id = {user_placeholder}
                     AND om.status = 'active'
                    JOIN organizations o
                      ON o.id = s.scope_id
                     AND o.is_active = 1
                    JOIN users u
                      ON u.id = {user_placeholder}
                     AND u.is_active = 1
                    WHERE s.scope_type = ?
                      AND s.scope_id = ?
                      AND s.provider IN ({provider_placeholders})
                """
                )

        if is_postgres:
            rows = await self.db_pool.fetchall(
                sql,
                scope_type,
                int(scope_id),
                *providers,
                int(user_id),
            )
        else:
            user_parameters = (
                (int(user_id), int(user_id), int(user_id))
                if scope_type == "team"
                else (int(user_id), int(user_id))
            )
            rows = await self.db_pool.fetchall(
                sql,
                (
                    *user_parameters,
                    scope_type,
                    int(scope_id),
                    *providers,
                ),
            )
        return [self._row_to_dict(row) for row in rows]

    async def _fetch_secrets_for_providers(
        self,
        scope_type: str,
        scope_id: int,
        providers: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        """Read every alias for one scope through a single database snapshot."""
        if getattr(self.db_pool, "pool", None) is not None:
            rows = await self.db_pool.fetchall(
                """
                SELECT id, scope_type, scope_id, provider, encrypted_blob, key_hint, metadata,
                       created_at, updated_at, last_used_at, created_by, updated_by, revoked_by, revoked_at
                FROM public.org_provider_secrets
                WHERE scope_type = $1 AND scope_id = $2
                  AND provider = ANY($3::text[])
                """,
                scope_type,
                int(scope_id),
                list(providers),
            )
        else:
            placeholders = ", ".join("?" for _provider in providers)
            sql = f"""
                SELECT id, scope_type, scope_id, provider, encrypted_blob, key_hint, metadata,
                       created_at, updated_at, last_used_at, created_by, updated_by, revoked_by, revoked_at
                FROM org_provider_secrets
                WHERE scope_type = ? AND scope_id = ?
                  AND provider IN ({placeholders})
                """  # nosec B608
            rows = await self.db_pool.fetchall(
                sql,
                (scope_type, int(scope_id), *providers),
            )
        return [self._row_to_dict(row) for row in rows]

    async def list_secrets(
        self,
        *,
        scope_type: str | None = None,
        scope_id: int | None = None,
        provider: str | None = None,
        include_revoked: bool = False,
    ) -> list[dict[str, Any]]:
        if scope_id is not None and not scope_type:
            raise ValueError("scope_type is required when scope_id is provided")

        scope_norm = _normalize_scope_type(scope_type) if scope_type else None
        provider_norm = canonical_provider_name(provider) if provider else None

        try:
            if getattr(self.db_pool, "pool", None) is not None:
                clauses = []
                params: list[Any] = []
                idx = 1
                if scope_norm:
                    clauses.append(f"scope_type = ${idx}")
                    params.append(scope_norm)
                    idx += 1
                if scope_id is not None:
                    clauses.append(f"scope_id = ${idx}")
                    params.append(int(scope_id))
                    idx += 1
                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                list_secrets_sql_template = """
                    SELECT id, scope_type, scope_id, provider, key_hint, metadata, created_at, updated_at, last_used_at,
                           created_by, updated_by, revoked_by, revoked_at
                    FROM public.org_provider_secrets
                    {where}
                    ORDER BY scope_type, scope_id, provider
                    """
                list_secrets_sql = list_secrets_sql_template.format_map(locals())  # nosec B608
                rows = await self.db_pool.fetchall(
                    list_secrets_sql,
                    *params,
                )
            else:
                clauses = []
                params = []
                if scope_norm:
                    clauses.append("scope_type = ?")
                    params.append(scope_norm)
                if scope_id is not None:
                    clauses.append("scope_id = ?")
                    params.append(int(scope_id))
                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                list_secrets_sql_template = """
                    SELECT id, scope_type, scope_id, provider, key_hint, metadata, created_at, updated_at, last_used_at,
                           created_by, updated_by, revoked_by, revoked_at
                    FROM org_provider_secrets
                    {where}
                    ORDER BY scope_type, scope_id, provider
                    """
                list_secrets_sql = list_secrets_sql_template.format_map(locals())  # nosec B608
                rows = await self.db_pool.fetchall(
                    list_secrets_sql,
                    tuple(params),
                )

            folded = fold_provider_credential_rows(
                [self._row_to_dict(row) for row in rows],
                identity_fields=("scope_type", "scope_id"),
                include_revoked=include_revoked,
            )
            if provider_norm:
                folded = [row for row in folded if row.get("provider") == provider_norm]
            return folded
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzOrgProviderSecretsRepo.list_secrets failed"
            )
            raise

    async def delete_secret(
        self,
        scope_type: str,
        scope_id: int,
        provider: str,
        *,
        revoked_by: int | None = None,
        revoked_at: datetime | None = None,
        authorization_context: ActorMembershipWriteContext | None = None,
    ) -> bool:
        scope_norm = _normalize_scope_type(scope_type)
        revoked_ts = revoked_at or datetime.now(timezone.utc)
        try:
            provider_norm = canonical_provider_name(provider)
            postgres = getattr(self.db_pool, "pool", None) is not None
            async with self.db_pool.transaction(
                acquire_timeout_seconds=(
                    get_authnz_transaction_policy().db_pool_acquire_timeout_seconds
                ),
            ) as conn:
                await self._lock_authorized_active_parent_scope(
                    conn,
                    scope_type=scope_norm,
                    scope_id=int(scope_id),
                    postgres=postgres,
                    authorization_context=authorization_context,
                )
                await self._canonicalize_provider_identity(
                    conn,
                    scope_norm,
                    scope_id,
                    provider_norm,
                    postgres=postgres,
                )
                if postgres:
                    ts = self._normalize_datetime_for_postgres(revoked_ts)
                    result = await conn.execute(
                        """
                        UPDATE public.org_provider_secrets
                        SET revoked_at = $1, revoked_by = $2, updated_at = $1, updated_by = $3
                        WHERE scope_type = $4 AND scope_id = $5 AND provider = $6
                          AND revoked_at IS NULL
                        """,
                        ts,
                        revoked_by,
                        revoked_by,
                        scope_norm,
                        int(scope_id),
                        provider_norm,
                    )
                    if isinstance(result, str):
                        parts = result.split()
                        if parts and parts[-1].isdigit():
                            return int(parts[-1]) > 0
                    return True

                cursor = await conn.execute(
                    """
                    UPDATE org_provider_secrets
                    SET revoked_at = ?, revoked_by = ?, updated_at = ?, updated_by = ?
                    WHERE scope_type = ? AND scope_id = ? AND provider = ?
                      AND revoked_at IS NULL
                    """,
                    (
                        revoked_ts.isoformat(),
                        revoked_by,
                        revoked_ts.isoformat(),
                        revoked_by,
                        scope_norm,
                        int(scope_id),
                        provider_norm,
                    ),
                )
                rowcount = getattr(cursor, "rowcount", 0)
                return rowcount > 0
        except TransactionError as exc:
            if isinstance(exc.__cause__, ProviderCredentialAliasConflictError):
                raise exc.__cause__ from None
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzOrgProviderSecretsRepo.delete_secret failed"
            )
            raise
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzOrgProviderSecretsRepo.delete_secret failed"
            )
            raise

    async def touch_last_used(self, scope_type: str, scope_id: int, provider: str, used_at: datetime) -> None:
        scope_norm = _normalize_scope_type(scope_type)
        try:
            row = await self.fetch_secret(scope_norm, scope_id, provider)
            if row is None:
                return
            stored_provider = str(row.get("provider") or "")
            if getattr(self.db_pool, "pool", None) is not None:
                ts = self._normalize_datetime_for_postgres(used_at)
                await self.db_pool.execute(
                    """
                    UPDATE public.org_provider_secrets
                    SET last_used_at = $1, updated_at = $1
                    WHERE scope_type = $2 AND scope_id = $3 AND provider = $4 AND revoked_at IS NULL
                    """,
                    ts,
                    scope_norm,
                    int(scope_id),
                    stored_provider,
                )
                return

            await self.db_pool.execute(
                """
                UPDATE org_provider_secrets
                SET last_used_at = ?, updated_at = ?
                WHERE scope_type = ? AND scope_id = ? AND provider = ? AND revoked_at IS NULL
                """,
                (used_at.isoformat(), used_at.isoformat(), scope_norm, int(scope_id), stored_provider),
            )
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "AuthnzOrgProviderSecretsRepo.touch_last_used failed"
            )
            raise

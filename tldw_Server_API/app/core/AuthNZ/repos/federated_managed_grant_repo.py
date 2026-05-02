from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool


def _normalize_grant_kind(grant_kind: str) -> str:
    normalized = str(grant_kind or "").strip().lower()
    if normalized not in {"org", "team", "role"}:
        raise ValueError(f"Unsupported grant_kind: {grant_kind}")
    return normalized


def _normalize_target_ref(target_ref: str) -> str:
    normalized = str(target_ref or "").strip()
    if not normalized:
        raise ValueError("target_ref must not be empty")
    return normalized


@dataclass
class FederatedManagedGrantRepo:
    """Repository for provider-managed membership and role provenance."""

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

            row = await self.db_pool.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='federated_managed_grants'"
            )
            if not row:
                raise RuntimeError(
                    "SQLite federated_managed_grants table is missing. "
                    "Run the AuthNZ migrations/bootstrap (see "
                    "'python -m tldw_Server_API.app.core.AuthNZ.initialize')."
                )
        except Exception as exc:
            logger.error(f"FederatedManagedGrantRepo.ensure_tables failed: {exc}")
            raise

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return dict(row)
        try:
            return {key: row[key] for key in row.keys()}
        except Exception as row_keys_error:
            logger.bind(error_type=type(row_keys_error).__name__).debug(
                "Managed grant row key materialization failed; falling back to dict(row)",
            )
        return dict(row)

    @classmethod
    def _normalize_row(cls, row: Any) -> dict[str, Any]:
        data = cls._row_to_dict(row)
        for key in ("created_at", "updated_at"):
            value = data.get(key)
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    async def list_for_provider_user(
        self,
        *,
        identity_provider_id: int,
        user_id: int,
    ) -> list[dict[str, Any]]:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                rows = await self.db_pool.fetchall(
                    """
                    SELECT id, identity_provider_id, user_id, grant_kind, target_ref, created_at, updated_at
                    FROM federated_managed_grants
                    WHERE identity_provider_id = $1 AND user_id = $2
                    ORDER BY grant_kind, target_ref
                    """,
                    int(identity_provider_id),
                    int(user_id),
                )
            else:
                rows = await self.db_pool.fetchall(
                    """
                    SELECT id, identity_provider_id, user_id, grant_kind, target_ref, created_at, updated_at
                    FROM federated_managed_grants
                    WHERE identity_provider_id = ? AND user_id = ?
                    ORDER BY grant_kind, target_ref
                    """,
                    (int(identity_provider_id), int(user_id)),
                )
            return [self._normalize_row(row) for row in rows]
        except Exception as exc:
            logger.error(f"FederatedManagedGrantRepo.list_for_provider_user failed: {exc}")
            raise

    async def upsert_grant(
        self,
        *,
        identity_provider_id: int,
        user_id: int,
        grant_kind: str,
        target_ref: str,
    ) -> dict[str, Any]:
        normalized_kind = _normalize_grant_kind(grant_kind)
        normalized_target_ref = _normalize_target_ref(target_ref)

        try:
            if getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    """
                    INSERT INTO federated_managed_grants (
                        identity_provider_id, user_id, grant_kind, target_ref
                    )
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (identity_provider_id, user_id, grant_kind, target_ref) DO UPDATE SET
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id, identity_provider_id, user_id, grant_kind, target_ref, created_at, updated_at
                    """,
                    int(identity_provider_id),
                    int(user_id),
                    normalized_kind,
                    normalized_target_ref,
                )
                return self._normalize_row(row) if row else {}

            await self.db_pool.execute(
                """
                INSERT INTO federated_managed_grants (
                    identity_provider_id, user_id, grant_kind, target_ref
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(identity_provider_id, user_id, grant_kind, target_ref) DO UPDATE SET
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    int(identity_provider_id),
                    int(user_id),
                    normalized_kind,
                    normalized_target_ref,
                ),
            )
            row = await self.db_pool.fetchone(
                """
                SELECT id, identity_provider_id, user_id, grant_kind, target_ref, created_at, updated_at
                FROM federated_managed_grants
                WHERE identity_provider_id = ? AND user_id = ? AND grant_kind = ? AND target_ref = ?
                """,
                (
                    int(identity_provider_id),
                    int(user_id),
                    normalized_kind,
                    normalized_target_ref,
                ),
            )
            return self._normalize_row(row) if row else {}
        except Exception as exc:
            logger.error(f"FederatedManagedGrantRepo.upsert_grant failed: {exc}")
            raise

    async def delete_grant(
        self,
        *,
        identity_provider_id: int,
        user_id: int,
        grant_kind: str,
        target_ref: str,
    ) -> bool:
        normalized_kind = _normalize_grant_kind(grant_kind)
        normalized_target_ref = _normalize_target_ref(target_ref)

        try:
            if getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    """
                    DELETE FROM federated_managed_grants
                    WHERE identity_provider_id = $1 AND user_id = $2 AND grant_kind = $3 AND target_ref = $4
                    RETURNING id
                    """,
                    int(identity_provider_id),
                    int(user_id),
                    normalized_kind,
                    normalized_target_ref,
                )
                return row is not None

            cursor = await self.db_pool.execute(
                """
                DELETE FROM federated_managed_grants
                WHERE identity_provider_id = ? AND user_id = ? AND grant_kind = ? AND target_ref = ?
                """,
                (
                    int(identity_provider_id),
                    int(user_id),
                    normalized_kind,
                    normalized_target_ref,
                ),
            )
            try:
                return bool((cursor.rowcount or 0) > 0)
            except AttributeError:
                row = await self.db_pool.fetchone(
                    """
                    SELECT 1
                    FROM federated_managed_grants
                    WHERE identity_provider_id = ? AND user_id = ? AND grant_kind = ? AND target_ref = ?
                    """,
                    (
                        int(identity_provider_id),
                        int(user_id),
                        normalized_kind,
                        normalized_target_ref,
                    ),
                )
                return row is None
        except Exception as exc:
            logger.error(f"FederatedManagedGrantRepo.delete_grant failed: {exc}")
            raise

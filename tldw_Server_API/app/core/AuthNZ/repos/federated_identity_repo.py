from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.repos._dual_backend import row_dict


def _normalize_status(status: str | None) -> str:
    normalized = (status or "active").strip().lower()
    if not normalized:
        raise ValueError("status must not be empty")
    return normalized


@dataclass
class FederatedIdentityRepo:
    """Repository for links between external subjects and local users."""

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
            identity_row = await self.db_pool.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='federated_identities'"
            )
            if not provider_row or not identity_row:
                raise RuntimeError(
                    "SQLite identity federation tables are missing. "
                    "Run the AuthNZ migrations/bootstrap (see "
                    "'python -m tldw_Server_API.app.core.AuthNZ.initialize')."
                )
        except Exception as exc:
            logger.error(f"FederatedIdentityRepo.ensure_tables failed: {exc}")
            raise

    _row_to_dict = staticmethod(row_dict)

    @staticmethod
    def _normalize_datetime_for_postgres(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=None) if getattr(value, "tzinfo", None) else value

    @classmethod
    def _normalize_row(cls, row: Any) -> dict[str, Any]:
        data = cls._row_to_dict(row)
        for key in ("last_seen_at", "created_at", "updated_at"):
            value = data.get(key)
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    async def upsert_identity(
        self,
        *,
        identity_provider_id: int,
        external_subject: str,
        user_id: int,
        external_username: str | None = None,
        external_email: str | None = None,
        last_claims_hash: str | None = None,
        status: str = "active",
        last_seen_at: datetime | None = None,
    ) -> dict[str, Any]:
        status_value = _normalize_status(status)
        seen_at = last_seen_at or datetime.now(timezone.utc)
        normalized_subject = external_subject.strip()

        existing = await self.get_by_provider_subject(
            identity_provider_id=int(identity_provider_id),
            external_subject=normalized_subject,
        )
        if existing is not None and int(existing.get("user_id") or 0) != int(user_id):
            raise ValueError("Federated subject is already linked to a different local user")

        try:
            if getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    """
                    INSERT INTO federated_identities (
                        identity_provider_id, external_subject, user_id, external_username,
                        external_email, last_claims_hash, last_seen_at, status
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (identity_provider_id, external_subject) DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        external_username = EXCLUDED.external_username,
                        external_email = EXCLUDED.external_email,
                        last_claims_hash = EXCLUDED.last_claims_hash,
                        last_seen_at = EXCLUDED.last_seen_at,
                        status = EXCLUDED.status,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id, identity_provider_id, external_subject, user_id, external_username,
                              external_email, last_claims_hash, last_seen_at, status, created_at, updated_at
                    """,
                    int(identity_provider_id),
                    normalized_subject,
                    int(user_id),
                    external_username,
                    external_email,
                    last_claims_hash,
                    self._normalize_datetime_for_postgres(seen_at),
                    status_value,
                )
                return self._normalize_row(row) if row else {}

            await self.db_pool.execute(
                """
                INSERT INTO federated_identities (
                    identity_provider_id, external_subject, user_id, external_username,
                    external_email, last_claims_hash, last_seen_at, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(identity_provider_id, external_subject) DO UPDATE SET
                    user_id = excluded.user_id,
                    external_username = excluded.external_username,
                    external_email = excluded.external_email,
                    last_claims_hash = excluded.last_claims_hash,
                    last_seen_at = excluded.last_seen_at,
                    status = excluded.status,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    int(identity_provider_id),
                    normalized_subject,
                    int(user_id),
                    external_username,
                    external_email,
                    last_claims_hash,
                    seen_at.isoformat(),
                    status_value,
                ),
            )
            row = await self.db_pool.fetchone(
                """
                SELECT id, identity_provider_id, external_subject, user_id, external_username,
                       external_email, last_claims_hash, last_seen_at, status, created_at, updated_at
                FROM federated_identities
                WHERE identity_provider_id = ? AND external_subject = ?
                """,
                (int(identity_provider_id), normalized_subject),
            )
            return self._normalize_row(row) if row else {}
        except Exception as exc:
            logger.error(f"FederatedIdentityRepo.upsert_identity failed: {exc}")
            raise

    async def get_by_provider_subject(
        self,
        *,
        identity_provider_id: int,
        external_subject: str,
    ) -> dict[str, Any] | None:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    """
                    SELECT id, identity_provider_id, external_subject, user_id, external_username,
                           external_email, last_claims_hash, last_seen_at, status, created_at, updated_at
                    FROM federated_identities
                    WHERE identity_provider_id = $1 AND external_subject = $2
                    """,
                    int(identity_provider_id),
                    external_subject.strip(),
                )
            else:
                row = await self.db_pool.fetchone(
                    """
                    SELECT id, identity_provider_id, external_subject, user_id, external_username,
                           external_email, last_claims_hash, last_seen_at, status, created_at, updated_at
                    FROM federated_identities
                    WHERE identity_provider_id = ? AND external_subject = ?
                    """,
                    (int(identity_provider_id), external_subject.strip()),
                )
            return self._normalize_row(row) if row else None
        except Exception as exc:
            logger.error(f"FederatedIdentityRepo.get_by_provider_subject failed: {exc}")
            raise

    async def list_for_user(self, *, user_id: int) -> list[dict[str, Any]]:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                rows = await self.db_pool.fetchall(
                    """
                    SELECT id, identity_provider_id, external_subject, user_id, external_username,
                           external_email, last_claims_hash, last_seen_at, status, created_at, updated_at
                    FROM federated_identities
                    WHERE user_id = $1
                    ORDER BY identity_provider_id, external_subject
                    """,
                    int(user_id),
                )
            else:
                rows = await self.db_pool.fetchall(
                    """
                    SELECT id, identity_provider_id, external_subject, user_id, external_username,
                           external_email, last_claims_hash, last_seen_at, status, created_at, updated_at
                    FROM federated_identities
                    WHERE user_id = ?
                    ORDER BY identity_provider_id, external_subject
                    """,
                    (int(user_id),),
                )
            return [self._normalize_row(row) for row in rows]
        except Exception as exc:
            logger.error(f"FederatedIdentityRepo.list_for_user failed: {exc}")
            raise

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos._dual_backend import row_dict


def _normalize_scope_type(scope_type: str) -> str:
    value = str(scope_type or "").strip().lower()
    if not value:
        raise ValueError("scope_type is required")
    return value


def _normalize_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


@dataclass
class TelegramApprovalsRepo:
    """Persistence for pending Telegram approval button clicks."""

    db_pool: DatabasePool

    async def ensure_tables(self) -> None:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                await self.db_pool.execute(
                    """
                    CREATE TABLE IF NOT EXISTS telegram_pending_approvals (
                        id BIGSERIAL PRIMARY KEY,
                        approval_token TEXT NOT NULL UNIQUE,
                        scope_type TEXT NOT NULL,
                        scope_id BIGINT NOT NULL,
                        approval_policy_id INTEGER NULL,
                        context_key TEXT NOT NULL,
                        conversation_id TEXT NULL,
                        tool_name TEXT NOT NULL,
                        scope_key TEXT NOT NULL,
                        initiating_auth_user_id BIGINT NOT NULL,
                        expires_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                        consumed_at TIMESTAMP WITHOUT TIME ZONE NULL,
                        created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
                    )
                    """,
                )
                await self.db_pool.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_telegram_pending_approvals_active
                    ON telegram_pending_approvals (approval_token, consumed_at, expires_at)
                    """,
                )
                return

            await self.db_pool.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_pending_approvals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    approval_token TEXT NOT NULL UNIQUE,
                    scope_type TEXT NOT NULL,
                    scope_id INTEGER NOT NULL,
                    approval_policy_id INTEGER NULL,
                    context_key TEXT NOT NULL,
                    conversation_id TEXT NULL,
                    tool_name TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    initiating_auth_user_id INTEGER NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT NULL,
                    created_at TEXT NOT NULL
                )
                """,
                (),
            )
            await self.db_pool.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_telegram_pending_approvals_active
                ON telegram_pending_approvals (approval_token, consumed_at, expires_at)
                """,
                (),
            )
        except Exception as exc:
            logger.error("TelegramApprovalsRepo.ensure_tables failed: {}", exc)
            raise

    @staticmethod
    def _normalize_datetime_for_postgres(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    _row_to_dict = staticmethod(row_dict)

    async def create_pending_approval(
        self,
        *,
        approval_token: str,
        scope_type: str,
        scope_id: int,
        approval_policy_id: int | None,
        context_key: str,
        conversation_id: str | None,
        tool_name: str,
        scope_key: str,
        initiating_auth_user_id: int,
        expires_at: datetime,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        created_at = now or datetime.now(timezone.utc)
        scope_type_value = _normalize_scope_type(scope_type)
        conversation_value = _normalize_optional_text(conversation_id)
        if getattr(self.db_pool, "pool", None) is not None:
            row = await self.db_pool.fetchone(
                """
                INSERT INTO telegram_pending_approvals (
                    approval_token, scope_type, scope_id, approval_policy_id, context_key,
                    conversation_id, tool_name, scope_key, initiating_auth_user_id,
                    expires_at, consumed_at, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NULL, $11)
                RETURNING id, approval_token, scope_type, scope_id, approval_policy_id, context_key,
                          conversation_id, tool_name, scope_key, initiating_auth_user_id,
                          expires_at, consumed_at, created_at
                """,
                str(approval_token).strip(),
                scope_type_value,
                int(scope_id),
                approval_policy_id,
                str(context_key).strip(),
                conversation_value,
                str(tool_name).strip(),
                str(scope_key).strip(),
                int(initiating_auth_user_id),
                self._normalize_datetime_for_postgres(expires_at),
                self._normalize_datetime_for_postgres(created_at),
            )
            return self._row_to_dict(row)

        await self.db_pool.execute(
            """
            INSERT INTO telegram_pending_approvals (
                approval_token, scope_type, scope_id, approval_policy_id, context_key,
                conversation_id, tool_name, scope_key, initiating_auth_user_id,
                expires_at, consumed_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                str(approval_token).strip(),
                scope_type_value,
                int(scope_id),
                approval_policy_id,
                str(context_key).strip(),
                conversation_value,
                str(tool_name).strip(),
                str(scope_key).strip(),
                int(initiating_auth_user_id),
                expires_at.isoformat(),
                created_at.isoformat(),
            ),
        )
        row = await self.db_pool.fetchone(
            """
            SELECT id, approval_token, scope_type, scope_id, approval_policy_id, context_key,
                   conversation_id, tool_name, scope_key, initiating_auth_user_id,
                   expires_at, consumed_at, created_at
            FROM telegram_pending_approvals
            WHERE approval_token = ?
            """,
            (str(approval_token).strip(),),
        )
        return self._row_to_dict(row)

    async def get_pending_approval_by_token(
        self,
        approval_token: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        current = now or datetime.now(timezone.utc)
        if getattr(self.db_pool, "pool", None) is not None:
            row = await self.db_pool.fetchone(
                """
                SELECT id, approval_token, scope_type, scope_id, approval_policy_id, context_key,
                       conversation_id, tool_name, scope_key, initiating_auth_user_id,
                       expires_at, consumed_at, created_at
                FROM telegram_pending_approvals
                WHERE approval_token = $1
                  AND consumed_at IS NULL
                  AND expires_at > $2
                LIMIT 1
                """,
                str(approval_token).strip(),
                self._normalize_datetime_for_postgres(current),
            )
            return self._row_to_dict(row) if row else None

        row = await self.db_pool.fetchone(
            """
            SELECT id, approval_token, scope_type, scope_id, approval_policy_id, context_key,
                   conversation_id, tool_name, scope_key, initiating_auth_user_id,
                   expires_at, consumed_at, created_at
            FROM telegram_pending_approvals
            WHERE approval_token = ?
              AND consumed_at IS NULL
              AND datetime(expires_at) > datetime(?)
            LIMIT 1
            """,
            (
                str(approval_token).strip(),
                current.isoformat(),
            ),
        )
        return self._row_to_dict(row) if row else None

    async def consume_pending_approval(
        self,
        approval_token: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        current = now or datetime.now(timezone.utc)
        token = str(approval_token).strip()

        if getattr(self.db_pool, "pool", None) is not None:
            row = await self.db_pool.fetchone(
                """
                UPDATE telegram_pending_approvals
                SET consumed_at = $1
                WHERE approval_token = $2
                  AND consumed_at IS NULL
                  AND expires_at > $1
                RETURNING id, approval_token, scope_type, scope_id, approval_policy_id, context_key,
                          conversation_id, tool_name, scope_key, initiating_auth_user_id,
                          expires_at, consumed_at, created_at
                """,
                self._normalize_datetime_for_postgres(current),
                token,
            )
            return self._row_to_dict(row) if row else None

        async with self.db_pool.transaction() as conn:
            await conn.execute(
                """
                UPDATE telegram_pending_approvals
                SET consumed_at = ?
                WHERE approval_token = ?
                  AND consumed_at IS NULL
                  AND datetime(expires_at) > datetime(?)
                """,
                (
                    current.isoformat(),
                    token,
                    current.isoformat(),
                ),
            )
            changes_row = await conn.execute("SELECT changes() AS changed")
            changed = await changes_row.fetchone()
            if int(self._row_to_dict(changed).get("changed") or 0) <= 0:
                return None
            row_cursor = await conn.execute(
                """
                SELECT id, approval_token, scope_type, scope_id, approval_policy_id, context_key,
                       conversation_id, tool_name, scope_key, initiating_auth_user_id,
                       expires_at, consumed_at, created_at
                FROM telegram_pending_approvals
                WHERE approval_token = ?
                """,
                (token,),
            )
            row = await row_cursor.fetchone()
            return self._row_to_dict(row) if row else None


async def get_telegram_approvals_repo() -> TelegramApprovalsRepo:
    pool = await get_db_pool()
    repo = TelegramApprovalsRepo(pool)
    await repo.ensure_tables()
    return repo

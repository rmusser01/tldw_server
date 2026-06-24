from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.crypto_utils import derive_hmac_key


def _normalize_scope_type(scope_type: str) -> str:
    value = str(scope_type or "").strip().lower()
    if not value:
        raise ValueError("scope_type is required")
    return value


def _normalize_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize_pairing_code(value: Any) -> str:
    """Normalize a user-entered Telegram pairing code for lookup or storage."""
    code = str(value or "").strip().upper()
    if not code:
        raise ValueError("pairing_code is required")
    return code


@lru_cache(maxsize=1)
def _telegram_pairing_hmac_key() -> bytes:
    """Return cached HMAC key material for Telegram pairing-code digests."""
    return derive_hmac_key()


def _hash_pairing_code(pairing_code: str) -> str:
    """Build a domain-separated HMAC digest for a normalized pairing code."""
    message = f"telegram_pairing_code:{pairing_code}".encode("utf-8")
    digest = hmac.new(_telegram_pairing_hmac_key(), message, hashlib.sha256).hexdigest()
    return f"hmac_sha256:{digest}"


@dataclass
class TelegramRuntimeRepo:
    """Persistence for Telegram webhook receipts, pairing codes, and actor links."""

    db_pool: DatabasePool

    async def ensure_tables(self) -> None:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                await self.db_pool.execute(
                    """
                    CREATE TABLE IF NOT EXISTS telegram_webhook_receipts (
                        id BIGSERIAL PRIMARY KEY,
                        dedupe_key TEXT NOT NULL UNIQUE,
                        scope_type TEXT NOT NULL,
                        scope_id BIGINT NOT NULL,
                        update_id BIGINT NOT NULL,
                        expires_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                        created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
                    )
                    """,
                )
                await self.db_pool.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_telegram_webhook_receipts_expiry
                    ON telegram_webhook_receipts (expires_at)
                    """,
                )
                await self.db_pool.execute(
                    """
                    CREATE TABLE IF NOT EXISTS telegram_pairing_codes (
                        id BIGSERIAL PRIMARY KEY,
                        pairing_code TEXT NOT NULL UNIQUE,
                        scope_type TEXT NOT NULL,
                        scope_id BIGINT NOT NULL,
                        auth_user_id BIGINT NOT NULL,
                        created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                        expires_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                        consumed_at TIMESTAMP WITHOUT TIME ZONE NULL
                    )
                    """,
                )
                await self.db_pool.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_telegram_pairing_codes_active
                    ON telegram_pairing_codes (pairing_code, consumed_at, expires_at)
                    """,
                )
                await self.db_pool.execute(
                    """
                    CREATE TABLE IF NOT EXISTS telegram_actor_links (
                        id BIGSERIAL PRIMARY KEY,
                        scope_type TEXT NOT NULL,
                        scope_id BIGINT NOT NULL,
                        telegram_user_id BIGINT NOT NULL,
                        auth_user_id BIGINT NOT NULL,
                        telegram_username TEXT NULL,
                        created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                        updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                        UNIQUE (scope_type, scope_id, telegram_user_id)
                    )
                    """,
                )
                return

            await self.db_pool.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_webhook_receipts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    scope_type TEXT NOT NULL,
                    scope_id INTEGER NOT NULL,
                    update_id INTEGER NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """,
                (),
            )
            await self.db_pool.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_telegram_webhook_receipts_expiry
                ON telegram_webhook_receipts (expires_at)
                """,
                (),
            )
            await self.db_pool.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_pairing_codes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pairing_code TEXT NOT NULL UNIQUE,
                    scope_type TEXT NOT NULL,
                    scope_id INTEGER NOT NULL,
                    auth_user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT NULL
                )
                """,
                (),
            )
            await self.db_pool.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_telegram_pairing_codes_active
                ON telegram_pairing_codes (pairing_code, consumed_at, expires_at)
                """,
                (),
            )
            await self.db_pool.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_actor_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_type TEXT NOT NULL,
                    scope_id INTEGER NOT NULL,
                    telegram_user_id INTEGER NOT NULL,
                    auth_user_id INTEGER NOT NULL,
                    telegram_username TEXT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (scope_type, scope_id, telegram_user_id)
                )
                """,
                (),
            )
        except Exception as exc:
            logger.error("TelegramRuntimeRepo.ensure_tables failed: {}", exc)
            raise

    @staticmethod
    def _normalize_datetime_for_postgres(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        if row is None:
            return {}
        if isinstance(row, dict):
            return dict(row)
        try:
            return dict(row)
        except Exception:
            try:
                keys = row.keys()
                return {key: row[key] for key in keys}
            except Exception:
                return {}

    async def store_webhook_receipt(
        self,
        *,
        dedupe_key: str,
        scope_type: str,
        scope_id: int,
        update_id: int,
        expires_at: datetime,
        now: datetime | None = None,
    ) -> bool:
        current = now or datetime.now(timezone.utc)
        scope_type_value = _normalize_scope_type(scope_type)
        if getattr(self.db_pool, "pool", None) is not None:
            async with self.db_pool.transaction() as conn:
                await conn.execute(
                    """
                    DELETE FROM telegram_webhook_receipts
                    WHERE dedupe_key = $1
                      AND expires_at <= $2
                    """,
                    str(dedupe_key).strip(),
                    self._normalize_datetime_for_postgres(current),
                )
                row = await conn.fetchrow(
                    """
                    INSERT INTO telegram_webhook_receipts (
                        dedupe_key, scope_type, scope_id, update_id, expires_at, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (dedupe_key) DO NOTHING
                    RETURNING dedupe_key
                    """,
                    str(dedupe_key).strip(),
                    scope_type_value,
                    int(scope_id),
                    int(update_id),
                    self._normalize_datetime_for_postgres(expires_at),
                    self._normalize_datetime_for_postgres(current),
                )
                return row is not None

        async with self.db_pool.transaction() as conn:
            await conn.execute(
                """
                DELETE FROM telegram_webhook_receipts
                WHERE dedupe_key = ?
                  AND datetime(expires_at) <= datetime(?)
                """,
                (
                    str(dedupe_key).strip(),
                    current.isoformat(),
                ),
            )
            await conn.execute(
                """
                INSERT OR IGNORE INTO telegram_webhook_receipts (
                    dedupe_key, scope_type, scope_id, update_id, expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(dedupe_key).strip(),
                    scope_type_value,
                    int(scope_id),
                    int(update_id),
                    expires_at.isoformat(),
                    current.isoformat(),
                ),
            )
            changes_row = await conn.execute("SELECT changes() AS changed")
            changed = await changes_row.fetchone()
            return int(self._row_to_dict(changed).get("changed") or 0) > 0

    async def create_pairing_code(
        self,
        *,
        pairing_code: str,
        scope_type: str,
        scope_id: int,
        auth_user_id: int,
        expires_at: datetime,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        created_at = now or datetime.now(timezone.utc)
        code = _normalize_pairing_code(pairing_code)
        code_hash = _hash_pairing_code(code)
        scope_type_value = _normalize_scope_type(scope_type)
        if getattr(self.db_pool, "pool", None) is not None:
            row = await self.db_pool.fetchone(
                """
                INSERT INTO telegram_pairing_codes (
                    pairing_code, scope_type, scope_id, auth_user_id, created_at, expires_at, consumed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, NULL)
                RETURNING id, pairing_code, scope_type, scope_id, auth_user_id, created_at, expires_at, consumed_at
                """,
                code_hash,
                scope_type_value,
                int(scope_id),
                int(auth_user_id),
                self._normalize_datetime_for_postgres(created_at),
                self._normalize_datetime_for_postgres(expires_at),
            )
            out = self._row_to_dict(row)
            out["pairing_code"] = code
            return out

        await self.db_pool.execute(
            """
            INSERT INTO telegram_pairing_codes (
                pairing_code, scope_type, scope_id, auth_user_id, created_at, expires_at, consumed_at
            ) VALUES (?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                code_hash,
                scope_type_value,
                int(scope_id),
                int(auth_user_id),
                created_at.isoformat(),
                expires_at.isoformat(),
            ),
            )
        row = await self.db_pool.fetchone(
            """
            SELECT id, pairing_code, scope_type, scope_id, auth_user_id, created_at, expires_at, consumed_at
            FROM telegram_pairing_codes
            WHERE pairing_code = ?
            """,
            (code_hash,),
        )
        out = self._row_to_dict(row)
        out["pairing_code"] = code
        return out

    async def consume_pairing_code(
        self,
        pairing_code: str,
        *,
        scope_type: str,
        scope_id: int,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        current = now or datetime.now(timezone.utc)
        code = _normalize_pairing_code(pairing_code)
        code_hash = _hash_pairing_code(code)
        scope_type_value = _normalize_scope_type(scope_type)
        scope_id_value = int(scope_id)
        if getattr(self.db_pool, "pool", None) is not None:
            row = await self.db_pool.fetchone(
                """
                UPDATE telegram_pairing_codes
                SET consumed_at = $2
                WHERE pairing_code = $1
                  AND scope_type = $3
                  AND scope_id = $4
                  AND consumed_at IS NULL
                  AND expires_at > $2
                RETURNING id, pairing_code, scope_type, scope_id, auth_user_id, created_at, expires_at, consumed_at
                """,
                code_hash,
                self._normalize_datetime_for_postgres(current),
                scope_type_value,
                scope_id_value,
            )
            if row is None:
                row = await self.db_pool.fetchone(
                    """
                    UPDATE telegram_pairing_codes
                    SET consumed_at = $2
                    WHERE pairing_code = $1
                      AND scope_type = $3
                      AND scope_id = $4
                      AND consumed_at IS NULL
                      AND expires_at > $2
                    RETURNING id, pairing_code, scope_type, scope_id, auth_user_id, created_at, expires_at, consumed_at
                    """,
                    code,
                    self._normalize_datetime_for_postgres(current),
                    scope_type_value,
                    scope_id_value,
                )
            if row is None:
                return None
            out = self._row_to_dict(row)
            out["pairing_code"] = code
            return out

        async with self.db_pool.transaction() as conn:
            await conn.execute(
                """
                UPDATE telegram_pairing_codes
                SET consumed_at = ?
                WHERE pairing_code = ?
                  AND scope_type = ?
                  AND scope_id = ?
                  AND consumed_at IS NULL
                  AND datetime(expires_at) > datetime(?)
                """,
                (
                    current.isoformat(),
                    code_hash,
                    scope_type_value,
                    scope_id_value,
                    current.isoformat(),
                ),
            )
            changes_row = await conn.execute("SELECT changes() AS changed")
            changed = await changes_row.fetchone()
            if int(self._row_to_dict(changed).get("changed") or 0) <= 0:
                await conn.execute(
                    """
                    UPDATE telegram_pairing_codes
                    SET consumed_at = ?
                    WHERE pairing_code = ?
                      AND scope_type = ?
                      AND scope_id = ?
                      AND consumed_at IS NULL
                      AND datetime(expires_at) > datetime(?)
                    """,
                    (
                        current.isoformat(),
                        code,
                        scope_type_value,
                        scope_id_value,
                        current.isoformat(),
                    ),
                )
                changes_row = await conn.execute("SELECT changes() AS changed")
                changed = await changes_row.fetchone()
                if int(self._row_to_dict(changed).get("changed") or 0) <= 0:
                    return None
            row_cursor = await conn.execute(
                """
                SELECT id, pairing_code, scope_type, scope_id, auth_user_id, created_at, expires_at, consumed_at
                FROM telegram_pairing_codes
                WHERE pairing_code = ?
                  AND scope_type = ?
                  AND scope_id = ?
                """,
                (code_hash, scope_type_value, scope_id_value),
            )
            row = await row_cursor.fetchone()
            if row is None:
                row_cursor = await conn.execute(
                    """
                    SELECT id, pairing_code, scope_type, scope_id, auth_user_id, created_at, expires_at, consumed_at
                    FROM telegram_pairing_codes
                    WHERE pairing_code = ?
                      AND scope_type = ?
                      AND scope_id = ?
                    """,
                    (code, scope_type_value, scope_id_value),
                )
                row = await row_cursor.fetchone()
            if row is None:
                return None
            out = self._row_to_dict(row)
            out["pairing_code"] = code
            return out

    async def upsert_actor_link(
        self,
        *,
        scope_type: str,
        scope_id: int,
        telegram_user_id: int,
        auth_user_id: int,
        telegram_username: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = now or datetime.now(timezone.utc)
        scope_type_value = _normalize_scope_type(scope_type)
        username = _normalize_optional_text(telegram_username)
        if getattr(self.db_pool, "pool", None) is not None:
            row = await self.db_pool.fetchone(
                """
                INSERT INTO telegram_actor_links (
                    scope_type, scope_id, telegram_user_id, auth_user_id, telegram_username, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (scope_type, scope_id, telegram_user_id) DO UPDATE
                SET auth_user_id = EXCLUDED.auth_user_id,
                    telegram_username = EXCLUDED.telegram_username,
                    updated_at = EXCLUDED.updated_at
                RETURNING id, scope_type, scope_id, telegram_user_id, auth_user_id,
                          telegram_username, created_at, updated_at
                """,
                scope_type_value,
                int(scope_id),
                int(telegram_user_id),
                int(auth_user_id),
                username,
                self._normalize_datetime_for_postgres(current),
                self._normalize_datetime_for_postgres(current),
            )
            return self._row_to_dict(row)

        await self.db_pool.execute(
            """
            INSERT INTO telegram_actor_links (
                scope_type, scope_id, telegram_user_id, auth_user_id, telegram_username, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope_type, scope_id, telegram_user_id) DO UPDATE SET
                auth_user_id = excluded.auth_user_id,
                telegram_username = excluded.telegram_username,
                updated_at = excluded.updated_at
            """,
            (
                scope_type_value,
                int(scope_id),
                int(telegram_user_id),
                int(auth_user_id),
                username,
                current.isoformat(),
                current.isoformat(),
            ),
        )
        row = await self.db_pool.fetchone(
            """
            SELECT id, scope_type, scope_id, telegram_user_id, auth_user_id,
                   telegram_username, created_at, updated_at
            FROM telegram_actor_links
            WHERE scope_type = ?
              AND scope_id = ?
              AND telegram_user_id = ?
            """,
            (
                scope_type_value,
                int(scope_id),
                int(telegram_user_id),
            ),
        )
        return self._row_to_dict(row)

    async def get_actor_link(
        self,
        *,
        scope_type: str,
        scope_id: int,
        telegram_user_id: int,
    ) -> dict[str, Any] | None:
        scope_type_value = _normalize_scope_type(scope_type)
        if getattr(self.db_pool, "pool", None) is not None:
            row = await self.db_pool.fetchone(
                """
                SELECT id, scope_type, scope_id, telegram_user_id, auth_user_id,
                       telegram_username, created_at, updated_at
                FROM telegram_actor_links
                WHERE scope_type = $1
                  AND scope_id = $2
                  AND telegram_user_id = $3
                LIMIT 1
                """,
                scope_type_value,
                int(scope_id),
                int(telegram_user_id),
            )
            return self._row_to_dict(row) if row else None

        row = await self.db_pool.fetchone(
            """
            SELECT id, scope_type, scope_id, telegram_user_id, auth_user_id,
                   telegram_username, created_at, updated_at
            FROM telegram_actor_links
            WHERE scope_type = ?
              AND scope_id = ?
              AND telegram_user_id = ?
            LIMIT 1
            """,
            (
                scope_type_value,
                int(scope_id),
                int(telegram_user_id),
            ),
        )
        return self._row_to_dict(row) if row else None

    async def list_actor_links(
        self,
        *,
        scope_type: str,
        scope_id: int,
    ) -> list[dict[str, Any]]:
        scope_type_value = _normalize_scope_type(scope_type)
        scope_id_value = int(scope_id)
        if getattr(self.db_pool, "pool", None) is not None:
            rows = await self.db_pool.fetchall(
                """
                SELECT id, scope_type, scope_id, telegram_user_id, auth_user_id,
                       telegram_username, created_at, updated_at
                FROM telegram_actor_links
                WHERE scope_type = $1
                  AND scope_id = $2
                ORDER BY updated_at DESC, id DESC
                """,
                scope_type_value,
                scope_id_value,
            )
            return [self._row_to_dict(row) for row in rows]

        rows = await self.db_pool.fetchall(
            """
            SELECT id, scope_type, scope_id, telegram_user_id, auth_user_id,
                   telegram_username, created_at, updated_at
            FROM telegram_actor_links
            WHERE scope_type = ?
              AND scope_id = ?
            ORDER BY datetime(updated_at) DESC, id DESC
            """,
            (scope_type_value, scope_id_value),
        )
        return [self._row_to_dict(row) for row in rows]

    async def delete_actor_link(
        self,
        *,
        link_id: int,
        scope_type: str,
        scope_id: int,
    ) -> bool:
        scope_type_value = _normalize_scope_type(scope_type)
        scope_id_value = int(scope_id)
        link_id_value = int(link_id)
        if getattr(self.db_pool, "pool", None) is not None:
            result = await self.db_pool.execute(
                """
                DELETE FROM telegram_actor_links
                WHERE id = $1
                  AND scope_type = $2
                  AND scope_id = $3
                """,
                link_id_value,
                scope_type_value,
                scope_id_value,
            )
            if isinstance(result, str):
                parts = result.split()
                if parts and parts[-1].isdigit():
                    return int(parts[-1]) > 0
            return True

        cursor = await self.db_pool.execute(
            """
            DELETE FROM telegram_actor_links
            WHERE id = ?
              AND scope_type = ?
              AND scope_id = ?
            """,
            (link_id_value, scope_type_value, scope_id_value),
        )
        return getattr(cursor, "rowcount", 0) > 0

    async def clear_all_for_tests(self) -> None:
        await self.db_pool.execute("DELETE FROM telegram_webhook_receipts")
        await self.db_pool.execute("DELETE FROM telegram_pairing_codes")
        await self.db_pool.execute("DELETE FROM telegram_actor_links")


async def get_telegram_runtime_repo() -> TelegramRuntimeRepo:
    pool = await get_db_pool()
    repo = TelegramRuntimeRepo(pool)
    await repo.ensure_tables()
    return repo

"""
Repository for user profile config overrides.
"""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_authnz_core_tables_pg


def _log_override_failure(repository: str, operation: str, exc: Exception) -> None:
    logger.bind(
        repository=repository,
        operation=operation,
        exception_type=type(exc).__name__,
    ).error("Profile override repository operation failed")


async def _ensure_postgres_override_schema(db_pool: DatabasePool) -> None:
    try:
        ready = await ensure_authnz_core_tables_pg(db_pool)
    except Exception as exc:  # noqa: BLE001
        logger.bind(exception_type=type(exc).__name__).error(
            "PostgreSQL AuthNZ profile override schema readiness failed"
        )
        raise RuntimeError(
            "PostgreSQL AuthNZ profile override schema readiness failed"
        ) from None
    if not ready:
        logger.error("PostgreSQL AuthNZ profile override schema readiness failed")
        raise RuntimeError(
            "PostgreSQL AuthNZ profile override schema readiness failed"
        )



# Schema readiness already confirmed for a given pool, keyed by table name.
#
# ensure_tables() is a readiness assertion, not a migration: on SQLite it reads
# sqlite_master and raises if the table is absent. The answer cannot change while
# the process runs, but each call cost a full DatabasePool.acquire() -- ~2.1 ms,
# because establishing a connection to a WAL database opens the file and maps the
# -shm index. _build_effective_config called it up to three times per request.
#
# The marker lives on the pool object rather than in a module-level dict, so the
# memo is scoped to exactly that pool's lifetime: replacing the pool (as
# reset_db_pool does between tests) discards it automatically, and there is no
# id() reuse hazard.
_SCHEMA_READY_ATTRIBUTE = "_userprofiles_schema_verified"


def _schema_already_verified(db_pool: Any, table: str) -> bool:
    verified = getattr(db_pool, _SCHEMA_READY_ATTRIBUTE, None)
    return bool(verified) and table in verified


def _mark_schema_verified(db_pool: Any, table: str) -> None:
    """Remember a successful readiness check for this pool."""
    verified = getattr(db_pool, _SCHEMA_READY_ATTRIBUTE, None)
    if verified is None:
        verified = set()
        try:
            setattr(db_pool, _SCHEMA_READY_ATTRIBUTE, verified)
        except AttributeError:
            # A pool that rejects attributes simply never memoizes.
            return
    verified.add(table)


def reset_schema_verification_cache(db_pool: Any) -> None:
    """Forget readiness for a pool. Intended for tests."""
    with suppress(AttributeError):
        delattr(db_pool, _SCHEMA_READY_ATTRIBUTE)


@dataclass
class UserProfileOverridesRepo:
    """Repository for user profile config overrides."""

    db_pool: DatabasePool

    async def ensure_tables(self) -> None:
        """Ensure user_config_overrides schema exists."""
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                await _ensure_postgres_override_schema(self.db_pool)
                return

            if _schema_already_verified(self.db_pool, "user_config_overrides"):
                return

            row = await self.db_pool.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='user_config_overrides'"
            )
            if not row:
                raise RuntimeError(
                    "SQLite user_config_overrides table is missing. "
                    "Run the AuthNZ migrations/bootstrap (see "
                    "'python -m tldw_Server_API.app.core.AuthNZ.initialize')."
                )
            _mark_schema_verified(self.db_pool, "user_config_overrides")
        except Exception as exc:
            _log_override_failure("user", "ensure_tables", exc)
            raise

    async def list_overrides_for_user(self, user_id: int) -> list[dict[str, Any]]:
        """List overrides for a user."""
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                rows = await self.db_pool.fetchall(
                    """
                    SELECT key, value_json, updated_at, updated_by
                    FROM public.user_config_overrides
                    WHERE user_id = $1
                    ORDER BY key
                    """,
                    user_id,
                )
                return [self._row_to_dict(dict(r)) for r in rows]

            rows = await self.db_pool.fetchall(
                """
                SELECT key, value_json, updated_at, updated_by
                FROM main.user_config_overrides
                WHERE user_id = ?
                ORDER BY key
                """,
                (user_id,),
            )
            return [
                self._row_to_dict(
                    {
                        "key": r[0],
                        "value_json": r[1],
                        "updated_at": r[2],
                        "updated_by": r[3],
                    }
                )
                for r in rows
            ]
        except Exception as exc:
            _log_override_failure("user", "list_overrides", exc)
            raise

    async def upsert_override(
        self,
        *,
        user_id: int,
        key: str,
        value: Any,
        updated_by: int | None,
        db_conn: Any | None = None,
    ) -> None:
        """Insert or update a config override."""
        payload = json.dumps(value)
        ts = datetime.now(timezone.utc)
        try:
            executor = db_conn or self.db_pool
            if getattr(self.db_pool, "pool", None) is not None:
                await executor.execute(
                    """
                    INSERT INTO public.user_config_overrides (
                        user_id, key, value_json, created_at, updated_at, created_by, updated_by
                    ) VALUES ($1, $2, $3, $4, $4, $5, $6)
                    ON CONFLICT (user_id, key) DO UPDATE SET
                        value_json = EXCLUDED.value_json,
                        updated_at = EXCLUDED.updated_at,
                        updated_by = EXCLUDED.updated_by
                    """,
                    user_id,
                    key,
                    payload,
                    ts,
                    updated_by,
                    updated_by,
                )
                return

            await executor.execute(
                """
                INSERT INTO main.user_config_overrides (
                    user_id, key, value_json, created_at, updated_at, created_by, updated_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at,
                    updated_by = excluded.updated_by
                """,
                (
                    user_id,
                    key,
                    payload,
                    ts.isoformat(),
                    ts.isoformat(),
                    updated_by,
                    updated_by,
                ),
            )
        except Exception as exc:
            _log_override_failure("user", "upsert_override", exc)
            raise

    async def delete_override(self, *, user_id: int, key: str, db_conn: Any | None = None) -> None:
        """Delete a config override."""
        try:
            executor = db_conn or self.db_pool
            if getattr(self.db_pool, "pool", None) is not None:
                await executor.execute(
                    "DELETE FROM public.user_config_overrides WHERE user_id = $1 AND key = $2",
                    user_id,
                    key,
                )
                return

            await executor.execute(
                "DELETE FROM main.user_config_overrides WHERE user_id = ? AND key = ?",
                (user_id, key),
            )
        except Exception as exc:
            _log_override_failure("user", "delete_override", exc)
            raise

    async def get_latest_update_for_user(
        self,
        user_id: int,
        *,
        db_conn: Any | None = None,
    ) -> Any | None:
        """Return the latest override update timestamp for a user."""
        try:
            executor = db_conn or self.db_pool
            if getattr(self.db_pool, "pool", None) is not None:
                if db_conn is not None and hasattr(executor, "fetchrow"):
                    row = await executor.fetchrow(
                        "SELECT MAX(updated_at) AS updated_at FROM public.user_config_overrides WHERE user_id = $1",
                        user_id,
                    )
                    row = dict(row) if row else None
                else:
                    row = await executor.fetchone(
                        "SELECT MAX(updated_at) AS updated_at FROM public.user_config_overrides WHERE user_id = $1",
                        user_id,
                    )
                return row.get("updated_at") if row else None

            if db_conn is not None:
                cursor = await executor.execute(
                    "SELECT MAX(updated_at) AS updated_at FROM main.user_config_overrides WHERE user_id = ?",
                    (user_id,),
                )
                row = await cursor.fetchone()
            else:
                row = await executor.fetchone(
                    "SELECT MAX(updated_at) AS updated_at FROM main.user_config_overrides WHERE user_id = ?",
                    (user_id,),
                )
            if row is None:
                return None
            if isinstance(row, dict):
                return row.get("updated_at")
            try:
                return row["updated_at"]
            except (TypeError, KeyError, IndexError):
                try:
                    return row[0]
                except (TypeError, KeyError, IndexError):
                    return None
        except Exception as exc:
            _log_override_failure("user", "get_latest_update", exc)
            raise

    @staticmethod
    def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
        value_json = row.get("value_json")
        value: Any = None
        if value_json is not None:
            try:
                value = json.loads(value_json)
            except Exception:
                value = value_json
        return {
            "key": row.get("key"),
            "value": value,
            "updated_at": row.get("updated_at"),
            "updated_by": row.get("updated_by"),
        }


@dataclass
class OrgProfileOverridesRepo:
    """Repository for organization-level config overrides."""

    db_pool: DatabasePool

    async def ensure_tables(self) -> None:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                await _ensure_postgres_override_schema(self.db_pool)
                return

            if _schema_already_verified(self.db_pool, "org_config_overrides"):
                return

            row = await self.db_pool.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='org_config_overrides'"
            )
            if not row:
                raise RuntimeError(
                    "SQLite org_config_overrides table is missing. "
                    "Run the AuthNZ migrations/bootstrap (see "
                    "'python -m tldw_Server_API.app.core.AuthNZ.initialize')."
                )
            _mark_schema_verified(self.db_pool, "org_config_overrides")
        except Exception as exc:
            _log_override_failure("organization", "ensure_tables", exc)
            raise

    async def list_overrides_for_orgs(self, org_ids: list[int]) -> list[dict[str, Any]]:
        if not org_ids:
            return []
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                rows = await self.db_pool.fetchall(
                    """
                    SELECT org_id, key, value_json, updated_at, updated_by
                    FROM public.org_config_overrides
                    WHERE org_id = ANY($1)
                    ORDER BY org_id, key
                    """,
                    org_ids,
                )
                return [self._row_to_dict(dict(r)) for r in rows]

            placeholders = ", ".join(["?"] * len(org_ids))
            org_ids_clause = f"({placeholders})"
            list_org_overrides_sql_template = """
                SELECT org_id, key, value_json, updated_at, updated_by
                FROM main.org_config_overrides
                WHERE org_id IN {org_ids_clause}
                ORDER BY org_id, key
                """
            list_org_overrides_sql = list_org_overrides_sql_template.format_map(locals())  # nosec B608
            rows = await self.db_pool.fetchall(
                list_org_overrides_sql,
                tuple(org_ids),
            )
            return [
                self._row_to_dict(
                    {
                        "org_id": r[0],
                        "key": r[1],
                        "value_json": r[2],
                        "updated_at": r[3],
                        "updated_by": r[4],
                    }
                )
                for r in rows
            ]
        except Exception as exc:
            _log_override_failure("organization", "list_overrides", exc)
            raise

    async def upsert_override(
        self,
        *,
        org_id: int,
        key: str,
        value: Any,
        updated_by: int | None,
    ) -> None:
        payload = json.dumps(value)
        ts = datetime.now(timezone.utc)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                await self.db_pool.execute(
                    """
                    INSERT INTO public.org_config_overrides (
                        org_id, key, value_json, created_at, updated_at, created_by, updated_by
                    ) VALUES ($1, $2, $3, $4, $4, $5, $6)
                    ON CONFLICT (org_id, key) DO UPDATE SET
                        value_json = EXCLUDED.value_json,
                        updated_at = EXCLUDED.updated_at,
                        updated_by = EXCLUDED.updated_by
                    """,
                    org_id,
                    key,
                    payload,
                    ts,
                    updated_by,
                    updated_by,
                )
                return

            await self.db_pool.execute(
                """
                INSERT INTO main.org_config_overrides (
                    org_id, key, value_json, created_at, updated_at, created_by, updated_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(org_id, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at,
                    updated_by = excluded.updated_by
                """,
                (
                    org_id,
                    key,
                    payload,
                    ts.isoformat(),
                    ts.isoformat(),
                    updated_by,
                    updated_by,
                ),
            )
        except Exception as exc:
            _log_override_failure("organization", "upsert_override", exc)
            raise

    async def delete_override(self, *, org_id: int, key: str) -> None:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                await self.db_pool.execute(
                    "DELETE FROM public.org_config_overrides WHERE org_id = $1 AND key = $2",
                    org_id,
                    key,
                )
                return

            await self.db_pool.execute(
                "DELETE FROM main.org_config_overrides WHERE org_id = ? AND key = ?",
                (org_id, key),
            )
        except Exception as exc:
            _log_override_failure("organization", "delete_override", exc)
            raise

    async def get_latest_update_for_orgs(self, org_ids: list[int]) -> Any | None:
        if not org_ids:
            return None
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    "SELECT MAX(updated_at) AS updated_at FROM public.org_config_overrides WHERE org_id = ANY($1)",
                    org_ids,
                )
                return row.get("updated_at") if row else None

            placeholders = ", ".join(["?"] * len(org_ids))
            org_ids_clause = f"({placeholders})"
            latest_org_update_sql_template = (
                "SELECT MAX(updated_at) AS updated_at FROM main.org_config_overrides WHERE org_id IN {org_ids_clause}"
            )
            latest_org_update_sql = latest_org_update_sql_template.format_map(locals())  # nosec B608
            row = await self.db_pool.fetchone(
                latest_org_update_sql,
                tuple(org_ids),
            )
            if row is None:
                return None
            if isinstance(row, dict):
                return row.get("updated_at")
            try:
                return row[0]
            except Exception:
                return None
        except Exception as exc:
            _log_override_failure("organization", "get_latest_update", exc)
            raise

    @staticmethod
    def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
        value_json = row.get("value_json")
        value: Any = None
        if value_json is not None:
            try:
                value = json.loads(value_json)
            except Exception:
                value = value_json
        return {
            "org_id": row.get("org_id"),
            "key": row.get("key"),
            "value": value,
            "updated_at": row.get("updated_at"),
            "updated_by": row.get("updated_by"),
        }


@dataclass
class TeamProfileOverridesRepo:
    """Repository for team-level config overrides."""

    db_pool: DatabasePool

    async def ensure_tables(self) -> None:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                await _ensure_postgres_override_schema(self.db_pool)
                return

            if _schema_already_verified(self.db_pool, "team_config_overrides"):
                return

            row = await self.db_pool.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='team_config_overrides'"
            )
            if not row:
                raise RuntimeError(
                    "SQLite team_config_overrides table is missing. "
                    "Run the AuthNZ migrations/bootstrap (see "
                    "'python -m tldw_Server_API.app.core.AuthNZ.initialize')."
                )
            _mark_schema_verified(self.db_pool, "team_config_overrides")
        except Exception as exc:
            _log_override_failure("team", "ensure_tables", exc)
            raise

    async def list_overrides_for_teams(self, team_ids: list[int]) -> list[dict[str, Any]]:
        if not team_ids:
            return []
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                rows = await self.db_pool.fetchall(
                    """
                    SELECT team_id, key, value_json, updated_at, updated_by
                    FROM public.team_config_overrides
                    WHERE team_id = ANY($1)
                    ORDER BY team_id, key
                    """,
                    team_ids,
                )
                return [self._row_to_dict(dict(r)) for r in rows]

            placeholders = ", ".join(["?"] * len(team_ids))
            team_ids_clause = f"({placeholders})"
            list_team_overrides_sql_template = """
                SELECT team_id, key, value_json, updated_at, updated_by
                FROM main.team_config_overrides
                WHERE team_id IN {team_ids_clause}
                ORDER BY team_id, key
                """
            list_team_overrides_sql = list_team_overrides_sql_template.format_map(locals())  # nosec B608
            rows = await self.db_pool.fetchall(
                list_team_overrides_sql,
                tuple(team_ids),
            )
            return [
                self._row_to_dict(
                    {
                        "team_id": r[0],
                        "key": r[1],
                        "value_json": r[2],
                        "updated_at": r[3],
                        "updated_by": r[4],
                    }
                )
                for r in rows
            ]
        except Exception as exc:
            _log_override_failure("team", "list_overrides", exc)
            raise

    async def upsert_override(
        self,
        *,
        team_id: int,
        key: str,
        value: Any,
        updated_by: int | None,
    ) -> None:
        payload = json.dumps(value)
        ts = datetime.now(timezone.utc)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                await self.db_pool.execute(
                    """
                    INSERT INTO public.team_config_overrides (
                        team_id, key, value_json, created_at, updated_at, created_by, updated_by
                    ) VALUES ($1, $2, $3, $4, $4, $5, $6)
                    ON CONFLICT (team_id, key) DO UPDATE SET
                        value_json = EXCLUDED.value_json,
                        updated_at = EXCLUDED.updated_at,
                        updated_by = EXCLUDED.updated_by
                    """,
                    team_id,
                    key,
                    payload,
                    ts,
                    updated_by,
                    updated_by,
                )
                return

            await self.db_pool.execute(
                """
                INSERT INTO main.team_config_overrides (
                    team_id, key, value_json, created_at, updated_at, created_by, updated_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team_id, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at,
                    updated_by = excluded.updated_by
                """,
                (
                    team_id,
                    key,
                    payload,
                    ts.isoformat(),
                    ts.isoformat(),
                    updated_by,
                    updated_by,
                ),
            )
        except Exception as exc:
            _log_override_failure("team", "upsert_override", exc)
            raise

    async def delete_override(self, *, team_id: int, key: str) -> None:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                await self.db_pool.execute(
                    "DELETE FROM public.team_config_overrides WHERE team_id = $1 AND key = $2",
                    team_id,
                    key,
                )
                return

            await self.db_pool.execute(
                "DELETE FROM main.team_config_overrides WHERE team_id = ? AND key = ?",
                (team_id, key),
            )
        except Exception as exc:
            _log_override_failure("team", "delete_override", exc)
            raise

    async def get_latest_update_for_teams(self, team_ids: list[int]) -> Any | None:
        if not team_ids:
            return None
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    "SELECT MAX(updated_at) AS updated_at FROM public.team_config_overrides WHERE team_id = ANY($1)",
                    team_ids,
                )
                return row.get("updated_at") if row else None

            placeholders = ", ".join(["?"] * len(team_ids))
            team_ids_clause = f"({placeholders})"
            latest_team_update_sql_template = (
                "SELECT MAX(updated_at) AS updated_at FROM main.team_config_overrides WHERE team_id IN {team_ids_clause}"
            )
            latest_team_update_sql = latest_team_update_sql_template.format_map(locals())  # nosec B608
            row = await self.db_pool.fetchone(
                latest_team_update_sql,
                tuple(team_ids),
            )
            if row is None:
                return None
            if isinstance(row, dict):
                return row.get("updated_at")
            try:
                return row[0]
            except Exception:
                return None
        except Exception as exc:
            _log_override_failure("team", "get_latest_update", exc)
            raise

    @staticmethod
    def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
        value_json = row.get("value_json")
        value: Any = None
        if value_json is not None:
            try:
                value = json.loads(value_json)
            except Exception:
                value = value_json
        return {
            "team_id": row.get("team_id"),
            "key": row.get("key"),
            "value": value,
            "updated_at": row.get("updated_at"),
            "updated_by": row.get("updated_by"),
        }

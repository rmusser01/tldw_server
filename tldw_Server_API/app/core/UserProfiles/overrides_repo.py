"""
Repository for user profile config overrides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_authnz_core_tables_pg


@dataclass
class UserProfileOverridesRepo:
    """Repository for user profile config overrides."""

    db_pool: DatabasePool

    async def ensure_tables(self) -> None:
        """Ensure user_config_overrides schema exists."""
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                await ensure_authnz_core_tables_pg(self.db_pool)
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
        except Exception as exc:
            logger.error(f"UserProfileOverridesRepo.ensure_tables failed: {exc}")
            raise

    async def list_overrides_for_user(self, user_id: int) -> list[dict[str, Any]]:
        """List overrides for a user."""
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                rows = await self.db_pool.fetchall(
                    """
                    SELECT key, value_json, updated_at, updated_by
                    FROM user_config_overrides
                    WHERE user_id = $1
                    ORDER BY key
                    """,
                    user_id,
                )
                return [self._row_to_dict(dict(r)) for r in rows]

            rows = await self.db_pool.fetchall(
                """
                SELECT key, value_json, updated_at, updated_by
                FROM user_config_overrides
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
            logger.error(f"UserProfileOverridesRepo.list_overrides_for_user failed: {exc}")
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
                    INSERT INTO user_config_overrides (
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
                INSERT INTO user_config_overrides (
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
            logger.error(f"UserProfileOverridesRepo.upsert_override failed: {exc}")
            raise

    async def delete_override(self, *, user_id: int, key: str, db_conn: Any | None = None) -> None:
        """Delete a config override."""
        try:
            executor = db_conn or self.db_pool
            if getattr(self.db_pool, "pool", None) is not None:
                await executor.execute(
                    "DELETE FROM user_config_overrides WHERE user_id = $1 AND key = $2",
                    user_id,
                    key,
                )
                return

            await executor.execute(
                "DELETE FROM user_config_overrides WHERE user_id = ? AND key = ?",
                (user_id, key),
            )
        except Exception as exc:
            logger.error(f"UserProfileOverridesRepo.delete_override failed: {exc}")
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
                        "SELECT MAX(updated_at) AS updated_at FROM user_config_overrides WHERE user_id = $1",
                        user_id,
                    )
                    row = dict(row) if row else None
                else:
                    row = await executor.fetchone(
                        "SELECT MAX(updated_at) AS updated_at FROM user_config_overrides WHERE user_id = $1",
                        user_id,
                    )
                return row.get("updated_at") if row else None

            if db_conn is not None:
                cursor = await executor.execute(
                    "SELECT MAX(updated_at) AS updated_at FROM user_config_overrides WHERE user_id = ?",
                    (user_id,),
                )
                row = await cursor.fetchone()
            else:
                row = await executor.fetchone(
                    "SELECT MAX(updated_at) AS updated_at FROM user_config_overrides WHERE user_id = ?",
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
            logger.error(f"UserProfileOverridesRepo.get_latest_update_for_user failed: {exc}")
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
                await ensure_authnz_core_tables_pg(self.db_pool)
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
        except Exception as exc:
            logger.error(f"OrgProfileOverridesRepo.ensure_tables failed: {exc}")
            raise

    async def list_overrides_for_orgs(self, org_ids: list[int]) -> list[dict[str, Any]]:
        if not org_ids:
            return []
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                rows = await self.db_pool.fetchall(
                    """
                    SELECT org_id, key, value_json, updated_at, updated_by
                    FROM org_config_overrides
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
                FROM org_config_overrides
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
            logger.error(f"OrgProfileOverridesRepo.list_overrides_for_orgs failed: {exc}")
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
                    INSERT INTO org_config_overrides (
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
                INSERT INTO org_config_overrides (
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
            logger.error(f"OrgProfileOverridesRepo.upsert_override failed: {exc}")
            raise

    async def delete_override(self, *, org_id: int, key: str) -> None:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                await self.db_pool.execute(
                    "DELETE FROM org_config_overrides WHERE org_id = $1 AND key = $2",
                    org_id,
                    key,
                )
                return

            await self.db_pool.execute(
                "DELETE FROM org_config_overrides WHERE org_id = ? AND key = ?",
                (org_id, key),
            )
        except Exception as exc:
            logger.error(f"OrgProfileOverridesRepo.delete_override failed: {exc}")
            raise

    async def get_latest_update_for_orgs(self, org_ids: list[int]) -> Any | None:
        if not org_ids:
            return None
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    "SELECT MAX(updated_at) AS updated_at FROM org_config_overrides WHERE org_id = ANY($1)",
                    org_ids,
                )
                return row.get("updated_at") if row else None

            placeholders = ", ".join(["?"] * len(org_ids))
            org_ids_clause = f"({placeholders})"
            latest_org_update_sql_template = (
                "SELECT MAX(updated_at) AS updated_at FROM org_config_overrides WHERE org_id IN {org_ids_clause}"
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
            logger.error(f"OrgProfileOverridesRepo.get_latest_update_for_orgs failed: {exc}")
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
                await ensure_authnz_core_tables_pg(self.db_pool)
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
        except Exception as exc:
            logger.error(f"TeamProfileOverridesRepo.ensure_tables failed: {exc}")
            raise

    async def list_overrides_for_teams(self, team_ids: list[int]) -> list[dict[str, Any]]:
        if not team_ids:
            return []
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                rows = await self.db_pool.fetchall(
                    """
                    SELECT team_id, key, value_json, updated_at, updated_by
                    FROM team_config_overrides
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
                FROM team_config_overrides
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
            logger.error(f"TeamProfileOverridesRepo.list_overrides_for_teams failed: {exc}")
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
                    INSERT INTO team_config_overrides (
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
                INSERT INTO team_config_overrides (
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
            logger.error(f"TeamProfileOverridesRepo.upsert_override failed: {exc}")
            raise

    async def delete_override(self, *, team_id: int, key: str) -> None:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                await self.db_pool.execute(
                    "DELETE FROM team_config_overrides WHERE team_id = $1 AND key = $2",
                    team_id,
                    key,
                )
                return

            await self.db_pool.execute(
                "DELETE FROM team_config_overrides WHERE team_id = ? AND key = ?",
                (team_id, key),
            )
        except Exception as exc:
            logger.error(f"TeamProfileOverridesRepo.delete_override failed: {exc}")
            raise

    async def get_latest_update_for_teams(self, team_ids: list[int]) -> Any | None:
        if not team_ids:
            return None
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                row = await self.db_pool.fetchone(
                    "SELECT MAX(updated_at) AS updated_at FROM team_config_overrides WHERE team_id = ANY($1)",
                    team_ids,
                )
                return row.get("updated_at") if row else None

            placeholders = ", ".join(["?"] * len(team_ids))
            team_ids_clause = f"({placeholders})"
            latest_team_update_sql_template = (
                "SELECT MAX(updated_at) AS updated_at FROM team_config_overrides WHERE team_id IN {team_ids_clause}"
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
            logger.error(f"TeamProfileOverridesRepo.get_latest_update_for_teams failed: {exc}")
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

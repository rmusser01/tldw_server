from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos._dual_backend import row_dict
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import normalize_provider_name


def _normalize_external_id(external_id: str) -> str:
    value = str(external_id or "").strip()
    if not value:
        raise ValueError("external_id is required")
    return value


@dataclass
class WorkspaceProviderInstallationsRepo:
    """Registry for workspace-scoped provider installations."""

    db_pool: DatabasePool

    async def ensure_tables(self) -> None:
        """Ensure workspace provider installation tables exist."""
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                await self.db_pool.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workspace_provider_installations (
                        id BIGSERIAL PRIMARY KEY,
                        org_id BIGINT NOT NULL,
                        provider TEXT NOT NULL,
                        external_id TEXT NOT NULL,
                        display_name TEXT NULL,
                        installed_by_user_id BIGINT NULL,
                        disabled BOOLEAN NOT NULL DEFAULT FALSE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        last_health_status TEXT NULL,
                        last_health_checked_at TIMESTAMPTZ NULL,
                        metadata_json TEXT NULL,
                        CONSTRAINT uq_workspace_provider_installations UNIQUE (org_id, provider, external_id)
                    )
                    """
                )
                await self.db_pool.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_workspace_provider_installations_org_provider
                    ON workspace_provider_installations (org_id, provider)
                    """
                )
                await self.db_pool.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_workspace_provider_installations_external_id
                    ON workspace_provider_installations (external_id)
                    """
                )
                return

            await self.db_pool.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_provider_installations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    external_id TEXT NOT NULL,
                    display_name TEXT NULL,
                    installed_by_user_id INTEGER NULL,
                    disabled INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_health_status TEXT NULL,
                    last_health_checked_at TEXT NULL,
                    metadata_json TEXT NULL,
                    UNIQUE (org_id, provider, external_id)
                )
                """
            )
            await self.db_pool.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workspace_provider_installations_org_provider
                ON workspace_provider_installations (org_id, provider)
                """
            )
            await self.db_pool.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workspace_provider_installations_external_id
                ON workspace_provider_installations (external_id)
                """
            )
        except Exception as exc:
            logger.error(f"WorkspaceProviderInstallationsRepo.ensure_tables failed: {exc}")
            raise

    @staticmethod
    def _normalize_datetime_for_postgres(dt: datetime) -> datetime:
        return dt.astimezone(timezone.utc).replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt

    _row_to_dict = staticmethod(row_dict)

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        for key in ("id", "org_id", "installed_by_user_id"):
            value = normalized.get(key)
            if value is not None:
                try:
                    normalized[key] = int(value)
                except Exception:
                    normalized[key] = value
        disabled_value = normalized.get("disabled")
        if isinstance(disabled_value, str):
            normalized["disabled"] = disabled_value.strip().lower() in {"1", "true", "t", "yes", "on"}
        else:
            normalized["disabled"] = bool(disabled_value)
        provider_value = normalized.get("provider")
        if provider_value is not None:
            normalized["provider"] = normalize_provider_name(str(provider_value))
        for key in ("external_id", "display_name", "last_health_status", "metadata_json"):
            if normalized.get(key) is not None:
                normalized[key] = str(normalized[key])
        return normalized

    async def upsert_installation(
        self,
        *,
        org_id: int,
        provider: str,
        external_id: str,
        display_name: str | None = None,
        installed_by_user_id: int | None = None,
        disabled: bool = False,
        metadata: dict[str, Any] | None = None,
        last_health_status: str | None = None,
        last_health_checked_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> dict[str, Any]:
        org_id_value = int(org_id)
        provider_norm = normalize_provider_name(provider)
        external_id_value = _normalize_external_id(external_id)
        display_name_value = str(display_name).strip() if display_name is not None else None
        metadata_json = json.dumps(metadata) if metadata is not None else None
        updated_at_value = updated_at or datetime.now(timezone.utc)
        last_health_checked_value = last_health_checked_at

        try:
            if getattr(self.db_pool, "pool", None) is not None:
                ts = self._normalize_datetime_for_postgres(updated_at_value)
                health_checked_ts = (
                    self._normalize_datetime_for_postgres(last_health_checked_value)
                    if last_health_checked_value is not None
                    else None
                )
                await self.db_pool.execute(
                    """
                    INSERT INTO workspace_provider_installations (
                        org_id, provider, external_id, display_name, installed_by_user_id,
                        disabled, created_at, updated_at, last_health_status,
                        last_health_checked_at, metadata_json
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $7, $8, $9, $10)
                    ON CONFLICT (org_id, provider, external_id) DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        installed_by_user_id = EXCLUDED.installed_by_user_id,
                        disabled = EXCLUDED.disabled,
                        updated_at = EXCLUDED.updated_at,
                        last_health_status = EXCLUDED.last_health_status,
                        last_health_checked_at = EXCLUDED.last_health_checked_at,
                        metadata_json = EXCLUDED.metadata_json
                    """,
                    org_id_value,
                    provider_norm,
                    external_id_value,
                    display_name_value,
                    installed_by_user_id,
                    bool(disabled),
                    ts,
                    last_health_status,
                    health_checked_ts,
                    metadata_json,
                )
                row = await self.db_pool.fetchone(
                    """
                    SELECT id, org_id, provider, external_id, display_name, installed_by_user_id,
                           disabled, created_at, updated_at, last_health_status, last_health_checked_at, metadata_json
                    FROM workspace_provider_installations
                    WHERE org_id = $1 AND provider = $2 AND external_id = $3
                    """,
                    org_id_value,
                    provider_norm,
                    external_id_value,
                )
            else:
                await self.db_pool.execute(
                    """
                    INSERT INTO workspace_provider_installations (
                        org_id, provider, external_id, display_name, installed_by_user_id,
                        disabled, created_at, updated_at, last_health_status,
                        last_health_checked_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (org_id, provider, external_id) DO UPDATE SET
                        display_name = excluded.display_name,
                        installed_by_user_id = excluded.installed_by_user_id,
                        disabled = excluded.disabled,
                        updated_at = excluded.updated_at,
                        last_health_status = excluded.last_health_status,
                        last_health_checked_at = excluded.last_health_checked_at,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        org_id_value,
                        provider_norm,
                        external_id_value,
                        display_name_value,
                        installed_by_user_id,
                        1 if disabled else 0,
                        updated_at_value.isoformat(),
                        updated_at_value.isoformat(),
                        last_health_status,
                        last_health_checked_value.isoformat() if last_health_checked_value else None,
                        metadata_json,
                    ),
                )
                row = await self.db_pool.fetchone(
                    """
                    SELECT id, org_id, provider, external_id, display_name, installed_by_user_id,
                           disabled, created_at, updated_at, last_health_status, last_health_checked_at, metadata_json
                    FROM workspace_provider_installations
                    WHERE org_id = ? AND provider = ? AND external_id = ?
                    """,
                    (org_id_value, provider_norm, external_id_value),
                )
            return self._normalize_row(self._row_to_dict(row)) if row else {}
        except Exception as exc:
            logger.error(f"WorkspaceProviderInstallationsRepo.upsert_installation failed: {exc}")
            raise

    async def list_installations(
        self,
        *,
        org_id: int,
        provider: str | None = None,
        include_disabled: bool = True,
    ) -> list[dict[str, Any]]:
        org_id_value = int(org_id)
        provider_norm = normalize_provider_name(provider) if provider is not None else None
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                if provider_norm is not None and include_disabled:
                    rows = await self.db_pool.fetchall(
                        """
                        SELECT id, org_id, provider, external_id, display_name, installed_by_user_id,
                               disabled, created_at, updated_at, last_health_status, last_health_checked_at, metadata_json
                        FROM workspace_provider_installations
                        WHERE org_id = $1 AND provider = $2
                        ORDER BY id
                        """,
                        org_id_value,
                        provider_norm,
                    )
                elif provider_norm is not None:
                    rows = await self.db_pool.fetchall(
                        """
                        SELECT id, org_id, provider, external_id, display_name, installed_by_user_id,
                               disabled, created_at, updated_at, last_health_status, last_health_checked_at, metadata_json
                        FROM workspace_provider_installations
                        WHERE org_id = $1 AND provider = $2 AND disabled = FALSE
                        ORDER BY id
                        """,
                        org_id_value,
                        provider_norm,
                    )
                elif include_disabled:
                    rows = await self.db_pool.fetchall(
                        """
                        SELECT id, org_id, provider, external_id, display_name, installed_by_user_id,
                               disabled, created_at, updated_at, last_health_status, last_health_checked_at, metadata_json
                        FROM workspace_provider_installations
                        WHERE org_id = $1
                        ORDER BY id
                        """,
                        org_id_value,
                    )
                else:
                    rows = await self.db_pool.fetchall(
                        """
                        SELECT id, org_id, provider, external_id, display_name, installed_by_user_id,
                               disabled, created_at, updated_at, last_health_status, last_health_checked_at, metadata_json
                        FROM workspace_provider_installations
                        WHERE org_id = $1 AND disabled = FALSE
                        ORDER BY id
                        """,
                        org_id_value,
                    )
            else:
                if provider_norm is not None and include_disabled:
                    rows = await self.db_pool.fetchall(
                        """
                        SELECT id, org_id, provider, external_id, display_name, installed_by_user_id,
                               disabled, created_at, updated_at, last_health_status, last_health_checked_at, metadata_json
                        FROM workspace_provider_installations
                        WHERE org_id = ? AND provider = ?
                        ORDER BY id
                        """,
                        (org_id_value, provider_norm),
                    )
                elif provider_norm is not None:
                    rows = await self.db_pool.fetchall(
                        """
                        SELECT id, org_id, provider, external_id, display_name, installed_by_user_id,
                               disabled, created_at, updated_at, last_health_status, last_health_checked_at, metadata_json
                        FROM workspace_provider_installations
                        WHERE org_id = ? AND provider = ? AND disabled = 0
                        ORDER BY id
                        """,
                        (org_id_value, provider_norm),
                    )
                elif include_disabled:
                    rows = await self.db_pool.fetchall(
                        """
                        SELECT id, org_id, provider, external_id, display_name, installed_by_user_id,
                               disabled, created_at, updated_at, last_health_status, last_health_checked_at, metadata_json
                        FROM workspace_provider_installations
                        WHERE org_id = ?
                        ORDER BY id
                        """,
                        (org_id_value,),
                    )
                else:
                    rows = await self.db_pool.fetchall(
                        """
                        SELECT id, org_id, provider, external_id, display_name, installed_by_user_id,
                               disabled, created_at, updated_at, last_health_status, last_health_checked_at, metadata_json
                        FROM workspace_provider_installations
                        WHERE org_id = ? AND disabled = 0
                        ORDER BY id
                        """,
                        (org_id_value,),
                    )
            return [self._normalize_row(self._row_to_dict(row)) for row in rows]
        except Exception as exc:
            logger.error(f"WorkspaceProviderInstallationsRepo.list_installations failed: {exc}")
            raise

    async def set_disabled(
        self,
        *,
        org_id: int,
        provider: str,
        external_id: str,
        disabled: bool,
        updated_at: datetime | None = None,
    ) -> bool:
        org_id_value = int(org_id)
        provider_norm = normalize_provider_name(provider)
        external_id_value = _normalize_external_id(external_id)
        updated_at_value = updated_at or datetime.now(timezone.utc)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                ts = self._normalize_datetime_for_postgres(updated_at_value)
                result = await self.db_pool.execute(
                    """
                    UPDATE workspace_provider_installations
                    SET disabled = $1, updated_at = $2
                    WHERE org_id = $3 AND provider = $4 AND external_id = $5
                    """,
                    bool(disabled),
                    ts,
                    org_id_value,
                    provider_norm,
                    external_id_value,
                )
                if isinstance(result, str):
                    parts = result.split()
                    if parts and parts[-1].isdigit():
                        return int(parts[-1]) > 0
                return True

            cursor = await self.db_pool.execute(
                """
                UPDATE workspace_provider_installations
                SET disabled = ?, updated_at = ?
                WHERE org_id = ? AND provider = ? AND external_id = ?
                """,
                (
                    1 if disabled else 0,
                    updated_at_value.isoformat(),
                    org_id_value,
                    provider_norm,
                    external_id_value,
                ),
            )
            return getattr(cursor, "rowcount", 0) > 0
        except Exception as exc:
            logger.error(f"WorkspaceProviderInstallationsRepo.set_disabled failed: {exc}")
            raise

    async def delete_installation(
        self,
        *,
        org_id: int,
        provider: str,
        external_id: str,
    ) -> bool:
        org_id_value = int(org_id)
        provider_norm = normalize_provider_name(provider)
        external_id_value = _normalize_external_id(external_id)
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                result = await self.db_pool.execute(
                    """
                    DELETE FROM workspace_provider_installations
                    WHERE org_id = $1 AND provider = $2 AND external_id = $3
                    """,
                    org_id_value,
                    provider_norm,
                    external_id_value,
                )
                if isinstance(result, str):
                    parts = result.split()
                    if parts and parts[-1].isdigit():
                        return int(parts[-1]) > 0
                return True

            cursor = await self.db_pool.execute(
                """
                DELETE FROM workspace_provider_installations
                WHERE org_id = ? AND provider = ? AND external_id = ?
                """,
                (org_id_value, provider_norm, external_id_value),
            )
            return getattr(cursor, "rowcount", 0) > 0
        except Exception as exc:
            logger.error(f"WorkspaceProviderInstallationsRepo.delete_installation failed: {exc}")
            raise


async def get_workspace_provider_installations_repo() -> WorkspaceProviderInstallationsRepo:
    pool = await get_db_pool()
    repo = WorkspaceProviderInstallationsRepo(pool)
    await repo.ensure_tables()
    return repo

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool

_VALID_SHARE_SCOPE_TYPES = {"team", "org"}
_VALID_ACCESS_LEVELS = {"view_chat", "view_chat_add", "full_edit"}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _load_json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not raw:
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _normalize_share_scope_type(scope_type: str | None) -> str:
    value = (scope_type or "").strip().lower()
    if value in {"organization", "orgs"}:
        return "org"
    if value in {"teams"}:
        return "team"
    if value in _VALID_SHARE_SCOPE_TYPES:
        return value
    raise ValueError(f"Invalid share_scope_type: {scope_type}")


def _normalize_access_level(access_level: str | None) -> str:
    value = (access_level or "").strip().lower()
    if value not in _VALID_ACCESS_LEVELS:
        raise ValueError(f"Invalid access_level: {access_level}")
    return value


@dataclass
class SharedWorkspaceRepo:
    """Data access for workspace sharing and share tokens in the AuthNZ DB."""

    db_pool: DatabasePool

    def _is_postgres_backend(self) -> bool:
        return bool(getattr(self.db_pool, "pool", None))

    def _ts(self) -> datetime | str:
        """Return a timestamp suitable for the current backend (native datetime for PG, ISO string for SQLite)."""
        now = datetime.now(timezone.utc)
        return now if self._is_postgres_backend() else now.isoformat()

    def _bool(self, val: bool) -> bool | int:
        """Return a boolean suitable for the current backend (native bool for PG, int for SQLite)."""
        return val if self._is_postgres_backend() else int(val)

    async def ensure_tables(self) -> None:
        required = {"shared_workspaces", "share_tokens", "share_audit_log", "sharing_config"}
        rows = await self.db_pool.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table'",
            (),
        )
        existing = {
            str(name)
            for row in rows
            if (name := self._row_to_dict(row).get("name")) and str(name) in required
        }
        missing = required - existing
        if missing:
            raise RuntimeError(
                "Sharing tables are missing. Run AuthNZ migrations. "
                f"Missing: {sorted(missing)}"
            )

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
            except Exception:
                return {}
            return {key: row[key] for key in keys}

    @staticmethod
    def _normalize_share_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        out = dict(row)
        out["allow_clone"] = _to_bool(out.get("allow_clone"))
        out["is_revoked"] = out.get("revoked_at") is not None
        return out

    @staticmethod
    def _normalize_token_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        out = dict(row)
        out["allow_clone"] = _to_bool(out.get("allow_clone"))
        out["use_count"] = int(out.get("use_count") or 0)
        out["is_password_protected"] = bool(out.get("password_hash"))
        out["is_revoked"] = out.get("revoked_at") is not None
        return out

    # ── shared_workspaces CRUD ──

    async def create_share(
        self,
        *,
        workspace_id: str,
        owner_user_id: int,
        share_scope_type: str,
        share_scope_id: int,
        access_level: str = "view_chat",
        allow_clone: bool = True,
        created_by: int,
    ) -> dict[str, Any]:
        scope_type = _normalize_share_scope_type(share_scope_type)
        level = _normalize_access_level(access_level)
        ts = self._ts()
        clone_val = self._bool(allow_clone)

        await self.db_pool.execute(
            """
            INSERT INTO shared_workspaces (
                workspace_id, owner_user_id, share_scope_type, share_scope_id,
                access_level, allow_clone, created_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workspace_id, owner_user_id, scope_type, share_scope_id,
                level, clone_val, created_by, ts, ts,
            ),
        )
        row = await self.db_pool.fetchone(
            """
            SELECT id FROM shared_workspaces
            WHERE workspace_id = ? AND owner_user_id = ?
              AND share_scope_type = ? AND share_scope_id = ?
            ORDER BY id DESC LIMIT 1
            """,
            (workspace_id, owner_user_id, scope_type, share_scope_id),
        )
        if not row:
            return {}
        created = await self.get_share(int(row["id"]))
        return created or {}

    async def get_share(self, share_id: int) -> dict[str, Any] | None:
        row = await self.db_pool.fetchone(
            """
            SELECT id, workspace_id, owner_user_id, share_scope_type, share_scope_id,
                   access_level, allow_clone, created_by, created_at, updated_at, revoked_at
            FROM shared_workspaces
            WHERE id = ?
            """,
            (int(share_id),),
        )
        return self._normalize_share_row(self._row_to_dict(row) if row else None)

    async def list_shares_for_workspace(
        self,
        workspace_id: str,
        owner_user_id: int,
        *,
        include_revoked: bool = False,
    ) -> list[dict[str, Any]]:
        if include_revoked:
            rows = await self.db_pool.fetchall(
                """
                SELECT id, workspace_id, owner_user_id, share_scope_type, share_scope_id,
                       access_level, allow_clone, created_by, created_at, updated_at, revoked_at
                FROM shared_workspaces
                WHERE workspace_id = ? AND owner_user_id = ?
                ORDER BY created_at DESC
                """,
                (workspace_id, owner_user_id),
            )
        else:
            rows = await self.db_pool.fetchall(
                """
                SELECT id, workspace_id, owner_user_id, share_scope_type, share_scope_id,
                       access_level, allow_clone, created_by, created_at, updated_at, revoked_at
                FROM shared_workspaces
                WHERE workspace_id = ? AND owner_user_id = ? AND revoked_at IS NULL
                ORDER BY created_at DESC
                """,
                (workspace_id, owner_user_id),
            )
        return [self._normalize_share_row(self._row_to_dict(r)) or {} for r in rows]

    async def list_shares_for_scope(
        self,
        share_scope_type: str,
        share_scope_id: int,
    ) -> list[dict[str, Any]]:
        scope_type = _normalize_share_scope_type(share_scope_type)
        rows = await self.db_pool.fetchall(
            """
            SELECT id, workspace_id, owner_user_id, share_scope_type, share_scope_id,
                   access_level, allow_clone, created_by, created_at, updated_at, revoked_at
            FROM shared_workspaces
            WHERE share_scope_type = ? AND share_scope_id = ? AND revoked_at IS NULL
            ORDER BY created_at DESC
            """,
            (scope_type, share_scope_id),
        )
        return [self._normalize_share_row(self._row_to_dict(r)) or {} for r in rows]

    async def update_share(
        self,
        share_id: int,
        *,
        access_level: str | None = None,
        allow_clone: bool | None = None,
    ) -> dict[str, Any] | None:
        existing = await self.get_share(share_id)
        if not existing:
            return None

        next_level = _normalize_access_level(access_level) if access_level is not None else existing["access_level"]
        next_clone = allow_clone if allow_clone is not None else existing["allow_clone"]
        ts = self._ts()
        clone_val = self._bool(next_clone)

        await self.db_pool.execute(
            """
            UPDATE shared_workspaces
            SET access_level = ?, allow_clone = ?, updated_at = ?
            WHERE id = ?
            """,
            (next_level, clone_val, ts, share_id),
        )
        return await self.get_share(share_id)

    async def revoke_share(self, share_id: int) -> bool:
        ts = self._ts()
        await self.db_pool.execute(
            "UPDATE shared_workspaces SET revoked_at = ?, updated_at = ? WHERE id = ? AND revoked_at IS NULL",
            (ts, ts, share_id),
        )
        row = await self.db_pool.fetchone(
            "SELECT revoked_at FROM shared_workspaces WHERE id = ?",
            (share_id,),
        )
        return self._row_to_dict(row).get("revoked_at") is not None if row else False

    async def revoke_shares_for_workspace(self, workspace_id: str, owner_user_id: int) -> int:
        ts = self._ts()
        await self.db_pool.execute(
            """
            UPDATE shared_workspaces
            SET revoked_at = ?, updated_at = ?
            WHERE workspace_id = ? AND owner_user_id = ? AND revoked_at IS NULL
            """,
            (ts, ts, workspace_id, owner_user_id),
        )
        remaining = await self.db_pool.fetchone(
            "SELECT COUNT(*) AS cnt FROM shared_workspaces WHERE workspace_id = ? AND owner_user_id = ? AND revoked_at IS NOT NULL",
            (workspace_id, owner_user_id),
        )
        return int(self._row_to_dict(remaining).get("cnt", 0)) if remaining else 0

    # ── share_tokens CRUD ──

    async def create_token(
        self,
        *,
        token_hash: str,
        token_prefix: str,
        resource_type: str,
        resource_id: str,
        owner_user_id: int,
        access_level: str = "view_chat",
        allow_clone: bool = True,
        password_hash: str | None = None,
        max_uses: int | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        level = _normalize_access_level(access_level)
        ts = self._ts()
        clone_val = self._bool(allow_clone)

        await self.db_pool.execute(
            """
            INSERT INTO share_tokens (
                token_hash, token_prefix, resource_type, resource_id, owner_user_id,
                access_level, allow_clone, password_hash, max_uses, expires_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                token_hash, token_prefix, resource_type, resource_id, owner_user_id,
                level, clone_val, password_hash, max_uses, expires_at, ts,
            ),
        )
        row = await self.db_pool.fetchone(
            "SELECT id FROM share_tokens WHERE token_hash = ?",
            (token_hash,),
        )
        if not row:
            return {}
        created = await self.get_token(int(row["id"]))
        return created or {}

    async def get_token(self, token_id: int) -> dict[str, Any] | None:
        row = await self.db_pool.fetchone(
            """
            SELECT id, token_hash, token_prefix, resource_type, resource_id, owner_user_id,
                   access_level, allow_clone, password_hash, max_uses, use_count,
                   expires_at, created_at, revoked_at
            FROM share_tokens
            WHERE id = ?
            """,
            (int(token_id),),
        )
        return self._normalize_token_row(self._row_to_dict(row) if row else None)

    async def find_tokens_by_prefix(self, prefix: str) -> list[dict[str, Any]]:
        rows = await self.db_pool.fetchall(
            """
            SELECT id, token_hash, token_prefix, resource_type, resource_id, owner_user_id,
                   access_level, allow_clone, password_hash, max_uses, use_count,
                   expires_at, created_at, revoked_at
            FROM share_tokens
            WHERE token_prefix = ? AND revoked_at IS NULL
            """,
            (prefix,),
        )
        return [self._normalize_token_row(self._row_to_dict(r)) or {} for r in rows]

    async def list_tokens_for_user(self, owner_user_id: int) -> list[dict[str, Any]]:
        rows = await self.db_pool.fetchall(
            """
            SELECT id, token_hash, token_prefix, resource_type, resource_id, owner_user_id,
                   access_level, allow_clone, password_hash, max_uses, use_count,
                   expires_at, created_at, revoked_at
            FROM share_tokens
            WHERE owner_user_id = ?
            ORDER BY created_at DESC
            """,
            (owner_user_id,),
        )
        return [self._normalize_token_row(self._row_to_dict(r)) or {} for r in rows]

    async def increment_token_use_count(self, token_id: int) -> None:
        await self.db_pool.execute(
            "UPDATE share_tokens SET use_count = use_count + 1 WHERE id = ?",
            (token_id,),
        )

    async def claim_token_use(self, token_id: int) -> bool:
        if not hasattr(self.db_pool, "transaction"):
            await self.db_pool.execute(
                """
                UPDATE share_tokens
                SET use_count = use_count + 1
                WHERE id = ?
                  AND revoked_at IS NULL
                  AND (max_uses IS NULL OR use_count < max_uses)
                """,
                (token_id,),
            )
            row = await self.db_pool.fetchone("SELECT changes() AS change_count", ())
            return int(self._row_to_dict(row).get("change_count", 0)) > 0 if row else False

        async with self.db_pool.transaction() as conn:
            if self._is_postgres_backend():
                row = await conn.fetchrow(
                    """
                    UPDATE share_tokens
                    SET use_count = use_count + 1
                    WHERE id = $1
                      AND revoked_at IS NULL
                      AND (max_uses IS NULL OR use_count < max_uses)
                    RETURNING id
                    """,
                    token_id,
                )
                return row is not None

            await conn.execute(
                """
                UPDATE share_tokens
                SET use_count = use_count + 1
                WHERE id = ?
                  AND revoked_at IS NULL
                  AND (max_uses IS NULL OR use_count < max_uses)
                """,
                (token_id,),
            )
            cursor = await conn.execute("SELECT changes() AS change_count")
            row = await cursor.fetchone()
            return int(self._row_to_dict(row).get("change_count", 0)) > 0 if row else False

    async def release_token_use(self, token_id: int) -> None:
        await self.db_pool.execute(
            """
            UPDATE share_tokens
            SET use_count = CASE
                WHEN use_count > 0 THEN use_count - 1
                ELSE 0
            END
            WHERE id = ?
            """,
            (token_id,),
        )

    async def revoke_token(self, token_id: int) -> bool:
        ts = self._ts()
        await self.db_pool.execute(
            "UPDATE share_tokens SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
            (ts, token_id),
        )
        row = await self.db_pool.fetchone(
            "SELECT revoked_at FROM share_tokens WHERE id = ?",
            (token_id,),
        )
        return self._row_to_dict(row).get("revoked_at") is not None if row else False

    async def revoke_tokens_for_resource(self, resource_type: str, resource_id: str, owner_user_id: int) -> int:
        ts = self._ts()
        await self.db_pool.execute(
            """
            UPDATE share_tokens SET revoked_at = ?
            WHERE resource_type = ? AND resource_id = ? AND owner_user_id = ? AND revoked_at IS NULL
            """,
            (ts, resource_type, resource_id, owner_user_id),
        )
        return 0  # count not critical

    # ── share_audit_log ──

    async def log_audit_event(
        self,
        *,
        event_type: str,
        resource_type: str,
        resource_id: str,
        owner_user_id: int,
        actor_user_id: int | None = None,
        share_id: int | None = None,
        token_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        ts = self._ts()
        meta_json = json.dumps(metadata or {})
        await self.db_pool.execute(
            """
            INSERT INTO share_audit_log (
                event_type, actor_user_id, resource_type, resource_id, owner_user_id,
                share_id, token_id, metadata_json, ip_address, user_agent, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_type, actor_user_id, resource_type, resource_id, owner_user_id,
                share_id, token_id, meta_json, ip_address, user_agent, ts,
            ),
        )

    async def list_audit_events(
        self,
        *,
        owner_user_id: int | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        rows = await self.db_pool.fetchall(
            """
            SELECT id, event_type, actor_user_id, resource_type, resource_id,
                   owner_user_id, share_id, token_id, metadata_json,
                   ip_address, user_agent, created_at
            FROM share_audit_log
            WHERE (? IS NULL OR owner_user_id = ?)
              AND (? IS NULL OR resource_type = ?)
              AND (? IS NULL OR resource_id = ?)
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (
                owner_user_id,
                owner_user_id,
                resource_type,
                resource_type,
                resource_id,
                resource_id,
                limit,
                offset,
            ),
        )
        result = []
        for r in rows:
            d = self._row_to_dict(r)
            d["metadata"] = _load_json_dict(d.get("metadata_json"))
            result.append(d)
        return result

    async def count_audit_events(
        self,
        *,
        owner_user_id: int | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
    ) -> int:
        row = await self.db_pool.fetchone(
            """
            SELECT COUNT(*) AS cnt
            FROM share_audit_log
            WHERE (? IS NULL OR owner_user_id = ?)
              AND (? IS NULL OR resource_type = ?)
              AND (? IS NULL OR resource_id = ?)
            """,
            (
                owner_user_id,
                owner_user_id,
                resource_type,
                resource_type,
                resource_id,
                resource_id,
            ),
        )
        return int(self._row_to_dict(row).get("cnt", 0)) if row else 0

    async def list_legacy_audit_events_for_migration(
        self,
        *,
        after_id: int = 0,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        rows = await self.db_pool.fetchall(
            """
            SELECT id, event_type, actor_user_id, resource_type, resource_id,
                   owner_user_id, share_id, token_id, metadata_json,
                   ip_address, user_agent, created_at
            FROM share_audit_log
            WHERE id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (int(after_id), int(limit)),
        )
        result = []
        for row in rows:
            data = self._row_to_dict(row)
            data["metadata"] = _load_json_dict(data.get("metadata_json"))
            result.append(data)
        return result

    # ── sharing_config ──

    async def get_config(
        self,
        scope_type: str = "global",
        scope_id: int | None = None,
    ) -> dict[str, str]:
        rows = await self.db_pool.fetchall(
            """
            SELECT config_key, config_value
            FROM sharing_config
            WHERE scope_type = ? AND (
                (scope_id IS NULL AND ? IS NULL) OR scope_id = ?
            )
            """,
            (scope_type, scope_id, scope_id),
        )
        return {self._row_to_dict(r)["config_key"]: self._row_to_dict(r)["config_value"] for r in rows}

    async def set_config(
        self,
        config_key: str,
        config_value: str,
        *,
        scope_type: str = "global",
        scope_id: int | None = None,
        updated_by: int | None = None,
    ) -> None:
        ts = self._ts()
        # Upsert: try update first, then insert
        await self.db_pool.execute(
            """
            INSERT INTO sharing_config (scope_type, scope_id, config_key, config_value, updated_by, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope_type, scope_id, config_key)
            DO UPDATE SET config_value = excluded.config_value, updated_by = excluded.updated_by, updated_at = excluded.updated_at
            """,
            (scope_type, scope_id, config_key, config_value, updated_by, ts),
        )

    # ── Admin queries ──

    async def list_all_shares(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        include_revoked: bool = False,
    ) -> list[dict[str, Any]]:
        if include_revoked:
            rows = await self.db_pool.fetchall(
                """
                SELECT id, workspace_id, owner_user_id, share_scope_type, share_scope_id,
                       access_level, allow_clone, created_by, created_at, updated_at, revoked_at
                FROM shared_workspaces
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
        else:
            rows = await self.db_pool.fetchall(
                """
                SELECT id, workspace_id, owner_user_id, share_scope_type, share_scope_id,
                       access_level, allow_clone, created_by, created_at, updated_at, revoked_at
                FROM shared_workspaces
                WHERE revoked_at IS NULL
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
        return [self._normalize_share_row(self._row_to_dict(r)) or {} for r in rows]

    async def count_all_shares(self, *, include_revoked: bool = False) -> int:
        if include_revoked:
            row = await self.db_pool.fetchone(
                "SELECT COUNT(*) AS cnt FROM shared_workspaces",
                (),
            )
        else:
            row = await self.db_pool.fetchone(
                "SELECT COUNT(*) AS cnt FROM shared_workspaces WHERE revoked_at IS NULL",
                (),
            )
        return int(self._row_to_dict(row).get("cnt", 0)) if row else 0

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool

_VALID_CREATION_SOURCES = {"prompt", "template", "existing_workspace"}
_VALID_ACTOR_TYPES = {"owner", "internal_collaborator", "external_collaborator"}
_VALID_PROMOTION_STATUSES = {"pending", "approved", "rejected", "promoted", "stale"}


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


def _load_json_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return list(raw)
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        return list(parsed) if isinstance(parsed, list) else []
    return []


def _normalize_creation_source(creation_source: str | None) -> str:
    value = (creation_source or "").strip().lower()
    if value not in _VALID_CREATION_SOURCES:
        raise ValueError(f"Invalid creation_source: {creation_source}")
    return value


def _normalize_actor_type(actor_type: str | None) -> str:
    value = (actor_type or "").strip().lower()
    if value not in _VALID_ACTOR_TYPES:
        raise ValueError(f"Invalid actor_type: {actor_type}")
    return value


def _is_blank(value: str | None) -> bool:
    return value is None or not str(value).strip()


def _normalize_promotion_status(status: str | None) -> str:
    value = (status or "").strip().lower()
    if value not in _VALID_PROMOTION_STATUSES:
        raise ValueError(f"Invalid status: {status}")
    return value


@dataclass
class PrototypeWorkspacesRepo:
    """Data access for prototype workspace metadata in the AuthNZ DB."""

    db_pool: DatabasePool

    def _is_postgres_backend(self) -> bool:
        return bool(getattr(self.db_pool, "pool", None))

    def _ts(self) -> datetime | str:
        now = datetime.now(timezone.utc)
        return now if self._is_postgres_backend() else now.isoformat()

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        if row is None:
            return {}
        if isinstance(row, dict):
            return dict(row)
        if hasattr(row, "keys"):
            try:
                keys = row.keys()
                return {key: row[key] for key in keys}
            except Exception:
                return {}
        try:
            return dict(row)
        except Exception:
            return {}

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex}"

    @staticmethod
    def _normalize_workspace_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        out = dict(row)
        if out.get("owner_user_id") is not None:
            out["owner_user_id"] = int(out["owner_user_id"])
        out["preview_policy"] = _load_json_dict(out.get("preview_policy_json"))
        out["share_policy"] = _load_json_dict(out.get("share_policy_json"))
        out["runtime_policy"] = _load_json_dict(out.get("runtime_policy_json"))
        out["designated_promoter_ids"] = _load_json_list(out.get("designated_promoter_ids_json"))
        out["is_archived"] = out.get("archived_at") is not None
        return out

    @staticmethod
    def _normalize_snapshot_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        out = dict(row)
        if out.get("author_user_id") is not None:
            out["author_user_id"] = int(out["author_user_id"])
        out["snapshot_id"] = out.get("id")
        out["diff_summary"] = _load_json_dict(out.get("diff_summary_json"))
        out["preview_health"] = _load_json_dict(out.get("preview_health_json"))
        return out

    @staticmethod
    def _normalize_shared_actor_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        out = dict(row)
        if out.get("share_link_id") is not None:
            out["share_link_id"] = int(out["share_link_id"])
        out["quota_policy"] = _load_json_dict(out.get("quota_policy_json"))
        out["is_revoked"] = out.get("revoked_at") is not None
        return out

    @staticmethod
    def _normalize_session_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        out = dict(row)
        if out.get("actor_user_id") is not None:
            out["actor_user_id"] = int(out["actor_user_id"])
        if out.get("share_link_id") is not None:
            out["share_link_id"] = int(out["share_link_id"])
        out["is_revoked"] = out.get("revoked_at") is not None
        return out

    @staticmethod
    def _normalize_promotion_request_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        out = dict(row)
        if out.get("requested_by_user_id") is not None:
            out["requested_by_user_id"] = int(out["requested_by_user_id"])
        if out.get("reviewed_by_user_id") is not None:
            out["reviewed_by_user_id"] = int(out["reviewed_by_user_id"])
        return out

    async def ensure_tables(self) -> None:
        required = {
            "prototype_workspaces",
            "prototype_snapshots",
            "prototype_sessions",
            "prototype_shared_actors",
            "prototype_promotion_requests",
        }
        if self._is_postgres_backend():
            table_query = """
            SELECT table_name AS name
            FROM information_schema.tables
            WHERE table_schema = current_schema()
              AND table_type = 'BASE TABLE'
            """
        else:
            table_query = "SELECT name FROM sqlite_master WHERE type='table'"
        rows = await self.db_pool.fetchall(
            table_query,
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
                "Prototype workspace tables are missing. Run AuthNZ migrations. "
                f"Missing: {sorted(missing)}"
            )

    async def create_workspace(
        self,
        *,
        owner_user_id: int,
        title: str,
        creation_source: str,
        description: str | None = None,
        preview_policy: dict[str, Any] | None = None,
        share_policy: dict[str, Any] | None = None,
        runtime_policy: dict[str, Any] | None = None,
        designated_promoter_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        creation_source_value = _normalize_creation_source(creation_source)
        workspace_id = self._new_id("pws")
        ts = self._ts()
        await self.db_pool.execute(
            """
            INSERT INTO prototype_workspaces (
                id, owner_user_id, title, description, creation_source,
                preview_policy_json, share_policy_json, runtime_policy_json,
                designated_promoter_ids_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workspace_id,
                int(owner_user_id),
                title,
                description,
                creation_source_value,
                json.dumps(preview_policy or {}),
                json.dumps(share_policy or {}),
                json.dumps(runtime_policy or {}),
                json.dumps(designated_promoter_ids or []),
                ts,
                ts,
            ),
        )
        created = await self.get_workspace(workspace_id)
        return created or {}

    async def get_workspace(self, prototype_workspace_id: str) -> dict[str, Any] | None:
        row = await self.db_pool.fetchone(
            """
            SELECT id, owner_user_id, title, description, creation_source,
                   canonical_snapshot_id, last_known_good_snapshot_id,
                   canonical_preview_status, publish_validation_status,
                   preview_policy_json, share_policy_json, runtime_policy_json,
                   designated_promoter_ids_json, created_at, updated_at, archived_at
            FROM prototype_workspaces
            WHERE id = ?
            """,
            (prototype_workspace_id,),
        )
        return self._normalize_workspace_row(self._row_to_dict(row) if row else None)

    async def update_workspace_state(
        self,
        prototype_workspace_id: str,
        *,
        canonical_snapshot_id: str | None = None,
        last_known_good_snapshot_id: str | None = None,
        canonical_preview_status: str | None = None,
        publish_validation_status: str | None = None,
    ) -> dict[str, Any] | None:
        ts = self._ts()
        await self.db_pool.execute(
            """
            UPDATE prototype_workspaces
            SET canonical_snapshot_id = COALESCE(?, canonical_snapshot_id),
                last_known_good_snapshot_id = COALESCE(?, last_known_good_snapshot_id),
                canonical_preview_status = COALESCE(?, canonical_preview_status),
                publish_validation_status = COALESCE(?, publish_validation_status),
                updated_at = ?
            WHERE id = ?
            """,
            (
                canonical_snapshot_id,
                last_known_good_snapshot_id,
                canonical_preview_status,
                publish_validation_status,
                ts,
                prototype_workspace_id,
            ),
        )
        return await self.get_workspace(prototype_workspace_id)

    async def create_snapshot(
        self,
        *,
        prototype_workspace_id: str,
        snapshot_id: str,
        created_by_user_id: int | None = None,
        created_by_shared_actor_id: str | None = None,
        parent_snapshot_id: str | None = None,
        created_from_session_id: str | None = None,
        storage_ref: str | None = None,
        diff_summary: dict[str, Any] | None = None,
        prompt_summary: str | None = None,
        preview_health: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if (created_by_user_id is None) == (created_by_shared_actor_id is None):
            raise ValueError("exactly one actor identity must be set")

        await self.db_pool.execute(
            """
            INSERT INTO prototype_snapshots (
                id, prototype_workspace_id, parent_snapshot_id, created_from_session_id,
                author_user_id, author_shared_actor_id, storage_ref, diff_summary_json,
                prompt_summary, preview_health_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                prototype_workspace_id,
                parent_snapshot_id,
                created_from_session_id,
                int(created_by_user_id) if created_by_user_id is not None else None,
                created_by_shared_actor_id,
                storage_ref,
                json.dumps(diff_summary or {}),
                prompt_summary,
                json.dumps(preview_health or {}),
            ),
        )
        created = await self.get_snapshot(snapshot_id)
        return created or {}

    async def get_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        row = await self.db_pool.fetchone(
            """
            SELECT id, prototype_workspace_id, parent_snapshot_id, created_from_session_id,
                   author_user_id, author_shared_actor_id, storage_ref, diff_summary_json,
                   prompt_summary, preview_health_json, created_at
            FROM prototype_snapshots
            WHERE id = ?
            """,
            (snapshot_id,),
        )
        return self._normalize_snapshot_row(self._row_to_dict(row) if row else None)

    async def list_snapshots_for_workspace(self, prototype_workspace_id: str) -> list[dict[str, Any]]:
        rows = await self.db_pool.fetchall(
            """
            SELECT id, prototype_workspace_id, parent_snapshot_id, created_from_session_id,
                   author_user_id, author_shared_actor_id, storage_ref, diff_summary_json,
                   prompt_summary, preview_health_json, created_at
            FROM prototype_snapshots
            WHERE prototype_workspace_id = ?
            ORDER BY created_at DESC
            """,
            (prototype_workspace_id,),
        )
        return [self._normalize_snapshot_row(self._row_to_dict(r)) or {} for r in rows]

    async def create_shared_actor(
        self,
        *,
        prototype_workspace_id: str,
        share_link_id: int,
        display_name: str,
        session_binding_id: str | None = None,
        runtime_policy_profile: str,
        quota_policy: dict[str, Any] | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        actor_id = self._new_id("psa")
        ts = self._ts()
        await self.db_pool.execute(
            """
            INSERT INTO prototype_shared_actors (
                id, prototype_workspace_id, share_link_id, display_name,
                session_binding_id, runtime_policy_profile, quota_policy_json,
                last_activity_at, expires_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                actor_id,
                prototype_workspace_id,
                int(share_link_id),
                display_name,
                session_binding_id,
                runtime_policy_profile,
                json.dumps(quota_policy or {}),
                ts,
                expires_at,
                ts,
                ts,
            ),
        )
        created = await self.get_shared_actor(actor_id)
        return created or {}

    async def touch_shared_actor(self, shared_actor_id: str) -> dict[str, Any] | None:
        ts = self._ts()
        await self.db_pool.execute(
            """
            UPDATE prototype_shared_actors
            SET last_activity_at = ?, updated_at = ?
            WHERE id = ? AND revoked_at IS NULL
            """,
            (ts, ts, shared_actor_id),
        )
        return await self.get_shared_actor(shared_actor_id)

    async def rotate_shared_actor_binding(
        self,
        shared_actor_id: str,
        *,
        new_session_binding_id: str,
    ) -> dict[str, Any] | None:
        ts = self._ts()
        await self.db_pool.execute(
            """
            UPDATE prototype_shared_actors
            SET session_binding_id = ?, last_activity_at = ?, updated_at = ?
            WHERE id = ? AND revoked_at IS NULL
            """,
            (new_session_binding_id, ts, ts, shared_actor_id),
        )
        return await self.get_shared_actor(shared_actor_id)

    async def get_shared_actor(self, shared_actor_id: str) -> dict[str, Any] | None:
        row = await self.db_pool.fetchone(
            """
            SELECT id, prototype_workspace_id, share_link_id, display_name,
                   session_binding_id, runtime_policy_profile, quota_policy_json,
                   last_activity_at, expires_at, revoked_at, created_at, updated_at
            FROM prototype_shared_actors
            WHERE id = ?
            """,
            (shared_actor_id,),
        )
        return self._normalize_shared_actor_row(self._row_to_dict(row) if row else None)

    async def create_session(
        self,
        *,
        prototype_workspace_id: str,
        base_snapshot_id: str,
        actor_type: str,
        actor_user_id: int | None = None,
        actor_shared_actor_id: str | None = None,
        share_link_id: int | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        actor_type_value = _normalize_actor_type(actor_type)
        if _is_blank(base_snapshot_id):
            raise ValueError("base_snapshot_id is required")

        workspace = await self.get_workspace(prototype_workspace_id)
        if workspace is None:
            raise ValueError("prototype_workspace_id does not exist")

        if actor_type_value == "external_collaborator":
            if actor_shared_actor_id is None or actor_user_id is not None:
                raise ValueError(
                    "external_collaborator requires actor_shared_actor_id and forbids actor_user_id"
                )
        else:
            if actor_user_id is None or actor_shared_actor_id is not None:
                raise ValueError(
                    "owner/internal_collaborator requires actor_user_id and forbids actor_shared_actor_id"
                )
            user_row = await self.db_pool.fetchone(
                "SELECT id FROM users WHERE id = ?",
                (int(actor_user_id),),
            )
            if not user_row:
                raise ValueError("actor_user_id must reference an existing user")

        snapshot_row = await self.db_pool.fetchone(
            """
            SELECT id
            FROM prototype_snapshots
            WHERE id = ? AND prototype_workspace_id = ?
            """,
            (base_snapshot_id, prototype_workspace_id),
        )
        if not snapshot_row:
            raise ValueError("base_snapshot_id must reference a snapshot in the same workspace")

        if actor_shared_actor_id is not None:
            shared_actor_row = await self.db_pool.fetchone(
                """
                SELECT id
                FROM prototype_shared_actors
                WHERE id = ? AND prototype_workspace_id = ? AND revoked_at IS NULL
                """,
                (actor_shared_actor_id, prototype_workspace_id),
            )
            if not shared_actor_row:
                raise ValueError(
                    "actor_shared_actor_id must reference an active shared actor in the same workspace"
                )

        if actor_type_value == "owner" and int(actor_user_id) != int(workspace["owner_user_id"]):
            raise ValueError("owner actor must match workspace owner")

        session_id = self._new_id("pss")
        ts = self._ts()
        await self.db_pool.execute(
            """
            INSERT INTO prototype_sessions (
                id, prototype_workspace_id, base_snapshot_id, actor_user_id,
                actor_shared_actor_id, actor_type, share_link_id, runtime_status,
                preview_status, last_activity_at, expires_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                prototype_workspace_id,
                base_snapshot_id,
                int(actor_user_id) if actor_user_id is not None else None,
                actor_shared_actor_id,
                actor_type_value,
                int(share_link_id) if share_link_id is not None else None,
                "pending",
                "uninitialized",
                ts,
                expires_at,
                ts,
                ts,
            ),
        )
        created = await self.get_session(session_id)
        return created or {}

    async def get_session(self, prototype_session_id: str) -> dict[str, Any] | None:
        row = await self.db_pool.fetchone(
            """
            SELECT id, prototype_workspace_id, base_snapshot_id, actor_user_id,
                   actor_shared_actor_id, actor_type, share_link_id, acp_session_id,
                   sandbox_session_id, sandbox_run_id, runtime_status, preview_handle,
                   preview_status, last_saved_snapshot_id, last_activity_at, expires_at,
                   revoked_at, created_at, updated_at
            FROM prototype_sessions
            WHERE id = ?
            """,
            (prototype_session_id,),
        )
        return self._normalize_session_row(self._row_to_dict(row) if row else None)

    async def list_sessions_for_workspace(
        self,
        prototype_workspace_id: str,
        *,
        include_revoked: bool = False,
    ) -> list[dict[str, Any]]:
        if include_revoked:
            rows = await self.db_pool.fetchall(
                """
                SELECT id, prototype_workspace_id, base_snapshot_id, actor_user_id,
                       actor_shared_actor_id, actor_type, share_link_id, acp_session_id,
                       sandbox_session_id, sandbox_run_id, runtime_status, preview_handle,
                       preview_status, last_saved_snapshot_id, last_activity_at, expires_at,
                       revoked_at, created_at, updated_at
                FROM prototype_sessions
                WHERE prototype_workspace_id = ?
                ORDER BY updated_at DESC, created_at DESC
                """,
                (prototype_workspace_id,),
            )
        else:
            rows = await self.db_pool.fetchall(
                """
                SELECT id, prototype_workspace_id, base_snapshot_id, actor_user_id,
                       actor_shared_actor_id, actor_type, share_link_id, acp_session_id,
                       sandbox_session_id, sandbox_run_id, runtime_status, preview_handle,
                       preview_status, last_saved_snapshot_id, last_activity_at, expires_at,
                       revoked_at, created_at, updated_at
                FROM prototype_sessions
                WHERE prototype_workspace_id = ? AND revoked_at IS NULL
                ORDER BY updated_at DESC, created_at DESC
                """,
                (prototype_workspace_id,),
            )
        return [self._normalize_session_row(self._row_to_dict(r)) or {} for r in rows]

    async def find_active_session(
        self,
        *,
        prototype_workspace_id: str,
        base_snapshot_id: str,
        actor_type: str,
        actor_user_id: int | None = None,
        actor_shared_actor_id: str | None = None,
    ) -> dict[str, Any] | None:
        actor_type_value = _normalize_actor_type(actor_type)
        actor_user_param = int(actor_user_id) if actor_user_id is not None else None
        row = await self.db_pool.fetchone(
            """
            SELECT id, prototype_workspace_id, base_snapshot_id, actor_user_id,
                   actor_shared_actor_id, actor_type, share_link_id, acp_session_id,
                   sandbox_session_id, sandbox_run_id, runtime_status, preview_handle,
                   preview_status, last_saved_snapshot_id, last_activity_at, expires_at,
                   revoked_at, created_at, updated_at
            FROM prototype_sessions
            WHERE prototype_workspace_id = ?
              AND base_snapshot_id = ?
              AND actor_type = ?
              AND (? IS NULL OR actor_user_id = ?)
              AND (? IS NULL OR actor_shared_actor_id = ?)
              AND revoked_at IS NULL
              AND (runtime_status IS NULL OR LOWER(runtime_status) NOT IN ('failed', 'revoked', 'closed'))
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1
            """,
            (
                prototype_workspace_id,
                base_snapshot_id,
                actor_type_value,
                actor_user_param,
                actor_user_param,
                actor_shared_actor_id,
                actor_shared_actor_id,
                self._ts(),
            ),
        )
        return self._normalize_session_row(self._row_to_dict(row) if row else None)

    async def update_session_state(
        self,
        prototype_session_id: str,
        *,
        acp_session_id: str | None = None,
        sandbox_session_id: str | None = None,
        sandbox_run_id: str | None = None,
        runtime_status: str | None = None,
        preview_handle: str | None = None,
        preview_status: str | None = None,
        last_saved_snapshot_id: str | None = None,
        last_activity_at: str | datetime | None = None,
        revoked_at: str | datetime | None = None,
    ) -> dict[str, Any] | None:
        existing = await self.get_session(prototype_session_id)
        if not existing:
            return None

        ts = self._ts()
        await self.db_pool.execute(
            """
            UPDATE prototype_sessions
            SET acp_session_id = ?,
                sandbox_session_id = ?,
                sandbox_run_id = ?,
                runtime_status = ?,
                preview_handle = ?,
                preview_status = ?,
                last_saved_snapshot_id = ?,
                last_activity_at = ?,
                revoked_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                acp_session_id if acp_session_id is not None else existing.get("acp_session_id"),
                sandbox_session_id if sandbox_session_id is not None else existing.get("sandbox_session_id"),
                sandbox_run_id if sandbox_run_id is not None else existing.get("sandbox_run_id"),
                runtime_status if runtime_status is not None else existing.get("runtime_status"),
                preview_handle if preview_handle is not None else existing.get("preview_handle"),
                preview_status if preview_status is not None else existing.get("preview_status"),
                last_saved_snapshot_id
                if last_saved_snapshot_id is not None
                else existing.get("last_saved_snapshot_id"),
                last_activity_at if last_activity_at is not None else ts,
                revoked_at if revoked_at is not None else existing.get("revoked_at"),
                ts,
                prototype_session_id,
            ),
        )
        return await self.get_session(prototype_session_id)

    async def create_promotion_request(
        self,
        *,
        prototype_workspace_id: str,
        prototype_session_id: str,
        candidate_snapshot_id: str,
        requested_by_user_id: int | None = None,
        requested_by_shared_actor_id: str | None = None,
        status: str = "pending",
    ) -> dict[str, Any]:
        if (requested_by_user_id is None) == (requested_by_shared_actor_id is None):
            raise ValueError("exactly one actor identity must be set")

        workspace = await self.get_workspace(prototype_workspace_id)
        if workspace is None:
            raise ValueError("prototype_workspace_id does not exist")

        session_row = await self.db_pool.fetchone(
            """
            SELECT id
            FROM prototype_sessions
            WHERE id = ? AND prototype_workspace_id = ?
            """,
            (prototype_session_id, prototype_workspace_id),
        )
        if not session_row:
            raise ValueError("prototype_session_id must reference a session in the same workspace")

        snapshot_row = await self.db_pool.fetchone(
            """
            SELECT id
            FROM prototype_snapshots
            WHERE id = ? AND prototype_workspace_id = ?
            """,
            (candidate_snapshot_id, prototype_workspace_id),
        )
        if not snapshot_row:
            raise ValueError("candidate_snapshot_id must reference a snapshot in the same workspace")

        if requested_by_shared_actor_id is not None:
            shared_actor_row = await self.db_pool.fetchone(
                """
                SELECT id
                FROM prototype_shared_actors
                WHERE id = ? AND prototype_workspace_id = ? AND revoked_at IS NULL
                """,
                (requested_by_shared_actor_id, prototype_workspace_id),
            )
            if not shared_actor_row:
                raise ValueError(
                    "requested_by_shared_actor_id must reference an active shared actor in the same workspace"
                )

        request_id = self._new_id("ppr")
        status_value = _normalize_promotion_status(status)
        ts = self._ts()
        await self.db_pool.execute(
            """
            INSERT INTO prototype_promotion_requests (
                id, prototype_workspace_id, prototype_session_id, candidate_snapshot_id,
                requested_by_user_id, requested_by_shared_actor_id, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                prototype_workspace_id,
                prototype_session_id,
                candidate_snapshot_id,
                int(requested_by_user_id) if requested_by_user_id is not None else None,
                requested_by_shared_actor_id,
                status_value,
                ts,
                ts,
            ),
        )
        created = await self.get_promotion_request(request_id)
        return created or {}

    async def get_promotion_request(self, promotion_request_id: str) -> dict[str, Any] | None:
        row = await self.db_pool.fetchone(
            """
            SELECT id, prototype_workspace_id, prototype_session_id, candidate_snapshot_id,
                   requested_by_user_id, requested_by_shared_actor_id, status,
                   reviewed_by_user_id, review_notes, created_at, updated_at
            FROM prototype_promotion_requests
            WHERE id = ?
            """,
            (promotion_request_id,),
        )
        return self._normalize_promotion_request_row(self._row_to_dict(row) if row else None)

    async def update_promotion_request(
        self,
        promotion_request_id: str,
        *,
        status: str | None = None,
        reviewed_by_user_id: int | None = None,
        review_notes: str | None = None,
    ) -> dict[str, Any] | None:
        existing = await self.get_promotion_request(promotion_request_id)
        if not existing:
            return None

        status_value = (
            _normalize_promotion_status(status)
            if status is not None
            else str(existing.get("status") or "pending")
        )
        ts = self._ts()
        await self.db_pool.execute(
            """
            UPDATE prototype_promotion_requests
            SET status = ?,
                reviewed_by_user_id = ?,
                review_notes = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                status_value,
                int(reviewed_by_user_id) if reviewed_by_user_id is not None else existing.get("reviewed_by_user_id"),
                review_notes if review_notes is not None else existing.get("review_notes"),
                ts,
                promotion_request_id,
            ),
        )
        return await self.get_promotion_request(promotion_request_id)

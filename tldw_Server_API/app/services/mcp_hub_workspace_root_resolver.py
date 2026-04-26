from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Sandbox.service import SandboxService


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


def _normalize_workspace_root(path_value: str) -> str:
    return str(Path(path_value).expanduser().resolve(strict=False))


class McpHubWorkspaceRootResolver:
    """Resolve a trusted workspace root for MCP Hub path-scope enforcement."""

    def __init__(
        self,
        sandbox_service: SandboxService | Any | None = None,
        repo: Any | None = None,
    ) -> None:
        self._sandbox_service = sandbox_service or SandboxService()
        self._repo = repo

    async def resolve_for_context(
        self,
        *,
        session_id: str | None,
        user_id: str | None,
        workspace_id: str | None,
        workspace_trust_source: str | None = None,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> dict[str, Any]:
        session_key = _first_nonempty(session_id)
        user_key = _first_nonempty(user_id)
        workspace_key = _first_nonempty(workspace_id)
        trust_source = _first_nonempty(workspace_trust_source)
        scope_type = _first_nonempty(owner_scope_type)

        result = {
            "workspace_root": None,
            "workspace_id": workspace_key,
            "source": None,
            "reason": None,
        }

        if trust_source == "shared_registry":
            if not workspace_key or not self._repo or not scope_type:
                result["reason"] = "workspace_root_unavailable"
                return result
            same_scope_matches = await self._repo.list_shared_workspace_entries(
                owner_scope_type=scope_type,
                owner_scope_id=owner_scope_id,
                workspace_id=workspace_key,
            )
            same_scope_roots = sorted(
                {
                    _normalize_workspace_root(str(row.get("absolute_root") or ""))
                    for row in same_scope_matches
                    if str(row.get("absolute_root") or "").strip() and bool(row.get("is_active", True))
                }
            )
            if len(same_scope_roots) > 1:
                result["source"] = "shared_registry"
                result["reason"] = "workspace_root_ambiguous"
                return result
            if same_scope_roots:
                result["workspace_root"] = same_scope_roots[0]
                result["source"] = "shared_registry"
                return result

            parent_matches = await self._repo.list_shared_workspace_entries(
                owner_scope_type="global",
                owner_scope_id=None,
                workspace_id=workspace_key,
            )
            parent_roots = sorted(
                {
                    _normalize_workspace_root(str(row.get("absolute_root") or ""))
                    for row in parent_matches
                    if str(row.get("absolute_root") or "").strip() and bool(row.get("is_active", True))
                }
            )
            if len(parent_roots) > 1:
                result["source"] = "shared_registry"
                result["reason"] = "workspace_root_ambiguous"
                return result
            if parent_roots:
                result["workspace_root"] = parent_roots[0]
                result["source"] = "shared_registry"
                return result

            result["source"] = "shared_registry"
            result["reason"] = "workspace_root_unavailable"
            return result

        if session_key:
            if not user_key:
                result["reason"] = "workspace_root_unavailable"
                return result

            workspace_root = self._sandbox_service.get_session_workspace_path_for_user(
                session_key,
                user_key,
            )
            if workspace_root:
                result["workspace_root"] = _normalize_workspace_root(str(workspace_root))
                result["source"] = "sandbox_session"
                return result

        if not user_key or not workspace_key:
            result["reason"] = "workspace_root_unavailable"
            return result

        candidates = self._sandbox_service.list_workspace_paths_for_user_workspace(
            user_id=user_key,
            workspace_id=workspace_key,
        )
        normalized_roots = sorted(
            {
                _normalize_workspace_root(str(path_value))
                for path_value in list(candidates or [])
                if str(path_value or "").strip()
            }
        )
        if not normalized_roots:
            # Fallback: try ACP workspace registry if workspace_key is numeric
            acp_root = self._resolve_acp_workspace(user_key, workspace_key)
            if acp_root:
                result["workspace_root"] = _normalize_workspace_root(acp_root)
                result["source"] = "acp_workspace"
                return result
            result["reason"] = "workspace_root_unavailable"
            return result
        if len(normalized_roots) > 1:
            result["source"] = "sandbox_workspace_lookup"
            result["reason"] = "workspace_root_ambiguous"
            return result

        result["workspace_root"] = normalized_roots[0]
        result["source"] = "sandbox_workspace_lookup"
        return result

    @staticmethod
    def _resolve_acp_workspace(user_key: str, workspace_key: str) -> str | None:
        """Try to resolve a workspace root from the ACP orchestration DB."""
        try:
            uid = int(user_key)
            wid = int(workspace_key)
        except (ValueError, TypeError):
            return None
        try:
            from tldw_Server_API.app.core.Agent_Orchestration.orchestration_service import (
                get_orchestration_db,
            )
            db = get_orchestration_db(uid)
            ws = db.get_workspace(wid)
            if ws and ws.root_path:
                return ws.root_path
        except Exception as exc:
            logger.debug("ACP workspace fallback resolution failed for user={} ws={}: {}", user_key, workspace_key, exc)
        return None

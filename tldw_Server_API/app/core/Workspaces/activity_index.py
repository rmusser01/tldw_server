"""Workspace contained-resource index and recent activity read model."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.Workspaces.membership_models import WORKSPACE_MEMBERSHIP_RESOURCE_TYPES
from tldw_Server_API.app.core.Workspaces.membership_service import (
    WorkspaceMembershipService,
    WorkspaceMembershipServiceError,
)
from tldw_Server_API.app.core.Workspaces.runtime_bindings import runtime_binding_response_payload


_GROUP_LIMIT_MAX = 25
_ACTIVITY_LIMIT_MAX = 100
_RUNTIME_WARNING_STATUSES = frozenset(
    {
        "missing",
        "blocked",
        "unavailable",
        "detached",
        "conflict",
        "runtime-missing",
        "unsupported",
    }
)
_OWNER_SURFACES = {
    "workspace_note": {"label": "Workspace notes", "href": "#/research-workspace"},
    "media": {"label": "Media library", "href": "#/media"},
    "workspace_source": {"label": "Research Workspace sources", "href": "#/research-workspace"},
    "workspace_artifact": {"label": "Research Workspace studio", "href": "#/research-workspace"},
    "chat": {"label": "Chat", "href": "#/chat"},
    "prompt": {"label": "Prompts", "href": "#/prompts"},
    "workflow": {"label": "Workflows", "href": "#/workflows"},
    "watchlist": {"label": "Watchlists", "href": "#/watchlists"},
    "acp_session": {"label": "ACP Playground", "href": "#/agent-playground"},
    "sandbox_session": {"label": "Sandbox", "href": "#/sandbox"},
}


class WorkspaceActivityIndexService:
    """Compose a read-only Workspace resource index without duplicating owner UIs."""

    def __init__(self, chacha_db: Any) -> None:
        self.chacha_db = chacha_db
        self.memberships = WorkspaceMembershipService(chacha_db)

    def build_index(
        self,
        workspace_id: str,
        *,
        user_id: str | None = None,
        media_db: Any | None = None,
        prompts_db: Any | None = None,
        workflows_db: Any | None = None,
        watchlists_db: Any | None = None,
        request_metadata: Mapping[str, Any] | None = None,
        group_limit: int = 5,
        activity_limit: int = 25,
    ) -> dict[str, Any]:
        """Return the Workspace index/activity contract for API responses."""
        workspace = self.chacha_db.get_workspace(workspace_id, include_deleted=True)
        if workspace is None:
            raise WorkspaceMembershipServiceError(
                "workspace_not_found",
                f"Workspace '{workspace_id}' was not found.",
                status_code=404,
            )

        normalized_group_limit = _bounded_limit(group_limit, default=5, max_value=_GROUP_LIMIT_MAX)
        normalized_activity_limit = _bounded_limit(
            activity_limit,
            default=25,
            max_value=_ACTIVITY_LIMIT_MAX,
        )
        if _workspace_truthy(workspace.get("deleted")):
            membership_summary: dict[str, Any] = {"total": 0, "by_resource_type": {}, "by_role": {}}
            resource_groups: list[dict[str, Any]] = []
            runtime_summary = {"total": 0, "by_kind": {}, "by_status": {}, "bindings": []}
        else:
            membership_summary = self.memberships.workspace_membership_summary(workspace_id)
            resource_groups = self._resource_groups(
                workspace_id,
                membership_summary=membership_summary,
                group_limit=normalized_group_limit,
                user_id=user_id,
                media_db=media_db,
                prompts_db=prompts_db,
                workflows_db=workflows_db,
                watchlists_db=watchlists_db,
                request_metadata=request_metadata,
            )
            runtime_summary = self._runtime_summary(workspace_id)
        recent_activity = self._recent_activity(workspace_id, normalized_activity_limit)
        warnings = self._warnings(
            workspace=workspace,
            resource_groups=resource_groups,
            runtime_bindings=runtime_summary["bindings"],
        )

        return {
            "workspace_id": workspace_id,
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "workspace": dict(workspace),
            "membership_summary": membership_summary,
            "resource_groups": resource_groups,
            "runtime_summary": runtime_summary,
            "warnings": warnings,
            "recent_activity": recent_activity,
            "partial_errors": [],
        }

    def _resource_groups(
        self,
        workspace_id: str,
        *,
        membership_summary: Mapping[str, Any],
        group_limit: int,
        user_id: str | None,
        media_db: Any | None,
        prompts_db: Any | None,
        workflows_db: Any | None,
        watchlists_db: Any | None,
        request_metadata: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        counts = dict(membership_summary.get("by_resource_type") or {})
        groups: list[dict[str, Any]] = []
        for resource_type in sorted(counts):
            if resource_type not in WORKSPACE_MEMBERSHIP_RESOURCE_TYPES:
                continue
            page = self.memberships.list_workspace_memberships(
                workspace_id,
                resource_type=resource_type,
                resolve=True,
                limit=group_limit,
                user_id=user_id,
                media_db=media_db,
                prompts_db=prompts_db,
                workflows_db=workflows_db,
                watchlists_db=watchlists_db,
                request_metadata=request_metadata,
            )
            groups.append(
                {
                    "resource_type": resource_type,
                    "count": int(counts.get(resource_type) or 0),
                    "owner_surface": _owner_surface(resource_type),
                    "items": page.get("items") or [],
                    "next_cursor": page.get("next_cursor"),
                }
            )
        return groups

    def _runtime_summary(self, workspace_id: str) -> dict[str, Any]:
        try:
            rows = self.chacha_db.list_workspace_runtime_bindings(workspace_id, limit=50)
        except AttributeError:
            rows = []
        bindings = [runtime_binding_response_payload(row) for row in rows]
        by_kind = Counter(str(item.get("binding_kind") or "") for item in bindings)
        by_status = Counter(str(item.get("status") or "") for item in bindings)
        by_kind.pop("", None)
        by_status.pop("", None)
        return {
            "total": len(bindings),
            "by_kind": dict(sorted(by_kind.items())),
            "by_status": dict(sorted(by_status.items())),
            "bindings": bindings,
        }

    def _recent_activity(self, workspace_id: str, activity_limit: int) -> list[dict[str, Any]]:
        try:
            return self.chacha_db.list_workspace_activity_events(workspace_id, limit=activity_limit)
        except AttributeError:
            return []

    def _warnings(
        self,
        *,
        workspace: Mapping[str, Any],
        resource_groups: list[Mapping[str, Any]],
        runtime_bindings: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        warnings: list[dict[str, Any]] = []
        if _workspace_truthy(workspace.get("deleted")):
            warnings.append(
                {
                    "severity": "error",
                    "reason_code": "workspace_deleted",
                    "message": "Workspace is deleted and only recovery-safe inspection is available.",
                    "action_href": "#/workspaces",
                }
            )
        elif _workspace_truthy(workspace.get("archived")):
            warnings.append(
                {
                    "severity": "warning",
                    "reason_code": "workspace_archived",
                    "message": "Workspace is archived; write actions are disabled until it is restored.",
                    "action_href": "#/workspaces",
                }
            )

        for group in resource_groups:
            for item in group.get("items") or []:
                summary = item.get("summary") or {}
                state = str(summary.get("state") or "available")
                if state == "available":
                    continue
                reason = "resource_unresolved" if state == "unresolved" else f"resource_{state}"
                warnings.append(
                    {
                        "severity": "warning",
                        "reason_code": reason,
                        "message": "A workspace resource needs attention in its owning surface.",
                        "resource_type": item.get("resource_type"),
                        "resource_id": item.get("resource_id"),
                        "action_href": (summary.get("href") or _owner_surface(str(item.get("resource_type") or ""))["href"]),
                    }
                )

        for binding in runtime_bindings:
            status = str(binding.get("status") or "").strip().lower()
            if status not in _RUNTIME_WARNING_STATUSES:
                continue
            reason = "runtime_binding_missing" if status in {"missing", "runtime-missing"} else (
                f"runtime_binding_{status.replace('-', '_')}"
            )
            warnings.append(
                {
                    "severity": "warning",
                    "reason_code": reason,
                    "message": "A workspace runtime binding needs attention before runtime-backed actions are reliable.",
                    "resource_type": "workspace_runtime_binding",
                    "resource_id": binding.get("binding_id"),
                    "action_href": "#/workspaces",
                }
            )
        return warnings


def _owner_surface(resource_type: str) -> dict[str, str]:
    return dict(
        _OWNER_SURFACES.get(
            resource_type,
            {"label": "Workspace resources", "href": "#/workspaces"},
        )
    )


def _bounded_limit(value: int, *, default: int, max_value: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(normalized, max_value))


def _workspace_truthy(value: Any) -> bool:
    return value in (True, 1, "1", "true", "True")

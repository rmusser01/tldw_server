"""Workspace active-context eligibility checks."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from tldw_Server_API.app.core.Workspaces.membership_models import WORKSPACE_MEMBERSHIP_RESOURCE_TYPES


WorkspaceEligibilityOperation = Literal[
    "browse",
    "search",
    "open",
    "edit",
    "stage",
    "rag_ground",
    "prompt_use",
    "tool_use",
    "agent_manipulate",
    "acp_run",
    "sandbox_operation",
    "workflow_launch",
    "watchlist_run",
]
WorkspaceEligibilityOperationCategory = Literal["visibility", "active_context"]
WorkspaceEligibilityRuntimeState = Literal["not_required", "ready", "missing"]
WorkspaceEligibilityPermissionState = Literal["granted", "denied"]
WorkspaceEligibilityReasonCode = Literal[
    "allowed",
    "visibility_allowed",
    "no_active_workspace",
    "workspace_not_found",
    "workspace_archived",
    "unsupported_resource_type",
    "resource_not_linked",
    "cross_workspace_resource",
    "missing_runtime",
    "permission_denied",
]

WORKSPACE_VISIBILITY_OPERATIONS = frozenset({"browse", "search", "open", "edit"})
WORKSPACE_ACTIVE_CONTEXT_OPERATIONS = frozenset(
    {
        "stage",
        "rag_ground",
        "prompt_use",
        "tool_use",
        "agent_manipulate",
        "acp_run",
        "sandbox_operation",
        "workflow_launch",
        "watchlist_run",
    }
)
WORKSPACE_RUNTIME_REQUIRED_OPERATIONS = frozenset(
    {
        "tool_use",
        "agent_manipulate",
        "acp_run",
        "sandbox_operation",
        "workflow_launch",
        "watchlist_run",
    }
)
WORKSPACE_ELIGIBILITY_OPERATIONS = WORKSPACE_VISIBILITY_OPERATIONS | WORKSPACE_ACTIVE_CONTEXT_OPERATIONS
_UNSET = object()


@dataclass(frozen=True)
class WorkspaceEligibilityRequest:
    """Input for active-context eligibility checks."""

    operation: WorkspaceEligibilityOperation
    resource_type: str
    resource_id: str
    runtime_state: WorkspaceEligibilityRuntimeState
    permission_state: WorkspaceEligibilityPermissionState
    active_workspace_id: str | None = None


@dataclass(frozen=True)
class WorkspaceEligibilityRecoveryAction:
    """Client-facing recovery action for an eligibility denial."""

    action: str
    label: str
    href: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "action": self.action,
            "label": self.label,
            "href": self.href,
        }


@dataclass(frozen=True)
class WorkspaceEligibilityMembership:
    """Compact membership reference returned with eligibility decisions."""

    workspace_id: str
    resource_type: str
    resource_id: str
    role: str = "member"
    label: str | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "WorkspaceEligibilityMembership":
        return cls(
            workspace_id=str(row.get("workspace_id") or ""),
            resource_type=str(row.get("resource_type") or ""),
            resource_id=str(row.get("resource_id") or ""),
            role=str(row.get("role") or "member"),
            label=str(row.get("label")) if row.get("label") is not None else None,
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "workspace_id": self.workspace_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "role": self.role,
            "label": self.label,
        }


@dataclass(frozen=True)
class WorkspaceEligibilityResult:
    """Stable eligibility decision returned to API and integration callers."""

    allowed: bool
    reason_code: WorkspaceEligibilityReasonCode
    message: str
    operation: WorkspaceEligibilityOperation
    operation_category: WorkspaceEligibilityOperationCategory
    active_workspace_id: str | None
    resource_type: str
    resource_id: str
    global_visibility_preserved: bool = True
    recovery_actions: list[WorkspaceEligibilityRecoveryAction] = field(default_factory=list)
    membership: WorkspaceEligibilityMembership | None = None
    resource_workspace_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason_code": self.reason_code,
            "message": self.message,
            "operation": self.operation,
            "operation_category": self.operation_category,
            "active_workspace_id": self.active_workspace_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "global_visibility_preserved": self.global_visibility_preserved,
            "recovery_actions": [action.to_dict() for action in self.recovery_actions],
            "membership": self.membership.to_dict() if self.membership is not None else None,
            "resource_workspace_ids": list(self.resource_workspace_ids),
        }


class WorkspaceEligibilityService:
    """Resolve whether a resource can be used by an active Workspace operation."""

    def __init__(self, chacha_db: Any) -> None:
        self.chacha_db = chacha_db

    def check(self, request: WorkspaceEligibilityRequest) -> WorkspaceEligibilityResult:
        operation_category = self._operation_category(request.operation)
        if request.permission_state == "denied":
            return self._deny(
                request,
                operation_category,
                "permission_denied",
                "The current user is not allowed to access this resource for the requested operation.",
                self._actions("permission_denied"),
            )

        if operation_category == "visibility":
            return WorkspaceEligibilityResult(
                allowed=True,
                reason_code="visibility_allowed",
                message="Global resource visibility is preserved for this operation.",
                operation=request.operation,
                operation_category=operation_category,
                active_workspace_id=request.active_workspace_id,
                resource_type=request.resource_type,
                resource_id=request.resource_id,
                global_visibility_preserved=True,
            )

        active_workspace_id = self._non_empty_or_none(request.active_workspace_id)
        if active_workspace_id is None:
            return self._deny(
                request,
                operation_category,
                "no_active_workspace",
                "Select or create a workspace before using resources in an active workspace context.",
                self._actions("no_active_workspace"),
                active_workspace_id=None,
            )

        workspace = self.chacha_db.get_workspace(active_workspace_id)
        if workspace is None:
            return self._deny(
                request,
                operation_category,
                "workspace_not_found",
                "The active workspace was not found.",
                self._actions("workspace_not_found"),
            )
        if self._workspace_is_archived(workspace):
            return self._deny(
                request,
                operation_category,
                "workspace_archived",
                "Archived workspaces cannot run active workspace operations.",
                self._actions("workspace_archived"),
            )

        if request.operation in WORKSPACE_RUNTIME_REQUIRED_OPERATIONS and request.runtime_state != "ready":
            return self._deny(
                request,
                operation_category,
                "missing_runtime",
                "The active workspace is missing the runtime required for this operation.",
                self._actions("missing_runtime"),
            )

        resource_type = request.resource_type.strip()
        if resource_type not in WORKSPACE_MEMBERSHIP_RESOURCE_TYPES:
            return self._deny(
                request,
                operation_category,
                "unsupported_resource_type",
                "This resource type does not yet have a workspace membership adapter.",
                self._actions("unsupported_resource_type"),
                resource_type=resource_type,
            )
        resource_id = self._canonical_resource_id(resource_type, request.resource_id)

        membership_row = self.chacha_db.get_workspace_resource_membership(
            active_workspace_id,
            resource_type,
            resource_id,
            include_deleted=False,
        )
        if membership_row is not None:
            membership = WorkspaceEligibilityMembership.from_row(membership_row)
            return WorkspaceEligibilityResult(
                allowed=True,
                reason_code="allowed",
                message="The resource is linked to the active workspace.",
                operation=request.operation,
                operation_category=operation_category,
                active_workspace_id=active_workspace_id,
                resource_type=resource_type,
                resource_id=resource_id,
                global_visibility_preserved=True,
                membership=membership,
                resource_workspace_ids=[membership.workspace_id],
            )

        resource_memberships = self._list_resource_workspace_memberships(resource_type, resource_id)
        resource_workspace_ids = [
            str(row.get("workspace_id"))
            for row in resource_memberships
            if row.get("workspace_id") is not None and str(row.get("workspace_id")) != active_workspace_id
        ]
        if resource_workspace_ids:
            return self._deny(
                request,
                operation_category,
                "cross_workspace_resource",
                "This resource is linked to a different workspace; copy it or switch workspaces before using it here.",
                self._actions("cross_workspace_resource"),
                active_workspace_id=active_workspace_id,
                resource_type=resource_type,
                resource_id=resource_id,
                resource_workspace_ids=resource_workspace_ids,
            )

        return self._deny(
            request,
            operation_category,
            "resource_not_linked",
            "Link this resource to the active workspace before using it in an active workspace context.",
            self._actions("resource_not_linked"),
            active_workspace_id=active_workspace_id,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_workspace_ids=[],
        )

    def _list_resource_workspace_memberships(self, resource_type: str, resource_id: str) -> list[Mapping[str, Any]]:
        rows = self.chacha_db.list_resource_workspace_memberships(
            resource_type,
            resource_id,
            include_deleted=False,
            limit=100,
            cursor=None,
        )
        return [row for row in rows if isinstance(row, Mapping)]

    @staticmethod
    def _operation_category(operation: WorkspaceEligibilityOperation) -> WorkspaceEligibilityOperationCategory:
        if operation in WORKSPACE_VISIBILITY_OPERATIONS:
            return "visibility"
        if operation in WORKSPACE_ACTIVE_CONTEXT_OPERATIONS:
            return "active_context"
        raise ValueError(f"Unsupported workspace eligibility operation: {operation}")

    @staticmethod
    def _canonical_resource_id(resource_type: str, resource_id: str) -> str:
        normalized = str(resource_id).strip()
        if resource_type in {"media", "workspace_note"}:
            try:
                parsed = int(normalized)
            except (TypeError, ValueError):
                return normalized
            if parsed >= 0:
                return str(parsed)
        return normalized

    @staticmethod
    def _workspace_is_archived(workspace: Mapping[str, Any]) -> bool:
        return workspace.get("archived") in (True, 1, "1", "true", "True")

    @staticmethod
    def _non_empty_or_none(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def _deny(
        self,
        request: WorkspaceEligibilityRequest,
        operation_category: WorkspaceEligibilityOperationCategory,
        reason_code: WorkspaceEligibilityReasonCode,
        message: str,
        recovery_actions: list[WorkspaceEligibilityRecoveryAction],
        *,
        active_workspace_id: str | None | object = _UNSET,
        resource_type: str | None = None,
        resource_id: str | None = None,
        resource_workspace_ids: list[str] | None = None,
    ) -> WorkspaceEligibilityResult:
        resolved_workspace_id = (
            request.active_workspace_id
            if active_workspace_id is _UNSET
            else cast(str | None, active_workspace_id)
        )
        return WorkspaceEligibilityResult(
            allowed=False,
            reason_code=reason_code,
            message=message,
            operation=request.operation,
            operation_category=operation_category,
            active_workspace_id=resolved_workspace_id,
            resource_type=resource_type if resource_type is not None else request.resource_type,
            resource_id=resource_id if resource_id is not None else request.resource_id,
            global_visibility_preserved=True,
            recovery_actions=recovery_actions,
            resource_workspace_ids=list(resource_workspace_ids or []),
        )

    @staticmethod
    def _actions(reason_code: WorkspaceEligibilityReasonCode) -> list[WorkspaceEligibilityRecoveryAction]:
        actions: dict[str, list[WorkspaceEligibilityRecoveryAction]] = {
            "no_active_workspace": [
                WorkspaceEligibilityRecoveryAction("select_workspace", "Select an active workspace"),
                WorkspaceEligibilityRecoveryAction("create_workspace", "Create a workspace"),
            ],
            "workspace_not_found": [
                WorkspaceEligibilityRecoveryAction("select_workspace", "Select an available workspace"),
            ],
            "workspace_archived": [
                WorkspaceEligibilityRecoveryAction("select_workspace", "Select an active workspace"),
                WorkspaceEligibilityRecoveryAction("unarchive_workspace", "Unarchive this workspace"),
            ],
            "unsupported_resource_type": [
                WorkspaceEligibilityRecoveryAction("wait_for_adapter", "Use a supported workspace resource type"),
            ],
            "resource_not_linked": [
                WorkspaceEligibilityRecoveryAction(
                    "link_to_active_workspace",
                    "Link this resource to the active workspace",
                ),
            ],
            "cross_workspace_resource": [
                WorkspaceEligibilityRecoveryAction(
                    "copy_to_active_workspace",
                    "Copy or link the resource to the active workspace",
                ),
                WorkspaceEligibilityRecoveryAction(
                    "switch_workspace",
                    "Switch to a workspace that already contains this resource",
                ),
            ],
            "missing_runtime": [
                WorkspaceEligibilityRecoveryAction("configure_workspace_runtime", "Configure the workspace runtime"),
            ],
            "permission_denied": [
                WorkspaceEligibilityRecoveryAction("request_access", "Request access or select a permitted workspace"),
            ],
            "allowed": [],
            "visibility_allowed": [],
        }
        return list(actions[reason_code])

from __future__ import annotations

from typing import Any, Literal, TypedDict

WorkspaceProfile = Literal["research", "project"]
WorkspaceKind = Literal["research_workspace", "project_workspace"]
WorkspaceAttentionState = Literal[
    "ready",
    "setup_pending",
    "working",
    "needs_attention",
    "blocked",
    "archived",
]
ProjectRootBackend = Literal["host_local", "sandbox_volume"]
ProjectRootState = Literal[
    "not_configured",
    "provisioning",
    "attached",
    "unavailable",
    "missing",
    "detached",
    "failed",
    "cleanup_pending",
    "archived",
]
ResolutionStatus = Literal["complete", "partial", "failed"]

_WORKING_INVENTORY_STATES = frozenset(
    {
        "queued",
        "running",
        "scanning",
        "processing",
        "retrying",
        "indexing",
    }
)
_SAFE_INVENTORY_STATES = frozenset(
    {
        "not_started",
        "current",
        "partial",
        "stale",
        "disabled",
    }
)
_PROJECT_ROOT_BACKENDS = frozenset({"host_local", "sandbox_volume"})
_READY_SANDBOX_MOUNT_STATES = frozenset({"ready", "mounted"})


class AllowedAction(TypedDict):
    allowed: bool
    reason_code: str | None


def normalize_workspace_profile(value: Any) -> WorkspaceProfile:
    return "project" if str(value or "").strip().lower() == "project" else "research"


def workspace_kind_for_profile(profile: WorkspaceProfile) -> WorkspaceKind:
    return "project_workspace" if profile == "project" else "research_workspace"


def normalize_project_root_state(value: Any) -> ProjectRootState:
    normalized = str(value or "").strip().lower()
    if normalized in {
        "not_configured",
        "provisioning",
        "attached",
        "unavailable",
        "missing",
        "detached",
        "failed",
        "cleanup_pending",
        "archived",
    }:
        return normalized  # type: ignore[return-value]
    return "failed"


def workspace_attention_state(
    *,
    workspace_profile: Any,
    project_root_state: Any,
    inventory_state: Any,
    service_errors: list[str] | None = None,
    archived: bool = False,
) -> WorkspaceAttentionState:
    """Project the manager-facing attention state from safe Workspace Core inputs."""
    raw_root_state = _normalized_string(project_root_state)
    if archived or raw_root_state == "archived":
        return "archived"

    service_attention = _attention_from_service_errors(service_errors)
    if service_attention is not None:
        return service_attention

    profile = _normalized_string(workspace_profile)
    if profile == "research":
        return "ready"
    if profile != "project":
        return "needs_attention"

    root_state = normalize_project_root_state(project_root_state)
    if root_state == "not_configured":
        return "setup_pending"
    if root_state == "provisioning":
        return "working"
    if root_state in {"missing", "detached"}:
        return "needs_attention"
    if root_state in {"failed", "unavailable"}:
        return "blocked"
    if root_state == "cleanup_pending":
        return "needs_attention"
    if root_state == "attached":
        return _attention_from_inventory_state(inventory_state)
    return "needs_attention"


def project_sandbox_volume_projection(
    sandbox_state: Any,
    *,
    usable_mount: bool,
) -> dict[str, Any]:
    """Project Sandbox durable-volume state into Workspace manager root fields."""
    normalized_state = _normalized_string(sandbox_state)
    mount_usable = bool(usable_mount)
    if normalized_state == "provisioning":
        root_state = "provisioning"
        mount_state = "not_ready"
        attention_state = "working"
    elif normalized_state == "ready":
        root_state = "attached"
        mount_state = "ready" if mount_usable else "not_ready"
        attention_state = (
            "ready"
            if mount_usable
            else workspace_attention_state(
                workspace_profile="project",
                project_root_state=root_state,
                inventory_state="not_started",
                service_errors=["sandbox_mount_not_ready"],
            )
        )
    elif normalized_state == "not_configured":
        root_state = "unavailable"
        mount_state = "not_configured"
        attention_state = "blocked"
    elif normalized_state == "unavailable":
        root_state = "unavailable"
        mount_state = "unavailable"
        attention_state = "blocked"
    elif normalized_state == "failed":
        root_state = "failed"
        mount_state = "failed"
        attention_state = "blocked"
    elif normalized_state == "cleanup_pending":
        root_state = "cleanup_pending"
        mount_state = "unavailable"
        attention_state = "needs_attention"
    else:
        root_state = "failed"
        mount_state = "unknown"
        attention_state = "blocked"

    return {
        "root_state": root_state,
        "mount_state": mount_state,
        "file_inventory": {
            "available": normalized_state == "ready" and mount_usable,
        },
        "attention_state": attention_state,
    }


def workspace_file_inventory_available(
    *,
    project_root_state: Any,
    root_id: Any,
    backend: Any,
    sandbox_mount_state: Any,
    inventory_state: Any,
) -> bool:
    """Return whether file inventory actions can use the projected root."""
    if _normalized_string(inventory_state) == "disabled":
        return False
    if normalize_project_root_state(project_root_state) != "attached":
        return False
    if not root_id:
        return False
    normalized_backend = _normalized_string(backend)
    if normalized_backend not in _PROJECT_ROOT_BACKENDS:
        return False
    if normalized_backend == "sandbox_volume":
        return _normalized_string(sandbox_mount_state) in _READY_SANDBOX_MOUNT_STATES
    return True


def _attention_from_service_errors(service_errors: list[str] | None) -> WorkspaceAttentionState | None:
    if not service_errors:
        return None
    normalized_errors = [
        _normalized_string(error)
        for error in service_errors
        if _normalized_string(error)
    ]
    if not normalized_errors:
        return None
    if any(
        "blocked" in error
        or "failed" in error
        or "unavailable" in error
        for error in normalized_errors
    ):
        return "blocked"
    return "needs_attention"


def _attention_from_inventory_state(inventory_state: Any) -> WorkspaceAttentionState:
    normalized = _normalized_string(inventory_state) or "not_started"
    if normalized in _WORKING_INVENTORY_STATES:
        return "working"
    if normalized == "failed":
        return "needs_attention"
    if normalized in _SAFE_INVENTORY_STATES:
        return "ready"
    return "needs_attention"


def _normalized_string(value: Any) -> str:
    return str(value or "").strip().lower()


def allowed_action() -> AllowedAction:
    return {"allowed": True, "reason_code": None}


def fail_closed_action(reason_code: str) -> AllowedAction:
    return {"allowed": False, "reason_code": reason_code}

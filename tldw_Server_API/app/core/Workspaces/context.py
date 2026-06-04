"""Read-only Workspace Core context resolver."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePath, PureWindowsPath
from typing import Any

from tldw_Server_API.app.core.Workspaces.models import (
    allowed_action,
    fail_closed_action,
    normalize_project_root_state,
    normalize_workspace_profile,
    workspace_attention_state,
    workspace_file_inventory_available,
    workspace_kind_for_profile,
)


_PROJECT_ROOT_BACKENDS = frozenset({"host_local", "sandbox_volume"})
_PARTIAL_REASON = "dependency_resolution_partial"
_PARTIAL_MESSAGE = "Workspace dependency resolution failed."


def build_workspace_core_context(
    *,
    workspace: Mapping[str, Any] | None,
    primary_root: Mapping[str, Any] | None,
    source_summary: Mapping[str, Any] | None,
    service_capabilities: Mapping[str, Any] | None,
    partial_errors: list[Mapping[str, Any]] | None,
    active_operations: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve the read-only Workspace Core context for status surfaces."""
    workspace_dict = dict(workspace or {})
    workspace_id = workspace_dict.get("id")
    profile = normalize_workspace_profile(workspace_dict.get("workspace_profile"))
    resolution = _resolution(workspace_id=workspace_id, partial_errors=partial_errors)
    project_root = _project_root_projection(
        profile=profile,
        primary_root=primary_root,
    )
    workspace_services = _workspace_services(service_capabilities)
    base_allowed_actions = _base_allowed_actions(service_capabilities)
    attention_state = _workspace_attention_state(
        profile=profile,
        workspace=workspace_dict,
        project_root=project_root,
        resolution=resolution,
    )
    allowed_actions = _allowed_actions(
        profile=profile,
        project_root=project_root,
        resolution=resolution,
        workspace_services=workspace_services,
        base_allowed_actions=base_allowed_actions,
    )

    return {
        "workspace_id": workspace_id,
        "workspace_profile": profile,
        "workspace_kind": workspace_kind_for_profile(profile),
        "access_level": workspace_dict.get("access_level", "owner"),
        "resolution": resolution,
        "project_root": project_root,
        "attention_state": attention_state,
        "source_summary": dict(source_summary or {}),
        "workspace_services": workspace_services,
        "allowed_actions": allowed_actions,
        "active_operations": [dict(operation) for operation in (active_operations or [])],
    }


def _resolution(
    *,
    workspace_id: Any,
    partial_errors: list[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    errors = _partial_errors(partial_errors)
    if not workspace_id:
        return {
            "status": "failed",
            "partial_errors": errors
            or [
                {
                    "scope": "workspace",
                    "code": "workspace_identity_unresolved",
                    "message": "Workspace identity could not be resolved.",
                }
            ],
        }
    if errors:
        return {"status": "partial", "partial_errors": errors}
    return {"status": "complete", "partial_errors": []}


def _project_root_projection(
    *,
    profile: str,
    primary_root: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if profile != "project" or not isinstance(primary_root, Mapping):
        return _empty_project_root()

    root = dict(primary_root)
    state = normalize_project_root_state(root.get("root_state", root.get("state")))
    backend = _normalized_backend(root.get("backend"))
    file_inventory = _file_inventory_projection(root)
    return {
        "state": state,
        "root_id": root.get("root_id") or root.get("id"),
        "backend": backend,
        "display_name": root.get("display_name") or root.get("name"),
        "path_hint": _path_hint(root),
        "git_state": root.get("git_state"),
        "file_inventory_state": file_inventory["state"] or root.get("file_inventory_state"),
        "indexing_state": root.get("indexing_state"),
        "sandbox_mount_state": root.get("sandbox_mount_state"),
        "mcp_trust_state": root.get("mcp_trust_state"),
        "file_inventory": file_inventory,
    }


def _empty_project_root() -> dict[str, Any]:
    return {
        "state": "not_configured",
        "root_id": None,
        "backend": None,
        "display_name": None,
        "path_hint": None,
        "git_state": None,
        "file_inventory_state": None,
        "indexing_state": None,
        "sandbox_mount_state": None,
        "mcp_trust_state": None,
        "file_inventory": _empty_file_inventory(),
    }


def _normalized_backend(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _PROJECT_ROOT_BACKENDS else None


def _path_hint(root: Mapping[str, Any]) -> str | None:
    explicit_hint = root.get("path_hint")
    if explicit_hint:
        return _redacted_path_hint(explicit_hint)
    if root.get("sandbox_volume_id"):
        return str(root["sandbox_volume_id"])
    if root.get("display_name"):
        return _redacted_path_hint(root["display_name"])
    absolute_root = root.get("absolute_root")
    if absolute_root:
        return _redacted_path_hint(absolute_root)
    return None


def _redacted_path_hint(value: Any) -> str:
    raw_value = str(value)
    windows_path = PureWindowsPath(raw_value)
    if raw_value.startswith(("/", "~", "\\\\")) or windows_path.is_absolute():
        if windows_path.is_absolute() or raw_value.startswith("\\\\"):
            return windows_path.name or "project_root"
        return PurePath(raw_value).name or "project_root"
    if "/" in raw_value or "\\" in raw_value:
        return windows_path.name or PurePath(raw_value).name or "project_root"
    return raw_value


def _empty_file_inventory() -> dict[str, Any]:
    return {
        "state": "not_started",
        "indexed_file_count": 0,
        "total_file_count": 0,
        "updated_at": None,
        "available": False,
    }


def _file_inventory_projection(root: Mapping[str, Any]) -> dict[str, Any]:
    raw_inventory = root.get("file_inventory")
    inventory = dict(raw_inventory) if isinstance(raw_inventory, Mapping) else {}
    state = str(
        inventory.get("state")
        or root.get("file_inventory_state")
        or "not_started"
    ).strip().lower()
    return {
        "state": state or "not_started",
        "indexed_file_count": _non_negative_int(inventory.get("indexed_file_count"), default=0),
        "total_file_count": _non_negative_int(inventory.get("total_file_count"), default=0),
        "updated_at": str(inventory["updated_at"]) if inventory.get("updated_at") else None,
        "available": workspace_file_inventory_available(
            project_root_state=root.get("root_state", root.get("state")),
            root_id=root.get("root_id") or root.get("id"),
            backend=root.get("backend"),
            sandbox_mount_state=root.get("sandbox_mount_state"),
            inventory_state=state,
        ),
    }


def _workspace_attention_state(
    *,
    profile: str,
    workspace: Mapping[str, Any],
    project_root: Mapping[str, Any],
    resolution: Mapping[str, Any],
) -> str:
    service_errors: list[str] = []
    resolution_status = str(resolution.get("status") or "").strip().lower()
    if resolution_status == "failed":
        service_errors.append("workspace_resolution_failed")
    elif resolution_status == "partial":
        service_errors.append("workspace_resolution_partial")

    if project_root.get("backend") == "sandbox_volume" and project_root.get("state") == "attached":
        mount_state = str(project_root.get("sandbox_mount_state") or "not_configured").strip().lower()
        if mount_state not in {"ready", "mounted"}:
            service_errors.append(f"sandbox_mount_{mount_state or 'not_configured'}")

    inventory = project_root.get("file_inventory")
    inventory_state = (
        project_root.get("file_inventory_state")
        or (inventory.get("state") if isinstance(inventory, Mapping) else None)
        or "not_started"
    )
    return workspace_attention_state(
        workspace_profile=profile,
        project_root_state=project_root.get("state"),
        inventory_state=inventory_state,
        service_errors=service_errors,
        archived=bool(workspace.get("archived")),
    )


def _non_negative_int(value: Any, *, default: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, coerced)


def _partial_errors(partial_errors: Any) -> list[dict[str, Any]]:
    if not partial_errors:
        return []
    if isinstance(partial_errors, Mapping):
        return [_sanitized_partial_error(partial_errors)]
    if not isinstance(partial_errors, list):
        return [
            {
                "scope": "workspace",
                "code": _PARTIAL_REASON,
                "message": _PARTIAL_MESSAGE,
            }
        ]

    errors: list[dict[str, Any]] = []
    for error in partial_errors:
        if isinstance(error, Mapping):
            errors.append(_sanitized_partial_error(error))
        else:
            errors.append(
                {
                    "scope": "workspace",
                    "code": _PARTIAL_REASON,
                    "message": _PARTIAL_MESSAGE,
                }
            )
    return errors


def _sanitized_partial_error(error: Mapping[str, Any]) -> dict[str, Any]:
    scope = str(error.get("scope") or "workspace").strip() or "workspace"
    return {
        "scope": scope,
        "code": _PARTIAL_REASON,
        "message": _PARTIAL_MESSAGE,
    }


def _workspace_services(service_capabilities: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(service_capabilities, Mapping):
        return {}
    services = service_capabilities.get("workspace_services")
    if not isinstance(services, Mapping):
        return {}
    return {
        str(key): dict(value)
        for key, value in services.items()
        if isinstance(value, Mapping)
    }


def _base_allowed_actions(service_capabilities: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(service_capabilities, Mapping):
        return {}
    actions = service_capabilities.get("allowed_actions")
    if not isinstance(actions, Mapping):
        return {}
    return {
        str(key): dict(value)
        for key, value in actions.items()
        if isinstance(value, Mapping)
    }


def _allowed_actions(
    *,
    profile: str,
    project_root: Mapping[str, Any],
    resolution: Mapping[str, Any],
    workspace_services: Mapping[str, Mapping[str, Any]],
    base_allowed_actions: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    actions = {
        str(name): _normalize_action(value)
        for name, value in base_allowed_actions.items()
    }
    root_ready, root_reason = _project_root_ready(profile, project_root)
    mcp_root_ready = root_ready and _mcp_trust_ready(project_root)
    mcp_root_reason = root_reason if not root_ready else "mcp_trust_not_verified"
    resolution_status = str(resolution.get("status") or "")
    partial_reason_by_scope = _partial_reason_by_scope(resolution)

    actions["write_files"] = _root_action(
        root_ready=root_ready,
        root_reason=root_reason,
        resolution_status=resolution_status,
        partial_reason_by_scope=partial_reason_by_scope,
    )
    actions["create_preview"] = _preview_action(
        root_ready=root_ready,
        root_reason=root_reason,
        resolution_status=resolution_status,
        partial_reason_by_scope=partial_reason_by_scope,
        workspace_services=workspace_services,
    )
    actions["index_file_content"] = _file_indexing_action(
        root_ready=root_ready,
        root_reason=root_reason,
        resolution_status=resolution_status,
        partial_reason_by_scope=partial_reason_by_scope,
        project_root=project_root,
    )
    actions["scan_files"] = _file_inventory_scan_action(
        root_ready=root_ready,
        root_reason=root_reason,
        resolution_status=resolution_status,
        partial_reason_by_scope=partial_reason_by_scope,
        project_root=project_root,
    )
    actions["view_file_inventory"] = _file_inventory_view_action(
        profile=profile,
        project_root=project_root,
        resolution_status=resolution_status,
        partial_reason_by_scope=partial_reason_by_scope,
    )
    actions["run_sandbox"] = _service_project_action(
        service_name="sandbox",
        service_action=actions.get("use_sandbox"),
        workspace_services=workspace_services,
        root_ready=root_ready,
        root_reason=root_reason,
        resolution_status=resolution_status,
        partial_reason_by_scope=partial_reason_by_scope,
        blocked_reason="sandbox_not_available",
    )
    actions["use_mcp_tools"] = _service_project_action(
        service_name="mcp",
        service_action=actions.get("run_mcp_tools"),
        workspace_services=workspace_services,
        root_ready=mcp_root_ready,
        root_reason=mcp_root_reason,
        resolution_status=resolution_status,
        partial_reason_by_scope=partial_reason_by_scope,
        blocked_reason="mcp_not_available",
    )

    actions["run_mcp_tools"] = _preserved_service_action(
        service_name="mcp",
        action=actions.get("run_mcp_tools"),
        resolution_status=resolution_status,
        partial_reason_by_scope=partial_reason_by_scope,
        default_reason="mcp_not_configured",
    )
    actions["use_acp_agents"] = _service_project_action(
        service_name="acp",
        service_action=actions.get("use_acp_agents"),
        workspace_services=workspace_services,
        root_ready=root_ready,
        root_reason=root_reason,
        resolution_status=resolution_status,
        partial_reason_by_scope=partial_reason_by_scope,
        blocked_reason="acp_not_available",
    )
    actions["use_sandbox"] = _service_project_action(
        service_name="sandbox",
        service_action=actions.get("use_sandbox"),
        workspace_services=workspace_services,
        root_ready=root_ready,
        root_reason=root_reason,
        resolution_status=resolution_status,
        partial_reason_by_scope=partial_reason_by_scope,
        blocked_reason="sandbox_not_available",
    )
    return actions


def _normalize_action(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return fail_closed_action("action_not_configured")
    allowed = bool(value.get("allowed"))
    reason = value.get("reason_code")
    return {
        "allowed": allowed,
        "reason_code": None if allowed else str(reason or "action_not_configured"),
    }


def _project_root_ready(profile: str, project_root: Mapping[str, Any]) -> tuple[bool, str]:
    if profile != "project":
        return False, "project_root_not_configured"
    if project_root.get("state") != "attached":
        return False, f"project_root_{project_root.get('state') or 'not_configured'}"
    if not project_root.get("root_id") or not project_root.get("backend"):
        return False, "project_root_unresolved"
    if project_root.get("backend") == "sandbox_volume":
        mount_state = str(project_root.get("sandbox_mount_state") or "not_configured").strip().lower()
        if mount_state not in {"ready", "mounted"}:
            return False, f"sandbox_mount_{mount_state or 'not_configured'}"
    return True, ""


def _project_root_exists(profile: str, project_root: Mapping[str, Any]) -> tuple[bool, str]:
    if profile != "project":
        return False, "project_root_not_configured"
    if project_root.get("state") == "not_configured":
        return False, "project_root_not_configured"
    if not project_root.get("root_id") or not project_root.get("backend"):
        return False, "project_root_unresolved"
    return True, ""


def _partial_reason_by_scope(resolution: Mapping[str, Any]) -> dict[str, str]:
    if resolution.get("status") != "partial":
        return {}
    reasons: dict[str, str] = {}
    for error in resolution.get("partial_errors") or []:
        if not isinstance(error, Mapping):
            continue
        scope = str(error.get("scope") or "").strip().lower()
        code = str(error.get("code") or "").strip()
        if scope and code:
            reasons[scope] = code
    return reasons


def _root_action(
    *,
    root_ready: bool,
    root_reason: str,
    resolution_status: str,
    partial_reason_by_scope: Mapping[str, str],
) -> dict[str, Any]:
    if resolution_status == "failed":
        return fail_closed_action("workspace_identity_unresolved")
    if resolution_status != "complete":
        return fail_closed_action(_PARTIAL_REASON)
    if not root_ready:
        return fail_closed_action(root_reason)
    return allowed_action()


def _preview_action(
    *,
    root_ready: bool,
    root_reason: str,
    resolution_status: str,
    partial_reason_by_scope: Mapping[str, str],
    workspace_services: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if resolution_status == "failed":
        return fail_closed_action("workspace_identity_unresolved")
    if resolution_status != "complete":
        return fail_closed_action(partial_reason_by_scope.get("preview") or _PARTIAL_REASON)
    if not root_ready:
        return fail_closed_action(root_reason)
    preview = workspace_services.get("preview", {})
    if str(preview.get("state") or "").strip().lower() != "available":
        return fail_closed_action(str(preview.get("reason_code") or "preview_not_configured"))
    return allowed_action()


def _file_indexing_action(
    *,
    root_ready: bool,
    root_reason: str,
    resolution_status: str,
    partial_reason_by_scope: Mapping[str, str],
    project_root: Mapping[str, Any],
) -> dict[str, Any]:
    if resolution_status == "failed":
        return fail_closed_action("workspace_identity_unresolved")
    if resolution_status != "complete":
        return fail_closed_action(partial_reason_by_scope.get("indexing") or _PARTIAL_REASON)
    if not root_ready:
        return fail_closed_action(root_reason)
    indexing_state = str(project_root.get("indexing_state") or "").strip().lower()
    if indexing_state not in {"enabled", "ready", "available"}:
        return fail_closed_action(
            "file_indexing_disabled" if indexing_state == "disabled" else "file_indexing_not_ready"
        )
    return allowed_action()


def _file_inventory_scan_action(
    *,
    root_ready: bool,
    root_reason: str,
    resolution_status: str,
    partial_reason_by_scope: Mapping[str, str],
    project_root: Mapping[str, Any],
) -> dict[str, Any]:
    if resolution_status == "failed":
        return fail_closed_action("workspace_identity_unresolved")
    if resolution_status != "complete":
        return fail_closed_action(partial_reason_by_scope.get("file_inventory") or _PARTIAL_REASON)
    if not root_ready:
        return fail_closed_action(root_reason)
    inventory_state = str(project_root.get("file_inventory_state") or "").strip().lower()
    if inventory_state == "disabled":
        return fail_closed_action("file_inventory_disabled")
    return allowed_action()


def _file_inventory_view_action(
    *,
    profile: str,
    project_root: Mapping[str, Any],
    resolution_status: str,
    partial_reason_by_scope: Mapping[str, str],
) -> dict[str, Any]:
    if resolution_status == "failed":
        return fail_closed_action("workspace_identity_unresolved")
    if resolution_status != "complete":
        return fail_closed_action(partial_reason_by_scope.get("file_inventory") or _PARTIAL_REASON)
    root_exists, root_reason = _project_root_exists(profile, project_root)
    if not root_exists:
        return fail_closed_action(root_reason)
    return allowed_action()


def _service_project_action(
    *,
    service_name: str,
    service_action: Mapping[str, Any] | None,
    workspace_services: Mapping[str, Mapping[str, Any]],
    root_ready: bool,
    root_reason: str,
    resolution_status: str,
    partial_reason_by_scope: Mapping[str, str],
    blocked_reason: str,
) -> dict[str, Any]:
    if resolution_status == "failed":
        return fail_closed_action("workspace_identity_unresolved")
    if resolution_status != "complete":
        return fail_closed_action(partial_reason_by_scope.get(service_name) or _PARTIAL_REASON)
    if partial_reason_by_scope:
        return fail_closed_action(partial_reason_by_scope.get(service_name) or _PARTIAL_REASON)
    if not root_ready:
        return fail_closed_action(root_reason or "project_root_unresolved")
    service = workspace_services.get(service_name, {})
    if str(service.get("state") or "").strip().lower() != "available":
        return fail_closed_action(str(service.get("reason_code") or blocked_reason))
    normalized_action = _normalize_action(service_action)
    if not normalized_action["allowed"]:
        return normalized_action
    return allowed_action()


def _preserved_service_action(
    *,
    service_name: str,
    action: Mapping[str, Any] | None,
    resolution_status: str,
    partial_reason_by_scope: Mapping[str, str],
    default_reason: str,
) -> dict[str, Any]:
    if resolution_status == "failed":
        return fail_closed_action("workspace_identity_unresolved")
    if resolution_status != "complete":
        return fail_closed_action(partial_reason_by_scope.get(service_name) or _PARTIAL_REASON)
    if partial_reason_by_scope:
        return fail_closed_action(partial_reason_by_scope.get(service_name) or _PARTIAL_REASON)
    if action is None:
        return fail_closed_action(default_reason)
    return _normalize_action(action)


def _mcp_trust_ready(project_root: Mapping[str, Any]) -> bool:
    trust_state = str(project_root.get("mcp_trust_state") or "").strip().lower()
    return trust_state in {"trusted", "approved", "allowed", "not_required"}

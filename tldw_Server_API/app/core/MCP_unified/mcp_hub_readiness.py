"""MCP Hub readiness policy helpers with sanitized response payload builders."""

from __future__ import annotations

from typing import Any

_READINESS_REASON_PRIORITY = [
    "auth_missing",
    "runtime_unavailable",
    "preflight_failed",
    "unreachable",
    "discovery_failed",
    "config_changed",
    "discovery_not_run",
    "no_tools_returned",
    "catalog_expired",
    "partial_capability",
]
_DISPLAY_STATE_BY_REASON = {
    "not_configured": "needs_setup",
    "preflight_failed": "needs_attention",
    "discovery_not_run": "needs_attention",
    "auth_missing": "needs_attention",
    "runtime_unavailable": "needs_attention",
    "unreachable": "needs_attention",
    "discovery_failed": "needs_attention",
    "no_tools_returned": "no_tools",
    "config_changed": "stale",
    "catalog_expired": "stale",
    "partial_capability": "ready",
}
_ACTIONS_BY_REASON = {
    "not_configured": ["add_server"],
    "preflight_failed": ["edit_config", "validate", "view_details"],
    "discovery_not_run": ["refresh_discovery", "edit_config"],
    "auth_missing": ["open_credentials", "view_details"],
    "runtime_unavailable": ["edit_config", "view_details"],
    "unreachable": ["edit_config", "refresh_discovery", "view_details"],
    "discovery_failed": ["refresh_discovery", "view_details"],
    "no_tools_returned": ["refresh_discovery", "view_details"],
    "config_changed": ["refresh_discovery", "edit_config"],
    "catalog_expired": ["refresh_discovery", "view_details"],
    "partial_capability": ["open_tool_catalog", "view_details"],
}


def _readiness_unique_reasons(reasons: list[str]) -> list[str]:
    """Return reason codes ordered by MCP Hub readiness priority."""
    seen = set()
    ordered: list[str] = []
    priority = {reason: index for index, reason in enumerate(_READINESS_REASON_PRIORITY)}
    for reason in sorted(reasons, key=lambda value: priority.get(value, len(priority))):
        if reason not in seen:
            seen.add(reason)
            ordered.append(reason)
    return ordered


def _readiness_primary_reason(reasons: list[str]) -> str | None:
    """Return the highest-priority readiness reason, if any."""
    ordered = _readiness_unique_reasons(reasons)
    return ordered[0] if ordered else None


def _readiness_allowed_actions(reasons: list[str]) -> list[str]:
    """Union readiness actions for the supplied reason codes."""
    actions: list[str] = []
    for reason in reasons:
        for action in _ACTIONS_BY_REASON.get(reason, []):
            if action not in actions:
                actions.append(action)
    return actions


def _readiness_message(primary_reason: str | None, credential_state: str) -> str:
    """Return a user-facing readiness status message without exposing config details."""
    if primary_reason == "auth_missing":
        return "Credentials are required before this server can be used."
    if primary_reason == "runtime_unavailable":
        return "Runtime is not available for this server."
    if primary_reason == "preflight_failed":
        return "Preflight validation failed. Check the server configuration."
    if primary_reason == "unreachable":
        return "Server cannot be reached."
    if primary_reason == "discovery_failed":
        return "Discovery ran but failed."
    if primary_reason == "config_changed":
        return "Server config or discovery state changed. Refresh discovery."
    if primary_reason == "discovery_not_run":
        if credential_state == "not_required":
            return "No credentials required. Discover tools to make this server available."
        return "Server is saved, but tool discovery has not run."
    if primary_reason == "no_tools_returned":
        return "Server responded, but exposed no tools."
    if primary_reason == "catalog_expired":
        return "Tool catalog is stale. Refresh discovery."
    if primary_reason == "partial_capability":
        return "Ready with limited capability."
    if primary_reason == "not_configured":
        return "Add an external server to start MCP Hub setup."
    return "Ready. No credentials required." if credential_state == "not_required" else "Ready."


def _credential_state_for_external_row(row: dict[str, Any]) -> str:
    """Normalize external-server credential state without returning secret material."""
    slots = [slot for slot in (row.get("credential_slots") or []) if isinstance(slot, dict)]
    required_slots = [slot for slot in slots if bool(slot.get("is_required", True))]
    if any(not bool(slot.get("secret_configured")) for slot in required_slots):
        return "required_missing"
    if any(bool(slot.get("secret_configured")) for slot in slots) or row.get("auth_template_valid") is True:
        return "configured"
    has_auth_template = bool(row.get("auth_template_present"))
    if bool(row.get("secret_configured")) and not has_auth_template and not slots:
        return "legacy_fallback"
    blocked_reason = str(row.get("auth_template_blocked_reason") or "").strip()
    if (
        str(row.get("transport") or "").strip().lower() == "stdio"
        and not has_auth_template
        and not slots
        and blocked_reason in {"", "no_auth_template"}
    ):
        return "not_required"
    return "unknown"


def _is_authoritative_external_tool(entry: dict[str, Any], server_id: str) -> bool:
    """Return True only for registry signals that authoritatively belong to an external server."""
    tool_name = str(entry.get("tool_name") or "")
    module_name = str(entry.get("module") or "")
    return tool_name.startswith(f"ext.{server_id}.") or module_name == f"external.{server_id}"


def _matching_external_tool_entries(
    registry_entries: list[dict[str, Any]],
    *,
    server_id: str,
) -> list[dict[str, Any]]:
    """Return registry entries that belong to one external server."""
    return [entry for entry in registry_entries if _is_authoritative_external_tool(entry, server_id)]


def _operation_metadata_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return in-flight operation metadata when the row exposes it."""
    current_operation = row.get("current_operation")
    return current_operation if isinstance(current_operation, dict) else None


def _coerce_nonnegative_int(value: Any) -> int:
    """Coerce untrusted numeric refresh metadata to a non-negative integer."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _extract_refresh_payload(raw_result: dict[str, Any]) -> dict[str, Any]:
    """Extract the manager refresh payload from direct or MCP content-wrapped results."""
    if not isinstance(raw_result, dict):
        return {}
    if any(key in raw_result for key in ("refreshed_servers", "total_servers", "virtual_tools", "errors")):
        return raw_result
    result = raw_result.get("result")
    if isinstance(result, dict):
        extracted = _extract_refresh_payload(result)
        if extracted:
            return extracted
    content = raw_result.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            json_payload = item.get("json")
            if isinstance(json_payload, dict):
                return json_payload
    return {}


def sanitize_discovery_refresh_result_payload(raw_result: dict[str, Any]) -> dict[str, Any]:
    """Return refresh metadata with generic error messages only."""
    payload = _extract_refresh_payload(raw_result)
    raw_errors = payload.get("errors") if isinstance(payload.get("errors"), dict) else {}
    errors = {str(server_id): "Discovery refresh failed." for server_id in raw_errors}
    return {
        "refreshed_servers": _coerce_nonnegative_int(payload.get("refreshed_servers")),
        "total_servers": _coerce_nonnegative_int(payload.get("total_servers")),
        "virtual_tools": _coerce_nonnegative_int(payload.get("virtual_tools")),
        "errors": errors,
    }


def build_server_readiness_payload(
    row: dict[str, Any],
    *,
    registry_entries: list[dict[str, Any]],
    current_operation: dict[str, Any] | None = None,
    last_validation_at: Any | None = None,
    refresh_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build sanitized readiness payload for one external server row."""
    server_id = str(row.get("id") or "")
    server_name = str(row.get("name") or server_id)
    credential_state = _credential_state_for_external_row(row)
    matching_entries = _matching_external_tool_entries(registry_entries, server_id=server_id)
    tool_count = len(matching_entries)
    operation = current_operation or _operation_metadata_from_row(row)
    if operation:
        operation_type = str(operation.get("operation_type") or "validation")
        if operation_type not in {"validation", "discovery"}:
            operation_type = "validation"
        return {
            "server_id": server_id,
            "server_name": server_name,
            "display_state": "checking",
            "credential_state": credential_state,
            "tool_count": tool_count,
            "reason_codes": [],
            "primary_reason_code": None,
            "allowed_actions": ["view_details"],
            "message": (
                "Tool discovery is running."
                if operation_type == "discovery"
                else "Preflight validation is running."
            ),
            "current_operation": {
                "operation_type": operation_type,
                "started_at": operation.get("started_at"),
                "message": operation.get("message"),
            },
            "last_validation_at": last_validation_at or row.get("last_validation_at"),
            "last_discovery_at": row.get("last_discovery_at"),
            "last_successful_discovery_at": row.get("last_successful_discovery_at"),
        }

    reasons: list[str] = []
    if credential_state == "required_missing":
        reasons.append("auth_missing")
    if not bool(row.get("enabled")) or row.get("runtime_executable") is False:
        reasons.append("runtime_unavailable")

    blocked_reason = row.get("auth_template_blocked_reason")
    if (row.get("auth_template_present") and row.get("auth_template_valid") is False) or (
        blocked_reason and str(blocked_reason) != "no_auth_template"
    ):
        reasons.append("preflight_failed")

    last_error_category = row.get("last_error_category")
    last_error_message = row.get("last_error_message")
    refresh_errors = refresh_result.get("errors") if isinstance(refresh_result, dict) else {}
    if refresh_result is not None:
        if isinstance(refresh_errors, dict) and server_id in refresh_errors:
            reasons.append("discovery_failed")
            last_error_category = "discovery_failed"
            last_error_message = "Discovery refresh failed."
        elif last_error_category == "discovery_failed":
            last_error_category = None
            last_error_message = None
    elif last_error_category == "discovery_failed":
        reasons.append("discovery_failed")
        last_error_message = "Discovery refresh failed."

    if tool_count == 0 and "runtime_unavailable" not in reasons and "discovery_failed" not in reasons:
        reasons.append("discovery_not_run")
    if tool_count > 0 and any(entry.get("metadata_warnings") for entry in matching_entries):
        reasons.append("partial_capability")

    reason_codes = _readiness_unique_reasons(reasons)
    primary_reason = _readiness_primary_reason(reason_codes)
    display_state = _DISPLAY_STATE_BY_REASON.get(primary_reason or "", "ready")
    allowed_actions = (
        _readiness_allowed_actions(reason_codes)
        if reason_codes
        else ["open_tool_catalog", "view_details"]
    )
    return {
        "server_id": server_id,
        "server_name": server_name,
        "display_state": display_state,
        "credential_state": credential_state,
        "tool_count": tool_count,
        "reason_codes": reason_codes,
        "primary_reason_code": primary_reason,
        "allowed_actions": allowed_actions,
        "message": _readiness_message(primary_reason, credential_state),
        "current_operation": None,
        "last_validation_at": last_validation_at or row.get("last_validation_at"),
        "last_discovery_at": row.get("last_discovery_at"),
        "last_successful_discovery_at": row.get("last_successful_discovery_at"),
        "last_error_category": last_error_category,
        "last_error_message": last_error_message,
        "refresh_result": refresh_result,
    }


def is_operational_managed_external_row(row: dict[str, Any]) -> bool:
    """Return True when a row counts toward active MCP Hub readiness."""
    return (
        str(row.get("server_source") or "managed") == "managed"
        and not row.get("superseded_by_server_id")
        and bool(row.get("enabled"))
    )


def is_available_for_discovery_refresh(row: dict[str, Any]) -> bool:
    """Return True when the server can run discovery refresh from the current process."""
    return is_operational_managed_external_row(row) and row.get("runtime_executable") is True


def build_hub_readiness_payload(
    rows: list[dict[str, Any]],
    registry_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build sanitized aggregate MCP Hub readiness payload for visible external servers."""
    server_readiness = [
        build_server_readiness_payload(row, registry_entries=registry_entries)
        for row in rows
    ]
    operational_server_ids = {
        str(row.get("id") or "")
        for row in rows
        if is_operational_managed_external_row(row)
    }
    operational_readiness = [
        readiness
        for readiness in server_readiness
        if readiness["server_id"] in operational_server_ids
    ]

    if not operational_readiness:
        return {
            "display_state": "needs_setup",
            "reason_codes": ["not_configured"],
            "primary_reason_code": "not_configured",
            "allowed_actions": ["add_server"],
            "message": _readiness_message("not_configured", "not_required"),
            "servers": server_readiness,
            "total_servers": 0,
        }

    aggregate_reasons = _readiness_unique_reasons(
        [
            reason
            for readiness in operational_readiness
            for reason in readiness["reason_codes"]
        ]
    )
    primary_reason = _readiness_primary_reason(aggregate_reasons)
    if any(readiness["display_state"] == "checking" for readiness in operational_readiness):
        display_state = "checking"
    elif primary_reason:
        display_state = _DISPLAY_STATE_BY_REASON.get(primary_reason, "needs_attention")
    else:
        display_state = "ready"

    return {
        "display_state": display_state,
        "reason_codes": aggregate_reasons,
        "primary_reason_code": primary_reason,
        "allowed_actions": (
            _readiness_allowed_actions(aggregate_reasons)
            if aggregate_reasons
            else ["open_tool_catalog", "view_details"]
        ),
        "message": (
            _readiness_message(primary_reason, "configured")
            if primary_reason
            else "MCP Hub is ready."
        ),
        "servers": server_readiness,
        "total_servers": len(operational_readiness),
        "ready_server_count": sum(1 for item in operational_readiness if item["display_state"] == "ready"),
        "checking_server_count": sum(1 for item in operational_readiness if item["display_state"] == "checking"),
        "attention_server_count": sum(
            1 for item in operational_readiness if item["display_state"] == "needs_attention"
        ),
        "no_tool_server_count": sum(1 for item in operational_readiness if item["display_state"] == "no_tools"),
        "stale_server_count": sum(1 for item in operational_readiness if item["display_state"] == "stale"),
    }

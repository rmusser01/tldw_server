from __future__ import annotations

from collections.abc import Mapping
from typing import Any

OperatorSection = dict[str, object]


def _as_dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_dict(value: object) -> dict[str, object]:
    return _as_dict(value)


def _safe_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        coerced = int(value or 0)
    except (OverflowError, TypeError, ValueError):
        return 0
    return max(coerced, 0)


def _safe_list(value: object) -> list[object]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _safe_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _safe_reason(value: object) -> list[str]:
    reason = str(value or "").strip()
    return [reason] if reason else []


def _section(status: str, *, severity: str = "info", **extra: object) -> OperatorSection:
    return {"status": status, "severity": severity, **extra}


def _action(
    code: str,
    severity: str,
    section: str,
    message: str,
    endpoint: str | None = None,
    dry_run_required: bool = False,
) -> dict[str, object]:
    return {
        "code": code,
        "severity": severity,
        "section": section,
        "message": message,
        "endpoint": endpoint,
        "dry_run_required": dry_run_required,
    }


def build_operator_status(
    *,
    runtime_diagnostics: Mapping[str, Any] | None,
    macos_diagnostics: Mapping[str, Any] | None,
    startup_warning_summary: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    runtime_payload = _safe_dict(runtime_diagnostics)
    macos_payload = _safe_dict(macos_diagnostics)
    startup_payload = _safe_dict(startup_warning_summary)

    runtime_failed = "_section_error" in runtime_payload
    macos_failed = "_section_error" in macos_payload
    runtime_failure_reasons = _safe_reason(runtime_payload.get("_section_error"))
    macos_failure_reasons = _safe_reason(macos_payload.get("_section_error"))

    runtime_summary = _safe_dict(runtime_payload.get("summary"))
    ready_count = _safe_int(runtime_summary.get("ready"))
    runtime_total = _safe_int(runtime_summary.get("total"))
    host_local_warning_runtimes = _safe_list(
        runtime_summary.get("host_local_warning_runtimes")
    )
    repair_supported_runtimes = _safe_list(
        runtime_summary.get("repair_supported_runtimes")
    )

    helper = _safe_dict(macos_payload.get("helper"))
    templates = _safe_dict(macos_payload.get("templates"))
    image_store = _safe_dict(macos_payload.get("image_store"))
    recovery = _safe_dict(macos_payload.get("recovery_summary"))

    repair_endpoint = recovery.get("repair_endpoint")
    cleanup_plan_endpoint = recovery.get("cleanup_plan_endpoint")
    gc_candidates = _safe_int(image_store.get("gc_candidates"))
    helper_configured = _safe_bool(helper.get("configured"))
    helper_ready = _safe_bool(helper.get("ready"))
    image_store_configured = _safe_bool(image_store.get("configured"))
    startup_present = (
        _safe_bool(startup_payload.get("present"))
        if "present" in startup_payload
        else False
    )
    startup_blocking = (
        _safe_bool(startup_payload.get("blocking"))
        if "blocking" in startup_payload
        else False
    )
    macos_vz_bool_malformed = helper_configured is None or helper_ready is None
    image_store_bool_malformed = image_store_configured is None
    startup_bool_malformed = startup_present is None or startup_blocking is None
    configured_templates = [
        str(name)
        for name, value in templates.items()
        if _safe_bool(_safe_dict(value).get("configured")) is True
    ]
    unready_templates = [
        str(name)
        for name, value in templates.items()
        if _safe_bool(_safe_dict(value).get("configured")) is True
        and _safe_bool(_safe_dict(value).get("ready")) is False
    ]
    configured_macos_vz = helper_configured is True or bool(configured_templates)
    macos_vz_action_required = bool(
        configured_macos_vz
        and (
            helper_ready is False
            or bool(unready_templates)
            or str(recovery.get("status") or "").strip().lower() == "unavailable"
            or str(recovery.get("severity") or "").strip().lower() == "error"
        )
    )

    recommended_actions: list[dict[str, object]] = []
    if macos_vz_action_required:
        recommended_actions.append(
            _action(
                "restore_helper_readiness",
                "error",
                "macos_vz",
                "Restore configured macOS VZ helper/template readiness before relying on that runtime path.",
                endpoint="/api/v1/sandbox/admin/macos-diagnostics",
            )
        )
    if isinstance(repair_endpoint, str) and repair_endpoint:
        recommended_actions.append(
            _action(
                "run_repair_dry_run",
                "warning",
                "reconciliation",
                "Run the macOS reconciliation repair endpoint in dry-run mode first.",
                endpoint=repair_endpoint,
                dry_run_required=True,
            )
        )
    if isinstance(cleanup_plan_endpoint, str) and cleanup_plan_endpoint:
        recommended_actions.append(
            _action(
                "inspect_image_store_cleanup_plan",
                "warning",
                "image_store",
                "Inspect the image-store cleanup plan before deleting candidates.",
                endpoint=cleanup_plan_endpoint,
            )
        )
    if startup_blocking is True:
        recommended_actions.append(
            _action(
                "resolve_startup_warnings",
                "error",
                "startup_warnings",
                "Resolve blocking sandbox startup warnings before relying on sandbox execution.",
            )
        )

    sections = {
        "runtime_readiness": _section(
            "unknown"
            if runtime_failed
            else ("ready" if ready_count > 0 else "unavailable"),
            severity="warning"
            if runtime_failed
            else ("info" if ready_count > 0 else "error"),
            ready=ready_count,
            total=runtime_total,
            host_local_warning_runtimes=host_local_warning_runtimes,
            repair_supported_runtimes=repair_supported_runtimes,
            reasons=runtime_failure_reasons,
        ),
        "macos_vz": _section(
            "unknown"
            if macos_failed or macos_vz_bool_malformed
            else (
                "action_required"
                if macos_vz_action_required
                else ("ready" if helper_ready else "not_configured")
            ),
            severity="warning"
            if macos_failed or macos_vz_bool_malformed
            else ("error" if macos_vz_action_required else "info"),
            configured=helper_configured,
            helper_ready=helper_ready,
            reasons=macos_failure_reasons or _safe_list(helper.get("reasons")),
            configured_templates=configured_templates,
            unready_templates=unready_templates,
        ),
        "image_store": _section(
            "unknown"
            if macos_failed or image_store_bool_malformed
            else ("ready" if image_store_configured else "not_configured"),
            severity="warning"
            if macos_failed or image_store_bool_malformed
            else "info",
            configured=image_store_configured,
            gc_candidates=gc_candidates,
            reasons=macos_failure_reasons or _safe_list(image_store.get("reasons")),
        ),
        "reconciliation": _section(
            "unknown"
            if macos_failed
            else str(recovery.get("status") or "not_configured"),
            severity="warning"
            if macos_failed
            else str(recovery.get("severity") or "info").replace("ok", "info"),
            counts=_safe_dict(recovery.get("counts")),
            repair_endpoint=repair_endpoint,
            cleanup_plan_endpoint=cleanup_plan_endpoint,
            reasons=macos_failure_reasons,
        ),
        "evidence": _section("not_configured"),
        "security_boundaries": _section(
            "ready",
            host_local_warning_runtimes=host_local_warning_runtimes,
        ),
        "startup_warnings": _section(
            "unknown"
            if startup_bool_malformed
            else (
                "action_required"
                if startup_blocking
                else ("degraded" if startup_present else "ready")
            ),
            severity="warning"
            if startup_bool_malformed
            else (
                "error"
                if startup_blocking
                else ("warning" if startup_present else "info")
            ),
            present=startup_present,
            blocking=startup_blocking,
            codes=_safe_list(startup_payload.get("codes")),
        ),
    }

    has_section_failures = (
        runtime_failed
        or macos_failed
        or macos_vz_bool_malformed
        or image_store_bool_malformed
        or startup_bool_malformed
    )
    has_repair_action = any(
        action["code"] == "run_repair_dry_run" for action in recommended_actions
    )
    has_macos_vz_action = any(
        action["code"] == "restore_helper_readiness" for action in recommended_actions
    )
    has_cleanup_candidates = gc_candidates > 0 or (
        isinstance(cleanup_plan_endpoint, str) and bool(cleanup_plan_endpoint)
    )
    has_host_local_warnings = bool(host_local_warning_runtimes)

    if runtime_failed:
        overall_status = "unknown"
        overall_severity = "warning"
    elif ready_count <= 0:
        overall_status = "unavailable"
        overall_severity = "error"
    elif startup_blocking is True or has_repair_action or has_macos_vz_action:
        overall_status = "action_required"
        overall_severity = (
            "error" if startup_blocking is True or has_macos_vz_action else "warning"
        )
    elif (
        has_section_failures
        or startup_present is True
        or has_cleanup_candidates
        or has_host_local_warnings
    ):
        overall_status = "degraded"
        overall_severity = "warning"
    else:
        overall_status = "ready"
        overall_severity = "info"

    return {
        "source": "sandbox_operator_status",
        "overall_status": overall_status,
        "overall_severity": overall_severity,
        "summary": {
            "runtime_total": runtime_total,
            "runtime_ready": ready_count,
            "actions": len(recommended_actions),
        },
        "sections": sections,
        "recommended_actions": recommended_actions,
        "notes": [],
    }

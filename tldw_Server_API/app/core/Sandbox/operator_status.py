from __future__ import annotations

from collections.abc import Mapping
from typing import Any

OperatorSection = dict[str, object]


def _as_dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _section(status: str, *, severity: str = "info", **extra: object) -> OperatorSection:
    return {"status": status, "severity": severity, **extra}


def build_operator_status(
    *,
    runtime_diagnostics: Mapping[str, Any] | None,
    macos_diagnostics: Mapping[str, Any] | None,
    startup_warning_summary: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    runtime_payload = _as_dict(runtime_diagnostics)
    macos_payload = _as_dict(macos_diagnostics)
    startup_payload = _as_dict(startup_warning_summary)

    runtime_summary = _as_dict(runtime_payload.get("summary"))
    ready_count = int(runtime_summary.get("ready") or 0)
    helper = _as_dict(macos_payload.get("helper"))
    image_store = _as_dict(macos_payload.get("image_store"))
    recovery = _as_dict(macos_payload.get("recovery_summary"))

    sections = {
        "runtime_readiness": _section(
            "ready" if ready_count else "unavailable",
            severity="info" if ready_count else "error",
            ready=ready_count,
            total=int(runtime_summary.get("total") or 0),
            host_local_warning_runtimes=list(
                runtime_summary.get("host_local_warning_runtimes") or []
            ),
            repair_supported_runtimes=list(
                runtime_summary.get("repair_supported_runtimes") or []
            ),
        ),
        "macos_vz": _section(
            "ready" if bool(helper.get("ready")) else "not_configured",
            configured=bool(helper.get("configured")),
            helper_ready=bool(helper.get("ready")),
            reasons=list(helper.get("reasons") or []),
        ),
        "image_store": _section(
            "ready" if bool(image_store.get("configured")) else "not_configured",
            configured=bool(image_store.get("configured")),
            gc_candidates=int(image_store.get("gc_candidates") or 0),
            reasons=list(image_store.get("reasons") or []),
        ),
        "reconciliation": _section(
            str(recovery.get("status") or "not_configured"),
            severity=str(recovery.get("severity") or "info").replace("ok", "info"),
            counts=dict(recovery.get("counts") or {}),
            repair_endpoint=recovery.get("repair_endpoint"),
            cleanup_plan_endpoint=recovery.get("cleanup_plan_endpoint"),
        ),
        "evidence": _section("not_configured"),
        "security_boundaries": _section(
            "ready",
            host_local_warning_runtimes=list(
                runtime_summary.get("host_local_warning_runtimes") or []
            ),
        ),
        "startup_warnings": _section(
            "action_required"
            if bool(startup_payload.get("blocking"))
            else ("degraded" if bool(startup_payload.get("present")) else "ready"),
            severity="error"
            if bool(startup_payload.get("blocking"))
            else ("warning" if bool(startup_payload.get("present")) else "info"),
            present=bool(startup_payload.get("present")),
            blocking=bool(startup_payload.get("blocking")),
            codes=list(startup_payload.get("codes") or []),
        ),
    }

    overall_status = "ready" if ready_count else "unavailable"
    overall_severity = "info" if ready_count else "error"
    return {
        "source": "sandbox_operator_status",
        "overall_status": overall_status,
        "overall_severity": overall_severity,
        "summary": {
            "runtime_total": int(runtime_summary.get("total") or 0),
            "runtime_ready": ready_count,
            "actions": 0,
        },
        "sections": sections,
        "recommended_actions": [],
        "notes": [],
    }

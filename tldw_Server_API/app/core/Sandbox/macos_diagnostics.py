"""Shared, side-effect-free diagnostics for macOS sandbox runtimes."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.testing import is_truthy

from .macos_virtualization.helper_client import (
    MacOSVirtualizationHelperClient,
    MacOSVirtualizationHelperUnavailable,
)
from .models import RuntimeType
from .runtime_capabilities import RuntimePreflightResult, collect_runtime_preflights
from .runners.vz_common import vz_host_facts

_VZ_LINUX_TEMPLATE_MISSING_REASON = "vz_linux_template_missing"
_VZ_MACOS_TEMPLATE_MISSING_REASON = "macos_template_missing"


def _truthy(value: str | None) -> bool:
    return is_truthy(value)


def _execution_mode_for_runtime(runtime: RuntimeType) -> str:
    env_key_by_runtime = {
        RuntimeType.vz_linux: "TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC",
        RuntimeType.vz_macos: "TLDW_SANDBOX_VZ_MACOS_FAKE_EXEC",
        RuntimeType.seatbelt: "TLDW_SANDBOX_SEATBELT_FAKE_EXEC",
    }
    env_key = env_key_by_runtime.get(runtime)
    if env_key and _truthy(os.getenv(env_key)):
        return "fake"
    return "none"


def _remediation_for_reasons(reasons: list[str]) -> str | None:
    if not reasons:
        return None
    if "macos_required" in reasons or "apple_silicon_required" in reasons:
        return "Run this runtime on an Apple silicon macOS host."
    if "macos_virtualization_helper_unavailable" in reasons:
        return "Install or start the macOS virtualization helper service."
    if "macos_helper_missing" in reasons:
        return "Configure the macOS virtualization helper and mark it ready."
    if _VZ_LINUX_TEMPLATE_MISSING_REASON in reasons or _VZ_MACOS_TEMPLATE_MISSING_REASON in reasons:
        return "Configure the required runtime template and mark it ready."
    if "real_execution_not_implemented" in reasons:
        return "Enable fake execution for scaffolding or implement the real runtime path."
    if "strict_allowlist_not_supported" in reasons:
        return "Use deny_all for this runtime; allowlist is not implemented."
    if "seatbelt_unavailable" in reasons:
        return "Enable the seatbelt runtime on supported macOS hosts."
    return "Review runtime preflight reasons and host readiness."


def probe_host() -> dict[str, object]:
    """Report coarse host facts and whether the host can support VZ runtimes."""

    facts = vz_host_facts()
    reasons: list[str] = []
    if facts.get("os") != "darwin":
        reasons.append("macos_required")
    if not bool(facts.get("apple_silicon")):
        reasons.append("apple_silicon_required")
    return {
        **facts,
        "macos_version": platform.mac_ver()[0] or None,
        "supported": not reasons,
        "reasons": reasons,
    }


def probe_helper() -> dict[str, object]:
    """Report helper readiness plus optional operator metadata such as local path facts."""

    raw_path = str(os.getenv("TLDW_SANDBOX_MACOS_HELPER_PATH") or "").strip()
    raw_socket = str(os.getenv("TLDW_SANDBOX_MACOS_HELPER_SOCKET") or "").strip()
    path = raw_path or raw_socket or None
    exists = bool(path and Path(path).exists())
    executable = bool(raw_path and exists and os.access(raw_path, os.X_OK))
    configured = bool(path)
    ready = False
    transport: str | None = None
    protocol_version: str | None = None
    helper_version: str | None = None
    reasons: list[str] = []

    if not configured:
        reasons.append("macos_helper_path_unconfigured")
    elif not exists:
        reasons.append("macos_helper_path_missing")
    elif raw_path and not executable:
        reasons.append("macos_helper_not_executable")

    try:
        ping = MacOSVirtualizationHelperClient().ping()
        ready = str(ping.status).strip().lower() == "ok"
        configured = configured or ready
        transport = str(ping.details.get("transport") or "").strip() or None
        protocol_version = str(ping.protocol_version or "").strip() or None
        helper_version = str(ping.helper_version or "").strip() or None
    except MacOSVirtualizationHelperUnavailable as exc:
        reasons.append(str(exc) or "macos_virtualization_helper_unavailable")

    if not ready and "macos_virtualization_helper_unavailable" not in reasons:
        reasons.append("macos_helper_missing")

    return {
        "configured": configured,
        "path": path,
        "exists": exists,
        "executable": executable,
        "ready": ready,
        "transport": transport,
        "protocol_version": protocol_version,
        "helper_version": helper_version,
        "reasons": reasons,
    }


def _template_status(
    *,
    source_env_key: str,
    ready_env_key: str,
    missing_reason: str,
) -> dict[str, object]:
    raw_source = str(os.getenv(source_env_key) or "").strip()
    source = raw_source or None
    ready = _truthy(os.getenv(ready_env_key))
    configured = bool(source) or ready
    reasons: list[str] = []

    if not configured:
        reasons.append("template_unconfigured")
    if not ready:
        reasons.append(missing_reason)

    return {
        "configured": configured,
        "ready": ready,
        "source": source,
        "reasons": reasons,
    }


def _vz_linux_template_status() -> dict[str, object]:
    raw_source = str(os.getenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE") or "").strip()
    source = raw_source or None
    reasons: list[str] = []
    ready = False
    configured = bool(source)

    if not configured:
        reasons.append("template_unconfigured")
        reasons.append(_VZ_LINUX_TEMPLATE_MISSING_REASON)
        return {
            "configured": configured,
            "ready": ready,
            "source": source,
            "reasons": reasons,
        }

    try:
        validation = MacOSVirtualizationHelperClient().validate_template(
            {"runtime": RuntimeType.vz_linux.value, "template": source}
        )
        ready = bool(validation.get("ready"))
        helper_reasons = [str(reason) for reason in validation.get("reasons", []) if str(reason).strip()]
        reasons.extend(helper_reasons)
    except MacOSVirtualizationHelperUnavailable as exc:
        reasons.append(str(exc) or "macos_virtualization_helper_unavailable")

    if not ready and not reasons:
        reasons.append(_VZ_LINUX_TEMPLATE_MISSING_REASON)

    return {
        "configured": configured,
        "ready": ready,
        "source": source,
        "reasons": reasons,
    }


def probe_templates() -> dict[str, dict[str, object]]:
    """Report template readiness for the VZ runtime families."""

    return {
        "vz_linux": _vz_linux_template_status(),
        "vz_macos": _template_status(
            source_env_key="TLDW_SANDBOX_VZ_MACOS_TEMPLATE_SOURCE",
            ready_env_key="TLDW_SANDBOX_VZ_MACOS_TEMPLATE_READY",
            missing_reason=_VZ_MACOS_TEMPLATE_MISSING_REASON,
        ),
    }


def probe_runtime_statuses(
    *,
    runtime_preflights: dict[RuntimeType, RuntimePreflightResult],
) -> dict[str, dict[str, object]]:
    """Summarize admin-facing runtime posture from shared runtime preflight results."""

    statuses: dict[str, dict[str, object]] = {}
    for runtime in (RuntimeType.vz_linux, RuntimeType.vz_macos, RuntimeType.seatbelt):
        preflight = runtime_preflights.get(runtime)
        reasons = list((preflight.reasons if preflight else []) or [])
        preflight_execution_mode = str((preflight.execution_mode if preflight else "") or "").strip().lower()
        statuses[runtime.value] = {
            "available": bool(preflight.available) if preflight is not None else False,
            "supported_trust_levels": list((preflight.supported_trust_levels if preflight else []) or []),
            "reasons": reasons,
            "execution_mode": (
                preflight_execution_mode
                if preflight_execution_mode in {"fake", "real"}
                else _execution_mode_for_runtime(runtime)
            ),
            "remediation": _remediation_for_reasons(reasons),
        }
    return statuses


def probe_reconciliation(orchestrator: Any | None = None) -> dict[str, object]:
    """Compare persisted VZ session rows with live helper VM state when available."""

    summary: dict[str, object] = {
        "computed": False,
        "persisted_sessions": 0,
        "live_vms": 0,
        "stale_session_ids": [],
        "orphaned_vm_ids": [],
        "reasons": [],
    }
    if orchestrator is None:
        return summary

    lister = getattr(orchestrator, "list_vz_session_controls", None)
    if not callable(lister):
        summary["reasons"] = ["vz_session_listing_unavailable"]
        return summary

    persisted_rows = [dict(row) for row in lister() or [] if isinstance(row, dict)]
    summary["persisted_sessions"] = len(persisted_rows)

    try:
        live = MacOSVirtualizationHelperClient().list_vms()
    except MacOSVirtualizationHelperUnavailable as exc:
        summary["reasons"] = [str(exc) or "macos_virtualization_helper_unavailable"]
        return summary

    live_vm_ids = {
        str(vm.vm_id).strip()
        for vm in list(live.vms or [])
        if str(getattr(vm, "vm_id", "")).strip()
    }
    persisted_vm_by_session = {
        str(row.get("id") or "").strip(): str(row.get("vm_id") or "").strip()
        for row in persisted_rows
        if str(row.get("id") or "").strip() and str(row.get("vm_id") or "").strip()
    }
    persisted_vm_ids = {vm_id for vm_id in persisted_vm_by_session.values() if vm_id}

    summary["computed"] = True
    summary["live_vms"] = len(live_vm_ids)
    summary["stale_session_ids"] = sorted(
        session_id
        for session_id, vm_id in persisted_vm_by_session.items()
        if vm_id not in live_vm_ids
    )
    summary["orphaned_vm_ids"] = sorted(vm_id for vm_id in live_vm_ids if vm_id not in persisted_vm_ids)
    return summary


def collect_macos_diagnostics(orchestrator: Any | None = None) -> dict[str, Any]:
    """Aggregate host, helper, template, and runtime diagnostics for admin callers."""

    host = probe_host()
    helper = probe_helper()
    templates = probe_templates()
    runtime_preflights = collect_runtime_preflights(network_policy="deny_all")
    runtimes = probe_runtime_statuses(runtime_preflights=runtime_preflights)
    reconciliation = probe_reconciliation(orchestrator)
    return {
        "host": host,
        "helper": helper,
        "templates": templates,
        "runtimes": runtimes,
        "reconciliation": reconciliation,
    }

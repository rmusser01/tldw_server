"""Shared, side-effect-free diagnostics for macOS sandbox runtimes."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.testing import is_truthy

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
    path = raw_path or None
    ready = _truthy(os.getenv("TLDW_SANDBOX_MACOS_HELPER_READY"))
    configured = bool(path) or ready
    exists = bool(path and Path(path).exists())
    executable = bool(path and exists and os.access(path, os.X_OK))
    reasons: list[str] = []

    if not configured:
        reasons.append("macos_helper_path_unconfigured")
    elif not exists:
        reasons.append("macos_helper_path_missing")
    elif not executable:
        reasons.append("macos_helper_not_executable")
    if not ready:
        reasons.append("macos_helper_missing")

    transport = "fake" if _truthy(os.getenv("TEST_MODE")) and ready else None
    return {
        "configured": configured,
        "path": path,
        "exists": exists,
        "executable": executable,
        "ready": ready,
        "transport": transport,
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


def probe_templates() -> dict[str, dict[str, object]]:
    """Report template readiness for the VZ runtime families."""

    return {
        "vz_linux": _template_status(
            source_env_key="TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE",
            ready_env_key="TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY",
            missing_reason=_VZ_LINUX_TEMPLATE_MISSING_REASON,
        ),
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


def collect_macos_diagnostics() -> dict[str, Any]:
    """Aggregate host, helper, template, and runtime diagnostics for admin callers."""

    host = probe_host()
    helper = probe_helper()
    templates = probe_templates()
    runtime_preflights = collect_runtime_preflights(network_policy="deny_all")
    runtimes = probe_runtime_statuses(runtime_preflights=runtime_preflights)
    return {
        "host": host,
        "helper": helper,
        "templates": templates,
        "runtimes": runtimes,
    }

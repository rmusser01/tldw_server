"""Shared, side-effect-free diagnostics for macOS sandbox runtimes."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.testing import is_truthy

from .image_store import ImageStoreValidationError, SandboxImageStore
from .macos_virtualization.helper_client import (
    MacOSVirtualizationHelperClient,
    MacOSVirtualizationHelperProtocolError,
    MacOSVirtualizationHelperUnavailable,
)
from .models import RuntimeType
from .runners.vz_common import vz_host_facts
from .runtime_capabilities import RuntimePreflightResult, collect_runtime_preflights
from .vz_reconciliation import collect_vz_reconciliation

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
    if "macos_virtualization_helper_protocol_mismatch" in reasons:
        return "Update the macOS virtualization helper and Python client to compatible protocol versions."
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
    except MacOSVirtualizationHelperProtocolError:
        reasons.append("macos_virtualization_helper_protocol_mismatch")

    if not ready and not any(
        reason in reasons
        for reason in (
            "macos_virtualization_helper_unavailable",
            "macos_virtualization_helper_protocol_mismatch",
        )
    ):
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
    except MacOSVirtualizationHelperProtocolError:
        reasons.append("macos_virtualization_helper_protocol_mismatch")

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

    return collect_vz_reconciliation(orchestrator)


def probe_image_store(reconciliation: dict[str, object] | None = None) -> dict[str, object]:
    """Report read-only image-store state plus correlation to reconciliation items."""

    root_text = str(os.getenv("TLDW_SANDBOX_IMAGE_STORE_ROOT") or "").strip()
    if not root_text:
        return {
            "configured": False,
            "root_path": None,
            "registered_templates": 0,
            "run_manifests": 0,
            "gc_candidates": 0,
            "items": [],
            "reasons": [],
        }

    try:
        store = SandboxImageStore(root_path=root_text)
    except (ImageStoreValidationError, OSError, ValueError) as exc:
        return {
            "configured": True,
            "root_path": root_text,
            "registered_templates": 0,
            "run_manifests": 0,
            "gc_candidates": 0,
            "items": [],
            "reasons": [f"image_store_unavailable: {exc}"],
        }

    reconciliation_items = [
        dict(item) for item in list((reconciliation or {}).get("items") or []) if isinstance(item, dict)
    ]
    active_run_ids = {
        str(item.get("run_id") or "").strip()
        for item in reconciliation_items
        if str(item.get("vm_id") or "").strip() and str(item.get("run_id") or "").strip()
    }
    active_run_ids.discard("")

    manifests = store.list_run_clone_manifests()
    gc_plan = store.plan_garbage_collection(active_run_ids=active_run_ids)
    gc_by_run_id = {candidate.run_id: candidate for candidate in gc_plan.run_candidates}
    unmatched_gc_run_ids = set(gc_by_run_id)

    def _match_reconciliation(
        *,
        run_manifest_path: str | None,
        run_id: str,
        template_id: str | None,
    ) -> dict[str, object] | None:
        if run_manifest_path:
            for item in reconciliation_items:
                if str(item.get("run_manifest_path") or "").strip() == run_manifest_path:
                    return item
        for item in reconciliation_items:
            if str(item.get("run_id") or "").strip() == run_id:
                return item
        if template_id:
            for item in reconciliation_items:
                if template_id in {
                    str(item.get("template_id") or "").strip(),
                    str(item.get("helper_template_id") or "").strip(),
                    str(item.get("persisted_template_id") or "").strip(),
                }:
                    return item
        return None

    items: list[dict[str, object]] = []
    for manifest in manifests:
        manifest_path = str(Path(root_text) / "runs" / manifest.run_id / "manifest.json")
        match = _match_reconciliation(
            run_manifest_path=manifest_path,
            run_id=manifest.run_id,
            template_id=manifest.template_id,
        )
        candidate = gc_by_run_id.get(manifest.run_id)
        if candidate is not None:
            unmatched_gc_run_ids.discard(manifest.run_id)
        items.append(
            {
                "run_id": manifest.run_id,
                "template_id": manifest.template_id,
                "run_manifest_path": manifest_path,
                "run_manifest_present": True,
                "gc_reason": (candidate.reason if candidate is not None else None),
                "gc_path": (candidate.path if candidate is not None else None),
                "matched_vm_id": (str(match.get("vm_id") or "").strip() or None) if match else None,
                "matched_reconciliation_status": (str(match.get("status") or "").strip() or None) if match else None,
                "matched_reconciliation_reason": (str(match.get("reason") or "").strip() or None) if match else None,
            }
        )

    for run_id in sorted(unmatched_gc_run_ids):
        candidate = gc_by_run_id[run_id]
        match = _match_reconciliation(run_manifest_path=None, run_id=run_id, template_id=candidate.template_id)
        items.append(
            {
                "run_id": run_id,
                "template_id": candidate.template_id,
                "run_manifest_path": None,
                "run_manifest_present": False,
                "gc_reason": candidate.reason,
                "gc_path": candidate.path,
                "matched_vm_id": (str(match.get("vm_id") or "").strip() or None) if match else None,
                "matched_reconciliation_status": (str(match.get("status") or "").strip() or None) if match else None,
                "matched_reconciliation_reason": (str(match.get("reason") or "").strip() or None) if match else None,
            }
        )

    return {
        "configured": True,
        "root_path": root_text,
        "registered_templates": len(store.list_templates()),
        "run_manifests": len(manifests),
        "gc_candidates": len(gc_plan.run_candidates),
        "items": sorted(items, key=lambda item: str(item.get("run_id") or "")),
        "reasons": [],
    }


def collect_macos_diagnostics(orchestrator: Any | None = None) -> dict[str, Any]:
    """Aggregate host, helper, template, and runtime diagnostics for admin callers."""

    host = probe_host()
    helper = probe_helper()
    templates = probe_templates()
    runtime_preflights = collect_runtime_preflights(network_policy="deny_all")
    runtimes = probe_runtime_statuses(runtime_preflights=runtime_preflights)
    reconciliation = probe_reconciliation(orchestrator)
    image_store = probe_image_store(reconciliation)
    return {
        "host": host,
        "helper": helper,
        "templates": templates,
        "runtimes": runtimes,
        "reconciliation": reconciliation,
        "image_store": image_store,
    }

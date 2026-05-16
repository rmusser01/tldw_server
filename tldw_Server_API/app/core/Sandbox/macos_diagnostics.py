"""Shared, side-effect-free diagnostics for macOS sandbox runtimes."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.testing import is_truthy

from .image_store import ImageStoreValidationError, SandboxImageStore
from .macos_virtualization.helper_client import (
    MacOSVirtualizationHelperClient,
    MacOSVirtualizationHelperFailure,
    MacOSVirtualizationHelperProtocolError,
    MacOSVirtualizationHelperUnavailable,
)
from .models import RuntimeType
from .runners.vz_common import vz_host_facts
from .runtime_capabilities import RuntimePreflightResult, collect_runtime_preflights
from .vz_reconciliation import (
    REASON_HELPER_FAILURE,
    REASON_HELPER_UNAVAILABLE,
    REASON_PROTOCOL_MISMATCH,
    REASON_RECONCILIATION_UNAVAILABLE,
    collect_vz_reconciliation,
)

_VZ_LINUX_TEMPLATE_MISSING_REASON = "vz_linux_template_missing"
_VZ_MACOS_TEMPLATE_MISSING_REASON = "macos_template_missing"
_VZ_LINUX_OBSERVABILITY_UNAVAILABLE_REASON = "vz_linux_observability_unavailable"
_MACOS_RECONCILIATION_REPAIR_ENDPOINT = "/api/v1/sandbox/admin/macos-reconciliation/repair"
_MACOS_IMAGE_STORE_CLEANUP_PLAN_ENDPOINT = "/api/v1/sandbox/admin/macos-image-store/cleanup-plan"
_HELPER_LOG_DIR_ENV = "TLDW_SANDBOX_MACOS_HELPER_LOG_DIR"
_VZ_LINUX_SERIAL_LOG_DIR_ENV = "TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR"
_VZ_LINUX_RESOURCE_DETAIL_KEYS = (
    "cpu_count",
    "cpu_time_sec",
    "wall_time_sec",
    "memory_size_mb",
    "peak_rss_mb",
    "memory_rss_mb",
    "disk_read_bytes",
    "disk_write_bytes",
    "network_rx_bytes",
    "network_tx_bytes",
)


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


def _path_from_text(value: str | None) -> Path | None:
    """Parse operator-provided path text without allowing invalid path payloads to escape."""

    raw = str(value or "").strip()
    if not raw or "\x00" in raw:
        return None
    try:
        return Path(raw).expanduser()
    except (OSError, ValueError):
        return None


def _file_pointer(path: Path | None) -> dict[str, object]:
    """Return existence and size metadata for a file without reading file contents."""

    if path is None:
        return {"path": None, "exists": False, "size_bytes": None}
    path_text = str(path)
    try:
        if not path.is_file():
            return {"path": path_text, "exists": False, "size_bytes": None}
        return {"path": path_text, "exists": True, "size_bytes": int(path.stat().st_size)}
    except (OSError, ValueError):
        return {"path": path_text, "exists": False, "size_bytes": None}


def _sanitize_serial_log_component(value: str) -> str:
    """Convert a helper VM id into the serial-log filename component used by helperctl."""

    sanitized = "".join(
        char if (char.isalnum() or char in "._-") else "_"
        for char in str(value or "")
    )
    return sanitized or "vm"


def _serial_log_pointer(vm_id: str, serial_log_dir: Path | None) -> dict[str, object]:
    """Build a read-only serial log pointer for one VM id."""

    if serial_log_dir is None:
        return _file_pointer(None)
    return _file_pointer(serial_log_dir / f"{_sanitize_serial_log_component(vm_id)}.serial.log")


def _detail_text(details: dict[str, object], key: str) -> str | None:
    """Extract a non-empty string detail from helper metadata."""

    value = details.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _detail_bool(details: dict[str, object], key: str) -> bool | None:
    """Coerce helper detail booleans without trusting arbitrary truthy strings."""

    value = details.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return None


def _detail_csv(details: dict[str, object], key: str) -> list[str]:
    """Extract a list-like helper detail from either a list or comma-separated string."""

    value = details.get(key)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _detail_int(details: dict[str, object], key: str) -> int | None:
    """Extract an integer helper detail while rejecting booleans."""

    value = details.get(key)
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _guest_observability(details: dict[str, object]) -> dict[str, object]:
    """Project helper guest-agent details into the admin observability schema."""

    return {
        "version": _detail_text(details, "guest_version"),
        "workspace_root": _detail_text(details, "guest_workspace_root"),
        "capabilities_known": _detail_bool(details, "guest_capabilities_known"),
        "capabilities": _detail_csv(details, "guest_capabilities"),
    }


def _resource_snapshot(details: dict[str, object]) -> dict[str, int]:
    """Project allowlisted helper resource counters into diagnostics."""

    snapshot: dict[str, int] = {}
    for key in _VZ_LINUX_RESOURCE_DETAIL_KEYS:
        value = _detail_int(details, key)
        if value is not None:
            snapshot[key] = value
    return snapshot


def _count_list(payload: dict[str, object], key: str) -> int:
    """Return a count for list-like diagnostics fields."""

    value = payload.get(key)
    if isinstance(value, list | tuple | set):
        return len(value)
    return 0


def _count_int(payload: dict[str, object], key: str) -> int:
    """Return a non-negative integer for diagnostics count fields."""

    value = payload.get(key)
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def summarize_recovery(
    *,
    reconciliation: dict[str, object] | None,
    image_store: dict[str, object] | None,
    observability: dict[str, object] | None,
) -> dict[str, object]:
    """Summarize already-collected diagnostics into operator recovery posture."""

    reconciliation_payload = dict(reconciliation or {})
    image_store_payload = dict(image_store or {})
    observability_payload = dict(observability or {})
    reasons = [
        str(reason).strip()
        for reason in list(reconciliation_payload.get("reasons") or [])
        if str(reason).strip()
    ]
    counts = {
        "persisted_session_controls": _count_int(reconciliation_payload, "persisted_sessions"),
        "healthy_session_controls": _count_list(reconciliation_payload, "healthy_session_ids"),
        "stale_session_controls": _count_list(reconciliation_payload, "stale_session_ids"),
        "unhealthy_session_controls": _count_list(reconciliation_payload, "unhealthy_session_ids"),
        "skipped_active_session_controls": _count_list(reconciliation_payload, "skipped_active_session_ids"),
        "orphaned_vms": _count_list(reconciliation_payload, "orphaned_vm_ids"),
        "owned_orphaned_vms": _count_list(reconciliation_payload, "owned_orphaned_vm_ids"),
        "unknown_orphaned_vms": _count_list(reconciliation_payload, "unknown_orphaned_vm_ids"),
        "foreign_orphaned_vms": _count_list(reconciliation_payload, "foreign_orphaned_vm_ids"),
        "image_store_gc_candidates": _count_int(image_store_payload, "gc_candidates"),
        "live_vms": _count_int(reconciliation_payload, "live_vms"),
    }
    notes: list[str] = []
    if not bool(reconciliation_payload.get("computed")) or reasons:
        if reasons:
            notes.append(f"Reconciliation reasons: {', '.join(reasons[:3])}.")
        else:
            notes.append("Reconciliation did not compute.")
        return {
            "status": "unavailable",
            "severity": "error",
            "codes": ["vz_recovery_unavailable"],
            "counts": counts,
            "recommended_action": "Fix helper and runtime diagnostics before running repair.",
            "repair_endpoint": None,
            "cleanup_plan_endpoint": None,
            "notes": notes,
        }

    codes: list[str] = []
    if counts["stale_session_controls"] > 0:
        codes.append("vz_stale_session_controls")
    if counts["unhealthy_session_controls"] > 0:
        codes.append("vz_unhealthy_session_controls")
    if counts["skipped_active_session_controls"] > 0:
        codes.append("vz_active_session_controls_skipped")
        notes.append("Active sessions were skipped by reconciliation and should not be repaired automatically.")
    if counts["owned_orphaned_vms"] > 0:
        codes.append("vz_owned_orphaned_vms")
    if counts["unknown_orphaned_vms"] > 0:
        codes.append("vz_unknown_orphaned_vms")
        notes.append("Unknown orphan VM ownership requires manual inspection before termination.")
    if counts["foreign_orphaned_vms"] > 0:
        codes.append("vz_foreign_orphaned_vms")
        notes.append("Foreign orphan VMs are reported for visibility only.")
    if counts["image_store_gc_candidates"] > 0:
        codes.append("vz_image_store_gc_candidates")

    observability_reasons = [
        str(reason).strip()
        for reason in list(observability_payload.get("reasons") or [])
        if str(reason).strip()
    ]
    if observability_reasons:
        notes.append(f"Observability reasons: {', '.join(observability_reasons[:3])}.")

    repair_needed = any(
        counts[key] > 0
        for key in (
            "stale_session_controls",
            "unhealthy_session_controls",
            "owned_orphaned_vms",
        )
    )
    cleanup_needed = counts["image_store_gc_candidates"] > 0
    inspect_needed = any(
        counts[key] > 0
        for key in (
            "unknown_orphaned_vms",
            "foreign_orphaned_vms",
            "skipped_active_session_controls",
        )
    )
    actionable_recovery_issue = repair_needed or cleanup_needed or inspect_needed

    if not actionable_recovery_issue:
        recommended_action = "No recovery action needed."
    elif repair_needed and cleanup_needed:
        recommended_action = "Run macOS reconciliation repair in dry-run mode, then inspect the image-store cleanup plan."
    elif repair_needed:
        recommended_action = "Run macOS reconciliation repair in dry-run mode and inspect planned actions."
    elif cleanup_needed:
        recommended_action = "Inspect the macOS image-store cleanup plan before deleting candidates."
    elif inspect_needed:
        recommended_action = "Inspect orphan VM ownership before taking manual action."
    else:
        recommended_action = "Review macOS sandbox diagnostics before taking recovery action."

    return {
        "status": "action_recommended" if actionable_recovery_issue else "healthy",
        "severity": "warning" if actionable_recovery_issue else "ok",
        "codes": codes,
        "counts": counts,
        "recommended_action": recommended_action,
        "repair_endpoint": _MACOS_RECONCILIATION_REPAIR_ENDPOINT if repair_needed else None,
        "cleanup_plan_endpoint": _MACOS_IMAGE_STORE_CLEANUP_PLAN_ENDPOINT if cleanup_needed else None,
        "notes": notes,
    }


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


def _fetch_live_vms_for_diagnostics() -> tuple[Any | None, str | None, str | None]:
    """Fetch helper VM state once and return reconciliation/observability failure reasons."""

    try:
        return MacOSVirtualizationHelperClient().list_vms(), None, None
    except MacOSVirtualizationHelperUnavailable:
        return None, REASON_HELPER_UNAVAILABLE, REASON_HELPER_UNAVAILABLE
    except MacOSVirtualizationHelperProtocolError:
        return None, REASON_PROTOCOL_MISMATCH, REASON_PROTOCOL_MISMATCH
    except MacOSVirtualizationHelperFailure as exc:
        logger.debug("VZ helper returned failure while listing VMs for diagnostics: {}", exc.error_code)
        return None, REASON_HELPER_FAILURE, REASON_HELPER_FAILURE
    except Exception as exc:  # noqa: BLE001
        # Diagnostics must remain best-effort when helper payload parsing changes unexpectedly.
        logger.debug("Unable to collect VZ helper VM list for diagnostics: {}", exc)
        return None, REASON_RECONCILIATION_UNAVAILABLE, _VZ_LINUX_OBSERVABILITY_UNAVAILABLE_REASON


def probe_reconciliation(
    orchestrator: Any | None = None,
    *,
    live: Any | None = None,
    live_failure_reason: str | None = None,
) -> dict[str, object]:
    """Compare persisted VZ session rows with live helper VM state when available."""

    return collect_vz_reconciliation(
        orchestrator,
        live=live,
        live_failure_reason=live_failure_reason,
    )


def probe_vz_linux_observability(
    *,
    live: Any | None = None,
    live_failure_reason: str | None = None,
) -> dict[str, object]:
    """Report read-only helper log pointers and live VM diagnostics."""

    raw_serial_log_dir = os.getenv(_VZ_LINUX_SERIAL_LOG_DIR_ENV)
    raw_helper_log_dir = os.getenv(_HELPER_LOG_DIR_ENV)
    serial_log_dir = _path_from_text(raw_serial_log_dir)
    helper_log_source = "env"
    helper_log_dir = _path_from_text(raw_helper_log_dir)
    configured = serial_log_dir is not None or helper_log_dir is not None
    if helper_log_dir is None and serial_log_dir is not None and serial_log_dir.name == "serial":
        helper_log_dir = serial_log_dir.parent
        helper_log_source = "inferred_from_serial_log_dir"
    elif helper_log_dir is None:
        helper_log_dir = Path.home() / "Library" / "Logs" / "tldw" / "macos-vz-helper"
        helper_log_source = "default"

    helper_logs = {
        "stdout": _file_pointer(helper_log_dir / "helper.stdout.log"),
        "stderr": _file_pointer(helper_log_dir / "helper.stderr.log"),
    }
    report: dict[str, object] = {
        "configured": configured,
        "serial_log_dir": str(serial_log_dir) if serial_log_dir is not None else None,
        "helper_log_dir": str(helper_log_dir) if helper_log_dir is not None else None,
        "helper_log_dir_source": helper_log_source,
        "helper_logs": helper_logs,
        "live_vms": 0,
        "vms": [],
        "reasons": [],
    }

    if live_failure_reason:
        report["reasons"] = [live_failure_reason]
        return report
    if live is None:
        live, _, observability_failure_reason = _fetch_live_vms_for_diagnostics()
        if observability_failure_reason:
            report["reasons"] = [observability_failure_reason]
            return report

    vm_items: list[dict[str, object]] = []
    for vm in list(getattr(live, "vms", None) or []):
        vm_id = str(getattr(vm, "vm_id", "") or "").strip()
        if not vm_id:
            continue
        metadata = getattr(vm, "metadata", None)
        details_raw = getattr(vm, "details", None)
        details = dict(details_raw) if isinstance(details_raw, dict) else {}
        vm_items.append(
            {
                "vm_id": vm_id,
                "state": str(getattr(vm, "state", "") or "").strip() or None,
                "healthy": bool(getattr(vm, "healthy", False)),
                "run_id": str(getattr(metadata, "run_id", "") or "").strip() or None,
                "session_id": str(getattr(metadata, "session_id", "") or "").strip() or None,
                "session_mode": bool(getattr(metadata, "session_mode", False)),
                "serial_log": _serial_log_pointer(vm_id, serial_log_dir),
                "guest": _guest_observability(details),
                "resource_snapshot": _resource_snapshot(details),
            }
        )

    report["vms"] = sorted(vm_items, key=lambda item: str(item.get("vm_id") or ""))
    report["live_vms"] = len(vm_items)
    return report


def _image_store_template_item(record: Any) -> dict[str, object]:
    """Return read-only admin diagnostics for one registered image-store template."""

    return {
        "template_id": record.template_id,
        "runtime": record.runtime,
        "template_name": record.template_name,
        "artifact_format": record.artifact_format,
        "source_path": record.source_path,
        "artifact_count": len(record.artifacts),
        "artifact_size_bytes": sum(
            int(artifact.size_bytes) for artifact in record.artifacts
        ),
        "oci_image_ref": record.oci_image_ref,
        "oci_platform": record.oci_platform,
        "oci_manifest_digest": record.oci_manifest_digest,
        "oci_config_digest": record.oci_config_digest,
        "oci_layer_digests": list(record.oci_layer_digests),
        "registry": record.registry,
        "imported_at": record.imported_at,
        "provenance": dict(record.provenance),
    }


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
            "templates": [],
            "items": [],
            "reasons": [],
        }

    root_path = Path(root_text).expanduser()
    try:
        if not root_path.exists():
            return {
                "configured": True,
                "root_path": root_text,
                "registered_templates": 0,
                "run_manifests": 0,
                "gc_candidates": 0,
                "templates": [],
                "items": [],
                "reasons": ["image_store_root_missing"],
            }
        if not root_path.is_dir():
            return {
                "configured": True,
                "root_path": root_text,
                "registered_templates": 0,
                "run_manifests": 0,
                "gc_candidates": 0,
                "templates": [],
                "items": [],
                "reasons": ["image_store_root_not_directory"],
            }
        store = SandboxImageStore(root_path=root_path, create_root=False)
    except (ImageStoreValidationError, OSError, ValueError) as exc:
        return {
            "configured": True,
            "root_path": root_text,
            "registered_templates": 0,
            "run_manifests": 0,
            "gc_candidates": 0,
            "templates": [],
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

    template_records = store.list_templates()
    manifests = store.list_run_clone_manifests()
    gc_plan = store.plan_garbage_collection(active_run_ids=active_run_ids)
    templates = [_image_store_template_item(record) for record in template_records]
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
        "registered_templates": len(template_records),
        "run_manifests": len(manifests),
        "gc_candidates": len(gc_plan.run_candidates),
        "templates": templates,
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
    live, reconciliation_live_failure, observability_live_failure = _fetch_live_vms_for_diagnostics()
    reconciliation = probe_reconciliation(
        orchestrator,
        live=live,
        live_failure_reason=reconciliation_live_failure,
    )
    image_store = probe_image_store(reconciliation)
    observability = probe_vz_linux_observability(
        live=live,
        live_failure_reason=observability_live_failure,
    )
    recovery_summary = summarize_recovery(
        reconciliation=reconciliation,
        image_store=image_store,
        observability=observability,
    )
    return {
        "host": host,
        "helper": helper,
        "templates": templates,
        "runtimes": runtimes,
        "reconciliation": reconciliation,
        "image_store": image_store,
        "observability": observability,
        "recovery_summary": recovery_summary,
    }

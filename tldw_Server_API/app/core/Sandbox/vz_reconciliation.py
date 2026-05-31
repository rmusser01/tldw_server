from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from .macos_virtualization.helper_client import (
    MacOSVirtualizationHelperClient,
    MacOSVirtualizationHelperFailure,
    MacOSVirtualizationHelperProtocolError,
    MacOSVirtualizationHelperUnavailable,
)

REASON_HELPER_UNAVAILABLE = "macos_virtualization_helper_unavailable"
REASON_PROTOCOL_MISMATCH = "macos_virtualization_helper_protocol_mismatch"
REASON_HELPER_FAILURE = "macos_virtualization_helper_failure"
REASON_RECONCILIATION_UNAVAILABLE = "vz_reconciliation_unavailable"
STATUS_OWNED_ORPHAN = "owned_orphaned_vm"
STATUS_UNKNOWN_ORPHAN = "unknown_orphaned_vm"
STATUS_FOREIGN_ORPHAN = "foreign_orphaned_vm"
REASON_OWNED_ORPHAN = "owned_orphan"
REASON_UNKNOWN_OWNERSHIP = "unknown_ownership"
REASON_FOREIGN_OWNER = "foreign_owner"
REASON_IMAGE_STORE_MANIFEST_MISSING = "image_store_manifest_missing"
ORPHAN_STATUSES = {
    STATUS_OWNED_ORPHAN,
    STATUS_UNKNOWN_ORPHAN,
    STATUS_FOREIGN_ORPHAN,
    "orphaned_vm",
}


def _empty_report() -> dict[str, object]:
    return {
        "computed": False,
        "persisted_sessions": 0,
        "live_vms": 0,
        "healthy_session_ids": [],
        "stale_session_ids": [],
        "unhealthy_session_ids": [],
        "skipped_active_session_ids": [],
        "orphaned_vm_ids": [],
        "owned_orphaned_vm_ids": [],
        "unknown_orphaned_vm_ids": [],
        "foreign_orphaned_vm_ids": [],
        "items": [],
        "reasons": [],
    }


def _clean_id(value: object) -> str:
    return str(value or "").strip()


def _append_item(
    items: list[dict[str, object]],
    *,
    status: str,
    session_id: str | None = None,
    vm_id: str | None = None,
    state: str | None = None,
    healthy: bool | None = None,
    reason: str | None = None,
    termination_eligible: bool | None = None,
    item_fields: dict[str, object] | None = None,
) -> None:
    item: dict[str, object] = {"status": status}
    if session_id:
        item["session_id"] = session_id
    if vm_id:
        item["vm_id"] = vm_id
    if state:
        item["state"] = state
    if healthy is not None:
        item["healthy"] = bool(healthy)
    if reason:
        item["reason"] = reason
    if termination_eligible is not None:
        item["termination_eligible"] = bool(termination_eligible)
    if item_fields:
        for key, value in item_fields.items():
            if value is not None:
                item[key] = value
    items.append(item)


def _sort_items(items: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        items,
        key=lambda item: (
            str(item.get("status") or ""),
            str(item.get("session_id") or ""),
            str(item.get("vm_id") or ""),
        ),
    )


def _metadata_context(vm: object) -> dict[str, object]:
    metadata = getattr(vm, "metadata", None)
    run_manifest_path = _clean_id(getattr(metadata, "run_manifest_path", ""))
    planning_source = _clean_id(getattr(metadata, "planning_source", ""))
    return {
        "template_id": _clean_id(getattr(metadata, "template_id", "")) or None,
        "run_id": _clean_id(getattr(metadata, "run_id", "")) or None,
        "helper_session_id": _clean_id(getattr(metadata, "session_id", "")) or None,
        "planning_source": planning_source or None,
        "run_manifest_path": run_manifest_path or None,
        "run_manifest_present": None,
    }


def _safe_path_is_file(path_text: str) -> bool:
    try:
        return Path(path_text).is_file()
    except (OSError, ValueError):
        return False


def _classify_orphan_vm(vm: object) -> tuple[str, bool, str, bool | None]:
    """Return orphan classification, repair eligibility, reason, and manifest presence."""
    metadata = getattr(vm, "metadata", None)
    owner = _clean_id(getattr(metadata, "owner", ""))
    runtime = _clean_id(getattr(metadata, "runtime", ""))
    if not owner or owner == "unknown" or not runtime:
        return STATUS_UNKNOWN_ORPHAN, False, REASON_UNKNOWN_OWNERSHIP, None
    if owner != "tldw" or runtime != "vz_linux":
        return STATUS_FOREIGN_ORPHAN, False, REASON_FOREIGN_OWNER, None

    run_id = _clean_id(getattr(metadata, "run_id", ""))
    created_at = _clean_id(getattr(metadata, "created_at", ""))
    session_mode = bool(getattr(metadata, "session_mode", False))
    session_id = _clean_id(getattr(metadata, "session_id", ""))
    if not run_id or not created_at or (session_mode and not session_id):
        return STATUS_UNKNOWN_ORPHAN, False, REASON_UNKNOWN_OWNERSHIP, None
    planning_source = _clean_id(getattr(metadata, "planning_source", ""))
    if planning_source == "image_store":
        run_manifest_path = _clean_id(getattr(metadata, "run_manifest_path", ""))
        template_id = _clean_id(getattr(metadata, "template_id", ""))
        if not template_id or not run_manifest_path:
            return STATUS_UNKNOWN_ORPHAN, False, REASON_UNKNOWN_OWNERSHIP, None
        if not _safe_path_is_file(run_manifest_path):
            return STATUS_UNKNOWN_ORPHAN, False, REASON_IMAGE_STORE_MANIFEST_MISSING, False
        return STATUS_OWNED_ORPHAN, True, REASON_OWNED_ORPHAN, True
    return STATUS_OWNED_ORPHAN, True, REASON_OWNED_ORPHAN, None


def collect_vz_reconciliation(
    orchestrator: Any | None,
    *,
    helper_client: Any | None = None,
    active_session_checker: Callable[[str], bool] | None = None,
    live: Any | None = None,
    live_failure_reason: str | None = None,
) -> dict[str, object]:
    """Compare persisted VZ session-control rows with live helper VM state.

    This collector is intentionally read-only: it only lists persisted rows and
    live VMs, then reports mismatches for later explicit repair paths.
    Callers that already fetched helper VM state can pass ``live`` or
    ``live_failure_reason`` to avoid duplicate helper RPCs.
    """

    report = _empty_report()
    if orchestrator is None:
        report["reasons"] = [REASON_RECONCILIATION_UNAVAILABLE]
        return report

    lister = getattr(orchestrator, "list_vz_session_controls", None)
    if not callable(lister):
        report["reasons"] = [REASON_RECONCILIATION_UNAVAILABLE]
        return report

    try:
        persisted_rows = [dict(row) for row in lister() or [] if isinstance(row, dict)]
    except Exception as exc:
        logger.debug("Unable to list VZ session controls for reconciliation: {}", exc)
        report["reasons"] = [REASON_RECONCILIATION_UNAVAILABLE]
        return report

    report["persisted_sessions"] = len(persisted_rows)

    if live_failure_reason:
        report["reasons"] = [live_failure_reason]
        return report
    if live is None:
        client = helper_client if helper_client is not None else MacOSVirtualizationHelperClient()
        try:
            live = client.list_vms()
        except MacOSVirtualizationHelperUnavailable:
            report["reasons"] = [REASON_HELPER_UNAVAILABLE]
            return report
        except MacOSVirtualizationHelperProtocolError:
            report["reasons"] = [REASON_PROTOCOL_MISMATCH]
            return report
        except MacOSVirtualizationHelperFailure as exc:
            logger.debug("VZ helper returned failure for reconciliation: {}", exc.error_code)
            report["reasons"] = [REASON_HELPER_FAILURE]
            return report
        except Exception as exc:
            logger.debug("Unable to collect VZ helper VM list for reconciliation: {}", exc)
            report["reasons"] = [REASON_RECONCILIATION_UNAVAILABLE]
            return report

    live_vm_by_id = {
        vm_id: vm
        for vm in list(getattr(live, "vms", None) or [])
        if (vm_id := _clean_id(getattr(vm, "vm_id", "")))
    }
    persisted_row_by_session = {
        session_id: row
        for row in persisted_rows
        if (session_id := _clean_id(row.get("id")))
    }
    persisted_vm_by_session = {
        session_id: vm_id
        for session_id, row in persisted_row_by_session.items()
        if (vm_id := _clean_id(row.get("vm_id")))
    }
    persisted_vm_ids = set(persisted_vm_by_session.values())

    healthy_session_ids: list[str] = []
    stale_session_ids: list[str] = []
    unhealthy_session_ids: list[str] = []
    skipped_active_session_ids: list[str] = []
    orphaned_vm_ids: list[str] = []
    owned_orphaned_vm_ids: list[str] = []
    unknown_orphaned_vm_ids: list[str] = []
    foreign_orphaned_vm_ids: list[str] = []
    items: list[dict[str, object]] = []

    for session_id, vm_id in persisted_vm_by_session.items():
        is_active_session = active_session_checker is not None and active_session_checker(session_id)
        persisted_row = persisted_row_by_session.get(session_id, {})
        persisted_template_id = _clean_id(persisted_row.get("template_id")) or None
        vm = live_vm_by_id.get(vm_id)
        if vm is None:
            if is_active_session:
                skipped_active_session_ids.append(session_id)
                _append_item(
                    items,
                    status="skipped_active_session",
                    session_id=session_id,
                    vm_id=vm_id,
                    reason="active_session",
                    item_fields={"persisted_template_id": persisted_template_id},
                )
                continue
            stale_session_ids.append(session_id)
            _append_item(
                items,
                status="stale_session",
                session_id=session_id,
                vm_id=vm_id,
                reason="vm_missing",
                item_fields={"persisted_template_id": persisted_template_id},
            )
            continue

        state = _clean_id(getattr(vm, "state", ""))
        healthy = bool(getattr(vm, "healthy", False))
        metadata_fields = _metadata_context(vm)
        helper_template_id = str(metadata_fields.get("template_id") or "").strip() or None
        template_id_matches = None
        if persisted_template_id is not None and helper_template_id is not None:
            template_id_matches = persisted_template_id == helper_template_id
        item_fields = {
            **metadata_fields,
            "persisted_template_id": persisted_template_id,
            "helper_template_id": helper_template_id,
            "template_id_matches_persisted": template_id_matches,
        }
        if healthy:
            healthy_session_ids.append(session_id)
            _append_item(
                items,
                status="healthy",
                session_id=session_id,
                vm_id=vm_id,
                state=state,
                healthy=healthy,
                item_fields=item_fields,
            )
            continue

        if is_active_session:
            skipped_active_session_ids.append(session_id)
            _append_item(
                items,
                status="skipped_active_session",
                session_id=session_id,
                vm_id=vm_id,
                state=state,
                healthy=healthy,
                reason="active_session",
                item_fields=item_fields,
            )
            continue

        unhealthy_session_ids.append(session_id)
        _append_item(
            items,
            status="unhealthy_vm",
            session_id=session_id,
            vm_id=vm_id,
            state=state,
            healthy=healthy,
            reason="vm_unhealthy",
            item_fields=item_fields,
        )

    for vm_id, vm in live_vm_by_id.items():
        if vm_id in persisted_vm_ids:
            continue
        status, termination_eligible, reason, run_manifest_present = _classify_orphan_vm(vm)
        orphaned_vm_ids.append(vm_id)
        if status == STATUS_OWNED_ORPHAN:
            owned_orphaned_vm_ids.append(vm_id)
        elif status == STATUS_FOREIGN_ORPHAN:
            foreign_orphaned_vm_ids.append(vm_id)
        else:
            unknown_orphaned_vm_ids.append(vm_id)
        _append_item(
            items,
            status=status,
            vm_id=vm_id,
            state=_clean_id(getattr(vm, "state", "")),
            healthy=bool(getattr(vm, "healthy", False)),
            reason=reason,
            termination_eligible=termination_eligible,
            item_fields={
                **_metadata_context(vm),
                "run_manifest_present": run_manifest_present,
            },
        )

    report["computed"] = True
    report["live_vms"] = len(live_vm_by_id)
    report["healthy_session_ids"] = sorted(healthy_session_ids)
    report["stale_session_ids"] = sorted(stale_session_ids)
    report["unhealthy_session_ids"] = sorted(unhealthy_session_ids)
    report["skipped_active_session_ids"] = sorted(skipped_active_session_ids)
    report["orphaned_vm_ids"] = sorted(orphaned_vm_ids)
    report["owned_orphaned_vm_ids"] = sorted(owned_orphaned_vm_ids)
    report["unknown_orphaned_vm_ids"] = sorted(unknown_orphaned_vm_ids)
    report["foreign_orphaned_vm_ids"] = sorted(foreign_orphaned_vm_ids)
    report["items"] = _sort_items(items)
    return report

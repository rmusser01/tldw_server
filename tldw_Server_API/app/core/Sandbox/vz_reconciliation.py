from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger

from .macos_virtualization.helper_client import (
    MacOSVirtualizationHelperClient,
    MacOSVirtualizationHelperProtocolError,
    MacOSVirtualizationHelperUnavailable,
)

REASON_HELPER_UNAVAILABLE = "macos_virtualization_helper_unavailable"
REASON_PROTOCOL_MISMATCH = "macos_virtualization_helper_protocol_mismatch"
REASON_RECONCILIATION_UNAVAILABLE = "vz_reconciliation_unavailable"


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


def collect_vz_reconciliation(
    orchestrator: Any | None,
    *,
    helper_client: Any | None = None,
    active_session_checker: Callable[[str], bool] | None = None,
) -> dict[str, object]:
    """Compare persisted VZ session-control rows with live helper VM state.

    This collector is intentionally read-only: it only lists persisted rows and
    live VMs, then reports mismatches for later explicit repair paths.
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

    client = helper_client if helper_client is not None else MacOSVirtualizationHelperClient()
    try:
        live = client.list_vms()
    except MacOSVirtualizationHelperUnavailable:
        report["reasons"] = [REASON_HELPER_UNAVAILABLE]
        return report
    except MacOSVirtualizationHelperProtocolError:
        report["reasons"] = [REASON_PROTOCOL_MISMATCH]
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
    persisted_vm_by_session = {
        session_id: vm_id
        for row in persisted_rows
        if (session_id := _clean_id(row.get("id"))) and (vm_id := _clean_id(row.get("vm_id")))
    }
    persisted_vm_ids = set(persisted_vm_by_session.values())

    healthy_session_ids: list[str] = []
    stale_session_ids: list[str] = []
    unhealthy_session_ids: list[str] = []
    skipped_active_session_ids: list[str] = []
    orphaned_vm_ids: list[str] = []
    items: list[dict[str, object]] = []

    for session_id, vm_id in persisted_vm_by_session.items():
        is_active_session = active_session_checker is not None and active_session_checker(session_id)
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
                )
                continue
            stale_session_ids.append(session_id)
            _append_item(
                items,
                status="stale_session",
                session_id=session_id,
                vm_id=vm_id,
                reason="vm_missing",
            )
            continue

        state = _clean_id(getattr(vm, "state", ""))
        healthy = bool(getattr(vm, "healthy", False))
        if healthy:
            healthy_session_ids.append(session_id)
            _append_item(
                items,
                status="healthy",
                session_id=session_id,
                vm_id=vm_id,
                state=state,
                healthy=healthy,
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
        )

    for vm_id, vm in live_vm_by_id.items():
        if vm_id in persisted_vm_ids:
            continue
        orphaned_vm_ids.append(vm_id)
        _append_item(
            items,
            status="orphaned_vm",
            vm_id=vm_id,
            state=_clean_id(getattr(vm, "state", "")),
            healthy=bool(getattr(vm, "healthy", False)),
            reason="session_missing",
        )

    report["computed"] = True
    report["live_vms"] = len(live_vm_by_id)
    report["healthy_session_ids"] = sorted(healthy_session_ids)
    report["stale_session_ids"] = sorted(stale_session_ids)
    report["unhealthy_session_ids"] = sorted(unhealthy_session_ids)
    report["skipped_active_session_ids"] = sorted(skipped_active_session_ids)
    report["orphaned_vm_ids"] = sorted(orphaned_vm_ids)
    report["items"] = _sort_items(items)
    return report

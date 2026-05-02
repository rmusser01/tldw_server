"""
Startup warning producer for sandbox VZ Linux reconciliation findings.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Sandbox import vz_reconciliation
from tldw_Server_API.app.services.startup_warning_models import StartupWarningRecord
from tldw_Server_API.app.services.startup_warning_registry import StartupWarningRegistry


def produce_sandbox_startup_warnings(
    *,
    orchestrator: Any | None,
    registry: StartupWarningRegistry,
    helper_client: Any | None = None,
    active_session_checker: Callable[[str], bool] | None = None,
) -> list[StartupWarningRecord]:
    """Translate bounded sandbox reconciliation truth into startup warning records."""
    if orchestrator is None:
        return []

    report = vz_reconciliation.collect_vz_reconciliation(
        orchestrator,
        helper_client=helper_client,
        active_session_checker=active_session_checker,
    )
    records: list[StartupWarningRecord] = []

    reasons = set(report.get("reasons") or [])
    if vz_reconciliation.REASON_PROTOCOL_MISMATCH in reasons:
        records.append(
            _record(
                code="vz_helper_protocol_mismatch",
                severity="error",
                startup_action="block_startup",
                summary="Startup detected a macOS virtualization helper protocol mismatch.",
                remediation=(
                    "Rebuild or restart the macOS VZ helper so its protocol matches the "
                    "server before retrying startup."
                ),
                details={"reason": vz_reconciliation.REASON_PROTOCOL_MISMATCH},
            )
        )
    elif vz_reconciliation.REASON_HELPER_UNAVAILABLE in reasons:
        records.append(
            _record(
                code="vz_helper_unavailable_at_startup",
                severity="warning",
                startup_action="warn",
                summary="Startup could not reach the macOS virtualization helper.",
                remediation=(
                    "Verify the helper socket, helper process, and operator workflow before "
                    "relying on vz_linux execution."
                ),
                details={"reason": vz_reconciliation.REASON_HELPER_UNAVAILABLE},
            )
        )
    elif vz_reconciliation.REASON_HELPER_FAILURE in reasons:
        records.append(
            _record(
                code="vz_helper_failure_at_startup",
                severity="warning",
                startup_action="warn",
                summary="Startup could not complete the macOS virtualization helper probe.",
                remediation=(
                    "Inspect helper logs and rerun sandbox diagnostics before relying on "
                    "vz_linux execution."
                ),
                details={"reason": vz_reconciliation.REASON_HELPER_FAILURE},
            )
        )
    elif vz_reconciliation.REASON_RECONCILIATION_UNAVAILABLE in reasons:
        records.append(
            _record(
                code="vz_reconciliation_unavailable_at_startup",
                severity="warning",
                startup_action="warn",
                summary="Startup could not collect vz_linux reconciliation state.",
                remediation=(
                    "Review sandbox diagnostics and startup logs to determine why "
                    "reconciliation state could not be collected."
                ),
                details={"reason": vz_reconciliation.REASON_RECONCILIATION_UNAVAILABLE},
            )
        )
    else:
        records.extend(_count_records(report))

    for record in records:
        registry.add_warning(record)
        _log_record(record)
    return records


def _count_records(report: dict[str, object]) -> list[StartupWarningRecord]:
    records: list[StartupWarningRecord] = []
    stale_count = len(report.get("stale_session_ids") or [])
    unhealthy_count = len(report.get("unhealthy_session_ids") or [])
    orphan_count = len(report.get("orphaned_vm_ids") or [])
    skipped_count = len(report.get("skipped_active_session_ids") or [])

    if stale_count:
        records.append(
            _record(
                code="vz_stale_session_controls_detected",
                severity="warning",
                startup_action="warn",
                summary="Startup detected stale persisted vz_linux session bindings.",
                remediation=(
                    "Review sandbox diagnostics and run explicit reconciliation repair after "
                    "confirming no active work."
                ),
                details={"stale_session_controls": stale_count},
            )
        )
    if unhealthy_count:
        records.append(
            _record(
                code="vz_unhealthy_session_controls_detected",
                severity="warning",
                startup_action="warn",
                summary="Startup detected unhealthy vz_linux session bindings.",
                remediation=(
                    "Inspect helper VM health in sandbox diagnostics, then run explicit "
                    "reconciliation repair if the sessions are no longer active."
                ),
                details={"unhealthy_session_controls": unhealthy_count},
            )
        )
    if orphan_count:
        records.append(
            _record(
                code="vz_orphaned_vms_detected",
                severity="warning",
                startup_action="warn",
                summary="Startup detected orphaned vz_linux helper VMs.",
                remediation=(
                    "Inspect orphan ownership in sandbox diagnostics before using explicit "
                    "reconciliation repair or orphan termination."
                ),
                details={
                    "orphaned_vms": orphan_count,
                    "owned_orphaned_vms": len(report.get("owned_orphaned_vm_ids") or []),
                    "unknown_orphaned_vms": len(
                        report.get("unknown_orphaned_vm_ids") or []
                    ),
                    "foreign_orphaned_vms": len(
                        report.get("foreign_orphaned_vm_ids") or []
                    ),
                },
            )
        )
    if skipped_count:
        records.append(
            _record(
                code="vz_skipped_active_reconciliation_items_detected",
                severity="warning",
                startup_action="warn",
                summary="Startup detected active vz_linux sessions that were skipped during reconciliation.",
                remediation=(
                    "Check the active session inventory before attempting any explicit "
                    "reconciliation repair."
                ),
                details={"skipped_active_sessions": skipped_count},
            )
        )
    return records


def _record(
    *,
    code: str,
    severity: str,
    startup_action: str,
    summary: str,
    remediation: str,
    details: dict[str, object],
) -> StartupWarningRecord:
    return StartupWarningRecord(
        component="sandbox.vz_linux",
        severity=severity,
        startup_action=startup_action,
        code=code,
        summary=summary,
        remediation=remediation,
        details=details,
    )


def _log_record(record: StartupWarningRecord) -> None:
    bound_logger = logger.bind(
        startup_warning=True,
        component=record.component,
        code=record.code,
        startup_action=record.startup_action,
        details=record.details,
    )
    if record.startup_action == "block_startup":
        bound_logger.error(record.summary)
        return
    bound_logger.warning(record.summary)

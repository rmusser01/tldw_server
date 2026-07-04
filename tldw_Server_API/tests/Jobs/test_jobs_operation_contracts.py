"""Contract tests for Jobs operation result dataclasses."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AdmissionRejectionReason,
    AdmissionResult,
    CreateJobCommand,
    LifecycleResult,
    NoTransitionReason,
    OperationOutcome,
)


def test_create_job_command_carries_public_job_facts() -> None:
    """Verify create commands carry public job facts without transformation."""

    command = CreateJobCommand(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"a": 1},
        owner_user_id="user-1",
        idempotency_key="same",
        priority=5,
        max_retries=2,
        request_id="request-1",
        trace_id="trace-1",
    )

    assert command.domain == "chatbooks"
    assert command.queue == "default"
    assert command.payload == {"a": 1}
    assert command.request_id == "request-1"
    assert command.trace_id == "trace-1"


def test_admission_result_distinguishes_inserted_and_existing_rows() -> None:
    """Verify admission results distinguish inserted, existing, and rejected states."""

    inserted = AdmissionResult.applied(
        row={"id": 1, "status": "queued"},
        durable_events=({"event_type": "job.created"},),
    )
    existing = AdmissionResult.existing(row={"id": 1, "status": "queued"})
    rejected = AdmissionResult.rejected(AdmissionRejectionReason.QUEUE_PAUSED)

    assert inserted.outcome is OperationOutcome.APPLIED
    assert inserted.inserted is True
    assert existing.outcome is OperationOutcome.NO_TRANSITION
    assert existing.inserted is False
    assert existing.no_transition_reason is NoTransitionReason.IDEMPOTENT_EXISTING
    assert rejected.outcome is OperationOutcome.ADMISSION_REJECTED
    assert rejected.admission_rejection_reason is AdmissionRejectionReason.QUEUE_PAUSED


def test_admission_result_rejects_inconsistent_states() -> None:
    """Verify invalid admission result state combinations are rejected."""

    with pytest.raises(ValueError, match="applied admission"):
        AdmissionResult(outcome=OperationOutcome.APPLIED, was_inserted=False)

    with pytest.raises(ValueError, match="applied admission"):
        AdmissionResult(outcome=OperationOutcome.APPLIED, was_inserted=True)

    with pytest.raises(ValueError, match="no-transition admission"):
        AdmissionResult(outcome=OperationOutcome.NO_TRANSITION)

    with pytest.raises(ValueError, match="rejected admission"):
        AdmissionResult(outcome=OperationOutcome.ADMISSION_REJECTED)

    with pytest.raises(ValueError, match="only applied admission"):
        AdmissionResult(
            outcome=OperationOutcome.NO_TRANSITION,
            was_inserted=True,
            no_transition_reason=NoTransitionReason.IDEMPOTENT_EXISTING,
        )

    with pytest.raises(ValueError, match="only applied admission"):
        AdmissionResult(
            outcome=OperationOutcome.NO_TRANSITION,
            no_transition_reason=NoTransitionReason.IDEMPOTENT_EXISTING,
            durable_events=({"event_type": "job.noop"},),
        )


def test_admission_result_copies_mutable_facts() -> None:
    """Verify admission results deep-copy nested mutable facts."""

    row = {"id": 1, "status": "queued", "payload": {"topic": "original"}}
    event = {"event_type": "job.created", "attrs": {"attempt": 1}}
    events = [event]

    result = AdmissionResult.applied(row=row, durable_events=events)
    row["status"] = "failed"
    row["payload"]["topic"] = "mutated"
    event["event_type"] = "job.failed"
    event["attrs"]["attempt"] = 2
    events.append({"event_type": "job.completed"})

    assert result.row == {"id": 1, "status": "queued", "payload": {"topic": "original"}}
    assert result.durable_events == ({"event_type": "job.created", "attrs": {"attempt": 1}},)


def test_lifecycle_result_names_no_transition_reason() -> None:
    """Verify lifecycle no-transition results retain their reason."""

    result = LifecycleResult.no_transition(
        NoTransitionReason.STALE_LEASE,
        row={"id": 1, "status": "processing"},
    )

    assert result.outcome is OperationOutcome.NO_TRANSITION
    assert result.no_transition_reason is NoTransitionReason.STALE_LEASE
    assert result.transition_applied is False


def test_lifecycle_result_rejects_inconsistent_states() -> None:
    """Verify invalid lifecycle result state combinations are rejected."""

    with pytest.raises(ValueError, match="applied lifecycle"):
        LifecycleResult(outcome=OperationOutcome.APPLIED, transition_applied=False)

    with pytest.raises(ValueError, match="applied lifecycle"):
        LifecycleResult(outcome=OperationOutcome.APPLIED, transition_applied=True)

    with pytest.raises(ValueError, match="no-transition lifecycle"):
        LifecycleResult(outcome=OperationOutcome.NO_TRANSITION, transition_applied=False)

    with pytest.raises(ValueError, match="only applied lifecycle"):
        LifecycleResult(
            outcome=OperationOutcome.NO_TRANSITION,
            transition_applied=True,
            no_transition_reason=NoTransitionReason.STALE_LEASE,
        )

    with pytest.raises(ValueError, match="only applied lifecycle"):
        LifecycleResult(
            outcome=OperationOutcome.NO_TRANSITION,
            transition_applied=False,
            no_transition_reason=NoTransitionReason.STALE_LEASE,
            durable_events=({"event_type": "job.noop"},),
        )


def test_lifecycle_result_copies_mutable_facts() -> None:
    """Verify lifecycle results deep-copy nested mutable facts."""

    row = {"id": 1, "status": "processing", "result": {"ok": True}}
    event = {"event_type": "job.completed", "attrs": {"attempt": 1}}
    events = [event]

    result = LifecycleResult.applied(row=row, durable_events=events)
    row["status"] = "failed"
    row["result"]["ok"] = False
    event["event_type"] = "job.failed"
    event["attrs"]["attempt"] = 2
    events.append({"event_type": "job.cancelled"})

    assert result.row == {"id": 1, "status": "processing", "result": {"ok": True}}
    assert result.durable_events == ({"event_type": "job.completed", "attrs": {"attempt": 1}},)


def test_operation_contracts_do_not_import_job_manager() -> None:
    """Verify operation contracts do not depend on the JobManager facade."""

    operations_dir = Path(__file__).resolve().parents[3] / "tldw_Server_API/app/core/Jobs/operations"
    assert operations_dir.exists()
    forbidden_refs: list[str] = []
    scanned_files = 0

    for path in operations_dir.rglob("*.py"):
        scanned_files += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[-1] == "manager" or alias.name.endswith(".Jobs.manager"):
                        forbidden_refs.append(f"{path}:{alias.name}")
                    if alias.asname in {"manager", "JobManager"}:
                        forbidden_refs.append(f"{path}:{alias.asname}")
            if isinstance(node, ast.ImportFrom):
                if node.module and (node.module.split(".")[-1] == "manager" or node.module.endswith(".Jobs.manager")):
                    forbidden_refs.append(f"{path}:{node.module}")
                for alias in node.names:
                    if alias.name in {"manager", "JobManager"}:
                        forbidden_refs.append(f"{path}:{alias.name}")
                    if alias.asname in {"manager", "JobManager"}:
                        forbidden_refs.append(f"{path}:{alias.asname}")
            if isinstance(node, ast.Name):
                if node.id == "JobManager":
                    forbidden_refs.append(f"{path}:{node.id}")
            if isinstance(node, ast.Attribute):
                if node.attr in {"manager", "JobManager"}:
                    forbidden_refs.append(f"{path}:{node.attr}")

    assert scanned_files > 0
    assert forbidden_refs == []

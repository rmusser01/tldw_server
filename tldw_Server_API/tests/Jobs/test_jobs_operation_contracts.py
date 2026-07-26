"""Contract tests for Jobs operation result dataclasses."""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AcquireJobCommand,
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


def test_admission_existing_can_report_idempotent_durable_event() -> None:
    """Verify idempotent create replays can report the persisted replay event."""

    result = AdmissionResult.existing(
        row={"id": 1, "status": "queued"},
        durable_events=({"event_type": "job.created", "attrs": {"idempotent": True}},),
    )

    assert result.outcome is OperationOutcome.NO_TRANSITION
    assert result.no_transition_reason is NoTransitionReason.IDEMPOTENT_EXISTING
    assert result.durable_events == ({"event_type": "job.created", "attrs": {"idempotent": True}},)


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

    with pytest.raises(ValueError, match="durable events"):
        AdmissionResult(
            outcome=OperationOutcome.NO_TRANSITION,
            no_transition_reason=NoTransitionReason.MISSING,
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


def test_acquire_job_command_preserves_all_public_job_facts() -> None:
    """Verify acquisition commands preserve every caller-provided field."""

    command = AcquireJobCommand(
        domain="chatbooks",
        queue="priority",
        lease_seconds=45,
        worker_id="worker-1",
        lease_id="lease-1",
        owner_user_id="user-1",
        job_type="export",
        max_inflight_quota=3,
        priority_direction="DESC",
        tie_break="lifo",
        single_update=True,
    )

    assert command.domain == "chatbooks"
    assert command.queue == "priority"
    assert command.lease_seconds == 45
    assert command.worker_id == "worker-1"
    assert command.lease_id == "lease-1"
    assert command.owner_user_id == "user-1"
    assert command.job_type == "export"
    assert command.max_inflight_quota == 3
    assert command.priority_direction == "DESC"
    assert command.tie_break == "lifo"
    assert command.single_update is True


def test_acquire_job_command_is_frozen() -> None:
    """Verify acquisition command fields cannot be reassigned."""

    command = AcquireJobCommand(
        domain="chatbooks",
        queue="default",
        lease_seconds=30,
        worker_id="worker-1",
        lease_id="lease-1",
    )

    with pytest.raises(FrozenInstanceError):
        command.queue = "other"


def test_acquire_job_command_defaults_to_backend_ordering() -> None:
    """Verify omitted tie-breaking delegates ordering to the backend default."""

    command = AcquireJobCommand(
        domain="chatbooks",
        queue="default",
        lease_seconds=30,
        worker_id="worker-1",
        lease_id="lease-1",
    )

    assert command.tie_break is None


def test_acquire_job_command_rejects_invalid_ordering_values() -> None:
    """Verify acquisition ordering controls accept only documented values."""

    with pytest.raises(ValueError, match="priority_direction must be ASC or DESC"):
        AcquireJobCommand(
            domain="chatbooks",
            queue="default",
            lease_seconds=30,
            worker_id="worker-1",
            lease_id="lease-1",
            priority_direction="invalid",
        )

    with pytest.raises(ValueError, match="tie_break must be fifo, lifo, or None"):
        AcquireJobCommand(
            domain="chatbooks",
            queue="default",
            lease_seconds=30,
            worker_id="worker-1",
            lease_id="lease-1",
            tie_break="invalid",
        )


def test_acquire_job_command_rejects_non_positive_lease_duration() -> None:
    """Verify acquisition leases must have a positive duration."""

    for lease_seconds in (0, -1):
        with pytest.raises(ValueError, match="lease_seconds must be positive"):
            AcquireJobCommand(
                domain="chatbooks",
                queue="default",
                lease_seconds=lease_seconds,
                worker_id="worker-1",
                lease_id="lease-1",
            )


def test_lifecycle_result_supports_no_eligible_job_reason() -> None:
    """Verify acquisition can report that no eligible job was available."""

    result = LifecycleResult.no_transition(NoTransitionReason.NO_ELIGIBLE_JOB)

    assert result.outcome is OperationOutcome.NO_TRANSITION
    assert result.no_transition_reason is NoTransitionReason.NO_ELIGIBLE_JOB


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


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_acquire_sql_does_not_interpolate_query_fragments(backend: str) -> None:
    """Verify acquisition SQL uses fixed fragments for ordering and candidate queries."""

    path = (
        Path(__file__).resolve().parents[3]
        / "tldw_Server_API/app/core/Jobs/operations"
        / backend
        / "lifecycle.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    formatted: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            expressions = [value.value for value in node.values if isinstance(value, ast.FormattedValue)]
            lease_parameter_only = expressions and all(
                isinstance(expression, ast.Attribute)
                and isinstance(expression.value, ast.Name)
                and expression.value.id == "command"
                and expression.attr == "lease_seconds"
                for expression in expressions
            )
            if not lease_parameter_only:
                formatted.append((node.lineno, "f-string"))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"format", "format_map"}
        ):
            formatted.append((node.lineno, node.func.attr))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            formatted.append((node.lineno, "percent-format"))

    assert formatted == []


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_acquire_sql_has_no_bandit_query_suppressions(backend: str) -> None:
    """Verify acquisition SQL does not rely on Bandit query suppressions."""

    path = (
        Path(__file__).resolve().parents[3]
        / "tldw_Server_API/app/core/Jobs/operations"
        / backend
        / "lifecycle.py"
    )

    source = path.read_text(encoding="utf-8")
    comments = [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and "nosec" in token.string.casefold()
    ]

    assert comments == []


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

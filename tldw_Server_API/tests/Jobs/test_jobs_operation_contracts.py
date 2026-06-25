from __future__ import annotations

import ast
from pathlib import Path

from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AdmissionRejectionReason,
    AdmissionResult,
    CreateJobCommand,
    LifecycleResult,
    NoTransitionReason,
    OperationOutcome,
)


def test_create_job_command_carries_public_job_facts():
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


def test_admission_result_distinguishes_inserted_and_existing_rows():
    inserted = AdmissionResult.inserted(
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


def test_lifecycle_result_names_no_transition_reason():
    result = LifecycleResult.no_transition(
        NoTransitionReason.STALE_LEASE,
        row={"id": 1, "status": "processing"},
    )

    assert result.outcome is OperationOutcome.NO_TRANSITION
    assert result.no_transition_reason is NoTransitionReason.STALE_LEASE
    assert result.transition_applied is False


def test_operation_contracts_do_not_import_job_manager():
    path = Path("tldw_Server_API/app/core/Jobs/operations/contracts.py")
    tree = ast.parse(path.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    assert "tldw_Server_API.app.core.Jobs.manager" not in imports
    assert not any(import_name.endswith(".Jobs.manager") for import_name in imports)

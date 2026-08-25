"""SQLite coverage for durable Jobs idempotency receipts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.Jobs.operations import contracts

pytestmark = pytest.mark.unit


def _receipt_contracts():
    required = (
        "IdempotentOperationAdmission",
        "IdempotentOperationCommand",
        "IdempotentOperationConflict",
        "IdempotentOperationConflictReason",
        "IdempotentOperationDisposition",
    )
    missing = [name for name in required if not hasattr(contracts, name)]
    assert not missing, f"missing durable idempotency contracts: {missing}"
    return tuple(getattr(contracts, name) for name in required)


def _create_job_command(*, owner_user_id: str | None = "recipient-1"):
    return contracts.CreateJobCommand(
        domain="sharing",
        queue="workspace-clone",
        job_type="workspace_clone",
        payload={"schema_version": 1},
        owner_user_id=owner_user_id,
        batch_group="share:share-1",
        max_retries=0,
    )


def _operation_command(**overrides):
    (
        _,
        command_type,
        _,
        _,
        _,
    ) = _receipt_contracts()
    values = {
        "job": _create_job_command(),
        "key_digest": "a" * 64,
        "request_fingerprint": "b" * 64,
        "operation_scope": "share:share-1",
        "receipt_expires_at": datetime.now(timezone.utc) + timedelta(days=30),
    }
    values.update(overrides)
    return command_type(**values)


@pytest.mark.parametrize("owner_user_id", (None, "", "   "))
def test_idempotent_operation_command_requires_owner(owner_user_id):
    with pytest.raises(ValueError, match="owner_user_id"):
        _operation_command(job=_create_job_command(owner_user_id=owner_user_id))


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("key_digest", "a" * 63),
        ("key_digest", "A" * 64),
        ("key_digest", "g" * 64),
        ("request_fingerprint", "b" * 65),
        ("request_fingerprint", "B" * 64),
    ),
)
def test_idempotent_operation_command_requires_sha256_hex_digests(
    field_name,
    invalid_value,
):
    with pytest.raises(ValueError, match=field_name):
        _operation_command(**{field_name: invalid_value})


@pytest.mark.parametrize("operation_scope", ("", "   ", "x" * 201, "share:\N{SNOWMAN}"))
def test_idempotent_operation_command_requires_bounded_ascii_scope(operation_scope):
    with pytest.raises(ValueError, match="operation_scope"):
        _operation_command(operation_scope=operation_scope)


def test_idempotent_operation_command_requires_timezone_aware_expiry():
    with pytest.raises(ValueError, match="receipt_expires_at"):
        _operation_command(receipt_expires_at=datetime.now())


def test_idempotent_operation_admission_defensively_copies_job_row():
    admission_type, _, _, _, disposition_type = _receipt_contracts()
    row = {"uuid": "job-1", "status": "queued", "payload": {"schema_version": 1}}

    admission = admission_type.created(row)
    row["status"] = "failed"
    row["payload"]["schema_version"] = 2

    assert admission.disposition is disposition_type.CREATED
    assert admission.job == {
        "uuid": "job-1",
        "status": "queued",
        "payload": {"schema_version": 1},
    }


def test_idempotent_operation_conflict_exposes_bounded_reason_and_job_uuid():
    _, _, conflict_type, reason_type, _ = _receipt_contracts()

    conflict = conflict_type(reason_type.KEY_REUSED, job_uuid="job-1")

    assert str(conflict) == "idempotency_key_reused"
    assert conflict.reason is reason_type.KEY_REUSED
    assert conflict.job_uuid == "job-1"

from __future__ import annotations

from uuid import uuid4

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AdmissionRejectionReason,
    AdmissionResult,
    ExpiredLeasePolicy,
    NoTransitionReason,
    OperationOutcome,
)

pytestmark = pytest.mark.unit


def _canonical_kwargs(*, key: str) -> dict:
    delivery_id = str(uuid4())
    return {
        "domain": "admin_webhooks",
        "queue": "delivery",
        "job_type": "admin_webhook_delivery",
        "payload": {"delivery_id": delivery_id},
        "owner_user_id": None,
        "idempotency_key": f"admin-webhook-delivery:{key}:{delivery_id}",
        "max_retries": 3,
        "expired_lease_policy": ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT,
        "quarantine_threshold": 5,
    }


def test_admin_webhook_delivery_queue_is_registered(tmp_path) -> None:
    manager = JobManager(tmp_path / "jobs.db")

    assert "delivery" in manager._get_allowed_queues("admin_webhooks")


def test_admit_job_returns_typed_result_and_create_job_preserves_dict_contract(
    tmp_path,
) -> None:
    manager = JobManager(tmp_path / "jobs.db")
    admitted = manager.admit_job(**_canonical_kwargs(key="typed"))
    created = manager.create_job(**_canonical_kwargs(key="legacy"))

    assert admitted.outcome is OperationOutcome.APPLIED
    assert admitted.inserted is True
    assert admitted.row is not None
    assert admitted.row["expired_lease_policy"] == "requeue_no_attempt"
    assert admitted.row["quarantine_threshold"] == 5
    assert created["status"] == "queued"
    assert created["expired_lease_policy"] == "requeue_no_attempt"
    assert created["quarantine_threshold"] == 5


def test_create_job_delegates_once_to_typed_admission(monkeypatch, tmp_path) -> None:
    manager = JobManager(tmp_path / "jobs.db")
    calls: list[dict] = []
    row = {"id": 1, "status": "queued"}

    def fake_admit_job(**kwargs):
        calls.append(kwargs)
        return AdmissionResult.applied(row=row)

    monkeypatch.setattr(manager, "admit_job", fake_admit_job)

    result = manager.create_job(
        domain="x",
        queue="default",
        job_type="work",
        payload={},
        owner_user_id=None,
    )

    assert result == row
    assert len(calls) == 1
    assert calls[0]["expired_lease_policy"] is ExpiredLeasePolicy.CONSUME_RETRY
    assert calls[0]["quarantine_threshold"] is None


def test_typed_backend_rejection_is_the_only_public_mapping_difference(
    monkeypatch,
    tmp_path,
) -> None:
    manager = JobManager(tmp_path / "jobs.db")
    rejected = AdmissionResult.rejected(
        AdmissionRejectionReason.QUOTA_EXCEEDED,
        message="Quota exceeded: stable",
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Jobs.manager._sqlite_create_job_admission",
        lambda *_args, **_kwargs: rejected,
    )
    kwargs = _canonical_kwargs(key="rejected")

    typed = manager.admit_job(**kwargs)
    with pytest.raises(ValueError, match="Quota exceeded: stable"):
        manager.create_job(**kwargs)

    assert typed is rejected
    assert typed.outcome is OperationOutcome.ADMISSION_REJECTED

def test_idempotent_existing_requires_execution_control_equality(tmp_path) -> None:
    manager = JobManager(tmp_path / "jobs.db")
    kwargs = _canonical_kwargs(key="controls")
    first = manager.admit_job(**kwargs)

    replay = manager.admit_job(**kwargs)
    conflict = manager.admit_job(
        **{
            **kwargs,
            "expired_lease_policy": ExpiredLeasePolicy.CONSUME_RETRY,
            "quarantine_threshold": None,
        }
    )

    assert first.outcome is OperationOutcome.APPLIED
    assert replay.no_transition_reason is NoTransitionReason.IDEMPOTENT_EXISTING
    assert conflict.outcome is OperationOutcome.BACKEND_CONFLICT
    assert conflict.row is not None


@pytest.mark.parametrize(
    ("policy", "threshold"),
    [("invalid", 5), (ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT, 0), (ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT, True)],
)
def test_invalid_execution_controls_fail_before_connect(
    monkeypatch,
    tmp_path,
    policy,
    threshold,
) -> None:
    manager = JobManager(tmp_path / "jobs.db")
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: pytest.fail("invalid controls reached SQL"),
    )

    with pytest.raises(ValueError):
        manager.admit_job(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            payload={"delivery_id": str(uuid4())},
            owner_user_id=None,
            expired_lease_policy=policy,
            quarantine_threshold=threshold,
        )


def test_admission_facades_share_transformation_and_side_effect_pipeline(
    monkeypatch,
    tmp_path,
) -> None:
    manager = JobManager(tmp_path / "jobs.db")
    calls: list[tuple[str, object]] = []
    original_scan = manager._scan_and_redact_secrets
    original_emit = manager._emit_create_side_effects
    original_invariants = manager._assert_invariants

    def scan(payload):
        calls.append(("scan", payload))
        return original_scan(payload)

    def emit(result, *, backend, idempotency_key):
        calls.append(("emit", result.outcome))
        return original_emit(
            result, backend=backend, idempotency_key=idempotency_key
        )

    def invariants(row):
        calls.append(("invariants", row["status"]))
        return original_invariants(row)

    monkeypatch.setattr(manager, "_scan_and_redact_secrets", scan)
    monkeypatch.setattr(manager, "_emit_create_side_effects", emit)
    monkeypatch.setattr(manager, "_assert_invariants", invariants)

    manager.admit_job(**_canonical_kwargs(key="typed-parity"))
    typed_calls = list(calls)
    calls.clear()
    manager.create_job(**_canonical_kwargs(key="legacy-parity"))

    assert [name for name, _ in typed_calls] == ["scan", "invariants", "emit"]
    assert [name for name, _ in calls] == ["scan", "invariants", "emit"]

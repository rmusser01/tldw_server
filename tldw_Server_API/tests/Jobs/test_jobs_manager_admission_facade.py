from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.Jobs.manager import (
    JobManager,
    SlidesGenerationJobsUnavailableError,
)
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AdmissionRejectionReason,
    AdmissionResult,
    ApplyPreparedDispositionCommand,
    EnsureLeaseHorizonCommand,
    ExpiredLeasePolicy,
    FindJobByIdentityCommand,
    NoTransitionReason,
    OperationOutcome,
    PreparedJobDisposition,
)

pytestmark = pytest.mark.unit


def _canonical_kwargs(*, key: str) -> dict:
    del key
    delivery_id = str(uuid4())
    return {
        "domain": "admin_webhooks",
        "queue": "delivery",
        "job_type": "admin_webhook_delivery",
        "payload": {"delivery_id": delivery_id},
        "owner_user_id": None,
        "idempotency_key": f"admin-webhook-delivery:{delivery_id}",
        "max_retries": 3,
        "expired_lease_policy": ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT,
        "quarantine_threshold": 5,
    }


def _noncanonical_admission(case: str) -> dict:
    kwargs = _canonical_kwargs(key=case)
    delivery_id = kwargs["payload"]["delivery_id"]
    if case == "job_type":
        kwargs["job_type"] = "other"
    elif case == "payload_missing":
        kwargs["payload"] = {}
    elif case == "payload_extra":
        kwargs["payload"] = {"delivery_id": delivery_id, "extra": True}
    elif case == "payload_id":
        kwargs["payload"] = {"delivery_id": "not-a-uuid"}
    elif case == "idempotency_key":
        kwargs["idempotency_key"] = f"admin-webhook-delivery:{delivery_id}:suffix"
    elif case == "owner":
        kwargs["owner_user_id"] = "owner-1"
    elif case == "project":
        kwargs["project_id"] = 1
    elif case == "batch":
        kwargs["batch_group"] = "batch-1"
    elif case == "schedule":
        kwargs["available_at"] = datetime.now(timezone.utc)
    elif case == "priority":
        kwargs["priority"] = 4
    elif case == "max_retries":
        kwargs["max_retries"] = 2
    elif case == "lease_policy":
        kwargs["expired_lease_policy"] = ExpiredLeasePolicy.CONSUME_RETRY
    elif case == "quarantine_threshold":
        kwargs["quarantine_threshold"] = 4
    else:  # pragma: no cover - test table guard
        raise AssertionError(case)
    return kwargs


def test_admin_webhook_delivery_queue_is_registered(tmp_path) -> None:
    manager = JobManager(tmp_path / "jobs.db")

    assert "delivery" in manager._get_allowed_queues("admin_webhooks")


@pytest.mark.parametrize(
    "case",
    (
        "job_type",
        "payload_missing",
        "payload_extra",
        "payload_id",
        "idempotency_key",
        "owner",
        "project",
        "batch",
        "schedule",
        "priority",
        "max_retries",
        "lease_policy",
        "quarantine_threshold",
    ),
)
def test_admin_webhook_delivery_admission_rejects_every_noncanonical_fact_before_sql(
    monkeypatch,
    tmp_path,
    case,
) -> None:
    manager = JobManager(tmp_path / "jobs.db")
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: pytest.fail("noncanonical admission reached SQL"),
    )

    with pytest.raises(ValueError, match="canonical"):
        manager.admit_job(**_noncanonical_admission(case))


def _guard_canonical_transform_side_effects(monkeypatch, manager) -> list[str]:
    calls: list[str] = []

    def fail_connect():
        calls.append("connect")
        pytest.fail("transformed canonical admission reached SQL")

    def fail_emit(*_args, **_kwargs):
        calls.append("emit")
        pytest.fail("transformed canonical admission emitted create side effects")

    monkeypatch.setattr(manager, "_connect", fail_connect)
    monkeypatch.setattr(manager, "_emit_create_side_effects", fail_emit)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Jobs.manager.increment_json_truncated",
        lambda *_args, **_kwargs: calls.append("truncate_metric"),
    )
    return calls


def test_canonical_admission_rejects_configured_redaction_before_side_effects(
    monkeypatch,
    tmp_path,
) -> None:
    manager = JobManager(tmp_path / "canonical-redaction.db")
    calls = _guard_canonical_transform_side_effects(monkeypatch, manager)
    monkeypatch.setenv("JOBS_SECRET_REDACT", "true")
    monkeypatch.setenv("JOBS_SECRET_PATTERNS", r"^[0-9a-f-]{36}$")

    with pytest.raises(ValueError, match="canonical"):
        manager.admit_job(**_canonical_kwargs(key="redaction"))

    assert calls == []


def test_canonical_admission_rejects_configured_encryption_before_side_effects(
    monkeypatch,
    tmp_path,
) -> None:
    manager = JobManager(tmp_path / "canonical-encryption.db")
    calls = _guard_canonical_transform_side_effects(monkeypatch, manager)
    monkeypatch.setenv("JOBS_ENCRYPT_ADMIN_WEBHOOKS", "true")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Jobs.manager.encrypt_json_blob",
        lambda _payload: {
            "_enc": "aesgcm:v1",
            "nonce": "nonce",
            "ct": "ciphertext",
            "tag": "tag",
        },
    )

    with pytest.raises(ValueError, match="canonical"):
        manager.admit_job(**_canonical_kwargs(key="encryption"))

    assert calls == []


def test_canonical_admission_rejects_truncation_before_metric_or_sql(
    monkeypatch,
    tmp_path,
) -> None:
    manager = JobManager(tmp_path / "canonical-truncation.db")
    calls = _guard_canonical_transform_side_effects(monkeypatch, manager)
    monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "1")
    monkeypatch.setenv("JOBS_JSON_TRUNCATE", "true")

    with pytest.raises(ValueError, match="canonical"):
        manager.admit_job(**_canonical_kwargs(key="truncation"))

    assert calls == []


def test_canonical_admission_rejects_mocked_payload_transform_before_side_effects(
    monkeypatch,
    tmp_path,
) -> None:
    manager = JobManager(tmp_path / "canonical-transform.db")
    calls = _guard_canonical_transform_side_effects(monkeypatch, manager)
    monkeypatch.setattr(
        manager,
        "_maybe_encrypt_json",
        lambda payload, _domain: {**payload, "unexpected": True},
    )

    with pytest.raises(ValueError, match="canonical"):
        manager.admit_job(**_canonical_kwargs(key="mocked-transform"))

    assert calls == []


@pytest.mark.parametrize("field", ("domain", "queue", "job_type", "payload", "key"))
def test_find_identity_rejects_noncanonical_facts_before_sql(
    monkeypatch,
    tmp_path,
    field,
) -> None:
    manager = JobManager(tmp_path / "jobs.db")
    delivery_id = str(uuid4())
    command = FindJobByIdentityCommand(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        idempotency_key=f"admin-webhook-delivery:{delivery_id}",
        expected_payload={"delivery_id": delivery_id},
    )
    changes = {
        "domain": {"domain": "other"},
        "queue": {"queue": "other"},
        "job_type": {"job_type": "other"},
        "payload": {"expected_payload": {"delivery_id": str(uuid4())}},
        "key": {"idempotency_key": f"admin-webhook-delivery:{delivery_id}:suffix"},
    }
    command = replace(command, **changes[field])
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: pytest.fail("noncanonical lookup reached SQL"),
    )

    with pytest.raises(ValueError, match="canonical"):
        manager.find_job_by_identity(command)


@pytest.mark.parametrize("facade", ("lease", "disposition", "delivery_mismatch"))
def test_lifecycle_facades_reject_noncanonical_facts_before_sql(
    monkeypatch,
    tmp_path,
    facade,
) -> None:
    manager = JobManager(tmp_path / "jobs.db")
    delivery_id = str(uuid4())
    payload = {"delivery_id": delivery_id}
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: pytest.fail("noncanonical lifecycle command reached SQL"),
    )

    if facade == "lease":
        command = EnsureLeaseHorizonCommand(
            job_id=1,
            domain="other",
            queue="delivery",
            job_type="admin_webhook_delivery",
            expected_payload=payload,
            worker_id="worker",
            lease_id="lease",
            minimum_seconds=30,
        )
        def invoke():
            return manager.ensure_lease_horizon(command)

    else:
        disposition_delivery_id = str(uuid4()) if facade == "delivery_mismatch" else delivery_id
        command = ApplyPreparedDispositionCommand(
            job_id=1,
            domain="admin_webhooks",
            queue="other" if facade == "disposition" else "delivery",
            job_type="admin_webhook_delivery",
            expected_payload=payload,
            disposition=PreparedJobDisposition.cancel(
                token="a" * 64,
                delivery_id=disposition_delivery_id,
                reason_code="registration_disabled",
            ),
        )
        def invoke():
            return manager.apply_prepared_disposition(command)

    with pytest.raises(ValueError, match="canonical"):
        invoke()


def test_slides_early_replay_validates_requested_execution_controls(tmp_path) -> None:
    manager = JobManager(tmp_path / "slides.db")
    kwargs = {
        "domain": "slides",
        "queue": "default",
        "job_type": "presentation.generate",
        "payload": {"receipt_id": "receipt-1"},
        "owner_user_id": "owner-1",
        "idempotency_key": "slides-controls",
    }
    created = manager.create_job(**kwargs)

    replayed = manager.create_job(**kwargs)
    assert replayed["uuid"] == created["uuid"]
    with pytest.raises(SlidesGenerationJobsUnavailableError):
        manager.admit_job(
            **kwargs,
            expired_lease_policy=ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT,
            quarantine_threshold=5,
        )


def test_slides_sqlite_race_callback_validates_requested_execution_controls(
    monkeypatch,
    tmp_path,
) -> None:
    manager = JobManager(tmp_path / "slides-race.db")
    kwargs = {
        "domain": "slides",
        "queue": "default",
        "job_type": "presentation.generate",
        "payload": {"receipt_id": "receipt-1"},
        "owner_user_id": "owner-1",
        "idempotency_key": "slides-race-controls",
    }
    manager.create_job(**kwargs)
    monkeypatch.setattr(
        manager,
        "_serialized_slides_generation_replay",
        lambda **_kwargs: None,
    )

    replayed = manager.admit_job(**kwargs)
    assert replayed.outcome is OperationOutcome.NO_TRANSITION
    assert replayed.no_transition_reason is NoTransitionReason.IDEMPOTENT_EXISTING
    assert replayed.inserted is False

    with pytest.raises(SlidesGenerationJobsUnavailableError):
        manager.admit_job(
            **kwargs,
            expired_lease_policy=ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT,
            quarantine_threshold=5,
        )


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
    kwargs = {
        "domain": "generic-controls",
        "queue": "default",
        "job_type": "work",
        "payload": {"value": 1},
        "owner_user_id": None,
        "idempotency_key": "generic-controls-1",
        "expired_lease_policy": ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT,
        "quarantine_threshold": 5,
    }
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

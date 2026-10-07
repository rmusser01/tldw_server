"""Public failed-parent retry validation preserves state and typed errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import BadRequestError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Jobs.test_failed_job_requeue_admission import failed_job, retry, snapshot

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("override", [
    {"job_id": True}, {"job_id": 0}, {"job_id": -1}, {"job_id": "1"},
    {"owner_user_id": " "}, {"expected_uuid": ""}, {"domain": ""},
    {"queue": None}, {"job_type": ""}, {"idempotency_key": ""},
    {"expected_payload": []}, {"domain": "other"}, {"job_type": "child"},
    {"expected_payload": {"pack_id": True, "batch_id": 7, "user_id": 42}},
    {"expected_payload": {"pack_id": 1, "batch_id": 7, "user_id": 43}},
    {"expected_payload": {"pack_id": 1, "batch_id": 7, "user_id": 42, "extra": 1}},
])
def test_retry_validation_is_typed_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: dict[str, Any],
) -> None:
    """Reject invalid public input using a ValueError-compatible domain type.

    Args:
        tmp_path: Isolated native SQLite Jobs directory.
        monkeypatch: Configure the existing allowed queue policy.
        override: One invalid identity or immutable parent-payload input.
    Returns:
        None; asserts the typed rejection and unchanged failed row/counters/events.
    """
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES", "default,generation")
    jobs = JobManager(db_path=tmp_path / "validation.db")
    job = failed_job(jobs)
    before = jobs.get_job(job["id"], owner_user_id="42")
    with pytest.raises(BadRequestError) as rejected:
        retry(jobs, job, **override)
    assert isinstance(rejected.value, ValueError)
    assert jobs.get_job(job["id"], owner_user_id="42") == before
    assert snapshot(jobs, job) == ("failed", 0, 0)


@pytest.mark.parametrize("policy", ["queue", "type"])
def test_retry_policy_rejection_is_typed_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str,
) -> None:
    """Keep admission policy failures typed without admitting the failed parent.

    Args:
        tmp_path: Isolated native SQLite Jobs directory.
        monkeypatch: Change public environment policy after the initial failure.
        policy: Queue or job-type policy to reject on explicit retry.
    Returns:
        None; asserts compatible typed failure and unchanged row/counters/events.
    """
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES", "default,generation,receipt-policy")
    jobs = JobManager(db_path=tmp_path / "policy.db")
    queue = "receipt-policy" if policy == "queue" else "default"
    job = jobs.create_job(
        domain="vn_assets", queue=queue, job_type="vn_asset_enqueue_batch", owner_user_id="42",
        payload={"pack_id": 1, "batch_id": 7, "user_id": 42}, idempotency_key="policy-parent", max_retries=0,
    )
    acquired = jobs.acquire_next_job(
        domain="vn_assets", queue=queue, worker_id="policy-worker", lease_seconds=60, owner_user_id="42",
    )
    assert acquired is not None
    assert jobs.fail_job(
        job["id"], error="partial fanout", retryable=True, backoff_seconds=0,
        worker_id=acquired["worker_id"], lease_id=acquired["lease_id"],
    )
    before = jobs.get_job(job["id"], owner_user_id="42")
    queue_before = jobs.get_queue_stats(domain="vn_assets", queue=queue)
    events_before = jobs.list_job_events_after(job_id=job["id"], owner_user_id="42")
    if policy == "queue":
        monkeypatch.setenv("JOBS_ALLOWED_QUEUES", "default,generation")
    else:
        monkeypatch.setenv("JOBS_ALLOWED_JOB_TYPES", "another-type")
    with pytest.raises(BadRequestError) as rejected:
        retry(jobs, job, queue=queue)
    assert isinstance(rejected.value, ValueError)
    assert jobs.get_job(job["id"], owner_user_id="42") == before
    assert jobs.get_queue_stats(domain="vn_assets", queue=queue) == queue_before
    assert jobs.list_job_events_after(job_id=job["id"], owner_user_id="42") == events_before

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_jobs
from tldw_Server_API.app.core.Claims_Extraction.claims_job_contracts import (
    CLAIMS_DELIVER_ALERT_JOB_TYPE,
    CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE,
    CLAIMS_JOBS_DOMAIN,
    CLAIMS_REBUILD_MEDIA_JOB_TYPE,
)

pytestmark = pytest.mark.unit


class FakeJobManager:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.status_counts = {"queued": 2, "processing": 1, "failed": 1}

    def create_job(self, **kwargs):
        self.created.append(kwargs)
        return {"id": len(self.created), **kwargs, "status": "queued"}

    def get_job(self, job_id: int):
        return {"id": job_id, **self.created[job_id - 1], "status": "queued"}

    def summarize_by_status(self, **kwargs):
        assert kwargs == {"domain": CLAIMS_JOBS_DOMAIN, "owner_user_id": "1"}  # nosec B101
        return dict(self.status_counts)


def test_enqueue_rebuild_media_creates_id_only_jobs_payload() -> None:
    fake = FakeJobManager()

    job = claims_jobs.enqueue_claims_rebuild_media(
        media_id=42,
        owner_user_id="1",
        job_manager=fake,
        settings_obj={
            "CLAIMS_JOBS_QUEUE": "default",
            "CLAIMS_JOBS_MAX_RETRIES_REBUILD": 4,
        },
    )

    assert job["id"] == 1  # nosec B101
    created = fake.created[0]
    assert created["domain"] == CLAIMS_JOBS_DOMAIN  # nosec B101
    assert created["queue"] == "default"  # nosec B101
    assert created["job_type"] == CLAIMS_REBUILD_MEDIA_JOB_TYPE  # nosec B101
    assert created["owner_user_id"] == "1"  # nosec B101
    assert created["payload"] == {"version": 1, "owner_user_id": "1", "media_id": 42}  # nosec B101
    assert "db_path" not in created["payload"]  # nosec B101
    assert created["idempotency_key"] is None  # nosec B101
    assert created["max_retries"] == 4  # nosec B101


def test_enqueue_rebuild_media_accepts_scoped_idempotency_key() -> None:
    fake = FakeJobManager()

    claims_jobs.enqueue_claims_rebuild_media(
        media_id=42,
        owner_user_id="1",
        idempotency_scope="stale-sweep-2026-06-25",
        job_manager=fake,
        settings_obj={"CLAIMS_JOBS_QUEUE": "default"},
    )

    created = fake.created[0]
    assert created["idempotency_key"] == "claims:rebuild:1:42:stale-sweep-2026-06-25"  # nosec B101


def test_enqueue_review_notification_uses_sorted_idempotency() -> None:
    fake = FakeJobManager()

    claims_jobs.enqueue_claims_review_notification(
        owner_user_id="1",
        notification_ids=[9, 3, 9],
        job_manager=fake,
        settings_obj={"CLAIMS_JOBS_QUEUE": "default"},
    )

    created = fake.created[0]
    assert created["job_type"] == CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE  # nosec B101
    assert created["payload"]["notification_ids"] == [3, 9]  # nosec B101
    assert created["idempotency_key"].startswith("claims:notify_review:1:")  # nosec B101


def test_claims_jobs_config_helpers_read_environment(monkeypatch) -> None:
    monkeypatch.setenv("CLAIMS_JOBS_ENABLED", "true")
    monkeypatch.setenv("CLAIMS_JOBS_WORKER_ENABLED", "1")
    monkeypatch.setenv("CLAIMS_JOBS_QUEUE", "claims-review")

    assert claims_jobs.claims_jobs_enabled() is True  # nosec B101
    assert claims_jobs.claims_jobs_worker_enabled() is True  # nosec B101
    assert claims_jobs.claims_jobs_queue() == "claims-review"  # nosec B101


def test_claims_jobs_queue_uses_default_for_blank_environment(monkeypatch) -> None:
    monkeypatch.setenv("CLAIMS_JOBS_QUEUE", "   ")

    assert claims_jobs.claims_jobs_queue() == "default"  # nosec B101


def test_enqueue_alert_delivery_rejects_email_channel() -> None:
    fake = FakeJobManager()

    with pytest.raises(Exception):
        claims_jobs.enqueue_claims_alert_delivery(
            owner_user_id="1",
            event_id=10,
            alert_id=5,
            channel="email",
            job_manager=fake,
        )

    assert fake.created == []  # nosec B101


def test_enqueue_alert_delivery_creates_normalized_jobs_payload(monkeypatch) -> None:
    monkeypatch.setenv("CLAIMS_JOBS_MAX_RETRIES_ALERT", "-1")
    fake = FakeJobManager()

    claims_jobs.enqueue_claims_alert_delivery(
        owner_user_id="1",
        event_id=10,
        alert_id=5,
        channel="SLACK",
        job_manager=fake,
    )

    created = fake.created[0]
    assert created["job_type"] == CLAIMS_DELIVER_ALERT_JOB_TYPE  # nosec B101
    assert created["payload"] == {  # nosec B101
        "version": 1,
        "owner_user_id": "1",
        "event_id": 10,
        "alert_id": 5,
        "channel": "slack",
    }
    assert created["max_retries"] == 3  # nosec B101
    assert created["idempotency_key"] == "claims:alert:1:10:5:slack"  # nosec B101


def test_enqueue_alert_delivery_defaults_negative_explicit_max_retries() -> None:
    fake = FakeJobManager()

    claims_jobs.enqueue_claims_alert_delivery(
        owner_user_id="1",
        event_id=10,
        alert_id=5,
        channel="webhook",
        job_manager=fake,
        settings_obj={
            "CLAIMS_JOBS_QUEUE": "default",
            "CLAIMS_JOBS_MAX_RETRIES_ALERT": -1,
        },
    )

    created = fake.created[0]
    assert created["max_retries"] == 3  # nosec B101


def test_claims_jobs_summary_is_read_only() -> None:
    fake = FakeJobManager()

    summary = claims_jobs.claims_jobs_summary(job_manager=fake, owner_user_id="1")

    assert summary == {  # nosec B101
        "domain": CLAIMS_JOBS_DOMAIN,
        "counts": {"queued": 2, "processing": 1, "failed": 1},
    }

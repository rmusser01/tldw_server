import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_jobs
from tldw_Server_API.app.core.Claims_Extraction.claims_job_contracts import (
    CLAIMS_DELIVER_ALERT_JOB_TYPE,
    CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE,
    CLAIMS_JOBS_DOMAIN,
    CLAIMS_REBUILD_MEDIA_JOB_TYPE,
    ClaimsJobError,
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


class NoRefreshJobManager(FakeJobManager):
    def __init__(self) -> None:
        super().__init__()
        self.get_job_calls = 0
        self.created_result = {"id": 91, "accepted": True}

    def create_job(self, **kwargs):
        self.created.append(kwargs)
        return self.created_result

    def get_job(self, job_id: int):
        self.get_job_calls += 1
        raise AssertionError(f"unexpected Jobs refresh for {job_id}")


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


@pytest.mark.parametrize(
    ("claims_enabled", "exports_enabled", "expected"),
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_claims_analytics_export_jobs_enabled_requires_both_flags(
    claims_enabled: bool,
    exports_enabled: bool,
    expected: bool,
) -> None:
    settings_obj = {
        "CLAIMS_JOBS_ENABLED": claims_enabled,
        "CLAIMS_ANALYTICS_EXPORT_JOBS_ENABLED": exports_enabled,
    }

    assert claims_jobs.claims_analytics_export_jobs_enabled(settings_obj) is expected  # nosec B101


def test_claims_analytics_export_jobs_enabled_defaults_to_false() -> None:
    assert (  # nosec B101
        claims_jobs.claims_analytics_export_jobs_enabled(
            {"CLAIMS_JOBS_ENABLED": True}
        )
        is False
    )


def test_enqueue_analytics_export_creates_exact_job_and_does_not_refresh() -> None:
    fake = NoRefreshJobManager()
    export_id = "0123456789abcdef0123456789abcdef"

    result = claims_jobs.enqueue_claims_analytics_export(
        owner_user_id="123",
        export_id=export_id,
        job_manager=fake,
        settings_obj={"CLAIMS_JOBS_QUEUE": "default"},
    )

    assert result is fake.created_result  # nosec B101
    assert fake.get_job_calls == 0  # nosec B101
    assert fake.created == [  # nosec B101
        {
            "domain": "claims",
            "queue": "default",
            "job_type": "claims_generate_analytics_export",
            "payload": {
                "version": 1,
                "owner_user_id": "123",
                "export_id": export_id,
            },
            "owner_user_id": "123",
            "priority": 5,
            "max_retries": 3,
            "batch_group": f"claims-analytics-export:{export_id}",
            "idempotency_key": f"claims:analytics_export:123:{export_id}",
        }
    ]


@pytest.mark.parametrize("configured", [-1, "-2", "invalid"])
def test_enqueue_analytics_export_defaults_invalid_or_negative_retries(
    configured: object,
) -> None:
    fake = NoRefreshJobManager()

    claims_jobs.enqueue_claims_analytics_export(
        owner_user_id="123",
        export_id="0123456789abcdef0123456789abcdef",
        job_manager=fake,
        settings_obj={
            "CLAIMS_JOBS_MAX_RETRIES_ANALYTICS_EXPORT": configured,
        },
    )

    assert fake.created[0]["max_retries"] == 3  # nosec B101


@pytest.mark.parametrize(
    ("owner_user_id", "export_id", "failure_code"),
    [
        ("0123", "0123456789abcdef0123456789abcdef", "claims_missing_owner"),
        ("123", "0123456789ABCDEF0123456789ABCDEF", "claims_export_invalid_payload"),
    ],
)
def test_enqueue_analytics_export_validates_before_creating_job(
    owner_user_id: str,
    export_id: str,
    failure_code: str,
) -> None:
    fake = NoRefreshJobManager()

    with pytest.raises(ClaimsJobError) as excinfo:
        claims_jobs.enqueue_claims_analytics_export(
            owner_user_id=owner_user_id,
            export_id=export_id,
            job_manager=fake,
        )

    assert excinfo.value.failure_code == failure_code  # nosec B101
    assert fake.created == []  # nosec B101


def test_claims_jobs_queue_uses_default_for_blank_environment(monkeypatch) -> None:
    monkeypatch.setenv("CLAIMS_JOBS_QUEUE", "   ")

    assert claims_jobs.claims_jobs_queue() == "default"  # nosec B101


def test_enqueue_alert_delivery_rejects_email_channel() -> None:
    fake = FakeJobManager()

    with pytest.raises(ClaimsJobError):
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

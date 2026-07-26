"""Shared Jobs parity scenarios for SQLite and Postgres backends."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier

from tldw_Server_API.app.core.Jobs.manager import JobManager

ManagerFactory = Callable[[], JobManager]
LeaseExpiry = Callable[[JobManager, int], None]
FUTURE_NOW_EPOCH = str(int((datetime.now(timezone.utc) + timedelta(days=365)).timestamp()))


def _as_utc_datetime(value: object) -> datetime:
    """Normalize backend timestamp values for cross-backend assertions."""

    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def run_idempotent_create_scope_scenario(make_manager: ManagerFactory) -> None:
    """Verify idempotency is scoped by domain, queue, type, and owner."""

    jm = make_manager()
    key = "idem-key-123"

    first = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="1",
        idempotency_key=key,
    )
    replay = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="1",
        idempotency_key=key,
    )
    assert int(first["id"]) == int(replay["id"])

    different_queue = jm.create_job(
        domain="chatbooks",
        queue="high",
        job_type="export",
        payload={},
        owner_user_id="1",
        idempotency_key=key,
    )
    different_type = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="import",
        payload={},
        owner_user_id="1",
        idempotency_key=key,
    )
    different_domain = jm.create_job(
        domain="other",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="2",
        idempotency_key=key,
    )

    assert int(different_queue["id"]) != int(first["id"])
    assert int(different_type["id"]) != int(first["id"])
    assert int(different_domain["id"]) != int(first["id"])


def run_idempotent_create_preserves_original_request_ids_scenario(make_manager: ManagerFactory) -> None:
    """Verify idempotent replay preserves the original request and trace ids."""

    jm = make_manager()
    key = "idem-request-id-key"

    first = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="1",
        idempotency_key=key,
        request_id="request-first",
        trace_id="trace-first",
    )
    replay = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="1",
        idempotency_key=key,
        request_id="request-second",
        trace_id="trace-second",
    )

    assert int(first["id"]) == int(replay["id"])
    assert first["request_id"] == "request-first"
    assert replay["request_id"] == "request-first"
    assert first["trace_id"] == "trace-first"
    assert replay["trace_id"] == "trace-first"


def run_idempotent_create_replay_event_uses_current_request_ids_scenario(make_manager: ManagerFactory) -> None:
    """Verify idempotent replay writes a current-context durable create event."""

    jm = make_manager()
    key = "idem-replay-event-key"

    first = jm.create_job(
        domain="parity",
        queue="default",
        job_type="idem-replay-event",
        payload={},
        owner_user_id="owner-1",
        idempotency_key=key,
        request_id="request-first",
        trace_id="trace-first",
    )
    replay = jm.create_job(
        domain="parity",
        queue="default",
        job_type="idem-replay-event",
        payload={},
        owner_user_id="owner-1",
        idempotency_key=key,
        request_id="request-replay",
        trace_id="trace-replay",
    )

    assert int(first["id"]) == int(replay["id"])
    assert replay["request_id"] == "request-first"
    assert replay["trace_id"] == "trace-first"

    events = jm.list_job_events_after(
        after_id=0,
        domain="parity",
        queue="default",
        job_type="idem-replay-event",
        limit=20,
    )
    created_events = [event for event in events if event.get("event_type") == "job.created"]

    assert len(created_events) == 2
    assert created_events[0]["request_id"] == "request-first"
    assert created_events[0]["trace_id"] == "trace-first"
    assert created_events[1]["request_id"] == "request-replay"
    assert created_events[1]["trace_id"] == "trace-replay"

    first_attrs = created_events[0].get("attrs_json")
    replay_attrs = created_events[1].get("attrs_json")
    if isinstance(first_attrs, str):
        first_attrs = json.loads(first_attrs)
    if isinstance(replay_attrs, str):
        replay_attrs = json.loads(replay_attrs)

    assert first_attrs["idempotent"] is False
    assert replay_attrs["idempotent"] is True


def run_acquire_complete_lifecycle_scenario(make_manager: ManagerFactory) -> None:
    """Verify the acquire-to-complete lifecycle shape remains backend-neutral."""

    jm = make_manager()
    job = jm.create_job(
        domain="parity",
        queue="default",
        job_type="lifecycle",
        payload={"value": 1},
        owner_user_id="owner-1",
    )

    acquired = jm.acquire_next_job(
        domain="parity",
        queue="default",
        lease_seconds=10,
        worker_id="worker-1",
    )

    assert acquired is not None
    assert int(acquired["id"]) == int(job["id"])
    assert acquired["status"] == "processing"
    assert acquired["worker_id"] == "worker-1"
    assert acquired.get("lease_id")

    token = str(acquired["lease_id"])
    assert jm.complete_job(
        int(job["id"]),
        result={"ok": True},
        worker_id="worker-1",
        lease_id=token,
        completion_token=token,
    ) is True

    stored = jm.get_job(int(job["id"]))
    assert stored is not None
    assert stored["status"] == "completed"
    assert stored.get("leased_until") is None


def run_acquire_contention_scenario(make_manager: ManagerFactory) -> None:
    """Verify concurrent workers acquire a pending job at most once."""

    seed = make_manager()
    job = seed.create_job(
        domain="parity-contention",
        queue="default",
        job_type="single",
        payload={},
        owner_user_id="owner-1",
    )
    managers = [make_manager(), make_manager()]
    barrier = Barrier(2)

    def acquire(item: tuple[JobManager, str]) -> dict[str, object] | None:
        manager, worker_id = item
        barrier.wait(timeout=10)
        return manager.acquire_next_job(
            domain="parity-contention",
            queue="default",
            lease_seconds=30,
            worker_id=worker_id,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(acquire, zip(managers, ("worker-1", "worker-2"), strict=True)))

    acquired = [result for result in results if result is not None]
    assert len(acquired) == 1
    assert int(acquired[0]["id"]) == int(job["id"])


def run_expired_lease_reclaim_scenario(
    make_manager: ManagerFactory,
    expire_lease: LeaseExpiry,
) -> None:
    """Verify a different worker can reclaim an expired lease."""

    manager = make_manager()
    job = manager.create_job(
        domain="parity-expiry",
        queue="default",
        job_type="reclaim",
        payload={},
        owner_user_id="owner-1",
    )
    first = manager.acquire_next_job(
        domain="parity-expiry",
        queue="default",
        lease_seconds=30,
        worker_id="worker-1",
    )
    assert first is not None
    expire_lease(manager, int(job["id"]))

    second = manager.acquire_next_job(
        domain="parity-expiry",
        queue="default",
        lease_seconds=30,
        worker_id="worker-2",
    )
    assert second is not None
    assert int(second["id"]) == int(job["id"])
    assert second["worker_id"] == "worker-2"
    assert second["lease_id"] != first["lease_id"]


def run_complete_idempotency_scenario(make_manager: ManagerFactory) -> None:
    """Verify completion retries are idempotent only with the same token."""

    jm = make_manager()
    job = jm.create_job(domain="test", queue="default", job_type="t", payload={}, owner_user_id="u")
    acquired = jm.acquire_next_job(domain="test", queue="default", lease_seconds=10, worker_id="w1")
    assert acquired is not None
    assert int(acquired["id"]) == int(job["id"])

    token = str(acquired["lease_id"])
    mismatch_marker = "other-token"
    assert jm.complete_job(int(job["id"]), worker_id="w1", lease_id=token, completion_token=token) is True
    assert jm.complete_job(int(job["id"]), worker_id="w1", lease_id=token, completion_token=token) is True
    assert jm.complete_job(int(job["id"]), worker_id="w1", lease_id=token, completion_token=mismatch_marker) is False


def run_renew_stale_lease_noop_scenario(make_manager: ManagerFactory) -> None:
    """Verify stale lease renewal attempts leave the current lease untouched."""

    jm = make_manager()
    job = jm.create_job(domain="parity", queue="default", job_type="renew", payload={}, owner_user_id="owner-1")
    acquired = jm.acquire_next_job(domain="parity", queue="default", lease_seconds=10, worker_id="worker-1")
    assert acquired is not None
    assert int(acquired["id"]) == int(job["id"])

    current_lease_id = str(acquired["lease_id"])
    assert jm.renew_job_lease(
        int(job["id"]),
        seconds=20,
        worker_id="worker-1",
        lease_id=current_lease_id,
        progress_percent=25.0,
        progress_message="still running",
        enforce=True,
    ) is True

    assert jm.renew_job_lease(
        int(job["id"]),
        seconds=20,
        worker_id="worker-2",
        lease_id=current_lease_id,
        enforce=True,
    ) is False

    assert jm.renew_job_lease(
        int(job["id"]),
        seconds=20,
        worker_id="worker-1",
        lease_id="stale-lease",
        enforce=True,
    ) is False

    stored = jm.get_job(int(job["id"]))
    assert stored is not None
    assert stored["status"] == "processing"
    assert float(stored["progress_percent"]) == 25.0
    assert stored["progress_message"] == "still running"


def run_renew_lease_characterization_scenario(make_manager: ManagerFactory) -> None:
    """Verify public renewal preserves a longer lease and compatibility behavior."""

    manager = make_manager()
    job = manager.create_job(
        domain="parity-renew-characterization",
        queue="default",
        job_type="renew",
        payload={"phase": "seeded"},
        owner_user_id="owner-1",
    )
    acquired = manager.acquire_next_job(
        domain="parity-renew-characterization",
        queue="default",
        lease_seconds=30,
        worker_id="worker-1",
    )
    assert acquired is not None
    assert int(acquired["id"]) == int(job["id"])
    lease_id = str(acquired["lease_id"])
    acquired_expiry = _as_utc_datetime(acquired["leased_until"])

    assert manager.renew_job_lease(
        int(job["id"]),
        seconds=300,
        worker_id="worker-1",
        lease_id=lease_id,
        enforce=True,
    ) is True
    long_lease = manager.get_job(int(job["id"]))
    assert long_lease is not None
    long_expiry = _as_utc_datetime(long_lease["leased_until"])
    expected_long_expiry = datetime.fromtimestamp(
        float(FUTURE_NOW_EPOCH), tz=timezone.utc
    ) + timedelta(seconds=300)
    assert long_expiry == expected_long_expiry
    assert long_expiry > acquired_expiry

    assert manager.renew_job_lease(
        int(job["id"]),
        seconds=30,
        worker_id="stale-worker",
        lease_id="stale-lease",
        progress_percent=37.5,
        progress_message="compatibility renewal",
        enforce=False,
    ) is True

    renewed = manager.get_job(int(job["id"]))
    assert renewed is not None
    assert _as_utc_datetime(renewed["leased_until"]) == long_expiry
    assert float(renewed["progress_percent"]) == 37.5
    assert renewed["progress_message"] == "compatibility renewal"

    assert manager.renew_job_lease(
        int(job["id"]),
        seconds=60,
        worker_id="stale-worker",
        lease_id=lease_id,
        enforce=True,
    ) is False
    assert manager.renew_job_lease(
        int(job["id"]),
        seconds=60,
        worker_id="worker-1",
        lease_id="stale-lease",
        enforce=True,
    ) is False

    after_stale_attempts = manager.get_job(int(job["id"]))
    assert after_stale_attempts is not None
    assert _as_utc_datetime(after_stale_attempts["leased_until"]) == long_expiry
    assert float(after_stale_attempts["progress_percent"]) == 37.5
    assert after_stale_attempts["progress_message"] == "compatibility renewal"


def run_release_lease_ownership_scenario(make_manager: ManagerFactory) -> None:
    """Verify release ownership, field clearing, and compatibility behavior."""

    credential_manager = make_manager()

    def fail_if_connection_opens() -> None:
        raise AssertionError("release opened a connection before validating credentials")

    credential_manager._connect = fail_if_connection_opens  # type: ignore[method-assign]
    assert credential_manager.release_job(999_001, lease_id="lease-1", enforce=True) is False
    assert credential_manager.release_job(999_001, worker_id="worker-1", enforce=True) is False

    manager = make_manager()
    job = manager.create_job(
        domain="parity-release",
        queue="default",
        job_type="release",
        payload={"document": "kept"},
        owner_user_id="owner-1",
        project_id=17,
        batch_group="batch-1",
        priority=7,
        max_retries=5,
        idempotency_key="release-preservation",
        request_id="request-1",
        trace_id="trace-1",
    )
    acquired = manager.acquire_next_job(
        domain="parity-release",
        queue="default",
        lease_seconds=300,
        worker_id="worker-1",
    )
    assert acquired is not None
    first_lease_id = str(acquired["lease_id"])
    error_stack = {"stage": "release-characterization", "attempt": 1}
    assert manager.fail_job(
        int(job["id"]),
        error="retry release characterization",
        retryable=True,
        backoff_seconds=0,
        worker_id="worker-1",
        lease_id=first_lease_id,
        enforce=True,
        error_code="release_retry",
        error_class="RetryableReleaseError",
        error_stack=error_stack,
    ) is True
    assert manager.reschedule_jobs(
        domain="parity-release",
        queue="default",
        job_type="release",
        status="queued",
        set_now=False,
        delta_seconds=-60,
    ) == 1
    reacquired = manager.acquire_next_job(
        domain="parity-release",
        queue="default",
        lease_seconds=300,
        worker_id="worker-1",
    )
    assert reacquired is not None
    assert int(reacquired["id"]) == int(job["id"])
    assert int(reacquired["retry_count"]) > 0
    assert reacquired["available_at"] is not None
    lease_id = str(reacquired["lease_id"])
    assert manager.renew_job_lease(
        int(job["id"]),
        seconds=30,
        worker_id="worker-1",
        lease_id=lease_id,
        progress_percent=48.0,
        progress_message="ready to yield",
        enforce=True,
    ) is True

    before_release = manager.get_job(int(job["id"]))
    assert before_release is not None
    assert int(before_release["retry_count"]) > 0
    assert before_release["available_at"] is not None
    assert before_release["acquired_at"] is not None
    assert before_release["started_at"] is not None
    assert before_release["last_error"] == "release_retry"
    assert before_release["error_message"] == "retry release characterization"
    assert before_release["error_code"] == "release_retry"
    assert before_release["error_class"] == "RetryableReleaseError"
    stored_error_stack = before_release["error_stack"]
    if isinstance(stored_error_stack, str):
        stored_error_stack = json.loads(stored_error_stack)
    assert stored_error_stack == error_stack
    assert before_release["failure_streak_code"] == "release_retry"
    assert int(before_release["failure_streak_count"]) == 1

    assert manager.release_job(
        int(job["id"]), worker_id="worker-2", lease_id=lease_id, enforce=True
    ) is False
    assert manager.release_job(
        int(job["id"]), worker_id="worker-1", lease_id="stale-lease", enforce=True
    ) is False
    current = manager.get_job(int(job["id"]))
    assert current is not None
    assert current["status"] == "processing"
    assert current["worker_id"] == "worker-1"
    assert current["lease_id"] == lease_id

    assert manager.release_job(
        int(job["id"]),
        worker_id="worker-1",
        lease_id=lease_id,
        reason="yield",
        enforce=True,
    ) is True
    released = manager.get_job(int(job["id"]))
    assert released is not None
    assert released["status"] == "queued"
    for field in (
        "available_at",
        "leased_until",
        "worker_id",
        "lease_id",
        "acquired_at",
        "started_at",
        "completion_token",
    ):
        assert released.get(field) is None

    for field in (
        "id",
        "uuid",
        "domain",
        "queue",
        "job_type",
        "payload",
        "owner_user_id",
        "project_id",
        "batch_group",
        "priority",
        "max_retries",
        "retry_count",
        "progress_percent",
        "progress_message",
        "last_error",
        "error_message",
        "error_code",
        "error_class",
        "error_stack",
        "failure_streak_code",
        "failure_streak_count",
        "idempotency_key",
        "request_id",
        "trace_id",
    ):
        assert released[field] == before_release[field]

    compatibility_job = manager.create_job(
        domain="parity-release-compatibility",
        queue="default",
        job_type="release",
        payload={},
        owner_user_id="owner-1",
    )
    compatibility_acquired = manager.acquire_next_job(
        domain="parity-release-compatibility",
        queue="default",
        lease_seconds=30,
        worker_id="worker-1",
    )
    assert compatibility_acquired is not None
    assert manager.release_job(
        int(compatibility_job["id"]),
        worker_id="stale-worker",
        lease_id="stale-lease",
        enforce=False,
    ) is True
    compatibility_released = manager.get_job(int(compatibility_job["id"]))
    assert compatibility_released is not None
    assert compatibility_released["status"] == "queued"
    assert compatibility_released["worker_id"] is None
    assert compatibility_released["lease_id"] is None


def run_cancel_terminal_noop_scenario(make_manager: ManagerFactory) -> None:
    """Verify repeated cancellation is a no-op once the job is terminal."""

    jm = make_manager()
    job = jm.create_job(domain="parity", queue="default", job_type="cancel", payload={}, owner_user_id="owner-1")

    assert jm.cancel_job(int(job["id"]), reason="user") is True
    stored = jm.get_job(int(job["id"]))
    assert stored is not None
    assert stored["status"] == "cancelled"

    assert jm.cancel_job(int(job["id"]), reason="again") is False
    stored_again = jm.get_job(int(job["id"]))
    assert stored_again is not None
    assert stored_again["status"] == "cancelled"


def run_events_outbox_create_complete_scenario(make_manager: ManagerFactory) -> None:
    """Verify create and complete emit exactly one durable outbox event each."""

    jm = make_manager()
    job = jm.create_job(domain="parity", queue="default", job_type="events", payload={}, owner_user_id="owner-1")
    acquired = jm.acquire_next_job(domain="parity", queue="default", lease_seconds=10, worker_id="worker-1")
    assert acquired is not None
    token = str(acquired["lease_id"])
    assert jm.complete_job(int(job["id"]), worker_id="worker-1", lease_id=token, completion_token=token) is True

    events = jm.list_job_events_after(after_id=0, domain="parity", queue="default", job_type="events", limit=20)
    event_types = [str(row.get("event_type")) for row in events]
    event_type_counts = Counter(event_types)
    assert event_type_counts["job.created"] == 1
    assert event_type_counts["job.completed"] == 1

    for event in events:
        assert event.get("attrs_json") is not None
        assert event.get("domain") == "parity"
        assert event.get("queue") == "default"
        assert event.get("job_type") == "events"

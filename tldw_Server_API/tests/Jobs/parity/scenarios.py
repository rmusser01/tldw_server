"""Shared Jobs parity scenarios for SQLite and Postgres backends."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable

from tldw_Server_API.app.core.Jobs.manager import JobManager

ManagerFactory = Callable[[], JobManager]


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
        worker_id="worker-1",
        lease_id="stale-lease",
        enforce=True,
    ) is False

    stored = jm.get_job(int(job["id"]))
    assert stored is not None
    assert stored["status"] == "processing"
    assert float(stored["progress_percent"]) == 25.0
    assert stored["progress_message"] == "still running"


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


def run_conditional_cancel_binding_scenario(make_manager: ManagerFactory) -> None:
    """Verify cancellation mutates only the exact expected Jobs binding."""

    jm = make_manager()
    job = jm.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={"run_id": "run-1", "occurrence_id": "occ-1", "attempt": 1},
        owner_user_id="owner-1",
        batch_group="batch-1",
        idempotency_key="run-1:occ-1:1",
    )
    binding = jm.normalize_job_binding_view(job, owner_user_id="owner-1")
    assert binding is not None
    mismatched = dict(binding)
    mismatched["payload"] = {**binding["payload"], "occurrence_id": "occ-other"}

    assert jm.cancel_job(int(job["id"]), expected_binding=mismatched) is False
    unchanged = jm.get_job(int(job["id"]))
    assert unchanged is not None
    assert unchanged["status"] == "queued"

    assert jm.cancel_job(int(job["id"]), expected_binding=binding) is True
    cancelled = jm.get_job(int(job["id"]))
    assert cancelled is not None
    assert cancelled["status"] == "cancelled"


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

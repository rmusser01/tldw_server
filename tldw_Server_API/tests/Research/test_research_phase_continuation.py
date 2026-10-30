"""Research phase transitions must remain executable through the real Jobs worker."""

import asyncio
from contextlib import suppress
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Research import jobs_worker
from tldw_Server_API.app.core.Research.service import ResearchService

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class _TerminalPhaseError(RuntimeError):
    retryable = False


def _fixture(tmp_path, monkeypatch):
    """Use the same persisted session and Job manager the worker will consume."""
    manager = JobManager(tmp_path / "jobs.db", backend="sqlite")
    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=manager,
    )
    session = service.create_session(
        owner_user_id="1", query="Rowan Observatory", source_policy="local_only", autonomy_mode="autonomous"
    )
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: manager)
    return manager, ResearchSessionsDB(tmp_path / "research.db"), session


async def _one_phase(tmp_path, monkeypatch, manager, session, *, fail=False, stop=None):
    """Drain a real SDK delivery; stop before another phase can execute."""
    stop = stop or asyncio.Event()
    calls = []
    actual = jobs_worker.handle_research_phase_job

    async def phase(job, **kwargs):
        calls.append(job["payload"]["phase"])
        try:
            if fail:
                raise _TerminalPhaseError("controlled terminal phase error")
            return await actual(job, **kwargs)
        finally:
            stop.set()
            await asyncio.sleep(0)

    monkeypatch.setattr(jobs_worker, "handle_research_phase_job", phase)

    async def terminal_watch():
        # Also stop a legitimately obsolete delivery that never invokes the provider path.
        while not stop.is_set():
            job = manager.get_job(int(session.active_job_id))
            if job["status"] in {"completed", "failed", "cancelled", "quarantined"}:
                stop.set()
                return
            await asyncio.sleep(0.01)

    worker = asyncio.create_task(
        jobs_worker.run_research_jobs_worker(
            stop,
            research_db_path=tmp_path / "research.db",
            outputs_dir=tmp_path / "outputs",
        )
    )
    watcher = asyncio.create_task(terminal_watch())
    try:
        await asyncio.wait_for(asyncio.gather(worker, watcher), timeout=5)
    finally:
        stop.set()
        for task in (worker, watcher):
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task
    return calls


async def test_autonomous_phase_queues_bound_successor_before_parent_completion(tmp_path, monkeypatch):
    manager, db, session = _fixture(tmp_path, monkeypatch)
    completion_receipts = []
    complete = manager.complete_job

    def observe_completion(job_id, **kwargs):
        current = db.get_session(session.id)
        successor = manager.get_job(int(current.active_job_id)) if current.active_job_id else None
        completion_receipts.append(
            bool(
                successor
                and str(successor["id"]) != session.active_job_id
                and successor["status"] == "queued"
                and successor["payload"]["phase"] == "collecting"
            )
        )
        return complete(job_id, **kwargs)

    monkeypatch.setattr(manager, "complete_job", observe_completion)
    calls = await _one_phase(tmp_path, monkeypatch, manager, session)
    assert completion_receipts == [True]  # nosec B101 -- Observe binding before the real parent completion, never in an after-completion callback.
    parent = manager.get_job(int(session.active_job_id))
    current = db.get_session(session.id)
    assert (parent["status"], current.phase, calls) == ("completed", "collecting", ["drafting_plan"])  # nosec B101 -- Assert actual SDK phase and durable completion behavior.
    assert current.active_job_id is not None and current.active_job_id != session.active_job_id  # nosec B101 -- A queued phase must retain an executable successor.
    successor = manager.get_job(int(current.active_job_id))
    assert (successor["status"], successor["owner_user_id"], successor["payload"]["phase"]) == (
        "queued",
        "1",
        "collecting",
    )  # nosec B101 -- Preserve the admitted successor's identity and owner.


async def test_obsolete_phase_delivery_does_not_repeat_planning_or_write_artifacts(tmp_path, monkeypatch):
    manager, db, session = _fixture(tmp_path, monkeypatch)
    db.update_phase(session.id, phase="collecting", status="queued")
    calls = await _one_phase(tmp_path, monkeypatch, manager, session)
    assert (calls, db.list_artifacts(session.id), db.get_session(session.id).phase) == ([], [], "collecting")  # nosec B101 -- A replay must reconcile or refuse, never repeat a committed phase.


async def test_terminal_sdk_failure_reconciles_exact_owned_research_phase(tmp_path, monkeypatch):
    manager, db, session = _fixture(tmp_path, monkeypatch)
    calls = await _one_phase(tmp_path, monkeypatch, manager, session, fail=True)
    stored = manager.get_job(int(session.active_job_id))
    current = db.get_session(session.id)
    assert (stored["status"], calls) == ("failed", ["drafting_plan"])  # nosec B101 -- Establish the real SDK terminal failure before checking Research state.
    assert (current.status, current.active_job_id) == ("failed", None)  # nosec B101 -- The UI must not retain a queued phase after its exact terminal Job.


@pytest.mark.parametrize("changed_field", ["control", "phase", "binding", "dispatch"])
async def test_phase_compare_and_set_refuses_newer_authority_without_recording_an_event(
    tmp_path, monkeypatch, changed_field
):
    manager, db, expected = _fixture(tmp_path, monkeypatch)
    if changed_field == "control":
        db.update_control_state(expected.id, control_state="paused")
    elif changed_field == "phase":
        db.update_phase(expected.id, phase="collecting", status="queued")
    elif changed_field == "binding":
        db.attach_active_job(expected.id, "99")
    else:
        assert (
            db.compare_and_set_phase(  # nosec B101 -- Establish a newer committed dispatch before replaying the old snapshot.
                expected,
                phase=expected.phase,
                status=expected.status,
                control_state=expected.control_state,
                active_job_id=expected.active_job_id,
                phase_dispatch_json={"schema_version": 1, "generation": "newer"},
            )
            is not None
        )
    before = db.get_session(expected.id)
    events = db.list_run_events_after(after_id=0, owner_user_id="1", session_id=expected.id)
    result = db.compare_and_set_phase(
        expected,
        phase="collecting",
        status="queued",
        control_state="running",
        active_job_id=None,
        phase_dispatch_json={},
    )
    assert (
        result,
        db.get_session(expected.id),
        db.list_run_events_after(after_id=0, owner_user_id="1", session_id=expected.id),
    ) == (
        None,
        before,
        events,
    )  # nosec B101 -- A stale transition must have no session or event side effects.


async def test_retired_lease_cannot_publish_phase_artifacts_or_advance_research(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Research import jobs as research_jobs

    manager, db, session = _fixture(tmp_path, monkeypatch)
    plan = research_jobs.build_initial_plan
    replacement = []

    def lose_lease(**kwargs):
        result = plan(**kwargs)
        original = manager.get_job(int(session.active_job_id))
        assert manager.release_job(  # nosec B101 -- Use the actual lease lifecycle to retire the admitted worker.
            original["id"],
            worker_id=original["worker_id"],
            lease_id=original["lease_id"],
            enforce=True,
        )
        claimed = manager.acquire_next_job(
            domain="research", queue="default", lease_seconds=60, worker_id="replacement-worker"
        )
        assert claimed and claimed["uuid"] == original["uuid"] and claimed["lease_id"] != original["lease_id"]  # nosec B101 -- Establish a real new lease on the exact same Job.
        replacement.append(claimed)
        return result

    monkeypatch.setattr(research_jobs, "build_initial_plan", lose_lease)
    try:
        await _one_phase(tmp_path, monkeypatch, manager, session)
        assert (db.list_artifacts(session.id), db.get_session(session.id).phase) == ([], "drafting_plan")  # nosec B101 -- A retired attempt cannot publish a phase or its successor.
    finally:
        for claimed in replacement:
            manager.finalize_cancelled(
                claimed["id"],
                expected_uuid=claimed["uuid"],
                worker_id=claimed["worker_id"],
                lease_id=claimed["lease_id"],
                reason="controlled_test_cleanup",
            )


@pytest.mark.parametrize("action", ["pause", "cancel"])
async def test_control_during_planning_preserves_cooperative_pause_and_cancellation(tmp_path, monkeypatch, action):
    from tldw_Server_API.app.core.Research import jobs as research_jobs

    manager, db, session = _fixture(tmp_path, monkeypatch)
    plan = research_jobs.build_initial_plan
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )

    def control_after_plan(**kwargs):
        result = plan(**kwargs)
        if action == "pause":
            service.pause_run(owner_user_id="1", session_id=session.id)
        else:
            service.cancel_run(owner_user_id="1", session_id=session.id)
        return result

    monkeypatch.setattr(research_jobs, "build_initial_plan", control_after_plan)
    await _one_phase(tmp_path, monkeypatch, manager, session)
    current = db.get_session(session.id)
    job = manager.get_job(int(session.active_job_id))
    if action == "pause":
        assert {row.artifact_name for row in db.list_artifacts(session.id)} == {"plan.json", "provider_config.json"}  # nosec B101 -- Pause-requested lets the admitted phase finish, then parks before its successor.
        assert current.phase == "collecting" and current.phase_dispatch_json["job_uuid"] is None  # nosec B101 -- Preserve the committed next dispatch without starting it.
    else:
        assert db.list_artifacts(session.id) == []  # nosec B101 -- Cancellation forbids subsequent canonical publication.
    assert len(manager.list_jobs(domain="research", owner_user_id="1")) == 1  # nosec B101 -- Neither control may leave an executable successor behind.
    assert (current.status, current.control_state, current.active_job_id, job["status"]) == (
        ("queued", "paused", None, "completed") if action == "pause" else ("cancelled", "cancelled", None, "cancelled")
    )  # nosec B101 -- Cooperative control is not a terminal worker failure.


def _edited_outline_fixture(tmp_path, monkeypatch):
    manager = JobManager(tmp_path / "outline-jobs.db", backend="sqlite")
    db = ResearchSessionsDB(tmp_path / "research.db")
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    session = db.create_session(
        owner_user_id="1",
        query="Rowan Observatory",
        source_policy="local_only",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="awaiting_outline_review",
        status="waiting_human",
    )
    checkpoint = db.create_checkpoint(
        session_id=session.id,
        checkpoint_type="outline_review",
        proposed_payload={"outline": {"sections": [{"title": "Background", "focus_area": "history"}]}},
    )
    approved = service.approve_checkpoint(
        owner_user_id="1",
        session_id=session.id,
        checkpoint_id=checkpoint.id,
        patch_payload={"sections": [{"title": "Evidence first", "focus_area": "history"}]},
    )
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: manager)
    return manager, db, service, approved, checkpoint


async def test_edited_outline_approval_persists_authorized_dispatch(tmp_path, monkeypatch):
    manager, db, service, approved, checkpoint = _edited_outline_fixture(tmp_path, monkeypatch)
    dispatch = approved.phase_dispatch_json
    assert dispatch and dispatch["checkpoint_id"] == checkpoint.id and dispatch["approved_outline_locked"] is True  # nosec B101 -- Persist the actual approved generation before workers can consume it.
    assert dispatch["job_uuid"] == manager.get_job(int(approved.active_job_id))["uuid"]  # nosec B101 -- Bind the exact admitted Job incarnation.


async def test_resume_retains_real_edited_outline_generation_and_policy(tmp_path, monkeypatch):
    manager, db, service, approved, checkpoint = _edited_outline_fixture(tmp_path, monkeypatch)
    service.pause_run(owner_user_id="1", session_id=approved.id)
    calls = await _one_phase(tmp_path, monkeypatch, manager, approved)
    paused = db.get_session(approved.id)
    assert calls == [] and paused.control_state == "paused"  # nosec B101 -- Park before synthesis without making a provider call.
    resumed = service.resume_run(owner_user_id="1", session_id=approved.id)
    job = manager.get_job(int(resumed.active_job_id))
    assert (
        job["payload"]["checkpoint_id"],
        job["payload"].get("approved_outline_locked"),
        job["payload"]["policy_version"],
    ) == (checkpoint.id, True, paused.phase_dispatch_json["policy_version"])  # nosec B101 -- Resume must retain checkpoint policy and locked-outline authority.
    assert (
        resumed.phase_dispatch_json["job_uuid"] == job["uuid"]
        and resumed.phase_dispatch_json["dispatch_id"] == job["payload"]["dispatch_id"]
    )  # nosec B101 -- A parked completed delivery needs one fresh, durably bound delivery identity.


async def test_initial_dispatch_is_committed_before_job_becomes_visible(tmp_path):
    manager = JobManager(tmp_path / "creation-jobs.db", backend="sqlite")
    db = ResearchSessionsDB(tmp_path / "research.db")
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    create = manager.create_job
    observations = []

    def observe_create(**kwargs):
        current = db.get_session(kwargs["payload"]["session_id"])
        observations.append(
            bool(current and current.phase_dispatch_json and current.phase_dispatch_json["job_uuid"] is None)
        )
        return create(**kwargs)

    manager.create_job = observe_create
    result = service.create_session(
        owner_user_id="1", query="Rowan", source_policy="local_only", autonomy_mode="autonomous"
    )
    assert observations == [True] and result.active_job_id  # nosec B101 -- A worker must see durable pending authority before queue admission.


async def test_pause_during_enqueue_preserves_the_exact_pending_candidate(tmp_path):
    manager = JobManager(tmp_path / "pause-enqueue-jobs.db", backend="sqlite")
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    create = manager.create_job
    candidates = []

    def pause_after_create(**kwargs):
        job = create(**kwargs)
        candidates.append(job)
        service.pause_run(owner_user_id="1", session_id=kwargs["payload"]["session_id"])
        return job

    manager.create_job = pause_after_create
    result = service.create_session(
        owner_user_id="1", query="Rowan", source_policy="local_only", autonomy_mode="autonomous"
    )
    candidate = manager.get_job(candidates[0]["id"])
    assert (result.control_state, result.active_job_id, candidate["status"]) == (
        "paused",
        str(candidate["id"]),
        "queued",
    )  # nosec B101 -- A racing pause must not poison the fixed idempotency key with cancellation.
    assert result.phase_dispatch_json["job_uuid"] == candidate["uuid"]  # nosec B101 -- Retain the exact parked candidate for Resume.


async def test_replacement_between_lease_reads_cannot_admit_retired_phase(tmp_path, monkeypatch):
    manager, db, session = _fixture(tmp_path, monkeypatch)
    get_job = manager.get_job
    replacement = []
    stop = asyncio.Event()

    def replace_after_identity_read(job_id, **kwargs):
        row = get_job(job_id, **kwargs)
        if kwargs.get("owner_user_id") == "1" and row and row["status"] == "processing" and not replacement:
            assert manager.release_job(  # nosec B101 -- Retire exactly the first snapshot's real lease.
                row["id"],
                worker_id=row["worker_id"],
                lease_id=row["lease_id"],
                enforce=True,
            )
            claimed = manager.acquire_next_job(
                domain="research", queue="default", lease_seconds=60, worker_id="between-reads-worker"
            )
            assert claimed and claimed["uuid"] == row["uuid"] and claimed["lease_id"] != row["lease_id"]  # nosec B101 -- The health read will now see a different live lease.
            replacement.append(claimed)
            stop.set()
        return row

    monkeypatch.setattr(manager, "get_job", replace_after_identity_read)
    try:
        calls = await _one_phase(tmp_path, monkeypatch, manager, session, stop=stop)
        assert calls == [] and db.list_artifacts(session.id) == []  # nosec B101 -- Admission needs one exact live lease, not two incompatible snapshots.
    finally:
        for claimed in replacement:
            manager.finalize_cancelled(
                claimed["id"],
                expected_uuid=claimed["uuid"],
                worker_id=claimed["worker_id"],
                lease_id=claimed["lease_id"],
                reason="controlled_test_cleanup",
            )


async def test_pause_between_admission_and_first_phase_halt_is_not_a_failure(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Research import jobs as research_jobs

    manager, db, session = _fixture(tmp_path, monkeypatch)
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    halt = research_jobs._halt_for_control_before_phase

    def pause_before_halt(**kwargs):
        service.pause_run(owner_user_id="1", session_id=session.id)
        return halt(**kwargs)

    monkeypatch.setattr(research_jobs, "_halt_for_control_before_phase", pause_before_halt)
    await _one_phase(tmp_path, monkeypatch, manager, session)
    current = db.get_session(session.id)
    job = manager.get_job(int(session.active_job_id))
    assert (current.status, current.control_state, current.active_job_id, job["status"]) == (
        "queued",
        "paused",
        None,
        "completed",
    )  # nosec B101 -- A legitimate pre-phase pause must park, not exhaust or fail the Job.
    assert db.list_artifacts(session.id) == []  # nosec B101 -- The phase did not start provider or publication work.


@pytest.mark.parametrize("surface", ["get", "snapshot", "list"])
async def test_owned_reload_recovers_initial_unknown_admission_ack_without_duplicate_job(tmp_path, surface):
    manager = JobManager(tmp_path / "unknown-admission.db", backend="sqlite")
    db = ResearchSessionsDB(tmp_path / "research.db")
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    create = manager.create_job
    committed = []

    def lose_ack(**kwargs):
        job = create(**kwargs)
        committed.append(job)
        raise RuntimeError("controlled lost admission acknowledgement")

    manager.create_job = lose_ack
    with pytest.raises(RuntimeError, match="lost admission"):
        service.create_session(owner_user_id="1", query="Rowan", source_policy="local_only", autonomy_mode="autonomous")
    manager.create_job = create
    sessions = db.list_sessions("1")
    assert len(sessions) == 1 and sessions[0].active_job_id is None  # nosec B101 -- Preserve the durable pending intent after unknown acknowledgement.
    if surface == "snapshot":
        snapshot = service.get_stream_snapshot(owner_user_id="1", session_id=sessions[0].id)
        assert "phase_dispatch_json" not in snapshot.model_dump_json()  # nosec B101 -- Internal execution authority must stay out of SSE snapshots.
        recovered = snapshot.run
    elif surface == "list":
        recovered = service.list_sessions(owner_user_id="1", session_id=sessions[0].id)[0]
    else:
        recovered = service.get_session(owner_user_id="1", session_id=sessions[0].id)
    assert (
        recovered.active_job_id == str(committed[0]["id"])
        and db.get_session(sessions[0].id).phase_dispatch_json["job_uuid"] == committed[0]["uuid"]
    )  # nosec B101 -- Reconcile the exact idempotent candidate rather than creating another Job.
    assert len(manager.list_jobs(domain="research", owner_user_id="1")) == 1  # nosec B101 -- Reload must not duplicate admission.


async def test_owned_reload_reconciles_terminal_failure_after_callback_interruption(tmp_path, monkeypatch):
    manager, db, session = _fixture(tmp_path, monkeypatch)
    reconcile = jobs_worker.reconcile_terminal_job

    def interrupted_callback(*args):
        raise RuntimeError("controlled callback interruption")

    monkeypatch.setattr(jobs_worker, "reconcile_terminal_job", interrupted_callback)
    await _one_phase(tmp_path, monkeypatch, manager, session, fail=True)
    assert (
        manager.get_job(int(session.active_job_id))["status"] == "failed"
        and db.get_session(session.id).status == "queued"
    )  # nosec B101 -- Establish the durable Jobs/Research gap left by an interrupted callback.
    monkeypatch.setattr(jobs_worker, "reconcile_terminal_job", reconcile)
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    recovered = service.get_session(owner_user_id="1", session_id=session.id)
    assert (recovered.status, recovered.active_job_id) == ("failed", None)  # nosec B101 -- Normal owner reload must replay exact terminal bookkeeping.
    events = db.list_run_events_after(owner_user_id="1", session_id=session.id, after_id=0)
    service.get_session(owner_user_id="1", session_id=session.id)
    assert db.list_run_events_after(owner_user_id="1", session_id=session.id, after_id=0) == events  # nosec B101 -- Reconciliation is idempotent and does not rerun work.


async def test_granted_handoff_retry_reconciles_dispatch_without_repeating_phase(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerSDK

    class TransientHandoffError(RuntimeError):
        backoff_seconds = 0

    manager, db, session = _fixture(tmp_path, monkeypatch)
    instances = []

    def observed_sdk(manager, config):
        sdk = WorkerSDK(manager, config)
        instances.append(sdk)
        return sdk

    monkeypatch.setattr(jobs_worker, "WorkerSDK", observed_sdk)
    create = manager.create_job
    failures = []

    def fail_first_handoff(**kwargs):
        if kwargs["payload"]["phase"] == "collecting" and not failures:
            failures.append(True)
            raise TransientHandoffError("controlled handoff admission failure")
        return create(**kwargs)

    monkeypatch.setattr(manager, "create_job", fail_first_handoff)
    first = await _one_phase(tmp_path, monkeypatch, manager, session)
    pending = db.get_session(session.id)
    assert first == ["drafting_plan"] and pending.phase == "collecting" and pending.active_job_id is None  # nosec B101 -- The phase committed before admission failed.
    assert manager.get_job(int(session.active_job_id))["status"] == "queued" and pending.status == "queued"  # nosec B101 -- A granted retry is not terminal Research failure.
    complete = manager.complete_job
    receipts = []

    def stop_after_parent(job_id, **kwargs):
        current = db.get_session(session.id)
        receipts.append(current.active_job_id is not None and current.active_job_id != session.active_job_id)
        result = complete(job_id, **kwargs)
        instances[-1].stop()
        return result

    monkeypatch.setattr(manager, "complete_job", stop_after_parent)
    second = await _one_phase(tmp_path, monkeypatch, manager, session)
    recovered = db.get_session(session.id)
    assert second == [] and receipts == [True] and recovered.active_job_id != session.active_job_id  # nosec B101 -- Retry binds the successor before completion without invoking phase work again.
    assert len(db.list_artifacts(session.id)) == 2  # nosec B101 -- Planning artifacts were not regenerated by the handoff retry.


@pytest.mark.parametrize("limit", ["quota", "fair-share"])
async def test_definitive_initial_admission_refusal_does_not_activate_on_reload(tmp_path, monkeypatch, limit):
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED" if limit == "quota" else "JOBS_MAX_PER_USER", "1")
    manager = JobManager(tmp_path / "refused-jobs.db", backend="sqlite")
    blocker = manager.create_job(
        domain="research", queue="default", job_type="research_phase", payload={}, owner_user_id="1"
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    with pytest.raises(ValueError):
        service.create_session(owner_user_id="1", query="Rowan", source_policy="local_only", autonomy_mode="autonomous")
    sessions = db.list_sessions("1")
    assert len(sessions) == 1 and len(manager.list_jobs(domain="research", owner_user_id="1")) == 1  # nosec B101 -- Establish a definitive refusal with zero newly admitted Jobs.
    manager.cancel_job(blocker["id"], expected_uuid=blocker["uuid"], reason="release_test_capacity")
    recovered = service.get_session(owner_user_id="1", session_id=sessions[0].id)
    assert recovered.status == "failed" and recovered.active_job_id is None  # nosec B101 -- A refused Start is inert until an explicit user Retry.
    assert len(manager.list_jobs(domain="research", owner_user_id="1")) == 1  # nosec B101 -- Freeing capacity and reloading must not secretly activate refused work.


def _pending_outline_fixture(tmp_path):
    manager = JobManager(tmp_path / "approval-jobs.db", backend="sqlite")
    db = ResearchSessionsDB(tmp_path / "research.db")
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    session = db.create_session(
        owner_user_id="1",
        query="Rowan",
        source_policy="local_only",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="awaiting_outline_review",
        status="waiting_human",
    )
    checkpoint = db.create_checkpoint(
        session_id=session.id,
        checkpoint_type="outline_review",
        proposed_payload={"outline": {"sections": [{"title": "Original", "focus_area": "history"}]}},
    )
    return manager, db, service, db.get_session(session.id), checkpoint


async def test_checkpoint_approval_losing_to_pause_keeps_checkpoint_pending(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Research import service as service_module

    manager, db, service, session, checkpoint = _pending_outline_fixture(tmp_path)
    apply_patch = service_module.apply_checkpoint_patch

    def pause_before_commit(**kwargs):
        result = apply_patch(**kwargs)
        service.pause_run(owner_user_id="1", session_id=session.id)
        return result

    monkeypatch.setattr(service_module, "apply_checkpoint_patch", pause_before_commit)
    with pytest.raises(ValueError):
        service.approve_checkpoint(
            owner_user_id="1",
            session_id=session.id,
            checkpoint_id=checkpoint.id,
            patch_payload={"sections": [{"title": "Losing edit", "focus_area": "history"}]},
        )
    assert db.get_checkpoint(checkpoint.id).status == "pending"  # nosec B101 -- A losing approval must not consume the pending checkpoint.
    assert db.list_artifacts(session.id) == [] and manager.list_jobs(domain="research", owner_user_id="1") == []  # nosec B101 -- Paused authority cannot publish an approved artifact or Job.


async def test_concurrent_checkpoint_approval_preserves_only_winning_edit(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Research import service as service_module
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

    manager, db, service, session, checkpoint = _pending_outline_fixture(tmp_path)
    apply_patch = service_module.apply_checkpoint_patch
    entered = []
    winner = {"sections": [{"title": "Winning edit", "focus_area": "history"}]}

    def concurrent_approval(**kwargs):
        result = apply_patch(**kwargs)
        if not entered:
            entered.append(True)
            service.approve_checkpoint(
                owner_user_id="1", session_id=session.id, checkpoint_id=checkpoint.id, patch_payload=winner
            )
        return result

    monkeypatch.setattr(service_module, "apply_checkpoint_patch", concurrent_approval)
    with pytest.raises(ValueError):
        service.approve_checkpoint(
            owner_user_id="1",
            session_id=session.id,
            checkpoint_id=checkpoint.id,
            patch_payload={"sections": [{"title": "Losing edit", "focus_area": "history"}]},
        )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    assert db.get_checkpoint(checkpoint.id).user_patch_payload == winner  # nosec B101 -- The losing approval cannot replace the winner's canonical authorization.
    assert (
        store.read_json(session_id=session.id, artifact_name="approved_outline.json")["sections"][0]["title"]
        == "Winning edit"
    )  # nosec B101 -- The next phase must read the winning immutable outline.
    assert len(db.list_artifacts(session.id)) == 1 and len(manager.list_jobs(domain="research", owner_user_id="1")) == 1  # nosec B101 -- Concurrent approval admits and publishes once.


async def test_new_checkpoint_authority_is_not_cancelled_by_previous_terminal_job(tmp_path, monkeypatch):
    manager = JobManager(tmp_path / "parent-failure.db", backend="sqlite")
    db = ResearchSessionsDB(tmp_path / "research.db")
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    session = service.create_session(
        owner_user_id="1", query="Rowan", source_policy="local_only", autonomy_mode="checkpointed"
    )
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: manager)

    def lose_parent_completion(job_id, **kwargs):
        manager.fail_job(
            job_id,
            error="controlled completion interruption",
            retryable=False,
            worker_id=kwargs.get("worker_id"),
            lease_id=kwargs.get("lease_id"),
            completion_token=kwargs.get("completion_token"),
            enforce=True,
        )
        return False

    monkeypatch.setattr(manager, "complete_job", lose_parent_completion)
    await _one_phase(tmp_path, monkeypatch, manager, session)
    parked = db.get_session(session.id)
    assert parked.status == "waiting_human" and manager.get_job(int(session.active_job_id))["status"] == "failed"  # nosec B101 -- The completed phase produced a real checkpoint before its parent acknowledgement failed.
    create = manager.create_job

    def lose_new_ack(**kwargs):
        raise RuntimeError("controlled new approval admission acknowledgement")

    monkeypatch.setattr(manager, "create_job", lose_new_ack)
    with pytest.raises(RuntimeError, match="new approval"):
        service.approve_checkpoint(owner_user_id="1", session_id=session.id, checkpoint_id=parked.latest_checkpoint_id)
    monkeypatch.setattr(manager, "create_job", create)
    recovered = service.get_session(owner_user_id="1", session_id=session.id)
    assert recovered.status == "queued" and recovered.active_job_id is not None  # nosec B101 -- A new explicit checkpoint decision must not inherit an old delivery's terminal failure.
    job = manager.get_job(int(recovered.active_job_id))
    assert job["payload"]["checkpoint_id"] == parked.latest_checkpoint_id and job["status"] == "queued"  # nosec B101 -- Recovery binds only the newly approved generation.


@pytest.mark.parametrize("failure", ["before_commit", "after_commit", "enqueue_ack"])
async def test_checkpoint_commit_recovery_preserves_one_winning_artifact(tmp_path, monkeypatch, failure):
    manager, db, service, session, checkpoint = _pending_outline_fixture(tmp_path)
    before_events = db.list_run_events_after(owner_user_id="1", session_id=session.id, after_id=0)
    if failure == "before_commit":

        def interrupt_registration(*args, **kwargs):
            raise RuntimeError("controlled approval failure")

        monkeypatch.setattr(ResearchSessionsDB, "_record_artifact_with_conn", interrupt_registration)
    elif failure == "after_commit":
        commit = ResearchSessionsDB.commit_checkpoint_approval

        def lose_commit_ack(*args, **kwargs):
            commit(*args, **kwargs)
            raise RuntimeError("controlled approval failure")

        monkeypatch.setattr(ResearchSessionsDB, "commit_checkpoint_approval", lose_commit_ack)
    else:
        create = manager.create_job

        def lose_enqueue_ack(**kwargs):
            create(**kwargs)
            raise RuntimeError("controlled approval failure")

        monkeypatch.setattr(manager, "create_job", lose_enqueue_ack)
    with pytest.raises(RuntimeError, match="controlled approval"):
        service.approve_checkpoint(
            owner_user_id="1",
            session_id=session.id,
            checkpoint_id=checkpoint.id,
            patch_payload={"sections": [{"title": "Winning edit", "focus_area": "history"}]},
        )
    if failure == "before_commit":
        assert (
            db.get_checkpoint(checkpoint.id).status == "pending"
            and db.get_session(session.id).status == "waiting_human"
        )  # nosec B101 -- A failed atomic commit must leave both authority records unchanged.
        assert (
            db.list_artifacts(session.id) == []
            and db.list_run_events_after(owner_user_id="1", session_id=session.id, after_id=0) == before_events
        )  # nosec B101 -- Rollback leaves no partial manifest or events.
        assert [path for path in (tmp_path / "outputs").rglob("*") if path.is_file()] == []  # nosec B101 -- The losing unique staged file is removed.
        return
    monkeypatch.undo()
    recovered = service.get_session(owner_user_id="1", session_id=session.id)
    artifacts = db.list_artifacts(session.id)
    assert (
        recovered.status == "queued"
        and recovered.active_job_id is not None
        and db.get_checkpoint(checkpoint.id).status == "resolved"
    )  # nosec B101 -- Reload recovers only the already committed approval.
    assert len(artifacts) == 1 and Path(artifacts[0].storage_path).is_file()  # nosec B101 -- Unknown acknowledgement cannot delete the canonical artifact.
    events = db.list_run_events_after(owner_user_id="1", session_id=session.id, after_id=0)
    assert (
        len([event for event in events if event.event_type == "artifact"]) == 1
        and len([event for event in events if event.event_type == "checkpoint"]) == 1
    )  # nosec B101 -- Recovery does not duplicate winning artifact/checkpoint events.
    assert len(manager.list_jobs(domain="research", owner_user_id="1")) == 1  # nosec B101 -- Unknown enqueue acknowledgement reuses the same admitted candidate.


@pytest.mark.parametrize("autonomy", ["autonomous", "checkpointed"])
async def test_real_sdk_completes_all_phases_at_one_job_capacity(tmp_path, monkeypatch, autonomy):
    from tldw_Server_API.app.core.Research import jobs as phase_jobs
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.broker import ResearchBroker
    from tldw_Server_API.app.core.Research.synthesizer import ResearchSynthesizer

    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")
    monkeypatch.setenv("JOBS_MAX_PER_USER", "1")
    manager = JobManager(tmp_path / "chain-jobs.db", backend="sqlite")
    db = ResearchSessionsDB(tmp_path / "research.db")
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    session = service.create_session(
        owner_user_id="1", query="Rowan Observatory telescope", source_policy="local_only", autonomy_mode=autonomy
    )
    stop = asyncio.Event()
    deliveries, synth_calls, approvals = [], [], []

    class LocalProvider:
        async def search(self, **kwargs):
            return [
                {
                    "id": "rowan",
                    "title": "Rowan Observatory",
                    "content": "Rowan Observatory telescope is Selene.",
                    "provider": "local_corpus",
                }
            ]

    class SynthesisProvider:
        async def summarize(self, *, plan, source_registry, evidence_notes, **kwargs):
            synth_calls.append(True)
            source = source_registry[0]
            note = next(note for note in evidence_notes if note.source_id == source.source_id)
            return {
                "outline_sections": [
                    {
                        "title": "Evidence",
                        "focus_area": source.focus_area,
                        "source_ids": [source.source_id],
                        "note_ids": [note.note_id],
                    }
                ],
                "claims": [
                    {
                        "text": "Rowan Observatory telescope is Selene.",
                        "focus_area": source.focus_area,
                        "source_ids": [source.source_id],
                        "citations": [{"source_id": source.source_id}],
                        "confidence": 0.9,
                    }
                ],
                "report_sections": [{"title": "Evidence", "markdown": "Rowan Observatory telescope is Selene."}],
                "unresolved_questions": [],
                "summary": {"mode": "llm_backed"},
            }

    broker = ResearchBroker(local_provider=LocalProvider())
    synthesizer = ResearchSynthesizer(synthesis_provider=SynthesisProvider())
    actual = phase_jobs.handle_research_phase_job

    async def controlled_providers(job, **kwargs):
        deliveries.append((job["payload"]["phase"], job["payload"].get("approved_outline_locked", False)))
        return await actual(job, broker=broker, synthesizer=synthesizer, **kwargs)

    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: manager)
    monkeypatch.setattr(jobs_worker, "handle_research_phase_job", controlled_providers)

    async def review_and_watch():
        while not stop.is_set():
            current = db.get_session(session.id)
            if current.status == "waiting_human":
                checkpoint = db.get_checkpoint(current.latest_checkpoint_id)
                patch = None
                if checkpoint.checkpoint_type == "outline_review":
                    section = dict(checkpoint.proposed_payload["outline"]["sections"][0])
                    patch = {"sections": [{"title": "Reviewed evidence", "focus_area": section["focus_area"]}]}
                service.approve_checkpoint(
                    owner_user_id="1", session_id=session.id, checkpoint_id=checkpoint.id, patch_payload=patch
                )
                approvals.append(checkpoint.checkpoint_type)
            elif current.status in {"failed", "cancelled"}:
                raise AssertionError(
                    "The real phase chain terminated before packaging: " + str(current.progress_message)
                )
            elif current.status == "completed":
                if all(job["status"] == "completed" for job in manager.list_jobs(domain="research", owner_user_id="1")):
                    stop.set()
                    return
            await asyncio.sleep(0.01)

    worker = asyncio.create_task(
        jobs_worker.run_research_jobs_worker(
            stop, research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs"
        )
    )
    watch = asyncio.create_task(review_and_watch())
    try:
        await asyncio.wait_for(asyncio.gather(worker, watch), timeout=10)
    finally:
        stop.set()
        for task in (worker, watch):
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    bundle = store.read_json(session_id=session.id, artifact_name="bundle.json")
    assert db.get_session(session.id).status == "completed" and bundle is not None  # nosec B101 -- Full SDK progression must create the actual final package at quota/fair-share capacity one.
    assert "Selene" in bundle["report_markdown"] and bundle["source_inventory"]  # nosec B101 -- The final package retains real pipeline evidence and cited report content.
    assert [phase for phase, locked in deliveries] == (
        ["drafting_plan", "collecting", "synthesizing", "packaging"]
        if autonomy == "autonomous"
        else ["drafting_plan", "collecting", "synthesizing", "synthesizing", "packaging"]
    )  # nosec B101 -- Each authorized phase runs once, with only the legitimate edited-outline synthesis repeated.
    assert approvals == ([] if autonomy == "autonomous" else ["plan_review", "sources_review", "outline_review"])  # nosec B101 -- Checkpointed progression retains all three human decisions.
    if autonomy == "checkpointed":
        assert (
            deliveries[-2] == ("synthesizing", True)
            and bundle["outline"]["sections"][0]["title"] == "Reviewed evidence"
        )  # nosec B101 -- Packaging must preserve the actually edited locked outline.


@pytest.mark.parametrize("resume_binds", [False, True])
async def test_old_handoff_cannot_cancel_candidate_retained_by_explicit_resume(tmp_path, monkeypatch, resume_binds):
    manager, db, session = _fixture(tmp_path, monkeypatch)
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    create = manager.create_job
    candidates = []

    def resume_after_enqueue(**kwargs):
        job = create(**kwargs)
        if kwargs["payload"]["phase"] == "collecting" and not candidates:
            candidates.append(job)
            service.pause_run(owner_user_id="1", session_id=session.id)
            if resume_binds:
                service.resume_run(owner_user_id="1", session_id=session.id)
            else:
                manager.create_job = lambda **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("controlled Resume acknowledgement")
                )
                try:
                    with pytest.raises(RuntimeError, match="Resume acknowledgement"):
                        service.resume_run(owner_user_id="1", session_id=session.id)
                finally:
                    manager.create_job = resume_after_enqueue
        return job

    monkeypatch.setattr(manager, "create_job", resume_after_enqueue)
    await _one_phase(tmp_path, monkeypatch, manager, session)
    candidate = manager.get_job(candidates[0]["id"])
    assert candidate["status"] == "queued"  # nosec B101 -- The retired dispatcher's cleanup cannot revoke the exact candidate now authorized by Resume.
    monkeypatch.setattr(manager, "create_job", create)
    recovered = service.get_session(owner_user_id="1", session_id=session.id)
    assert (
        recovered.status == "queued"
        and recovered.active_job_id == str(candidate["id"])
        and recovered.phase_dispatch_json["authorization"] == "resume"
    )  # nosec B101 -- Preserve the latest explicit decision while binding the same unknown-ACK candidate.
    assert len(manager.list_jobs(domain="research", owner_user_id="1")) == 2  # nosec B101 -- Resume recovery adds no duplicate successor.


@pytest.mark.parametrize("outcome", ["failed", "cancelled", "completed"])
async def test_terminal_candidate_is_not_reported_executable_after_unknown_bind(tmp_path, outcome):
    manager = JobManager(tmp_path / "terminal-candidate.db", backend="sqlite")
    db = ResearchSessionsDB(tmp_path / "research.db")
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    create = manager.create_job

    def terminal_before_bind(**kwargs):
        created = create(**kwargs)
        leased = manager.acquire_next_job(
            domain="research", queue="default", lease_seconds=60, worker_id="controlled-candidate"
        )
        if outcome == "failed":
            assert manager.fail_job(
                leased["id"],
                error="controlled candidate failure",
                retryable=False,
                worker_id=leased["worker_id"],
                lease_id=leased["lease_id"],
                enforce=True,
            )  # nosec B101 -- Establish a real terminal outcome before binding acknowledgement.
        elif outcome == "cancelled":
            assert manager.finalize_cancelled(
                leased["id"],
                expected_uuid=leased["uuid"],
                worker_id=leased["worker_id"],
                lease_id=leased["lease_id"],
                reason="controlled candidate cancellation",
            )  # nosec B101 -- Establish exact cancellation of the admitted candidate.
        else:
            assert manager.complete_job(
                leased["id"],
                expected_uuid=leased["uuid"],
                worker_id=leased["worker_id"],
                lease_id=leased["lease_id"],
                enforce=True,
            )  # nosec B101 -- Establish an ended delivery without claiming a Research phase completed.
        return created

    manager.create_job = terminal_before_bind
    result = service.create_session(
        owner_user_id="1", query="Rowan", source_policy="local_only", autonomy_mode="autonomous"
    )
    assert result.status == ("cancelled" if outcome == "cancelled" else "failed") and result.active_job_id is None  # nosec B101 -- A terminal idempotent candidate must never be displayed as executable queued work.
    manager.create_job = create
    service.get_session(owner_user_id="1", session_id=result.id)
    assert len(manager.list_jobs(domain="research", owner_user_id="1")) == 1 and db.list_artifacts(result.id) == []  # nosec B101 -- Reload neither hides failure nor silently retries the ended delivery.


async def test_acquired_unbound_candidate_defers_without_spending_retry(tmp_path, monkeypatch):
    from dataclasses import replace

    manager = JobManager(tmp_path / "unbound.db", backend="sqlite")
    db = ResearchSessionsDB(tmp_path / "research.db")
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    create = manager.create_job
    candidates = []

    def unknown_ack(**kwargs):
        candidates.append(create(**kwargs))
        raise RuntimeError("controlled unknown acknowledgement")

    manager.create_job = unknown_ack
    with pytest.raises(RuntimeError, match="unknown acknowledgement"):
        service.create_session(owner_user_id="1", query="Rowan", source_policy="local_only", autonomy_mode="autonomous")
    manager.create_job = create
    pending = db.list_sessions("1")[0]
    stop = asyncio.Event()
    release = manager.release_job

    def stop_after_release(*args, **kwargs):
        result = release(*args, **kwargs)
        stop.set()
        return result

    monkeypatch.setattr(manager, "release_job", stop_after_release)
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: manager)
    calls = await _one_phase(
        tmp_path, monkeypatch, manager, replace(pending, active_job_id=str(candidates[0]["id"])), stop=stop
    )
    job = manager.get_job(candidates[0]["id"])
    assert calls == [] and job["status"] == "queued" and job["retry_count"] == 0 and db.list_artifacts(pending.id) == []  # nosec B101 -- Acquisition before binding is a deferred delivery, never provider work or a consumed retry.
    recovered = service.get_session(owner_user_id="1", session_id=pending.id)
    assert recovered.active_job_id == str(job["id"])  # nosec B101 -- Normal reload binds the exact same candidate.


async def test_held_acquired_candidate_remains_valid_across_explicit_resume(tmp_path, monkeypatch):
    from dataclasses import replace

    from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerSDK

    manager = JobManager(tmp_path / "held.db", backend="sqlite")
    db = ResearchSessionsDB(tmp_path / "research.db")
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    create = manager.create_job
    candidates = []

    def unknown_ack(**kwargs):
        candidates.append(create(**kwargs))
        raise RuntimeError("controlled unknown acknowledgement")

    manager.create_job = unknown_ack
    with pytest.raises(RuntimeError, match="unknown acknowledgement"):
        service.create_session(owner_user_id="1", query="Rowan", source_policy="local_only", autonomy_mode="autonomous")
    manager.create_job = create
    pending = db.list_sessions("1")[0]
    observations = []

    def observed_sdk(jm, config):
        sdk = WorkerSDK(jm, config)
        run = sdk.run

        async def observe_run(*, handler, **kwargs):
            async def held(job):
                service.pause_run(owner_user_id="1", session_id=pending.id)
                resumed = service.resume_run(owner_user_id="1", session_id=pending.id)
                observations.append(
                    resumed.active_job_id == str(job["id"]) and resumed.phase_dispatch_json["authorization"] == "resume"
                )
                return await handler(job)

            return await run(handler=held, **kwargs)

        sdk.run = observe_run
        return sdk

    monkeypatch.setattr(jobs_worker, "WorkerSDK", observed_sdk)
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: manager)
    calls = await _one_phase(tmp_path, monkeypatch, manager, replace(pending, active_job_id=str(candidates[0]["id"])))
    assert observations == [True] and calls == ["drafting_plan"]  # nosec B101 -- The held exact candidate runs once only after the current explicit Resume authorizes it.
    assert (
        manager.get_job(candidates[0]["id"])["status"] == "completed"
        and db.get_session(pending.id).phase == "collecting"
    )  # nosec B101 -- Exact lease and successor progression survive the decision race.
    assert len(manager.list_jobs(domain="research", owner_user_id="1")) == 2  # nosec B101 -- Rebinding the held candidate creates only the legitimate successor.


async def test_old_dispatcher_cannot_cancel_phase_that_published_successor(tmp_path, monkeypatch):
    import threading

    from tldw_Server_API.app.core.Research.broker import ResearchBroker

    manager = JobManager(tmp_path / "advanced.db", backend="sqlite")
    db = ResearchSessionsDB(tmp_path / "research.db")
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    create, complete = manager.create_job, manager.complete_job
    reached, release = threading.Event(), threading.Event()
    candidates, threads, errors = [], [], []
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: manager)

    def admit_and_advance(**kwargs):
        created = create(**kwargs)
        if not candidates:
            candidates.append(created)
            service.get_session(owner_user_id="1", session_id=kwargs["payload"]["session_id"])

            def work():
                async def run():
                    stop = asyncio.Event()

                    def hold_completion(job_id, **values):
                        reached.set()
                        if not release.wait(timeout=5):
                            raise RuntimeError("controlled completion barrier timed out")
                        result = complete(job_id, **values)
                        stop.set()
                        return result

                    manager.complete_job = hold_completion
                    await jobs_worker.run_research_jobs_worker(
                        stop, research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs"
                    )

                try:
                    asyncio.run(run())
                except BaseException as error:  # noqa: BLE001 -- Propagate every owned test-thread failure through the asserted result.
                    errors.append(error)

            thread = threading.Thread(target=work)
            threads.append(thread)
            thread.start()
            if not reached.wait(timeout=5):
                raise RuntimeError("controlled phase did not reach completion")
        return created

    manager.create_job = admit_and_advance
    try:
        result = service.create_session(
            owner_user_id="1", query="Rowan", source_policy="local_only", autonomy_mode="autonomous"
        )
        parent = manager.get_job(candidates[0]["id"])
        assert parent["status"] == "processing" and not parent.get("cancel_requested_at")  # nosec B101 -- A bind-CAS loser cannot cancel a phase that already legitimately published its successor.
        assert result.phase == "collecting" and result.active_job_id != str(parent["id"])  # nosec B101 -- Observe the actual committed successor while parent completion remains held.
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=5)
        manager.create_job, manager.complete_job = create, complete
    assert (
        errors == []
        and not any(thread.is_alive() for thread in threads)
        and manager.get_job(candidates[0]["id"])["status"] == "completed"
    )  # nosec B101 -- Release and drain the real SDK parent without changing its truthful outcome.
    actual = jobs_worker.handle_research_phase_job
    searches = []

    class LocalProvider:
        async def search(self, **kwargs):
            searches.append(True)
            return []

    async def collect(job, **kwargs):
        return await actual(job, broker=ResearchBroker(local_provider=LocalProvider()), **kwargs)

    monkeypatch.setattr(jobs_worker, "handle_research_phase_job", collect)
    calls = await _one_phase(tmp_path, monkeypatch, manager, db.get_session(result.id))
    assert calls == ["collecting"] and searches and db.get_session(result.id).phase == "synthesizing"  # nosec B101 -- The successor executes normally; terminal reconciliation cannot hide an induced parent cancellation.


async def test_reload_refuses_completed_bound_delivery_without_research_completion(tmp_path):
    manager = JobManager(tmp_path / "ended-bound.db", backend="sqlite")
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    session = service.create_session(
        owner_user_id="1", query="Rowan", source_policy="local_only", autonomy_mode="autonomous"
    )
    job = manager.acquire_next_job(
        domain="research", queue="default", lease_seconds=60, worker_id="controlled-ended-delivery"
    )
    assert manager.complete_job(
        job["id"], expected_uuid=job["uuid"], worker_id=job["worker_id"], lease_id=job["lease_id"], enforce=True
    )  # nosec B101 -- Establish an ended delivery without falsely advancing Research state.
    recovered = service.get_session(owner_user_id="1", session_id=session.id)
    assert recovered.status == "failed" and recovered.active_job_id is None  # nosec B101 -- A later terminal outcome cannot leave reload displaying permanently queued work.
    assert len(manager.list_jobs(domain="research", owner_user_id="1")) == 1  # nosec B101 -- Refusal does not invent a replacement delivery or consume another retry.


async def test_resume_refuses_legacy_phase_without_verified_dispatch(tmp_path):
    manager = JobManager(tmp_path / "legacy-ambiguous.db", backend="sqlite")
    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Rowan",
        source_policy="local_only",
        autonomy_mode="autonomous",
        limits_json={},
        phase="synthesizing",
        status="queued",
    )
    db.update_control_state(session.id, control_state="paused")
    before = db.get_session(session.id)
    service = ResearchService(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs", job_manager=manager
    )
    with pytest.raises(ValueError, match="research_phase_recovery_ambiguous"):
        service.resume_run(owner_user_id="1", session_id=session.id)
    assert db.get_session(session.id) == before and manager.list_jobs(domain="research", owner_user_id="1") == []  # nosec B101 -- Legacy recovery must not invent checkpoint or locked-outline authority.

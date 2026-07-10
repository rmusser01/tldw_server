from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management import Watchlists_DB as watchlists_module
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

pytestmark = pytest.mark.unit


def _make_backend(tmp_path):
    return DatabaseBackendFactory.create_backend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(tmp_path / "briefing_occurrences.db"),
        )
    )


def _make_db(tmp_path, *, user_id: int = 1) -> WatchlistsDatabase:
    return WatchlistsDatabase(user_id=user_id, backend=_make_backend(tmp_path))


def _create_job_run(db: WatchlistsDatabase, *, label: str) -> tuple[int, int]:
    job = db.create_job(
        name=f"{label} job",
        description=None,
        scope_json=json.dumps({}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=1,
        per_host_delay_ms=0,
        retry_policy_json=json.dumps({}),
        output_prefs_json=json.dumps({}),
        job_filters_json=None,
    )
    run = db.create_run(int(job.id), status="finished")
    return int(job.id), int(run.id)


def test_sqlite_schema_contains_briefing_occurrence_contract_and_indexes(tmp_path):
    db = _make_db(tmp_path)

    columns = {row["name"] for row in db.backend.get_table_info("watchlist_briefing_occurrences")}
    indexes = {
        row["name"]
        for row in db.backend.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = ?",
            ("watchlist_briefing_occurrences",),
        ).rows
    }

    assert columns == {
        "id",
        "user_id",
        "job_id",
        "run_id",
        "occurrence_key",
        "contract_json",
        "stages_json",
        "artifact_status",
        "delivery_status",
        "output_id",
        "audio_task_id",
        "delivery_task_id",
        "selected_count",
        "omitted_count",
        "created_at",
        "updated_at",
    }
    assert {
        "idx_briefing_occurrences_user_job",
        "idx_briefing_occurrences_run",
    }.issubset(indexes)

    attempt_columns = {
        row["name"] for row in db.backend.get_table_info("watchlist_briefing_attempts")
    }
    assert {
        "id",
        "user_id",
        "occurrence_id",
        "artifact_version",
        "adapter",
        "attempt",
        "state",
        "requested_stage",
        "scheduler_task_id",
        "request_id",
        "workflow_run_id",
        "artifact_id",
        "code",
        "created_at",
        "updated_at",
    } == attempt_columns


def test_attempt_claim_and_transition_are_atomic_and_attempt_scoped(tmp_path):
    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label="attempt")
    occurrence = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=f"user:1:job:{job_id}:run:{run_id}:attempt",
        contract_json='{"version":1}',
    )

    first = db.claim_briefing_attempt(
        occurrence_id=int(occurrence.id),
        artifact_version=1,
        adapter="audio",
        requested_stage="generate_audio",
    )
    duplicate = db.claim_briefing_attempt(
        occurrence_id=int(occurrence.id),
        artifact_version=1,
        adapter="audio",
        requested_stage="generate_audio",
    )
    queued = db.transition_briefing_attempt(
        int(first.id),
        expected_states={"intent"},
        state="queued",
        scheduler_task_id="task-1",
        request_id="wla_attempt_1",
    )
    stale = db.transition_briefing_attempt(
        int(first.id),
        expected_states={"intent"},
        state="queued",
        scheduler_task_id="stale-task",
    )
    terminal = db.transition_briefing_attempt(
        int(first.id),
        expected_states={"queued"},
        state="successful",
        workflow_run_id="workflow-1",
        artifact_id="audio-1",
    )
    db.update_briefing_occurrence(
        int(occurrence.id),
        stages={"generate_audio": {"status": "ready", "attempt_count": 1}},
        artifact_status="ready",
    )
    stale_occurrence = db.update_briefing_occurrence_for_attempt(
        int(occurrence.id),
        int(first.id),
        expected_attempt_states={"queued"},
        stages={"generate_audio": {"status": "queued", "attempt_count": 1}},
        artifact_status="running",
        audio_task_id="stale-task",
    )
    retry = db.claim_briefing_attempt(
        occurrence_id=int(occurrence.id),
        artifact_version=1,
        adapter="audio",
        requested_stage="generate_audio",
        allow_retry=True,
    )

    assert duplicate.id == first.id
    assert queued is not None and queued.scheduler_task_id == "task-1"
    assert stale is None
    assert stale_occurrence is None
    assert terminal is not None and terminal.state == "successful"
    assert json.loads(db.get_briefing_occurrence(int(occurrence.id)).stages_json)["generate_audio"][
        "status"
    ] == "ready"
    assert retry.id == first.id
    assert retry.attempt == 1
    assert retry.state == "successful"


def test_successful_attempt_cannot_be_claimed_as_retry(tmp_path):
    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label="successful-attempt")
    occurrence = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=f"user:1:job:{job_id}:run:{run_id}:successful",
        contract_json='{"version":1}',
    )
    attempt = db.claim_briefing_attempt(
        occurrence_id=int(occurrence.id),
        artifact_version=1,
        adapter="email",
    )
    db.transition_briefing_attempt(
        int(attempt.id),
        expected_states={"intent"},
        state="successful",
    )

    claimed = db.claim_briefing_attempt(
        occurrence_id=int(occurrence.id),
        artifact_version=1,
        adapter="email",
        allow_retry=True,
    )

    assert claimed.id == attempt.id
    assert claimed.attempt == 1
    assert claimed.state == "successful"


def test_attempt_terminal_and_occurrence_stage_roll_back_together(tmp_path, monkeypatch):
    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label="atomic-terminal")
    occurrence = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=f"user:1:job:{job_id}:run:{run_id}:atomic",
        contract_json='{"version":1}',
    )
    db.update_briefing_occurrence(
        int(occurrence.id),
        stages={"deliver:email": {"status": "running", "outcome": "sending", "attempt_count": 1}},
    )
    attempt = db.claim_briefing_attempt(
        occurrence_id=int(occurrence.id),
        artifact_version=1,
        adapter="email",
    )
    db.transition_briefing_attempt(
        int(attempt.id),
        expected_states={"intent"},
        state="sending",
    )
    original_execute = db.backend.execute

    def fail_occurrence_write(query, params=(), **kwargs):
        normalized = " ".join(str(query).split())
        if normalized.startswith("UPDATE watchlist_briefing_occurrences SET stages_json"):
            raise RuntimeError("simulated occurrence write crash")
        return original_execute(query, params, **kwargs)

    monkeypatch.setattr(db.backend, "execute", fail_occurrence_write)
    with pytest.raises(RuntimeError, match="simulated occurrence write crash"):
        db.finalize_briefing_attempt(
            int(attempt.id),
            expected_states={"sending"},
            state="successful",
            stage_updates={
                "deliver:email": {
                    "status": "ready",
                    "outcome": "successful",
                    "attempt_count": 1,
                }
            },
            delivery_status="delivered",
        )
    monkeypatch.setattr(db.backend, "execute", original_execute)

    assert db.get_briefing_attempt(int(attempt.id)).state == "sending"
    persisted = json.loads(db.get_briefing_occurrence(int(occurrence.id)).stages_json)
    assert persisted["deliver:email"]["outcome"] == "sending"


def test_delivery_aggregate_uses_sibling_stage_committed_after_caller_snapshot(tmp_path):
    from tldw_Server_API.app.core.Watchlists.briefing_delivery import (
        reconcile_successful_delivery_attempt,
    )

    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label="atomic-delivery-aggregate")
    occurrence = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=f"user:1:job:{job_id}:run:{run_id}:atomic-delivery-aggregate",
        contract_json=json.dumps(
            {
                "version": 1,
                "delivery": {
                    "email": {"enabled": True},
                    "chatbook": {"enabled": True},
                },
            }
        ),
    )
    db.update_briefing_occurrence(
        int(occurrence.id),
        stages={
            "persist_text": {"status": "ready", "output_version": 1},
            "deliver": {"status": "running", "code": "delivering"},
            "deliver:email": {"status": "running", "outcome": "sending", "attempt_count": 1},
            "deliver:chatbook": {"status": "running", "outcome": "sending", "attempt_count": 1},
        },
        delivery_status="delivering",
    )
    email_attempt = db.claim_briefing_attempt(occurrence_id=int(occurrence.id), artifact_version=1, adapter="email")
    chatbook_attempt = db.claim_briefing_attempt(
        occurrence_id=int(occurrence.id), artifact_version=1, adapter="chatbook"
    )
    db.transition_briefing_attempt(int(email_attempt.id), expected_states={"intent"}, state="successful")
    db.transition_briefing_attempt(int(chatbook_attempt.id), expected_states={"intent"}, state="sending")
    snapshot_barrier = Barrier(2)
    sibling_committed = Event()

    def reconcile_email():
        stale_occurrence = db.get_briefing_occurrence(int(occurrence.id))
        snapshot_barrier.wait()
        assert sibling_committed.wait(timeout=5)
        return reconcile_successful_delivery_attempt(
            watchlists_db=db,
            occurrence=stale_occurrence,
            adapter="email",
        )

    def finalize_chatbook():
        snapshot_barrier.wait()
        try:
            return db.finalize_briefing_attempt(
                int(chatbook_attempt.id),
                expected_states={"sending"},
                state="successful",
                stage_updates={
                    "deliver:chatbook": {
                        "status": "ready",
                        "outcome": "successful",
                        "attempt_count": 1,
                    }
                },
                delivery_status="partially_delivered",
            )
        finally:
            sibling_committed.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        email_future = executor.submit(reconcile_email)
        chatbook_future = executor.submit(finalize_chatbook)
        assert chatbook_future.result(timeout=10) is not None
        assert email_future.result(timeout=10) is not None

    persisted = db.get_briefing_occurrence(int(occurrence.id))
    stages = json.loads(persisted.stages_json)
    assert stages["deliver:email"]["outcome"] == "successful"
    assert stages["deliver:chatbook"]["outcome"] == "successful"
    assert persisted.delivery_status == "delivered"
    assert stages["deliver"]["status"] == "ready"


@pytest.mark.parametrize(
    ("chatbook_outcome", "expected_status"),
    [("unknown", "unknown"), ("failed", "partially_delivered")],
)
def test_finalize_briefing_attempt_recomputes_delivery_aggregate_in_transaction(
    tmp_path,
    chatbook_outcome,
    expected_status,
):
    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label=f"aggregate-{chatbook_outcome}")
    occurrence = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=f"user:1:job:{job_id}:run:{run_id}:aggregate:{chatbook_outcome}",
        contract_json='{"version":1}',
    )
    db.update_briefing_occurrence(
        int(occurrence.id),
        stages={
            "deliver:email": {"status": "running", "outcome": "sending", "attempt_count": 1},
            "deliver:chatbook": {"status": "failed", "outcome": chatbook_outcome, "attempt_count": 1},
        },
        delivery_status="delivering",
    )
    attempt = db.claim_briefing_attempt(occurrence_id=int(occurrence.id), artifact_version=1, adapter="email")
    db.transition_briefing_attempt(int(attempt.id), expected_states={"intent"}, state="sending")

    finalized = db.finalize_briefing_attempt(
        int(attempt.id),
        expected_states={"sending"},
        state="successful",
        stage_updates={
            "deliver:email": {
                "status": "ready",
                "outcome": "successful",
                "attempt_count": 1,
            }
        },
        configured_delivery_adapters={"email", "chatbook"},
    )

    stages = json.loads(finalized.stages_json)
    assert finalized.delivery_status == expected_status
    assert stages["deliver"]["status"] == "failed"
    assert stages["deliver"]["code"] == expected_status


@pytest.mark.parametrize("terminal_state", ["successful", "failed", "cancelled"])
def test_superseded_audio_callback_is_atomic_noop(tmp_path, terminal_state):
    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label=f"stale-{terminal_state}")
    occurrence = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=f"user:1:job:{job_id}:run:{run_id}:stale:{terminal_state}",
        contract_json='{"version":1}',
    )
    first = db.claim_briefing_attempt(
        occurrence_id=int(occurrence.id),
        artifact_version=1,
        adapter="audio",
    )
    db.transition_briefing_attempt(
        int(first.id),
        expected_states={"intent"},
        state="failed",
    )
    retry = db.claim_briefing_attempt(
        occurrence_id=int(occurrence.id),
        artifact_version=1,
        adapter="audio",
        allow_retry=True,
    )
    db.update_briefing_occurrence(
        int(occurrence.id),
        stages={"generate_audio": {"status": "queued", "attempt_count": int(retry.attempt)}},
        artifact_status="running",
    )

    result = db.finalize_briefing_attempt(
        int(first.id),
        expected_states={"intent", "queued", "sending"},
        state=terminal_state,
        stage_updates={
            "generate_audio": {
                "status": "ready" if terminal_state == "successful" else terminal_state,
                "attempt_count": int(first.attempt),
            }
        },
        artifact_status="ready" if terminal_state == "successful" else terminal_state,
    )

    assert result is None
    persisted = db.get_briefing_occurrence(int(occurrence.id))
    assert persisted.artifact_status == "running"
    assert json.loads(persisted.stages_json)["generate_audio"]["attempt_count"] == retry.attempt


def test_concurrent_attempt_claim_returns_one_intent(tmp_path):
    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label="concurrent-attempt")
    occurrence = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=f"user:1:job:{job_id}:run:{run_id}:attempt",
        contract_json='{"version":1}',
    )
    barrier = Barrier(4)

    def claim() -> int:
        barrier.wait()
        return int(
            db.claim_briefing_attempt(
                occurrence_id=int(occurrence.id),
                artifact_version=1,
                adapter="email",
            ).id
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        ids = list(executor.map(lambda _index: claim(), range(4)))

    assert len(set(ids)) == 1
    assert (
        db.backend.execute(
            "SELECT COUNT(*) AS count FROM watchlist_briefing_attempts WHERE occurrence_id = ?",
            (int(occurrence.id),),
        ).scalar
        == 1
    )


def test_create_or_get_occurrence_is_idempotent_and_preserves_initial_contract(tmp_path):
    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label="daily")
    occurrence_key = f"user:1:job:{job_id}:run:{run_id}:v1"

    first = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=occurrence_key,
        contract_json='{"version":1}',
    )
    second = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=occurrence_key,
        contract_json='{"version":2}',
    )

    assert isinstance(first, watchlists_module.BriefingOccurrenceRow)
    assert second.id == first.id
    assert first.user_id == "1"
    assert first.job_id == job_id
    assert first.run_id == run_id
    assert first.contract_json == '{"version":1}'
    assert first.stages_json == "{}"
    assert first.artifact_status == "running"
    assert first.delivery_status == "waiting_for_artifacts"
    assert first.output_id is None
    assert first.audio_task_id is None
    assert first.delivery_task_id is None
    assert first.selected_count == 0
    assert first.omitted_count == 0
    assert second.contract_json == first.contract_json
    assert second.created_at == first.created_at
    assert db.backend.execute("SELECT COUNT(*) AS count FROM watchlist_briefing_occurrences").scalar == 1


def test_concurrent_create_or_get_occurrence_returns_one_logical_row(tmp_path):
    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label="race")
    occurrence_key = f"user:1:job:{job_id}:run:{run_id}:v1"
    barrier = Barrier(6)

    def create() -> int:
        barrier.wait()
        row = db.create_or_get_briefing_occurrence(
            run_id=run_id,
            occurrence_key=occurrence_key,
            contract_json='{"version":1}',
        )
        return int(row.id)

    with ThreadPoolExecutor(max_workers=6) as pool:
        occurrence_ids = list(pool.map(lambda _: create(), range(6)))

    assert len(set(occurrence_ids)) == 1
    assert db.backend.execute("SELECT COUNT(*) AS count FROM watchlist_briefing_occurrences").scalar == 1


def test_occurrence_create_read_and_update_are_scoped_to_owned_run(tmp_path):
    backend = _make_backend(tmp_path)
    owner = WatchlistsDatabase(user_id=1, backend=backend)
    outsider = WatchlistsDatabase(user_id=2, backend=backend)
    owner_job_id, owner_run_id = _create_job_run(owner, label="owner")
    outsider_job_id, outsider_run_id = _create_job_run(outsider, label="outsider")
    occurrence_key = "shared-logical-key"
    owner_occurrence = owner.create_or_get_briefing_occurrence(
        run_id=owner_run_id,
        occurrence_key=occurrence_key,
        contract_json='{"owner":1}',
    )
    outsider_occurrence = outsider.create_or_get_briefing_occurrence(
        run_id=outsider_run_id,
        occurrence_key=occurrence_key,
        contract_json='{"owner":2}',
    )

    assert owner_occurrence.id != outsider_occurrence.id
    assert owner_occurrence.job_id == owner_job_id
    assert outsider_occurrence.job_id == outsider_job_id
    with pytest.raises(KeyError, match="briefing_occurrence_not_found"):
        outsider.get_briefing_occurrence(int(owner_occurrence.id))
    with pytest.raises(KeyError, match="briefing_occurrence_not_found"):
        outsider.update_briefing_occurrence(
            int(owner_occurrence.id),
            artifact_status="failed",
        )
    with pytest.raises(KeyError, match="run_not_found"):
        outsider.create_or_get_briefing_occurrence(
            run_id=owner_run_id,
            occurrence_key=occurrence_key,
            contract_json='{"owner":2}',
        )
    assert owner.get_briefing_occurrence(int(owner_occurrence.id)).artifact_status == "running"


def test_get_latest_occurrence_is_job_and_user_scoped(tmp_path):
    backend = _make_backend(tmp_path)
    owner = WatchlistsDatabase(user_id=1, backend=backend)
    outsider = WatchlistsDatabase(user_id=2, backend=backend)
    job_id, first_run_id = _create_job_run(owner, label="latest")
    second_run = owner.create_run(job_id, status="finished")
    first = owner.create_or_get_briefing_occurrence(
        run_id=first_run_id,
        occurrence_key="latest:first",
        contract_json='{"version":1}',
    )
    second = owner.create_or_get_briefing_occurrence(
        run_id=int(second_run.id),
        occurrence_key="latest:second",
        contract_json='{"version":1}',
    )

    assert owner.get_latest_briefing_occurrence(job_id).id == second.id
    assert owner.get_latest_briefing_occurrence(job_id).id != first.id
    with pytest.raises(KeyError, match="briefing_occurrence_not_found"):
        outsider.get_latest_briefing_occurrence(job_id)


def test_get_occurrence_for_run_is_exact_and_user_scoped(tmp_path):
    backend = _make_backend(tmp_path)
    owner = WatchlistsDatabase(user_id=1, backend=backend)
    outsider = WatchlistsDatabase(user_id=2, backend=backend)
    job_id, first_run_id = _create_job_run(owner, label="exact-run")
    second_run = owner.create_run(job_id, status="finished")
    first = owner.create_or_get_briefing_occurrence(
        run_id=first_run_id,
        occurrence_key="exact:first",
        contract_json='{"version":1}',
    )
    owner.create_or_get_briefing_occurrence(
        run_id=int(second_run.id),
        occurrence_key="exact:second",
        contract_json='{"version":1}',
    )

    assert owner.get_briefing_occurrence_for_run(first_run_id).id == first.id
    with pytest.raises(KeyError, match="briefing_occurrence_not_found"):
        outsider.get_briefing_occurrence_for_run(first_run_id)


def test_update_occurrence_serializes_stages_and_only_changes_named_fields(tmp_path, monkeypatch):
    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label="update")
    occurrence = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=f"user:1:job:{job_id}:run:{run_id}:v1",
        contract_json='{"version":1}',
    )
    stages = {
        "collect": {"status": "ready"},
        "persist_text": {"status": "ready", "output_id": 901},
    }
    updated_at = "2026-07-10T12:34:56+00:00"
    monkeypatch.setattr(watchlists_module, "_utcnow_iso", lambda: updated_at)

    updated = db.update_briefing_occurrence(
        int(occurrence.id),
        stages=stages,
        artifact_status="ready",
        delivery_status="delivered",
        output_id=901,
        audio_task_id="audio-123",
        delivery_task_id="delivery-456",
        selected_count=4,
        omitted_count=2,
    )

    assert json.loads(updated.stages_json) == stages
    assert updated.artifact_status == "ready"
    assert updated.delivery_status == "delivered"
    assert updated.output_id == 901
    assert updated.audio_task_id == "audio-123"
    assert updated.delivery_task_id == "delivery-456"
    assert updated.selected_count == 4
    assert updated.omitted_count == 2
    assert updated.contract_json == occurrence.contract_json
    assert updated.created_at == occurrence.created_at
    assert updated.updated_at == updated_at
    with pytest.raises(TypeError):
        db.update_briefing_occurrence(  # type: ignore[call-arg]
            int(occurrence.id),
            contract_json='{"version":2}',
        )


def test_update_occurrence_clears_nullable_ids_but_omission_preserves_them(tmp_path):
    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label="nullable")
    occurrence = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=f"user:1:job:{job_id}:run:{run_id}:nullable",
        contract_json='{"version":1}',
    )
    stored = db.update_briefing_occurrence(
        int(occurrence.id),
        output_id=901,
        audio_task_id="audio-123",
        delivery_task_id="delivery-456",
    )

    preserved = db.update_briefing_occurrence(
        int(occurrence.id),
        stages=None,
        artifact_status=None,
        delivery_status=None,
        selected_count=None,
        omitted_count=None,
    )
    assert preserved.output_id == stored.output_id
    assert preserved.audio_task_id == stored.audio_task_id
    assert preserved.delivery_task_id == stored.delivery_task_id
    assert preserved.updated_at == stored.updated_at

    cleared = db.update_briefing_occurrence(
        int(occurrence.id),
        output_id=None,
        audio_task_id=None,
        delivery_task_id=None,
    )
    assert cleared.output_id is None
    assert cleared.audio_task_id is None
    assert cleared.delivery_task_id is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_status", "running"),
        ("artifact_status", "ready"),
        ("artifact_status", "failed"),
        ("artifact_status", "cancelled"),
        ("delivery_status", "not_configured"),
        ("delivery_status", "waiting_for_artifacts"),
        ("delivery_status", "delivering"),
        ("delivery_status", "delivered"),
        ("delivery_status", "partially_delivered"),
        ("delivery_status", "failed"),
        ("delivery_status", "unknown"),
    ],
)
def test_update_occurrence_accepts_exact_lifecycle_values(tmp_path, field, value):
    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label=f"valid-{value}")
    occurrence = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=f"user:1:job:{job_id}:run:{run_id}:{field}:{value}",
        contract_json='{"version":1}',
    )

    updated = db.update_briefing_occurrence(int(occurrence.id), **{field: value})

    assert getattr(updated, field) == value


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"artifact_status": "complete"}, "invalid_briefing_artifact_status"),
        ({"delivery_status": "sent"}, "invalid_briefing_delivery_status"),
        ({"selected_count": -1}, "selected_count_must_be_non_negative"),
        ({"omitted_count": -1}, "omitted_count_must_be_non_negative"),
    ],
)
def test_update_occurrence_rejects_invalid_durable_values_before_sql(
    tmp_path,
    monkeypatch,
    fields,
    error,
):
    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label=error)
    occurrence = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=f"user:1:job:{job_id}:run:{run_id}:{error}",
        contract_json='{"version":1}',
    )

    with monkeypatch.context() as patch:
        patch.setattr(
            db.backend,
            "execute",
            lambda *args, **kwargs: pytest.fail("invalid update reached SQL"),
        )
        with pytest.raises(ValueError, match=error):
            db.update_briefing_occurrence(int(occurrence.id), **fields)

    assert db.get_briefing_occurrence(int(occurrence.id)) == occurrence


def test_update_occurrence_accepts_zero_counts(tmp_path):
    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label="zero-counts")
    occurrence = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=f"user:1:job:{job_id}:run:{run_id}:zero-counts",
        contract_json='{"version":1}',
    )
    db.update_briefing_occurrence(
        int(occurrence.id),
        selected_count=3,
        omitted_count=2,
    )

    updated = db.update_briefing_occurrence(
        int(occurrence.id),
        selected_count=0,
        omitted_count=0,
    )

    assert updated.selected_count == 0
    assert updated.omitted_count == 0


class _CapturingPostgresBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(self) -> None:
        self.ddl = ""
        self.executed: list[str] = []

    def create_tables(self, ddl: str) -> None:
        self.ddl = ddl

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append(query)

    def get_table_info(self, table_name: str) -> list[dict[str, Any]]:
        existing_columns = {
            "scrape_jobs": {"wf_schedule_id", "job_filters_json", "watchlist_id"},
            "sources": {"defer_until", "consec_not_modified", "consec_errors"},
            "scrape_run_items": {"source_id"},
            "scraped_items": {"content", "queued_for_briefing"},
        }
        return [{"name": name} for name in existing_columns.get(table_name, set())]


def test_postgres_schema_includes_briefing_occurrence_contract():
    backend = _CapturingPostgresBackend()

    WatchlistsDatabase(user_id=1, backend=backend)  # type: ignore[arg-type]

    occurrence_table_marker = "CREATE TABLE IF NOT EXISTS watchlist_briefing_occurrences"
    assert occurrence_table_marker in backend.ddl
    occurrence_ddl = backend.ddl.partition(occurrence_table_marker)[2].partition(";")[0]
    assert "id BIGSERIAL PRIMARY KEY" in occurrence_ddl
    assert "job_id BIGINT NOT NULL" in occurrence_ddl
    assert "run_id BIGINT NOT NULL" in occurrence_ddl
    assert "output_id BIGINT" in occurrence_ddl
    assert "UNIQUE (user_id, occurrence_key)" in occurrence_ddl
    assert "CREATE INDEX IF NOT EXISTS idx_briefing_occurrences_user_job" in backend.ddl
    assert "CREATE INDEX IF NOT EXISTS idx_briefing_occurrences_run" in backend.ddl
    attempt_marker = "CREATE TABLE IF NOT EXISTS watchlist_briefing_attempts"
    assert attempt_marker in backend.ddl
    attempt_ddl = backend.ddl.partition(attempt_marker)[2].partition(";")[0]
    assert "id BIGSERIAL PRIMARY KEY" in attempt_ddl
    assert "occurrence_id BIGINT NOT NULL" in attempt_ddl
    assert "UNIQUE (user_id, occurrence_id, artifact_version, adapter, attempt)" in attempt_ddl
    assert "CREATE INDEX IF NOT EXISTS idx_briefing_attempts_latest" in backend.ddl

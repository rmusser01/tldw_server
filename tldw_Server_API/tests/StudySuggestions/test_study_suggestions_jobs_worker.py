from __future__ import annotations

import asyncio
from contextlib import suppress
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK


pytestmark = pytest.mark.asyncio


def _load_modules():
    try:
        actions_mod = import_module("tldw_Server_API.app.core.StudySuggestions.actions")
        snapshot_service_mod = import_module("tldw_Server_API.app.core.StudySuggestions.snapshot_service")
        jobs_mod = import_module("tldw_Server_API.app.core.StudySuggestions.jobs")
        worker_mod = import_module("tldw_Server_API.app.services.study_suggestions_jobs_worker")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Study suggestions jobs worker modules are missing: {exc}")
    except ImportError as exc:
        pytest.fail(f"Study suggestions jobs worker imports are not yet usable: {exc}")
    return actions_mod, snapshot_service_mod, jobs_mod, worker_mod


@pytest.fixture
def db(tmp_path):
    chacha = CharactersRAGDB(str(tmp_path / "study-suggestions-worker.db"), client_id="study-suggestions-worker-tests")
    try:
        yield chacha
    finally:
        chacha.close_connection()


@pytest.fixture
def jobs_db_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "study-suggestions-worker-jobs.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(path))
    return path


def _create_snapshot(
    db: CharactersRAGDB,
    *,
    anchor_id: int = 101,
) -> int:
    return db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=anchor_id,
        suggestion_type="study_suggestions",
        payload_json={"topics": [{"display_label": "Original topic"}]},
        user_selection_json={"selected_topic_ids": ["topic-1"]},
    )


def _fake_refresh_job(
    jobs_mod: Any,
    *,
    job_id: int = 1,
    anchor_id: int = 101,
    snapshot_id: int | None = None,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "owner_user_id": "1",
        "job_type": jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
        "payload": jobs_mod.build_study_suggestions_job_payload(
            job_type=jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
            anchor_type="quiz_attempt",
            anchor_id=anchor_id,
            snapshot_id=snapshot_id,
        ),
    }


async def test_handle_refresh_job_creates_new_snapshot_with_refresh_lineage(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    _actions_mod, snapshot_service_mod, jobs_mod, worker_mod = _load_modules()
    original_snapshot_id = _create_snapshot(db)

    async def fake_get_databases_for_user(user_id: str):
        assert user_id == "1"  # nosec B101
        return db, SimpleNamespace()

    monkeypatch.setattr(worker_mod, "_get_databases_for_user", fake_get_databases_for_user)

    def fake_refresh_snapshot(*, note_db, anchor_type, anchor_id, refreshed_from_snapshot_id, principal):
        assert note_db is db  # nosec B101
        assert anchor_type == "quiz_attempt"  # nosec B101
        assert anchor_id == 101  # nosec B101
        return db.create_suggestion_snapshot(
            service="quiz",
            activity_type="quiz_attempt",
            anchor_type=anchor_type,
            anchor_id=anchor_id,
            suggestion_type="study_suggestions",
            payload_json={"topics": [{"display_label": "Refreshed topic"}]},
            refreshed_from_snapshot_id=refreshed_from_snapshot_id,
        )

    monkeypatch.setattr(snapshot_service_mod, "refresh_snapshot_for_anchor", fake_refresh_snapshot)

    result = await worker_mod.handle_study_suggestions_job(
        _fake_refresh_job(jobs_mod, snapshot_id=original_snapshot_id)
    )

    refreshed_snapshot = db.get_suggestion_snapshot(int(result["snapshot_id"]))
    original_snapshot = db.get_suggestion_snapshot(original_snapshot_id)

    assert int(result["snapshot_id"]) != original_snapshot_id  # nosec B101
    assert refreshed_snapshot["refreshed_from_snapshot_id"] == original_snapshot_id  # nosec B101
    assert refreshed_snapshot["user_selection_json"] == {"selected_topic_ids": ["topic-1"]}  # nosec B101
    assert original_snapshot["payload_json"]["topics"][0]["display_label"] == "Original topic"  # nosec B101


async def test_failed_jobs_surface_failed_without_mutating_prior_snapshots(
    db: CharactersRAGDB,
    jobs_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _actions_mod, snapshot_service_mod, jobs_mod, worker_mod = _load_modules()
    original_snapshot_id = _create_snapshot(db)

    async def fake_get_databases_for_user(user_id: str):
        return db, SimpleNamespace()

    monkeypatch.setattr(worker_mod, "_get_databases_for_user", fake_get_databases_for_user)

    def fail_refresh_snapshot(*, note_db, anchor_type, anchor_id, refreshed_from_snapshot_id, principal):
        raise RuntimeError("refresh exploded")

    monkeypatch.setattr(snapshot_service_mod, "refresh_snapshot_for_anchor", fail_refresh_snapshot)

    jm = JobManager(db_path=jobs_db_path)
    created = jm.create_job(
        domain=jobs_mod.STUDY_SUGGESTIONS_DOMAIN,
        queue=jobs_mod.study_suggestions_jobs_queue(),
        job_type=jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
        payload=jobs_mod.build_study_suggestions_job_payload(
            job_type=jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
            anchor_type="quiz_attempt",
            anchor_id=101,
            snapshot_id=original_snapshot_id,
        ),
        owner_user_id="1",
        priority=5,
        max_retries=1,
    )
    acquired = jm.acquire_next_job(
        domain=jobs_mod.STUDY_SUGGESTIONS_DOMAIN,
        queue=jobs_mod.study_suggestions_jobs_queue(),
        lease_seconds=30,
        worker_id="study-suggestions-worker-test",
    )
    assert acquired is not None  # nosec B101

    with pytest.raises(RuntimeError, match="refresh exploded"):
        await worker_mod.handle_study_suggestions_job(dict(acquired))

    jm.fail_job(
        int(acquired["id"]),
        error="refresh exploded",
        retryable=False,
        worker_id=str(acquired["worker_id"]),
        lease_id=str(acquired["lease_id"]),
    )

    status = snapshot_service_mod.get_anchor_status(
        note_db=db,
        job_manager=jm,
        anchor_type="quiz_attempt",
        anchor_id=101,
    )
    snapshots = db.list_suggestion_snapshots_for_anchor("quiz_attempt", 101)

    assert status["status"] == "failed"  # nosec B101
    assert len(snapshots) == 1  # nosec B101
    assert snapshots[0]["id"] == original_snapshot_id  # nosec B101
    assert snapshots[0]["payload_json"]["topics"][0]["display_label"] == "Original topic"  # nosec B101


async def test_anchor_status_pages_past_first_hundred_failed_jobs_for_matching_anchor(
    db: CharactersRAGDB,
    jobs_db_path: Path,
):
    _actions_mod, snapshot_service_mod, jobs_mod, _worker_mod = _load_modules()
    jm = JobManager(db_path=jobs_db_path)

    target = jm.create_job(
        domain=jobs_mod.STUDY_SUGGESTIONS_DOMAIN,
        queue=jobs_mod.study_suggestions_jobs_queue(),
        job_type=jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
        payload=jobs_mod.build_study_suggestions_job_payload(
            job_type=jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
            anchor_type="quiz_attempt",
            anchor_id=101,
            snapshot_id=11,
        ),
        owner_user_id="1",
        priority=5,
        max_retries=1,
    )
    acquired = jm.acquire_next_job(
        domain=jobs_mod.STUDY_SUGGESTIONS_DOMAIN,
        queue=jobs_mod.study_suggestions_jobs_queue(),
        lease_seconds=30,
        worker_id="target-job",
    )
    assert acquired is not None  # nosec B101
    jm.fail_job(
        int(acquired["id"]),
        error="target failed",
        retryable=False,
        worker_id=str(acquired["worker_id"]),
        lease_id=str(acquired["lease_id"]),
    )

    for anchor_id in range(200, 321):
        created = jm.create_job(
            domain=jobs_mod.STUDY_SUGGESTIONS_DOMAIN,
            queue=jobs_mod.study_suggestions_jobs_queue(),
            job_type=jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
            payload=jobs_mod.build_study_suggestions_job_payload(
                job_type=jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
                anchor_type="quiz_attempt",
                anchor_id=anchor_id,
                snapshot_id=anchor_id,
            ),
            owner_user_id="1",
            priority=5,
            max_retries=1,
        )
        acquired = jm.acquire_next_job(
            domain=jobs_mod.STUDY_SUGGESTIONS_DOMAIN,
            queue=jobs_mod.study_suggestions_jobs_queue(),
            lease_seconds=30,
            worker_id=f"bulk-{anchor_id}",
        )
        assert acquired is not None  # nosec B101
        assert int(acquired["id"]) == int(created["id"])  # nosec B101
        jm.fail_job(
            int(acquired["id"]),
            error="bulk failed",
            retryable=False,
            worker_id=str(acquired["worker_id"]),
            lease_id=str(acquired["lease_id"]),
        )

    status = snapshot_service_mod.get_anchor_status(
        note_db=db,
        job_manager=jm,
        anchor_type="quiz_attempt",
        anchor_id=101,
    )

    assert status["status"] == "failed"  # nosec B101
    assert int(status["job_id"]) == int(target["id"])  # nosec B101


async def test_worker_sdk_cancellation_marks_study_suggestions_job_cancelled_without_running_handler(
    jobs_db_path: Path,
):
    _actions_mod, _snapshot_service_mod, jobs_mod, worker_mod = _load_modules()
    jm = JobManager(db_path=jobs_db_path)
    job = jm.create_job(
        domain=jobs_mod.STUDY_SUGGESTIONS_DOMAIN,
        queue=jobs_mod.study_suggestions_jobs_queue(),
        job_type=jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
        payload=jobs_mod.build_study_suggestions_job_payload(
            job_type=jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
            anchor_type="quiz_attempt",
            anchor_id=101,
            snapshot_id=11,
        ),
        owner_user_id="1",
        priority=5,
        max_retries=1,
    )

    cfg = WorkerConfig(
        domain=jobs_mod.STUDY_SUGGESTIONS_DOMAIN,
        queue=jobs_mod.study_suggestions_jobs_queue(),
        worker_id="study-suggestions-worker-test",
        lease_seconds=5,
        renew_threshold_seconds=1,
        renew_jitter_seconds=0,
    )
    sdk = WorkerSDK(jm, cfg)

    async def handler(job_row: dict[str, Any]):
        pytest.fail("Study suggestions handler should not run when cancel_check returns true")

    cancel_check_started = asyncio.Event()

    async def cancel_check(job_row: dict[str, Any]) -> bool:
        cancel_check_started.set()
        jm.cancel_job(int(job_row["id"]), reason="requested")
        return await worker_mod._should_cancel(job_row, job_manager=jm)

    task = asyncio.create_task(sdk.run(handler=handler, cancel_check=cancel_check))
    await asyncio.wait_for(cancel_check_started.wait(), timeout=1)
    sdk.stop()
    task.cancel()
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    stored = jm.get_job(int(job["id"]))

    assert stored is not None  # nosec B101
    assert stored["status"] == "cancelled"  # nosec B101


async def test_selection_fingerprint_includes_snapshot_target_topics_action_generator_and_normalization_version():
    actions_mod, _snapshot_service_mod, _jobs_mod, _worker_mod = _load_modules()

    fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=17,
        target_service="quiz",
        target_type="quiz",
        selected_topics=[" Renal Basics ", "acid base", "renal basics"],
        action_kind="follow_up_quiz",
        generator_version="v2",
        normalization_version="norm-v2",
    )

    assert "snapshot_id=17" in fingerprint  # nosec B101
    assert "target_service=quiz" in fingerprint  # nosec B101
    assert "target_type=quiz" in fingerprint  # nosec B101
    assert "topics=acid base,renal basics" in fingerprint  # nosec B101
    assert "action_kind=follow_up_quiz" in fingerprint  # nosec B101
    assert "generator_version=v2" in fingerprint  # nosec B101
    assert "normalization_version=norm-v2" in fingerprint  # nosec B101


async def test_resolve_selected_topic_labels_defaults_to_snapshot_selected_topics():
    actions_mod, _snapshot_service_mod, _jobs_mod, _worker_mod = _load_modules()

    labels = actions_mod.resolve_selected_topic_labels(
        {
            "payload_json": {
                "topics": [
                    {"id": "topic-1", "display_label": " Renal Basics ", "selected": True},
                    {"id": "topic-2", "display_label": "Acid Base", "selected": False},
                ]
            }
        },
        selected_topic_ids=[],
    )

    assert labels == ["renal basics"]  # nosec B101


async def test_resolve_selected_topic_labels_uses_id_anchored_edits_and_manual_labels():
    actions_mod, _snapshot_service_mod, _jobs_mod, _worker_mod = _load_modules()

    labels = actions_mod.resolve_selected_topic_labels(
        {
            "payload_json": {
                "topics": [
                    {"id": "topic-1", "display_label": "Renal Basics", "selected": True},
                    {"id": "topic-2", "display_label": "Acid Base", "selected": True},
                ]
            }
        },
        selected_topic_ids=["topic-2"],
        selected_topic_edits=[{"id": "topic-2", "label": " Acid Base Remix "}],
        manual_topic_labels=[" Electrolyte Ladder "],
        has_explicit_selection=True,
    )

    assert labels == ["acid base remix", "electrolyte ladder"]  # nosec B101


async def test_resolve_selected_topic_labels_respects_explicit_empty_selection():
    actions_mod, _snapshot_service_mod, _jobs_mod, _worker_mod = _load_modules()

    labels = actions_mod.resolve_selected_topic_labels(
        {
            "payload_json": {
                "topics": [
                    {"id": "topic-1", "display_label": "Renal Basics", "selected": True},
                    {"id": "topic-2", "display_label": "Acid Base", "selected": True},
                ]
            }
        },
        selected_topic_ids=[],
        selected_topic_edits=[],
        manual_topic_labels=[],
        has_explicit_selection=True,
    )

    assert labels == []  # nosec B101

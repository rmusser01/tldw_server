from __future__ import annotations

import asyncio
import json

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_job_handlers, claims_jobs
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import OperationOutcome
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK

pytestmark = pytest.mark.integration


def seed_reviews(db):
    media_id, _, _ = db.add_media_with_keywords(
        title="Review source", media_type="text", content="Original claim.", keywords=None
    )
    db.upsert_claims(
        [
            {
                "media_id": media_id,
                "chunk_index": 0,
                "claim_text": "Original claim.",
                "confidence": 0.9,
                "extractor": "heuristic",
                "extractor_version": "v1",
                "chunk_hash": "hash",
            }
        ]
    )
    claim = db.execute_query("SELECT id FROM Claims WHERE media_id = ?", (media_id,)).fetchone()
    db.update_claim_review(
        int(claim["id"]),
        review_status="approved",
        reviewer_id=42,
        corrected_text="Corrected claim.",
        review_reason_code="typo",
    )
    db.execute_query("UPDATE claims_review_log SET created_at = ?", ("2026-09-07 01:00:00",), commit=True)


def enqueue(manager, *, start="2026-09-07"):
    return claims_jobs.enqueue_claims_review_metrics(
        owner_user_id="42",
        scheduled_for="2026-09-07T00:00:00Z",
        start_date=start,
        end_date="2026-09-07",
        interval_seconds=86400,
        job_manager=manager,
        settings_obj={"CLAIMS_JOBS_QUEUE": "default"},
    )


async def run_worker(manager, dispatched):
    sdk = WorkerSDK(
        manager,
        WorkerConfig(
            domain="claims",
            queue="default",
            worker_id="metrics-e2e",
            lease_seconds=5,
            renew_threshold_seconds=1,
            renew_jitter_seconds=0,
        ),
    )

    async def handler(job):
        dispatched.append(job["id"])
        return await claims_job_handlers.process_claims_job(job)

    async def completed(job, result):
        sdk.stop()

    await asyncio.wait_for(sdk.run(handler=handler, on_completed=completed), 30)


@pytest.mark.asyncio
async def test_real_admission_worker_delayed_window_replay_and_queued_cancellation(monkeypatch, tmp_path):
    manager = JobManager(tmp_path / "jobs.sqlite")
    owner_dir = tmp_path / "42"
    db = MediaDatabase(db_path=owner_dir / "Media_DB_v2.db", client_id="42")
    monkeypatch.setattr(claims_job_handlers.media_db_runtime_defaults, "postgres_content_mode", False)
    monkeypatch.setattr(claims_job_handlers.DatabasePaths, "resolve_user_base_directory", lambda owner: owner_dir)
    dispatched = []
    try:
        seed_reviews(db)
        cancelled = enqueue(manager, start="2026-09-06")
        assert cancelled.outcome is OperationOutcome.APPLIED
        assert manager.cancel_job(int(cancelled.row["id"]))
        accepted = enqueue(manager)
        replay = enqueue(manager)
        assert replay.outcome is OperationOutcome.NO_TRANSITION
        assert replay.row["id"] == accepted.row["id"]
        await run_worker(manager, dispatched)
        assert dispatched == [accepted.row["id"]]
        assert manager.get_job(int(cancelled.row["id"]))["status"] == "cancelled"
        completed = manager.get_job(int(accepted.row["id"]))
        assert completed["status"] == "completed"
        rows = db.list_claims_review_extractor_metrics_daily(user_id="42")
        assert len(rows) == 1 and rows[0]["report_date"] == "2026-09-07"
        assert rows[0]["approved_count"] == rows[0]["edited_count"] == 1
        assert json.loads(rows[0]["reason_code_counts_json"]) == {"typo": 1}
        result = await claims_job_handlers.process_claims_job(accepted.row)
        assert result["groups_written"] == 1
        assert len(db.list_claims_review_extractor_metrics_daily(user_id="42")) == 1
    finally:
        db.close_connection()

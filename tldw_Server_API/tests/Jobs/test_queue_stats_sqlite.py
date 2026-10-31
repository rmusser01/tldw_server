"""core/Jobs/queue_stats.py over a real SQLite jobs table, plus its conversion contract."""

from __future__ import annotations

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.jobs_queue_stats_repository import JobsQueueStatsRepository
from tldw_Server_API.app.core.Jobs import queue_stats
from tldw_Server_API.app.core.Jobs.manager import JobManager

_SLICE = {"domain": "prompt_studio", "queue": "default", "owner_user_id": "1"}


@pytest.fixture
def jm(tmp_path) -> JobManager:
    return JobManager(tmp_path / "jobs.db")


def _seed(jm: JobManager) -> None:
    for job_type in ("evaluation", "evaluation", "optimization"):
        jm.create_job(domain="prompt_studio", queue="default", job_type=job_type, payload={}, owner_user_id="1")
    assert jm.acquire_next_job(domain="prompt_studio", queue="default", lease_seconds=30, worker_id="w") is not None


def test_lease_warning_window_is_bound_and_applied(jm):
    _seed(jm)

    wide = queue_stats.get_lease_stats(jm, warn_seconds=60, **_SLICE)
    narrow = queue_stats.get_lease_stats(jm, warn_seconds=10, **_SLICE)

    assert wide == {"active": 1, "expiring_soon": 1, "stale_processing": 0}
    assert narrow == {"active": 1, "expiring_soon": 0, "stale_processing": 0}


def test_counts_and_empty_aggregates(jm):
    _seed(jm)

    by_status = queue_stats.get_by_status(jm, **_SLICE)
    totals, queued, processing = queue_stats.get_by_type_and_status(jm, **_SLICE)

    assert by_status == {"queued": 2, "processing": 1}
    assert sum(totals.values()) == 3 and sum(queued.values()) == 2 and sum(processing.values()) == 1
    # No finished jobs: the aggregates are NULL, which is a legitimate 0.
    assert queue_stats.get_success_rate(jm, **_SLICE) == 0.0
    assert queue_stats.get_avg_processing_time_seconds(jm, **_SLICE) == 0.0
    assert queue_stats.get_lease_stats(jm, warn_seconds=60, domain="none", queue="none", owner_user_id=None) == {
        "active": 0,
        "expiring_soon": 0,
        "stale_processing": 0,
    }


def test_corrupt_value_is_logged_and_raised_not_reported_as_zero(jm, monkeypatch):
    monkeypatch.setattr(
        JobsQueueStatsRepository,
        "count_by_status",
        lambda self, **_kw: [{"status": "queued", "c": "secret-garbage"}],
    )
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        with pytest.raises(ValueError, match="by_status"):
            queue_stats.get_by_status(jm, **_SLICE)
    finally:
        logger.remove(sink)

    assert any("by_status" in m for m in messages)
    assert not any("secret-garbage" in m for m in messages)

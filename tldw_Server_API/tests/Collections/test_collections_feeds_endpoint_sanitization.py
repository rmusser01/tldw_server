from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from tldw_Server_API.app.api.v1.endpoints import collections_feeds

pytestmark = pytest.mark.unit


class _FakeFeedDb:
    def __init__(self) -> None:
        self.schedule_ids: list[tuple[int, str]] = []

    def set_job_schedule_id(self, job_id: int, schedule_id: str) -> None:
        self.schedule_ids.append((job_id, schedule_id))


class _FailingScheduler:
    def create(self, **kwargs):
        raise RuntimeError("scheduler backend exploded at /private/scheduler.db")


class _FailingUpdateScheduler(_FailingScheduler):
    def update(self, schedule_id: str, payload: dict):
        assert schedule_id == "schedule-private"
        raise RuntimeError("scheduler update exploded at /private/scheduler-update.db")


def _job_row() -> SimpleNamespace:
    return SimpleNamespace(
        id=9,
        name="Private Feed",
        schedule_expr="0 * * * *",
        schedule_timezone="UTC",
        active=True,
    )


def _scheduled_job_row() -> SimpleNamespace:
    job = _job_row()
    job.wf_schedule_id = "schedule-private"
    return job


def test_register_schedule_sanitizes_scheduler_failure_log(monkeypatch):
    from tldw_Server_API.app.core.DB_Management import Workflows_Scheduler_DB as wfdb_module
    from tldw_Server_API.app.services import workflows_scheduler

    class _FallbackSchedulerDb:
        def __init__(self, *, user_id: int) -> None:
            assert user_id == 42

        def create_schedule(self, **kwargs) -> None:
            return None

    fake_db = _FakeFeedDb()
    fake_logger = MagicMock()
    monkeypatch.setattr(workflows_scheduler, "get_workflows_scheduler", lambda: _FailingScheduler())
    monkeypatch.setattr(wfdb_module, "WorkflowsSchedulerDB", _FallbackSchedulerDb)
    monkeypatch.setattr(collections_feeds, "logger", fake_logger)

    collections_feeds._register_schedule(
        fake_db,
        _job_row(),
        current_user=SimpleNamespace(id=42),
    )

    fake_logger.debug.assert_called_once_with("Collections feeds schedule registration failed")
    assert fake_db.schedule_ids


def test_register_schedule_sanitizes_db_fallback_failure_log(monkeypatch):
    from tldw_Server_API.app.core.DB_Management import Workflows_Scheduler_DB as wfdb_module
    from tldw_Server_API.app.services import workflows_scheduler

    class _FailingFallbackSchedulerDb:
        def __init__(self, *, user_id: int) -> None:
            assert user_id == 42

        def create_schedule(self, **kwargs) -> None:
            raise RuntimeError("scheduler DB fallback exploded at /private/scheduler-fallback.db")

    fake_logger = MagicMock()
    monkeypatch.setattr(workflows_scheduler, "get_workflows_scheduler", lambda: _FailingScheduler())
    monkeypatch.setattr(wfdb_module, "WorkflowsSchedulerDB", _FailingFallbackSchedulerDb)
    monkeypatch.setattr(collections_feeds, "logger", fake_logger)

    collections_feeds._register_schedule(
        _FakeFeedDb(),
        _job_row(),
        current_user=SimpleNamespace(id=42),
    )

    assert fake_logger.debug.call_args_list == [
        call("Collections feeds schedule registration failed"),
        call("Collections feeds schedule DB fallback failed"),
    ]


def test_sync_job_schedule_sanitizes_scheduler_update_failure_log(monkeypatch):
    from tldw_Server_API.app.services import workflows_scheduler

    class _FakeFeedDbWithJob(_FakeFeedDb):
        def get_job(self, job_id: int) -> SimpleNamespace:
            assert job_id == 9
            return SimpleNamespace(id=job_id)

    fake_logger = MagicMock()
    monkeypatch.setattr(workflows_scheduler, "get_workflows_scheduler", lambda: _FailingUpdateScheduler())
    monkeypatch.setattr(collections_feeds, "logger", fake_logger)

    result = collections_feeds._sync_job_schedule(
        _FakeFeedDbWithJob(),
        _scheduled_job_row(),
        current_user=SimpleNamespace(id=42),
    )

    assert result.id == 9
    fake_logger.debug.assert_called_once_with("Collections feeds schedule update failed")

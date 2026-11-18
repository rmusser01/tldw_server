from datetime import datetime
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.Workflows_Scheduler_DB import WorkflowsSchedulerDB
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


def _close_scheduler_backend(db: WorkflowsSchedulerDB) -> None:
    try:
        pool = db.backend.get_pool()
        pool.close_all()
    except Exception:
        _ = None


def test_scheduler_uses_dedicated_default_sqlite_path_when_global_database_url_is_set(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./Databases/users.db")
    monkeypatch.delenv("WORKFLOWS_SCHEDULER_DATABASE_URL", raising=False)
    monkeypatch.delenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", raising=False)

    db = WorkflowsSchedulerDB(user_id=1)
    try:
        actual = Path(db.backend.config.sqlite_path).resolve()  # type: ignore[arg-type]
        expected = DatabasePaths.get_workflows_scheduler_db_path(1).resolve()
        assert actual == expected
    finally:
        _close_scheduler_backend(db)



@pytest.mark.parametrize("backend_type", [BackendType.SQLITE, BackendType.POSTGRESQL])
def test_scheduler_creates_and_updates_boolean_flags(
    backend_type: BackendType,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    if backend_type is BackendType.POSTGRESQL:
        config = request.getfixturevalue("pg_database_config")
    else:
        config = DatabaseConfig(backend_type=backend_type, sqlite_path=str(tmp_path / "scheduler.db"))
    db = WorkflowsSchedulerDB(backend=DatabaseBackendFactory.create_backend(config))
    try:
        db.create_schedule(
            id="watchlist-monitor",
            tenant_id="default",
            user_id="1",
            workflow_id=None,
            name="Watchlist monitor",
            cron="*/15 * * * *",
            timezone="UTC",
            inputs={"watchlist_job_id": 1},
            enabled=True,
            require_online=False,
            coalesce=True,
        )
        created = db.get_schedule("watchlist-monitor")
        assert created is not None
        assert (created.enabled, created.require_online, created.coalesce) == (True, False, True)
        assert db.update_schedule(
            "watchlist-monitor", {"enabled": False, "require_online": True, "coalesce": False},
        )
        paused = db.get_schedule("watchlist-monitor")
        assert paused is not None
        assert (paused.enabled, paused.require_online, paused.coalesce) == (False, True, False)

        from tldw_Server_API.app.api.v1.endpoints.acp_schedules import ACPScheduleResponse, _schedule_to_response

        last_run = "2026-11-17T19:30:00+00:00"
        next_run = "2026-11-17T19:45:00+00:00"
        db.set_history("watchlist-monitor", last_run_at=last_run, next_run_at=next_run)
        rows = (
            db.get_schedule("watchlist-monitor"),
            *db.list_schedules(tenant_id="default", user_id="1"),
            *db.list_all_schedules(),
        )
        assert len(rows) == 3
        for row in rows:
            assert row is not None
            response = ACPScheduleResponse.model_validate(_schedule_to_response(row))
            assert datetime.fromisoformat(response.last_run_at) == datetime.fromisoformat(last_run)
            assert datetime.fromisoformat(response.next_run_at) == datetime.fromisoformat(next_run)
            assert isinstance(response.created_at, str)
            assert isinstance(row.updated_at, str)
    finally:
        _close_scheduler_backend(db)

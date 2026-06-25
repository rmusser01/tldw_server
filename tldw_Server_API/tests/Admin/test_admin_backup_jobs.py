from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@dataclass
class _BackupResult:
    filename: str
    dataset: str
    user_id: int | None
    size_bytes: int
    created_at: str
    path: str


def test_backup_schedule_jobs_storage_import_is_compatibility_shim() -> None:
    from tldw_Server_API.app.core.Admin_Backups import backup_schedule_jobs as canonical
    from tldw_Server_API.app.core.Storage import backup_schedule_jobs as storage_shim

    assert storage_shim.BACKUP_SCHEDULE_DOMAIN == canonical.BACKUP_SCHEDULE_DOMAIN
    assert storage_shim.BACKUP_SCHEDULE_JOB_TYPE == canonical.BACKUP_SCHEDULE_JOB_TYPE
    assert storage_shim.build_backup_schedule_job_payload is canonical.build_backup_schedule_job_payload


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "missing_schedule_id"),
        ({"schedule_id": "1"}, "missing_run_id"),
        ({"schedule_id": "1", "run_id": "2"}, "missing_dataset"),
        (
            {
                "schedule_id": "1",
                "run_id": "2",
                "dataset": "media",
                "scheduled_for": "not-a-date",
            },
            "invalid_scheduled_for",
        ),
        (
            {
                "schedule_id": "1",
                "run_id": "2",
                "dataset": "media",
                "scheduled_for": "2026-06-24T00:00:00Z",
                "target_user_id": "not-an-int",
            },
            "invalid_target_user_id",
        ),
        (
            {
                "schedule_id": "1",
                "run_id": "2",
                "dataset": "media",
                "scheduled_for": "2026-06-24T00:00:00Z",
                "retention_count": "not-an-int",
            },
            "invalid_retention_count",
        ),
    ],
)
def test_parse_backup_schedule_job_payload_uses_validation_error(payload, message) -> None:
    from tldw_Server_API.app.core.Admin_Backups.backup_schedule_jobs import (
        parse_backup_schedule_job_payload,
    )
    from tldw_Server_API.app.core.exceptions import ValidationError

    with pytest.raises(ValidationError, match=message):
        parse_backup_schedule_job_payload(payload)


@pytest.fixture()
async def backup_jobs_repo(tmp_path, monkeypatch: pytest.MonkeyPatch):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.backup_schedules_repo import (
        AuthnzBackupSchedulesRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users_backup_jobs.db"
    jobs_db_path = tmp_path / "jobs_backup_jobs.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_db_path))

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = AuthnzBackupSchedulesRepo(pool)
    await repo.ensure_schema()
    return repo


async def _create_queued_run(repo, *, dataset: str = "media", target_user_id: int | None = 7):
    from tldw_Server_API.app.core.Storage.backup_schedule_jobs import (
        build_backup_schedule_job_payload,
        build_backup_schedule_run_slot_key,
        normalize_backup_schedule_slot,
    )

    due = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    schedule = await repo.create_schedule(
        dataset=dataset,
        target_user_id=target_user_id,
        frequency="daily",
        time_of_day="02:00",
        timezone="UTC",
        anchor_day_of_week=None,
        anchor_day_of_month=None,
        retention_count=10,
        created_by_user_id=1,
        updated_by_user_id=1,
        next_run_at=(due + timedelta(days=1)).isoformat(),
    )
    scheduled_for = normalize_backup_schedule_slot(due)
    run = await repo.claim_run_slot(
        schedule_id=str(schedule["id"]),
        scheduled_for=scheduled_for,
        run_slot_key=build_backup_schedule_run_slot_key(
            schedule_id=str(schedule["id"]),
            scheduled_for=scheduled_for,
        ),
        enqueued_at=scheduled_for,
    )
    assert run is not None
    await repo.mark_run_queued(
        run_id=str(run["id"]),
        job_id="job-1",
        next_run_at=(due + timedelta(days=1)).isoformat(),
        last_run_at=scheduled_for,
    )
    job = {
        "id": "job-1",
        "owner_user_id": target_user_id,
        "payload": build_backup_schedule_job_payload(
            schedule_id=str(schedule["id"]),
            run_id=str(run["id"]),
            scheduled_for=scheduled_for,
            dataset=dataset,
            target_user_id=target_user_id,
            retention_count=5,
        ),
    }
    return schedule, run, job


@pytest.mark.asyncio
async def test_handle_backup_schedule_job_uses_current_schedule_row(backup_jobs_repo) -> None:
    from tldw_Server_API.app.services.admin_backup_jobs_worker import handle_backup_schedule_job

    schedule, run, job = await _create_queued_run(backup_jobs_repo)
    updated_schedule = await backup_jobs_repo.update_schedule(
        str(schedule["id"]),
        retention_count=25,
        updated_by_user_id=1,
    )
    assert updated_schedule is not None

    captured: dict[str, object] = {}

    def _fake_create_backup_snapshot(*, dataset, user_id, backup_type, max_backups):
        captured["dataset"] = dataset
        captured["user_id"] = user_id
        captured["backup_type"] = backup_type
        captured["max_backups"] = max_backups
        return _BackupResult(
            filename="backup-1.db",
            dataset=dataset,
            user_id=user_id,
            size_bytes=123,
            created_at=datetime.now(timezone.utc).isoformat(),
            path="/tmp/backup-1.db",
        )

    result = await handle_backup_schedule_job(
        job,
        repo=backup_jobs_repo,
        create_backup_snapshot_fn=_fake_create_backup_snapshot,
    )

    assert result["status"] == "succeeded"
    assert result["run_id"] == str(run["id"])
    assert captured["dataset"] == "media"
    assert captured["user_id"] == 7
    assert captured["backup_type"] == "full"
    assert captured["max_backups"] == 25


@pytest.mark.asyncio
async def test_handle_backup_schedule_job_marks_run_succeeded(backup_jobs_repo) -> None:
    from tldw_Server_API.app.services.admin_backup_jobs_worker import handle_backup_schedule_job

    schedule, run, job = await _create_queued_run(backup_jobs_repo)

    def _fake_create_backup_snapshot(*, dataset, user_id, backup_type, max_backups):
        return _BackupResult(
            filename="backup-2.db",
            dataset=dataset,
            user_id=user_id,
            size_bytes=456,
            created_at=datetime.now(timezone.utc).isoformat(),
            path="/tmp/backup-2.db",
        )

    await handle_backup_schedule_job(
        job,
        repo=backup_jobs_repo,
        create_backup_snapshot_fn=_fake_create_backup_snapshot,
    )

    updated_run = await backup_jobs_repo.get_run(str(run["id"]))
    assert updated_run is not None
    assert updated_run["status"] == "succeeded"

    updated_schedule = await backup_jobs_repo.get_schedule(str(schedule["id"]))
    assert updated_schedule is not None
    assert updated_schedule["last_status"] == "succeeded"
    assert updated_schedule["last_error"] is None


@pytest.mark.asyncio
async def test_handle_backup_schedule_job_marks_failure_and_preserves_error(backup_jobs_repo) -> None:
    from tldw_Server_API.app.services.admin_backup_jobs_worker import handle_backup_schedule_job

    schedule, run, job = await _create_queued_run(backup_jobs_repo)

    def _failing_create_backup_snapshot(*, dataset, user_id, backup_type, max_backups):
        raise RuntimeError("backup exploded")

    with pytest.raises(RuntimeError, match="backup exploded"):
        await handle_backup_schedule_job(
            job,
            repo=backup_jobs_repo,
            create_backup_snapshot_fn=_failing_create_backup_snapshot,
        )

    updated_run = await backup_jobs_repo.get_run(str(run["id"]))
    assert updated_run is not None
    assert updated_run["status"] == "failed"
    assert updated_run["error"] == "backup exploded"

    updated_schedule = await backup_jobs_repo.get_schedule(str(schedule["id"]))
    assert updated_schedule is not None
    assert updated_schedule["last_status"] == "failed"
    assert updated_schedule["last_error"] == "backup exploded"

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_backup_schedules_repo_sqlite_roundtrip_and_uniqueness(tmp_path, monkeypatch, fresh_services) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.backup_schedules_repo import (
        AuthnzBackupSchedulesRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = AuthnzBackupSchedulesRepo(pool)
    await repo.ensure_schema()

    created = await repo.create_schedule(
        dataset="media",
        target_user_id=7,
        frequency="daily",
        time_of_day="02:00",
        timezone="UTC",
        anchor_day_of_week=None,
        anchor_day_of_month=None,
        retention_count=14,
        created_by_user_id=1,
        updated_by_user_id=1,
        next_run_at="2026-03-11T02:00:00+00:00",
    )

    assert created["dataset"] == "media"
    assert created["target_user_id"] == 7
    assert created["retention_count"] == 14
    assert created["deleted_at"] is None

    fetched = await repo.get_schedule(str(created["id"]))
    assert fetched is not None
    assert fetched["id"] == created["id"]

    listed, total = await repo.list_schedules(limit=50, offset=0)
    assert total == 1
    assert listed[0]["id"] == created["id"]

    with pytest.raises(Exception):
        await repo.create_schedule(
            dataset="media",
            target_user_id=7,
            frequency="weekly",
            time_of_day="03:00",
            timezone="UTC",
            anchor_day_of_week=2,
            anchor_day_of_month=None,
            retention_count=7,
            created_by_user_id=1,
            updated_by_user_id=1,
            next_run_at="2026-03-12T03:00:00+00:00",
        )

    deleted = await repo.delete_schedule(str(created["id"]), deleted_at="2026-03-10T12:00:00+00:00")
    assert deleted is True

    replacement = await repo.create_schedule(
        dataset="media",
        target_user_id=7,
        frequency="weekly",
        time_of_day="03:00",
        timezone="UTC",
        anchor_day_of_week=2,
        anchor_day_of_month=None,
        retention_count=7,
        created_by_user_id=1,
        updated_by_user_id=1,
        next_run_at="2026-03-12T03:00:00+00:00",
    )
    assert replacement["id"] != created["id"]


@pytest.mark.asyncio
async def test_backup_schedules_repo_sqlite_claim_run_slot_is_idempotent(tmp_path, monkeypatch, fresh_services) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.backup_schedules_repo import (
        AuthnzBackupSchedulesRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = AuthnzBackupSchedulesRepo(pool)
    await repo.ensure_schema()

    created = await repo.create_schedule(
        dataset="authnz",
        target_user_id=None,
        frequency="daily",
        time_of_day="02:00",
        timezone="UTC",
        anchor_day_of_week=None,
        anchor_day_of_month=None,
        retention_count=30,
        created_by_user_id=1,
        updated_by_user_id=1,
        next_run_at="2026-03-11T02:00:00+00:00",
    )

    claimed = await repo.claim_run_slot(
        schedule_id=str(created["id"]),
        scheduled_for="2026-03-11T02:00:00+00:00",
        run_slot_key=f"{created['id']}:2026-03-11T02:00:00+00:00",
        enqueued_at=datetime.now(timezone.utc).isoformat(),
    )
    assert claimed is not None
    assert claimed["status"] == "queued"

    duplicate = await repo.claim_run_slot(
        schedule_id=str(created["id"]),
        scheduled_for="2026-03-11T02:00:00+00:00",
        run_slot_key=f"{created['id']}:2026-03-11T02:00:00+00:00",
        enqueued_at=datetime.now(timezone.utc).isoformat(),
    )
    assert duplicate is None

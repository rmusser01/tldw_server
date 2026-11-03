from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_maintenance_rotation_runs_repo_sqlite_roundtrip(tmp_path, monkeypatch, fresh_services) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.maintenance_rotation_runs_repo import (
        AuthnzMaintenanceRotationRunsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = AuthnzMaintenanceRotationRunsRepo(pool)
    await repo.ensure_schema()

    created = await repo.create_run(
        mode="dry_run",
        domain="jobs",
        queue="default",
        job_type="encryption_rotation",
        fields_json='["payload","result"]',
        limit=250,
        requested_by_user_id=7,
        requested_by_label="ops-admin@example.com",
        confirmation_recorded=False,
        scope_summary="domain=jobs, queue=default, fields=payload,result, limit=250",
        key_source="env:jobs_crypto_rotate",
    )

    assert created["mode"] == "dry_run"
    assert created["status"] == "queued"
    assert created["domain"] == "jobs"
    assert created["fields_json"] == '["payload","result"]'
    assert created["limit"] == 250
    assert created["requested_by_user_id"] == 7
    assert created["requested_by_label"] == "ops-admin@example.com"
    assert created["confirmation_recorded"] is False
    assert created["scope_summary"] == "domain=jobs, queue=default, fields=payload,result, limit=250"
    assert created["key_source"] == "env:jobs_crypto_rotate"
    assert created["affected_count"] is None
    assert created["job_id"] is None
    assert created["error_message"] is None
    assert created["started_at"] is None
    assert created["completed_at"] is None

    fetched = await repo.get_run(str(created["id"]))
    assert fetched is not None
    assert fetched["id"] == created["id"]
    assert fetched["created_at"] is not None


@pytest.mark.asyncio
async def test_maintenance_rotation_runs_repo_sqlite_lists_newest_first_and_updates_status(
    tmp_path,
    monkeypatch, fresh_services
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.maintenance_rotation_runs_repo import (
        AuthnzMaintenanceRotationRunsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = AuthnzMaintenanceRotationRunsRepo(pool)
    await repo.ensure_schema()

    first = await repo.create_run(
        mode="dry_run",
        domain="jobs",
        queue="default",
        job_type="encryption_rotation",
        fields_json='["payload"]',
        limit=100,
        requested_by_user_id=9,
        requested_by_label="first-admin@example.com",
        confirmation_recorded=False,
        scope_summary="domain=jobs, queue=default, fields=payload, limit=100",
        key_source="env:jobs_crypto_rotate",
    )
    second = await repo.create_run(
        mode="execute",
        domain="jobs",
        queue="priority",
        job_type="encryption_rotation",
        fields_json='["payload","result"]',
        limit=50,
        requested_by_user_id=11,
        requested_by_label="second-admin@example.com",
        confirmation_recorded=True,
        scope_summary="domain=jobs, queue=priority, fields=payload,result, limit=50",
        key_source="env:jobs_crypto_rotate",
    )

    await repo.mark_running(str(first["id"]), job_id="job-dry-run")
    await repo.mark_complete(str(first["id"]), affected_count=14)
    await repo.mark_running(str(second["id"]), job_id="job-execute")
    await repo.mark_failed(str(second["id"]), error_message="rotation worker failed")

    listed, total = await repo.list_runs(limit=20, offset=0)
    assert total == 2
    assert [row["id"] for row in listed] == [second["id"], first["id"]]

    first_fetched = await repo.get_run(str(first["id"]))
    assert first_fetched is not None
    assert first_fetched["status"] == "complete"
    assert first_fetched["job_id"] == "job-dry-run"
    assert first_fetched["affected_count"] == 14
    assert first_fetched["started_at"] is not None
    assert first_fetched["completed_at"] is not None

    second_fetched = await repo.get_run(str(second["id"]))
    assert second_fetched is not None
    assert second_fetched["status"] == "failed"
    assert second_fetched["job_id"] == "job-execute"
    assert second_fetched["error_message"] == "rotation worker failed"
    assert second_fetched["started_at"] is not None
    assert second_fetched["completed_at"] is not None


@pytest.mark.asyncio
async def test_maintenance_rotation_runs_repo_sqlite_enforces_single_active_execute_run(
    tmp_path,
    monkeypatch, fresh_services
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.maintenance_rotation_runs_repo import (
        AuthnzMaintenanceRotationRunsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = AuthnzMaintenanceRotationRunsRepo(pool)
    await repo.ensure_schema()

    first_execute = await repo.create_run(
        mode="execute",
        domain="jobs",
        queue="default",
        job_type="encryption_rotation",
        fields_json='["payload","result"]',
        limit=500,
        requested_by_user_id=5,
        requested_by_label="ops@example.com",
        confirmation_recorded=True,
        scope_summary="domain=jobs, queue=default, fields=payload,result, limit=500",
        key_source="env:jobs_crypto_rotate",
    )

    assert await repo.has_active_execute_run() is True

    with pytest.raises(Exception):
        await repo.create_run(
            mode="execute",
            domain="jobs",
            queue="default",
            job_type="encryption_rotation",
            fields_json='["payload"]',
            limit=100,
            requested_by_user_id=6,
            requested_by_label="other-ops@example.com",
            confirmation_recorded=True,
            scope_summary="domain=jobs, queue=default, fields=payload, limit=100",
            key_source="env:jobs_crypto_rotate",
        )

    await repo.mark_running(str(first_execute["id"]), job_id="job-active-execute")
    assert await repo.has_active_execute_run() is True

    await repo.mark_complete(str(first_execute["id"]), affected_count=88)
    assert await repo.has_active_execute_run() is False

    replacement = await repo.create_run(
        mode="execute",
        domain="jobs",
        queue="default",
        job_type="encryption_rotation",
        fields_json='["payload"]',
        limit=100,
        requested_by_user_id=6,
        requested_by_label="other-ops@example.com",
        confirmation_recorded=True,
        scope_summary="domain=jobs, queue=default, fields=payload, limit=100",
        key_source="env:jobs_crypto_rotate",
    )

    assert replacement["id"] != first_execute["id"]
    assert replacement["status"] == "queued"

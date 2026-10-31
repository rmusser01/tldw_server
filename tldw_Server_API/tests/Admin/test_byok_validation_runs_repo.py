from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_byok_validation_runs_repo_sqlite_roundtrip(tmp_path, monkeypatch, fresh_services) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.byok_validation_runs_repo import (
        AuthnzByokValidationRunsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = AuthnzByokValidationRunsRepo(pool)
    await repo.ensure_schema()

    created = await repo.create_run(
        org_id=42,
        provider="openai",
        requested_by_user_id=7,
        requested_by_label="ops-admin@example.com",
        scope_summary="org=42, provider=openai",
    )

    assert created["status"] == "queued"
    assert created["org_id"] == 42
    assert created["provider"] == "openai"
    assert created["keys_checked"] is None
    assert created["valid_count"] is None
    assert created["invalid_count"] is None
    assert created["error_count"] is None
    assert created["requested_by_user_id"] == 7
    assert created["requested_by_label"] == "ops-admin@example.com"
    assert created["job_id"] is None
    assert created["scope_summary"] == "org=42, provider=openai"
    assert created["error_message"] is None
    assert created["started_at"] is None
    assert created["completed_at"] is None

    fetched = await repo.get_run(str(created["id"]))
    assert fetched is not None
    assert fetched["id"] == created["id"]
    assert fetched["created_at"] is not None


@pytest.mark.asyncio
async def test_byok_validation_runs_repo_sqlite_lists_newest_first_and_updates_status(
    tmp_path,
    monkeypatch, fresh_services
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.byok_validation_runs_repo import (
        AuthnzByokValidationRunsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = AuthnzByokValidationRunsRepo(pool)
    await repo.ensure_schema()

    first = await repo.create_run(
        org_id=42,
        provider="openai",
        requested_by_user_id=9,
        requested_by_label="first-admin@example.com",
        scope_summary="org=42, provider=openai",
    )

    await repo.mark_running(str(first["id"]), job_id="job-byok-openai")
    await repo.mark_complete(
        str(first["id"]),
        keys_checked=14,
        valid_count=11,
        invalid_count=2,
        error_count=1,
    )

    second = await repo.create_run(
        org_id=None,
        provider=None,
        requested_by_user_id=11,
        requested_by_label="second-admin@example.com",
        scope_summary="global",
    )
    await repo.mark_running(str(second["id"]), job_id="job-byok-global")
    await repo.mark_failed(str(second["id"]), error_message="validation worker failed")

    listed, total = await repo.list_runs(limit=20, offset=0)
    assert total == 2
    assert [row["id"] for row in listed] == [second["id"], first["id"]]

    first_fetched = await repo.get_run(str(first["id"]))
    assert first_fetched is not None
    assert first_fetched["status"] == "complete"
    assert first_fetched["job_id"] == "job-byok-openai"
    assert first_fetched["keys_checked"] == 14
    assert first_fetched["valid_count"] == 11
    assert first_fetched["invalid_count"] == 2
    assert first_fetched["error_count"] == 1
    assert first_fetched["started_at"] is not None
    assert first_fetched["completed_at"] is not None

    second_fetched = await repo.get_run(str(second["id"]))
    assert second_fetched is not None
    assert second_fetched["status"] == "failed"
    assert second_fetched["job_id"] == "job-byok-global"
    assert second_fetched["error_message"] == "validation worker failed"
    assert second_fetched["started_at"] is not None
    assert second_fetched["completed_at"] is not None


@pytest.mark.asyncio
async def test_byok_validation_runs_repo_sqlite_enforces_single_active_run(
    tmp_path,
    monkeypatch, fresh_services
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.byok_validation_runs_repo import (
        AuthnzByokValidationRunsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = AuthnzByokValidationRunsRepo(pool)
    await repo.ensure_schema()

    first_run = await repo.create_run(
        org_id=42,
        provider="openai",
        requested_by_user_id=5,
        requested_by_label="ops@example.com",
        scope_summary="org=42, provider=openai",
    )

    assert await repo.has_active_run() is True

    with pytest.raises(Exception):
        await repo.create_run(
            org_id=84,
            provider="anthropic",
            requested_by_user_id=6,
            requested_by_label="other-ops@example.com",
            scope_summary="org=84, provider=anthropic",
        )

    await repo.mark_running(str(first_run["id"]), job_id="job-byok-active")
    assert await repo.has_active_run() is True

    await repo.mark_complete(
        str(first_run["id"]),
        keys_checked=8,
        valid_count=8,
        invalid_count=0,
        error_count=0,
    )
    assert await repo.has_active_run() is False

    replacement = await repo.create_run(
        org_id=84,
        provider="anthropic",
        requested_by_user_id=6,
        requested_by_label="other-ops@example.com",
        scope_summary="org=84, provider=anthropic",
    )

    assert replacement["id"] != first_run["id"]
    assert replacement["status"] == "queued"

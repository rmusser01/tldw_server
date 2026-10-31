from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_authnz_quotas_repo_sqlite_increment_and_check(tmp_path, monkeypatch, fresh_services):
    """AuthnzQuotasRepo increment helpers should work on SQLite."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.quotas_repo import AuthnzQuotasRepo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()

    pool = await get_db_pool()
    repo = AuthnzQuotasRepo(db_pool=pool)
    # Ensure vk_* counters schema exists via the repo helper.
    await repo.ensure_schema()

    # JWT quota: limit 2, ensure third call is denied.
    allowed1, count1 = await repo.increment_and_check_jwt_quota(
        jti="jwt-sqlite-1",
        counter_type="test",
        limit=2,
        bucket=None,
    )
    allowed2, count2 = await repo.increment_and_check_jwt_quota(
        jti="jwt-sqlite-1",
        counter_type="test",
        limit=2,
        bucket=None,
    )
    allowed3, count3 = await repo.increment_and_check_jwt_quota(
        jti="jwt-sqlite-1",
        counter_type="test",
        limit=2,
        bucket=None,
    )

    assert allowed1 is True and count1 == 1
    assert allowed2 is True and count2 == 2
    assert allowed3 is False and count3 == 3

    # API key quota: same pattern, with a different bucket label.
    allowed4, count4 = await repo.increment_and_check_api_key_quota(
        api_key_id=123,
        counter_type="audio",
        limit=2,
        bucket="unit",
    )
    allowed5, count5 = await repo.increment_and_check_api_key_quota(
        api_key_id=123,
        counter_type="audio",
        limit=2,
        bucket="unit",
    )
    allowed6, count6 = await repo.increment_and_check_api_key_quota(
        api_key_id=123,
        counter_type="audio",
        limit=2,
        bucket="unit",
    )

    assert allowed4 is True and count4 == 1
    assert allowed5 is True and count5 == 2
    assert allowed6 is False and count6 == 3

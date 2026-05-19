from __future__ import annotations

import uuid
from pathlib import Path

import pytest

pytest_plugins = ("tldw_Server_API.tests.AuthNZ.conftest",)


@pytest.mark.asyncio
async def test_org_stt_settings_repo_sqlite(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.org_stt_settings_repo import (
        AuthnzOrgSttSettingsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username="org-stt-sqlite",
        email="org-stt-sqlite@example.com",
        password_hash="hashed-password",
        role="admin",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    user_id = int(created_user["id"])

    await pool.execute(
        """
        INSERT INTO organizations (uuid, name, slug, owner_user_id, is_active, metadata)
        VALUES (?, ?, ?, ?, 1, ?)
        """,
        (
            str(uuid.uuid4()),
            "SQLite STT Org",
            "sqlite-stt-org",
            user_id,
            "{}",
        ),
    )
    org_row = await pool.fetchone("SELECT id FROM organizations WHERE slug = ?", ("sqlite-stt-org",))
    org_id = int(org_row["id"] if isinstance(org_row, dict) else org_row[0])

    repo = AuthnzOrgSttSettingsRepo(pool)
    await repo.ensure_tables()

    created = await repo.upsert_settings(
        org_id=org_id,
        delete_audio_after_success=False,
        audio_retention_hours=24.0,
        redact_pii=True,
        allow_unredacted_partials=False,
        redact_categories=["email", "phone"],
        updated_by=user_id,
    )
    assert created["org_id"] == org_id
    assert created["delete_audio_after_success"] is False
    assert created["audio_retention_hours"] == 24.0
    assert created["redact_pii"] is True
    assert created["allow_unredacted_partials"] is False
    assert created["redact_categories"] == ["email", "phone"]

    fetched = await repo.get_settings(org_id)
    assert fetched == created

    updated = await repo.upsert_settings(
        org_id=org_id,
        delete_audio_after_success=True,
        audio_retention_hours=0.0,
        redact_pii=False,
        allow_unredacted_partials=True,
        redact_categories=["ssn"],
        updated_by=user_id,
    )
    assert updated["delete_audio_after_success"] is True
    assert updated["audio_retention_hours"] == 0.0
    assert updated["redact_pii"] is False
    assert updated["allow_unredacted_partials"] is True
    assert updated["redact_categories"] == ["ssn"]

    # Teardown singleton state to avoid polluting other tests
    reset_settings()
    await reset_db_pool()

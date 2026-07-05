from __future__ import annotations

from datetime import datetime, timezone
import uuid

import pytest

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)


@pytest.mark.asyncio
async def test_authnz_mfa_repo_basic_sqlite(tmp_path, monkeypatch) -> None:
    """AuthnzMfaRepo should update and read MFA fields on a SQLite-backed AuthNZ DB."""
    from pathlib import Path

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.mfa_repo import AuthnzMfaRepo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    # Ensure core AuthNZ tables (including MFA columns) exist via migrations.
    ensure_authnz_tables(Path(str(db_path)))

    # Seed a user row via UsersDB so schema details remain centralized.
    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username="mfa-user",
        email="mfa@example.com",
        password_hash="hashed-password",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    user_id = int(created_user["id"])

    repo = AuthnzMfaRepo(pool)
    now = datetime.now(timezone.utc)

    await repo.set_mfa_config(
        user_id=user_id,
        encrypted_secret="enc-secret",
        backup_codes_json='["code1","code2"]',
        updated_at=now,
    )

    encrypted = await repo.get_encrypted_totp_secret(user_id)
    assert encrypted == "enc-secret"

    status_row = await repo.get_mfa_status_row(user_id)
    assert status_row is not None
    assert bool(status_row.get("two_factor_enabled"))
    assert bool(status_row.get("has_secret"))
    assert bool(status_row.get("has_backup_codes"))

    # Repo should expose raw backup_codes JSON and allow updates
    raw_codes = await repo.get_backup_codes_json(user_id)
    assert raw_codes == '["code1","code2"]'

    await repo.update_backup_codes_json(
        user_id=user_id,
        backup_codes_json='["code2"]',
    )
    raw_codes_updated = await repo.get_backup_codes_json(user_id)
    assert raw_codes_updated == '["code2"]'

    await repo.clear_mfa_config(user_id=user_id, updated_at=now)

    encrypted_after = await repo.get_encrypted_totp_secret(user_id)
    assert encrypted_after is None

    status_after = await repo.get_mfa_status_row(user_id)
    assert status_after is not None
    assert not bool(status_after.get("two_factor_enabled"))
    assert not bool(status_after.get("has_secret"))
    assert not bool(status_after.get("has_backup_codes"))

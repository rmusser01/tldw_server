from __future__ import annotations

from datetime import datetime, timezone
import base64
import json
import uuid

import pytest

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


@pytest.mark.asyncio
async def test_org_provider_secrets_repo_sqlite(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
        AuthnzOrgProviderSecretsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        build_secret_payload,
        encrypt_byok_payload,
        key_hint_for_api_key,
    )
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username="byok-org",
        email="byok-org@example.com",
        password_hash="hashed-password",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    user_id = int(created_user["id"])

    repo = AuthnzOrgProviderSecretsRepo(pool)
    await repo.ensure_tables()

    payload = build_secret_payload("sk-test", {"org_id": "org-1"})
    envelope = encrypt_byok_payload(payload)
    encrypted_blob = json.dumps(envelope)
    key_hint = key_hint_for_api_key("sk-test")
    now = datetime.now(timezone.utc)

    await repo.upsert_secret(
        scope_type="org",
        scope_id=1,
        provider="OpenAI",
        encrypted_blob=encrypted_blob,
        key_hint=key_hint,
        metadata={"label": "org-shared"},
        updated_at=now,
        created_by=user_id,
        updated_by=user_id,
    )

    row = await repo.fetch_secret("org", 1, "openai")
    assert row is not None
    assert row["provider"] == "openai"
    assert row["encrypted_blob"] == encrypted_blob
    assert row["key_hint"] == key_hint
    assert row["created_by"] == user_id
    assert row["updated_by"] == user_id

    items = await repo.list_secrets(scope_type="org", scope_id=1)
    assert len(items) == 1
    assert items[0]["provider"] == "openai"

    items_filtered = await repo.list_secrets(scope_type="org", scope_id=1, provider="openai")
    assert len(items_filtered) == 1

    await repo.touch_last_used("org", 1, "openai", now)
    refreshed = await repo.fetch_secret("org", 1, "openai")
    assert refreshed is not None
    assert refreshed["last_used_at"] is not None

    deleted = await repo.delete_secret("org", 1, "openai")
    assert deleted
    missing = await repo.fetch_secret("org", 1, "openai")
    assert missing is None
    revoked_rows = await repo.list_secrets(scope_type="org", scope_id=1, include_revoked=True)
    assert len(revoked_rows) == 1
    assert revoked_rows[0]["revoked_at"] is not None

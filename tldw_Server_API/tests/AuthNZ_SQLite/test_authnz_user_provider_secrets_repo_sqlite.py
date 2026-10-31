from __future__ import annotations

import base64
import json
import sqlite3
import uuid
from datetime import datetime, timezone

import pytest

from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import set_authnz_test_user_active

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


@pytest.mark.asyncio
async def test_legacy_nullable_owner_fails_active_read_and_cas_sqlite(
    tmp_path,
    monkeypatch,
) -> None:
    """Legacy NULL owner state is never treated as active credential ownership."""
    from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
        AuthnzUserProviderSecretsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import Settings, reset_settings
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        build_secret_payload,
        encrypt_byok_payload,
    )

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    reset_settings()
    db_path = tmp_path / "legacy-nullable-users.db"
    now = datetime.now(timezone.utc)
    original_blob = json.dumps(
        encrypt_byok_payload(build_secret_payload("sk-original-legacy-key"))
    )
    replacement_blob = json.dumps(
        encrypt_byok_payload(build_secret_payload("sk-replacement-must-not-persist"))
    )
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                is_active INTEGER DEFAULT 1
            );
            CREATE TABLE user_provider_secrets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                encrypted_blob TEXT NOT NULL,
                key_hint TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT,
                created_by INTEGER,
                updated_by INTEGER,
                revoked_by INTEGER,
                revoked_at TEXT,
                UNIQUE(user_id, provider)
            );
            """
        )
        conn.execute("INSERT INTO users (id, is_active) VALUES (?, ?)", (1, None))
        conn.execute(
            """
            INSERT INTO user_provider_secrets (
                user_id, provider, encrypted_blob, key_hint, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (1, "openai", original_blob, "legacy", now.isoformat(), now.isoformat()),
        )

    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{db_path}",
        JWT_SECRET_KEY="legacy-nullable-test-key-32-characters!",
    )
    pool = DatabasePool(settings)
    pool.db_path = str(db_path)
    pool._sqlite_fs_path = str(db_path)
    pool._sqlite_uri = False
    pool._initialized = True
    repo = AuthnzUserProviderSecretsRepo(pool)
    try:
        columns = await pool.fetchall("PRAGMA table_info(users)")
        active_column = next(row for row in columns if row["name"] == "is_active")
        assert active_column["notnull"] == 0

        assert await repo.fetch_secret_for_active_user(
            1,
            "openai",
            include_revoked=True,
        ) is None
        assert not await repo.update_secret_if_active_and_unchanged(
            user_id=1,
            provider="openai",
            encrypted_blob=replacement_blob,
            expected_encrypted_blob=original_blob,
            key_hint="replacement",
            metadata=None,
            updated_at=now,
            updated_by=1,
        )

        stored = await repo.fetch_secret_for_user(1, "openai", include_revoked=True)
        assert stored is not None
        assert stored["encrypted_blob"] == original_blob
    finally:
        await pool.close()
        reset_settings()


@pytest.mark.asyncio
async def test_user_provider_secrets_repo_sqlite(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
        AuthnzUserProviderSecretsRepo,
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
        username="byok-user",
        email="byok@example.com",
        password_hash="hashed-password",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    user_id = int(created_user["id"])

    repo = AuthnzUserProviderSecretsRepo(pool)
    await repo.ensure_tables()

    payload = build_secret_payload("sk-test", {"org_id": "org-1"})
    envelope = encrypt_byok_payload(payload)
    encrypted_blob = json.dumps(envelope)
    key_hint = key_hint_for_api_key("sk-test")
    now = datetime.now(timezone.utc)

    await repo.upsert_secret(
        user_id=user_id,
        provider="OpenAI",
        encrypted_blob=encrypted_blob,
        key_hint=key_hint,
        metadata={"label": "test"},
        updated_at=now,
        created_by=user_id,
        updated_by=user_id,
    )

    row = await repo.fetch_secret_for_user(user_id, "openai")
    assert row is not None
    assert row["provider"] == "openai"
    assert row["encrypted_blob"] == encrypted_blob
    assert row["key_hint"] == key_hint
    assert row["created_by"] == user_id
    assert row["updated_by"] == user_id

    active_owner_row = await repo.fetch_secret_for_active_user(
        user_id,
        "openai",
        include_revoked=True,
    )
    assert active_owner_row is not None
    await set_authnz_test_user_active(pool, user_id, False)
    assert await repo.fetch_secret_for_active_user(
        user_id,
        "openai",
        include_revoked=True,
    ) is None
    await set_authnz_test_user_active(pool, user_id, True)

    items = await repo.list_secrets_for_user(user_id)
    assert len(items) == 1
    assert items[0]["provider"] == "openai"

    unknown_row = await repo.upsert_secret(
        user_id=user_id,
        provider=" Foo_Bar ",
        encrypted_blob=encrypted_blob,
        key_hint=key_hint,
        metadata=None,
        updated_at=now,
        created_by=user_id,
        updated_by=user_id,
    )
    assert unknown_row["provider"] == "foo_bar"
    unknown = await repo.fetch_secret_for_user(user_id, "foo_bar")
    assert unknown is not None
    assert unknown["provider"] == "foo_bar"

    raw_now = now.isoformat()
    await pool.execute(
        """
        INSERT INTO user_provider_secrets (
            user_id, provider, encrypted_blob, key_hint, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, "aws_bedrock", encrypted_blob, key_hint, raw_now, raw_now),
    )
    accepted_alias = await repo.fetch_secret_for_user(user_id, "bedrock")
    assert accepted_alias is not None
    assert accepted_alias["provider"] == "aws_bedrock"
    active_by_provider = {
        item["provider"]: item for item in await repo.list_secrets_for_user(user_id)
    }
    assert active_by_provider["bedrock"]["key_hint"] == key_hint

    revoked_at = datetime.now(timezone.utc).isoformat()
    await pool.execute(
        """
        UPDATE user_provider_secrets
        SET revoked_at = ?, updated_at = ?
        WHERE user_id = ? AND provider = ?
        """,
        (revoked_at, revoked_at, user_id, "aws_bedrock"),
    )
    assert await repo.fetch_secret_for_user(user_id, "bedrock") is None
    accepted_tombstone = await repo.fetch_secret_for_user(
        user_id,
        "bedrock",
        include_revoked=True,
    )
    assert accepted_tombstone is not None
    assert accepted_tombstone["provider"] == "aws_bedrock"

    await repo.touch_last_used(user_id, "openai", now)
    refreshed = await repo.fetch_secret_for_user(user_id, "openai")
    assert refreshed is not None
    assert refreshed["last_used_at"] is not None

    deleted = await repo.delete_secret(user_id, "openai")
    assert deleted
    missing = await repo.fetch_secret_for_user(user_id, "openai")
    assert missing is None
    revoked_rows = await repo.list_secrets_for_user(user_id, include_revoked=True)
    revoked_by_provider = {row["provider"]: row for row in revoked_rows}
    assert set(revoked_by_provider) == {"openai", "foo_bar", "bedrock"}
    assert revoked_by_provider["openai"]["revoked_at"] is not None
    assert revoked_by_provider["bedrock"]["revoked_at"] is not None

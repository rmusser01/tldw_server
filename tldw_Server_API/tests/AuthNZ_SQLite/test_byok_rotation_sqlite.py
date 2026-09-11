from __future__ import annotations

import asyncio
import base64
import contextlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.byok_rotation import rotate_byok_secrets
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
    AuthnzUserProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    build_secret_payload,
    decrypt_byok_payload,
    dumps_envelope,
    encrypt_byok_payload,
    loads_envelope,
)
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB, reset_users_db


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


@pytest.mark.asyncio
async def test_byok_rotation_reencrypts_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "users.db"
    old_key = _b64_key(b"a")
    new_key = _b64_key(b"b")

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", old_key)

    reset_settings()
    await reset_db_pool()
    await reset_users_db()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))
    users_db = UsersDB(pool)
    await users_db.initialize()
    user = await users_db.create_user(
        username="byok-rotate-user",
        email="byok-rotate@example.com",
        password_hash="hashed",
        role="user",
        is_active=True,
        is_verified=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )

    payload = build_secret_payload("sk-rotate-test")
    encrypted_blob = dumps_envelope(encrypt_byok_payload(payload))

    repo = AuthnzUserProviderSecretsRepo(pool)
    await repo.upsert_secret(
        user_id=int(user["id"]),
        provider="openai",
        encrypted_blob=encrypted_blob,
        key_hint="test",
        metadata=None,
        updated_at=datetime.now(timezone.utc),
    )

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", new_key)
    monkeypatch.setenv("BYOK_SECONDARY_ENCRYPTION_KEY", old_key)
    reset_settings()

    summary = await rotate_byok_secrets(batch_size=10)
    assert summary.tables["user_provider_secrets"].updated == 1

    monkeypatch.setenv("BYOK_SECONDARY_ENCRYPTION_KEY", "")
    reset_settings()

    row = await pool.fetchone(
        "SELECT encrypted_blob FROM user_provider_secrets WHERE user_id = ? AND provider = ?",
        (int(user["id"]), "openai"),
    )
    assert row
    decrypted = decrypt_byok_payload(loads_envelope(row["encrypted_blob"]))
    assert decrypted["api_key"] == "sk-rotate-test"


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_byok_rotation_serializes_with_live_openai_refresh_sqlite(tmp_path, monkeypatch):
    """Rotation re-reads the refresh winner instead of restoring its stale snapshot."""
    db_path = tmp_path / "users.db"
    old_key = _b64_key(b"a")
    new_key = _b64_key(b"b")

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENABLED", "1")
    monkeypatch.setenv("BYOK_ALLOWED_PROVIDERS", "openai")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", old_key)
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "1")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "oauth-client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "oauth-client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example.com/token")

    reset_settings()
    await reset_db_pool()
    await reset_users_db()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))
    users_db = UsersDB(pool)
    await users_db.initialize()
    user = await users_db.create_user(
        username="byok-rotate-race-user",
        email="byok-rotate-race@example.com",
        password_hash="hashed",
        role="user",
        is_active=True,
        is_verified=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    user_id = int(user["id"])

    now = datetime.now(timezone.utc)
    original_payload = {
        "credential_version": 2,
        "active_auth_source": "oauth",
        "credentials": {
            "oauth": {
                "access_token": "stale-access-token",
                "refresh_token": "single-use-refresh-token",
                "expires_at": "2000-01-01T00:00:00+00:00",
            }
        },
    }
    repo = AuthnzUserProviderSecretsRepo(pool)
    await repo.upsert_secret(
        user_id=user_id,
        provider="openai",
        encrypted_blob=dumps_envelope(encrypt_byok_payload(original_payload)),
        key_hint="oauth",
        metadata=None,
        updated_at=now,
        created_by=user_id,
        updated_by=user_id,
    )

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", new_key)
    monkeypatch.setenv("BYOK_SECONDARY_ENCRYPTION_KEY", old_key)
    reset_settings()

    from tldw_Server_API.app.core.AuthNZ import byok_rotation, byok_runtime

    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()
    rotation_lock_attempted = asyncio.Event()
    original_rotation_lock = byok_rotation.openai_credential_mutation_lock

    @contextlib.asynccontextmanager
    async def tracked_rotation_lock(**kwargs):
        rotation_lock_attempted.set()
        async with original_rotation_lock(**kwargs) as locked_repo:
            yield locked_repo

    async def runtime_token_refresh(**_kwargs):
        refresh_started.set()
        await release_refresh.wait()
        return {
            "access_token": "runtime-winning-access-token",
            "refresh_token": "runtime-winning-refresh-token",
            "expires_in": 3600,
        }

    monkeypatch.setattr(byok_runtime, "_openai_oauth_token_refresh", runtime_token_refresh)
    monkeypatch.setattr(byok_rotation, "openai_credential_mutation_lock", tracked_rotation_lock)

    runtime_task = asyncio.create_task(
        byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=user_id,
            force_oauth_refresh=True,
        )
    )
    await asyncio.wait_for(refresh_started.wait(), timeout=10)
    rotation_task = asyncio.create_task(rotate_byok_secrets(batch_size=10, pool=pool))
    await asyncio.wait_for(rotation_lock_attempted.wait(), timeout=10)
    assert not rotation_task.done()
    release_refresh.set()
    runtime_result, summary = await asyncio.gather(runtime_task, rotation_task)

    assert runtime_result.api_key == "runtime-winning-access-token"
    assert summary.tables["user_provider_secrets"].updated == 1

    monkeypatch.setenv("BYOK_SECONDARY_ENCRYPTION_KEY", "")
    reset_settings()
    row = await repo.fetch_secret_for_user(user_id, "openai")
    stored_payload = decrypt_byok_payload(loads_envelope(row["encrypted_blob"]))
    assert stored_payload["credentials"]["oauth"]["access_token"] == "runtime-winning-access-token"
    assert stored_payload["credentials"]["oauth"]["refresh_token"] == "runtime-winning-refresh-token"

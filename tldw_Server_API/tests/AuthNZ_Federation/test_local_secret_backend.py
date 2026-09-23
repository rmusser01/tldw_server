from __future__ import annotations

import base64
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.settings import Settings, reset_settings
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB


pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


def _sqlite_settings(db_path: Path) -> Settings:
    return Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{db_path}",
        JWT_SECRET_KEY="x" * 32,
    )


async def _insert_test_user(pool: DatabasePool, *, username: str, email: str) -> int:
    users = UsersDB(pool)
    await users.initialize(ensure_schema=False)
    user = await users.create_user(  # nosec B106 # Inert fixture hash, never used for login
        username=username,
        email=email,
        password_hash="not-a-real-password-hash",
    )
    return int(user["id"])


async def test_local_secret_backend_resolve_for_use_returns_ephemeral_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    reset_settings()

    from tldw_Server_API.app.core.AuthNZ.secret_backends.local_encrypted import (
        LocalEncryptedSecretBackend,
    )

    db_path = tmp_path / "local_secret_backend.db"
    pool = DatabasePool(_sqlite_settings(db_path))
    await pool.initialize()

    try:
        user_id = await _insert_test_user(
            pool,
            username="secret-owner",
            email="secret-owner@example.com",
        )
        backend = LocalEncryptedSecretBackend(db_pool=pool)
        await backend.ensure_tables()

        ref = await backend.store_ref(
            owner_scope_type="user",
            owner_scope_id=user_id,
            provider_key="openai",
            payload={
                "api_key": "sk-test",
                "credential_fields": {"base_url": "https://api.example.com/v1"},
            },
        )
        resolved = await backend.resolve_for_use(ref["id"])

        assert ref["backend_name"] == "local_encrypted_v1"
        assert resolved["backend_name"] == "local_encrypted_v1"
        assert resolved["material"]["api_key"] == "sk-test"
        assert resolved["material"]["credential_fields"]["base_url"] == "https://api.example.com/v1"
        assert resolved["expires_at"] is not None
    finally:
        await pool.close()
        reset_settings()

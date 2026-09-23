from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB


pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


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


async def test_create_and_fetch_identity_provider(tmp_path: Path) -> None:
    from tldw_Server_API.app.core.AuthNZ.repos.identity_provider_repo import IdentityProviderRepo

    db_path = tmp_path / "identity_provider_repo.db"
    pool = DatabasePool(_sqlite_settings(db_path))
    await pool.initialize()

    try:
        repo = IdentityProviderRepo(db_pool=pool)
        await repo.ensure_tables()

        created = await repo.create_provider(
            slug="corp",
            provider_type="oidc",
            owner_scope_type="global",
            owner_scope_id=None,
            enabled=False,
            issuer="https://issuer.example.com",
            claim_mapping={"email": "email"},
            provisioning_policy={"mode": "jit_grant_only"},
        )
        fetched = await repo.get_provider(created["id"])

        assert created["slug"] == "corp"
        assert fetched is not None
        assert fetched["slug"] == "corp"
        assert fetched["provider_type"] == "oidc"
        assert fetched["owner_scope_type"] == "global"
    finally:
        await pool.close()


async def test_link_and_fetch_federated_identity(tmp_path: Path) -> None:
    from tldw_Server_API.app.core.AuthNZ.repos.federated_identity_repo import FederatedIdentityRepo
    from tldw_Server_API.app.core.AuthNZ.repos.identity_provider_repo import IdentityProviderRepo

    db_path = tmp_path / "federated_identity_repo.db"
    pool = DatabasePool(_sqlite_settings(db_path))
    await pool.initialize()

    try:
        provider_repo = IdentityProviderRepo(db_pool=pool)
        identity_repo = FederatedIdentityRepo(db_pool=pool)
        await provider_repo.ensure_tables()
        await identity_repo.ensure_tables()

        provider = await provider_repo.create_provider(
            slug="corp",
            provider_type="oidc",
            owner_scope_type="global",
            owner_scope_id=None,
            enabled=True,
            issuer="https://issuer.example.com",
            claim_mapping={"email": "email"},
            provisioning_policy={"mode": "jit_grant_only"},
        )
        user_id = await _insert_test_user(
            pool,
            username="alice",
            email="alice@example.com",
        )

        created = await identity_repo.upsert_identity(
            identity_provider_id=provider["id"],
            external_subject="sub-123",
            user_id=user_id,
            external_username="alice",
            external_email="alice@example.com",
            last_claims_hash="hash-1",
            status="active",
        )
        fetched = await identity_repo.get_by_provider_subject(
            identity_provider_id=provider["id"],
            external_subject="sub-123",
        )

        assert created["external_subject"] == "sub-123"
        assert fetched is not None
        assert fetched["user_id"] == user_id
        assert fetched["external_email"] == "alice@example.com"
        assert fetched["status"] == "active"
    finally:
        await pool.close()


async def test_federated_identity_upsert_rejects_transferring_subject_to_another_user(tmp_path: Path) -> None:
    from tldw_Server_API.app.core.AuthNZ.repos.federated_identity_repo import FederatedIdentityRepo
    from tldw_Server_API.app.core.AuthNZ.repos.identity_provider_repo import IdentityProviderRepo

    db_path = tmp_path / "federated_identity_transfer_repo.db"
    pool = DatabasePool(_sqlite_settings(db_path))
    await pool.initialize()

    try:
        provider_repo = IdentityProviderRepo(db_pool=pool)
        identity_repo = FederatedIdentityRepo(db_pool=pool)
        await provider_repo.ensure_tables()
        await identity_repo.ensure_tables()

        provider = await provider_repo.create_provider(
            slug="corp",
            provider_type="oidc",
            owner_scope_type="global",
            owner_scope_id=None,
            enabled=True,
            issuer="https://issuer.example.com",
            claim_mapping={"email": "email"},
            provisioning_policy={"mode": "jit_grant_only"},
        )
        first_user_id = await _insert_test_user(pool, username="alice", email="alice@example.com")
        second_user_id = await _insert_test_user(pool, username="bob", email="bob@example.com")

        await identity_repo.upsert_identity(
            identity_provider_id=provider["id"],
            external_subject="sub-123",
            user_id=first_user_id,
            external_username="alice",
            external_email="alice@example.com",
            last_claims_hash="hash-1",
            status="active",
        )

        with pytest.raises(ValueError, match="already linked to a different local user"):
            await identity_repo.upsert_identity(
                identity_provider_id=provider["id"],
                external_subject="sub-123",
                user_id=second_user_id,
                external_username="bob",
                external_email="bob@example.com",
                last_claims_hash="hash-2",
                status="active",
            )
    finally:
        await pool.close()

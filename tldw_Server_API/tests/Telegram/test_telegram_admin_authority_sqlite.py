"""Persisted platform authority governs Telegram secret mutations."""

import base64
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import telegram_support as support
from tldw_Server_API.app.api.v1.schemas.telegram_schemas import TelegramBotConfigUpdate
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import AuthnzOrgProviderSecretsRepo
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.AuthNZ.settings import Settings, reset_settings
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def telegram_sqlite_state(tmp_path, monkeypatch):
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", base64.b64encode(b"t" * 32).decode("ascii"))
    reset_settings()
    pool = DatabasePool(Settings(
        AUTH_MODE="multi_user", DATABASE_URL=f"sqlite:///{tmp_path / 'telegram.db'}",
        JWT_SECRET_KEY="x" * 32,
    ))
    await pool.initialize()
    try:
        admin_id = await ensure_test_user(pool, "telegram-admin", role="admin")
        user_id = await ensure_test_user(pool, "telegram-user", role="user")
        scopes = AuthnzOrgsTeamsRepo(pool)
        org = await scopes.create_organization(name="Telegram org", owner_user_id=None)
        team = await scopes.create_team(org_id=int(org["id"]), name="Telegram team")
        yield pool, {"admin": admin_id, "user": user_id, "org": int(org["id"]), "team": int(team["id"])}
    finally:
        await pool.close()
        reset_settings()


def _principal(user_id: int, scope_type: str, ids: dict[str, int]) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user", user_id=user_id, roles=["admin"], permissions=["system.configure"],
        org_ids=[ids["org"]], team_ids=[ids["team"]] if scope_type == "team" else [],
        active_org_id=ids["org"], active_team_id=ids["team"] if scope_type == "team" else None,
    )


@pytest.mark.parametrize("scope_type", ("org", "team"))
@pytest.mark.parametrize("actor_state", ("forged", "revoked"))
async def test_telegram_denies_unpersisted_platform_authority_without_mutation(
    telegram_sqlite_state, scope_type: str, actor_state: str,
) -> None:
    pool, ids = telegram_sqlite_state
    actor_id = ids["admin"] if actor_state == "revoked" else ids["user"]
    principal = _principal(actor_id, scope_type, ids)
    if actor_state == "revoked":
        await UsersDB(pool).update_user(actor_id, role="user")
    repo = AuthnzOrgProviderSecretsRepo(pool)
    await repo.upsert_secret(
        scope_type=scope_type, scope_id=ids[scope_type], provider="telegram",
        encrypted_blob="unchanged-secret", key_hint="before", metadata={"enabled": False},
        updated_at=datetime.now(timezone.utc),
    )
    before = await repo.fetch_secret(scope_type, ids[scope_type], "telegram", include_revoked=True)
    version_before = await pool.fetchval("SELECT profile_version FROM users WHERE id = ?", (actor_id,))

    async def get_repo():
        return repo

    with pytest.raises(HTTPException) as exc:
        await support.telegram_admin_put_bot_impl(
            principal=principal,
            payload=TelegramBotConfigUpdate(
                bot_token=":".join(("123", "replacement")),
                webhook_secret="-".join(("webhook", "replacement")),
            ),
            get_org_secret_repo=get_repo,
        )
    assert (exc.value.status_code, exc.value.detail) == (403, "Not authorized to manage shared BYOK keys")
    assert await repo.fetch_secret(scope_type, ids[scope_type], "telegram", include_revoked=True) == before
    assert await pool.fetchval("SELECT profile_version FROM users WHERE id = ?", (actor_id,)) == version_before


@pytest.mark.parametrize("scope_type", ("org", "team"))
async def test_persisted_telegram_admin_can_store_and_read_config(
    telegram_sqlite_state, scope_type: str,
) -> None:
    pool, ids = telegram_sqlite_state
    repo = AuthnzOrgProviderSecretsRepo(pool)

    async def get_repo():
        return repo

    principal = _principal(ids["admin"], scope_type, ids)
    response = await support.telegram_admin_put_bot_impl(
        principal=principal,
        payload=TelegramBotConfigUpdate(
            bot_token=":".join(("123", "example")),
            webhook_secret="-".join(("webhook", "example")), bot_username="@ExampleBot",
        ),
        get_org_secret_repo=get_repo,
    )
    fetched = await support.telegram_admin_get_bot_impl(principal=principal, get_org_secret_repo=get_repo)
    assert response.model_dump() == {
        "ok": True, "provider": "telegram",
        "scope_type": scope_type, "scope_id": ids[scope_type], "bot_username": "examplebot", "enabled": True,
    }
    assert fetched == response
    row = await repo.fetch_secret(scope_type, ids[scope_type], "telegram")
    assert (row["created_by"], row["updated_by"]) == (ids["admin"], ids["admin"])

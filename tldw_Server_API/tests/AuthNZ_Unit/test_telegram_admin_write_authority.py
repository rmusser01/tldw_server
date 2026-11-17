"""AuthNZ unit coverage for the request-bound Telegram BYOK actor boundary."""

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import integrations_control_plane, telegram
from tldw_Server_API.app.api.v1.utils.chatops import telegram_support as support
from tldw_Server_API.app.api.v1.schemas.telegram_schemas import TelegramBotConfigUpdate
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    ConnectionPoolExhaustedError,
    DatabaseConcurrencyConflict,
    DatabaseLockError,
)
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    ActorMembershipWriteContext,
    MembershipAuthority,
    MembershipAuthorizationError,
    MembershipScopeNotFound,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
_BUSY_DETAIL = "Authentication database is busy. Please retry shortly."


def _principal(scope_type: str = "org") -> AuthPrincipal:
    return AuthPrincipal(
        kind="user", user_id=101, roles=["admin"], permissions=["system.configure"],
        org_ids=[11], team_ids=[22] if scope_type == "team" else [],
        active_org_id=11, active_team_id=22 if scope_type == "team" else None,
    )


def _payload() -> TelegramBotConfigUpdate:
    return TelegramBotConfigUpdate(
        bot_token=":".join(("123", "example")),
        webhook_secret="-".join(("webhook", "example")), bot_username="@ResearchBot",
    )


@pytest.mark.parametrize("scope_type,scope_id", (("org", 11), ("team", 22)))
@pytest.mark.parametrize("path", (
    "/api/v1/telegram/admin/bot", "/api/v1/integrations/workspace/telegram/bot",
))
async def test_both_telegram_routes_forward_platform_actor_context(
    monkeypatch: pytest.MonkeyPatch, path: str, scope_type: str, scope_id: int,
) -> None:
    captured = []

    class Repo:
        async def ensure_tables(self):
            return None

        async def upsert_secret(self, **kwargs):
            captured.append(kwargs)
            return {}

    async def get_pool():
        return object()

    async def principal():
        return _principal(scope_type)

    monkeypatch.setattr(support, "get_db_pool", get_pool)
    monkeypatch.setattr(support, "AuthnzOrgProviderSecretsRepo", lambda _pool: Repo())
    monkeypatch.setattr(support, "encrypt_byok_payload", lambda _payload: {})
    app = FastAPI()
    app.include_router(telegram.router, prefix="/api/v1")
    app.include_router(integrations_control_plane.router, prefix="/api/v1")
    app.dependency_overrides[get_auth_principal] = principal
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.put(path, json=_payload().model_dump())
    assert response.status_code == 200
    assert response.json() == {
        "ok": True, "provider": "telegram",
        "scope_type": scope_type, "scope_id": scope_id,
        "bot_username": "researchbot", "enabled": True,
    }
    assert len(captured) == 1
    context = captured[0]["authorization_context"]
    assert isinstance(context, ActorMembershipWriteContext)
    assert context.actor_user_id == 101
    assert context.required_authority is MembershipAuthority.PLATFORM_ADMIN
    assert (captured[0]["created_by"], captured[0]["updated_by"]) == (101, 101)


@pytest.mark.parametrize("user_id", (None, 0, -1, True, "101"))
async def test_telegram_rejects_invalid_actor_before_encryption_or_storage(user_id) -> None:
    principal = _principal().model_copy(update={"user_id": user_id})

    async def forbidden_repo():
        pytest.fail("invalid actor reached storage")

    def forbidden_encryption(_payload):
        pytest.fail("invalid actor reached encryption")

    with pytest.raises(HTTPException) as exc:
        await support.telegram_admin_put_bot_impl(
            principal=principal, payload=_payload(), get_org_secret_repo=forbidden_repo,
            encrypt_telegram_payload=forbidden_encryption,
        )
    assert (exc.value.status_code, exc.value.detail) == (
        403, "Not authorized to manage shared BYOK keys",
    )


async def test_telegram_rejects_missing_principal_before_storage() -> None:
    async def forbidden_repo():
        pytest.fail("missing principal reached storage")

    with pytest.raises(HTTPException) as exc:
        await support.telegram_admin_put_bot_impl(
            principal=None, payload=_payload(), get_org_secret_repo=forbidden_repo,
        )
    assert (exc.value.status_code, exc.value.detail) == (
        403, "Not authorized to manage shared BYOK keys",
    )


@pytest.mark.parametrize("scope_type", ("org", "team"))
@pytest.mark.parametrize("stage", ("acquire", "write"))
@pytest.mark.parametrize("failure_type,expected_status,expected_detail", (
    (MembershipAuthorizationError, 403, "Not authorized to manage shared BYOK keys"),
    (MembershipScopeNotFound, 404, "Shared BYOK scope not found"),
    (DatabaseLockError, 503, _BUSY_DETAIL),
    (DatabaseConcurrencyConflict, 503, _BUSY_DETAIL),
    (ConnectionPoolExhaustedError, 503, _BUSY_DETAIL),
    (TimeoutError, 503, _BUSY_DETAIL),
))
async def test_telegram_write_preserves_sanitized_repository_control_errors(
    monkeypatch: pytest.MonkeyPatch, scope_type: str, stage: str,
    failure_type: type[Exception], expected_status: int, expected_detail: str,
) -> None:
    failure = failure_type()
    failure.args = ("private-token at /private/telegram-auth.db",)

    class Repo:
        async def upsert_secret(self, **_kwargs):
            raise failure

    async def get_repo():
        if stage == "acquire":
            raise failure
        return Repo()

    for retry_after in (7, 19):
        monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS", str(retry_after))
        with pytest.raises(HTTPException) as exc:
            await support.telegram_admin_put_bot_impl(
                principal=_principal(scope_type), payload=_payload(),
                get_org_secret_repo=get_repo, encrypt_telegram_payload=lambda _payload: "sealed",
            )
        assert (exc.value.status_code, exc.value.detail) == (expected_status, expected_detail)
        assert exc.value.headers == ({"Retry-After": str(retry_after)} if expected_status == 503 else None)


async def test_telegram_missing_scope_keeps_existing_400_contract() -> None:
    principal = _principal().model_copy(update={"org_ids": [], "active_org_id": None})
    with pytest.raises(HTTPException) as exc:
        await support.telegram_admin_put_bot_impl(principal=principal, payload=_payload())
    assert (exc.value.status_code, exc.value.detail) == (400, "An active org/team scope is required")


async def test_telegram_blank_credentials_keep_existing_422_contract() -> None:
    with pytest.raises(HTTPException) as exc:
        await support.telegram_admin_put_bot_impl(
            principal=_principal(),
            payload=SimpleNamespace(bot_token=" ".join(("", "")), webhook_secret="\t".join(("", ""))),
        )
    assert (exc.value.status_code, exc.value.detail) == (422, "bot_token and webhook_secret are required")

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import HTTPException
import pytest

from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as user_handling


class _Headers(dict):
    def get(self, key: str, default=None):
        return super().get(key, default)


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(request_id="req-1"),
        client=SimpleNamespace(host="127.0.0.1"),
        headers=_Headers({"User-Agent": "pytest", "X-Request-ID": "req-1"}),
        scope={},
    )


def _patch_jwt_auth_dependencies(monkeypatch, payload_updates: dict):
    class JwtStub:
        def decode_access_token(self, _token: str):
            payload = {
                "sub": "42",
                "username": "target",
            }
            payload.update(payload_updates)
            return payload

    class RepoStub:
        async def get_user_by_id(self, user_id: int):
            assert user_id == 42  # nosec B101
            return {
                "id": 42,
                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                "username": "target",
                "email": "target@example.com",
                "role": "user",
                "is_active": True,
                "is_verified": True,
                "is_superuser": False,
            }

        async def get_user_by_uuid(self, _identifier: str):
            return None

        async def get_user_by_username(self, _username: str):
            return None

    async def _repo_from_pool():
        return RepoStub()

    monkeypatch.setattr(user_handling, "get_jwt_service", lambda: JwtStub())
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.repos.users_repo.AuthnzUsersRepo.from_pool",
        _repo_from_pool,
    )
    monkeypatch.setattr(
        user_handling,
        "get_session_manager",
        AsyncMock(return_value=SimpleNamespace(is_token_blacklisted=AsyncMock(return_value=False))),
    )
    monkeypatch.setattr(user_handling, "list_memberships_for_user", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        user_handling,
        "apply_scoped_permissions",
        AsyncMock(return_value=SimpleNamespace(permissions=[], active_org_id=None, active_team_id=None)),
    )
    monkeypatch.setattr(user_handling, "_enrich_user_with_rbac", lambda *_, **__: (["user"], [], False))
    monkeypatch.setattr(user_handling, "set_scope", lambda **_: None)


@pytest.mark.asyncio
async def test_impersonation_claims_populate_auth_context(monkeypatch):
    request = _request()
    _patch_jwt_auth_dependencies(
        monkeypatch,
        {
            "impersonation": True,
            "impersonated_by": 7,
        },
    )

    user = await user_handling.verify_jwt_and_fetch_user(request, token="token")  # nosec B106

    assert user.id == 42  # nosec B101
    assert request.state.impersonation is True  # nosec B101
    assert request.state.impersonated_by_user_id == 7  # nosec B101
    assert request.state.auth.principal.user_id == 42  # nosec B101
    assert request.state.auth.principal.impersonation is True  # nosec B101
    assert request.state.auth.principal.impersonated_by_user_id == 7  # nosec B101


@pytest.mark.asyncio
async def test_impersonated_by_digit_string_populates_auth_context(monkeypatch):
    request = _request()
    _patch_jwt_auth_dependencies(
        monkeypatch,
        {
            "impersonation": True,
            "impersonated_by": "7",
        },
    )

    await user_handling.verify_jwt_and_fetch_user(request, token="token")  # nosec B106

    assert request.state.impersonation is True  # nosec B101
    assert request.state.impersonated_by_user_id == 7  # nosec B101
    assert request.state.auth.principal.impersonated_by_user_id == 7  # nosec B101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload_updates",
    [
        {"impersonation": "false", "impersonated_by": 7},
        {"impersonation": True, "impersonated_by": True},
        {"impersonation": True, "impersonated_by": 7.9},
        {"impersonation": True, "impersonated_by": 0},
        {"impersonation": True, "impersonated_by": -1},
        {"impersonation": True},
        {"impersonation": False, "impersonated_by": None},
        {"impersonation": False, "impersonated_by": 7},
        {"impersonated_by": 7},
    ],
)
async def test_malformed_impersonation_claims_fail_closed(monkeypatch, payload_updates: dict):
    request = _request()
    _patch_jwt_auth_dependencies(monkeypatch, payload_updates)

    with pytest.raises(HTTPException) as exc_info:
        await user_handling.verify_jwt_and_fetch_user(request, token="token")  # nosec B106

    assert exc_info.value.status_code == 401  # nosec B101

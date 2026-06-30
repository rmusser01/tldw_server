import pytest
from fastapi import HTTPException
from starlette.requests import Request
from types import SimpleNamespace

from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as user_handling


def _build_request(client_ip: str = "127.0.0.1") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [],
        "client": (client_ip, 0),
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_verify_jwt_rejects_stale_token_org_team_claims(monkeypatch):
    payload = {
        "sub": "1",
        "org_ids": [999],
        "team_ids": [888],
        "active_org_id": 999,
        "active_team_id": 888,
    }

    class _StubJWT:
        def decode_access_token(self, _token: str):
            return payload

    class _StubUsersRepo:
        async def get_user_by_id(self, _user_id: int):
            return {
                "id": 1,
                "username": "tester",
                "email": "tester@example.com",
                "is_active": True,
                "role": "user",
            }

        async def get_user_by_uuid(self, _identifier: str):
            return None

        async def get_user_by_username(self, _username: str):
            return None

    async def _fake_from_pool():
        return _StubUsersRepo()

    async def _fake_list_memberships(_user_id: int):
        return [
            {"org_id": 1, "team_id": 10},
            {"org_id": 2, "team_id": 20},
        ]

    class _StubSessionManager:
        async def is_token_blacklisted(self, *_args, **_kwargs):
            return False

    async def _fake_get_session_manager():
        return _StubSessionManager()

    monkeypatch.setattr(user_handling, "get_jwt_service", lambda: _StubJWT())
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.repos.users_repo.AuthnzUsersRepo.from_pool",
        _fake_from_pool,
    )
    monkeypatch.setattr(user_handling, "list_memberships_for_user", _fake_list_memberships)
    monkeypatch.setattr(user_handling, "get_session_manager", _fake_get_session_manager)
    monkeypatch.setattr(
        user_handling,
        "_enrich_user_with_rbac",
        lambda *_args, **_kwargs: (["user"], [], False),
    )
    monkeypatch.setattr(user_handling, "set_scope", lambda *_, **__: None)

    request = _build_request()
    jwt_value = ".".join(("fake", "jwt", "token"))

    with pytest.raises(HTTPException) as exc_info:
        await user_handling.verify_jwt_and_fetch_user(request, token=jwt_value)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_verify_jwt_denies_when_membership_lookup_fails(monkeypatch):
    payload = {"sub": "1"}

    class _StubJWT:
        def decode_access_token(self, _token: str):
            return payload

    class _StubUsersRepo:
        async def get_user_by_id(self, _user_id: int):
            return {
                "id": 1,
                "username": "tester",
                "email": "tester@example.com",
                "is_active": True,
                "role": "user",
            }

        async def get_user_by_uuid(self, _identifier: str):
            return None

        async def get_user_by_username(self, _username: str):
            return None

    async def _fake_from_pool():
        return _StubUsersRepo()

    async def _failing_memberships(_user_id: int):
        raise RuntimeError("membership backend down")

    async def _unexpected_apply_scoped_permissions(**_kwargs):
        raise AssertionError("apply_scoped_permissions should not run after membership failure")

    class _StubSessionManager:
        async def is_token_blacklisted(self, *_args, **_kwargs):
            return False

    async def _fake_get_session_manager():
        return _StubSessionManager()

    monkeypatch.setattr(user_handling, "get_jwt_service", lambda: _StubJWT())
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.repos.users_repo.AuthnzUsersRepo.from_pool",
        _fake_from_pool,
    )
    monkeypatch.setattr(user_handling, "list_memberships_for_user", _failing_memberships)
    monkeypatch.setattr(user_handling, "apply_scoped_permissions", _unexpected_apply_scoped_permissions)
    monkeypatch.setattr(user_handling, "get_session_manager", _fake_get_session_manager)
    monkeypatch.setattr(user_handling, "_enrich_user_with_rbac", lambda *_args, **_kwargs: (["user"], [], False))
    monkeypatch.setattr(user_handling, "set_scope", lambda *_, **__: None)

    request = _build_request()

    with pytest.raises(HTTPException) as exc_info:
        await user_handling.verify_jwt_and_fetch_user(request, token="fake.jwt.token")

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Token membership could not be validated"


@pytest.mark.asyncio
async def test_verify_jwt_sanitizes_user_model_validation_failure(monkeypatch):
    payload = {"sub": "1"}

    class _StubJWT:
        def decode_access_token(self, _token: str):
            return payload

    class _StubUsersRepo:
        async def get_user_by_id(self, _user_id: int):
            return {
                "id": 1,
                "email": "tester@example.com",
                "is_active": True,
                "role": "user",
            }

        async def get_user_by_uuid(self, _identifier: str):
            return None

        async def get_user_by_username(self, _username: str):
            return None

    async def _fake_from_pool():
        return _StubUsersRepo()

    class _StubSessionManager:
        async def is_token_blacklisted(self, *_args, **_kwargs):
            return False

    async def _fake_get_session_manager():
        return _StubSessionManager()

    monkeypatch.setattr(user_handling, "get_jwt_service", lambda: _StubJWT())
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.repos.users_repo.AuthnzUsersRepo.from_pool",
        _fake_from_pool,
    )
    monkeypatch.setattr(user_handling, "get_session_manager", _fake_get_session_manager)
    monkeypatch.setattr(
        user_handling,
        "_enrich_user_with_rbac",
        lambda *_args, **_kwargs: (["user"], [], False),
    )

    request = _build_request()
    jwt_value = ".".join(("fake", "jwt", "token"))

    with pytest.raises(HTTPException) as exc_info:
        await user_handling.verify_jwt_and_fetch_user(request, token=jwt_value)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Error processing user data: Invalid format."
    assert "username" not in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_verify_jwt_accepts_valid_membership_claims(monkeypatch):
    payload = {
        "sub": "1",
        "org_ids": [1],
        "team_ids": [10],
        "active_org_id": 1,
        "active_team_id": 10,
    }

    class _StubJWT:
        def decode_access_token(self, _token: str):
            return payload

    class _StubUsersRepo:
        async def get_user_by_id(self, _user_id: int):
            return {
                "id": 1,
                "username": "tester",
                "email": "tester@example.com",
                "is_active": True,
                "role": "user",
            }

        async def get_user_by_uuid(self, _identifier: str):
            return None

        async def get_user_by_username(self, _username: str):
            return None

    async def _fake_from_pool():
        return _StubUsersRepo()

    async def _fake_list_memberships(_user_id: int):
        return [{"org_id": 1, "team_id": 10}]

    async def _fake_apply_scoped_permissions(**kwargs):
        return SimpleNamespace(
            permissions=list(kwargs.get("base_permissions") or []) + ["media.read"],
            active_org_id=1,
            active_team_id=10,
        )

    class _StubSessionManager:
        async def is_token_blacklisted(self, *_args, **_kwargs):
            return False

    async def _fake_get_session_manager():
        return _StubSessionManager()

    monkeypatch.setattr(user_handling, "get_jwt_service", lambda: _StubJWT())
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.repos.users_repo.AuthnzUsersRepo.from_pool",
        _fake_from_pool,
    )
    monkeypatch.setattr(user_handling, "list_memberships_for_user", _fake_list_memberships)
    monkeypatch.setattr(user_handling, "apply_scoped_permissions", _fake_apply_scoped_permissions)
    monkeypatch.setattr(user_handling, "get_session_manager", _fake_get_session_manager)
    monkeypatch.setattr(user_handling, "_enrich_user_with_rbac", lambda *_args, **_kwargs: (["user"], [], False))
    monkeypatch.setattr(user_handling, "set_scope", lambda *_, **__: None)

    request = _build_request()
    user = await user_handling.verify_jwt_and_fetch_user(request, token="fake.jwt.token")

    assert user.id == 1
    assert user.username == "tester"
    assert "media.read" in list(user.permissions or [])
    assert request.state.org_ids == [1]
    assert request.state.team_ids == [10]

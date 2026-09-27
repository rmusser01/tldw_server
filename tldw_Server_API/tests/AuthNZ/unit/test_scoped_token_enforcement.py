import asyncio
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import require_token_scope
from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidTokenError
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


class _StubSessionManager:
    async def is_token_blacklisted(self, _token: str, _jti=None) -> bool:
        return False


def _patch_jwt_user_resolution(monkeypatch) -> None:
    import tldw_Server_API.app.core.AuthNZ.User_DB_Handling as user_mod
    from tldw_Server_API.app.core.AuthNZ.repos import users_repo as users_repo_mod

    async def _fake_get_session_manager():
        return _StubSessionManager()

    class _StubUsersRepo:
        async def get_user_by_id(self, user_id: int):
            return {
                "id": user_id,
                "username": "tester",
                "email": "tester@example.com",
                "role": "user",
                "is_active": True,
                "is_verified": True,
            }

        async def get_user_by_uuid(self, _user_uuid: str):
            return None

        async def get_user_by_username(self, _username: str):
            return None

    @classmethod
    async def _fake_from_pool(cls):
        return _StubUsersRepo()

    async def _fake_list_memberships_for_user(_user_id: int):
        return []

    async def _fake_apply_scoped_permissions(**kwargs):
        return SimpleNamespace(
            permissions=list(kwargs.get("base_permissions") or []),
            active_org_id=None,
            active_team_id=None,
        )

    monkeypatch.setattr(user_mod, "get_session_manager", _fake_get_session_manager)
    monkeypatch.setattr(user_mod, "_enrich_user_with_rbac", lambda *_args, **_kwargs: (["user"], [], False))
    monkeypatch.setattr(user_mod, "list_memberships_for_user", _fake_list_memberships_for_user)
    monkeypatch.setattr(user_mod, "apply_scoped_permissions", _fake_apply_scoped_permissions)
    monkeypatch.setattr(users_repo_mod.AuthnzUsersRepo, "from_pool", _fake_from_pool)


def _build_app(with_scope_dep: bool) -> FastAPI:
    app = FastAPI()

    deps = []
    if with_scope_dep:
        deps.append(
            Depends(
                require_token_scope(
                    "workflows",
                    require_if_present=True,
                    endpoint_id="unit.scoped",
                )
            )
        )

    @app.get("/protected", dependencies=deps)
    async def protected(_user=Depends(get_request_user)):  # noqa: B008
        return {"ok": True}

    # Avoid touching the real AuthNZ database pool in unit tests.
    async def _fake_db_pool():
        return None

    app.dependency_overrides[auth_deps.get_db_pool] = _fake_db_pool
    return app


def _make_scoped_token(scope: str = "workflows", role: str = "user") -> str:


    svc = JWTService(get_settings())
    return svc.create_access_token(
        user_id=1,
        username="tester",
        role=role,
        additional_claims={"scope": scope},
    )


def test_scoped_token_requires_scope_dependency(monkeypatch):


    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_SECRET_KEY", "unit-test-secret-key-1234567890-abcdef")
    reset_settings()

    _patch_jwt_user_resolution(monkeypatch)

    token = _make_scoped_token()
    app = _build_app(with_scope_dep=False)
    client = TestClient(app)

    r = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403
    assert r.json().get("detail") == "Scoped token requires endpoint scope enforcement"


def test_scoped_token_allows_when_scope_dependency_present(monkeypatch):


    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_SECRET_KEY", "unit-test-secret-key-1234567890-abcdef")
    reset_settings()

    _patch_jwt_user_resolution(monkeypatch)

    token = _make_scoped_token()
    app = _build_app(with_scope_dep=True)
    client = TestClient(app)

    r = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_scoped_token_does_not_bypass_with_admin_claim(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_SECRET_KEY", "unit-test-secret-key-1234567890-abcdef")
    reset_settings()

    _patch_jwt_user_resolution(monkeypatch)

    token = _make_scoped_token(scope="notes", role="admin")
    app = _build_app(with_scope_dep=True)
    client = TestClient(app)

    r = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403
    assert "invalid token scope" in r.json().get("detail", "").lower()


def test_require_token_scope_enforces_bearer_api_key(monkeypatch):
    captured = {"record_usage": None}

    class _StubAPIKeyManager:
        async def validate_api_key(self, api_key: str, ip_address=None, record_usage=True):
            assert api_key == "tldw_test.key"
            captured["record_usage"] = record_usage
            return {
                "id": 1,
                "user_id": 42,
                "scope": "read",
                "llm_allowed_endpoints": ["unit.api_key"],
                "metadata": {
                    "allowed_methods": ["POST"],
                    "allowed_paths": ["/protected"],
                },
            }

    async def _fake_get_api_key_manager():
        return _StubAPIKeyManager()

    monkeypatch.setattr(auth_deps, "get_api_key_manager", _fake_get_api_key_manager)

    dep = require_token_scope(
        "any",
        require_if_present=True,
        endpoint_id="unit.api_key",
    )
    req = SimpleNamespace(
        method="GET",
        headers={},
        scope={"path": "/protected"},
        client=SimpleNamespace(host="127.0.0.1"),
    )
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="tldw_test.key")

    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        asyncio.run(dep(request=req, credentials=creds, jwt_service=object(), db_pool=object()))
    assert exc.value.status_code == 403
    assert "method not permitted" in exc.value.detail.lower()
    assert captured["record_usage"] is False


def test_require_token_scope_and_get_request_user_record_usage_once(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_SECRET_KEY", "unit-test-secret-key-1234567890-abcdef")
    reset_settings()

    usage_flags: list[bool] = []
    usage_increment_count = {"value": 0}
    usage_details_for_recorded_call = {"value": None}

    class _StubAPIKeyManager:
        async def validate_api_key(
            self,
            api_key: str,
            ip_address=None,
            record_usage=True,
            usage_details=None,
        ):
            assert api_key == "tldw_test.key"
            usage_flags.append(bool(record_usage))
            if record_usage:
                usage_increment_count["value"] += 1
                usage_details_for_recorded_call["value"] = usage_details
            return {
                "id": 7,
                "user_id": 42,
                "scope": "read",
                "org_id": 1,
                "team_id": None,
                "metadata": {},
                "llm_allowed_endpoints": ["unit.api_key.double_usage"],
            }

    async def _fake_get_api_key_manager():
        return _StubAPIKeyManager()

    monkeypatch.setattr(auth_deps, "get_api_key_manager", _fake_get_api_key_manager)

    import tldw_Server_API.app.core.AuthNZ.User_DB_Handling as user_mod
    from tldw_Server_API.app.core.AuthNZ.repos import users_repo as users_repo_mod

    monkeypatch.setattr(user_mod, "get_api_key_manager", _fake_get_api_key_manager)
    monkeypatch.setattr(user_mod, "_enrich_user_with_rbac", lambda *_args, **_kwargs: (["user"], ["media.read"], False))

    async def _fake_list_memberships_for_user(_user_id: int):
        return [{"org_id": 1, "team_id": None}]

    async def _fake_apply_scoped_permissions(**kwargs):
        return SimpleNamespace(
            permissions=list(kwargs.get("base_permissions") or []),
            active_org_id=1,
            active_team_id=None,
        )

    class _StubUsersRepo:
        async def get_user_by_id(self, user_id: int):
            return {
                "id": user_id,
                "username": "api-user",
                "email": "api-user@example.com",
                "role": "user",
                "is_active": True,
                "is_verified": True,
            }

    @classmethod
    async def _fake_from_pool(cls):
        return _StubUsersRepo()

    monkeypatch.setattr(user_mod, "list_memberships_for_user", _fake_list_memberships_for_user)
    monkeypatch.setattr(user_mod, "apply_scoped_permissions", _fake_apply_scoped_permissions)
    monkeypatch.setattr(users_repo_mod.AuthnzUsersRepo, "from_pool", _fake_from_pool)

    app = FastAPI()

    @app.get(
        "/protected",
        dependencies=[
            Depends(
                require_token_scope(
                    "any",
                    require_if_present=True,
                    endpoint_id="unit.api_key.double_usage",
                    count_as="voice_call",
                )
            )
        ],
    )
    async def protected(_user=Depends(get_request_user)):  # noqa: B008
        return {"ok": True}

    async def _fake_db_pool():
        return object()

    app.dependency_overrides[auth_deps.get_db_pool] = _fake_db_pool
    client = TestClient(app)

    response = client.get("/protected", headers={"Authorization": "Bearer tldw_test.key"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert usage_increment_count["value"] == 1
    assert usage_flags.count(False) == 1
    assert usage_flags.count(True) == 1
    assert usage_details_for_recorded_call["value"] == {
        "endpoint_id": "unit.api_key.double_usage",
        "action": "voice_call",
        "scope": "any",
        "path": "/protected",
        "method": "GET",
    }


def test_require_token_scope_rejects_invalid_jwt():
    class _BadJWTService:
        def decode_access_token(self, _token: str):
            raise InvalidTokenError("bad token")

    dep = require_token_scope(
        "workflows",
        require_if_present=True,
        endpoint_id="unit.invalid_jwt",
    )
    req = SimpleNamespace(
        method="GET",
        headers={},
        scope={"path": "/protected"},
        path_params={},
        client=SimpleNamespace(host="127.0.0.1"),
        state=SimpleNamespace(),
    )
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid.token.here")

    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        asyncio.run(dep(request=req, credentials=creds, jwt_service=_BadJWTService(), db_pool=object()))
    assert exc.value.status_code == 401


def test_require_token_scope_fails_closed_when_credentials_missing():
    dep = require_token_scope(
        "workflows",
        require_if_present=True,
        endpoint_id="unit.missing_creds",
    )
    req = SimpleNamespace(
        method="GET",
        headers={},
        scope={"path": "/protected"},
        path_params={},
        client=SimpleNamespace(host="127.0.0.1"),
        state=SimpleNamespace(),
    )

    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        asyncio.run(dep(request=req, credentials=None, jwt_service=object(), db_pool=object()))
    assert exc.value.status_code == 401
    assert exc.value.detail == "Authentication required"


def test_require_token_scope_allows_missing_credentials_when_optional():
    dep = require_token_scope(
        "workflows",
        require_if_present=False,
        endpoint_id="unit.optional_scope",
    )
    req = SimpleNamespace(
        method="GET",
        headers={},
        scope={"path": "/protected"},
        path_params={},
        client=SimpleNamespace(host="127.0.0.1"),
        state=SimpleNamespace(),
    )

    # Optional mode should not force credentials.
    asyncio.run(dep(request=req, credentials=None, jwt_service=object(), db_pool=object()))


def test_require_token_scope_allows_missing_credentials_with_request_user_override_in_tests(
    monkeypatch,
):
    monkeypatch.setenv("TEST_MODE", "1")
    dep = require_token_scope(
        "workflows",
        require_if_present=True,
        endpoint_id="unit.test_override_scope",
    )
    req = SimpleNamespace(
        method="GET",
        headers={},
        scope={"path": "/protected"},
        path_params={},
        client=SimpleNamespace(host="127.0.0.1"),
        state=SimpleNamespace(),
        app=SimpleNamespace(
            dependency_overrides={get_request_user: lambda: SimpleNamespace(id=1)}
        ),
    )

    # Test-mode compatibility: legacy get_request_user overrides should satisfy
    # scoped routes that otherwise fail closed on missing credentials.
    asyncio.run(dep(request=req, credentials=None, jwt_service=object(), db_pool=object()))


def test_require_token_scope_fails_closed_for_invalid_api_key(monkeypatch):
    class _StubAPIKeyManager:
        async def validate_api_key(self, api_key: str, ip_address=None, record_usage=True):
            assert api_key == "tldw_invalid.key"
            assert record_usage is False
            return None

    async def _fake_get_api_key_manager():
        return _StubAPIKeyManager()

    monkeypatch.setattr(auth_deps, "get_api_key_manager", _fake_get_api_key_manager)

    dep = require_token_scope(
        "any",
        require_if_present=True,
        endpoint_id="unit.invalid_api_key",
        allow_admin_bypass=False,
    )
    req = SimpleNamespace(
        method="GET",
        headers={},
        scope={"path": "/protected"},
        url=SimpleNamespace(path="/protected"),
        path_params={},
        client=SimpleNamespace(host="127.0.0.1"),
        state=SimpleNamespace(),
    )
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="tldw_invalid.key")

    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        asyncio.run(dep(request=req, credentials=creds, jwt_service=object(), db_pool=object()))
    assert exc.value.status_code == 401
    assert exc.value.detail == "Could not validate credentials"


def test_require_token_scope_fails_closed_on_api_key_metadata_parse_error(monkeypatch):
    class _StubAPIKeyManager:
        async def validate_api_key(self, api_key: str, ip_address=None, record_usage=True):
            assert api_key == "tldw_parse.key"
            assert record_usage is False
            return {
                "id": 123,
                "user_id": 99,
                "scope": "read",
                "llm_allowed_endpoints": ["unit.scope.parse_error"],
                "metadata": '{"allowed_methods":["GET"]',  # malformed JSON
            }

    async def _fake_get_api_key_manager():
        return _StubAPIKeyManager()

    monkeypatch.setattr(auth_deps, "get_api_key_manager", _fake_get_api_key_manager)

    dep = require_token_scope(
        "any",
        require_if_present=True,
        endpoint_id="unit.scope.parse_error",
        allow_admin_bypass=False,
    )
    req = SimpleNamespace(
        method="GET",
        headers={},
        url=SimpleNamespace(path="/protected"),
        scope={"path": "/protected"},
        client=SimpleNamespace(host="127.0.0.1"),
        state=SimpleNamespace(),
    )
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="tldw_parse.key")

    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        asyncio.run(dep(request=req, credentials=creds, jwt_service=object(), db_pool=object()))
    assert exc.value.status_code == 403
    assert "invalid API key metadata constraints" in exc.value.detail


def test_require_token_scope_fails_closed_on_api_key_backend_error(monkeypatch):
    class _StubAPIKeyManager:
        async def validate_api_key(self, api_key: str, ip_address=None, record_usage=True):
            assert api_key == "tldw_backend.key"
            assert record_usage is False
            raise RuntimeError("backend outage secret-details")

    async def _fake_get_api_key_manager():
        return _StubAPIKeyManager()

    monkeypatch.setattr(auth_deps, "get_api_key_manager", _fake_get_api_key_manager)

    dep = require_token_scope(
        "any",
        require_if_present=True,
        endpoint_id="unit.scope.backend_error",
        allow_admin_bypass=False,
    )
    req = SimpleNamespace(
        method="GET",
        headers={},
        url=SimpleNamespace(path="/protected"),
        scope={"path": "/protected"},
        client=SimpleNamespace(host="127.0.0.1"),
        state=SimpleNamespace(),
    )
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="tldw_backend.key")

    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        asyncio.run(dep(request=req, credentials=creds, jwt_service=object(), db_pool=object()))
    assert exc.value.status_code == 403
    assert exc.value.detail == "Forbidden: unable to validate API key constraints"


@pytest.mark.parametrize(
    "method,constraints,detail",
    [
        ("POST", {}, "lacks required scope"),
        ("GET", {"llm_allowed_endpoints": ["other.endpoint"]}, "endpoint not permitted"),
        ("GET", {"metadata": {"allowed_methods": ["POST"]}}, "method not permitted"),
        ("GET", {"metadata": {"allowed_paths": ["/other"]}}, "path not permitted"),
        ("GET", {"metadata": {"max_calls": 0}}, "quota exceeded"),
    ],
)
def test_header_api_key_keeps_constraints_for_admin_owner(monkeypatch, method, constraints, detail):
    """Resolving a cookie must not widen a restricted X-API-KEY's authority."""
    from fastapi import HTTPException

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    reset_settings()

    class APIKeyManager:
        async def validate_api_key(self, api_key, ip_address=None, record_usage=True):
            return {"id": 99, "user_id": 42, "scope": "read", **constraints}

    async def key_manager():
        return APIKeyManager()

    async def admin_owner(_request):
        return SimpleNamespace(roles=["admin"], permissions=[], is_admin=True)

    async def exhausted_quota(*args, **kwargs):
        raise HTTPException(status_code=403, detail="Forbidden: API key quota exceeded")

    monkeypatch.setattr(auth_deps, "get_api_key_manager", key_manager)
    monkeypatch.setattr(auth_deps, "_consume_or_defer_token_quota", exhausted_quota)
    request = SimpleNamespace(
        method=method,
        headers={"X-API-KEY": "tldw_restricted.key"},
        scope={"path": "/protected"},
        url=SimpleNamespace(path="/protected"),
        client=SimpleNamespace(host="127.0.0.1"),
        state=SimpleNamespace(),
        app=SimpleNamespace(dependency_overrides={auth_deps.get_auth_principal: admin_owner}),
    )
    guard = require_token_scope("any", endpoint_id="unit.restricted", count_as="run")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(guard(request=request, credentials=None, jwt_service=object(), db_pool=object()))

    assert exc.value.status_code == 403
    assert detail in exc.value.detail

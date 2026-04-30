from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import text2sql as text2sql_mod
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


def _make_principal(*, permissions: list[str], roles: list[str] | None = None) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject="single_user",
        token_type="single_user",
        jti=None,
        roles=roles or ["user"],
        permissions=permissions,
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )


def _build_test_app(*, principal: AuthPrincipal, user: User, db_path: Path) -> FastAPI:
    app = FastAPI()
    app.include_router(text2sql_mod.router, prefix="/api/v1", tags=["text2sql"])

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        request.state.auth = AuthContext(
            principal=principal,
            ip=None,
            user_agent=None,
            request_id=None,
        )
        return principal

    async def _fake_get_request_user() -> User:
        return user

    async def _fake_get_media_db_for_user():
        return SimpleNamespace(db_path=str(db_path))

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    app.dependency_overrides[text2sql_mod.get_request_user] = _fake_get_request_user
    app.dependency_overrides[text2sql_mod.get_media_db_for_user] = _fake_get_media_db_for_user
    return app


class _FakeText2SQLCoreService:
    def __init__(self, *args, **kwargs):
        _ = args
        _ = kwargs

    async def generate_and_execute(self, **kwargs):
        _ = kwargs
        return {
            "sql": "SELECT 1 AS n",
            "columns": ["n"],
            "rows": [{"n": 1}],
            "row_count": 1,
            "duration_ms": 1,
            "target_id": "media_db",
            "guardrail": {"limit_injected": False, "limit_clamped": False},
            "truncated": False,
        }


class _ExplodingText2SQLCoreService:
    def __init__(self, *args, **kwargs):
        _ = args
        _ = kwargs

    async def generate_and_execute(self, **kwargs):
        _ = kwargs
        raise RuntimeError("text2sql backend exploded at /private/db/path")


@pytest.mark.security
def test_text2sql_requires_sql_read_permission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(text2sql_mod, "Text2SQLCoreService", _FakeText2SQLCoreService)

    app = _build_test_app(
        principal=_make_principal(permissions=["media.read"]),
        user=User(
            id=1,
            username="test-user",
            email=None,
            is_active=True,
            roles=["user"],
            permissions=["media.read"],
            is_admin=False,
        ),
        db_path=tmp_path / "media.db",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/text2sql/query",
            json={"query": "SELECT 1 AS n", "target_id": "media_db"},
        )

    assert response.status_code == 403
    assert "sql.read" in str(response.json().get("detail", ""))


@pytest.mark.security
def test_text2sql_requires_explicit_target_acl_permission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(text2sql_mod, "Text2SQLCoreService", _FakeText2SQLCoreService)

    app = _build_test_app(
        principal=_make_principal(permissions=["sql.read", "media.read"]),
        user=User(
            id=1,
            username="test-user",
            email=None,
            is_active=True,
            roles=["user"],
            permissions=["sql.read", "media.read"],
            is_admin=False,
        ),
        db_path=tmp_path / "media.db",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/text2sql/query",
            json={"query": "SELECT 1 AS n", "target_id": "media_db"},
        )

    assert response.status_code == 403
    detail = response.json().get("detail")
    if isinstance(detail, dict):
        assert detail.get("code") == "unauthorized_target"
    else:
        assert "unauthorized_target" in str(detail)


@pytest.mark.security
def test_text2sql_enforces_connector_acl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(text2sql_mod, "Text2SQLCoreService", _FakeText2SQLCoreService)
    monkeypatch.setattr(
        text2sql_mod,
        "_connector_acl_allows",
        lambda current_user, target_id: False,
        raising=False,
    )

    app = _build_test_app(
        principal=_make_principal(permissions=["sql.read", "media.read", "sql.target:media_db"]),
        user=User(
            id=1,
            username="test-user",
            email=None,
            is_active=True,
            roles=["user"],
            permissions=["sql.read", "media.read", "sql.target:media_db"],
            is_admin=False,
        ),
        db_path=tmp_path / "media.db",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/text2sql/query",
            json={"query": "SELECT 1 AS n", "target_id": "media_db"},
        )

    assert response.status_code == 403
    detail = response.json().get("detail")
    if isinstance(detail, dict):
        assert detail.get("code") == "unauthorized_target"
    else:
        assert "unauthorized_target" in str(detail)


@pytest.mark.security
def test_text2sql_allows_explicit_target_acl_permission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(text2sql_mod, "Text2SQLCoreService", _FakeText2SQLCoreService)

    app = _build_test_app(
        principal=_make_principal(permissions=["sql.read", "sql.target:media_db"]),
        user=User(
            id=1,
            username="test-user",
            email=None,
            is_active=True,
            roles=["user"],
            permissions=["sql.read", "sql.target:media_db"],
            is_admin=False,
        ),
        db_path=tmp_path / "media.db",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/text2sql/query",
            json={"query": "SELECT 1 AS n", "target_id": "media_db"},
        )

        assert response.status_code == 200


@pytest.mark.security
def test_text2sql_sanitizes_execution_backend_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(text2sql_mod, "Text2SQLCoreService", _ExplodingText2SQLCoreService)

    app = _build_test_app(
        principal=_make_principal(permissions=["sql.read", "sql.target:media_db"]),
        user=User(
            id=1,
            username="test-user",
            email=None,
            is_active=True,
            roles=["user"],
            permissions=["sql.read", "sql.target:media_db"],
            is_admin=False,
        ),
        db_path=tmp_path / "media.db",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/text2sql/query",
            json={"query": "SELECT 1 AS n", "target_id": "media_db"},
        )

    assert response.status_code == 500
    assert response.json().get("detail") == {
        "code": "sql_execution_failed",
        "message": "SQL execution failed",
    }

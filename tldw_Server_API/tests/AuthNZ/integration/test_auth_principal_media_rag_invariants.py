from typing import Any, Dict, Tuple

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext


pytestmark = pytest.mark.integration


def _install_auth_capture(app: FastAPI) -> Tuple[Dict[str, Any], Any]:
    """
    Install a lightweight wrapper around get_auth_principal that records
    principal/state alignment for the last request.
    """
    captured: Dict[str, Any] = {}
    original_get_auth_principal = auth_deps.get_auth_principal

    async def _capturing_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        principal = await original_get_auth_principal(request)

        captured["principal"] = {
            "principal_id": principal.principal_id,
            "kind": principal.kind,
            "user_id": principal.user_id,
            "api_key_id": principal.api_key_id,
            "roles": list(principal.roles),
            "permissions": list(principal.permissions),
            "org_ids": list(principal.org_ids),
            "team_ids": list(principal.team_ids),
        }
        captured["state"] = {
            "user_id": getattr(request.state, "user_id", None),
            "api_key_id": getattr(request.state, "api_key_id", None),
            "org_ids": getattr(request.state, "org_ids", None),
            "team_ids": getattr(request.state, "team_ids", None),
        }

        ctx = getattr(request.state, "auth", None)
        if isinstance(ctx, AuthContext):
            cp = ctx.principal
            captured["state_auth_principal"] = {
                "principal_id": cp.principal_id,
                "kind": cp.kind,
                "user_id": cp.user_id,
                "api_key_id": cp.api_key_id,
                "roles": list(cp.roles),
                "permissions": list(cp.permissions),
                "org_ids": list(cp.org_ids),
                "team_ids": list(cp.team_ids),
            }
        else:
            captured["state_auth_principal"] = None

        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _capturing_get_auth_principal
    return captured, original_get_auth_principal


def _restore_auth_capture(app: FastAPI, original_get_auth_principal: Any) -> None:
    """Remove the auth capture wrapper from dependency overrides."""
    app.dependency_overrides.pop(auth_deps.get_auth_principal, None)


def _run_async(coro):


    """Run an async coroutine from sync tests, tolerating an active loop."""
    import asyncio
    import threading

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Any] = {}

    def _runner():

        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - propagated below
            result["error"] = exc

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


async def _grant_user_permission(db_name: str, username: str, permission: str) -> int:
    """Ensure a user has a concrete permission via user_permissions."""
    import asyncpg

    from tldw_Server_API.tests.helpers.pg_env import get_pg_env

    env = get_pg_env()
    conn = await asyncpg.connect(
        host=env.host,
        port=env.port,
        user=env.user,
        password=env.password,
        database=db_name,
    )
    try:
        user_id = await conn.fetchval("SELECT id FROM users WHERE username=$1", username)
        if user_id is None:
            raise RuntimeError(f"User {username} not found")
        perm_id = await conn.fetchval(
            """
            INSERT INTO permissions (name, description, category)
            VALUES ($1, $2, $3)
            ON CONFLICT (name) DO UPDATE SET description=EXCLUDED.description
            RETURNING id
            """,
            permission,
            permission,
            permission.split(".")[0] if "." in permission else "general",
        )
        await conn.execute(
            """
            INSERT INTO user_permissions (user_id, permission_id, granted)
            VALUES ($1, $2, TRUE)
            ON CONFLICT (user_id, permission_id) DO UPDATE SET granted=EXCLUDED.granted
            """,
            user_id,
            perm_id,
        )
        return int(user_id)
    finally:
        await conn.close()


async def _grant_admin_role(db_name: str, username: str) -> int:
    """Ensure the given user has the 'admin' role in Postgres-backed RBAC tables."""
    import asyncpg

    from tldw_Server_API.tests.helpers.pg_env import get_pg_env

    env = get_pg_env()
    conn = await asyncpg.connect(
        host=env.host,
        port=env.port,
        user=env.user,
        password=env.password,
        database=db_name,
    )
    try:
        user_id = await conn.fetchval("SELECT id FROM users WHERE username=$1", username)
        if user_id is None:
            raise RuntimeError(f"User {username} not found")

        # Ensure admin role exists
        await conn.execute(
            """
            INSERT INTO roles (name, description, is_system)
            VALUES ('admin', 'Administrator', TRUE)
            ON CONFLICT (name) DO NOTHING
            """
        )
        role_id = await conn.fetchval("SELECT id FROM roles WHERE name = 'admin'")
        if role_id is None:
            raise RuntimeError("admin role not found after insert")

        # Attach admin role to user and update legacy role column for fallback paths
        await conn.execute(
            """
            INSERT INTO user_roles (user_id, role_id)
            VALUES ($1, $2)
            ON CONFLICT (user_id, role_id) DO NOTHING
            """,
            user_id,
            role_id,
        )
        await conn.execute(
            "UPDATE users SET role = 'admin' WHERE id = $1",
            user_id,
        )

        return int(user_id)
    finally:
        await conn.close()


async def _create_api_key(db_name: str, username: str) -> Dict[str, Any]:
    """Create a real API key for the given user via the manager."""
    import asyncpg

    from tldw_Server_API.tests.helpers.pg_env import get_pg_env

    env = get_pg_env()
    conn = await asyncpg.connect(
        host=env.host,
        port=env.port,
        user=env.user,
        password=env.password,
        database=db_name,
    )
    try:
        user_id = await conn.fetchval("SELECT id FROM users WHERE username=$1", username)
        if user_id is None:
            raise RuntimeError(f"User {username} not found")
    finally:
        await conn.close()

    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    mgr = APIKeyManager()
    await mgr.initialize()
    return await mgr.create_api_key(
        user_id=user_id,
        name="principal-invariants-key",
        description="auth_principal_media_rag_invariants",
        scope="write",
        expires_in_days=30,
    )


class _StubRagResult:
    """Lightweight stub emulating RAG pipeline results for auth-focused tests."""

    def __init__(self, query: str):
        self.documents = []
        self.query = query
        self.expanded_queries = []
        self.metadata = {}
        self.timings = {}
        self.citations = []
        self.academic_citations = []
        self.total_time = 0.01
        self.cache_hit = False
        self.errors: list[str] = []
        self.feedback_id = None
        self.generated_answer = "ok"
        self.security_report = None
        self.claims = None
        self.factuality = None


def test_rag_search_jwt_principal_and_state_alignment(isolated_test_environment, monkeypatch):


    """
    Multi-user JWT happy path for a representative RAG route:

    - Register and log in a user.
    - Grant media.read permission.
    - Call /api/v1/rag/search and assert that request.state.* and
      request.state.auth.principal stay aligned with AuthPrincipal.
    """
    client, db_name = isolated_test_environment
    assert isinstance(client, TestClient)

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    settings = get_settings()
    assert settings.AUTH_MODE == "multi_user"

    # Lightweight stubs for RAG dependencies to keep the test focused on auth.
    from tldw_Server_API.app.main import app as fastapi_app
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.api.v1.endpoints import rag_unified as rag_mod

    fastapi_app.dependency_overrides[get_media_db_for_user] = lambda: type("DB", (), {"db_path": ":memory:"})()
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: type("DB", (), {"db_path": ":memory:"})()

    async def _fake_pipeline(**kwargs):
        return _StubRagResult(kwargs.get("query", ""))

    monkeypatch.setattr(rag_mod, "unified_rag_pipeline", _fake_pipeline)

    try:
        # 1. Register and log in via real auth endpoints.
        username = "rag_invariants_user"
        password = "Str0ngP@ssw0rd!"
        reg = client.post(
            "/api/v1/auth/register",
            json={"username": username, "email": "rag_invariants_user@example.com", "password": password},
        )
        assert reg.status_code == 200, reg.text

        login = client.post("/api/v1/auth/login", data={"username": username, "password": password})
        assert login.status_code == 200, login.text
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Grant media.read permission so the RAG search passes require_permissions.
        _run_async(_grant_user_permission(db_name, username, "media.read"))

        # 3. Install the auth capture wrapper and call /api/v1/rag/search.
        app = fastapi_app
        captured, original = _install_auth_capture(app)
        try:
            body = {"query": "hello world"}
            resp = client.post("/api/v1/rag/search", headers=headers, json=body)
            assert resp.status_code == 200, resp.text

            principal = captured.get("principal")
            state = captured.get("state")
            state_auth_principal = captured.get("state_auth_principal")

            assert principal is not None
            assert state is not None
            assert state_auth_principal is not None

            # AuthPrincipal kind/user identity
            assert principal["kind"] == "user"
            assert principal["user_id"] is not None

            # request.state mirrors principal identity (user_id, no api_key_id)
            assert str(state["user_id"]) == str(principal["user_id"])
            assert state["api_key_id"] is None

            # request.state.auth.principal mirrors principal and state
            assert state_auth_principal["kind"] == principal["kind"]
            assert str(state_auth_principal["user_id"]) == str(principal["user_id"])
            assert state_auth_principal["api_key_id"] == principal["api_key_id"]
            assert state_auth_principal["org_ids"] == principal["org_ids"]
            assert state_auth_principal["team_ids"] == principal["team_ids"]
            assert state["org_ids"] == principal["org_ids"]
            assert state["team_ids"] == principal["team_ids"]
        finally:
            _restore_auth_capture(app, original)
    finally:
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)
        fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)


def test_media_process_videos_api_key_principal_and_state_alignment(isolated_test_environment):


    """
    Multi-user API-key happy path for a representative media route:

    - Register a user and create a real API key.
    - Grant media.create permission.
    - Call /api/v1/media/process-videos and assert that request.state.* and
      request.state.auth.principal stay aligned with AuthPrincipal, including
      api_key_id.
    """
    client, db_name = isolated_test_environment
    assert isinstance(client, TestClient)

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    settings = get_settings()
    assert settings.AUTH_MODE == "multi_user"

    # 1. Register a user via the real auth endpoint.
    username = "media_invariants_user"
    password = "Str0ngP@ssw0rd!"
    reg = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": "media_invariants_user@example.com", "password": password},
    )
    assert reg.status_code == 200, reg.text

    # 2. Create an API key for this user and grant media.create.
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    app_pool = client.portal.call(get_db_pool)
    api_key_info = client.portal.call(_create_api_key, db_name, username)
    assert client.portal.call(get_db_pool) is app_pool
    api_key = api_key_info["key"]

    _run_async(_grant_user_permission(db_name, username, "media.create"))

    # 3. Install auth capture and call /api/v1/media/process-videos.
    from tldw_Server_API.app.main import app as fastapi_app

    app = fastapi_app
    captured, original = _install_auth_capture(app)
    try:
        resp = client.post(
            "/api/v1/media/process-videos",
            headers={"X-API-KEY": api_key},
            data={"urls": ""},
        )
        # The endpoint may return validation errors; we only require that auth succeeded.
        assert resp.status_code not in (401, 403), resp.text

        principal = captured.get("principal")
        state = captured.get("state")
        state_auth_principal = captured.get("state_auth_principal")

        assert principal is not None
        assert state is not None
        assert state_auth_principal is not None

        # AuthPrincipal reflects API-key based identity.
        assert principal["kind"] == "api_key"
        assert principal["user_id"] is not None
        assert principal["api_key_id"] is not None

        # request.state mirrors principal identity (user_id and api_key_id).
        assert str(state["user_id"]) == str(principal["user_id"])
        assert state["api_key_id"] is not None
        assert str(state["api_key_id"]) == str(principal["api_key_id"])

        # request.state.auth.principal mirrors both principal and state.
        assert state_auth_principal["kind"] == principal["kind"]
        assert str(state_auth_principal["user_id"]) == str(principal["user_id"])
        assert str(state_auth_principal["api_key_id"]) == str(principal["api_key_id"])
        assert str(state_auth_principal["api_key_id"]) == str(state["api_key_id"])
    finally:
        _restore_auth_capture(app, original)

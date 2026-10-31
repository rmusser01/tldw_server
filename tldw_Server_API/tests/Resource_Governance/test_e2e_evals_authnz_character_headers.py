import contextlib
import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.rate_limit


def _reset_rg_state(app):


    """
    Ensure each test starts with a fresh ResourceGovernor / policy loader.

    Tests in this module mutate RG_POLICY_PATH and related envs; reusing the
    same FastAPI app instance without resetting RG state can cause cross-test
    rate-limit bleed (unexpected 429s). This helper clears governor-related
    attributes so middleware will lazily reinitialize from the current env.
    """
    for attr in ("rg_governor", "rg_policy_loader", "rg_policy_store", "rg_policy_version", "rg_policy_count"):
        try:
            if hasattr(app.state, attr):
                setattr(app.state, attr, None)
        except Exception:
            continue


@pytest.fixture(params=["memory", "redis"], ids=["rg-memory", "rg-redis"])
def rg_backend(request) -> str:
    """Exercise deny-header behavior under both RG backends."""
    return str(request.param)


@contextlib.contextmanager
def _with_rg_middleware(app):
    """Temporarily install RGSimpleMiddleware for tests that set RG_ENABLED after app import."""
    try:
        from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware
        from starlette.middleware import Middleware
    except Exception:
        yield
        return

    original_user_middleware = getattr(app, "user_middleware", [])[:]
    changed = False
    try:
        already = any(getattr(m, "cls", None) is RGSimpleMiddleware for m in original_user_middleware)
        if not already:
            app.user_middleware = [Middleware(RGSimpleMiddleware), *original_user_middleware]
            changed = True
            try:
                app.middleware_stack = app.build_middleware_stack()
            except Exception:
                _ = None
        yield
    finally:
        if changed:
            try:
                app.user_middleware = original_user_middleware
                app.middleware_stack = app.build_middleware_stack()
            except Exception:
                _ = None


async def _init_authnz_sqlite(db_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        reset_settings()
    except Exception:
        _ = None
    try:
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_authnz_schema_ready_once

        await ensure_authnz_schema_ready_once()
    except Exception:
        _ = None


async def _create_user_and_key(*, username: str, email: str) -> tuple[int, str]:
    from uuid import uuid4

    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    pool = await get_db_pool()
    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username=username,
        email=email,
        password_hash="x",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid4(),
    )
    user_id = int(created_user["id"])
    mgr = APIKeyManager(pool)
    await mgr.initialize()
    key_rec = await mgr.create_api_key(user_id=user_id, name=f"{username}-key")
    return user_id, str(key_rec["key"])


@pytest.mark.asyncio
async def test_e2e_evaluations_deny_headers_retry_after(monkeypatch, tmp_path, rg_backend, fresh_services):
    db_path = tmp_path / "authnz_evals_e2e.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    _user_id, api_key = await _create_user_and_key(username="evals-user", email="evals-user@example.com")

    # Minimal app with RG middleware; enforce requests-only deny semantics for a
    # representative Evaluations endpoint via route_map.
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_STORE", "file")

    policy_id = f"evals.small.{rg_backend}.{tmp_path.name.replace('-', '_')}"
    policy = (
        "schema_version: 1\n"
        "policies:\n"
        f"  {policy_id}:\n"
        "    requests: { rpm: 1 }\n"
        "    scopes: [user, api_key]\n"
        "route_map:\n"
        "  by_path:\n"
        f"    /api/v1/evaluations/rate-limits: {policy_id}\n"
    )
    p = tmp_path / "rg_evals.yaml"
    p.write_text(policy, encoding="utf-8")

    monkeypatch.setenv("RG_POLICY_PATH", str(p))
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")

    from tldw_Server_API.app.main import app

    _reset_rg_state(app)

    with _with_rg_middleware(app):
        with TestClient(app) as client:
            # First request should not be rate-limited by RG; allow non-429 errors
            # from downstream Evaluations plumbing as long as RG does not deny.
            r1 = client.get(
                "/api/v1/evaluations/rate-limits",
                headers={"X-API-KEY": api_key},
            )
            assert r1.status_code != 429

            # Second request should be governed by RG via route_map.
            r2 = client.get(
                "/api/v1/evaluations/rate-limits",
                headers={"X-API-KEY": api_key},
            )

    assert r2.status_code in (429, 503)
    if r2.status_code == 429:
        retry_after = r2.headers.get("Retry-After")
        assert retry_after is not None
        assert r2.headers.get("X-RateLimit-Limit") == "1"
        assert r2.headers.get("X-RateLimit-Remaining") == "0"
        reset = r2.headers.get("X-RateLimit-Reset")
        assert reset is not None and int(reset) >= 1


@pytest.mark.asyncio
async def test_e2e_authnz_debug_deny_headers_retry_after(monkeypatch, tmp_path, rg_backend):
    # Minimal app with RG middleware; enforce requests-only deny semantics for a
    # lightweight AuthNZ debug endpoint that does not require full login flow.
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_STORE", "file")

    policy_id = f"authnz.small.{rg_backend}.{tmp_path.name.replace('-', '_')}"
    policy = (
        "schema_version: 1\n"
        "policies:\n"
        f"  {policy_id}:\n"
        "    requests: { rpm: 1 }\n"
        "    scopes: [ip]\n"
        "route_map:\n"
        "  by_path:\n"
        f"    /api/v1/authnz/debug/api-key-id: {policy_id}\n"
    )
    p = tmp_path / "rg_authnz.yaml"
    p.write_text(policy, encoding="utf-8")

    monkeypatch.setenv("RG_POLICY_PATH", str(p))
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")

    monkeypatch.setenv("RG_ENABLED", "1")

    # Stub out heavy AuthNZ budget helpers so the debug endpoint does not hit DBs.
    import tldw_Server_API.app.api.v1.endpoints.authnz_debug as authnz_debug_ep
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

    async def _fake_resolve_api_key(api_key: str):
        _ = api_key
        return None

    async def _fake_get_key_limits(key_id: int):
        _ = key_id
        return {}

    async def _fake_summarize_usage_for_key_day(key_id: int):
        _ = key_id
        return {}

    async def _fake_summarize_usage_for_key_month(key_id: int):
        _ = key_id
        return {}

    async def _fake_is_key_over_budget(key_id: int):
        _ = key_id
        return {"over": False, "reasons": []}

    monkeypatch.setattr(authnz_debug_ep, "resolve_api_key_by_hash", _fake_resolve_api_key)
    monkeypatch.setattr(authnz_debug_ep, "get_key_limits", _fake_get_key_limits)
    monkeypatch.setattr(authnz_debug_ep, "summarize_usage_for_key_day", _fake_summarize_usage_for_key_day)
    monkeypatch.setattr(authnz_debug_ep, "summarize_usage_for_key_month", _fake_summarize_usage_for_key_month)
    monkeypatch.setattr(authnz_debug_ep, "is_key_over_budget", _fake_is_key_over_budget)

    from tldw_Server_API.app.main import app

    _reset_rg_state(app)

    async def _fake_get_auth_principal(_request):
        return AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject=None,
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=[],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal

    try:
        with _with_rg_middleware(app):
            with TestClient(app) as client:
                r1 = client.get("/api/v1/authnz/debug/api-key-id")
                assert r1.status_code != 429

                r2 = client.get("/api/v1/authnz/debug/api-key-id")
    finally:
        app.dependency_overrides.pop(auth_deps.get_auth_principal, None)

    assert r2.status_code in (429, 503)
    if r2.status_code == 429:
        retry_after = r2.headers.get("Retry-After")
        assert retry_after is not None
        assert r2.headers.get("X-RateLimit-Limit") == "1"
        assert r2.headers.get("X-RateLimit-Remaining") == "0"
        reset = r2.headers.get("X-RateLimit-Reset")
        assert reset is not None and int(reset) >= 1


@pytest.mark.asyncio
async def test_e2e_character_chat_deny_headers_retry_after(monkeypatch, tmp_path, rg_backend, fresh_services):
    db_path = tmp_path / "authnz_character_e2e.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    user_id, api_key = await _create_user_and_key(username="character-user", email="character-user@example.com")

    # Minimal app with RG middleware; enforce requests-only deny semantics for
    # the legacy character chat completion endpoint used in tests.
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_STORE", "file")

    policy_id = f"character.small.{rg_backend}.{tmp_path.name.replace('-', '_')}"
    policy = (
        "schema_version: 1\n"
        "policies:\n"
        f"  {policy_id}:\n"
        "    requests: { rpm: 1 }\n"
        "    scopes: [user, api_key]\n"
        "route_map:\n"
        "  by_path:\n"
        f"    /api/v1/chats/*: {policy_id}\n"
    )
    p = tmp_path / "rg_character.yaml"
    p.write_text(policy, encoding="utf-8")

    monkeypatch.setenv("RG_POLICY_PATH", str(p))
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")

    from tldw_Server_API.app.main import app

    _reset_rg_state(app)

    # Override ChaCha DB dependency with a lightweight stub so the endpoint does
    # not touch the real database layer.
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    class _StubChaChaDB:
        def get_conversation_by_id(self, chat_id: str):
            # Provide a minimal conversation owned by user id 1 so the endpoint
            # proceeds past ownership checks.
            return {"id": chat_id, "client_id": str(user_id)}

    async def _stub_get_db(current_user=None):  # noqa: ARG001
        return _StubChaChaDB()

    app.dependency_overrides[get_chacha_db_for_user] = _stub_get_db

    try:
        with _with_rg_middleware(app):
            with TestClient(app) as client:
                chat_id = "rg-e2e-chat"
                url = f"/api/v1/chats/{chat_id}/complete"

                # First request should not be rate-limited by RG; allow downstream
                # errors as long as RG does not immediately return 429.
                r1 = client.post(url, headers={"X-API-KEY": api_key})
                assert r1.status_code != 429

                # Second request should be governed by RG policy via middleware.
                r2 = client.post(url, headers={"X-API-KEY": api_key})
    finally:
        app.dependency_overrides.pop(get_chacha_db_for_user, None)

    assert r2.status_code in (429, 503)
    if r2.status_code == 429:
        retry_after = r2.headers.get("Retry-After")
        assert retry_after is not None
        assert r2.headers.get("X-RateLimit-Limit") == "1"
        assert r2.headers.get("X-RateLimit-Remaining") == "0"
        reset = r2.headers.get("X-RateLimit-Reset")
        assert reset is not None and int(reset) >= 1


@pytest.mark.asyncio
async def test_e2e_auth_deny_headers_retry_after(monkeypatch, tmp_path, rg_backend, fresh_services):
    db_path = tmp_path / "authnz_auth_e2e.db"
    await _init_authnz_sqlite(db_path, monkeypatch)

    # Minimal app with RG middleware; enforce requests-only deny semantics for
    # an auth endpoint.
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_STORE", "file")

    policy_id = f"auth.small.{rg_backend}.{tmp_path.name.replace('-', '_')}"
    policy = (
        "schema_version: 1\n"
        "policies:\n"
        f"  {policy_id}:\n"
        "    requests: { rpm: 1 }\n"
        "    scopes: [ip]\n"
        "route_map:\n"
        "  by_path:\n"
        f"    /api/v1/auth/forgot-password: {policy_id}\n"
    )
    p = tmp_path / "rg_auth.yaml"
    p.write_text(policy, encoding="utf-8")

    monkeypatch.setenv("RG_POLICY_PATH", str(p))
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")

    from tldw_Server_API.app.main import app

    _reset_rg_state(app)

    with _with_rg_middleware(app):
        with TestClient(app) as client:
            payload = {"email": "rate-limit@example.com"}
            r1 = client.post("/api/v1/auth/forgot-password", json=payload)
            assert r1.status_code != 429

            r2 = client.post("/api/v1/auth/forgot-password", json=payload)

    assert r2.status_code in (429, 503)
    if r2.status_code == 429:
        retry_after = r2.headers.get("Retry-After")
        assert retry_after is not None
        assert r2.headers.get("X-RateLimit-Limit") == "1"
        assert r2.headers.get("X-RateLimit-Remaining") == "0"
        reset = r2.headers.get("X-RateLimit-Reset")
        assert reset is not None and int(reset) >= 1

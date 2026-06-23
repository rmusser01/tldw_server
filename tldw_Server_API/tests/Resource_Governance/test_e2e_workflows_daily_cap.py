import contextlib
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.rate_limit


@pytest.fixture(params=["memory", "redis"], ids=["rg-memory", "rg-redis"])
def rg_backend(request) -> str:
    """Exercise workflows daily-cap behavior under both RG backends."""
    return str(request.param)


def _reset_rg_state(app):


    for attr in ("rg_governor", "rg_policy_loader", "rg_policy_store", "rg_policy_version", "rg_policy_count"):
        try:
            if hasattr(app.state, attr):
                setattr(app.state, attr, None)
        except Exception:
            continue


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
        from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        await reset_db_pool()
        reset_settings()
    except Exception:
        _ = None
    try:
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_authnz_schema_ready_once

        await ensure_authnz_schema_ready_once()
    except Exception:
        _ = None
    # Reset cached RG daily ledger between tests when DATABASE_URL changes.
    try:
        import tldw_Server_API.app.core.Resource_Governance.daily_caps as _dc

        _dc._daily_ledger = None  # type: ignore[attr-defined]
    except Exception:
        _ = None
    try:
        import tldw_Server_API.app.core.Workflows.daily_ledger as _dl

        _dl._workflows_daily_ledger = None  # type: ignore[attr-defined]
        _dl._workflows_backfill_done = set()  # type: ignore[attr-defined]
    except Exception:
        _ = None


@pytest.mark.asyncio
async def test_e2e_workflows_daily_cap_denies_with_headers(monkeypatch, tmp_path, rg_backend):
    # Ensure ledger is available for enforcement.
    db_path = tmp_path / "authnz_wf_e2e.db"
    await _init_authnz_sqlite(db_path, monkeypatch)

    # Create a fresh user + API key so legacy workflow runs do not affect the cap.
    from uuid import uuid4
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    pool = await get_db_pool()
    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username="wf-cap-user",
        email="wf-cap-user@example.com",
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
    key_rec = await mgr.create_api_key(user_id=user_id, name="wf-cap-key", scope="write")
    api_key = key_rec["key"]

    # Isolate workflows content DB under a temporary user DB base dir so legacy
    # counts/backfill do not pick up runs from other tests.
    user_db_base = tmp_path / "user_dbs"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_db_base))

    # Minimal app + RG middleware.
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_STORE", "file")
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")

    # Auth is multi-user (API key) and test-mode stability.
    monkeypatch.setenv("TEST_MODE", "true")

    from tldw_Server_API.app.main import app

    _reset_rg_state(app)
    try:
        import configparser
        from tldw_Server_API.app.core.DB_Management.DB_Manager import reset_content_backend

        cfg = configparser.ConfigParser()
        cfg["Database"] = {
            "type": "sqlite",
            "workflows_path": str(tmp_path / "workflows.db"),
        }
        reset_content_backend(config=cfg, reload=False)
    except Exception:
        _ = None

    policy_id = f"workflows.small.{rg_backend}.{tmp_path.name.replace('-', '_')}"
    policy = (
        "schema_version: 1\n"
        "policies:\n"
        f"  {policy_id}:\n"
        "    requests: { rpm: 100000, burst: 1.0 }\n"
        "    workflows_runs: { daily_cap: 1 }\n"
        "    scopes: [user, api_key]\n"
        "route_map:\n"
        "  by_path:\n"
        f"    \"/api/v1/workflows/*\": {policy_id}\n"
    )
    p = tmp_path / "rg_workflows.yaml"
    p.write_text(policy, encoding="utf-8")
    monkeypatch.setenv("RG_POLICY_PATH", str(p))

    body = {
        "definition": {
            "name": "wf-small",
            "version": 1,
            "steps": [{"id": "log", "type": "log", "config": {"message": "hi"}}],
        },
        "inputs": {},
    }

    with _with_rg_middleware(app):
        with TestClient(app) as c:
            r1 = c.post(
                "/api/v1/workflows/run",
                headers={"X-API-KEY": api_key},
                json=body,
            )
            assert r1.status_code == 200, r1.text

            r2 = c.post(
                "/api/v1/workflows/run",
                headers={"X-API-KEY": api_key},
                json=body,
            )
            assert r2.status_code == 429, r2.text
            limit_hdr = r2.headers.get("X-RateLimit-Limit")
            assert limit_hdr is not None
            limit_vals = [v.strip() for v in limit_hdr.split(",")]
            assert "1" in limit_vals
            assert r2.headers.get("Retry-After") is not None

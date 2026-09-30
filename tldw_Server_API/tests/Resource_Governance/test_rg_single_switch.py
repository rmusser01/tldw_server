"""RG_ENABLED=false means no enforcement path touches a governor."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware
from tldw_Server_API.app.core.Resource_Governance.policy_loader import PolicyLoader, PolicyReloadConfig

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]

APP = Path(__file__).resolve().parents[2] / "app"

_DIVERGENT_DEFAULT_RE = re.compile(r"rg_enabled\w*\(\s*False\s*\)")


def test_no_direct_env_reads_or_divergent_defaults():
    offenders = []
    for path in APP.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.name != "config.py" and re.search(r"""getenv\(\s*["']RG_ENABLED["']""", text):
            offenders.append(f"{path.relative_to(APP)}: reads RG_ENABLED directly")
        if _DIVERGENT_DEFAULT_RE.search(text):
            offenders.append(f"{path.relative_to(APP)}: rg_enabled(False)")
    assert offenders == []


@pytest.mark.asyncio
async def test_disabled_governance_attaches_no_governor(monkeypatch):
    monkeypatch.setenv("RG_ENABLED", "false")
    from tldw_Server_API.app.services.startup_resource_governor import init_resource_governor

    app = SimpleNamespace(state=SimpleNamespace())
    await init_resource_governor(app)
    try:
        assert getattr(app.state, "rg_governor", None) is None
        assert getattr(app.state, "rg_policy_loader", None) is not None  # diag still works
    finally:
        await app.state.rg_policy_loader.shutdown()  # stop the auto-reload task


@pytest.mark.asyncio
async def test_disabled_governance_skips_auth_reservations(monkeypatch):
    monkeypatch.setenv("RG_ENABLED", "false")
    from tldw_Server_API.app.api.v1.endpoints import auth as auth_ep

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()), state=SimpleNamespace())
    assert await auth_ep._get_auth_endpoint_rg_governor(request) is None


def test_diag_lazy_governor_is_not_attached_when_disabled(monkeypatch):
    monkeypatch.setenv("RG_ENABLED", "false")
    from tldw_Server_API.app.api.v1.endpoints import resource_governor as rg_ep

    app = SimpleNamespace(state=SimpleNamespace(rg_policy_loader=SimpleNamespace(get_policy=lambda _pid: None)))
    monkeypatch.setattr(rg_ep, "_get_app", lambda: app)
    assert rg_ep._get_or_init_governor() is not None  # diagnostics still work
    assert getattr(app.state, "rg_governor", None) is None  # but enforcement stays off


_RG_SWITCH_PATH = "/api/v1/rg-switch-check"


async def _rg_switch_app(tmp_path, monkeypatch, *, fail_closed: bool = False) -> FastAPI:
    """A FastAPI app with RGSimpleMiddleware, a loader/route_map, and no attached governor.

    Mirrors the tmp_path + PolicyLoader pattern used by test_middleware_tag_enforcement.py.
    """
    monkeypatch.delenv("RG_POLICY_PATH", raising=False)
    policy: dict = {"requests": {"rpm": 1, "burst": 1.0}, "scopes": ["ip"]}
    if fail_closed:
        policy["fail_mode"] = "fail_closed"
    policy_path = tmp_path / "rg.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "policies": {"test.rg": policy},
                "route_map": {"by_path": {_RG_SWITCH_PATH: "test.rg"}},
            }
        ),
        encoding="utf-8",
    )
    loader = PolicyLoader(policy_path, PolicyReloadConfig(enabled=False))
    await loader.load_once()

    app = FastAPI()

    @app.get(_RG_SWITCH_PATH)
    def _check() -> dict:
        return {"ok": True}

    app.add_middleware(RGSimpleMiddleware)
    app.state.rg_policy_loader = loader
    app.state.rg_governor = None  # nothing initialized this run; middleware may lazily build one
    return app


@pytest.mark.asyncio
async def test_disabled_middleware_never_lazily_attaches_governor(tmp_path, monkeypatch):
    """A governed route under a disabled switch must never see a governor, ever, at ingress."""
    monkeypatch.setenv("RG_ENABLED", "false")
    app = await _rg_switch_app(tmp_path, monkeypatch)
    client = TestClient(app)

    codes = [client.get(_RG_SWITCH_PATH).status_code for _ in range(3)]

    assert codes == [200, 200, 200]
    assert getattr(app.state, "rg_governor", None) is None


@pytest.mark.asyncio
async def test_disabled_middleware_ignores_fail_closed_policy(tmp_path, monkeypatch):
    """Disabled overrides fail_mode: a fail_closed policy must not 503 when RG is off."""
    monkeypatch.setenv("RG_ENABLED", "false")
    app = await _rg_switch_app(tmp_path, monkeypatch, fail_closed=True)
    client = TestClient(app)

    codes = [client.get(_RG_SWITCH_PATH).status_code for _ in range(3)]

    assert codes == [200, 200, 200]
    assert getattr(app.state, "rg_governor", None) is None


@pytest.mark.asyncio
async def test_enabled_middleware_still_lazily_governs(tmp_path, monkeypatch):
    """Sanity: the disabled guard must not disable the enabled lazy-attach path."""
    monkeypatch.setenv("RG_ENABLED", "true")
    app = await _rg_switch_app(tmp_path, monkeypatch)
    client = TestClient(app)

    codes = [client.get(_RG_SWITCH_PATH).status_code for _ in range(2)]

    assert codes == [200, 429]
    assert getattr(app.state, "rg_governor", None) is not None


# ---------------------------------------------------------------------------
# R27: the auth_deps ingress fallback (`check_rate_limit` / `check_auth_rate_limit`)
# is itself an enforcement path and must honor the single switch too -- except
# the auth-endpoint brute-force floor, which is a deliberate exception.


def _auth_deps_fallback_request(path: str) -> SimpleNamespace:
    """Minimal Request stand-in for auth_deps._enforce_auth_deps_ingress_guard.

    No `state.rg_policy_id` (ingress did not govern this request) and no
    `state.auth` (not single-user, not RG-derived) -- mirrors the "route ingress
    left ungoverned" case the fallback exists for.
    """
    return SimpleNamespace(
        state=SimpleNamespace(),
        client=SimpleNamespace(host="127.0.0.1"),
        url=SimpleNamespace(path=path),
    )


@pytest.mark.asyncio
async def test_check_rate_limit_fallback_honors_rg_switch(monkeypatch):
    """RG_ENABLED=false must silence the *general* auth_deps fallback too."""
    monkeypatch.setenv("RG_ENABLED", "false")
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.setenv("TESTING", "0")
    monkeypatch.setenv("AUTH_DEPS_FALLBACK_RATE_LIMIT", "1")
    monkeypatch.setenv("AUTH_DEPS_FALLBACK_RATE_WINDOW_SECONDS", "60")

    from tldw_Server_API.app.api.v1.API_Deps import auth_deps

    async def _boom_get_auth_governor():
        raise AssertionError("get_auth_governor should not run once RG_ENABLED=false short-circuits")

    monkeypatch.setattr(auth_deps, "get_auth_governor", _boom_get_auth_governor)
    auth_deps._AUTH_DEPS_FALLBACK_RATE_WINDOWS.clear()

    request = _auth_deps_fallback_request("/api/v1/rag/search")

    # limit=1/min: a leftover enforcement path would 429 on the second call.
    await auth_deps.check_rate_limit(request=request)
    await auth_deps.check_rate_limit(request=request)


@pytest.mark.asyncio
async def test_check_auth_rate_limit_fallback_ignores_rg_switch(monkeypatch):
    """The auth brute-force floor (check_auth_rate_limit) keeps enforcing when RG is off."""
    monkeypatch.setenv("RG_ENABLED", "false")
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.setenv("TESTING", "0")
    monkeypatch.setenv("AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT", "1")
    monkeypatch.setenv("AUTH_DEPS_AUTH_FALLBACK_RATE_WINDOW_SECONDS", "60")

    from tldw_Server_API.app.api.v1.API_Deps import auth_deps

    async def _fake_get_auth_governor():
        return object()

    monkeypatch.setattr(auth_deps, "get_auth_governor", _fake_get_auth_governor)
    auth_deps._AUTH_DEPS_FALLBACK_RATE_WINDOWS.clear()

    request = _auth_deps_fallback_request("/api/v1/auth/forgot-password")

    await auth_deps.check_auth_rate_limit(request=request)
    with pytest.raises(HTTPException) as exc_info:
        await auth_deps.check_auth_rate_limit(request=request)
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_check_rate_limit_fallback_still_enforces_when_rg_enabled(monkeypatch):
    """Sanity: RG_ENABLED=true, request ungoverned -> the general fallback is unchanged."""
    monkeypatch.setenv("RG_ENABLED", "true")
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.setenv("TESTING", "0")
    monkeypatch.setenv("AUTH_DEPS_FALLBACK_RATE_LIMIT", "1")
    monkeypatch.setenv("AUTH_DEPS_FALLBACK_RATE_WINDOW_SECONDS", "60")

    from tldw_Server_API.app.api.v1.API_Deps import auth_deps

    async def _fake_get_auth_governor():
        return object()

    monkeypatch.setattr(auth_deps, "get_auth_governor", _fake_get_auth_governor)
    auth_deps._AUTH_DEPS_FALLBACK_RATE_WINDOWS.clear()

    request = _auth_deps_fallback_request("/api/v1/rag/search")

    await auth_deps.check_rate_limit(request=request)
    with pytest.raises(HTTPException) as exc_info:
        await auth_deps.check_rate_limit(request=request)
    assert exc_info.value.status_code == 429

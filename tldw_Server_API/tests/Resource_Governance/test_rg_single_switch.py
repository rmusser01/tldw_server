"""RG_ENABLED=false means no enforcement path touches a governor."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

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

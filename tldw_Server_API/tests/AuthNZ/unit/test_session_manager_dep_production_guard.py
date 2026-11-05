import pytest

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager


@pytest.mark.asyncio
async def test_session_manager_dep_returns_the_slot_manager_under_pytest_and_test_mode(
    fresh_services, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = SessionManager.__new__(SessionManager)
    marker._initialized = True
    fresh_services.session_manager = marker
    monkeypatch.setenv("TEST_MODE", "1")

    assert await auth_deps.get_session_manager_dep() is marker


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_override", ["0", "1"])
async def test_session_manager_dep_never_uses_test_stub_in_production_like_env(
    monkeypatch: pytest.MonkeyPatch,
    allow_override: str,
) -> None:
    marker = object()
    calls = {"real": 0}

    async def _real_session_manager():
        calls["real"] += 1
        return marker

    monkeypatch.setattr(auth_deps, "get_session_manager", _real_session_manager)
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("ALLOW_TEST_MODE_IN_PRODUCTION", allow_override)

    resolved = await auth_deps.get_session_manager_dep()

    assert resolved is marker
    assert calls["real"] == 1

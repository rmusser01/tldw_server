import pytest

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager


@pytest.mark.asyncio
async def test_session_manager_dep_returns_the_slot_manager_under_pytest_and_test_mode(
    fresh_services, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = SessionManager.__new__(SessionManager)
    fresh_services.session_manager = marker
    monkeypatch.setenv("TEST_MODE", "1")

    assert await auth_deps.get_session_manager_dep() is marker

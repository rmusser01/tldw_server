"""The single-user test key is only a settings default, never a credential of its own."""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from tldw_Server_API.app.core.AuthNZ import auth_principal_resolver, User_DB_Handling
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

pytestmark = pytest.mark.unit

_CONFIGURED = "configured-key-1234567890"
_TEST_KEY = "other-test-key-0987654321"


@pytest.fixture(autouse=True)
def _differing_keys(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", _CONFIGURED)
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", _TEST_KEY)
    monkeypatch.setenv("TEST_MODE", "1")
    reset_settings()
    yield
    reset_settings()


def _request(headers: dict[str, str] | None = None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "GET", "path": "/t", "headers": raw, "client": ("127.0.0.1", 0)})


@pytest.mark.asyncio
async def test_test_key_is_not_accepted_when_it_differs_from_the_configured_key():
    with pytest.raises(HTTPException) as exc:
        await User_DB_Handling.authenticate_api_key_user(_request(), _TEST_KEY)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_resolver_rejects_the_test_key_under_pytest():
    with pytest.raises(HTTPException) as exc:
        await auth_principal_resolver.get_auth_principal(_request({"X-API-KEY": _TEST_KEY}))
    assert exc.value.status_code == 401

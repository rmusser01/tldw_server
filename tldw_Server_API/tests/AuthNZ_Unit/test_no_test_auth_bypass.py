"""Test flags never replace authentication in the shared request-user dependency."""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from tldw_Server_API.app.core.AuthNZ import User_DB_Handling
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


@pytest.mark.asyncio
async def test_get_request_user_has_no_testing_bypass_under_pytest(monkeypatch):
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "false")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    reset_settings()
    request = Request({"type": "http", "headers": [], "path": "/", "method": "GET"})

    try:
        with pytest.raises(HTTPException) as exc:
            await User_DB_Handling.get_request_user(request, api_key=None, token=None)
    finally:
        reset_settings()

    assert exc.value.status_code == 401
    assert exc.value.detail == "Not authenticated (provide Bearer token or X-API-KEY)"

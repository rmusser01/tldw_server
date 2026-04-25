from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_auth as eval_auth
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


def _request() -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/evaluations",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_verify_api_key_testing_bypass_requires_explicit_pytest_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "false")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    reset_settings()

    with pytest.raises(HTTPException) as exc:
        await eval_auth.verify_api_key(credentials=None, x_api_key=None, request=_request())

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_api_key_testing_bypass_allows_under_explicit_pytest_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "false")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests::evals::test")
    reset_settings()

    value = await eval_auth.verify_api_key(credentials=None, x_api_key=None, request=_request())
    assert value == "test_user"


@pytest.mark.asyncio
async def test_verify_api_key_test_mode_single_user_bypass_requires_explicit_pytest_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "primary-key-123456")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    reset_settings()

    with pytest.raises(HTTPException) as exc:
        await eval_auth.verify_api_key(
            credentials=None,
            x_api_key="different-key",
            request=_request(),
        )

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_get_eval_request_user_testing_bypass_requires_explicit_pytest_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "false")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    reset_settings()

    with pytest.raises(HTTPException) as exc:
        await eval_auth.get_eval_request_user(
            request=_request(),
            _user_ctx="ctx",
            api_key=None,
            token=None,
            legacy_token_header=None,
        )

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_get_eval_request_user_testing_bypass_allows_under_explicit_pytest_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "false")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests::evals::test")
    reset_settings()

    async def _stub_get_request_user(**_kwargs):
        return SimpleNamespace(id=7, username="eval-user")

    monkeypatch.setattr(eval_auth, "get_request_user", _stub_get_request_user)

    user = await eval_auth.get_eval_request_user(
        request=_request(),
        _user_ctx="ctx",
        api_key=None,
        token=None,
        legacy_token_header=None,
    )

    assert getattr(user, "id", None) == 7


def test_enforce_heavy_admin_only_treats_on_as_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "on")

    with pytest.raises(HTTPException) as exc:
        eval_auth.enforce_heavy_evaluations_admin(
            SimpleNamespace(roles=[], permissions=[])
        )

    assert exc.value.status_code == 403


def test_get_evaluation_identity_uses_stable_user_scope() -> None:
    identity = eval_auth.get_evaluation_identity(
        User(id="tenant-user", username="tenant", email=None, is_active=True)
    )

    assert identity.user_scope == "tenant-user"
    assert identity.created_by == "tenant-user"
    assert identity.rate_limit_subject == "tenant-user"
    assert identity.webhook_user_id == "user_tenant-user"

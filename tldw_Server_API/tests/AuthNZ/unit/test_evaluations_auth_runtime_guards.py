from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_auth as eval_auth
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


_EVAL_AUTH_SENSITIVE_MARKERS = (
    "eval auth leaked",
    "jwt decode leaked",
    "jwt verify leaked",
    "rate limit header leaked",
    "/private/evaluations.db",
    "/private/jwt-secrets.db",
    "/private/rate-limits.db",
    "tenant-jwt",
)


class _LoggerStub:
    def __init__(self):
        self.debugs: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.errors: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.debugs.append((message, args, kwargs))

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.errors.append((message, args, kwargs))


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


def _multi_user_settings() -> SimpleNamespace:
    return SimpleNamespace(
        AUTH_MODE="multi_user",
        AUTH_TRUST_X_FORWARDED_FOR=False,
    )


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


def test_sanitize_error_message_logs_type_without_raw_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(eval_auth, "logger", logger_stub)

    result = eval_auth.sanitize_error_message(
        RuntimeError("eval auth leaked /private/evaluations.db"),
        "webhook registration",
    )

    assert result == "An error occurred during webhook registration"
    assert logger_stub.errors == [("Error in {}: {}", ("webhook registration", "RuntimeError"), {})]
    rendered = " ".join(
        [
            logger_stub.errors[0][0],
            *(str(arg) for arg in logger_stub.errors[0][1]),
        ]
    )
    for marker in _EVAL_AUTH_SENSITIVE_MARKERS:
        assert marker not in rendered


@pytest.mark.asyncio
async def test_verify_api_key_logs_jwt_decode_failure_without_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    class _JWTServiceStub:
        def decode_access_token(self, _token: str) -> None:
            raise RuntimeError("jwt decode leaked /private/jwt-secrets.db")

    monkeypatch.setattr(eval_auth, "logger", logger_stub)
    monkeypatch.setattr(eval_auth, "get_settings", _multi_user_settings)
    monkeypatch.setattr(eval_auth, "get_jwt_service", lambda: _JWTServiceStub())

    with pytest.raises(HTTPException) as exc:
        await eval_auth.verify_api_key(
            credentials=HTTPAuthorizationCredentials(scheme="Bearer", credentials="tenant-jwt"),
            x_api_key=None,
            request=_request(),
        )

    assert exc.value.status_code == 401
    assert exc.value.detail == {
        "error": {
            "message": "Invalid API key or token",
            "type": "authentication_error",
            "code": "invalid_token",
        }
    }
    assert logger_stub.errors == [("Unexpected error decoding JWT for evaluations auth", (), {})]
    rendered = repr(logger_stub.errors)
    for marker in _EVAL_AUTH_SENSITIVE_MARKERS:
        assert marker not in rendered


@pytest.mark.asyncio
async def test_verify_api_key_logs_jwt_verification_failure_without_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    class _JWTServiceStub:
        def decode_access_token(self, _token: str) -> None:
            return None

    async def _raise_user_verification(_request: Request, _token: str) -> None:
        raise RuntimeError("jwt verify leaked /private/jwt-secrets.db")

    monkeypatch.setattr(eval_auth, "logger", logger_stub)
    monkeypatch.setattr(eval_auth, "get_settings", _multi_user_settings)
    monkeypatch.setattr(eval_auth, "get_jwt_service", lambda: _JWTServiceStub())
    monkeypatch.setattr(eval_auth, "verify_jwt_and_fetch_user", _raise_user_verification)

    with pytest.raises(HTTPException) as exc:
        await eval_auth.verify_api_key(
            credentials=HTTPAuthorizationCredentials(scheme="Bearer", credentials="tenant-jwt"),
            x_api_key=None,
            request=_request(),
        )

    assert exc.value.status_code == 401
    assert exc.value.detail == {
        "error": {
            "message": "Invalid API key or token",
            "type": "authentication_error",
            "code": "invalid_credentials",
        }
    }
    assert logger_stub.errors == [("Unexpected error verifying JWT for evaluations auth", (), {})]
    rendered = repr(logger_stub.errors)
    for marker in _EVAL_AUTH_SENSITIVE_MARKERS:
        assert marker not in rendered


@pytest.mark.asyncio
async def test_apply_rate_limit_headers_logs_failure_without_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    class _LimiterStub:
        async def get_usage_summary(self, _user_id: str) -> None:
            raise RuntimeError("rate limit header leaked /private/rate-limits.db")

    monkeypatch.setattr(eval_auth, "logger", logger_stub)

    response = Response()
    await eval_auth._apply_rate_limit_headers(_LimiterStub(), "tenant-user", response, meta={})

    assert logger_stub.debugs == [("Failed to populate rate limit response headers", (), {})]
    rendered = repr(logger_stub.debugs)
    for marker in _EVAL_AUTH_SENSITIVE_MARKERS:
        assert marker not in rendered

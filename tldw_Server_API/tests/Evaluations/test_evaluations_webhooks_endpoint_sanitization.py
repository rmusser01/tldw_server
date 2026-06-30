import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_webhooks
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import (
    WebhookEventType,
    WebhookRegistrationRequest,
    WebhookTestRequest,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


_WEBHOOK_SENSITIVE_MARKERS = (
    "test mode leaked",
    "webhook backend leaked",
    "/private/evals-webhook.db",
)


class _LoggerStub:
    def __init__(self):
        self.debugs: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.errors: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.debugs.append((message, args, kwargs))

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.errors.append((message, args, kwargs))


class _ExplodingWebhookManager:
    def _raise(self) -> None:
        raise RuntimeError("webhook backend leaked /private/evals-webhook.db")

    def register_webhook(self, **_kwargs: object) -> None:
        self._raise()

    def get_webhook_status(self, **_kwargs: object) -> None:
        self._raise()

    def unregister_webhook(self, *_args: object, **_kwargs: object) -> None:
        self._raise()

    def test_webhook(self, **_kwargs: object) -> None:
        self._raise()


class _WebhookService:
    def __init__(self, manager: object):
        self.webhook_manager = manager


def _user() -> User:
    return User(id="tenant-user", username="tenant", email=None, is_active=True)


def _patch_failing_manager(monkeypatch: pytest.MonkeyPatch) -> _LoggerStub:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(evaluations_webhooks, "logger", logger_stub)
    monkeypatch.setattr(
        evaluations_webhooks,
        "_get_webhook_manager_for_user",
        lambda _user_id: _ExplodingWebhookManager(),
    )
    return logger_stub


def _assert_sanitized_detail(exc_info: pytest.ExceptionInfo[HTTPException], expected: str) -> None:
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == expected
    detail = str(exc_info.value.detail)
    for marker in _WEBHOOK_SENSITIVE_MARKERS:
        assert marker not in detail


def _assert_sanitized_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.errors == [(expected_message, (), {})]
    rendered = " ".join(
        [
            logger_stub.errors[0][0],
            *(str(arg) for arg in logger_stub.errors[0][1]),
        ]
    )
    for marker in _WEBHOOK_SENSITIVE_MARKERS:
        assert marker not in rendered


def _assert_sanitized_debug(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.debugs == [(expected_message, (), {})]
    rendered = repr(logger_stub.debugs)
    for marker in _WEBHOOK_SENSITIVE_MARKERS:
        assert marker not in rendered


async def test_get_webhook_manager_sanitizes_test_mode_detection_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import testing as testing_module

    logger_stub = _LoggerStub()
    manager = object()

    def _raise_test_mode_error() -> None:
        raise RuntimeError("test mode leaked /private/evals-webhook.db")

    monkeypatch.setattr(evaluations_webhooks, "logger", logger_stub)
    monkeypatch.setattr(testing_module, "is_test_mode", _raise_test_mode_error)
    monkeypatch.setattr(
        evaluations_webhooks,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: _WebhookService(manager),
    )

    result = evaluations_webhooks._get_webhook_manager_for_user("tenant-user")

    assert result is manager
    _assert_sanitized_debug(logger_stub, "Webhook test mode detection skipped")


async def test_register_webhook_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_manager(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_webhooks.register_webhook(
            request=WebhookRegistrationRequest(
                url="https://example.com/hook",
                events=[WebhookEventType.EVALUATION_COMPLETED],
            ),
            _user_ctx=object(),
            current_user=_user(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to register webhook: An error occurred during webhook registration",
    )
    _assert_sanitized_log(logger_stub, "Failed to register webhook")


async def test_list_webhooks_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_manager(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_webhooks.list_webhooks(
            _user_ctx=object(),
            current_user=_user(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to list webhooks: An error occurred during listing webhooks",
    )
    _assert_sanitized_log(logger_stub, "Failed to list webhooks")


async def test_unregister_webhook_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_manager(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_webhooks.unregister_webhook(
            url="https://example.com/hook",
            _user_ctx=object(),
            current_user=_user(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to unregister webhook: An error occurred during webhook removal",
    )
    _assert_sanitized_log(logger_stub, "Failed to unregister webhook")


async def test_test_webhook_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_manager(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_webhooks.test_webhook(
            payload=WebhookTestRequest(url="https://example.com/hook"),
            _user_ctx=object(),
            current_user=_user(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to test webhook: An error occurred during webhook testing",
    )
    _assert_sanitized_log(logger_stub, "Failed to test webhook")

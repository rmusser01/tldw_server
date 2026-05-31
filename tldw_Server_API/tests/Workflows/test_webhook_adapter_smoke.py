import os
import asyncio
import pytest

from tldw_Server_API.app.core.Workflows.adapters import run_notify_adapter, run_webhook_adapter
from tldw_Server_API.app.core.Workflows.adapters.integration import webhook as webhook_mod
from tldw_Server_API.app.core.Security import egress as egress_mod


class _FailingWebhookClient:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def post(self, *_args, **_kwargs):
        raise self._exc

    def request(self, *_args, **_kwargs):
        raise self._exc


class _FailingRegisteredWebhookManager:
    async def send_webhook(self, **_kwargs):
        raise RuntimeError("registered webhook token at /private/workflows-registered.key")


def _disable_test_mode(monkeypatch):
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)


def _allow_webhook_egress(monkeypatch):
    monkeypatch.setattr(webhook_mod, "is_url_allowed", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(webhook_mod, "is_url_allowed_for_tenant", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(egress_mod, "is_webhook_url_allowed_for_tenant", lambda *_args, **_kwargs: True)


@pytest.mark.timeout(10)
def test_webhook_adapter_test_mode(monkeypatch):
     # Force TEST_MODE so adapter short-circuits without network
    monkeypatch.setenv("TEST_MODE", "1")
    cfg = {"url": "https://example.com/echo", "method": "POST", "headers": {}, "body": {"hello": "world"}}
    ctx = {"tenant_id": "default", "user_id": "1"}
    out = asyncio.run(run_webhook_adapter(cfg, ctx))
    assert out.get("dispatched") is False
    assert out.get("test_mode") is True


@pytest.mark.timeout(10)
def test_notify_adapter_sanitizes_dispatch_errors(monkeypatch):
    _disable_test_mode(monkeypatch)
    _allow_webhook_egress(monkeypatch)
    monkeypatch.setattr(
        webhook_mod,
        "_wf_create_client",
        lambda **_kwargs: _FailingWebhookClient(RuntimeError("notify token at /private/workflows-notify.key")),
    )

    out = asyncio.run(
        run_notify_adapter(
            {"url": "https://example.com/notify", "message": "hello"},
            {"tenant_id": "default", "user_id": "1"},
        )
    )

    assert out == {"dispatched": False, "error": "Notification dispatch failed"}


@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            RuntimeError("workflow webhook token at /private/workflows-webhook.key"),
            "Workflow webhook dispatch failed",
        ),
        (
            TimeoutError("workflow webhook timeout at /private/workflows-timeout.key"),
            "Workflow webhook dispatch timed out",
        ),
    ],
)
def test_webhook_adapter_sanitizes_http_dispatch_errors(monkeypatch, exc, expected):
    _disable_test_mode(monkeypatch)
    _allow_webhook_egress(monkeypatch)
    monkeypatch.setattr(webhook_mod, "_wf_create_client", lambda **_kwargs: _FailingWebhookClient(exc))

    out = asyncio.run(
        run_webhook_adapter(
            {"url": "https://example.com/hook", "method": "POST", "body": {"hello": "world"}},
            {"tenant_id": "default", "user_id": "1"},
        )
    )

    assert out == {"dispatched": False, "error": expected}


@pytest.mark.timeout(10)
def test_webhook_adapter_sanitizes_registered_dispatch_errors(monkeypatch):
    _disable_test_mode(monkeypatch)
    import tldw_Server_API.app.core.Evaluations.webhook_manager as eval_webhook_mod

    monkeypatch.setattr(eval_webhook_mod, "webhook_manager", _FailingRegisteredWebhookManager())

    out = asyncio.run(
        run_webhook_adapter(
            {"event": "evaluation.progress", "data": {"hello": "world"}},
            {"tenant_id": "default", "user_id": "1"},
        )
    )

    assert out == {"dispatched": False, "error": "Workflow webhook dispatch failed"}


@pytest.mark.timeout(10)
def test_webhook_adapter_sanitizes_policy_error_reason(monkeypatch):
    _disable_test_mode(monkeypatch)
    _allow_webhook_egress(monkeypatch)

    def fail_policy(*_args, **_kwargs):
        raise RuntimeError("policy parser token at /private/workflows-policy.key")

    monkeypatch.setattr(egress_mod, "evaluate_url_policy", fail_policy)

    out = asyncio.run(
        run_webhook_adapter(
            {
                "url": "https://example.com/hook",
                "method": "POST",
                "body": {"hello": "world"},
                "egress_policy": {"allowlist": ["example.com"]},
            },
            {"tenant_id": "default", "user_id": "1"},
        )
    )

    assert out == {"dispatched": False, "error": "blocked_egress", "reason": "policy_error"}

"""Unit tests for admin_webhooks_service — HMAC signing, CRUD, delivery."""
from __future__ import annotations

import base64
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldw_Server_API.app.core.AuthNZ.admin_webhook_secrets import encrypt_admin_webhook_secret
from tldw_Server_API.app.services.admin_webhooks_service import (
    AdminWebhooksService,
    DeliveryLogEntry,
    WebhookRecord,
    generate_signature,
)

# ---------------------------------------------------------------------------
# HMAC signing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generate_signature_deterministic():
    sig1 = generate_signature("secret", "12345", '{"hello": "world"}')
    sig2 = generate_signature("secret", "12345", '{"hello": "world"}')
    assert sig1 == sig2
    assert sig1.startswith("v1=")


@pytest.mark.unit
def test_generate_signature_changes_with_secret():
    sig1 = generate_signature("secret-a", "12345", "body")
    sig2 = generate_signature("secret-b", "12345", "body")
    assert sig1 != sig2


@pytest.mark.unit
def test_generate_signature_changes_with_timestamp():
    sig1 = generate_signature("secret", "12345", "body")
    sig2 = generate_signature("secret", "99999", "body")
    assert sig1 != sig2


# ---------------------------------------------------------------------------
# Row parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_row_to_record_from_dict():
    encrypted = encrypt_admin_webhook_secret("s3cret")
    row = {
        "id": 1,
        "url": "https://example.com/hook",
        "secret_encrypted": encrypted.encrypted_blob,
        "secret_key_id": encrypted.key_id,
        "event_types": '["incident.created", "alert.fired"]',
        "description": "Test hook",
        "active": 1,
        "retry_count": 3,
        "timeout_seconds": 10,
        "created_by": 42,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    record = AdminWebhooksService._row_to_record(row)
    assert record.id == 1
    assert record.url == "https://example.com/hook"
    assert record.secret == "s3cret"
    assert record.event_types == ["incident.created", "alert.fired"]
    assert record.active is True
    assert record.created_by == 42


@pytest.mark.unit
def test_row_to_record_handles_invalid_json():
    encrypted = encrypt_admin_webhook_secret("s3cret")
    row = {
        "id": 2,
        "url": "https://example.com/hook",
        "secret_encrypted": encrypted.encrypted_blob,
        "secret_key_id": encrypted.key_id,
        "event_types": "NOT-JSON",
        "description": "",
        "active": 0,
        "retry_count": 1,
        "timeout_seconds": 5,
        "created_by": None,
        "created_at": None,
        "updated_at": None,
    }
    record = AdminWebhooksService._row_to_record(row)
    assert record.event_types == []
    assert record.active is False


# ---------------------------------------------------------------------------
# CRUD (with mocked pool)
# ---------------------------------------------------------------------------


def _make_pool_mock() -> MagicMock:
    """Return a mock DatabasePool with async helpers."""
    pool = MagicMock()
    pool.fetchone = AsyncMock()
    pool.fetchall = AsyncMock()
    pool.execute = AsyncMock()
    return pool


class _FakeAsyncWebhookClient:
    def __init__(self, response_or_responses):
        if isinstance(response_or_responses, list):
            self.post = AsyncMock(side_effect=response_or_responses)
        else:
            self.post = AsyncMock(return_value=response_or_responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _install_fake_webhook_client(svc: AdminWebhooksService, response_or_responses):
    factory_calls: list[dict[str, object]] = []
    client = _FakeAsyncWebhookClient(response_or_responses)

    def _factory(*, timeout: int):
        factory_calls.append({"timeout": timeout})
        return client

    svc.http_client_factory = _factory
    return client, factory_calls


@pytest.fixture
def svc() -> AdminWebhooksService:
    return AdminWebhooksService(db_pool=_make_pool_mock())


def _b64_key(seed: bytes) -> str:
    return base64.b64encode((seed * 32)[:32]).decode("ascii")


@pytest.fixture(autouse=True)
def webhook_secret_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"w"))


@pytest.mark.unit
async def test_create_webhook_generates_secret(svc: AdminWebhooksService):
    async def _return_inserted_row(_query: str, params: tuple[object, ...]) -> dict[str, object]:
        return {
            "id": 10,
            "url": params[0],
            "secret_encrypted": params[1],
            "secret_key_id": params[2],
            "event_types": params[3],
            "description": params[4],
            "active": params[5],
            "retry_count": params[6],
            "timeout_seconds": params[7],
            "created_by": params[8],
            "created_at": params[9],
            "updated_at": params[10],
        }

    svc.db_pool.fetchone.side_effect = _return_inserted_row
    record = await svc.create_webhook(url="https://example.com/hook", event_types=["*"])
    assert record.id == 10
    assert record.url == "https://example.com/hook"
    assert record.secret
    svc.db_pool.fetchone.assert_awaited_once()
    params = svc.db_pool.fetchone.await_args.args[1]
    assert params[1] != record.secret


@pytest.mark.unit
async def test_list_webhooks(svc: AdminWebhooksService):
    svc.db_pool.fetchone.return_value = {"cnt": 2}
    secret_one = encrypt_admin_webhook_secret("s1")
    secret_two = encrypt_admin_webhook_secret("s2")
    svc.db_pool.fetchall.return_value = [
        {
            "id": 1, "url": "https://a.com", "secret_encrypted": secret_one.encrypted_blob,
            "secret_key_id": secret_one.key_id,
            "event_types": '["*"]', "description": "", "active": 1,
            "retry_count": 3, "timeout_seconds": 10,
            "created_by": None, "created_at": None, "updated_at": None,
        },
        {
            "id": 2, "url": "https://b.com", "secret_encrypted": secret_two.encrypted_blob,
            "secret_key_id": secret_two.key_id,
            "event_types": '["incident.created"]', "description": "hook 2",
            "active": 0, "retry_count": 1, "timeout_seconds": 5,
            "created_by": 1, "created_at": None, "updated_at": None,
        },
    ]
    items, total = await svc.list_webhooks(limit=10, offset=0)
    assert total == 2
    assert len(items) == 2
    assert items[1].active is False


@pytest.mark.unit
async def test_update_webhook_skips_none_fields(svc: AdminWebhooksService):
    encrypted = encrypt_admin_webhook_secret("s")
    svc.db_pool.fetchone.return_value = {
        "id": 1, "url": "https://updated.com", "secret_encrypted": encrypted.encrypted_blob,
        "secret_key_id": encrypted.key_id,
        "event_types": '["*"]', "description": "updated", "active": 1,
        "retry_count": 3, "timeout_seconds": 10,
        "created_by": None, "created_at": None, "updated_at": None,
    }
    await svc.update_webhook(1, url="https://updated.com", description=None)
    # description=None should leave the existing value untouched.
    call_args = svc.db_pool.execute.call_args
    params = call_args[0][1]
    assert params[2] is None


@pytest.mark.unit
async def test_delete_webhook(svc: AdminWebhooksService):
    result = await svc.delete_webhook(42)
    assert result is True
    svc.db_pool.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_deliver_success(svc: AdminWebhooksService):
    wh = WebhookRecord(
        id=1, url="https://example.com/hook", secret="test-secret",
        event_types=["*"], description="", active=True,
        retry_count=0, timeout_seconds=5,
        created_by=None, created_at=None, updated_at=None,
    )

    # Mock the delivery log insert
    svc.db_pool.fetchone.return_value = {
        "id": 100, "webhook_id": 1, "event_type": "test.event",
        "status_code": 200, "latency_ms": 50, "retry_attempt": 0,
        "error_message": None, "delivered_at": "2026-01-01T00:00:00Z",
        "created_at": "2026-01-01T00:00:00Z",
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "OK"

    mock_client, factory_calls = _install_fake_webhook_client(svc, mock_response)

    entry = await svc.deliver(wh, "test.event", {"key": "value"})

    assert entry.status_code == 200
    assert factory_calls == [{"timeout": 5}]
    mock_client.post.assert_awaited_once()


@pytest.mark.unit
async def test_deliver_refreshes_signature_and_timestamp_for_each_retry_attempt(svc: AdminWebhooksService):
    wh = WebhookRecord(
        id=1, url="https://example.com/hook", secret="test-secret",
        event_types=["*"], description="", active=True,
        retry_count=1, timeout_seconds=5,
        created_by=None, created_at=None, updated_at=None,
    )

    svc.db_pool.fetchone.return_value = {
        "id": 101, "webhook_id": 1, "event_type": "test.event",
        "status_code": 200, "latency_ms": 50, "retry_attempt": 1,
        "error_message": None, "delivered_at": "2026-01-01T00:00:05Z",
        "created_at": "2026-01-01T00:00:05Z",
    }

    first_response = MagicMock()
    first_response.status_code = 500
    first_response.text = "server error"

    second_response = MagicMock()
    second_response.status_code = 200
    second_response.text = "OK"

    mock_client, factory_calls = _install_fake_webhook_client(svc, [first_response, second_response])

    with (
        patch("tldw_Server_API.app.services.admin_webhooks_service.asyncio.sleep", new=AsyncMock()),
        # Only the service's own clock: afetch reads time.time too.
        patch(
            "tldw_Server_API.app.services.admin_webhooks_service.time",
            SimpleNamespace(time=MagicMock(side_effect=[1000, 1005]), monotonic=time.monotonic),
        ),
    ):
        await svc.deliver(wh, "test.event", {"key": "value"})

    assert factory_calls == [{"timeout": 5}]
    first_call = mock_client.post.await_args_list[0]
    second_call = mock_client.post.await_args_list[1]
    assert first_call.kwargs["headers"]["X-Admin-Webhook-Timestamp"] == "1000"
    assert second_call.kwargs["headers"]["X-Admin-Webhook-Timestamp"] == "1005"
    assert first_call.kwargs["headers"]["X-Admin-Webhook-Signature"] != second_call.kwargs["headers"]["X-Admin-Webhook-Signature"]


@pytest.mark.unit
async def test_dispatch_event_filters_by_event_type(svc: AdminWebhooksService):
    svc.db_pool.fetchone.side_effect = [
        {"cnt": 2},  # list_webhooks count
        # deliver log entry for matching webhook
        {
            "id": 1, "webhook_id": 1, "event_type": "incident.created",
            "status_code": 200, "latency_ms": 10, "retry_attempt": 0,
            "error_message": None, "delivered_at": "2026-01-01T00:00:00Z",
            "created_at": "2026-01-01T00:00:00Z",
        },
    ]
    svc.db_pool.fetchall.return_value = [
        {
            "id": 1, "url": "https://a.com", "secret": "s1",
            "event_types": '["incident.created"]', "description": "",
            "active": 1, "retry_count": 0, "timeout_seconds": 5,
            "created_by": None, "created_at": None, "updated_at": None,
        },
        {
            "id": 2, "url": "https://b.com", "secret": "s2",
            "event_types": '["alert.fired"]', "description": "",
            "active": 1, "retry_count": 0, "timeout_seconds": 5,
            "created_by": None, "created_at": None, "updated_at": None,
        },
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "OK"

    mock_client, factory_calls = _install_fake_webhook_client(svc, mock_response)

    delivered = await svc.dispatch_event("incident.created", {"id": 1})

    assert factory_calls == [{"timeout": 5}]
    mock_client.post.assert_awaited_once()
    # Only webhook 1 matches "incident.created", webhook 2 only wants "alert.fired"
    assert delivered == 1


@pytest.mark.unit
async def test_dispatch_event_counts_only_2xx_deliveries(svc: AdminWebhooksService):
    webhooks = [
        WebhookRecord(
            id=1, url="https://a.com", secret="s1", event_types=["incident.created"],
            description="", active=True, retry_count=0, timeout_seconds=5,
            created_by=None, created_at=None, updated_at=None,
        ),
        WebhookRecord(
            id=2, url="https://b.com", secret="s2", event_types=["incident.created"],
            description="", active=True, retry_count=0, timeout_seconds=5,
            created_by=None, created_at=None, updated_at=None,
        ),
    ]
    svc.list_webhooks = AsyncMock(return_value=(webhooks, 2))  # type: ignore[method-assign]
    svc.deliver = AsyncMock(side_effect=[  # type: ignore[method-assign]
        DeliveryLogEntry(1, 1, "incident.created", 200, 10, 0, None, "2026-01-01T00:00:00Z", None),
        DeliveryLogEntry(2, 2, "incident.created", 500, 10, 0, "HTTP 500", None, None),
    ])

    delivered = await svc.dispatch_event("incident.created", {"id": 1})

    assert delivered == 1


# ---------------------------------------------------------------------------
# Delivery through core.http_client (TASK-13950.11)
# ---------------------------------------------------------------------------

_PUBLIC_HOOK_URL = "http://93.184.216.34/hook"


def _hook(url: str, *, retry_count: int = 0) -> WebhookRecord:
    return WebhookRecord(
        id=7, url=url, secret="test-secret", event_types=["*"], description="", active=True,
        retry_count=retry_count, timeout_seconds=5, created_by=None, created_at=None, updated_at=None,
    )


@pytest.fixture
def http_route(monkeypatch: pytest.MonkeyPatch, svc: AdminWebhooksService):
    """Deliver through the default client factory and the real afetch over a MockTransport.

    Egress and redirect handling run for real; only the socket layer is replaced, and any
    real connection attempt fails loudly. ``replies`` holds (status, headers, body) or an
    exception to raise, per request; the delivery log row is captured instead of written.
    """
    import socket

    import httpx

    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.services import admin_webhooks_service

    def _no_real_network(*_args, **_kwargs):
        raise AssertionError("admin webhook delivery bypassed core.http_client")

    monkeypatch.setattr(socket, "create_connection", _no_real_network)
    monkeypatch.setattr(socket.socket, "connect", _no_real_network)
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    requests: list[httpx.Request] = []
    replies: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        status, headers, body = reply
        return httpx.Response(status, headers=headers, content=body, request=request)

    def _mock_transport_client(**kwargs):
        return http_client.create_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(admin_webhooks_service, "create_async_client", _mock_transport_client, raising=False)
    log = AsyncMock(return_value=MagicMock())
    svc._log_delivery = log  # type: ignore[method-assign]
    return requests, replies, log


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1/hook", "http://169.254.169.254/latest/meta-data/", "http://10.0.0.5/hook"],
)
async def test_deliver_refuses_private_target_at_delivery_time(svc: AdminWebhooksService, http_route, url):
    """A stored webhook URL on loopback, link-local metadata or a private range is never contacted."""
    requests, _replies, log = http_route
    with patch("tldw_Server_API.app.services.admin_webhooks_service.asyncio.sleep", new=AsyncMock()) as sleep:
        await svc.deliver(_hook(url, retry_count=2), "test.event", {"key": "value"})

    assert requests == []
    logged = log.await_args.kwargs
    assert logged["status_code"] is None
    assert logged["delivered_at"] is None
    assert logged["error_message"].startswith("egress policy refused: ")
    assert "private or reserved" in logged["error_message"]
    # A policy refusal is final: no backoff, no second attempt.
    assert logged["retry_attempt"] == 0
    sleep.assert_not_awaited()


@pytest.mark.unit
async def test_deliver_does_not_follow_redirect_to_private_address(svc: AdminWebhooksService, http_route):
    """A public receiver that redirects to link-local metadata is logged as the 302 it returned."""
    requests, replies, log = http_route
    replies.append((302, {"Location": "http://169.254.169.254/latest/meta-data/"}, b""))

    await svc.deliver(_hook(_PUBLIC_HOOK_URL), "test.event", {"key": "value"})

    assert [str(r.url) for r in requests] == [_PUBLIC_HOOK_URL]
    logged = log.await_args.kwargs
    assert logged["status_code"] == 302
    assert logged["error_message"] == "HTTP 302"
    assert logged["delivered_at"] is None


@pytest.mark.unit
async def test_deliver_signs_and_posts_to_a_public_target(svc: AdminWebhooksService, http_route):
    """A valid delivery sends the same body, signature headers and timeout as before."""
    import warnings

    requests, replies, log = http_route
    replies.append((200, {}, b"OK"))

    with warnings.catch_warnings():
        # The body goes as content=, not httpx's deprecated data=<str>.
        warnings.simplefilter("error", DeprecationWarning)
        await svc.deliver(_hook(_PUBLIC_HOOK_URL), "test.event", {"key": "value"})

    (request,) = requests
    body = '{"key": "value"}'
    timestamp = request.headers["X-Admin-Webhook-Timestamp"]
    assert request.method == "POST"
    assert str(request.url) == _PUBLIC_HOOK_URL
    assert request.content == body.encode()
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["X-Admin-Webhook-Event"] == "test.event"
    assert request.headers["X-Admin-Webhook-Signature"] == generate_signature("test-secret", timestamp, body)
    assert request.extensions["timeout"] == {"connect": 5.0, "read": 5.0, "write": 5.0, "pool": 5.0}
    logged = log.await_args.kwargs
    assert logged["status_code"] == 200
    assert logged["response_body"] == "OK"
    assert logged["error_message"] is None
    assert logged["delivered_at"] is not None
    assert logged["retry_attempt"] == 0
    assert logged["signature"] == request.headers["X-Admin-Webhook-Signature"]


@pytest.mark.unit
async def test_deliver_retries_with_backoff_and_a_fresh_signature(
    svc: AdminWebhooksService, http_route, monkeypatch: pytest.MonkeyPatch,
):
    """Non-2xx answers are retried with 2**attempt backoff; each attempt is signed afresh."""
    import itertools

    from tldw_Server_API.app.services import admin_webhooks_service

    clock = itertools.count(1000, 5)
    monkeypatch.setattr(
        admin_webhooks_service,
        "time",
        SimpleNamespace(time=lambda: next(clock), monotonic=time.monotonic),
    )
    requests, replies, log = http_route
    replies.extend([(500, {}, b"boom"), (503, {}, b"busy"), (200, {}, b"OK")])

    with patch("tldw_Server_API.app.services.admin_webhooks_service.asyncio.sleep", new=AsyncMock()) as sleep:
        await svc.deliver(_hook(_PUBLIC_HOOK_URL, retry_count=2), "test.event", {"key": "value"})

    assert [call.args[0] for call in sleep.await_args_list] == [2, 4]
    timestamps = [r.headers["X-Admin-Webhook-Timestamp"] for r in requests]
    assert timestamps == ["1000", "1005", "1010"]
    for request, timestamp in zip(requests, timestamps):
        expected = generate_signature("test-secret", timestamp, '{"key": "value"}')
        assert request.headers["X-Admin-Webhook-Signature"] == expected
    logged = log.await_args.kwargs
    assert logged["status_code"] == 200
    assert logged["retry_attempt"] == 2
    assert logged["error_message"] == "HTTP 503"


def _resolve_hostnames(monkeypatch: pytest.MonkeyPatch, resolver) -> None:
    """Make the egress check resolve hostnames through ``resolver``, as it does outside pytest."""
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Security import egress

    monkeypatch.setattr(http_client, "is_explicit_pytest_runtime", lambda: False)
    monkeypatch.setattr(http_client, "env_flag_enabled", lambda _name: False)
    monkeypatch.setattr(http_client, "is_test_mode", lambda: False)
    monkeypatch.setattr(egress, "_resolve_host_ips", lambda host, **_kwargs: resolver(host))


@pytest.mark.unit
async def test_deliver_refuses_a_hostname_resolving_to_a_private_address(
    svc: AdminWebhooksService, http_route, monkeypatch: pytest.MonkeyPatch,
):
    """The DNS answer is checked at delivery, not only the literal URL."""
    requests, _replies, log = http_route
    _resolve_hostnames(monkeypatch, lambda _host: ["10.0.0.5"])

    await svc.deliver(_hook("http://hooks.internal.example/hook"), "test.event", {"key": "value"})

    assert requests == []
    logged = log.await_args.kwargs
    assert logged["status_code"] is None
    assert logged["error_message"].startswith("egress policy refused: ")


@pytest.mark.unit
async def test_deliver_rechecks_egress_on_every_attempt(
    svc: AdminWebhooksService, http_route, monkeypatch: pytest.MonkeyPatch,
):
    """A host that answers 500 and then rebinds to a private address is refused on the retry.

    The refusal row must not keep the first attempt's status and body.
    """
    requests, replies, log = http_route
    replies.append((500, {}, b"boom"))
    _resolve_hostnames(monkeypatch, lambda _host: ["10.0.0.5"] if requests else ["93.184.216.34"])

    with patch("tldw_Server_API.app.services.admin_webhooks_service.asyncio.sleep", new=AsyncMock()) as sleep:
        await svc.deliver(_hook("http://hooks.example.com/hook", retry_count=3), "test.event", {"key": "value"})

    assert len(requests) == 1
    assert requests[0].headers["Host"] == "hooks.example.com"
    # One backoff before attempt 2, then the refusal stops the loop.
    assert [call.args[0] for call in sleep.await_args_list] == [2]
    logged = log.await_args.kwargs
    assert logged["retry_attempt"] == 1
    assert logged["status_code"] is None
    assert logged["response_body"] is None
    assert logged["error_message"].startswith("egress policy refused: ")


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["egress", "transport"])
async def test_delivery_warnings_log_only_the_url_origin(
    svc: AdminWebhooksService, http_route, monkeypatch: pytest.MonkeyPatch, failure,
):
    """Slack/Discord-style URLs carry their secret in the path; warnings show only the origin."""
    import httpx

    from tldw_Server_API.app.services.admin_webhooks_service import logger

    _requests, replies, _log = http_route
    if failure == "egress":
        monkeypatch.setenv("EGRESS_DENYLIST", "hooks.slack.com")
    else:
        replies.append(httpx.ConnectError("refused"))
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        await svc.deliver(
            _hook("https://hooks.slack.com/services/T000/B000/SECRETTOKEN?x=1"), "test.event", {"key": "value"},
        )
    finally:
        logger.remove(sink)

    warning = "".join(messages)
    assert "https://hooks.slack.com" in warning
    assert "SECRETTOKEN" not in warning
    assert "/services/" not in warning

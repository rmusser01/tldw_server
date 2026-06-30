"""Unit tests for admin_webhooks_service — HMAC signing, CRUD, delivery."""
from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldw_Server_API.app.core.AuthNZ.admin_webhook_secrets import encrypt_admin_webhook_secret
from tldw_Server_API.app.services.admin_webhooks_service import (
    AdminWebhooksService,
    DeliveryLogEntry,
    WebhookRecord,
    generate_signature,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# HMAC signing
# ---------------------------------------------------------------------------


def test_generate_signature_deterministic():
    sig1 = generate_signature("secret", "12345", '{"hello": "world"}')
    sig2 = generate_signature("secret", "12345", '{"hello": "world"}')
    assert sig1 == sig2
    assert sig1.startswith("v1=")


def test_generate_signature_changes_with_secret():
    sig1 = generate_signature("secret-a", "12345", "body")
    sig2 = generate_signature("secret-b", "12345", "body")
    assert sig1 != sig2


def test_generate_signature_changes_with_timestamp():
    sig1 = generate_signature("secret", "12345", "body")
    sig2 = generate_signature("secret", "99999", "body")
    assert sig1 != sig2


# ---------------------------------------------------------------------------
# Row parsing
# ---------------------------------------------------------------------------


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

    def _factory(*, timeout: int, follow_redirects: bool):
        factory_calls.append(
            {
                "timeout": timeout,
                "follow_redirects": follow_redirects,
            }
        )
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


# ---------------------------------------------------------------------------
# Route response pagination
# ---------------------------------------------------------------------------


class _FakeAdminWebhooksRouteService:
    def __init__(self):
        self.webhook_items = [
            WebhookRecord(
                id=1,
                url="https://example.com/a",
                secret="",
                event_types=["*"],
                description="first",
                active=True,
                retry_count=3,
                timeout_seconds=10,
                created_by=None,
                created_at=None,
                updated_at=None,
            ),
            WebhookRecord(
                id=2,
                url="https://example.com/b",
                secret="",
                event_types=["incident.created"],
                description="second",
                active=False,
                retry_count=1,
                timeout_seconds=5,
                created_by=1,
                created_at=None,
                updated_at=None,
            ),
        ]
        self.delivery_items = [
            DeliveryLogEntry(
                id=10,
                webhook_id=1,
                event_type="test.one",
                status_code=200,
                latency_ms=12,
                retry_attempt=0,
                error_message=None,
                delivered_at=None,
                created_at=None,
            ),
            DeliveryLogEntry(
                id=11,
                webhook_id=1,
                event_type="test.two",
                status_code=500,
                latency_ms=24,
                retry_attempt=1,
                error_message="boom",
                delivered_at=None,
                created_at=None,
            ),
        ]

    async def list_webhooks(self, *, limit: int, offset: int, active_only: bool = False):
        self.list_webhooks_args = {"limit": limit, "offset": offset, "active_only": active_only}
        return self.webhook_items, 4

    async def get_webhook(self, webhook_id: int):
        self.get_webhook_id = webhook_id
        return self.webhook_items[0]

    async def list_delivery_log(self, webhook_id: int, *, limit: int, offset: int):
        self.list_delivery_args = {"webhook_id": webhook_id, "limit": limit, "offset": offset}
        return self.delivery_items, 5


def _make_admin_webhooks_route_client(monkeypatch: pytest.MonkeyPatch, fake_service):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_webhooks

    monkeypatch.setattr(admin_webhooks, "get_admin_webhooks_service", lambda: fake_service)
    app = FastAPI()
    app.include_router(admin_webhooks.router, prefix="/api/v1/admin")
    return TestClient(app)


def test_list_webhooks_route_includes_canonical_pagination(monkeypatch: pytest.MonkeyPatch):
    """Admin webhook list route preserves totals and exposes canonical pagination."""
    fake_service = _FakeAdminWebhooksRouteService()
    client = _make_admin_webhooks_route_client(monkeypatch, fake_service)

    response = client.get("/api/v1/admin/webhooks?limit=2&offset=1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 4
    assert len(payload["items"]) == 2
    assert payload["pagination"] == {
        "mode": "offset",
        "limit": 2,
        "offset": 1,
        "total": 4,
        "has_more": True,
        "next_offset": 3,
    }
    assert payload["has_more"] is True
    assert payload["next_offset"] == 3
    assert fake_service.list_webhooks_args == {"limit": 2, "offset": 1, "active_only": False}


def test_list_webhook_deliveries_route_includes_canonical_pagination(monkeypatch: pytest.MonkeyPatch):
    """Admin webhook delivery-log route returns canonical pagination metadata."""
    fake_service = _FakeAdminWebhooksRouteService()
    client = _make_admin_webhooks_route_client(monkeypatch, fake_service)

    response = client.get("/api/v1/admin/webhooks/1/deliveries?limit=2&offset=1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 5
    assert len(payload["items"]) == 2
    assert payload["pagination"] == {
        "mode": "offset",
        "limit": 2,
        "offset": 1,
        "total": 5,
        "has_more": True,
        "next_offset": 3,
    }
    assert payload["has_more"] is True
    assert payload["next_offset"] == 3
    assert fake_service.get_webhook_id == 1
    assert fake_service.list_delivery_args == {"webhook_id": 1, "limit": 2, "offset": 1}


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_update_webhook_skips_none_fields(svc: AdminWebhooksService):
    encrypted = encrypt_admin_webhook_secret("s")
    svc.db_pool.fetchone.return_value = {
        "id": 1, "url": "https://updated.com", "secret_encrypted": encrypted.encrypted_blob,
        "secret_key_id": encrypted.key_id,
        "event_types": '["*"]', "description": "updated", "active": 1,
        "retry_count": 3, "timeout_seconds": 10,
        "created_by": None, "created_at": None, "updated_at": None,
    }
    record = await svc.update_webhook(1, url="https://updated.com", description=None)
    # description=None should leave the existing value untouched.
    call_args = svc.db_pool.execute.call_args
    params = call_args[0][1]
    assert params[2] is None


@pytest.mark.asyncio
async def test_delete_webhook(svc: AdminWebhooksService):
    result = await svc.delete_webhook(42)
    assert result is True
    svc.db_pool.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
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
    assert factory_calls == [{"timeout": 5, "follow_redirects": False}]
    mock_client.post.assert_awaited_once()


@pytest.mark.asyncio
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
        patch("tldw_Server_API.app.services.admin_webhooks_service.time.time", side_effect=[1000, 1005]),
    ):
        await svc.deliver(wh, "test.event", {"key": "value"})

    assert factory_calls == [{"timeout": 5, "follow_redirects": False}]
    first_call = mock_client.post.await_args_list[0]
    second_call = mock_client.post.await_args_list[1]
    assert first_call.kwargs["headers"]["X-Admin-Webhook-Timestamp"] == "1000"
    assert second_call.kwargs["headers"]["X-Admin-Webhook-Timestamp"] == "1005"
    assert first_call.kwargs["headers"]["X-Admin-Webhook-Signature"] != second_call.kwargs["headers"]["X-Admin-Webhook-Signature"]


@pytest.mark.asyncio
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

    assert factory_calls == [{"timeout": 5, "follow_redirects": False}]
    mock_client.post.assert_awaited_once()
    # Only webhook 1 matches "incident.created", webhook 2 only wants "alert.fired"
    assert delivered == 1


@pytest.mark.asyncio
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

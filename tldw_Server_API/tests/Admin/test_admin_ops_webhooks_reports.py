"""Tests for webhook CRUD, webhook deliveries, report schedules, and digest
preferences in admin_system_ops_service.

Tests exercise service functions directly (not HTTP) following the monkeypatch +
JSON store isolation pattern established in test_admin_ops_new_endpoints.py.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from fastapi import HTTPException


# ---------------------------------------------------------------------------
# Shared fixture: isolate the JSON store to a temp directory
# ---------------------------------------------------------------------------


def _configure_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, Path]:
    """Redirect the system-ops JSON store to *tmp_path*."""
    from tldw_Server_API.app.services import admin_system_ops_service

    store_path = tmp_path / "system_ops.json"
    monkeypatch.setattr(admin_system_ops_service, "_STORE_PATH", store_path)
    return admin_system_ops_service, store_path


# ═══════════════════════════════════════════════════════════════════════════
# 1. Webhook Create — secret generated (hex, 64 chars), events validated
# ═══════════════════════════════════════════════════════════════════════════


class TestWebhookCreate:
    """Tests for create_webhook service function."""

    def test_create_webhook_returns_secret(self, monkeypatch, tmp_path):
        """Created webhook contains a hex secret of 64 characters (32 bytes)."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        webhook = service.create_webhook(
            url="https://example.com/hook",
            events=["user.created"],
            enabled=True,
        )

        assert webhook["id"].startswith("wh_")
        assert "secret" in webhook
        assert len(webhook["secret"]) == 64  # 32 bytes hex
        # Verify it is valid hex
        int(webhook["secret"], 16)

    def test_create_webhook_validates_events(self, monkeypatch, tmp_path):
        """Invalid events raise ValueError('invalid_events')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="invalid_events"):
            service.create_webhook(
                url="https://example.com/hook",
                events=["not.a.valid.event"],
            )

    def test_create_webhook_empty_events_rejected(self, monkeypatch, tmp_path):
        """Empty events list raises ValueError."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="invalid_events"):
            service.create_webhook(
                url="https://example.com/hook",
                events=[],
            )

    def test_create_webhook_invalid_url_rejected(self, monkeypatch, tmp_path):
        """Non-HTTP URL raises ValueError('invalid_url')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="invalid_url"):
            service.create_webhook(
                url="ftp://example.com/hook",
                events=["user.created"],
            )

    def test_create_webhook_empty_url_rejected(self, monkeypatch, tmp_path):
        """Empty URL raises ValueError('invalid_url')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="invalid_url"):
            service.create_webhook(
                url="",
                events=["user.created"],
            )

    def test_create_webhook_normalizes_events(self, monkeypatch, tmp_path):
        """Events are lowercased and sorted."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        webhook = service.create_webhook(
            url="https://example.com/hook",
            events=["Incident.Created", "USER.CREATED"],
        )

        assert webhook["events"] == ["incident.created", "user.created"]

    def test_create_webhook_multiple_valid_events(self, monkeypatch, tmp_path):
        """Multiple valid events are accepted."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        webhook = service.create_webhook(
            url="https://example.com/hook",
            events=["user.created", "incident.created", "incident.resolved"],
        )

        assert len(webhook["events"]) == 3


# ═══════════════════════════════════════════════════════════════════════════
# 2. Webhook List — secrets are redacted
# ═══════════════════════════════════════════════════════════════════════════


class TestWebhookList:
    """Tests for list_webhooks service function."""

    def test_list_webhooks_redacts_secrets(self, monkeypatch, tmp_path):
        """Listed webhooks should NOT contain the secret field."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        service.create_webhook(
            url="https://example.com/hook1",
            events=["user.created"],
        )
        service.create_webhook(
            url="https://example.com/hook2",
            events=["incident.created"],
        )

        items = service.list_webhooks()

        assert len(items) == 2
        for item in items:
            assert "secret" not in item

    def test_list_webhooks_empty(self, monkeypatch, tmp_path):
        """No webhooks yields empty list."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        items = service.list_webhooks()
        assert items == []


# ═══════════════════════════════════════════════════════════════════════════
# 3. Webhook Update — change URL, toggle enabled
# ═══════════════════════════════════════════════════════════════════════════


class TestWebhookUpdate:
    """Tests for update_webhook service function."""

    def test_update_webhook_url(self, monkeypatch, tmp_path):
        """Updating the URL changes it and returns updated webhook."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        webhook = service.create_webhook(
            url="https://example.com/old",
            events=["user.created"],
        )

        updated = service.update_webhook(
            webhook_id=webhook["id"],
            url="https://example.com/new",
        )

        assert updated["url"] == "https://example.com/new"
        # Secret should be redacted in update response
        assert "secret" not in updated

    def test_update_webhook_toggle_enabled(self, monkeypatch, tmp_path):
        """Toggling enabled flag works."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        webhook = service.create_webhook(
            url="https://example.com/hook",
            events=["user.created"],
            enabled=True,
        )

        updated = service.update_webhook(
            webhook_id=webhook["id"],
            enabled=False,
        )

        assert updated["enabled"] is False

    def test_update_webhook_not_found(self, monkeypatch, tmp_path):
        """Updating a nonexistent webhook raises ValueError('not_found')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="not_found"):
            service.update_webhook(
                webhook_id="wh_nonexistent",
                url="https://example.com/new",
            )

    def test_update_webhook_invalid_url_rejected(self, monkeypatch, tmp_path):
        """Updating with an invalid URL raises ValueError."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        webhook = service.create_webhook(
            url="https://example.com/hook",
            events=["user.created"],
        )

        with pytest.raises(ValueError, match="invalid_url"):
            service.update_webhook(
                webhook_id=webhook["id"],
                url="not-a-url",
            )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Webhook Delete — verify 404 after delete
# ═══════════════════════════════════════════════════════════════════════════


class TestWebhookDelete:
    """Tests for delete_webhook service function."""

    def test_delete_webhook_removes_it(self, monkeypatch, tmp_path):
        """Deleted webhook is no longer in the list."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        webhook = service.create_webhook(
            url="https://example.com/hook",
            events=["user.created"],
        )

        service.delete_webhook(webhook_id=webhook["id"])

        items = service.list_webhooks()
        assert len(items) == 0

    def test_delete_webhook_not_found_after_delete(self, monkeypatch, tmp_path):
        """Deleting the same webhook twice raises ValueError('not_found')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        webhook = service.create_webhook(
            url="https://example.com/hook",
            events=["user.created"],
        )

        service.delete_webhook(webhook_id=webhook["id"])

        with pytest.raises(ValueError, match="not_found"):
            service.delete_webhook(webhook_id=webhook["id"])

    def test_delete_nonexistent_webhook(self, monkeypatch, tmp_path):
        """Deleting a nonexistent webhook raises ValueError('not_found')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="not_found"):
            service.delete_webhook(webhook_id="wh_does_not_exist")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Webhook Delivery Recording — record 5, list by webhook_id, cap at 1000
# ═══════════════════════════════════════════════════════════════════════════


class TestWebhookDeliveryRecording:
    """Tests for record_webhook_delivery and list_webhook_deliveries."""

    def test_record_and_list_deliveries(self, monkeypatch, tmp_path):
        """Record 5 deliveries and list them back, newest first."""
        service, _ = _configure_store(monkeypatch, tmp_path)
        timestamps = iter(
            f"2026-05-19T00:00:0{idx}.000000+00:00"
            for idx in range(10)
        )
        monkeypatch.setattr(service, "_now_iso", lambda: next(timestamps))

        webhook = service.create_webhook(
            url="https://example.com/hook",
            events=["user.created"],
        )

        for i in range(5):
            service.record_webhook_delivery(
                webhook_id=webhook["id"],
                event_type="user.created",
                status_code=200,
                response_time_ms=50 + i,
                success=True,
            )

        deliveries, total = service.list_webhook_deliveries(webhook_id=webhook["id"])

        assert total == 5
        assert len(deliveries) == 5
        # Newest first
        assert deliveries[0]["response_time_ms"] == 54

    def test_deliveries_filtered_by_webhook_id(self, monkeypatch, tmp_path):
        """Deliveries for different webhooks are independent."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        wh1 = service.create_webhook(
            url="https://example.com/hook1",
            events=["user.created"],
        )
        wh2 = service.create_webhook(
            url="https://example.com/hook2",
            events=["incident.created"],
        )

        for _ in range(3):
            service.record_webhook_delivery(
                webhook_id=wh1["id"],
                event_type="user.created",
                status_code=200,
                response_time_ms=10,
                success=True,
            )
        for _ in range(2):
            service.record_webhook_delivery(
                webhook_id=wh2["id"],
                event_type="incident.created",
                status_code=200,
                response_time_ms=20,
                success=True,
            )

        deliveries_1, total_1 = service.list_webhook_deliveries(webhook_id=wh1["id"])
        deliveries_2, total_2 = service.list_webhook_deliveries(webhook_id=wh2["id"])

        assert total_1 == 3
        assert total_2 == 2
        assert len(deliveries_1) == 3
        assert len(deliveries_2) == 2

    def test_delivery_cap_at_1000(self, monkeypatch, tmp_path):
        """Deliveries per webhook are capped at _WEBHOOK_DELIVERIES_CAP (1000)."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        cap = service._WEBHOOK_DELIVERIES_CAP
        assert cap == 1000

        webhook = service.create_webhook(
            url="https://example.com/hook",
            events=["user.created"],
        )

        # Record cap + 5 deliveries
        for i in range(cap + 5):
            service.record_webhook_delivery(
                webhook_id=webhook["id"],
                event_type="user.created",
                status_code=200,
                response_time_ms=10,
                success=True,
            )

        deliveries, total = service.list_webhook_deliveries(
            webhook_id=webhook["id"],
            limit=cap + 100,
        )

        assert total == cap
        assert len(deliveries) <= cap

    def test_delivery_record_fields(self, monkeypatch, tmp_path):
        """Each delivery record has all expected fields."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        webhook = service.create_webhook(
            url="https://example.com/hook",
            events=["user.created"],
        )

        record = service.record_webhook_delivery(
            webhook_id=webhook["id"],
            event_type="user.created",
            status_code=200,
            response_time_ms=42,
            success=True,
            error=None,
            payload_preview='{"test": true}',
        )

        assert record["id"].startswith("wd_")
        assert record["webhook_id"] == webhook["id"]
        assert record["event_type"] == "user.created"
        assert record["status_code"] == 200
        assert record["response_time_ms"] == 42
        assert record["success"] is True
        assert record["error"] is None
        assert record["attempted_at"] is not None
        assert record["payload_preview"] == '{"test": true}'

    def test_delivery_error_recorded(self, monkeypatch, tmp_path):
        """Failed deliveries record the error message."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        webhook = service.create_webhook(
            url="https://example.com/hook",
            events=["user.created"],
        )

        record = service.record_webhook_delivery(
            webhook_id=webhook["id"],
            event_type="user.created",
            status_code=500,
            response_time_ms=100,
            success=False,
            error="Internal Server Error",
        )

        assert record["success"] is False
        assert record["error"] == "Internal Server Error"


class TestAdminOpsWebhookDeliveryPagination:
    """Direct endpoint tests for admin_ops webhook delivery pagination."""

    @pytest.mark.asyncio
    async def test_list_webhook_deliveries_includes_canonical_pagination(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        monkeypatch.setattr(admin_ops, "_require_platform_admin", lambda _: None)

        def _fake_deliveries(*, webhook_id: str, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
            assert webhook_id == "wh_1"
            assert limit == 2
            assert offset == 1
            items = [
                {
                    "id": f"wd_{idx}",
                    "webhook_id": "wh_1",
                    "event_type": "webhook.test",
                    "status_code": 200,
                    "response_time_ms": idx,
                    "success": True,
                    "error": None,
                    "attempted_at": f"2026-01-01T00:00:0{idx}Z",
                    "payload_preview": None,
                }
                for idx in range(5)
            ]
            return items[offset:offset + limit], len(items)

        monkeypatch.setattr(admin_ops, "svc_list_webhook_deliveries", _fake_deliveries)

        response = await admin_ops.list_webhook_deliveries(
            webhook_id="wh_1",
            limit=2,
            offset=1,
            principal=mock.MagicMock(),
        )

        payload = response.model_dump(mode="json")
        assert payload["total"] == 5
        assert [item["id"] for item in payload["items"]] == ["wd_1", "wd_2"]
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


# ═══════════════════════════════════════════════════════════════════════════
# 6. Webhook Test Send — mock HTTP, verify delivery recorded
# ═══════════════════════════════════════════════════════════════════════════


class TestWebhookTestSend:
    """Tests for send_test_webhook service function."""

    def test_send_test_webhook_success(self, monkeypatch, tmp_path):
        """Successful test send records a delivery with success=True."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        webhook = service.create_webhook(
            url="https://example.com/hook",
            events=["user.created"],
        )

        class _FakeResponse:
            status_code = 200

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

            def post(self, url, content, headers):
                del url, content, headers
                return _FakeResponse()

        monkeypatch.setattr(service, "_create_webhook_http_client", lambda **_: _FakeClient())
        delivery = service.send_test_webhook(webhook_id=webhook["id"])

        assert delivery["success"] is True
        assert delivery["status_code"] == 200
        assert delivery["event_type"] == "webhook.test"
        assert delivery["webhook_id"] == webhook["id"]

        # Verify delivery was also recorded in the store
        deliveries, total = service.list_webhook_deliveries(webhook_id=webhook["id"])
        assert total == 1
        assert len(deliveries) == 1

    def test_send_test_webhook_http_error(self, monkeypatch, tmp_path):
        """HTTP error response records the error in delivery."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        webhook = service.create_webhook(
            url="https://example.com/hook",
            events=["user.created"],
        )

        class _FakeResponse:
            status_code = 500

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

            def post(self, url, content, headers):
                del url, content, headers
                return _FakeResponse()

        monkeypatch.setattr(service, "_create_webhook_http_client", lambda **_: _FakeClient())
        delivery = service.send_test_webhook(webhook_id=webhook["id"])

        assert delivery["success"] is False
        assert delivery["status_code"] == 500
        assert "HTTP 500" in delivery["error"]

    def test_send_test_webhook_not_found(self, monkeypatch, tmp_path):
        """Sending to a nonexistent webhook raises ValueError('not_found')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="not_found"):
            service.send_test_webhook(webhook_id="wh_nonexistent")

    def test_send_test_webhook_connection_error(self, monkeypatch, tmp_path):
        """Connection error is recorded as a failed delivery."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        webhook = service.create_webhook(
            url="https://example.com/hook",
            events=["user.created"],
        )

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

            def post(self, url, content, headers):
                del url, content, headers
                raise OSError("Connection refused")

        monkeypatch.setattr(service, "_create_webhook_http_client", lambda **_: _FakeClient())
        delivery = service.send_test_webhook(webhook_id=webhook["id"])

        assert delivery["success"] is False
        assert delivery["status_code"] is None
        assert "Connection refused" in delivery["error"]


# ═══════════════════════════════════════════════════════════════════════════
# 7. Report Schedule Create — validate frequency/format/recipients
# ═══════════════════════════════════════════════════════════════════════════


class TestReportScheduleCreate:
    """Tests for create_report_schedule service function."""

    def test_create_report_schedule_basic(self, monkeypatch, tmp_path):
        """Create a schedule with valid frequency, format, and recipients."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedule = service.create_report_schedule(
            frequency="weekly",
            recipients=["admin@example.com"],
            report_format="html",
            enabled=True,
        )

        assert schedule["frequency"] == "weekly"
        assert schedule["recipients"] == ["admin@example.com"]
        assert schedule["format"] == "html"
        assert schedule["enabled"] is True
        assert schedule["last_sent_at"] is None

    def test_create_report_schedule_invalid_frequency(self, monkeypatch, tmp_path):
        """Invalid frequency raises ValueError('invalid_frequency')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="invalid_frequency"):
            service.create_report_schedule(
                frequency="hourly",
                recipients=["admin@example.com"],
            )

    def test_create_report_schedule_invalid_format(self, monkeypatch, tmp_path):
        """Invalid format raises ValueError('invalid_format')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="invalid_format"):
            service.create_report_schedule(
                frequency="weekly",
                recipients=["admin@example.com"],
                report_format="pdf",
            )

    def test_create_report_schedule_empty_recipients(self, monkeypatch, tmp_path):
        """Empty recipients list raises ValueError('recipients_required')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="recipients_required"):
            service.create_report_schedule(
                frequency="weekly",
                recipients=[],
            )

    def test_create_report_schedule_invalid_recipient_email(self, monkeypatch, tmp_path):
        """Recipient without @ raises ValueError('invalid_recipient_email')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="invalid_recipient_email"):
            service.create_report_schedule(
                frequency="weekly",
                recipients=["not-an-email"],
            )

    def test_create_report_schedule_all_frequencies(self, monkeypatch, tmp_path):
        """All valid frequencies are accepted: daily, weekly, monthly."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        for freq in ("daily", "weekly", "monthly"):
            schedule = service.create_report_schedule(
                frequency=freq,
                recipients=["admin@example.com"],
            )
            assert schedule["frequency"] == freq

    def test_create_report_schedule_json_format(self, monkeypatch, tmp_path):
        """JSON format is accepted."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedule = service.create_report_schedule(
            frequency="weekly",
            recipients=["admin@example.com"],
            report_format="json",
        )

        assert schedule["format"] == "json"


# ═══════════════════════════════════════════════════════════════════════════
# 8. Report Schedule List / Update / Delete
# ═══════════════════════════════════════════════════════════════════════════


class TestReportScheduleListUpdateDelete:
    """Tests for list, update, and delete of report schedules."""

    def test_list_report_schedules_empty(self, monkeypatch, tmp_path):
        """No schedules yields empty list."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedules = service.list_report_schedules()
        assert schedules == []

    def test_list_report_schedules_returns_all(self, monkeypatch, tmp_path):
        """Multiple schedules are returned."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        service.create_report_schedule(
            frequency="weekly",
            recipients=["a@b.com"],
        )
        service.create_report_schedule(
            frequency="daily",
            recipients=["c@d.com"],
        )

        schedules = service.list_report_schedules()
        assert len(schedules) == 2

    def test_update_report_schedule_frequency(self, monkeypatch, tmp_path):
        """Updating the frequency works."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedule = service.create_report_schedule(
            frequency="weekly",
            recipients=["admin@example.com"],
        )

        updated = service.update_report_schedule(
            schedule_id=schedule["id"],
            frequency="monthly",
        )

        assert updated["frequency"] == "monthly"

    def test_update_report_schedule_recipients(self, monkeypatch, tmp_path):
        """Updating recipients replaces the list."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedule = service.create_report_schedule(
            frequency="weekly",
            recipients=["old@example.com"],
        )

        updated = service.update_report_schedule(
            schedule_id=schedule["id"],
            recipients=["new1@example.com", "new2@example.com"],
        )

        assert updated["recipients"] == ["new1@example.com", "new2@example.com"]

    def test_update_report_schedule_toggle_enabled(self, monkeypatch, tmp_path):
        """Toggling enabled flag works."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedule = service.create_report_schedule(
            frequency="weekly",
            recipients=["admin@example.com"],
            enabled=True,
        )

        updated = service.update_report_schedule(
            schedule_id=schedule["id"],
            enabled=False,
        )

        assert updated["enabled"] is False

    def test_update_report_schedule_not_found(self, monkeypatch, tmp_path):
        """Updating a nonexistent schedule raises ValueError('not_found')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="not_found"):
            service.update_report_schedule(
                schedule_id="nonexistent",
                frequency="daily",
            )

    def test_update_report_schedule_invalid_frequency(self, monkeypatch, tmp_path):
        """Updating with invalid frequency raises ValueError."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedule = service.create_report_schedule(
            frequency="weekly",
            recipients=["admin@example.com"],
        )

        with pytest.raises(ValueError, match="invalid_frequency"):
            service.update_report_schedule(
                schedule_id=schedule["id"],
                frequency="biweekly",
            )

    def test_delete_report_schedule(self, monkeypatch, tmp_path):
        """Deleting a schedule removes it and returns the deleted record."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedule = service.create_report_schedule(
            frequency="weekly",
            recipients=["admin@example.com"],
        )

        deleted = service.delete_report_schedule(schedule_id=schedule["id"])

        assert deleted["id"] == schedule["id"]

        # Should be gone from the list
        schedules = service.list_report_schedules()
        assert len(schedules) == 0

    def test_delete_report_schedule_not_found(self, monkeypatch, tmp_path):
        """Deleting a nonexistent schedule raises ValueError('not_found')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="not_found"):
            service.delete_report_schedule(schedule_id="nonexistent")

    def test_mark_report_schedule_sent(self, monkeypatch, tmp_path):
        """mark_report_schedule_sent updates last_sent_at."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        schedule = service.create_report_schedule(
            frequency="weekly",
            recipients=["admin@example.com"],
        )

        assert schedule["last_sent_at"] is None

        marked = service.mark_report_schedule_sent(schedule_id=schedule["id"])

        assert marked["last_sent_at"] is not None

    def test_mark_report_schedule_sent_not_found(self, monkeypatch, tmp_path):
        """Marking a nonexistent schedule raises ValueError('not_found')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="not_found"):
            service.mark_report_schedule_sent(schedule_id="nonexistent")


class TestAdminOpsScheduleErrorSanitization:
    """Direct endpoint tests for narrow report-schedule fallback sanitization."""

    @pytest.mark.asyncio
    async def test_get_report_schedules_sanitizes_backend_error(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops._require_platform_admin",
            lambda _: None,
        )

        def _raise_schedules() -> list[dict[str, Any]]:
            raise OSError("report schedules backend exploded")

        monkeypatch.setattr(admin_ops, "svc_list_report_schedules", _raise_schedules)

        with pytest.raises(HTTPException) as exc_info:
            await admin_ops.get_report_schedules(principal=mock.MagicMock())

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to list report schedules"


class TestAdminOpsDigestErrorSanitization:
    """Direct endpoint tests for digest preference fallback sanitization."""

    @pytest.mark.asyncio
    async def test_get_digest_preference_sanitizes_backend_error(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        def _raise_pref(*, user_id: str) -> dict[str, Any] | None:
            _ = user_id
            raise OSError("digest preference backend exploded")

        monkeypatch.setattr(admin_ops, "svc_get_digest_preference", _raise_pref)

        with pytest.raises(HTTPException) as exc_info:
            await admin_ops.get_digest_preference(
                principal=mock.MagicMock(user_id="user_42"),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to get digest preference"

    @pytest.mark.asyncio
    async def test_set_digest_preference_sanitizes_backend_error(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        class _Request:
            async def json(self) -> dict[str, Any]:
                return {"email": "user42@example.com", "frequency": "weekly"}

        def _raise_pref(*, user_id: str, email: str, frequency: str) -> dict[str, Any]:
            _ = (user_id, email, frequency)
            raise OSError("digest preference write exploded")

        monkeypatch.setattr(admin_ops, "svc_set_digest_preference", _raise_pref)

        with pytest.raises(HTTPException) as exc_info:
            await admin_ops.set_digest_preference(
                request=_Request(),
                principal=mock.MagicMock(user_id="user_42"),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to set digest preference"


# ═══════════════════════════════════════════════════════════════════════════
# 9. Digest Preference Get / Set — per-user scoping
# ═══════════════════════════════════════════════════════════════════════════


class TestDigestPreferences:
    """Tests for get_digest_preference and set_digest_preference."""

    def test_set_and_get_digest_preference(self, monkeypatch, tmp_path):
        """Setting a preference and retrieving it returns the same values."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        pref = service.set_digest_preference(
            user_id="user_42",
            email="user42@example.com",
            frequency="weekly",
        )

        assert pref["user_id"] == "user_42"
        assert pref["email"] == "user42@example.com"
        assert pref["frequency"] == "weekly"
        assert pref["enabled"] is True

        retrieved = service.get_digest_preference(user_id="user_42")
        assert retrieved is not None
        assert retrieved["user_id"] == "user_42"
        assert retrieved["frequency"] == "weekly"

    def test_set_digest_preference_updates_existing(self, monkeypatch, tmp_path):
        """Setting a preference for an existing user updates it in place."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        service.set_digest_preference(
            user_id="user_42",
            email="user42@example.com",
            frequency="weekly",
        )

        updated = service.set_digest_preference(
            user_id="user_42",
            email="user42@example.com",
            frequency="daily",
        )

        assert updated["frequency"] == "daily"

        # Should still be only one entry
        retrieved = service.get_digest_preference(user_id="user_42")
        assert retrieved["frequency"] == "daily"

    def test_digest_preference_per_user_scoped(self, monkeypatch, tmp_path):
        """Different users have independent digest preferences."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        service.set_digest_preference(
            user_id="user_1",
            email="user1@example.com",
            frequency="daily",
        )
        service.set_digest_preference(
            user_id="user_2",
            email="user2@example.com",
            frequency="weekly",
        )

        pref_1 = service.get_digest_preference(user_id="user_1")
        pref_2 = service.get_digest_preference(user_id="user_2")

        assert pref_1["frequency"] == "daily"
        assert pref_2["frequency"] == "weekly"

    def test_set_digest_preference_off_disables(self, monkeypatch, tmp_path):
        """Setting frequency to 'off' sets enabled=False."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        pref = service.set_digest_preference(
            user_id="user_42",
            email="user42@example.com",
            frequency="off",
        )

        assert pref["frequency"] == "off"
        assert pref["enabled"] is False

    def test_set_digest_preference_invalid_email(self, monkeypatch, tmp_path):
        """Invalid email raises ValueError('invalid_email')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="invalid_email"):
            service.set_digest_preference(
                user_id="user_42",
                email="not-an-email",
                frequency="weekly",
            )

    def test_set_digest_preference_invalid_frequency(self, monkeypatch, tmp_path):
        """Invalid frequency raises ValueError('invalid_frequency')."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="invalid_frequency"):
            service.set_digest_preference(
                user_id="user_42",
                email="user42@example.com",
                frequency="hourly",
            )


# ═══════════════════════════════════════════════════════════════════════════
# 10. Digest Preference Default — returns None (off) for unknown user
# ═══════════════════════════════════════════════════════════════════════════


class TestDigestPreferenceDefault:
    """Tests for default digest preference behavior."""

    def test_unknown_user_returns_none(self, monkeypatch, tmp_path):
        """Getting preference for unknown user returns None."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        pref = service.get_digest_preference(user_id="unknown_user_999")

        assert pref is None

    def test_empty_store_returns_none(self, monkeypatch, tmp_path):
        """Fresh store returns None for any user."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        assert service.get_digest_preference(user_id="any_user") is None
        assert service.get_digest_preference(user_id="") is None

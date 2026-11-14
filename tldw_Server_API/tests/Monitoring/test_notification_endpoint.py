import pytest

from tldw_Server_API.app.api.v1.endpoints import monitoring as monitoring_ep
from tldw_Server_API.app.api.v1.schemas.monitoring_schemas import NotificationTestRequest


@pytest.mark.asyncio
async def test_send_test_notification_returns_status_from_notifier(monkeypatch):
    alerts: list[object] = []

    class _StubNotifier:
        def notify(self, alert):
            alerts.append(alert)
            return "skipped"

    monkeypatch.setattr(monitoring_ep, "get_notification_service", lambda: _StubNotifier())

    payload = NotificationTestRequest(severity="info", message="Test", user_id="user-1")
    resp = await monitoring_ep.send_test_notification(payload)

    assert resp.status == "skipped"
    # The endpoint must forward the request payload into the alert it notifies with.
    assert len(alerts) == 1
    assert alerts[0].user_id == "user-1"
    assert alerts[0].rule_severity == "info"
    assert alerts[0].text_snippet == "Test"

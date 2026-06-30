from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints.self_monitoring import list_alerts
from tldw_Server_API.app.core.DB_Management.Guardian_DB import SelfMonitoringAlert


pytestmark = pytest.mark.unit


def _alert(alert_id: int) -> SelfMonitoringAlert:
    return SelfMonitoringAlert(
        id=f"alert-{alert_id}",
        user_id="user1",
        rule_id="rule-1",
        rule_name="Rule 1",
        category="health",
        severity="warning",
        matched_pattern="pattern",
        context_snippet="matched text",
        notification_sent=False,
        notification_channels_used=[],
        crisis_resources_shown=False,
        created_at=f"2026-05-02T00:00:0{alert_id}+00:00",
    )


class _FakeGuardianDB:
    def __init__(self) -> None:
        self.rows = [_alert(1), _alert(2), _alert(3)]

    def list_self_monitoring_alerts(
        self,
        user_id: str,
        rule_id: str | None = None,
        unread_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SelfMonitoringAlert]:
        assert user_id == "user1"
        assert rule_id == "rule-1"
        assert unread_only is False
        return self.rows[offset:offset + limit]

    def count_self_monitoring_alerts(
        self,
        user_id: str,
        rule_id: str | None = None,
        unread_only: bool = False,
    ) -> int:
        assert user_id == "user1"
        assert rule_id == "rule-1"
        assert unread_only is False
        return len(self.rows)


def test_list_alerts_includes_canonical_pagination_and_total() -> None:
    response = list_alerts(
        rule_id="rule-1",
        unread_only=False,
        limit=2,
        offset=0,
        user=SimpleNamespace(id="user1"),  # type: ignore[arg-type]
        db=_FakeGuardianDB(),  # type: ignore[arg-type]
    )

    assert response.total == 3
    assert response.limit == 2
    assert response.offset == 0
    assert response.pagination.model_dump(mode="json") == {
        "mode": "offset",
        "limit": 2,
        "offset": 0,
        "total": 3,
        "has_more": True,
        "next_offset": 2,
    }
    assert response.has_more is True
    assert response.next_offset == 2

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.endpoints.notifications import list_notifications
from tldw_Server_API.app.core.DB_Management.Collections_DB import UserNotificationRow


pytestmark = pytest.mark.unit


def _notification_row(notification_id: int) -> UserNotificationRow:
    return UserNotificationRow(
        id=notification_id,
        user_id="778",
        kind="job_completed",
        title=f"Notification {notification_id}",
        message="Background job completed",
        severity="info",
        source_task_id=None,
        source_task_run_id=None,
        source_job_id=None,
        source_domain=None,
        source_job_type=None,
        link_type=None,
        link_id=None,
        link_url=None,
        dedupe_key=None,
        retention_until=None,
        archived_at=None,
        created_at=f"2026-05-02T00:00:0{notification_id}+00:00",
        read_at=None,
        dismissed_at=None,
    )


class _FakeNotificationsDB:
    user_id = 778

    def __init__(self) -> None:
        self.rows = [_notification_row(1), _notification_row(2), _notification_row(3)]

    def list_user_notifications(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[UserNotificationRow]:
        return self.rows[offset:offset + limit]

    def count_user_notifications(self, *, include_archived: bool = False) -> int:
        return len(self.rows)


@pytest.mark.asyncio
async def test_list_notifications_includes_canonical_pagination() -> None:
    response = await list_notifications(
        limit=2,
        offset=0,
        include_archived=False,
        only_snoozed=False,
        db=_FakeNotificationsDB(),  # type: ignore[arg-type]
        _principal=object(),
    )

    assert response.pagination.model_dump(mode="json") == {
        "mode": "offset",
        "limit": 2,
        "offset": 0,
        "total": 3,
        "has_more": True,
        "next_offset": 2,
    }
    assert response.total == 3
    assert response.limit == 2
    assert response.offset == 0
    assert response.has_more is True
    assert response.next_offset == 2

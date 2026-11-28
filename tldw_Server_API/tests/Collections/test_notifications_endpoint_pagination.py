from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.endpoints.notifications import list_notifications
from tldw_Server_API.app.api.v1.schemas.reminders_schemas import NotificationKind
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase

pytestmark = pytest.mark.integration


@pytest.fixture
def notifications_db(tmp_path: Path) -> Iterator[CollectionsDatabase]:
    backend = SQLiteBackend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(tmp_path / "notifications.db"))
    )
    try:
        yield CollectionsDatabase.from_backend("778", backend)
    finally:
        backend.get_pool().close_all()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "offset", "expected_titles", "total", "next_offset"),
    [
        (None, 0, ["Latest job", "Reminder"], 3, 2),
        (None, 2, ["First job"], 3, None),
        ("job_completed", 0, ["Latest job", "First job"], 2, None),
        ("job_failed", 0, [], 0, None),
    ],
)
async def test_list_notifications_includes_canonical_pagination(
    notifications_db: CollectionsDatabase,
    monkeypatch: pytest.MonkeyPatch,
    kind: NotificationKind | None,
    offset: int,
    expected_titles: list[str],
    total: int,
    next_offset: int | None,
) -> None:
    timestamps = (f"2026-05-02T00:00:0{second}+00:00" for second in range(3))
    monkeypatch.setattr("tldw_Server_API.app.core.DB_Management.Collections_DB._utcnow_iso", timestamps.__next__)
    for row_kind, title in [
        ("job_completed", "First job"),
        ("reminder_due", "Reminder"),
        ("job_completed", "Latest job"),
    ]:
        notifications_db.create_user_notification(kind=row_kind, title=title, message=title, severity="info")

    response = await list_notifications(
        limit=2,
        offset=offset,
        include_archived=False,
        only_snoozed=False,
        kind=kind,
        db=notifications_db,
        _principal=AuthPrincipal(kind="user", user_id=778),
    )

    assert [item.title for item in response.items] == expected_titles
    pagination = {
        "limit": 2,
        "offset": offset,
        "total": total,
        "has_more": next_offset is not None,
        "next_offset": next_offset,
    }
    assert response.pagination.model_dump(mode="json") == {"mode": "offset", **pagination}
    assert response.model_dump(mode="json", include=set(pagination)) == pagination

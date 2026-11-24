"""Calendar composition keeps shared facts below persistence and domain code."""

from __future__ import annotations

import importlib

import pytest

from tldw_Server_API.app.core.Calendar.calendar_service import CalendarService
from tldw_Server_API.app.core.Calendar.errors import CalendarValidationError
from tldw_Server_API.app.core.Calendar.view_service import CalendarViewService
from tldw_Server_API.app.core.DB_Management.Calendar_DB import CalendarDatabase
from tldw_Server_API.app.core.Scheduled_Tasks.control_plane_models import ScheduledTaskListResponse

pytestmark = pytest.mark.unit


@pytest.fixture
def calendar_db(tmp_path) -> CalendarDatabase:
    db = CalendarDatabase(db_path=tmp_path / "calendar.db")
    db.ensure_schema()
    return db


def test_calendar_constant_facade_preserves_shared_facts() -> None:
    """Persistence and domain imports refer to one set of inert constants."""
    canonical = importlib.import_module("tldw_Server_API.app.core.Utils.calendar_constants")
    facade = importlib.import_module("tldw_Server_API.app.core.Calendar.constants")
    expected = {
        "CALENDAR_DOMAIN": "calendar",
        "CALENDAR_SYNC_JOB_TYPE": "calendar_sync",
        "DEFAULT_SYNC_INTERVAL_MINUTES": 60,
        "MAX_SYNC_INTERVAL_MINUTES": 3700 * 24 * 60,
        "DEFAULT_SYNC_LOOKBACK_DAYS": 90,
        "DEFAULT_SYNC_LOOKAHEAD_DAYS": 365,
        "MAX_QUERY_WINDOW_DAYS": 370,
        "MAX_EXPANDED_OCCURRENCES": 2000,
        "CALENDAR_SOURCE_OWNER_TLDW": "tldw",
        "CALENDAR_SOURCE_OWNER_PROVIDER": "provider",
        "CALENDAR_SOURCE_OWNER_LINKED_PROJECTION": "linked_projection",
        "CALENDAR_ROLE_OWNER": "owner",
        "CALENDAR_ROLE_EDITOR": "editor",
        "CALENDAR_ROLE_COMMENTER": "commenter",
        "CALENDAR_ROLE_VIEWER": "viewer",
    }
    for name, value in expected.items():
        assert getattr(canonical, name) == value
        assert getattr(facade, name) is getattr(canonical, name)


def test_media_and_calendar_router_specs_keep_their_registration_contract(monkeypatch) -> None:
    """Extraction preserves policy, prefixes, tags and registration order."""
    from tldw_Server_API.app.api.v1.router_groups import content

    calls = []
    specs = []
    monkeypatch.setattr(content, "append_imported_router_spec", lambda target, spec: calls.append((target, spec)))
    content._append_media_and_calendar_router_specs(specs)
    assert len(calls) == 3
    assert all(target is specs for target, _spec in calls)
    assert [spec.import_path for _target, spec in calls] == [
        "tldw_Server_API.app.api.v1.endpoints.media",
        "tldw_Server_API.app.api.v1.endpoints.media.ingest_jobs",
        "tldw_Server_API.app.api.v1.endpoints.calendar",
    ]
    assert [(spec.prefix, spec.tags, spec.route_key) for _target, spec in calls[:2]] == [
        ("/api/v1/media", ("media",), "media"),
        ("/api/v1/media", ("media",), "media-ingest-jobs"),
    ]
    _target, spec = calls[2]
    assert spec.import_path == "tldw_Server_API.app.api.v1.endpoints.calendar"
    assert (spec.log_name, spec.prefix, spec.tags, spec.route_key, spec.default_stable) == (
        "calendar", "/api/v1", ("calendar",), "calendar", False,
    )


@pytest.mark.asyncio
async def test_calendar_projection_requires_an_injected_source(calendar_db) -> None:
    """Missing composition cannot silently construct an upper-layer service."""
    view = CalendarViewService(calendar_service=CalendarService(db=calendar_db))
    with pytest.raises(CalendarValidationError, match="Scheduled-task projections require"):
        await view.load_scheduled_task_projections(
            actor_user_id=1, start_at="2026-06-01", end_at="2026-06-08",
        )


@pytest.mark.asyncio
async def test_calendar_projection_preserves_a_falsy_injected_source(calendar_db) -> None:
    """A valid collaborator is not replaced based on its truth value."""
    calls = []

    class Source:
        def __bool__(self) -> bool:
            return False

        async def list_tasks(self, *, user_id: int) -> ScheduledTaskListResponse:
            calls.append(user_id)
            return ScheduledTaskListResponse(items=[], total=0)

    source = Source()
    view = CalendarViewService(calendar_service=CalendarService(db=calendar_db), scheduled_tasks_service=source)
    result = await view.load_scheduled_task_projections(
        actor_user_id=17, start_at="2026-06-01", end_at="2026-06-08",
    )
    assert result == []
    assert calls == [17]

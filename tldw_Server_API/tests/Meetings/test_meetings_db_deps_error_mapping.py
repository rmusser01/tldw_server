import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import Meetings_DB_Deps as deps
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.Meetings_DB import (
    MeetingsDatabaseError,
    SchemaError,
)


def _user() -> User:
    return User(id=84, username="meetings-user")


class _FakeUrl:
    path = "/api/v1/meetings/ws"


class _FakeWebSocket:
    headers = {}
    scope = {"query_string": b""}
    url = _FakeUrl()


@pytest.mark.asyncio
async def test_get_meetings_db_maps_schema_error(monkeypatch):
    class FailingMeetingsDatabase:
        @staticmethod
        def for_user(user_id):
            raise SchemaError("schema exploded")

    monkeypatch.setattr(deps, "MeetingsDatabase", FailingMeetingsDatabase)

    with pytest.raises(HTTPException) as exc_info:
        await deps.get_meetings_db_for_user(current_user=_user())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database schema error"


@pytest.mark.asyncio
async def test_get_meetings_db_maps_base_database_error(monkeypatch):
    class FailingMeetingsDatabase:
        @staticmethod
        def for_user(user_id):
            raise MeetingsDatabaseError("backend exploded")

    monkeypatch.setattr(deps, "MeetingsDatabase", FailingMeetingsDatabase)

    with pytest.raises(HTTPException) as exc_info:
        await deps.get_meetings_db_for_user(current_user=_user())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Meetings DB unavailable"


@pytest.mark.asyncio
async def test_get_meetings_db_for_websocket_maps_schema_error(monkeypatch):
    async def fake_get_request_user(**kwargs):
        return _user()

    class FailingMeetingsDatabase:
        @staticmethod
        def for_user(user_id):
            raise SchemaError("schema exploded")

    monkeypatch.setattr(deps, "get_request_user", fake_get_request_user)
    monkeypatch.setattr(deps, "MeetingsDatabase", FailingMeetingsDatabase)

    with pytest.raises(HTTPException) as exc_info:
        await deps.get_meetings_db_for_websocket(websocket=_FakeWebSocket())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database schema error"


@pytest.mark.asyncio
async def test_get_meetings_db_for_websocket_maps_base_database_error(monkeypatch):
    async def fake_get_request_user(**kwargs):
        return _user()

    class FailingMeetingsDatabase:
        @staticmethod
        def for_user(user_id):
            raise MeetingsDatabaseError("backend exploded")

    monkeypatch.setattr(deps, "get_request_user", fake_get_request_user)
    monkeypatch.setattr(deps, "MeetingsDatabase", FailingMeetingsDatabase)

    with pytest.raises(HTTPException) as exc_info:
        await deps.get_meetings_db_for_websocket(websocket=_FakeWebSocket())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Meetings DB unavailable"

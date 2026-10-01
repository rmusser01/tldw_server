import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import email as email_endpoint
from tldw_Server_API.app.api.v1.endpoints.email import (
    get_email_message_detail,
    list_email_sources,
    search_email_messages,
    trigger_email_source_sync,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError, InputError
from tldw_Server_API.tests.MediaIngestion_NEW.integration.test_email_sensitive_logging_13376_2 import (
    captured_logs as captured_logs,
)

pytestmark = pytest.mark.unit


class _BrokenEmailDb:
    def __init__(
        self,
        *,
        search_exc: Exception | None = None,
        detail_exc: Exception | None = None,
    ) -> None:
        self._search_exc = search_exc
        self._detail_exc = detail_exc

    def search_email_messages(self, **_kwargs):
        if self._search_exc is not None:
            raise self._search_exc
        return [], 0

    def get_email_message_detail(self, **_kwargs):
        if self._detail_exc is not None:
            raise self._detail_exc
        return {"email_message_id": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised_exc", "expected_status", "expected_detail"),
    [
        (InputError("invalid email search"), 400, "invalid email search"),
        (DatabaseError("driver failed"), 500, "A database error occurred during email search."),
    ],
)
async def test_search_email_messages_maps_db_errors(
    monkeypatch,
    raised_exc,
    expected_status,
    expected_detail,
):
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_OPERATOR_SEARCH_ENABLED", True)

    with pytest.raises(HTTPException) as exc_info:
        await search_email_messages(
            q="budget",
            limit=50,
            offset=0,
            db=_BrokenEmailDb(search_exc=raised_exc),
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail


@pytest.mark.asyncio
async def test_search_email_messages_database_error_log_is_sanitized(monkeypatch, captured_logs):
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_OPERATOR_SEARCH_ENABLED", True)

    with pytest.raises(HTTPException) as exc_info:
        await search_email_messages(
            q="budget",
            limit=50,
            offset=0,
            db=_BrokenEmailDb(search_exc=DatabaseError("driver failed at /private/email.db")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "A database error occurred during email search."
    record = next(message.record for message in captured_logs
                  if message.record["message"].startswith("Database error during email search"))
    assert record["level"].name == "ERROR"
    assert "error_type=DatabaseError" in record["message"]
    assert record["extra"]["stage"] == "email_search"
    assert record["extra"]["stack_frames"]
    rendered = "\n".join(map(str, captured_logs))
    assert "/private/email.db" not in rendered
    assert "driver failed" not in rendered


@pytest.mark.asyncio
async def test_list_email_sources_backend_error_log_is_sanitized(monkeypatch, captured_logs):
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_GMAIL_CONNECTOR_ENABLED", True)

    async def _raise_list_sources(_db, _user_id):
        raise RuntimeError("connector backend failed at /private/connectors.db")

    monkeypatch.setattr(email_endpoint, "list_connector_sources", _raise_list_sources)

    with pytest.raises(HTTPException) as exc_info:
        await list_email_sources(
            db=object(),
            principal=type("Principal", (), {"user_id": 133770042})(),
            media_db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list email sources."
    record = next(message.record for message in captured_logs
                  if message.record["message"].startswith("Failed to list email sources"))
    assert record["level"].name == "ERROR"
    assert "error_type=RuntimeError" in record["message"]
    rendered = "\n".join(map(str, captured_logs))
    assert "133770042" not in rendered
    assert "/private/connectors.db" not in rendered
    assert "backend failed" not in rendered


@pytest.mark.asyncio
async def test_trigger_email_source_sync_backend_error_log_is_sanitized(monkeypatch, captured_logs):
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_GMAIL_CONNECTOR_ENABLED", True)

    async def _get_source(_db, _user_id, _source_id):
        return {"provider": "gmail"}

    async def _raise_import_job(_user_id, _source_id, *, request_id=None):
        raise RuntimeError("queue backend failed at /private/connectors.db")

    monkeypatch.setattr(email_endpoint, "get_source_by_id", _get_source)
    monkeypatch.setattr(email_endpoint, "create_import_job", _raise_import_job)

    with pytest.raises(HTTPException) as exc_info:
        await trigger_email_source_sync(
            request=None,
            source_id=133770055,
            db=object(),
            principal=type("Principal", (), {"user_id": 133770042})(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to queue email sync job."
    record = next(message.record for message in captured_logs
                  if message.record["message"].startswith("Failed to queue email sync job"))
    assert record["level"].name == "ERROR"
    assert "error_type=RuntimeError" in record["message"]
    rendered = "\n".join(map(str, captured_logs))
    assert "133770042" not in rendered
    assert "133770055" not in rendered
    assert "/private/connectors.db" not in rendered
    assert "backend failed" not in rendered


@pytest.mark.asyncio
async def test_list_email_sources_sync_state_log_is_sanitized(monkeypatch, captured_logs):
    class _BrokenSyncStateDb:
        def get_email_sync_state(self, **_kwargs):
            raise DatabaseError("sync state failed at /private/email.db")

    monkeypatch.setitem(email_endpoint.settings, "EMAIL_GMAIL_CONNECTOR_ENABLED", True)

    async def _list_sources(_db, _user_id):
        return [
            {
                "id": 133770055,
                "account_id": 7,
                "provider": "gmail",
                "remote_id": "remote-1",
                "type": "mailbox",
                "path": "inbox",
                "options": {},
                "enabled": True,
                "last_synced_at": None,
            }
        ]

    monkeypatch.setattr(email_endpoint, "list_connector_sources", _list_sources)

    response = await list_email_sources(
        db=object(),
        principal=type("Principal", (), {"user_id": 133770042})(),
        media_db=_BrokenSyncStateDb(),
    )

    assert response["total"] == 1
    assert response["items"][0]["sync"]["state"] == "never_synced"
    record = next(message.record for message in captured_logs
                  if message.record["message"].startswith("Failed to fetch email sync state"))
    assert record["level"].name == "WARNING"
    assert "error_type=DatabaseError" in record["message"]
    rendered = "\n".join(map(str, captured_logs))
    assert "133770042" not in rendered
    assert "133770055" not in rendered
    assert "/private/email.db" not in rendered
    assert "sync state failed" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised_exc", "expected_status", "expected_detail"),
    [
        (InputError("invalid email detail"), 400, "invalid email detail"),
        (
            DatabaseError("driver failed"),
            500,
            "A database error occurred while fetching email message detail.",
        ),
    ],
)
async def test_get_email_message_detail_maps_db_errors(
    monkeypatch,
    raised_exc,
    expected_status,
    expected_detail,
):
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_OPERATOR_SEARCH_ENABLED", True)

    with pytest.raises(HTTPException) as exc_info:
        await get_email_message_detail(
            email_message_id=1,
            db=_BrokenEmailDb(detail_exc=raised_exc),
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail


@pytest.mark.asyncio
async def test_get_email_message_detail_database_error_log_is_sanitized(monkeypatch, captured_logs):
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_OPERATOR_SEARCH_ENABLED", True)

    with pytest.raises(HTTPException) as exc_info:
        await get_email_message_detail(
            email_message_id=133770042,
            db=_BrokenEmailDb(detail_exc=DatabaseError("driver failed at /private/email.db")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "A database error occurred while fetching email message detail."
    record = next(message.record for message in captured_logs
                  if message.record["message"].startswith("Database error during email detail lookup"))
    assert record["level"].name == "ERROR"
    assert "error_type=DatabaseError" in record["message"]
    assert record["extra"]["stage"] == "email_detail"
    assert record["extra"]["stack_frames"]
    rendered = "\n".join(map(str, captured_logs))
    assert "133770042" not in rendered
    assert "/private/email.db" not in rendered
    assert "driver failed" not in rendered

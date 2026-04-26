import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import email as email_endpoint
from tldw_Server_API.app.api.v1.endpoints.email import (
    get_email_message_detail,
    search_email_messages,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError, InputError


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.errors.append((message, args, kwargs))


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
async def test_search_email_messages_database_error_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_OPERATOR_SEARCH_ENABLED", True)
    monkeypatch.setattr(email_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await search_email_messages(
            q="budget",
            limit=50,
            offset=0,
            db=_BrokenEmailDb(search_exc=DatabaseError("driver failed at /private/email.db")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "A database error occurred during email search."
    assert logger_stub.errors == [("Database error during email search", (), {})]
    rendered = " ".join([logger_stub.errors[0][0], *(str(arg) for arg in logger_stub.errors[0][1])])
    assert "/private/email.db" not in rendered
    assert "driver failed" not in rendered


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
async def test_get_email_message_detail_database_error_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_OPERATOR_SEARCH_ENABLED", True)
    monkeypatch.setattr(email_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await get_email_message_detail(
            email_message_id=42,
            db=_BrokenEmailDb(detail_exc=DatabaseError("driver failed at /private/email.db")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "A database error occurred while fetching email message detail."
    assert logger_stub.errors == [("Database error during email detail lookup", (), {})]
    rendered = " ".join([logger_stub.errors[0][0], *(str(arg) for arg in logger_stub.errors[0][1])])
    assert "42" not in rendered
    assert "/private/email.db" not in rendered
    assert "driver failed" not in rendered

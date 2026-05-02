import hashlib

import pytest
from fastapi import HTTPException, Response

from tldw_Server_API.app.api.v1.endpoints.media import item as media_item_endpoint
from tldw_Server_API.app.api.v1.endpoints.media.item import (
    delete_media_item,
    get_media_item,
    permanently_delete_media_item,
    restore_media_item,
    update_media_item,
    update_media_keywords,
)
from tldw_Server_API.app.api.v1.schemas.media_request_models import (
    MediaKeywordsUpdateRequest,
    MediaUpdateRequest,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.debug_calls = []
        self.error_calls = []
        self.info_calls = []
        self.warning_calls = []
        self.exception_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def info(self, *args, **kwargs):
        self.info_calls.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exception_calls.append((args, kwargs))


_SENSITIVE_MARKERS = (
    "backend exploded",
    "/private/tmp/media-item.db",
)


def _unexpected_failure(label: str) -> RuntimeError:
    return RuntimeError(f"{label} backend exploded /private/tmp/media-item.db")


def _patch_endpoint_logger(monkeypatch: pytest.MonkeyPatch) -> _LoggerStub:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(media_item_endpoint, "logger", logger_stub, raising=True)
    return logger_stub


def _assert_sanitized_error_log(
    logger_stub: _LoggerStub,
    expected_message: str,
) -> None:
    assert logger_stub.exception_calls == []
    assert logger_stub.error_calls

    matching_messages = [args[0] for args, _kwargs in logger_stub.error_calls if args]
    assert expected_message in matching_messages
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.error_calls)

    rendered_calls = repr(logger_stub.error_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


def _assert_sanitized_debug_log(
    logger_stub: _LoggerStub,
    expected_message: str,
) -> None:
    assert logger_stub.debug_calls

    matching_messages = [args[0] for args, _kwargs in logger_stub.debug_calls if args]
    assert expected_message in matching_messages
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.debug_calls)

    rendered_calls = repr(logger_stub.debug_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


class _FakeMediaUpdateCursor:
    def __init__(self, update_exc: Exception | None):
        self._update_exc = update_exc
        self._row = None

    def execute(self, sql: str, params=()):
        if "SELECT id, uuid, content_hash, version" in sql:
            self._row = {
                "id": int(params[0]),
                "uuid": "media-uuid-1",
                "content_hash": hashlib.sha256(b"existing content").hexdigest(),
                "version": 3,
            }
            return
        if sql.startswith("UPDATE Media SET"):
            if self._update_exc is not None:
                raise self._update_exc
            return
        raise AssertionError(f"Unexpected SQL in test double: {sql}")

    def fetchone(self):
        return self._row


class _FakeMediaUpdateConnection:
    def __init__(self, update_exc: Exception | None):
        self._update_exc = update_exc

    def cursor(self):
        return _FakeMediaUpdateCursor(self._update_exc)


class _FakeMediaUpdateTransaction:
    def __init__(self, update_exc: Exception | None):
        self._update_exc = update_exc

    def __enter__(self):
        return _FakeMediaUpdateConnection(self._update_exc)

    def __exit__(self, exc_type, exc, tb):
        return False


class _BrokenMediaUpdateDb:
    client_id = "test-client"

    def __init__(self, update_exc: Exception | None):
        self._update_exc = update_exc

    def transaction(self):
        return _FakeMediaUpdateTransaction(self._update_exc)

    def _get_current_utc_timestamp_str(self):
        return "2026-04-21T00:00:00Z"


class _BrokenMediaDeleteDb:
    def __init__(self, delete_exc: Exception | None):
        self._delete_exc = delete_exc

    def mark_as_trash(self, media_id: int):
        if self._delete_exc is not None:
            raise self._delete_exc
        return True


class _BrokenMediaRestoreDb:
    def __init__(self, restore_exc: Exception | None):
        self._restore_exc = restore_exc

    def restore_from_trash(self, media_id: int):
        if self._restore_exc is not None:
            raise self._restore_exc
        return True


class _BrokenMediaKeywordsDb:
    def __init__(self, update_exc: Exception | None):
        self._update_exc = update_exc

    def update_keywords_for_media(self, *, media_id: int, keywords: list[str]):
        _ = (media_id, keywords)
        if self._update_exc is not None:
            raise self._update_exc


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised_exc", "expected_status", "expected_detail"),
    [
        (InputError("invalid media update"), 500, "Database error during update"),
        (ConflictError("stale update"), 409, "Conflict detected during update"),
        (DatabaseError("driver failed"), 500, "Database error during update"),
    ],
)
async def test_update_media_item_maps_db_errors(raised_exc, expected_status, expected_detail):
    with pytest.raises(HTTPException) as exc_info:
        await update_media_item(
            payload=MediaUpdateRequest(title="Updated title"),
            media_id=42,
            db=_BrokenMediaUpdateDb(raised_exc),
            current_user=User(id=1, username="tester", email=None, is_active=True),
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised_exc", "expected_status", "expected_detail"),
    [
        (InputError("invalid media trash target"), 400, "Invalid media identifier"),
        (ConflictError("stale trash"), 409, "Media was modified concurrently"),
        (DatabaseError("driver failed"), 500, "Database error moving media to trash"),
    ],
)
async def test_delete_media_item_maps_db_errors(
    monkeypatch,
    raised_exc,
    expected_status,
    expected_detail,
):
    monkeypatch.setattr(
        media_item_endpoint,
        "get_media_by_id",
        lambda *_args, **_kwargs: {"id": 42, "is_trash": False},
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await delete_media_item(
            media_id=42,
            db=_BrokenMediaDeleteDb(raised_exc),
            current_user=User(id=1, username="tester", email=None, is_active=True),
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised_exc", "expected_status", "expected_detail"),
    [
        (InputError("invalid media restore target"), 400, "Invalid media identifier"),
        (ConflictError("stale restore"), 409, "Media was modified concurrently"),
        (DatabaseError("driver failed"), 500, "Database error restoring media from trash"),
    ],
)
async def test_restore_media_item_maps_db_errors(
    monkeypatch,
    raised_exc,
    expected_status,
    expected_detail,
):
    monkeypatch.setattr(
        media_item_endpoint,
        "get_media_by_id",
        lambda *_args, **_kwargs: {"id": 42, "is_trash": True},
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await restore_media_item(
            media_id=42,
            include_content=True,
            include_versions=True,
            include_version_content=False,
            db=_BrokenMediaRestoreDb(raised_exc),
            current_user=User(id=1, username="tester", email=None, is_active=True),
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised_exc", "expected_status", "expected_detail"),
    [
        (InputError("invalid permanent delete target"), 400, "Invalid media identifier"),
        (ConflictError("stale permanent delete"), 409, "Media was modified concurrently"),
        (DatabaseError("driver failed"), 500, "Database error permanently deleting media"),
    ],
)
async def test_permanently_delete_media_item_maps_db_errors(
    monkeypatch,
    raised_exc,
    expected_status,
    expected_detail,
):
    monkeypatch.setattr(
        media_item_endpoint,
        "get_media_by_id",
        lambda *_args, **_kwargs: {"id": 42, "is_trash": True},
        raising=True,
    )

    def _raise_permanent_delete(_db, _media_id):
        raise raised_exc

    monkeypatch.setattr(
        media_item_endpoint,
        "permanently_delete_item",
        _raise_permanent_delete,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await permanently_delete_media_item(
            media_id=42,
            db=object(),
            current_user=User(id=1, username="tester", email=None, is_active=True),
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail


@pytest.mark.asyncio
async def test_get_media_item_maps_database_error(monkeypatch):
    monkeypatch.setattr(
        media_item_endpoint,
        "get_full_media_details_rich",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DatabaseError("driver failed")),
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_media_item(
            request=type("_Req", (), {"headers": {}})(),
            response=Response(),
            media_id=42,
            include_content=True,
            include_versions=True,
            include_version_content=False,
            db=object(),
            current_user=User(id=1, username="tester", email=None, is_active=True),
            if_none_match=None,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error retrieving media details"


@pytest.mark.asyncio
async def test_update_media_keywords_maps_not_found_input_error(monkeypatch):
    monkeypatch.setattr(
        media_item_endpoint,
        "fetch_keywords_for_media",
        lambda *_args, **_kwargs: ["alpha"],
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await update_media_keywords(
            payload=MediaKeywordsUpdateRequest(keywords=["beta"], mode="add"),
            media_id=42,
            db=_BrokenMediaKeywordsDb(InputError("Cannot update keywords: Media ID 42 not found or deleted.")),
            _current_user=User(id=1, username="tester", email=None, is_active=True),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Media not found or deleted"


@pytest.mark.asyncio
async def test_update_media_keywords_maps_database_error(monkeypatch):
    monkeypatch.setattr(
        media_item_endpoint,
        "fetch_keywords_for_media",
        lambda *_args, **_kwargs: ["alpha"],
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await update_media_keywords(
            payload=MediaKeywordsUpdateRequest(keywords=["beta"], mode="add"),
            media_id=42,
            db=_BrokenMediaKeywordsDb(DatabaseError("driver failed")),
            _current_user=User(id=1, username="tester", email=None, is_active=True),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to update keywords"


@pytest.mark.asyncio
async def test_get_media_item_sanitizes_unexpected_error_log(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)
    monkeypatch.setattr(
        media_item_endpoint,
        "get_full_media_details_rich",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_unexpected_failure("detail")),
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_media_item(
            request=type("_Req", (), {"headers": {}})(),
            response=Response(),
            media_id=42,
            include_content=True,
            include_versions=True,
            include_version_content=False,
            db=object(),
            current_user=User(id=1, username="tester", email=None, is_active=True),
            if_none_match=None,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "An unexpected error occurred retrieving media details"
    _assert_sanitized_error_log(
        logger_stub,
        "Unexpected error fetching details for media {}",
    )


@pytest.mark.asyncio
async def test_get_media_item_auth_header_diagnostic_failure_log_is_sanitized(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    class ExplodingHeaders:
        def get(self, _key: str) -> bool:
            raise RuntimeError("header diagnostics backend exploded /private/tmp/media-item.db")

    class FakeMediaDetailResponse:
        def __init__(self, **details):
            self._details = details

        def model_dump(self):
            return self._details

    monkeypatch.setattr(media_item_endpoint, "_is_test_mode", lambda: True, raising=True)
    monkeypatch.setattr(
        media_item_endpoint,
        "get_full_media_details_rich",
        lambda *_args, **_kwargs: {"id": 42, "title": "test item"},
        raising=True,
    )
    monkeypatch.setattr(
        media_item_endpoint,
        "MediaDetailResponse",
        FakeMediaDetailResponse,
        raising=True,
    )

    payload = await get_media_item(
        request=type("_Req", (), {"headers": ExplodingHeaders()})(),
        response=Response(),
        media_id=42,
        include_content=True,
        include_versions=True,
        include_version_content=False,
        db=object(),
        current_user=User(id=1, username="tester", email=None, is_active=True),
        if_none_match=None,
    )

    assert payload["id"] == 42
    _assert_sanitized_debug_log(
        logger_stub,
        "Failed to emit media item auth header diagnostics",
    )


@pytest.mark.asyncio
async def test_delete_media_item_sanitizes_unexpected_error_log(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)
    monkeypatch.setattr(
        media_item_endpoint,
        "get_media_by_id",
        lambda *_args, **_kwargs: {"id": 42, "is_trash": False},
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await delete_media_item(
            media_id=42,
            db=_BrokenMediaDeleteDb(_unexpected_failure("trash")),
            current_user=User(id=1, username="tester", email=None, is_active=True),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Unexpected error moving media to trash"
    _assert_sanitized_error_log(
        logger_stub,
        "Unexpected error trashing media {}",
    )


@pytest.mark.asyncio
async def test_restore_media_item_sanitizes_unexpected_error_log(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)
    monkeypatch.setattr(
        media_item_endpoint,
        "get_media_by_id",
        lambda *_args, **_kwargs: {"id": 42, "is_trash": True},
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await restore_media_item(
            media_id=42,
            include_content=True,
            include_versions=True,
            include_version_content=False,
            db=_BrokenMediaRestoreDb(_unexpected_failure("restore")),
            current_user=User(id=1, username="tester", email=None, is_active=True),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Unexpected error restoring media from trash"
    _assert_sanitized_error_log(
        logger_stub,
        "Unexpected error restoring media {}",
    )


@pytest.mark.asyncio
async def test_permanently_delete_media_item_sanitizes_unexpected_error_log(
    monkeypatch,
):
    logger_stub = _patch_endpoint_logger(monkeypatch)
    monkeypatch.setattr(
        media_item_endpoint,
        "get_media_by_id",
        lambda *_args, **_kwargs: {"id": 42, "is_trash": True},
        raising=True,
    )

    def _raise_permanent_delete(_db, _media_id):
        raise _unexpected_failure("permanent delete")

    monkeypatch.setattr(
        media_item_endpoint,
        "permanently_delete_item",
        _raise_permanent_delete,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await permanently_delete_media_item(
            media_id=42,
            db=object(),
            current_user=User(id=1, username="tester", email=None, is_active=True),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Unexpected error permanently deleting media"
    _assert_sanitized_error_log(
        logger_stub,
        "Unexpected error permanently deleting media {}",
    )


@pytest.mark.asyncio
async def test_update_media_item_sanitizes_unexpected_error_log(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await update_media_item(
            payload=MediaUpdateRequest(title="Updated title"),
            media_id=42,
            db=_BrokenMediaUpdateDb(_unexpected_failure("update")),
            current_user=User(id=1, username="tester", email=None, is_active=True),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "An unexpected error occurred"
    _assert_sanitized_error_log(
        logger_stub,
        "Unexpected error updating media {}",
    )


@pytest.mark.asyncio
async def test_update_media_keywords_sanitizes_unexpected_error_log(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)
    monkeypatch.setattr(
        media_item_endpoint,
        "fetch_keywords_for_media",
        lambda *_args, **_kwargs: ["alpha"],
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await update_media_keywords(
            payload=MediaKeywordsUpdateRequest(keywords=["beta"], mode="add"),
            media_id=42,
            db=_BrokenMediaKeywordsDb(_unexpected_failure("keywords")),
            _current_user=User(id=1, username="tester", email=None, is_active=True),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to update keywords"
    _assert_sanitized_error_log(
        logger_stub,
        "Unexpected error updating keywords for media {}",
    )

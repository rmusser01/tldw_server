import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.media import versions as media_versions_endpoint
from tldw_Server_API.app.api.v1.endpoints.media.versions import delete_version
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
    "driver failed",
    "version delete exploded",
    "/private/tmp/media-versions-delete.db",
)


def _database_failure() -> DatabaseError:
    return DatabaseError("driver failed /private/tmp/media-versions-delete.db")


def _patch_endpoint_logger(monkeypatch: pytest.MonkeyPatch) -> _LoggerStub:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(media_versions_endpoint, "logger", logger_stub, raising=True)
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


class _FakeDeleteVersionQueryCursor:
    def fetchone(self):
        return {"uuid": "version-uuid-1"}


class _FakeDeleteVersionTransaction:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _BrokenDeleteVersionDb:
    def __init__(self, delete_exc: Exception):
        self._delete_exc = delete_exc

    def execute_query(self, query: str, params):
        _ = (query, params)
        return _FakeDeleteVersionQueryCursor()

    def transaction(self):
        return _FakeDeleteVersionTransaction()

    def soft_delete_document_version(self, *, version_uuid: str):
        _ = version_uuid
        raise self._delete_exc


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised_exc", "expected_status", "expected_detail"),
    [
        (InputError("invalid version delete"), 400, "invalid version delete"),
        (ConflictError("stale delete conflict"), 409, "Conflict during deletion"),
        (_database_failure(), 500, "Database error occurred"),
    ],
)
async def test_delete_version_maps_db_errors(
    monkeypatch,
    raised_exc,
    expected_status,
    expected_detail,
):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await delete_version(
            media_id=42,
            version_number=2,
            db=_BrokenDeleteVersionDb(raised_exc),
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail
    if expected_status == 500:
        _assert_sanitized_error_log(
            logger_stub,
            "Database error deleting version {} for media {}",
        )


@pytest.mark.asyncio
async def test_delete_version_sanitizes_unexpected_error_log(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await delete_version(
            media_id=42,
            version_number=2,
            db=_BrokenDeleteVersionDb(
                RuntimeError("version delete exploded /private/tmp/media-versions-delete.db"),
            ),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal server error deleting version"
    _assert_sanitized_error_log(
        logger_stub,
        "Unexpected error deleting version {} for media {}",
    )

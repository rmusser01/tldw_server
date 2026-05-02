import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.media import versions as media_versions_endpoint
from tldw_Server_API.app.api.v1.endpoints.media.versions import get_version, list_versions
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError

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
    "versions read exploded",
    "/private/tmp/media-versions.db",
)


def _database_failure() -> DatabaseError:
    return DatabaseError("driver failed /private/tmp/media-versions.db")


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


@pytest.mark.asyncio
async def test_list_versions_maps_database_error(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)
    monkeypatch.setattr(
        media_versions_endpoint,
        "check_media_exists",
        lambda _db, media_id: media_id,
        raising=True,
    )

    def _raise_database_error(*_args, **_kwargs):
        raise _database_failure()

    monkeypatch.setattr(
        media_versions_endpoint,
        "list_document_versions",
        _raise_database_error,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await list_versions(
            media_id=42,
            include_content=False,
            limit=10,
            page=1,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error occurred"
    _assert_sanitized_error_log(
        logger_stub,
        "Database error listing versions for media {}",
    )


@pytest.mark.asyncio
async def test_get_version_maps_database_error(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _raise_database_error(*_args, **_kwargs):
        raise _database_failure()

    monkeypatch.setattr(
        media_versions_endpoint,
        "get_document_version",
        _raise_database_error,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_version(
            media_id=42,
            version_number=2,
            include_content=True,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error occurred"
    _assert_sanitized_error_log(
        logger_stub,
        "Database error getting version {} for media {}",
    )


@pytest.mark.asyncio
async def test_list_versions_sanitizes_unexpected_error_log(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)
    monkeypatch.setattr(
        media_versions_endpoint,
        "check_media_exists",
        lambda _db, media_id: media_id,
        raising=True,
    )

    def _raise_unexpected_error(*_args, **_kwargs):
        raise RuntimeError("versions read exploded /private/tmp/media-versions.db")

    monkeypatch.setattr(
        media_versions_endpoint,
        "list_document_versions",
        _raise_unexpected_error,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await list_versions(
            media_id=42,
            include_content=False,
            limit=10,
            page=1,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal server error listing versions"
    _assert_sanitized_error_log(
        logger_stub,
        "Unexpected error listing versions for media {}",
    )


@pytest.mark.asyncio
async def test_get_version_sanitizes_unexpected_error_log(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    def _raise_unexpected_error(*_args, **_kwargs):
        raise RuntimeError("versions read exploded /private/tmp/media-versions.db")

    monkeypatch.setattr(
        media_versions_endpoint,
        "get_document_version",
        _raise_unexpected_error,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_version(
            media_id=42,
            version_number=2,
            include_content=True,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal server error getting version"
    _assert_sanitized_error_log(
        logger_stub,
        "Unexpected error getting version {} for media {}",
    )

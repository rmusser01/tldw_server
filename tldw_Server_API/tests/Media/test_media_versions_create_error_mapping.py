import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.media import versions as media_versions_endpoint
from tldw_Server_API.app.api.v1.endpoints.media.versions import create_version
from tldw_Server_API.app.api.v1.schemas.media_request_models import VersionCreateRequest
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
    "driver failed",
    "version create conflict",
    "version create exploded",
    "/private/tmp/media-versions-create.db",
)


def _database_failure() -> DatabaseError:
    return DatabaseError("driver failed /private/tmp/media-versions-create.db")


def _conflict_failure() -> ConflictError:
    return ConflictError("version create conflict /private/tmp/media-versions-create.db")


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


class _FakeRequest:
    headers = {}


class _FakeCreateVersionTransaction:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _BrokenCreateVersionDb:
    def __init__(self, create_exc: Exception | None = None):
        self._create_exc = create_exc
        self.create_calls = 0

    def transaction(self):
        return _FakeCreateVersionTransaction()

    def create_document_version(self, **_kwargs):
        self.create_calls += 1
        if self._create_exc is not None:
            raise self._create_exc
        return {"version_number": 2, "uuid": "version-uuid-2"}


@pytest.mark.asyncio
async def test_create_version_returns_404_before_db_call_when_media_missing(monkeypatch):
    db = _BrokenCreateVersionDb()
    monkeypatch.setattr(
        media_versions_endpoint,
        "check_media_exists",
        lambda _db, media_id: None,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_version(
            media_id=42,
            request_body=VersionCreateRequest(
                content="v2 content",
                prompt="prompt",
                analysis_content="analysis",
            ),
            request=_FakeRequest(),
            db=db,
            current_user=User(id=1, username="tester", email=None, is_active=True),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Media not found or deleted"
    assert db.create_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised_exc", "expected_status", "expected_detail"),
    [
        (InputError("invalid version create"), 400, "invalid version create"),
        (_conflict_failure(), 500, "Database error occurred"),
        (_database_failure(), 500, "Database error occurred"),
    ],
)
async def test_create_version_maps_db_errors_when_media_exists(
    monkeypatch,
    raised_exc,
    expected_status,
    expected_detail,
):
    logger_stub = _patch_endpoint_logger(monkeypatch)
    monkeypatch.setattr(
        media_versions_endpoint,
        "check_media_exists",
        lambda _db, media_id: media_id,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_version(
            media_id=42,
            request_body=VersionCreateRequest(
                content="v2 content",
                prompt="prompt",
                analysis_content="analysis",
            ),
            request=_FakeRequest(),
            db=_BrokenCreateVersionDb(raised_exc),
            current_user=User(id=1, username="tester", email=None, is_active=True),
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail
    if expected_status == 500:
        _assert_sanitized_error_log(
            logger_stub,
            "Database error creating version for media {}",
        )


@pytest.mark.asyncio
async def test_create_version_sanitizes_unexpected_error_log(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)
    monkeypatch.setattr(
        media_versions_endpoint,
        "check_media_exists",
        lambda _db, media_id: media_id,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_version(
            media_id=42,
            request_body=VersionCreateRequest(
                content="v2 content",
                prompt="prompt",
                analysis_content="analysis",
            ),
            request=_FakeRequest(),
            db=_BrokenCreateVersionDb(
                RuntimeError("version create exploded /private/tmp/media-versions-create.db"),
            ),
            current_user=User(id=1, username="tester", email=None, is_active=True),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal server error during version creation"
    _assert_sanitized_error_log(
        logger_stub,
        "Unexpected error creating version for media {}",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [False, True])
async def test_create_version_requires_the_analyzed_source(monkeypatch, changed):
    from unittest.mock import Mock

    from tldw_Server_API.app.api.v1.schemas.media_response_models import MediaDetailResponse
    from tldw_Server_API.app.api.v1.utils.media_analysis import media_source_sha256

    db = _BrokenCreateVersionDb()
    db.create_document_version = Mock(return_value={"version_number": 2, "uuid": "version-uuid-2"})
    text = "x" * 1_405_967
    monkeypatch.setattr(media_versions_endpoint, "check_media_exists", lambda *_args, **_kwargs: True)
    read_text = Mock(return_value=text)
    monkeypatch.setattr(media_versions_endpoint, "read_saved_media_text", read_text)
    monkeypatch.setattr(media_versions_endpoint, "get_full_media_details_rich", lambda **_kwargs: {"id": 42})
    monkeypatch.setattr(media_versions_endpoint, "MediaDetailResponse", lambda **_kwargs: MediaDetailResponse.model_construct(id=42))

    request = create_version(
        media_id=42,
        request_body=VersionCreateRequest(
            prompt="prompt", analysis_content="analysis",
            source_sha256=media_source_sha256("previous text" if changed else text),
        ),
        request=_FakeRequest(),
        db=db,
        current_user=User(id=1, username="tester", email=None, is_active=True),
    )
    if changed:
        with pytest.raises(HTTPException) as error:
            await request
        assert error.value.status_code == 409
        read_text.assert_called_once_with(db, 42, owner_user_id=1, for_update=True)
        db.create_document_version.assert_not_called()
        return

    await request
    read_text.assert_called_once_with(db, 42, owner_user_id=1, for_update=True)
    db.create_document_version.assert_called_once_with(
        media_id=42, content=text, prompt="prompt", analysis_content="analysis", safe_metadata=None
    )

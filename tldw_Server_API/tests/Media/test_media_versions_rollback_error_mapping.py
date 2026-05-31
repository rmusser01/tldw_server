import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.media import versions as media_versions_endpoint
from tldw_Server_API.app.api.v1.endpoints.media.versions import rollback_version
from tldw_Server_API.app.api.v1.schemas.media_request_models import VersionRollbackRequest
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
    "rollback exploded",
    "/private/tmp/media-versions-rollback.db",
)


def _database_failure() -> DatabaseError:
    return DatabaseError("driver failed /private/tmp/media-versions-rollback.db")


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


class _FakeRollbackTransaction:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _BrokenRollbackDb:
    def __init__(self, rollback_exc: Exception):
        self._rollback_exc = rollback_exc

    def transaction(self):
        return _FakeRollbackTransaction()

    def rollback_to_version(self, *, media_id: int, target_version_number: int):
        _ = (media_id, target_version_number)
        raise self._rollback_exc


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised_exc", "expected_status", "expected_detail"),
    [
        (InputError("invalid rollback target"), 400, "invalid rollback target"),
        (ConflictError("rollback conflict"), 409, "Conflict during rollback"),
        (_database_failure(), 500, "Database error during rollback"),
    ],
)
async def test_rollback_version_maps_db_errors(
    monkeypatch,
    raised_exc,
    expected_status,
    expected_detail,
):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await rollback_version(
            media_id=42,
            request_body=VersionRollbackRequest(version_number=1),
            db=_BrokenRollbackDb(raised_exc),
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail
    if expected_status == 500:
        _assert_sanitized_error_log(
            logger_stub,
            "Database error rolling back media {} to version {}",
        )


@pytest.mark.asyncio
async def test_rollback_version_sanitizes_unexpected_error_log(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await rollback_version(
            media_id=42,
            request_body=VersionRollbackRequest(version_number=1),
            db=_BrokenRollbackDb(
                RuntimeError("rollback exploded /private/tmp/media-versions-rollback.db"),
            ),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal server error during rollback"
    _assert_sanitized_error_log(
        logger_stub,
        "Unexpected error rolling back media {} to version {}",
    )

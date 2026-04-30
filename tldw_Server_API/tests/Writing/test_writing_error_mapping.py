"""Error-mapping characterization tests for writing endpoint helpers."""

from __future__ import annotations

import pytest
from fastapi import HTTPException, status

from tldw_Server_API.app.api.v1.endpoints import writing, writing_manuscripts
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    ConflictError,
    InputError,
)


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.debug_calls = []
        self.error_calls = []
        self.exception_calls = []
        self.warning_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exception_calls.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))


_SENSITIVE_LOG_MARKERS = (
    "private.scope",
    "rate limiter backend leaked",
    "sqlite backend exploded",
    "unexpected writing backend leaked",
    "/private/limits.db",
    "/private/writing.db",
    "wordcloud backend leaked",
    "wordcloud-private-id",
    "/private/wordcloud.db",
)


def _assert_sanitized_error_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.exception_calls == []
    assert logger_stub.error_calls == [((expected_message,), {})]

    rendered_calls = repr(logger_stub.error_calls)
    for marker in _SENSITIVE_LOG_MARKERS:
        assert marker not in rendered_calls


@pytest.mark.parametrize(
    ("module", "entity_label"),
    [
        (writing, "writing session"),
        (writing_manuscripts, "manuscript project"),
    ],
)
def test_writing_handle_db_errors_maps_input_error(module, entity_label):
    with pytest.raises(HTTPException) as exc_info:
        module._handle_db_errors(InputError("invalid writing payload"), entity_label)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc_info.value.detail == "invalid writing payload"


@pytest.mark.parametrize(
    ("module", "entity_label"),
    [
        (writing, "writing session"),
        (writing_manuscripts, "manuscript project"),
    ],
)
def test_writing_handle_db_errors_maps_base_db_error(module, entity_label):
    with pytest.raises(HTTPException) as exc_info:
        module._handle_db_errors(CharactersRAGDBError("sqlite backend exploded"), entity_label)

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == f"Database error while processing {entity_label}"


@pytest.mark.parametrize(
    ("module", "entity_label"),
    [
        (writing, "writing session"),
        (writing_manuscripts, "manuscript project"),
    ],
)
def test_writing_handle_db_errors_sanitizes_database_error_log(monkeypatch, module, entity_label):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(module, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        module._handle_db_errors(
            CharactersRAGDBError("sqlite backend exploded /private/writing.db"),
            entity_label,
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == f"Database error while processing {entity_label}"
    _assert_sanitized_error_log(logger_stub, "Database error while processing writing entity")


@pytest.mark.parametrize(
    ("module", "entity_label"),
    [
        (writing, "writing session"),
        (writing_manuscripts, "manuscript project"),
    ],
)
def test_writing_handle_db_errors_sanitizes_unexpected_error_log(monkeypatch, module, entity_label):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(module, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        module._handle_db_errors(
            RuntimeError("unexpected writing backend leaked /private/writing.db"),
            entity_label,
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == f"Unexpected error while processing {entity_label}"
    _assert_sanitized_error_log(logger_stub, "Unexpected error while processing writing entity")


@pytest.mark.asyncio
@pytest.mark.parametrize("module", [writing, writing_manuscripts])
async def test_writing_enforce_rate_limit_sanitizes_backend_failure_log(monkeypatch, module):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(module, "logger", logger_stub)

    class _FailingRateLimiter:
        async def check_user_rate_limit(self, user_id: int, scope: str):
            assert user_id == 881
            assert scope == "private.scope"
            raise RuntimeError("rate limiter backend leaked /private/limits.db")

    with pytest.raises(HTTPException) as exc_info:
        await module._enforce_rate_limit(_FailingRateLimiter(), 881, "private.scope")

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc_info.value.detail == "Rate limiter unavailable"
    assert exc_info.value.headers == {"Retry-After": "60"}
    _assert_sanitized_error_log(logger_stub, "Rate limiter check failed")


def test_writing_wordcloud_job_failure_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(writing, "logger", logger_stub)

    def _raise_compute_failure(_text, _options):
        raise RuntimeError("wordcloud backend leaked /private/wordcloud.db")

    class _DB:
        def __init__(self):
            self.status_calls = []
            self.result_calls = []

        def set_writing_wordcloud_status(self, *args):
            self.status_calls.append(args)

        def set_writing_wordcloud_result(self, *args, **kwargs):
            self.result_calls.append((args, kwargs))

    monkeypatch.setattr(writing, "_compute_wordcloud", _raise_compute_failure)
    db = _DB()

    writing._run_wordcloud_job(
        db,
        "wordcloud-private-id",
        "alpha beta",
        writing.WritingWordcloudOptions(),
    )

    assert db.result_calls == [
        (
            ("wordcloud-private-id",),
            {
                "status": writing.WORDCLOUD_STATUS_FAILED,
                "error": writing.WORDCLOUD_GENERATION_FAILED_DETAIL,
            },
        )
    ]
    _assert_sanitized_error_log(logger_stub, "Wordcloud job failed")


def test_writing_wordcloud_failure_persist_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(writing, "logger", logger_stub)

    def _raise_compute_failure(_text, _options):
        raise RuntimeError("wordcloud backend leaked /private/wordcloud.db")

    class _DB:
        def set_writing_wordcloud_status(self, *_args):
            return None

        def set_writing_wordcloud_result(self, *_args, **_kwargs):
            raise RuntimeError("wordcloud backend leaked /private/wordcloud.db")

    monkeypatch.setattr(writing, "_compute_wordcloud", _raise_compute_failure)

    writing._run_wordcloud_job(
        _DB(),
        "wordcloud-private-id",
        "alpha beta",
        writing.WritingWordcloudOptions(),
    )

    assert logger_stub.exception_calls == []
    assert logger_stub.error_calls == [
        (("Wordcloud job failed",), {}),
        (("Failed to persist wordcloud failure",), {}),
    ]
    rendered_calls = repr(logger_stub.error_calls)
    for marker in _SENSITIVE_LOG_MARKERS:
        assert marker not in rendered_calls


@pytest.mark.parametrize(
    ("module", "entity_label"),
    [
        (writing, "writing session"),
        (writing_manuscripts, "manuscript project"),
    ],
)
def test_writing_handle_db_errors_preserves_not_found_conflict(module, entity_label):
    with pytest.raises(HTTPException) as exc_info:
        module._handle_db_errors(ConflictError("record not found"), entity_label)

    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
    assert exc_info.value.detail == f"{entity_label} not found"


@pytest.mark.parametrize(
    ("module", "entity_label"),
    [
        (writing, "writing session"),
        (writing_manuscripts, "manuscript project"),
    ],
)
def test_writing_handle_db_errors_preserves_version_conflict(module, entity_label):
    with pytest.raises(HTTPException) as exc_info:
        module._handle_db_errors(ConflictError("version conflict"), entity_label)

    assert exc_info.value.status_code == status.HTTP_409_CONFLICT
    assert exc_info.value.detail == "version conflict"

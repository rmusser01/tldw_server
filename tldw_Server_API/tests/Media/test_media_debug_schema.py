from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.media import debug as debug_endpoint

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.error_calls = []
        self.warning_calls = []
        self.exception_calls = []

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exception_calls.append((args, kwargs))


_SENSITIVE_MARKERS = (
    "debug schema leaked",
    "/private/tmp/media-debug.db",
)


class _FailingDebugDb:
    @contextmanager
    def get_connection(self):
        raise RuntimeError("debug schema leaked /private/tmp/media-debug.db")


def _assert_sanitized_error_log(logger_stub: _LoggerStub) -> None:
    assert logger_stub.exception_calls == []
    assert logger_stub.error_calls
    assert [args[0] for args, _kwargs in logger_stub.error_calls if args] == ["debug_schema failed"]
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.error_calls)

    rendered_calls = repr(logger_stub.error_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


def test_media_debug_schema_basic(client_user_only) -> None:

    response = client_user_only.get("/api/v1/media/debug/schema")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, dict)
    assert "tables" in data
    assert "media_columns" in data
    assert "media_mods_columns" in data
    assert "media_count" in data

    assert isinstance(data["tables"], list)
    assert isinstance(data["media_columns"], list)
    assert isinstance(data["media_mods_columns"], list)
    assert isinstance(data["media_count"], int)


async def test_media_debug_schema_sanitizes_failure_log(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(debug_endpoint, "logger", logger_stub, raising=True)

    with pytest.raises(HTTPException) as exc_info:
        await debug_endpoint.debug_schema(db=_FailingDebugDb())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal error while reading media schema."
    _assert_sanitized_error_log(logger_stub)

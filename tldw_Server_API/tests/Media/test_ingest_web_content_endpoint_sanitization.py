from typing import Any

import pytest
from fastapi import BackgroundTasks, HTTPException

from tldw_Server_API.app.api.v1.endpoints.media import ingest_web_content as ingest_endpoint
from tldw_Server_API.app.api.v1.schemas.media_request_models import IngestWebContentRequest

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.error_calls = []
        self.info_calls = []
        self.exception_calls = []

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def info(self, *args, **kwargs):
        self.info_calls.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exception_calls.append((args, kwargs))


_SENSITIVE_MARKERS = (
    "ingest backend leaked",
    "/private/tmp/ingest-web-content.db",
)


def _assert_sanitized_error_log(logger_stub: _LoggerStub) -> None:
    assert logger_stub.exception_calls == []
    assert logger_stub.error_calls
    assert [args[0] for args, _kwargs in logger_stub.error_calls if args] == ["Web content ingestion failed"]
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.error_calls)

    rendered_calls = repr(logger_stub.error_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


async def test_ingest_web_content_sanitizes_orchestration_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    async def _raise_orchestration_failure(**_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("ingest backend leaked /private/tmp/ingest-web-content.db")

    monkeypatch.setattr(ingest_endpoint, "logger", logger_stub, raising=True)
    monkeypatch.setattr(
        ingest_endpoint,
        "ingest_web_content_orchestrate",
        _raise_orchestration_failure,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await ingest_endpoint.ingest_web_content(
            request=IngestWebContentRequest(urls=["https://example.com/"]),
            background_tasks=BackgroundTasks(),
            token=object(),
            db=object(),
            usage_log=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to ingest web content"
    _assert_sanitized_error_log(logger_stub)

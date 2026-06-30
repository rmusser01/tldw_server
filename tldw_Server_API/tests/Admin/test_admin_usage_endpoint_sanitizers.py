from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.admin import admin_usage

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.error_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.error_records.append((message, args, kwargs))


@pytest.mark.asyncio
async def test_run_usage_aggregate_sanitizes_generic_error_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _raise_backend_error(day: str | None):
        raise RuntimeError("usage aggregate backend exploded at /private/admin-usage.db")

    monkeypatch.setattr(admin_usage, "logger", logger_stub)
    monkeypatch.setattr(admin_usage.admin_usage_service, "run_usage_aggregate", _raise_backend_error)

    with pytest.raises(HTTPException) as exc_info:
        await admin_usage.run_usage_aggregate(day="2026-04-25")

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to run usage aggregate"
    assert logger_stub.error_records == [("Failed to run usage aggregate", (), {})]


@pytest.mark.asyncio
async def test_run_llm_usage_aggregate_sanitizes_generic_error_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _raise_backend_error(day: str | None):
        raise RuntimeError("LLM usage aggregate backend exploded at /private/llm-usage.db")

    monkeypatch.setattr(admin_usage, "logger", logger_stub)
    monkeypatch.setattr(admin_usage.admin_usage_service, "run_llm_usage_aggregate", _raise_backend_error)

    with pytest.raises(HTTPException) as exc_info:
        await admin_usage.run_llm_usage_aggregate(day="2026-04-25")

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to run LLM usage aggregate"
    assert logger_stub.error_records == [("Failed to run LLM usage aggregate", (), {})]

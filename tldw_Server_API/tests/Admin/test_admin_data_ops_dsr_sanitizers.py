from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.admin import admin_data_ops


class _LoggerStub:
    def __init__(self) -> None:
        self.error_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.error_records.append((message, args, kwargs))


async def _raise_build_dsr_repos() -> tuple[Any, Any]:
    raise RuntimeError("data subject backend exploded at /private/dsr.db")


@pytest.mark.asyncio
async def test_preview_data_subject_request_sanitizes_generic_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_data_ops, "logger", logger_stub)
    monkeypatch.setattr(admin_data_ops, "_build_dsr_repos", _raise_build_dsr_repos)

    with pytest.raises(HTTPException) as exc_info:
        await admin_data_ops.preview_data_subject_request(
            admin_data_ops.DataSubjectRequestPreviewRequest(
                requester_identifier="subject@example.test",
            ),
            request=object(),
            principal=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to preview data subject request"
    assert logger_stub.error_records == [("Failed to preview data subject request", (), {})]


@pytest.mark.asyncio
async def test_create_data_subject_request_sanitizes_generic_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_data_ops, "logger", logger_stub)
    monkeypatch.setattr(admin_data_ops, "_build_dsr_repos", _raise_build_dsr_repos)

    with pytest.raises(HTTPException) as exc_info:
        await admin_data_ops.create_data_subject_request(
            admin_data_ops.DataSubjectRequestCreateRequest(
                client_request_id="dsr-123",
                requester_identifier="subject@example.test",
                request_type="export",
            ),
            request=object(),
            principal=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to record data subject request"
    assert logger_stub.error_records == [("Failed to record data subject request", (), {})]


@pytest.mark.asyncio
async def test_list_data_subject_requests_sanitizes_generic_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_data_ops, "logger", logger_stub)
    monkeypatch.setattr(admin_data_ops, "_build_dsr_repos", _raise_build_dsr_repos)

    with pytest.raises(HTTPException) as exc_info:
        await admin_data_ops.list_data_subject_requests(
            limit=50,
            offset=0,
            principal=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list data subject requests"
    assert logger_stub.error_records == [("Failed to list data subject requests", (), {})]


@pytest.mark.asyncio
async def test_execute_data_subject_request_sanitizes_generic_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_data_ops, "logger", logger_stub)
    monkeypatch.setattr(admin_data_ops, "_build_dsr_repos", _raise_build_dsr_repos)

    with pytest.raises(HTTPException) as exc_info:
        await admin_data_ops.execute_data_subject_request(
            42,
            request=object(),
            principal=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to execute data subject request"
    assert logger_stub.error_records == [("Failed to execute data subject request", (), {})]

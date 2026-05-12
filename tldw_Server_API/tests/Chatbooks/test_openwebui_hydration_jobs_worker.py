from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.Chatbooks.openwebui_hydration_jobs import (
    OPENWEBUI_ATTACHMENT_HYDRATION_JOB_TYPE,
)
from tldw_Server_API.app.core.Chatbooks.services import jobs_worker


pytestmark = pytest.mark.unit


class _FakeHydrationService:
    def __init__(self, *, result: dict[str, Any] | None = None, exc: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result or {
            "summary": {
                "referenced_files": 2,
                "resolved_files": 2,
                "hydrated_images": 1,
                "registered_media_files": 1,
                "already_hydrated": 0,
                "missing_files": 0,
                "unsupported_files": 0,
                "failed_files": 0,
                "processed_files": 0,
            },
            "warnings": [],
        }
        self.exc = exc

    def run_openwebui_attachment_hydration(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.result


@pytest.mark.asyncio
async def test_handle_job_dispatches_openwebui_hydration(monkeypatch):
    service = _FakeHydrationService()
    monkeypatch.setattr(jobs_worker, "_get_service", lambda user_id: service)

    result = await jobs_worker._handle_job(
        {
            "id": 42,
            "uuid": "job-uuid-42",
            "job_type": OPENWEBUI_ATTACHMENT_HYDRATION_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {
                "openwebui_data_root": "/srv/openwebui",
                "scope": {
                    "conversation_ids": ["conv-a"],
                    "source_user_id": "ow-user",
                },
                "process_supported_files": True,
            },
        }
    )

    assert result["referenced_files"] == 2
    assert result["hydrated_images"] == 1
    assert result["registered_media_files"] == 1
    assert service.calls == [
        {
            "openwebui_data_root": "/srv/openwebui",
            "scope": {
                "conversation_ids": ["conv-a"],
                "source_user_id": "ow-user",
            },
            "process_supported_files": True,
            "job_id": "job-uuid-42",
        }
    ]


@pytest.mark.asyncio
async def test_handle_openwebui_hydration_requires_data_root(monkeypatch):
    service = _FakeHydrationService()
    monkeypatch.setattr(jobs_worker, "_get_service", lambda user_id: service)

    with pytest.raises(jobs_worker.ChatbooksJobError, match="Missing openwebui_data_root") as exc_info:
        await jobs_worker._handle_job(
            {
                "id": 43,
                "job_type": OPENWEBUI_ATTACHMENT_HYDRATION_JOB_TYPE,
                "owner_user_id": "7",
                "payload": {"scope": {"conversation_ids": ["conv-a"]}},
            }
        )

    assert exc_info.value.retryable is False
    assert service.calls == []


@pytest.mark.asyncio
async def test_handle_openwebui_hydration_root_validation_is_nonretryable(monkeypatch):
    service = _FakeHydrationService(exc=ValueError("OpenWebUI data root must contain webui.db."))
    monkeypatch.setattr(jobs_worker, "_get_service", lambda user_id: service)

    with pytest.raises(jobs_worker.ChatbooksJobError, match="OpenWebUI data root must contain webui.db") as exc_info:
        await jobs_worker._handle_job(
            {
                "id": 44,
                "job_type": OPENWEBUI_ATTACHMENT_HYDRATION_JOB_TYPE,
                "owner_user_id": "7",
                "payload": {
                    "openwebui_data_root": "/srv/openwebui",
                    "scope": {"conversation_ids": ["conv-a"]},
                },
            }
        )

    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_handle_openwebui_hydration_caps_and_redacts_warnings(monkeypatch):
    service = _FakeHydrationService(
        result={
            "summary": {
                "referenced_files": 101,
                "resolved_files": 0,
                "hydrated_images": 0,
                "registered_media_files": 0,
                "already_hydrated": 0,
                "missing_files": 0,
                "unsupported_files": 0,
                "failed_files": 101,
                "processed_files": 0,
            },
            "warnings": [f"Failed /private/openwebui/uploads/file-{index}.png" for index in range(101)],
        }
    )
    monkeypatch.setattr(jobs_worker, "_get_service", lambda user_id: service)

    result = await jobs_worker._handle_job(
        {
            "id": 45,
            "job_type": OPENWEBUI_ATTACHMENT_HYDRATION_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {
                "openwebui_data_root": "/srv/openwebui",
                "scope": {"conversation_ids": ["conv-a"]},
            },
        }
    )

    assert len(result["warnings"]) == 100
    assert "/private/openwebui" not in repr(result["warnings"])


def test_redact_hydration_warning_redacts_windows_and_unc_paths():
    warning = jobs_worker._redact_hydration_warning(
        r"Failed C:\OpenWebUI\uploads\file.png and \\server\share\secret.pdf and /srv/openwebui/file.txt"
    )

    assert "C:\\OpenWebUI" not in warning
    assert "\\\\server\\share" not in warning
    assert "/srv/openwebui" not in warning
    assert warning.count("[redacted-path]") == 3


@pytest.mark.asyncio
async def test_handle_job_unsupported_job_type_still_errors(monkeypatch):
    service = _FakeHydrationService()
    monkeypatch.setattr(jobs_worker, "_get_service", lambda user_id: service)

    with pytest.raises(jobs_worker.ChatbooksJobError, match="Unsupported chatbooks job action: unknown"):
        await jobs_worker._handle_job(
            {
                "id": 46,
                "job_type": "unknown",
                "owner_user_id": "7",
                "payload": {"chatbooks_job_id": "legacy-job"},
            }
        )

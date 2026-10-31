"""Cancellation and storage-quota behaviour of the ingestion persistence layer.

_PERSISTENCE_NONCRITICAL_EXCEPTIONS once listed asyncio.CancelledError (a
BaseException), so a client disconnect mid-/media/add was logged as a "non-fatal"
quota-check failure and the request kept writing for a client that was gone.

The tuple deliberately keeps HTTPException: inside process_document_like_item a
per-item rejection (SSRF block, per-URL quota) is reported as that item's Error
result, which is the batch contract the /media/add gather, the ingest-jobs worker
and the reading service all consume. Request-level rejections (the aggregate
upload quota) re-raise before the tuple catch and reach the client as 413.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import BackgroundTasks, HTTPException

from tldw_Server_API.app.core.Ingestion_Media_Processing import Upload_Sink as upload_sink
from tldw_Server_API.app.core.Ingestion_Media_Processing import (
    download_utils,
    input_sourcing,
    persistence,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import ValidationResult
from tldw_Server_API.app.services import storage_quota_service

pytestmark = pytest.mark.unit

_OVER_QUOTA_INFO = {
    "current_usage_mb": 10,
    "new_size_mb": 1,
    "quota_mb": 10,
    "available_mb": 0,
}


class _FakeDB:
    db_path_str = ":memory:"
    client_id = "test-client"


class _RejectingQuotaService:
    async def check_quota(self, user_id: int, new_bytes: int, raise_on_exceed: bool = False):
        return False, dict(_OVER_QUOTA_INFO)


class _BlockingQuotaService:
    """Parks inside check_quota so the test can cancel the request mid-persistence."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def check_quota(self, user_id: int, new_bytes: int, raise_on_exceed: bool = False):
        self.entered.set()
        await asyncio.Event().wait()


def _patch_upload(monkeypatch: pytest.MonkeyPatch, quota_service: Any) -> list[str]:
    processed: list[str] = []

    async def fake_save_uploaded_files(_files: list[Any], temp_dir: Path, **_kwargs: Any):
        path = Path(temp_dir) / "doc.txt"
        path.write_text("hello", encoding="utf-8")
        return [{"path": path, "original_filename": "doc.txt"}], []

    async def fake_process_doc_item(*, item_input_ref: str, **_kwargs: Any) -> dict[str, Any]:
        processed.append(item_input_ref)
        return {"status": "Success", "input_ref": item_input_ref, "db_id": 1}

    monkeypatch.setattr(input_sourcing, "save_uploaded_files", fake_save_uploaded_files)
    monkeypatch.setattr(persistence, "process_document_like_item", fake_process_doc_item)
    monkeypatch.setattr(storage_quota_service, "get_storage_quota_service", lambda: quota_service)
    return processed


def _orchestrate() -> Any:
    form_data = SimpleNamespace(
        media_type="document",
        urls=[],
        keep_original_file=False,
        perform_chunking=False,
        perform_analysis=False,
        generate_embeddings=False,
    )
    return persistence.add_media_orchestrate(
        background_tasks=BackgroundTasks(),
        form_data=form_data,
        files=[object()],
        db=_FakeDB(),
        current_user=SimpleNamespace(id=1),
        usage_log=SimpleNamespace(log_event=lambda *_args, **_kwargs: None),
    )


@pytest.mark.asyncio
async def test_cancelling_add_media_mid_persistence_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    quota_service = _BlockingQuotaService()
    processed = _patch_upload(monkeypatch, quota_service)

    task = asyncio.create_task(_orchestrate())
    await asyncio.wait_for(quota_service.entered.wait(), timeout=10)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=10)
    assert processed == [], "ingestion kept going after the request was cancelled"


@pytest.mark.asyncio
async def test_over_quota_upload_is_rejected_with_413(monkeypatch: pytest.MonkeyPatch) -> None:
    processed = _patch_upload(monkeypatch, _RejectingQuotaService())

    with pytest.raises(HTTPException) as exc_info:
        await _orchestrate()

    assert exc_info.value.status_code == 413
    assert "Storage quota exceeded" in str(exc_info.value.detail)
    assert processed == []


@pytest.mark.asyncio
async def test_over_quota_url_item_is_reported_as_that_items_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://example.com/file.txt"
    downloaded_file = tmp_path / "downloaded.txt"
    downloaded_file.write_text("hello from url", encoding="utf-8")

    async def _fake_download_url_async(**_kwargs: Any) -> Path:
        return downloaded_file

    monkeypatch.setattr(download_utils, "download_url_async", _fake_download_url_async)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Security.url_validation.assert_url_safe",
        lambda _url: None,
    )
    monkeypatch.setattr(
        upload_sink,
        "process_and_validate_file",
        lambda *_a, **_k: ValidationResult(True, file_path=downloaded_file),
    )
    monkeypatch.setattr(
        storage_quota_service,
        "get_storage_quota_service",
        lambda: _RejectingQuotaService(),
    )

    result = await persistence.process_document_like_item(
        item_input_ref=url,
        processing_source=url,
        media_type="document",
        is_url=True,
        form_data=SimpleNamespace(title=None, author=None, keywords=None),
        chunk_options=None,
        temp_dir=tmp_path,
        loop=asyncio.get_running_loop(),
        db_path=":memory:",
        client_id="test-client",
        user_id=1,
    )

    assert result["status"] == "Error"
    assert "Storage quota exceeded" in result["error"]

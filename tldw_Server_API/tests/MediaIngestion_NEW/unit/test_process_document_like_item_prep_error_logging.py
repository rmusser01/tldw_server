from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing import download_utils, persistence

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_process_document_like_item_logs_prep_error_context(tmp_path, monkeypatch):
    async def _boom_download(**_kwargs: Any):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(download_utils, "download_url_async", _boom_download, raising=True)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Security.url_validation.assert_url_safe",
        lambda _url: None,
    )

    captured: dict[str, Any] = {}

    def _capture_exception(message: str, *args: Any, **kwargs: Any) -> None:
        captured["message"] = message
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(persistence.logger, "exception", _capture_exception)

    form_data = SimpleNamespace(
        title=None,
        author=None,
        keywords=None,
        perform_chunking=False,
        perform_analysis=False,
        api_name=None,
        custom_prompt=None,
        system_prompt=None,
        summarize_recursively=False,
    )

    url = "https://example.com/file.pdf"
    result = await persistence.process_document_like_item(
        item_input_ref=url,
        processing_source=url,
        media_type="pdf",
        is_url=True,
        form_data=form_data,
        chunk_options=None,
        temp_dir=tmp_path,
        loop=asyncio.get_running_loop(),
        db_path=":memory:",
        client_id="test-client",
        user_id=None,
    )

    assert result.get("status") == "Error"
    assert "File preparation/download failed" in str(result.get("error", ""))
    assert "No such file or directory" in str(result.get("error", ""))

    assert "context:" in captured.get("message", "")
    log_args = captured.get("args", ())
    assert log_args[2] == "FileNotFoundError"
    assert log_args[3] is True
    assert log_args[4] == tmp_path
    assert log_args[5] is True
    assert log_args[6] == url
    assert log_args[8] is None
    assert log_args[9] is None
    assert log_args[10] is None
    assert log_args[11] is None


@pytest.mark.asyncio
async def test_email_prep_error_retains_safe_stack_and_correlatable_failure_id(tmp_path, monkeypatch):
    from uuid import UUID

    from loguru import logger

    sentinel = "SensitiveEmailBodyFilenameAndCredential"

    async def fail_download(**_kwargs):
        raise FileNotFoundError(sentinel)

    monkeypatch.setattr(download_utils, "download_url_async", fail_download)
    monkeypatch.setattr("tldw_Server_API.app.core.Security.url_validation.assert_url_safe", lambda _url: None)
    output = []
    token = logger.add(
        output.append,
        level="ERROR",
        format="{message} {extra}",
        backtrace=True,
        diagnose=True,
        filter=lambda record: record["name"] == persistence.__name__,
    )
    try:
        result = await persistence.process_document_like_item(
            item_input_ref=sentinel,
            processing_source=f"https://example.test/{sentinel}.eml",
            media_type="email",
            is_url=True,
            form_data=SimpleNamespace(),
            chunk_options=None,
            temp_dir=tmp_path,
            loop=asyncio.get_running_loop(),
            db_path=":memory:",
            client_id="synthetic",
        )
    finally:
        logger.remove(token)
    assert result["status"] == "Error"
    record = output[-1].record
    identifier = record["extra"]["correlation_id"]
    assert UUID(identifier).version == 4 and identifier in result["error"]
    assert sentinel not in result["error"]
    assert identifier in record["message"] and "fail_download:" in record["message"]
    assert any(frame.startswith("fail_download:") for frame in record["extra"]["stack_frames"])
    assert record["extra"]["error_type"] == "FileNotFoundError"
    assert sentinel not in " ".join(str(message) for message in output)

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing import download_utils, persistence
from tldw_Server_API.app.core.Ingestion_Media_Processing import Upload_Sink as upload_sink
from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import (
    DEFAULT_MEDIA_TYPE_CONFIG,
    ValidationResult,
)


class _MetricsCapture:
    def __init__(self) -> None:
        self.increment_calls: list[tuple[str, float, dict[str, Any] | None]] = []

    def increment(
        self,
        metric_name: str,
        value: float = 1,
        labels: dict[str, Any] | None = None,
    ) -> None:
        self.increment_calls.append((metric_name, value, labels))

    def observe(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_process_document_like_item_url_oversize_rejected(tmp_path, monkeypatch):
    url = "http://example.com/file.pdf"

    body = b"x" * (1024 * 1024 + 1)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method.upper() == "GET":
            return httpx.Response(
                200,
                headers={
                    "content-type": "application/pdf",
                    "content-length": str(len(body)),
                },
                content=body,
            )
        return httpx.Response(405)

    transport = httpx.MockTransport(handler)

    def _mk_async_client(**kwargs):
        timeout = kwargs.get("timeout", 10.0)
        return httpx.AsyncClient(timeout=timeout, transport=transport)

    monkeypatch.setattr(download_utils, "_create_async_client", _mk_async_client, raising=True)
    monkeypatch.setenv("EGRESS_ALLOWLIST", "example.com")
    monkeypatch.setitem(DEFAULT_MEDIA_TYPE_CONFIG["pdf"], "max_size_mb", 1)

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
        pdf_parsing_engine="pymupdf4llm",
    )

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
    assert "exceeds maximum allowed size" in str(result.get("error", "")).lower()


@pytest.mark.asyncio
async def test_process_document_like_item_url_post_download_validation_rejected(
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.endpoints import media as media_endpoints

    url = "https://example.com/file.txt"
    downloaded_file = tmp_path / "downloaded.txt"
    downloaded_file.write_text("hello from url", encoding="utf-8")
    metrics = _MetricsCapture()

    async def _fake_download_url_async(**_kwargs: Any):
        return downloaded_file

    def _fake_process_and_validate_file(*_args: Any, **_kwargs: Any) -> ValidationResult:
        return ValidationResult(
            False,
            issues=["blocked by validator"],
            file_path=downloaded_file,
        )

    def _should_not_process(*_args: Any, **_kwargs: Any):
        raise AssertionError("processor should not run when URL validation fails")

    monkeypatch.setattr(download_utils, "download_url_async", _fake_download_url_async, raising=True)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Security.url_validation.assert_url_safe",
        lambda _url: None,
    )
    monkeypatch.setattr(persistence, "get_metrics_registry", lambda: metrics)
    monkeypatch.setattr(upload_sink, "process_and_validate_file", _fake_process_and_validate_file, raising=True)
    monkeypatch.setattr(media_endpoints, "process_document_content", _should_not_process, raising=True)

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

    result = await persistence.process_document_like_item(
        item_input_ref=url,
        processing_source=url,
        media_type="document",
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
    assert "downloaded file failed validation" in str(result.get("error", "")).lower()
    assert "blocked by validator" in str(result.get("error", "")).lower()
    assert (
        "ingestion_validation_failures_total",
        1,
        {"reason": "validator_rejected", "path_kind": "url"},
    ) in metrics.increment_calls


@pytest.mark.asyncio
async def test_process_document_like_item_url_post_download_validation_success_path(
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.endpoints import media as media_endpoints

    url = "https://example.com/file.txt"
    downloaded_file = tmp_path / "downloaded-ok.txt"
    downloaded_file.write_text("hello from validated url", encoding="utf-8")
    captured: dict[str, Any] = {}

    async def _fake_download_url_async(**kwargs: Any):
        captured["download_kwargs"] = kwargs
        return downloaded_file

    def _fake_process_and_validate_file(*args: Any, **_kwargs: Any) -> ValidationResult:
        captured["validated_path"] = args[0] if args else None
        return ValidationResult(True, file_path=downloaded_file)

    def _fake_process_document_content(**_kwargs: Any):
        return {
            "status": "Success",
            "content": "validated content",
            "metadata": {"title": "Validated URL Doc"},
            "analysis": None,
            "summary": None,
            "analysis_details": {},
            "error": None,
            "warnings": None,
        }

    async def _fake_persist_doc_item_and_children(**kwargs: Any) -> None:
        captured["persisted_metadata"] = kwargs["final_result"]["metadata"]
        return None

    async def _fake_extract_claims_if_requested(*_args: Any, **_kwargs: Any):
        return None

    monkeypatch.setattr(download_utils, "download_url_async", _fake_download_url_async, raising=True)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Security.url_validation.assert_url_safe",
        lambda _url: None,
    )
    monkeypatch.setattr(upload_sink, "process_and_validate_file", _fake_process_and_validate_file, raising=True)
    monkeypatch.setattr(media_endpoints, "process_document_content", _fake_process_document_content, raising=True)
    monkeypatch.setattr(persistence, "persist_doc_item_and_children", _fake_persist_doc_item_and_children, raising=True)
    monkeypatch.setattr(persistence, "extract_claims_if_requested", _fake_extract_claims_if_requested, raising=True)

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

    result = await persistence.process_document_like_item(
        item_input_ref=url,
        processing_source=url,
        media_type="document",
        is_url=True,
        form_data=form_data,
        chunk_options=None,
        temp_dir=tmp_path,
        loop=asyncio.get_running_loop(),
        db_path=":memory:",
        client_id="test-client",
        user_id=None,
        max_download_bytes=50 * 1024 * 1024,
        allowed_download_content_types={"application/pdf"},
        trusted_source_metadata_by_url={
            url: {"doi": "10.1000/example", "title": "Trusted Discovery Title"}
        },
    )

    assert captured.get("validated_path") == downloaded_file
    assert captured["download_kwargs"]["max_bytes"] == 50 * 1024 * 1024
    assert captured["download_kwargs"]["allowed_content_types"] == {"application/pdf"}
    assert captured["persisted_metadata"]["doi"] == "10.1000/example"
    assert captured["persisted_metadata"]["title"] == "Trusted Discovery Title"
    assert result.get("status") == "Success"
    assert result.get("processing_source") == str(downloaded_file)


@pytest.mark.asyncio
async def test_process_document_like_item_document_url_passes_html_xml_allowlist(
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.endpoints import media as media_endpoints

    url = "https://example.com/thread/12345"
    downloaded_file = tmp_path / "downloaded-thread.html"
    downloaded_file.write_text("<html><body>thread</body></html>", encoding="utf-8")
    captured: dict[str, Any] = {}

    async def _fake_download_url_async(**kwargs: Any):
        captured["allowed_extensions"] = set(kwargs.get("allowed_extensions") or [])
        captured["check_extension"] = kwargs.get("check_extension")
        return downloaded_file

    def _fake_process_and_validate_file(*_args: Any, **_kwargs: Any) -> ValidationResult:
        return ValidationResult(True, file_path=downloaded_file)

    def _fake_process_document_content(**_kwargs: Any):
        return {
            "status": "Success",
            "content": "validated html content",
            "metadata": {"title": "Validated HTML URL Doc"},
            "analysis": None,
            "summary": None,
            "analysis_details": {},
            "error": None,
            "warnings": None,
        }

    async def _fake_persist_doc_item_and_children(**_kwargs: Any) -> None:
        return None

    async def _fake_extract_claims_if_requested(*_args: Any, **_kwargs: Any):
        return None

    monkeypatch.setattr(download_utils, "download_url_async", _fake_download_url_async, raising=True)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Security.url_validation.assert_url_safe",
        lambda _url: None,
    )
    monkeypatch.setattr(upload_sink, "process_and_validate_file", _fake_process_and_validate_file, raising=True)
    monkeypatch.setattr(media_endpoints, "process_document_content", _fake_process_document_content, raising=True)
    monkeypatch.setattr(persistence, "persist_doc_item_and_children", _fake_persist_doc_item_and_children, raising=True)
    monkeypatch.setattr(persistence, "extract_claims_if_requested", _fake_extract_claims_if_requested, raising=True)

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

    result = await persistence.process_document_like_item(
        item_input_ref=url,
        processing_source=url,
        media_type="document",
        is_url=True,
        form_data=form_data,
        chunk_options=None,
        temp_dir=tmp_path,
        loop=asyncio.get_running_loop(),
        db_path=":memory:",
        client_id="test-client",
        user_id=None,
    )

    allowed_extensions = captured.get("allowed_extensions")
    assert captured.get("check_extension") is True
    assert isinstance(allowed_extensions, set)
    assert {".html", ".htm", ".xml"}.issubset(allowed_extensions)
    assert result.get("status") == "Success"
    assert result.get("processing_source") == str(downloaded_file)


@pytest.mark.asyncio
async def test_process_document_like_item_email_archive_url_uses_archive_content_validation(
    tmp_path,
    monkeypatch,
):
    url = "https://example.com/emails.zip"
    downloaded_file = tmp_path / "emails.zip"
    downloaded_file.write_bytes(b"fake zip bytes")
    called: dict[str, Any] = {"validator_dispatch": False}

    async def _fake_download_url_async(**_kwargs: Any):
        return downloaded_file

    def _fake_process_and_validate_file(*_args: Any, **_kwargs: Any) -> ValidationResult:
        called["validator_dispatch"] = True
        return ValidationResult(
            False,
            issues=["archive content rejected"],
            file_path=downloaded_file,
        )

    monkeypatch.setattr(
        download_utils,
        "download_url_async",
        _fake_download_url_async,
        raising=True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Security.url_validation.assert_url_safe",
        lambda _url: None,
    )
    monkeypatch.setattr(
        upload_sink,
        "process_and_validate_file",
        _fake_process_and_validate_file,
        raising=True,
    )
    monkeypatch.setattr(
        persistence,
        "loaded_config_data",
        {"media_processing": {"validate_email_archive_contents": True}},
        raising=False,
    )

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
        accept_archives=True,
        accept_mbox=False,
        accept_pst=False,
        ingest_attachments=False,
        max_depth=1,
    )

    result = await persistence.process_document_like_item(
        item_input_ref=url,
        processing_source=url,
        media_type="email",
        is_url=True,
        form_data=form_data,
        chunk_options=None,
        temp_dir=tmp_path,
        loop=asyncio.get_running_loop(),
        db_path=":memory:",
        client_id="test-client",
        user_id=None,
    )

    assert called["validator_dispatch"] is True
    assert result.get("status") == "Error"
    assert "downloaded file failed validation" in str(result.get("error", "")).lower()
    assert "archive content rejected" in str(result.get("error", "")).lower()

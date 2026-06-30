from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.services import _placeholder_guard
from tldw_Server_API.app.services import xml_processing_service


class _NamedTempFileStub:
    """Minimal context-manager stub for tempfile.NamedTemporaryFile."""

    def __init__(self, path: Path):
        self.name = str(path)
        self._fh = path.open("wb")

    def write(self, data: bytes) -> int:
        return self._fh.write(data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._fh.close()
        return False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_xml_task_cleans_temp_file_on_success(monkeypatch, tmp_path):
    temp_path = tmp_path / "xml-success.xml"

    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.setenv("PLACEHOLDER_SERVICES_ENABLED", "1")
    monkeypatch.setattr(
        _placeholder_guard,
        "get_settings",
        lambda: SimpleNamespace(PLACEHOLDER_SERVICES_ENABLED=True),
    )
    monkeypatch.setattr(
        xml_processing_service.tempfile,
        "NamedTemporaryFile",
        lambda suffix, delete: _NamedTempFileStub(temp_path),
    )
    monkeypatch.setattr(
        xml_processing_service,
        "improved_chunking_process",
        lambda _text, _opts: [{"text": "chunk-1", "metadata": {"source": "test"}}],
    )

    result = await xml_processing_service.process_xml_task(
        file_bytes=b"<root><child>ok</child></root>",
        filename="sample.xml",
        title="Title",
        author="Author",
        keywords=["k1"],
        system_prompt=None,
        custom_prompt=None,
        auto_summarize=False,
        api_name=None,
        api_key=None,
    )

    assert result["info_dict"]["file_type"] == "xml"
    assert result["segments"][0]["Text"] == "chunk-1"
    assert not temp_path.exists()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_xml_task_cleans_temp_file_on_parse_error(monkeypatch, tmp_path):
    temp_path = tmp_path / "xml-invalid.xml"

    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.setenv("PLACEHOLDER_SERVICES_ENABLED", "1")
    monkeypatch.setattr(
        _placeholder_guard,
        "get_settings",
        lambda: SimpleNamespace(PLACEHOLDER_SERVICES_ENABLED=True),
    )
    monkeypatch.setattr(
        xml_processing_service.tempfile,
        "NamedTemporaryFile",
        lambda suffix, delete: _NamedTempFileStub(temp_path),
    )

    with pytest.raises(HTTPException) as exc_info:
        await xml_processing_service.process_xml_task(
            file_bytes=b"<root><broken>",
            filename="broken.xml",
            title=None,
            author=None,
            keywords=[],
            system_prompt=None,
            custom_prompt=None,
            auto_summarize=False,
            api_name=None,
            api_key=None,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid XML"
    assert "mismatched tag" not in str(exc_info.value.detail)
    assert "line" not in str(exc_info.value.detail)
    assert not temp_path.exists()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_xml_task_sanitizes_unexpected_processing_error(monkeypatch, tmp_path):
    temp_path = tmp_path / "xml-chunking-error.xml"

    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.setenv("PLACEHOLDER_SERVICES_ENABLED", "1")
    monkeypatch.setattr(
        _placeholder_guard,
        "get_settings",
        lambda: SimpleNamespace(PLACEHOLDER_SERVICES_ENABLED=True),
    )
    monkeypatch.setattr(
        xml_processing_service.tempfile,
        "NamedTemporaryFile",
        lambda suffix, delete: _NamedTempFileStub(temp_path),
    )

    def raise_chunking_error(_text, _opts):
        raise RuntimeError("chunker leaked /private/xml/cache/token")

    monkeypatch.setattr(
        xml_processing_service,
        "improved_chunking_process",
        raise_chunking_error,
    )

    with pytest.raises(HTTPException) as exc_info:
        await xml_processing_service.process_xml_task(
            file_bytes=b"<root><child>ok</child></root>",
            filename="sample.xml",
            title=None,
            author=None,
            keywords=[],
            system_prompt=None,
            custom_prompt=None,
            auto_summarize=False,
            api_name=None,
            api_key=None,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to process XML file"
    assert not temp_path.exists()

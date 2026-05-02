import io
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.media import document_figures as figures_endpoint
from tldw_Server_API.app.core.Storage.storage_interface import StorageError

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.debug_calls = []
        self.error_calls = []
        self.info_calls = []
        self.warning_calls = []
        self.exception_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def info(self, *args, **kwargs):
        self.info_calls.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exception_calls.append((args, kwargs))


_SENSITIVE_MARKERS = (
    "figures db leaked",
    "figures storage leaked",
    "figures extraction leaked",
    "figures helper leaked",
    "figures image leaked",
    "/private/tmp/document-figures.db",
    "/private/tmp/document-figures-storage",
    "/private/tmp/document-figures-extract",
    "/private/tmp/pdf-figures",
    "/private/tmp/pdf-image",
)


def _patch_logger(monkeypatch: pytest.MonkeyPatch) -> _LoggerStub:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(figures_endpoint, "logger", logger_stub, raising=True)
    return logger_stub


def _assert_sanitized_error_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.exception_calls == []
    assert logger_stub.error_calls

    matching_messages = [args[0] for args, _kwargs in logger_stub.error_calls if args]
    assert expected_message in matching_messages
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.error_calls)

    rendered_calls = repr(logger_stub.error_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


def _assert_sanitized_debug_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.exception_calls == []
    assert logger_stub.debug_calls

    matching_messages = [args[0] for args, _kwargs in logger_stub.debug_calls if args]
    assert expected_message in matching_messages

    rendered_calls = repr(logger_stub.debug_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


def _user() -> object:
    return type("User", (), {"id": 1})()


class _DbFailure:
    def get_media_by_id(
        self,
        _media_id: int,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any]:
        raise RuntimeError("figures db leaked /private/tmp/document-figures.db")


class _PdfDb:
    def get_media_by_id(
        self,
        media_id: int,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any]:
        return {
            "id": media_id,
            "title": "PDF",
            "type": "pdf",
            "deleted": int(include_deleted),
            "is_trash": int(include_trash),
        }

    def get_media_file(self, _media_id: int, _file_kind: str) -> dict[str, str]:
        return {
            "storage_path": "user_1/media/private/original.pdf",
            "mime_type": "application/pdf",
        }


class _FailingStorage:
    async def exists(self, _storage_path: str) -> bool:
        raise StorageError("figures storage leaked /private/tmp/document-figures-storage")


class _WorkingStorage:
    async def exists(self, _storage_path: str) -> bool:
        return True

    async def get_size(self, _storage_path: str) -> int:
        return 1024

    async def retrieve(self, _storage_path: str) -> io.BytesIO:
        return io.BytesIO(b"%PDF-1.4\nfake pdf")


async def test_document_figures_sanitizes_database_error_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)
    monkeypatch.setattr(figures_endpoint, "_check_pymupdf_available", lambda: True)

    with pytest.raises(HTTPException) as exc_info:
        await figures_endpoint.get_document_figures(
            media_id=42,
            db=_DbFailure(),
            current_user=_user(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error while fetching media item"
    _assert_sanitized_error_log(logger_stub, "Database error fetching media item")


async def test_document_figures_sanitizes_storage_error_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)
    monkeypatch.setattr(figures_endpoint, "_check_pymupdf_available", lambda: True)
    monkeypatch.setattr(figures_endpoint, "get_storage_backend", lambda: _FailingStorage())

    with pytest.raises(HTTPException) as exc_info:
        await figures_endpoint.get_document_figures(
            media_id=42,
            db=_PdfDb(),
            current_user=_user(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Error accessing file storage"
    _assert_sanitized_error_log(logger_stub, "Storage error retrieving file for figures")


async def test_document_figures_sanitizes_extraction_error_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)

    def _raise_extract_failure(_pdf_file, _min_size):
        raise RuntimeError("figures extraction leaked /private/tmp/document-figures-extract")

    monkeypatch.setattr(figures_endpoint, "_check_pymupdf_available", lambda: True)
    monkeypatch.setattr(figures_endpoint, "get_storage_backend", lambda: _WorkingStorage())
    monkeypatch.setattr(figures_endpoint, "_extract_pdf_figures", _raise_extract_failure)

    with pytest.raises(HTTPException) as exc_info:
        await figures_endpoint.get_document_figures(
            media_id=42,
            min_size=50,
            db=_PdfDb(),
            current_user=_user(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Error extracting document figures"
    _assert_sanitized_error_log(logger_stub, "Error extracting document figures")


def test_extract_pdf_figures_sanitizes_open_failure_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)

    with patch(
        "pymupdf.open",
        side_effect=RuntimeError("figures helper leaked /private/tmp/pdf-figures"),
    ):
        figures = figures_endpoint._extract_pdf_figures(b"%PDF-1.4\nfake pdf")

    assert figures == []
    _assert_sanitized_error_log(logger_stub, "Error extracting PDF figures")


def test_extract_pdf_figures_sanitizes_unreadable_image_debug_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)
    page = MagicMock()
    page.get_images.return_value = [(7,)]
    doc = MagicMock()
    doc.page_count = 1
    doc.__getitem__.return_value = page
    doc.extract_image.side_effect = RuntimeError("figures image leaked /private/tmp/pdf-image")

    with patch("pymupdf.open", return_value=doc):
        figures = figures_endpoint._extract_pdf_figures(b"%PDF-1.4\nfake pdf")

    assert figures == []
    _assert_sanitized_debug_log(
        logger_stub,
        "Skipping unreadable extracted image xref {} on page {}",
    )

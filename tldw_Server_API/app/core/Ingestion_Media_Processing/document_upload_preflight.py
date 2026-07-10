"""Document upload preflight decisions for chat attachments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

from tldw_Server_API.app.api.v1.schemas.document_upload_processing import (
    DocumentModeCapability,
    DocumentProcessingMode,
    DocumentProcessingStatus,
    DocumentUploadPreflightFile,
    DocumentUploadPreflightItem,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import (
    list_backends as _list_backends,
)

SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".docx",
    ".rtf",
    ".html",
    ".htm",
    ".xhtml",
    ".xml",
    ".json",
}
SUPPORTED_EBOOK_EXTENSIONS = {".epub"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}

DEFAULT_MAX_CHAT_UPLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_CHAT_UPLOAD_PAGES = 200
DEFAULT_MAX_DIRECT_CHAT_TOKENS = 24_000

MediaType = Literal["pdf", "document", "ebook", "unsupported"]
ListOcrBackends = Callable[[], dict[str, Any]]


def preflight_document_upload_files(
    files: list[DocumentUploadPreflightFile],
    list_ocr_backends: ListOcrBackends = _list_backends,
) -> list[DocumentUploadPreflightItem]:
    """Return per-file processing capabilities and defaults."""
    return [_preflight_file(file, list_ocr_backends) for file in files]


def _media_type_for(filename: str) -> MediaType:
    """Classify a filename into the supported upload media groups."""
    suffix = Path(filename).suffix.lower()
    if suffix in SUPPORTED_PDF_EXTENSIONS:
        return "pdf"
    if suffix in SUPPORTED_DOCUMENT_EXTENSIONS:
        return "document"
    if suffix in SUPPORTED_EBOOK_EXTENSIONS:
        return "ebook"
    return "unsupported"


def _capability(
    available: bool,
    status_value: DocumentProcessingStatus,
    reason: str | None = None,
) -> DocumentModeCapability:
    """Build a processing capability response."""
    return DocumentModeCapability(
        available=available,
        status=status_value,
        reason=reason,
    )


def _available(reason: str | None = None) -> DocumentModeCapability:
    """Return an available processing capability."""
    return _capability(True, "available", reason)


def _unavailable(reason: str) -> DocumentModeCapability:
    """Return an unavailable processing capability with its reason."""
    return _capability(False, "unavailable", reason)


def _blocked(reason: str) -> DocumentModeCapability:
    """Return a policy-blocked processing capability with its reason."""
    return _capability(False, "blocked", reason)


def _ocr_available(list_ocr_backends: ListOcrBackends) -> bool:
    """Return whether any registered OCR backend is currently available."""
    backends = list_ocr_backends()
    return any(
        isinstance(details, dict) and details.get("available") is True
        for details in backends.values()
    )


def _size_limit_reason(file: DocumentUploadPreflightFile) -> str | None:
    """Return the size-limit failure reason for a file, when applicable."""
    if file.size_bytes <= DEFAULT_MAX_CHAT_UPLOAD_BYTES:
        return None
    size_mb = DEFAULT_MAX_CHAT_UPLOAD_BYTES // (1024 * 1024)
    return f"{file.filename} exceeds {size_mb} MB limit"


def _page_limit_reason(file: DocumentUploadPreflightFile) -> str | None:
    """Return the page-limit failure reason for a file, when applicable."""
    if file.page_count is None or file.page_count <= DEFAULT_MAX_CHAT_UPLOAD_PAGES:
        return None
    return f"{file.filename} exceeds {DEFAULT_MAX_CHAT_UPLOAD_PAGES} page limit"


def _token_limit_reason(file: DocumentUploadPreflightFile) -> str | None:
    """Return the direct-chat token-limit reason, when applicable."""
    if file.estimated_tokens is None or file.estimated_tokens <= DEFAULT_MAX_DIRECT_CHAT_TOKENS:
        return None
    return f"{file.filename} exceeds {DEFAULT_MAX_DIRECT_CHAT_TOKENS} token direct-chat limit"


def _unsupported_modes() -> dict[DocumentProcessingMode, DocumentModeCapability]:
    """Return uniformly unavailable modes for unsupported file types."""
    reason = "Unsupported document type"
    return {
        "add_to_chat": _unavailable(reason),
        "ocr_pages": _unavailable(reason),
        "ingest_to_library": _unavailable(reason),
    }


def _preflight_file(
    file: DocumentUploadPreflightFile,
    list_ocr_backends: ListOcrBackends,
) -> DocumentUploadPreflightItem:
    """Build the processing decision and limits for one upload candidate."""
    media_type = _media_type_for(file.filename)
    if media_type == "unsupported":
        return DocumentUploadPreflightItem(
            client_id=file.client_id,
            filename=file.filename,
            media_type="unsupported",
            default_mode=None,
            modes=_unsupported_modes(),
            max_size_bytes=DEFAULT_MAX_CHAT_UPLOAD_BYTES,
            max_pages=DEFAULT_MAX_CHAT_UPLOAD_PAGES,
            max_chat_tokens=DEFAULT_MAX_DIRECT_CHAT_TOKENS,
            estimated_pages=file.page_count,
            estimated_tokens=file.estimated_tokens,
            requires_send_time_estimate=False,
        )

    size_reason = _size_limit_reason(file)
    page_reason = _page_limit_reason(file)
    token_reason = _token_limit_reason(file)
    direct_reason = size_reason or page_reason or token_reason

    add_to_chat = _blocked(direct_reason) if direct_reason else _available()
    ingest = _blocked(size_reason) if size_reason else _available()
    ocr = _ocr_capability(file, media_type, size_reason or page_reason, list_ocr_backends)

    modes: dict[DocumentProcessingMode, DocumentModeCapability] = {
        "add_to_chat": add_to_chat,
        "ocr_pages": ocr,
        "ingest_to_library": ingest,
    }
    default_mode = next(
        (
            mode
            for mode in ("add_to_chat", "ingest_to_library", "ocr_pages")
            if modes[mode].available
        ),
        None,
    )
    return DocumentUploadPreflightItem(
        client_id=file.client_id,
        filename=file.filename,
        media_type=media_type,
        default_mode=default_mode,
        modes=modes,
        max_size_bytes=DEFAULT_MAX_CHAT_UPLOAD_BYTES,
        max_pages=DEFAULT_MAX_CHAT_UPLOAD_PAGES,
        max_chat_tokens=DEFAULT_MAX_DIRECT_CHAT_TOKENS,
        estimated_pages=file.page_count,
        estimated_tokens=file.estimated_tokens,
        requires_send_time_estimate=file.page_count is None or file.estimated_tokens is None,
    )


def _ocr_capability(
    file: DocumentUploadPreflightFile,
    media_type: str,
    blocked_reason: str | None,
    list_ocr_backends: ListOcrBackends,
) -> DocumentModeCapability:
    """Return OCR availability after media, limit, and backend checks."""
    if media_type != "pdf":
        suffix = Path(file.filename).suffix.upper() or "file"
        return _unavailable(f"OCR unavailable: server cannot render {suffix} pages")
    if blocked_reason:
        return _blocked(blocked_reason)
    if not _ocr_available(list_ocr_backends):
        return _unavailable("OCR unavailable: no OCR backend configured")
    return _available()

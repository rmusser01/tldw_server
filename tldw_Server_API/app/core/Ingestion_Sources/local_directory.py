from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext.Plaintext_Files import (
    convert_document_to_text,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import resolve_safe_local_path
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import open_safe_local_path
from tldw_Server_API.app.core.config import get_ingestion_source_allowed_roots

NOTES_SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".docx",
        ".htm",
        ".html",
        ".json",
        ".markdown",
        ".md",
        ".rtf",
        ".txt",
        ".xml",
    }
)
MEDIA_SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    NOTES_SUPPORTED_SUFFIXES | {".epub", ".pdf"}
)
_LOCAL_FILE_MAX_BYTES_ENV = "INGESTION_SOURCES_LOCAL_FILE_MAX_BYTES"
_DEFAULT_LOCAL_FILE_MAX_BYTES = 50 * 1024 * 1024


def _local_file_max_bytes() -> int:
    raw_value = os.getenv(_LOCAL_FILE_MAX_BYTES_ENV)
    if raw_value is None or raw_value.strip() == "":
        return _DEFAULT_LOCAL_FILE_MAX_BYTES
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{_LOCAL_FILE_MAX_BYTES_ENV} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{_LOCAL_FILE_MAX_BYTES_ENV} must be a positive integer")
    return value


def process_pdf(
    file_input: str | bytes | Path,
    *,
    filename: str,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Delegate PDF extraction to the shared PDF ingestion library."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import (
        process_pdf as _process_pdf,
    )

    return _process_pdf(file_input, filename=filename, **kwargs)


def process_epub(file_path: str, **kwargs: Any) -> dict[str, Any]:
    """Delegate EPUB extraction to the shared book ingestion library."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Books.Book_Processing_Lib import (
        process_epub as _process_epub,
    )

    return _process_epub(file_path, **kwargs)


def validate_local_directory_source(config: dict[str, Any]) -> Path:
    raw_path = str(config.get("path") or "").strip()
    if not raw_path:
        raise ValueError("Local directory source requires a non-empty path.")

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate

    resolved_candidate = candidate.resolve(strict=False)
    allowed_roots = get_ingestion_source_allowed_roots()
    if not allowed_roots:
        raise ValueError("No ingestion source allowed roots are configured.")

    for allowed_root in allowed_roots:
        safe_path = resolve_safe_local_path(resolved_candidate, allowed_root)
        if safe_path is None:
            continue
        if not safe_path.exists():
            raise ValueError(f"Local directory source path does not exist: {safe_path}")
        if not safe_path.is_dir():
            raise ValueError(f"Local directory source path is not a directory: {safe_path}")
        return safe_path

    allowed_display = ", ".join(str(root) for root in allowed_roots)
    raise ValueError(
        "Local directory source path must resolve under one of the configured allowed roots: "
        f"{allowed_display}"
    )


def _supported_suffixes_for_sink(sink_type: str) -> frozenset[str]:
    normalized = str(sink_type or "").strip().lower()
    if normalized == "notes":
        return NOTES_SUPPORTED_SUFFIXES
    return MEDIA_SUPPORTED_SUFFIXES


def _read_markdown_alias(path: Path, base_dir: Path) -> str:
    handle = open_safe_local_path(path, base_dir, mode="rb")
    if handle is None:
        raise ValueError(f"Local directory source path rejected: {path}")
    with handle:
        data = handle.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _raw_metadata_from_processing_result(
    result: dict[str, Any],
    *,
    source_format: str,
) -> dict[str, Any]:
    raw_metadata: dict[str, Any] = {"source_format": source_format}
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        raw_metadata.update(metadata)
        title = metadata.get("title")
        author = metadata.get("author")
        if title and "extracted_title" not in raw_metadata:
            raw_metadata["extracted_title"] = title
        if author and "extracted_author" not in raw_metadata:
            raw_metadata["extracted_author"] = author
    parser_used = result.get("parser_used")
    if parser_used:
        raw_metadata["parser_used"] = parser_used
    return raw_metadata


def _media_document_to_text(
    path: Path,
    *,
    base_dir: Path,
) -> tuple[str, str, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        result = process_pdf(
            path,
            filename=path.name,
            perform_chunking=False,
            perform_analysis=False,
            base_dir=base_dir,
        )
        if not isinstance(result, dict) or str(result.get("status") or "").strip().lower() == "error":
            raise ValueError(str((result or {}).get("error") or f"Failed to process PDF '{path.name}'"))
        return (
            str(result.get("content") or ""),
            "pdf",
            _raw_metadata_from_processing_result(result, source_format="pdf"),
        )

    if suffix == ".epub":
        result = process_epub(
            str(path),
            perform_chunking=False,
            perform_analysis=False,
            base_dir=base_dir,
        )
        if not isinstance(result, dict) or str(result.get("status") or "").strip().lower() == "error":
            raise ValueError(str((result or {}).get("error") or f"Failed to process EPUB '{path.name}'"))
        return (
            str(result.get("content") or ""),
            "epub",
            _raw_metadata_from_processing_result(result, source_format="epub"),
        )

    return convert_document_to_text(path, base_dir=base_dir)


def build_local_directory_snapshot_with_failures(
    config: dict[str, Any],
    *,
    sink_type: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    source_root = validate_local_directory_source(config)
    supported_suffixes = _supported_suffixes_for_sink(sink_type)
    max_file_bytes = _local_file_max_bytes()
    snapshot_items: dict[str, dict[str, Any]] = {}
    failed_items: dict[str, dict[str, Any]] = {}

    for candidate in sorted(source_root.rglob("*")):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        safe_path = resolve_safe_local_path(candidate, source_root)
        if safe_path is None:
            continue
        suffix = safe_path.suffix.lower()
        if suffix not in supported_suffixes:
            continue

        stat = safe_path.stat()
        relative_path = safe_path.relative_to(source_root).as_posix()
        if int(stat.st_size) > max_file_bytes:
            failed_items[relative_path] = {
                "relative_path": relative_path,
                "source_format": suffix.lstrip(".") or "unknown",
                "size": int(stat.st_size),
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "error": (
                    f"File '{relative_path}' exceeds local file size limit of "
                    f"{max_file_bytes} bytes"
                ),
            }
            continue
        try:
            if suffix == ".markdown":
                text_content = _read_markdown_alias(safe_path, source_root)
                source_format = "markdown"
                raw_metadata: dict[str, Any] = {}
            elif str(sink_type or "").strip().lower() == "media" and suffix in {".epub", ".pdf"}:
                text_content, source_format, raw_metadata = _media_document_to_text(
                    safe_path,
                    base_dir=source_root,
                )
            else:
                text_content, source_format, raw_metadata = convert_document_to_text(
                    safe_path,
                    base_dir=source_root,
                )
        except (AttributeError, LookupError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            failed_items[relative_path] = {
                "relative_path": relative_path,
                "source_format": suffix.lstrip(".") or "unknown",
                "size": int(stat.st_size),
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(exc),
            }
            continue

        snapshot_items[relative_path] = {
            "relative_path": relative_path,
            "absolute_path": str(safe_path),
            "content_hash": hashlib.sha256(text_content.encode("utf-8")).hexdigest(),
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime,
                tz=timezone.utc,
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "size": int(stat.st_size),
            "source_format": source_format,
            "raw_metadata": raw_metadata,
            "text": text_content,
        }

    return snapshot_items, failed_items


def build_local_directory_snapshot(
    config: dict[str, Any],
    *,
    sink_type: str,
) -> dict[str, dict[str, Any]]:
    snapshot_items, failed_items = build_local_directory_snapshot_with_failures(
        config,
        sink_type=sink_type,
    )
    if failed_items:
        first_failure = next(iter(failed_items.values()))
        raise ValueError(str(first_failure.get("error") or "Local directory source ingestion failed."))
    return snapshot_items

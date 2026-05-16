"""Materialize Persona Visual provider archive handoffs for import preview.

Provider envelope normalization and handoff construction intentionally stop at
an MCP resource descriptor. This module implements the next backend-only bridge:
given a ready handoff and an injected resource reader, it writes bounded archive
bytes into local import-preview staging, verifies the provider checksum, and
returns the existing Persona Visual import-preview Jobs payload shape. It does
not execute providers, create preview rows, enqueue Jobs, commit imports,
activate packs, change renderers, or expose raw provider content in diagnostics.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Callable, cast

from tldw_Server_API.app.core.Persona.visual_jobs import (
    PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE,
    build_visual_pack_import_preview_payload,
)

from .archive import DEFAULT_MAX_ARCHIVE_SIZE_BYTES
from .constants import PERSONA_VISUAL_PACK_EXTENSION
from .provider_envelope import COMPATIBLE_PERSONA_VISUAL_ARCHIVE_MEDIA_TYPES


ProviderArchiveResourceReader = Callable[[str], bytes | bytearray | Iterable[bytes]]

_SHA256_HEX_LENGTH = 64
_SAFE_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
_MAX_ID_LENGTH = 128


def materialize_provider_archive_import_preview_handoff(
    handoff: Mapping[str, Any],
    *,
    resource_reader: ProviderArchiveResourceReader,
    staging_root: Path,
    max_archive_size_bytes: int = DEFAULT_MAX_ARCHIVE_SIZE_BYTES,
) -> dict[str, Any]:
    """Fetch a ready provider archive handoff into an import-preview job payload."""
    blockers: list[dict[str, str]] = []
    warnings = _diagnostics_list(handoff.get("warnings"))
    request = _mapping(handoff.get("request"))
    archive = _mapping(handoff.get("archive"))

    if handoff.get("ready") is not True:
        blockers.append(_diagnostic("handoff_not_ready", "Provider archive handoff is not ready."))
    if handoff.get("operation") != "import_preview":
        blockers.append(
            _diagnostic(
                "unsupported_archive_handoff_operation",
                "Provider archive handoff must target import preview.",
            )
        )
    if handoff.get("job_type") != PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE:
        blockers.append(
            _diagnostic(
                "unsupported_archive_handoff_job_type",
                "Provider archive handoff uses an unsupported job type.",
            )
        )
    blockers.extend(_diagnostics_list(handoff.get("blockers")))

    user_id = _safe_identifier(request.get("user_id") if request else None)
    preview_id = _safe_identifier(request.get("preview_id") if request else None)
    request_id = _safe_identifier(request.get("request_id") if request else None)
    target_persona_id = _safe_identifier(request.get("target_persona_id") if request else None)
    if not user_id or not preview_id or not request_id:
        blockers.append(
            _diagnostic(
                "invalid_handoff_request",
                "Provider archive handoff request identifiers are required.",
            )
        )

    resource_uri = _safe_resource_uri(archive.get("mcp_resource_uri") if archive else None)
    expected_sha256 = _safe_sha256(archive.get("sha256") if archive else None)
    media_type = _safe_text(archive.get("media_type") if archive else None, max_length=120)
    if not archive or archive.get("source_type") != "mcp_resource" or not resource_uri:
        blockers.append(
            _diagnostic(
                "archive_resource_uri_missing",
                "Portable archive MCP resource URI is required.",
            )
        )
    if not expected_sha256:
        blockers.append(
            _diagnostic(
                "archive_sha256_invalid",
                "Portable archive SHA-256 checksum is required.",
            )
        )
    if media_type not in COMPATIBLE_PERSONA_VISUAL_ARCHIVE_MEDIA_TYPES:
        blockers.append(
            _diagnostic(
                "unsupported_archive_media_type",
                "Portable archive media type must be a supported zip payload type.",
            )
        )

    if blockers:
        return _blocked_result(blockers=blockers, warnings=warnings)

    user_id = cast(str, user_id)
    preview_id = cast(str, preview_id)
    request_id = cast(str, request_id)
    resource_uri = cast(str, resource_uri)
    expected_sha256 = cast(str, expected_sha256)

    archive_path = _materialized_archive_path(
        staging_root=Path(staging_root),
        user_id=user_id,
        preview_id=preview_id,
        request_id=request_id,
        resource_uri=resource_uri,
    )
    try:
        size_bytes, actual_sha256 = _write_resource_archive(
            archive_path=archive_path,
            resource_reader=resource_reader,
            resource_uri=resource_uri,
            max_archive_size_bytes=max_archive_size_bytes,
        )
    except _ProviderArchiveRetrievalError as exc:
        _remove_file(archive_path)
        return _blocked_result(blockers=[_diagnostic(exc.code, exc.message)], warnings=warnings)

    if actual_sha256 != expected_sha256:
        _remove_file(archive_path)
        return _blocked_result(
            blockers=[
                _diagnostic(
                    "archive_sha256_mismatch",
                    "Retrieved archive checksum does not match provider metadata.",
                )
            ],
            warnings=warnings,
        )

    job_payload = build_visual_pack_import_preview_payload(
        user_id=user_id,
        preview_id=preview_id,
        archive_path=str(archive_path),
        request_id=request_id,
        target_persona_id=target_persona_id,
    )
    return {
        "ready": True,
        "operation": "import_preview",
        "job_type": PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE,
        "job_payload": job_payload,
        "archive": {
            "source_type": "mcp_resource",
            "media_type": media_type,
            "sha256": actual_sha256,
            "size_bytes": size_bytes,
        },
        "diagnostics": {
            "status": "ready_for_import_preview_job",
            "blockers": [],
            "warnings": warnings,
        },
        "blockers": [],
        "warnings": warnings,
    }


class _ProviderArchiveRetrievalError(Exception):
    """Trace-safe retrieval failure with a stable diagnostic code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code)
        self.code = code
        self.message = message


def _write_resource_archive(
    *,
    archive_path: Path,
    resource_reader: ProviderArchiveResourceReader,
    resource_uri: str,
    max_archive_size_bytes: int,
) -> tuple[int, str]:
    """Write resource chunks to disk while enforcing size and checksum accounting."""
    if max_archive_size_bytes <= 0:
        raise _ProviderArchiveRetrievalError(
            "archive_too_large",
            "Retrieved archive exceeds size limits.",
        )

    try:
        resource = resource_reader(resource_uri)
        chunks = _resource_chunks(resource)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size_bytes = 0
        with archive_path.open("wb") as file_obj:
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise _ProviderArchiveRetrievalError(
                        "archive_resource_invalid",
                        "Retrieved archive resource must yield bytes.",
                    )
                size_bytes += len(chunk)
                if size_bytes > max_archive_size_bytes:
                    raise _ProviderArchiveRetrievalError(
                        "archive_too_large",
                        "Retrieved archive exceeds size limits.",
                    )
                digest.update(chunk)
                file_obj.write(chunk)
    except _ProviderArchiveRetrievalError:
        raise
    except OSError as exc:
        raise _ProviderArchiveRetrievalError(
            "archive_materialization_failed",
            "Retrieved archive could not be written to staging.",
        ) from exc
    except Exception as exc:
        raise _ProviderArchiveRetrievalError(
            "archive_retrieval_failed",
            "Provider archive resource retrieval failed.",
        ) from exc

    return size_bytes, digest.hexdigest()


def _resource_chunks(value: bytes | bytearray | Iterable[bytes]) -> Iterable[bytes]:
    """Return byte chunks from a reader result or raise a trace-safe error."""
    if isinstance(value, bytes):
        return (value,)
    if isinstance(value, bytearray):
        return (bytes(value),)
    if isinstance(value, (str, Mapping)):
        raise _ProviderArchiveRetrievalError(
            "archive_resource_invalid",
            "Retrieved archive resource must be bytes.",
        )
    if isinstance(value, Iterable):
        return value
    raise _ProviderArchiveRetrievalError(
        "archive_resource_invalid",
        "Retrieved archive resource must be bytes.",
    )


def _materialized_archive_path(
    *,
    staging_root: Path,
    user_id: str,
    preview_id: str,
    request_id: str,
    resource_uri: str,
) -> Path:
    """Return a deterministic local archive path without embedding provider URI text."""
    fingerprint = hashlib.sha256(
        f"{user_id}\0{preview_id}\0{request_id}\0{resource_uri}".encode("utf-8")
    ).hexdigest()[:24]
    filename = f"{fingerprint}{PERSONA_VISUAL_PACK_EXTENSION}"
    return Path(staging_root).resolve(strict=False) / filename


def _blocked_result(
    *,
    blockers: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    """Return a trace-safe blocked materialization result."""
    return {
        "ready": False,
        "operation": "import_preview",
        "job_type": PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE,
        "job_payload": None,
        "archive": None,
        "diagnostics": {
            "status": "blocked",
            "blockers": _dedupe_diagnostics(blockers),
            "warnings": warnings,
        },
        "blockers": _dedupe_diagnostics(blockers),
        "warnings": warnings,
    }


def _mapping(value: Any) -> Mapping[str, Any] | None:
    """Return mappings only, preserving read-only access for caller-owned data."""
    return value if isinstance(value, Mapping) else None


def _diagnostics_list(value: Any) -> list[dict[str, str]]:
    """Copy bounded diagnostic code/message objects without leaking arbitrary payloads."""
    if not isinstance(value, list):
        return []
    diagnostics: list[dict[str, str]] = []
    for item in value[:64]:
        if not isinstance(item, Mapping):
            continue
        code = _safe_code(item.get("code"))
        message = _safe_text(item.get("message"), max_length=300)
        if code and message:
            diagnostics.append(_diagnostic(code, message))
    return diagnostics


def _dedupe_diagnostics(items: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return diagnostics in first-seen order without duplicate codes."""
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for item in items:
        code = _safe_code(item.get("code"))
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(_diagnostic(code, item.get("message", "")))
    return out


def _diagnostic(code: str, message: str) -> dict[str, str]:
    """Build a stable machine-readable diagnostic object."""
    return {
        "code": _safe_code(code) or "archive_materialization_failed",
        "message": (
            _safe_text(message, max_length=300)
            or "Provider archive materialization failed."
        ),
    }


def _safe_identifier(value: Any) -> str | None:
    """Return a bounded identifier for job payload fields."""
    text = _safe_text(value, max_length=_MAX_ID_LENGTH)
    if not text:
        return None
    if any(char not in _SAFE_ID_CHARS for char in text):
        return None
    return text


def _safe_resource_uri(value: Any) -> str | None:
    """Return an MCP resource URI handle without accepting paths or remote URLs."""
    text = _safe_text(value, max_length=500)
    if not text or not text.startswith("mcp://"):
        return None
    if any(part in text for part in ("\n", "\r", "\x00", "\\", "../", "/..")):
        return None
    return text


def _safe_sha256(value: Any) -> str | None:
    """Return a lowercase SHA-256 digest when it is syntactically valid."""
    text = _safe_text(value, max_length=80).lower()
    if len(text) != _SHA256_HEX_LENGTH:
        return None
    if any(char not in "0123456789abcdef" for char in text):
        return None
    return text


def _safe_code(value: Any) -> str:
    """Return a bounded diagnostic code."""
    text = _safe_text(value, max_length=80)
    if not text:
        return ""
    if not text[0].isalpha():
        return ""
    if any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_:-" for char in text):
        return ""
    return text


def _safe_text(value: Any, *, max_length: int) -> str:
    """Return stripped scalar text without path expansion or newlines."""
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) > max_length:
        return ""
    if any(char in text for char in ("\n", "\r", "\x00")):
        return ""
    return text


def _remove_file(path: Path) -> None:
    """Remove a partial materialized archive without surfacing filesystem details."""
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except OSError:
        return

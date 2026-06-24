"""Shared image parsing/format conversion helpers for image-generation adapters."""

from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.http_client import (
    DEFAULT_MAX_REDIRECTS,
    _resolve_redirect_url,
    _validate_egress_or_raise,
    create_client,
)
from tldw_Server_API.app.core.Image_Generation.capabilities import ResolvedReferenceImage
from tldw_Server_API.app.core.Image_Generation.exceptions import ImageGenerationError

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency guard
    Image = None  # type: ignore


def decode_data_url(data_url: str, *, max_bytes: int | None = None) -> tuple[bytes, str]:
    header, _, encoded = data_url.partition(",")
    if not header.startswith("data:"):
        raise ImageGenerationError("invalid data URL")
    meta = header[5:]
    content_type = "application/octet-stream"
    content_type = meta.split(";", 1)[0] or content_type if ";" in meta else meta or content_type
    if ";base64" not in header:
        raise ImageGenerationError("unsupported data URL encoding")
    encoded_clean = "".join(encoded.split())
    _enforce_base64_encoded_limit(encoded_clean, max_bytes)
    try:
        content = base64.b64decode(encoded_clean, validate=True)
    except (binascii.Error, TypeError, ValueError) as exc:
        raise ImageGenerationError("invalid base64 data") from exc
    _enforce_max_bytes(content, max_bytes)
    return content, content_type


def decode_base64_image(encoded: str, *, max_bytes: int | None = None) -> bytes:
    _enforce_base64_encoded_limit(encoded, max_bytes)
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, TypeError, ValueError) as exc:
        raise ImageGenerationError("invalid base64 image data") from exc
    _enforce_max_bytes(content, max_bytes)
    return content


def maybe_decode_base64_image(encoded: str | None, *, max_bytes: int | None = None) -> bytes | None:
    if not isinstance(encoded, str):
        return None
    raw = encoded.strip()
    if not raw:
        return None
    if raw.startswith("data:"):
        try:
            content, _content_type = decode_data_url(raw, max_bytes=max_bytes)
            return content
        except ImageGenerationError as exc:
            if "too large" in str(exc):
                raise
            return None
    if any(ch.isspace() for ch in raw):
        return None
    _enforce_base64_encoded_limit(raw, max_bytes)
    try:
        content = base64.b64decode(raw, validate=True)
    except (binascii.Error, TypeError, ValueError):
        return None
    _enforce_max_bytes(content, max_bytes)
    return content


def reference_image_data_url(reference_image: ResolvedReferenceImage) -> str:
    """Encode a normalized reference image into a Model Studio-compatible data URL."""

    content = reference_image.content
    if content is None:
        if not reference_image.temp_path:
            raise ImageGenerationError("invalid reference image data")
        try:
            content = Path(reference_image.temp_path).read_bytes()
        except Exception as exc:
            raise ImageGenerationError("invalid reference image data") from exc
    if not content:
        raise ImageGenerationError("invalid reference image data")

    mime_type = (reference_image.mime_type or "application/octet-stream").split(";", 1)[0].strip().lower()
    if not mime_type:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def format_from_content_type(content_type: str) -> str | None:
    if not content_type:
        return None
    ctype = content_type.split(";", 1)[0].strip().lower()
    if ctype == "image/png":
        return "png"
    if ctype == "image/jpeg":
        return "jpg"
    if ctype == "image/webp":
        return "webp"
    return None


def format_from_bytes(content: bytes) -> str | None:
    if not content:
        return None
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if content.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "webp"
    return None


def content_type_for_format(fmt: str) -> str:
    if fmt == "png":
        return "image/png"
    if fmt == "jpg":
        return "image/jpeg"
    if fmt == "webp":
        return "image/webp"
    return "application/octet-stream"


def maybe_convert_format(
    content: bytes,
    content_type: str,
    actual_format: str | None,
    requested_format: str,
) -> tuple[bytes, str]:
    if requested_format == actual_format:
        return content, content_type or content_type_for_format(requested_format)
    if requested_format not in {"png", "jpg", "webp"}:
        raise ImageGenerationError(f"unsupported output format: {requested_format}")
    if actual_format is None:
        raise ImageGenerationError("invalid image content")
    if Image is None:
        raise ImageGenerationError("Pillow is required for image format conversion")
    try:
        with Image.open(io.BytesIO(content)) as img:
            if requested_format == "jpg" and img.mode not in {"RGB"}:
                img = img.convert("RGB")
            if requested_format == "png" and img.mode in {"P"}:
                img = img.convert("RGBA")
            buf = io.BytesIO()
            save_format = {
                "jpg": "JPEG",
                "png": "PNG",
                "webp": "WEBP",
            }[requested_format]
            img.save(buf, format=save_format)
            converted = buf.getvalue()
    except Exception as exc:
        raise ImageGenerationError(f"failed to convert image: {exc}") from exc
    return converted, content_type_for_format(requested_format)


def validate_and_convert_image_output(
    content: bytes,
    content_type: str,
    requested_format: str,
    *,
    max_bytes: int | None = None,
) -> tuple[bytes, str]:
    """Validate actual image bytes, enforce size caps, and convert when requested."""

    _enforce_max_bytes(content, max_bytes)
    output_format = "jpg" if requested_format == "jpeg" else requested_format
    actual_format = format_from_bytes(content)
    if actual_format is None:
        raise ImageGenerationError("invalid image content")
    actual_content_type = content_type_for_format(actual_format)
    converted, converted_content_type = maybe_convert_format(
        content,
        actual_content_type,
        actual_format,
        output_format,
    )
    _enforce_max_bytes(converted, max_bytes)
    return converted, converted_content_type


def fetch_image_bytes(
    url: str,
    *,
    timeout: int | float,
    headers: dict[str, Any] | None = None,
    cookies: dict[str, Any] | None = None,
    max_bytes: int | None = None,
) -> tuple[bytes, str]:
    current_url = url
    try:
        with create_client(timeout=timeout) as client:
            for _redirect_count in range(DEFAULT_MAX_REDIRECTS + 1):
                _validate_egress_or_raise(current_url)
                with client.stream(
                    "GET",
                    current_url,
                    headers=headers,
                    cookies=cookies,
                    timeout=timeout,
                    follow_redirects=False,
                ) as response:
                    status = int(getattr(response, "status_code", 0) or 0)
                    if status in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location") or response.headers.get("Location")
                        if not location:
                            raise ImageGenerationError("image fetch failed: redirect without location")
                        next_url = _resolve_redirect_url(str(getattr(response, "url", current_url)), str(location))
                        if not next_url:
                            raise ImageGenerationError("image fetch failed: invalid redirect")
                        current_url = next_url
                        continue
                    if status >= 400:
                        raise ImageGenerationError(f"image fetch failed with status {status}")

                    headers_obj = getattr(response, "headers", {}) or {}
                    _reject_declared_oversize(headers_obj, max_bytes)
                    content = _read_stream_with_limit(response.iter_bytes(), max_bytes)
                    content_type = (
                        headers_obj.get("content-type")
                        or headers_obj.get("Content-Type")
                        or "application/octet-stream"
                    )
                    return content, content_type.split(";", 1)[0].strip().lower()
            raise ImageGenerationError("image fetch failed: too many redirects")
    except ImageGenerationError:
        raise
    except Exception as exc:
        raise ImageGenerationError(f"image fetch failed: {exc}") from exc


def _enforce_base64_encoded_limit(encoded: str, max_bytes: int | None) -> None:
    if max_bytes is None:
        return
    try:
        limit = int(max_bytes)
    except (TypeError, ValueError):
        return
    if limit <= 0:
        return
    max_encoded_chars = ((limit + 2) // 3) * 4
    if len(encoded.strip()) > max_encoded_chars:
        raise ImageGenerationError("image content too large")


def _reject_declared_oversize(headers: Any, max_bytes: int | None) -> None:
    limit = _positive_byte_limit(max_bytes)
    if limit is None:
        return
    content_length = headers.get("content-length") or headers.get("Content-Length")
    if content_length is None:
        return
    try:
        declared_size = int(str(content_length).strip())
    except ValueError:
        return
    if declared_size > limit:
        raise ImageGenerationError("image content too large")


def _read_stream_with_limit(chunks: Any, max_bytes: int | None) -> bytes:
    limit = _positive_byte_limit(max_bytes)
    total = 0
    parts: list[bytes] = []
    for chunk in chunks:
        if not chunk:
            continue
        total += len(chunk)
        if limit is not None and total > limit:
            raise ImageGenerationError("image content too large")
        parts.append(chunk)
    return b"".join(parts)


def _enforce_max_bytes(content: bytes, max_bytes: int | None) -> None:
    limit = _positive_byte_limit(max_bytes)
    if limit is not None and len(content) > limit:
        raise ImageGenerationError("image content too large")


def _positive_byte_limit(max_bytes: int | None) -> int | None:
    if max_bytes is None:
        return None
    try:
        limit = int(max_bytes)
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None

"""Storage helpers for VN asset pack generated files."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import SOURCE_FEATURE_VN_ASSETS
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

VN_ASSET_CONTENT_NOT_FOUND = "vn_asset_content_not_found"
VN_ASSET_INVALID_IMAGE = "vn_asset_invalid_image"
VN_ASSET_UNSUPPORTED_IMAGE_MIME = "vn_asset_unsupported_image_mime"
VN_ASSET_IMAGE_MIME_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
}


def vn_asset_source_ref(item_id: int) -> str:
    """Return the generated-file source reference for a VN asset item."""
    return f"vn_asset_item:{item_id}"


def generated_file_matches_vn_asset(record: dict[str, Any], *, user_id: int, item_id: int) -> bool:
    """Check AuthNZ generated-file metadata against the expected VN asset owner/item."""
    return (
        int(record.get("user_id") or 0) == user_id
        and record.get("source_feature") == SOURCE_FEATURE_VN_ASSETS
        and record.get("source_ref") == vn_asset_source_ref(item_id)
        and not bool(record.get("is_deleted"))
    )


def resolve_vn_asset_storage_path(*, user_id: int, storage_path: str) -> Path:
    """Resolve a VN asset storage path under the user's generated outputs directory."""
    relative_path = Path(storage_path)
    if not storage_path or relative_path.is_absolute():
        raise ValueError(VN_ASSET_CONTENT_NOT_FOUND)

    base_dir = DatabasePaths.get_user_outputs_dir(user_id).resolve()
    full_path = (base_dir / relative_path).resolve()

    try:
        full_path.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError(VN_ASSET_CONTENT_NOT_FOUND) from exc

    return full_path


def image_format_from_mime_type(mime_type: str) -> str:
    """Return the storage image extension for an allowed uploaded VN asset MIME type."""
    image_format = VN_ASSET_IMAGE_MIME_TYPES.get(mime_type.lower())
    if image_format is None:
        raise ValueError(VN_ASSET_UNSUPPORTED_IMAGE_MIME)
    return image_format


def detect_image_dimensions(image_bytes: bytes, mime_type: str) -> tuple[int, int]:
    """Detect dimensions for supported VN asset upload images without decoding pixels."""
    normalized_mime = mime_type.lower()
    if normalized_mime == "image/png":
        return _detect_png_dimensions(image_bytes)
    if normalized_mime == "image/jpeg":
        return _detect_jpeg_dimensions(image_bytes)
    if normalized_mime == "image/webp":
        return _detect_webp_dimensions(image_bytes)
    raise ValueError(VN_ASSET_UNSUPPORTED_IMAGE_MIME)


def unlink_vn_asset_storage_file(*, user_id: int, storage_path: str) -> bool:
    """Remove a VN asset storage file from disk if it exists under the user's outputs dir."""
    full_path = resolve_vn_asset_storage_path(user_id=user_id, storage_path=storage_path)
    if not full_path.exists():
        return False
    if not full_path.is_file():
        raise ValueError(VN_ASSET_CONTENT_NOT_FOUND)
    try:
        full_path.unlink()
    except FileNotFoundError:
        return False
    return True


def generated_file_size_bytes(record: dict[str, Any], fallback: int | None = None) -> int:
    """Extract a non-negative generated-file byte count from AuthNZ metadata."""
    raw_size = record.get("file_size_bytes")
    if raw_size is None:
        raw_size = fallback
    try:
        size = int(raw_size or 0)
    except (TypeError, ValueError):
        return 0
    return max(size, 0)


def _detect_png_dimensions(image_bytes: bytes) -> tuple[int, int]:
    if len(image_bytes) < 24 or not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(VN_ASSET_INVALID_IMAGE)
    width, height = struct.unpack(">II", image_bytes[16:24])
    return _validate_dimensions(width, height)


def _detect_jpeg_dimensions(image_bytes: bytes) -> tuple[int, int]:
    if len(image_bytes) < 4 or not image_bytes.startswith(b"\xff\xd8"):
        raise ValueError(VN_ASSET_INVALID_IMAGE)

    index = 2
    while index + 9 < len(image_bytes):
        if image_bytes[index] != 0xFF:
            raise ValueError(VN_ASSET_INVALID_IMAGE)
        marker = image_bytes[index + 1]
        index += 2
        while marker == 0xFF and index < len(image_bytes):
            marker = image_bytes[index]
            index += 1
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(image_bytes):
            raise ValueError(VN_ASSET_INVALID_IMAGE)
        segment_length = struct.unpack(">H", image_bytes[index:index + 2])[0]
        if segment_length < 2 or index + segment_length > len(image_bytes):
            raise ValueError(VN_ASSET_INVALID_IMAGE)
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if segment_length < 7:
                raise ValueError(VN_ASSET_INVALID_IMAGE)
            height = struct.unpack(">H", image_bytes[index + 3:index + 5])[0]
            width = struct.unpack(">H", image_bytes[index + 5:index + 7])[0]
            return _validate_dimensions(width, height)
        index += segment_length
    raise ValueError(VN_ASSET_INVALID_IMAGE)


def _detect_webp_dimensions(image_bytes: bytes) -> tuple[int, int]:
    if len(image_bytes) < 30 or image_bytes[:4] != b"RIFF" or image_bytes[8:12] != b"WEBP":
        raise ValueError(VN_ASSET_INVALID_IMAGE)
    chunk_type = image_bytes[12:16]
    if chunk_type == b"VP8 ":
        if len(image_bytes) < 30 or image_bytes[23:26] != b"\x9d\x01\x2a":
            raise ValueError(VN_ASSET_INVALID_IMAGE)
        width = struct.unpack("<H", image_bytes[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", image_bytes[28:30])[0] & 0x3FFF
        return _validate_dimensions(width, height)
    if chunk_type == b"VP8L":
        if len(image_bytes) < 25 or image_bytes[20] != 0x2F:
            raise ValueError(VN_ASSET_INVALID_IMAGE)
        bits = image_bytes[21] | (image_bytes[22] << 8) | (image_bytes[23] << 16) | (image_bytes[24] << 24)
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return _validate_dimensions(width, height)
    if chunk_type == b"VP8X":
        if len(image_bytes) < 30:
            raise ValueError(VN_ASSET_INVALID_IMAGE)
        width = int.from_bytes(image_bytes[24:27], "little") + 1
        height = int.from_bytes(image_bytes[27:30], "little") + 1
        return _validate_dimensions(width, height)
    raise ValueError(VN_ASSET_INVALID_IMAGE)


def _validate_dimensions(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(VN_ASSET_INVALID_IMAGE)
    return width, height

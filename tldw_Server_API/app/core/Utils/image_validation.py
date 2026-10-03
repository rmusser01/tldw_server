# image_validation.py
# Description: Image validation utilities for secure image processing
#
# Imports
import base64
import io
import os
import re
import struct
from typing import Optional
from urllib.parse import urlparse

from loguru import logger
from PIL import BmpImagePlugin, IcoImagePlugin, Image

_IMAGE_VALIDATION_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ImportError,
    KeyError,
    OSError,
    TypeError,
    ValueError,
)
_IMAGE_INTEGRITY_EXCEPTIONS = _IMAGE_VALIDATION_NONCRITICAL_EXCEPTIONS + (SyntaxError, Image.DecompressionBombError)

#######################################################################################################################
#
# Constants:

MAX_IMAGE_PIXELS = 16777216  # 16 megapixels max


def validate_image_pixel_limit(image_bytes: bytes) -> None:
    """Bound eager PNG/APNG, GIF, ICO, and WebP bitmap allocation before ``Image.open``.

    Inspect raster metadata before its first frame;
    Pillow still verifies and decodes the complete image afterward. ICO's DIB
    reader is lazy, whereas its outer reader and animated PNG/GIF readers may
    allocate pixels during opening. WebP allocates its canvas in native code. Other codecs keep their existing lazy-open
    dimension checks.

    Raises:
        Image.DecompressionBombError: A raster exceeds the shared pixel ceiling.
        ValueError or OSError: Container metadata is truncated or invalid.
    """
    def check_size(width: int, height: int) -> None:
        pixels = width * height
        if pixels > MAX_IMAGE_PIXELS:
            raise Image.DecompressionBombError(f"Image too large: {pixels} pixels")

    def check_png(data: memoryview) -> None:
        if len(data) < 33 or data[8:16] != b"\x00\x00\x00\rIHDR":
            raise ValueError("Invalid PNG dimension header")
        check_size(*struct.unpack_from(">II", data, 16))
        position = 33
        while position + 8 <= len(data):
            size = struct.unpack_from(">I", data, position)[0]
            kind = data[position + 4:position + 8]
            end = position + 12 + size
            if end > len(data):
                raise ValueError("Truncated PNG metadata chunk")
            if kind == b"IHDR":
                raise ValueError("Duplicate PNG dimension header")
            if kind in (b"IDAT", b"fdAT"):
                return
            if kind == b"IEND":
                break
            position = end
        raise ValueError("PNG has no complete image data chunk")

    try:
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            check_png(memoryview(image_bytes))
        elif image_bytes[:6] in (b"GIF87a", b"GIF89a"):
            if len(image_bytes) < 13:
                raise ValueError("Truncated GIF dimension header")
            width, height = struct.unpack_from("<HH", image_bytes, 6)
            check_size(width, height)
            flags = image_bytes[10]
            position = 13 + (3 * (1 << ((flags & 7) + 1)) if flags & 128 else 0)
            while position < len(image_bytes):
                marker = image_bytes[position]
                position += 1
                if marker == 0x2C:  # First image descriptor, including its canvas offset.
                    if position + 9 > len(image_bytes):
                        raise ValueError("Truncated GIF image descriptor")
                    x, y, frame_width, frame_height = struct.unpack_from("<HHHH", image_bytes, position)
                    check_size(max(width, x + frame_width), max(height, y + frame_height))
                    return
                if marker != 0x21:
                    raise ValueError("Invalid GIF block before its first image")
                position += 1  # Extension label; remaining data consists of length-prefixed blocks.
                while True:
                    block_size = image_bytes[position]
                    position += 1
                    if position + block_size > len(image_bytes):
                        raise ValueError("Truncated GIF extension block")
                    position += block_size
                    if not block_size:
                        break
            raise ValueError("GIF has no complete image descriptor")
        elif image_bytes.startswith(b"\x00\x00\x01\x00"):
            icon = IcoImagePlugin.IcoFile(io.BytesIO(image_bytes))
            if not icon.entry:
                raise ValueError("ICO has no image frames")
            # Pillow's default open decodes only its first sorted entry.
            entry = icon.entry[0]
            if entry.offset < 6 + 16 * icon.nb_items or not entry.size or entry.offset + entry.size > len(image_bytes):
                raise ValueError("Invalid ICO frame bounds")
            header = image_bytes[entry.offset:entry.offset + min(entry.size, 33)]
            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                check_png(memoryview(image_bytes)[entry.offset:entry.offset + entry.size])
            else:
                icon.buf.seek(entry.offset)
                with BmpImagePlugin.DibImageFile(icon.buf) as bitmap:
                    # ICO's DIB height includes its XOR pixels and AND mask.
                    check_size(bitmap.width, bitmap.height // 2)
        elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            # The first master chunk holds the native decoder's canvas dimensions.
            if len(image_bytes) < 20:
                raise ValueError("Truncated WebP dimension header")
            kind = image_bytes[12:16]
            size = struct.unpack_from("<I", image_bytes, 16)[0]
            if size + 20 > len(image_bytes):
                raise ValueError("Truncated WebP master chunk")
            if kind == b"VP8X" and size >= 10:
                width = int.from_bytes(image_bytes[24:27], "little") + 1
                height = int.from_bytes(image_bytes[27:30], "little") + 1
            elif kind == b"VP8L" and size >= 5 and image_bytes[20] == 0x2F:
                bits = struct.unpack_from("<I", image_bytes, 21)[0]
                width, height = (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
            elif kind == b"VP8 " and size >= 10 and image_bytes[23:26] == b"\x9d\x01\x2a":
                width, height = struct.unpack_from("<HH", image_bytes, 26)
                width, height = width & 0x3FFF, height & 0x3FFF
            else:
                raise ValueError("Invalid WebP dimension header")
            check_size(width, height)
    except (IndexError, struct.error) as exc:
        raise ValueError("Truncated image dimension metadata") from exc


def get_max_base64_bytes() -> int:
    """Resolve max base64 image bytes via env/config (default 3MB)."""
    # Env override (in megabytes)
    try:
        env_mb = os.getenv("CHAT_IMAGE_MAX_MB")
        if env_mb is not None:
            mb = max(1, int(env_mb))
            return mb * 1024 * 1024
    except (TypeError, ValueError):
        pass
    try:
        from tldw_Server_API.app.core.config import load_comprehensive_config
        cfg = load_comprehensive_config()
        if cfg and cfg.has_section('Chat-Module'):
            raw = cfg.get('Chat-Module', 'max_base64_image_size_mb', fallback=None)
            if raw is not None:
                mb = max(1, int(raw))
                return mb * 1024 * 1024
    except _IMAGE_VALIDATION_NONCRITICAL_EXCEPTIONS:
        pass
    return 3 * 1024 * 1024


def get_allowed_image_mime_types() -> set[str]:
    """Return allowed image MIME types (static set; hook for future config)."""
    # Optionally allow env override as comma-separated list
    try:
        env_val = os.getenv("CHAT_ALLOWED_IMAGE_MIME_TYPES")
        if env_val:
            return {m.strip().lower() for m in env_val.split(',') if m.strip()}
    except (TypeError, ValueError):
        pass
    return {"image/png", "image/jpeg", "image/webp"}


def get_max_flashcard_asset_bytes() -> int:
    """Resolve the flashcard asset byte cap (defaulting to chat image limits)."""
    raw_bytes = os.getenv("FLASHCARD_ASSET_MAX_BYTES")
    if raw_bytes is not None:
        try:
            return max(1, int(raw_bytes))
        except (TypeError, ValueError):
            logger.warning("Invalid FLASHCARD_ASSET_MAX_BYTES={!r}; falling back to defaults.", raw_bytes)

    raw_mb = os.getenv("FLASHCARD_ASSET_MAX_MB")
    if raw_mb is not None:
        try:
            return max(1, int(raw_mb)) * 1024 * 1024
        except (TypeError, ValueError):
            logger.warning("Invalid FLASHCARD_ASSET_MAX_MB={!r}; falling back to defaults.", raw_mb)

    return get_max_base64_bytes()

# Expose module-level variables for convenience (computed at import time)
MAX_BASE64_BYTES = get_max_base64_bytes()
ALLOWED_IMAGE_MIME_TYPES = get_allowed_image_mime_types()
# Back-compat constant used by tests to build oversized base64 strings
# Approximate maximum base64 length that maps to MAX_BASE64_BYTES decoded bytes
MAX_BASE64_STRING_LENGTH = int(MAX_BASE64_BYTES * 4 / 3) + 100

# Regex pattern for data URI validation
DATA_URI_PATTERN = re.compile(r'^data:([^;]+);base64,(.+)$')

#######################################################################################################################
#
# Functions:

def validate_mime_type(mime_type: str) -> bool:
    """
    Validate if the MIME type is allowed.

    Args:
        mime_type: MIME type to validate

    Returns:
        True if MIME type is allowed, False otherwise
    """
    return mime_type.lower() in get_allowed_image_mime_types()


def estimate_decoded_size(base64_string: str) -> int:
    """
    Estimate the decoded size of a base64 string without actually decoding it.

    Args:
        base64_string: Base64-encoded string

    Returns:
        Estimated size in bytes of the decoded data
    """
    # Remove padding characters
    base64_string = base64_string.rstrip('=')
    # Each base64 character represents 6 bits, so 4 characters = 3 bytes
    return int(len(base64_string) * 3 / 4)


def validate_data_uri(data_uri: str) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Validate a data URI and extract its components safely.

    Args:
        data_uri: Data URI string to validate

    Returns:
        Tuple of (is_valid, mime_type, base64_data)
    """
    # Check if it starts with 'data:'
    if not data_uri.startswith('data:'):
        return False, None, None

    # Parse the data URI
    match = DATA_URI_PATTERN.match(data_uri)
    if not match:
        logger.warning("Invalid data URI format")
        return False, None, None

    mime_type = match.group(1)
    base64_data = match.group(2)

    # Validate MIME type
    if not validate_mime_type(mime_type):
        logger.warning(f"Disallowed MIME type: {mime_type}")
        return False, mime_type, None

    # Check base64 string length BEFORE decoding
    max_bytes = get_max_base64_bytes()
    max_str_len = int(max_bytes * 4 / 3) + 100
    if len(base64_data) > max_str_len:
        logger.warning(f"Base64 string too long: {len(base64_data)} > {max_str_len}")
        return False, mime_type, None

    # Estimate decoded size
    estimated_size = estimate_decoded_size(base64_data)
    if estimated_size > max_bytes:
        logger.warning(f"Estimated decoded size too large: {estimated_size} > {max_bytes}")
        return False, mime_type, None

    return True, mime_type, base64_data


def safe_decode_base64_image(base64_data: str, mime_type: str) -> Optional[bytes]:
    """
    Safely decode a base64-encoded image with size validation.

    Args:
        base64_data: Base64-encoded image data
        mime_type: MIME type of the image

    Returns:
        Decoded bytes if valid, None otherwise
    """
    try:
        normalized_mime = str(mime_type or "").lower().strip()
        # Final validation of MIME type
        if not validate_mime_type(normalized_mime):
            logger.warning(f"Invalid MIME type for decoding: {mime_type}")
            return None

        # Decode the base64 data
        decoded_data = base64.b64decode(base64_data, validate=True)

        # Final size check on decoded data
        max_bytes = get_max_base64_bytes()
        if len(decoded_data) > max_bytes:
            logger.warning(f"Decoded image too large: {len(decoded_data)} > {max_bytes}")
            return None

        try:
            validate_image_pixel_limit(decoded_data)
            with Image.open(io.BytesIO(decoded_data)) as img:
                detected_mime = Image.MIME.get(img.format or "")
                img.verify()
        except _IMAGE_INTEGRITY_EXCEPTIONS as exc:
            logger.warning("Decoded image failed integrity validation: {}", exc)
            return None

        if detected_mime and detected_mime.lower() != normalized_mime:
            logger.warning(
                "Decoded image MIME mismatch: expected {}, detected {}",
                normalized_mime,
                detected_mime.lower(),
            )
            return None

        # TODO: Add optional virus/malware scanning here
        # if ENABLE_VIRUS_SCAN:
        #     if not scan_for_malware(decoded_data, mime_type):
        #         logger.warning("Image failed malware scan")
        #         return None

        return decoded_data

    except base64.binascii.Error as e:
        logger.warning(f"Invalid base64 data: {e}")
        return None
    except (MemoryError, OSError, TypeError, ValueError) as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None


def validate_image_url(url: str) -> tuple[bool, Optional[str], Optional[bytes]]:
    """
    Validate and process an image URL (data URI or HTTP URL).

    Args:
        url: Image URL to validate

    Returns:
        Tuple of (is_valid, mime_type, decoded_bytes)
    """
    if url.startswith('data:'):
        # Handle data URI
        is_valid, mime_type, base64_data = validate_data_uri(url)
        if not is_valid:
            return False, mime_type, None

        decoded_bytes = safe_decode_base64_image(base64_data, mime_type)
        if decoded_bytes is None:
            return False, mime_type, None

        return True, mime_type, decoded_bytes
    else:
        # For now, we don't support external URLs for security reasons
        try:
            parsed = urlparse(str(url or ""))
            host = parsed.hostname or "unknown"
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
            logger.warning(
                "External image URLs not supported: scheme={}, host={}",
                parsed.scheme or "unknown",
                host,
            )
        except (TypeError, ValueError, AttributeError):
            logger.warning("External image URLs not supported: unparsable URL")
        return False, None, None


def validate_uploaded_image_bytes(
    image_bytes: bytes,
    mime_type: str,
) -> tuple[bool, Optional[str], Optional[int], Optional[int]]:
    """Validate uploaded raster image bytes and return dimensions when possible."""
    normalized_mime = str(mime_type or "").lower().strip()
    if not validate_mime_type(normalized_mime):
        return False, f"Unsupported image MIME type: {mime_type}", None, None

    max_bytes = get_max_flashcard_asset_bytes()
    if len(image_bytes) > max_bytes:
        return False, f"Image exceeds max size of {max_bytes} bytes", None, None

    try:
        validate_image_pixel_limit(image_bytes)
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            detected_mime = Image.MIME.get(img.format or "")
            img.verify()
    except _IMAGE_INTEGRITY_EXCEPTIONS as exc:
        logger.warning("Uploaded image failed integrity validation: {}", exc)
        return False, "Uploaded image failed integrity validation", None, None

    if detected_mime and detected_mime.lower() != normalized_mime:
        return False, f"Uploaded image MIME mismatch: expected {normalized_mime}, detected {detected_mime.lower()}", None, None

    return True, None, int(width), int(height)


#
# End of image_validation.py
#######################################################################################################################

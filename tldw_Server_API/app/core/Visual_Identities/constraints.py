"""Format and size constraints for visual identity expression assets."""

from __future__ import annotations

BASELINE_VISUAL_IDENTITY_MIME_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
    }
)
BASELINE_ANIMATED_VISUAL_IDENTITY_MIME_TYPES = frozenset(
    {
        "image/webp",
        "image/gif",
    }
)
AVIF_MIME_TYPE = "image/avif"

MAX_EXPRESSION_ASSET_BYTES = 25 * 1024 * 1024
MAX_EXPRESSION_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_EXPRESSION_IMAGE_DIMENSION = 4096
MAX_EXPRESSION_FRAME_COUNT = 512


def supports_avif() -> bool:
    """Return whether Pillow can decode AVIF in the current runtime."""
    try:
        from PIL import features
    except ImportError:
        return False
    return bool(features.check("avif"))


def supported_visual_identity_mime_types() -> tuple[str, ...]:
    """Return MIME types accepted by this runtime for visual identity assets."""
    supported = set(BASELINE_VISUAL_IDENTITY_MIME_TYPES)
    if supports_avif():
        supported.add(AVIF_MIME_TYPE)
    return tuple(sorted(supported))


def build_visual_identity_capabilities() -> dict[str, object]:
    """Build the backend capability payload shared by API and validation layers."""
    avif_enabled = supports_avif()
    supported_mime_types = set(BASELINE_VISUAL_IDENTITY_MIME_TYPES)
    if avif_enabled:
        supported_mime_types.add(AVIF_MIME_TYPE)

    return {
        "upload_max_bytes": MAX_EXPRESSION_ASSET_BYTES,
        "archive_max_bytes": MAX_EXPRESSION_ARCHIVE_BYTES,
        "max_dimension": MAX_EXPRESSION_IMAGE_DIMENSION,
        "max_frame_count": MAX_EXPRESSION_FRAME_COUNT,
        "supported_mime_types": sorted(supported_mime_types),
        "avif_enabled": avif_enabled,
    }

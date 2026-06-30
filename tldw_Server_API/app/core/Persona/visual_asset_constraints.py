"""Shared asset constraints for Persona Visual upload and capability contracts.

The upload service validates incoming raster assets with these values, and the
renderer capability registry exposes the same values to clients. Keeping them in
one module prevents the API-advertised limits from drifting away from server
validation behavior.
"""

from __future__ import annotations

from types import MappingProxyType


VISUAL_RASTER_MIME_TYPES = (
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
)
VISUAL_RASTER_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
)
ALLOWED_VISUAL_MIME_TYPES = frozenset(VISUAL_RASTER_MIME_TYPES)
VISUAL_MIME_EXTENSIONS = MappingProxyType(
    {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
)
MAX_VISUAL_IMAGE_DIMENSION = 4096

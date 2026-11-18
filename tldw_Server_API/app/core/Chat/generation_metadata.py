"""Compatibility exports for shared chat generation metadata."""

from tldw_Server_API.app.core.Utils.generation_metadata import (
    GENERATION_METADATA_KEYS,
    GENERATION_STATUSES,
    GENERATION_STATUS_COMPLETE,
    GENERATION_STATUS_ERROR,
    GENERATION_STATUS_INTERRUPTED,
    GENERATION_STATUS_LENGTH,
    GENERATION_STATUS_STOPPED,
    USAGE_KEYS,
    GenerationMetadataError,
    build_generation_metadata,
    sanitize_generation_metadata,
    sanitize_usage,
    validate_generation_metadata,
)

__all__ = [
    "GENERATION_METADATA_KEYS",
    "GENERATION_STATUSES",
    "GENERATION_STATUS_COMPLETE",
    "GENERATION_STATUS_ERROR",
    "GENERATION_STATUS_INTERRUPTED",
    "GENERATION_STATUS_LENGTH",
    "GENERATION_STATUS_STOPPED",
    "USAGE_KEYS",
    "GenerationMetadataError",
    "build_generation_metadata",
    "sanitize_generation_metadata",
    "sanitize_usage",
    "validate_generation_metadata",
]

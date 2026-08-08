"""Canonical facade for HTML extraction foundations."""

from .caches import clear_extraction_caches, get_extraction_cache_stats
from .dependencies import ExtractionDependencies, build_default_dependencies

__all__ = [
    "ExtractionDependencies",
    "build_default_dependencies",
    "clear_extraction_caches",
    "get_extraction_cache_stats",
]

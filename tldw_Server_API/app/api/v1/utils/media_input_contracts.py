"""Shared media endpoint input contract helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def normalize_urls_field(urls: list[str] | None) -> list[str] | None:
    """Normalize legacy empty-list sentinel used by some clients."""

    if urls and urls == [""]:
        return None
    return urls


def validate_media_inputs(
    validate_inputs_fn: Callable[[str, Any, Any], None],
    media_type: str,
    urls: Any,
    files: Any,
) -> None:
    """Validate endpoint inputs via the shared compatibility validator."""

    validate_inputs_fn(media_type, urls, files)


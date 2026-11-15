"""Media endpoint deprecation signaling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.api.v1.utils.deprecation import build_deprecation_headers


@dataclass(frozen=True)
class MediaLegacySignal:
    """HTTP header and payload signals for media legacy compatibility paths."""

    headers: dict[str, str]
    payload: dict[str, str]


def build_media_legacy_signal(
    *,
    successor: str,
    warning_code: str,
    default_sunset_days: int = 90,
) -> MediaLegacySignal:
    """Build standard deprecation signal structures for media endpoints."""

    return MediaLegacySignal(
        headers=build_deprecation_headers(
            successor,
            default_sunset_days=default_sunset_days,
        ),
        payload={
            "warning": "deprecated_endpoint",
            "code": warning_code,
            "successor": successor,
        },
    )


def apply_media_legacy_headers(response: Any, signal: MediaLegacySignal) -> Any:
    """Apply deprecation headers to a response-like object."""

    response.headers.update(signal.headers)
    return response

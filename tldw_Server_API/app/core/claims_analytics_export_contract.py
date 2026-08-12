"""Dependency-neutral contracts shared by Claims analytics export layers."""

from __future__ import annotations

from typing import Any

CLAIMS_MAX_OWNER_USER_ID = 9_223_372_036_854_775_807
_CLAIMS_MAX_OWNER_USER_ID_TEXT = str(CLAIMS_MAX_OWNER_USER_ID)

CLAIMS_ANALYTICS_EXPORT_WORKSPACE_ID_MAX_CHARS = len(
    _CLAIMS_MAX_OWNER_USER_ID_TEXT
)
CLAIMS_ANALYTICS_EXPORT_EVENT_TYPE_MAX_CHARS = 128
CLAIMS_ANALYTICS_EXPORT_SEVERITY_MAX_CHARS = 64
CLAIMS_ANALYTICS_EXPORT_PROVIDER_MAX_CHARS = 128
CLAIMS_ANALYTICS_EXPORT_MODEL_MAX_CHARS = 256
CLAIMS_ANALYTICS_EXPORT_TIMESTAMP_MAX_CHARS = 64
CLAIMS_ANALYTICS_EXPORT_REQUEST_JSON_MAX_BYTES = 8_192
CLAIMS_ANALYTICS_EXPORT_FILTERS_JSON_MAX_BYTES = (
    CLAIMS_ANALYTICS_EXPORT_REQUEST_JSON_MAX_BYTES
)
CLAIMS_ANALYTICS_EXPORT_PAGINATION_JSON_MAX_BYTES = (
    CLAIMS_ANALYTICS_EXPORT_REQUEST_JSON_MAX_BYTES
)
_CLAIMS_ANALYTICS_EXPORT_FILTER_MAX_CHARS = {
    "workspace_id": CLAIMS_ANALYTICS_EXPORT_WORKSPACE_ID_MAX_CHARS,
    "event_type": CLAIMS_ANALYTICS_EXPORT_EVENT_TYPE_MAX_CHARS,
    "severity": CLAIMS_ANALYTICS_EXPORT_SEVERITY_MAX_CHARS,
    "provider": CLAIMS_ANALYTICS_EXPORT_PROVIDER_MAX_CHARS,
    "model": CLAIMS_ANALYTICS_EXPORT_MODEL_MAX_CHARS,
    "start_time": CLAIMS_ANALYTICS_EXPORT_TIMESTAMP_MAX_CHARS,
    "end_time": CLAIMS_ANALYTICS_EXPORT_TIMESTAMP_MAX_CHARS,
}


def is_routable_claims_owner_id_text(value: Any) -> bool:
    """Return whether value is a canonical positive signed-BIGINT owner id."""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or not value.isascii()
        or not value.isdigit()
        or value == "0"
        or (len(value) > 1 and value.startswith("0"))
    ):
        return False
    maximum = _CLAIMS_MAX_OWNER_USER_ID_TEXT
    return len(value) < len(maximum) or (
        len(value) == len(maximum) and value <= maximum
    )


def is_valid_persisted_claims_analytics_export_filters(value: Any) -> bool:
    """Return whether persisted filters match the public response schema shape."""
    if not isinstance(value, dict) or not set(value).issubset(
        _CLAIMS_ANALYTICS_EXPORT_FILTER_MAX_CHARS
    ):
        return False
    return all(
        field_value is None
        or (
            isinstance(field_value, str)
            and len(field_value)
            <= _CLAIMS_ANALYTICS_EXPORT_FILTER_MAX_CHARS[field_name]
        )
        for field_name, field_value in value.items()
    )


def is_valid_persisted_claims_analytics_export_pagination(value: Any) -> bool:
    """Return whether persisted pagination matches the public history shape."""
    if not isinstance(value, dict) or not set(value).issubset(
        {"limit", "offset", "total"}
    ):
        return False
    for field_name, field_value in value.items():
        if field_value is None:
            continue
        if isinstance(field_value, bool) or not isinstance(field_value, int):
            return False
        if field_name == "limit" and not 1 <= field_value <= 10_000:
            return False
        if field_name != "limit" and field_value < 0:
            return False
    return True

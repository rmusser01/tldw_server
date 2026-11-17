"""Dependency-neutral contracts shared by Claims analytics export layers."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.Utils.claims_owner_identity import (
    _CLAIMS_MAX_OWNER_USER_ID_TEXT,
)
from tldw_Server_API.app.core.Utils.claims_owner_identity import (
    CLAIMS_MAX_OWNER_USER_ID as CLAIMS_MAX_OWNER_USER_ID,
)
from tldw_Server_API.app.core.Utils.claims_owner_identity import (
    is_routable_claims_owner_id_text as is_routable_claims_owner_id_text,
)

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

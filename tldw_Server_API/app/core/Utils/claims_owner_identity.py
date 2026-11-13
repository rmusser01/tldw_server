"""Dependency-neutral canonical Claims owner identity validation."""

from typing import Any

CLAIMS_MAX_OWNER_USER_ID = 9_223_372_036_854_775_807
_CLAIMS_MAX_OWNER_USER_ID_TEXT = str(CLAIMS_MAX_OWNER_USER_ID)


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
    return len(value) < len(maximum) or (len(value) == len(maximum) and value <= maximum)

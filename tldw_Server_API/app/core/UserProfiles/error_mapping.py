"""
Stable error taxonomy for UserProfiles update failures.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from fastapi import status


class ProfileErrorCode(str, Enum):
    UNKNOWN_KEY = "unknown_key"
    UNSUPPORTED_KEY = "unsupported_key"
    INVALID_PAYLOAD = "invalid_payload"
    INVALID_ACTION = "invalid_action"
    INVALID_VALUE = "invalid_value"
    TYPE_MISMATCH = "type_mismatch"
    ENUM_VIOLATION = "enum_violation"
    MIN_VIOLATION = "min_violation"
    MAX_VIOLATION = "max_violation"
    INVALID_ROLE = "invalid_role"
    FORBIDDEN_KEY = "forbidden"
    FORBIDDEN_SCOPE = "forbidden_scope"
    FORBIDDEN_ROLE_ESCALATION = "forbidden_role_escalation"
    TARGET_NOT_FOUND = "user_not_found"
    MEMBERSHIP_NOT_FOUND = "membership_not_found"
    TEAM_NOT_FOUND = "team_not_found"
    ORG_NOT_FOUND = "org_not_found"
    VERSION_MISMATCH = "profile_version_mismatch"


@dataclass(frozen=True)
class ProfileErrorMapping:
    status_code: int
    error_code: str
    detail: str


_UNKNOWN_KEY_MAPPING = ProfileErrorMapping(
    status_code=status.HTTP_400_BAD_REQUEST,
    error_code="profile_update_unknown_key",
    detail="One or more keys are not recognized",
)
_INVALID_PAYLOAD_MAPPING = ProfileErrorMapping(
    status_code=status.HTTP_400_BAD_REQUEST,
    error_code="profile_update_invalid",
    detail="Invalid profile update payload",
)
_INVALID_ACTION_MAPPING = ProfileErrorMapping(
    status_code=status.HTTP_400_BAD_REQUEST,
    error_code="profile_update_invalid",
    detail="Invalid profile update action",
)
_FORBIDDEN_MAPPING = ProfileErrorMapping(
    status_code=status.HTTP_403_FORBIDDEN,
    error_code="profile_update_forbidden",
    detail="Caller cannot edit one or more fields",
)
_NOT_FOUND_MAPPING = ProfileErrorMapping(
    status_code=status.HTTP_404_NOT_FOUND,
    error_code="profile_update_not_found",
    detail="Target resource not found",
)
_VERSION_MISMATCH_MAPPING = ProfileErrorMapping(
    status_code=status.HTTP_409_CONFLICT,
    error_code="profile_version_mismatch",
    detail="profile_version_mismatch",
)
_INVALID_MAPPING = ProfileErrorMapping(
    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    error_code="profile_update_invalid",
    detail="One or more updates failed validation",
)

_PROFILE_ERROR_MAPPINGS = {
    ProfileErrorCode.UNKNOWN_KEY: _UNKNOWN_KEY_MAPPING,
    ProfileErrorCode.UNSUPPORTED_KEY: _UNKNOWN_KEY_MAPPING,
    ProfileErrorCode.INVALID_PAYLOAD: _INVALID_PAYLOAD_MAPPING,
    ProfileErrorCode.INVALID_ACTION: _INVALID_ACTION_MAPPING,
    ProfileErrorCode.INVALID_VALUE: _INVALID_MAPPING,
    ProfileErrorCode.TYPE_MISMATCH: _INVALID_MAPPING,
    ProfileErrorCode.ENUM_VIOLATION: _INVALID_MAPPING,
    ProfileErrorCode.MIN_VIOLATION: _INVALID_MAPPING,
    ProfileErrorCode.MAX_VIOLATION: _INVALID_MAPPING,
    ProfileErrorCode.INVALID_ROLE: _INVALID_MAPPING,
    ProfileErrorCode.FORBIDDEN_KEY: _FORBIDDEN_MAPPING,
    ProfileErrorCode.FORBIDDEN_SCOPE: _FORBIDDEN_MAPPING,
    ProfileErrorCode.FORBIDDEN_ROLE_ESCALATION: _FORBIDDEN_MAPPING,
    ProfileErrorCode.TARGET_NOT_FOUND: _NOT_FOUND_MAPPING,
    ProfileErrorCode.MEMBERSHIP_NOT_FOUND: _INVALID_MAPPING,
    ProfileErrorCode.TEAM_NOT_FOUND: _NOT_FOUND_MAPPING,
    ProfileErrorCode.ORG_NOT_FOUND: _NOT_FOUND_MAPPING,
    ProfileErrorCode.VERSION_MISMATCH: _VERSION_MISMATCH_MAPPING,
}

_FORBIDDEN_SKIP_MESSAGES = {
    "forbidden",
    "forbidden_scope",
    "forbidden_role_escalation",
    "owner_required",
    "org_membership_required",
}
_UNKNOWN_SKIP_MESSAGES = {
    "unknown_key",
    "unsupported_key",
    "unsupported_type",
}
_LEGACY_USER_NOT_FOUND_MESSAGES = {
    "user_not_found",
}
_LEGACY_USER_NOT_FOUND_MAPPING = ProfileErrorMapping(
    status_code=status.HTTP_404_NOT_FOUND,
    error_code="profile_update_not_found",
    detail="Target user not found",
)
_DOMAIN_CODE_PRECEDENCE = (
    ProfileErrorCode.TEAM_NOT_FOUND,
    ProfileErrorCode.ORG_NOT_FOUND,
    ProfileErrorCode.TARGET_NOT_FOUND,
    ProfileErrorCode.VERSION_MISMATCH,
    ProfileErrorCode.FORBIDDEN_KEY,
    ProfileErrorCode.FORBIDDEN_SCOPE,
    ProfileErrorCode.FORBIDDEN_ROLE_ESCALATION,
    ProfileErrorCode.UNKNOWN_KEY,
    ProfileErrorCode.UNSUPPORTED_KEY,
    ProfileErrorCode.INVALID_PAYLOAD,
    ProfileErrorCode.INVALID_ACTION,
    ProfileErrorCode.MEMBERSHIP_NOT_FOUND,
    ProfileErrorCode.INVALID_VALUE,
    ProfileErrorCode.TYPE_MISMATCH,
    ProfileErrorCode.ENUM_VIOLATION,
    ProfileErrorCode.MIN_VIOLATION,
    ProfileErrorCode.MAX_VIOLATION,
    ProfileErrorCode.INVALID_ROLE,
)


def map_profile_error_code(code: ProfileErrorCode | str) -> ProfileErrorMapping:
    profile_code = ProfileErrorCode(code)
    return _PROFILE_ERROR_MAPPINGS[profile_code]


def _first_matching_domain_code(messages: set[str]) -> ProfileErrorCode | None:
    for code in _DOMAIN_CODE_PRECEDENCE:
        if code.value in messages:
            return code
    return None


def classify_legacy_profile_update_skips(
    skipped: Iterable[Mapping[str, str]],
) -> ProfileErrorMapping | None:
    """Map legacy per-key skip reasons into the shared update error envelope."""
    skipped_list = list(skipped)
    if not skipped_list:
        return None

    messages = {str(item.get("message") or "") for item in skipped_list}
    if messages & _LEGACY_USER_NOT_FOUND_MESSAGES:
        return _LEGACY_USER_NOT_FOUND_MAPPING
    if messages & _FORBIDDEN_SKIP_MESSAGES:
        return _FORBIDDEN_MAPPING
    if messages & _UNKNOWN_SKIP_MESSAGES:
        return _UNKNOWN_KEY_MAPPING

    mapped_code = _first_matching_domain_code(messages)
    if mapped_code is not None:
        return map_profile_error_code(mapped_code)

    return _INVALID_MAPPING


__all__ = [
    "classify_legacy_profile_update_skips",
    "ProfileErrorCode",
    "ProfileErrorMapping",
    "map_profile_error_code",
]

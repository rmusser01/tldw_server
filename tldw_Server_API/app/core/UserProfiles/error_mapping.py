"""
Stable error taxonomy for UserProfiles update failures.
"""

from __future__ import annotations

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


def map_profile_error_code(code: ProfileErrorCode | str) -> ProfileErrorMapping:
    profile_code = ProfileErrorCode(code)
    return _PROFILE_ERROR_MAPPINGS[profile_code]


__all__ = [
    "ProfileErrorCode",
    "ProfileErrorMapping",
    "map_profile_error_code",
]

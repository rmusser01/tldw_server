from __future__ import annotations

import pytest
from fastapi import status

from tldw_Server_API.app.core.UserProfiles.error_mapping import (
    ProfileErrorCode,
    map_profile_error_code,
)


def test_profile_error_code_http_mapping() -> None:
    assert (
        map_profile_error_code(ProfileErrorCode.UNKNOWN_KEY).status_code
        == status.HTTP_400_BAD_REQUEST
    )
    assert (
        map_profile_error_code(ProfileErrorCode.INVALID_VALUE).status_code
        == status.HTTP_422_UNPROCESSABLE_ENTITY
    )
    assert (
        map_profile_error_code(ProfileErrorCode.FORBIDDEN_SCOPE).status_code
        == status.HTTP_403_FORBIDDEN
    )
    assert (
        map_profile_error_code(ProfileErrorCode.TARGET_NOT_FOUND).status_code
        == status.HTTP_404_NOT_FOUND
    )
    assert (
        map_profile_error_code(ProfileErrorCode.VERSION_MISMATCH).status_code
        == status.HTTP_409_CONFLICT
    )


def test_forbidden_role_escalation_maps_to_forbidden_profile_update() -> None:
    mapped = map_profile_error_code(ProfileErrorCode.FORBIDDEN_ROLE_ESCALATION)

    assert mapped.status_code == status.HTTP_403_FORBIDDEN
    assert mapped.error_code == "profile_update_forbidden"
    assert mapped.detail == "Caller cannot edit one or more fields"


def test_known_raw_string_maps_to_profile_error_code() -> None:
    mapped = map_profile_error_code("unsupported_key")

    assert mapped.status_code == status.HTTP_400_BAD_REQUEST
    assert mapped.error_code == "profile_update_unknown_key"
    assert mapped.detail == "One or more keys are not recognized"


def test_unknown_raw_string_is_rejected() -> None:
    with pytest.raises(ValueError):
        map_profile_error_code("not_a_profile_error")

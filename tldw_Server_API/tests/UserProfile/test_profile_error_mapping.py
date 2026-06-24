from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from fastapi import status

from tldw_Server_API.app.api.v1.utils.profile_errors import (
    classify_profile_update_skips,
)
from tldw_Server_API.app.core.UserProfiles.error_mapping import (
    ProfileErrorCode,
    map_profile_error_code,
)


def _classify_mixed_domain_skips() -> tuple[int, str, str]:
    classified = classify_profile_update_skips(
        [
            {"key": "team_id", "message": "team_not_found"},
            {"key": "profile_type", "message": "type_mismatch"},
        ]
    )

    assert classified is not None
    return classified[0], classified[1], classified[2]


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


def test_mixed_domain_skip_messages_use_deterministic_not_found_precedence() -> None:
    expected = (
        status.HTTP_404_NOT_FOUND,
        "profile_update_not_found",
        "Target resource not found",
    )

    assert _classify_mixed_domain_skips() == expected

    script = """
import json

from tldw_Server_API.app.api.v1.utils.profile_errors import classify_profile_update_skips

result = classify_profile_update_skips(
    [
        {"key": "team_id", "message": "team_not_found"},
        {"key": "profile_type", "message": "type_mismatch"},
    ]
)

print(json.dumps(result[:3]))
"""
    observed = set()
    for seed in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        )
        observed.add(tuple(json.loads(completed.stdout)))

    assert observed == {expected}


def test_bad_request_direct_domain_codes_map_to_invalid_details() -> None:
    invalid_payload = map_profile_error_code(ProfileErrorCode.INVALID_PAYLOAD)
    invalid_action = map_profile_error_code(ProfileErrorCode.INVALID_ACTION)

    assert invalid_payload.status_code == status.HTTP_400_BAD_REQUEST
    assert invalid_payload.error_code == "profile_update_invalid"
    assert invalid_payload.error_code != "profile_update_unknown_key"
    assert invalid_payload.detail == "Invalid profile update payload"

    assert invalid_action.status_code == status.HTTP_400_BAD_REQUEST
    assert invalid_action.error_code == "profile_update_invalid"
    assert invalid_action.error_code != "profile_update_unknown_key"
    assert invalid_action.detail == "Invalid profile update action"

    assert invalid_payload.detail != invalid_action.detail


def test_not_found_details_diverge_for_direct_and_legacy_user_mapping() -> None:
    direct = map_profile_error_code(ProfileErrorCode.TARGET_NOT_FOUND)
    legacy = classify_profile_update_skips(
        [{"key": "user_id", "message": "user_not_found"}]
    )

    assert direct.status_code == status.HTTP_404_NOT_FOUND
    assert direct.error_code == "profile_update_not_found"
    assert direct.detail == "Target resource not found"

    assert legacy is not None
    legacy_status, legacy_error_code, legacy_detail, _legacy_errors = legacy
    assert legacy_status == status.HTTP_404_NOT_FOUND
    assert legacy_error_code == "profile_update_not_found"
    assert legacy_detail == "Target user not found"

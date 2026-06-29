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


def test_core_legacy_skip_classifier_preserves_compatibility_buckets() -> None:
    from tldw_Server_API.app.core.UserProfiles.error_mapping import (
        classify_legacy_profile_update_skips,
    )

    assert classify_legacy_profile_update_skips([]) is None

    unsupported_type = classify_legacy_profile_update_skips(
        [{"key": "preferences.ui.theme", "message": "unsupported_type"}]
    )
    assert unsupported_type is not None
    assert unsupported_type.status_code == status.HTTP_400_BAD_REQUEST
    assert unsupported_type.error_code == "profile_update_unknown_key"
    assert unsupported_type.detail == "One or more keys are not recognized"

    user_not_found = classify_legacy_profile_update_skips(
        [{"key": "user_id", "message": "user_not_found"}]
    )
    assert user_not_found is not None
    assert user_not_found.status_code == status.HTTP_404_NOT_FOUND
    assert user_not_found.error_code == "profile_update_not_found"
    assert user_not_found.detail == "Target user not found"


def test_core_legacy_skip_classifier_uses_domain_precedence() -> None:
    from tldw_Server_API.app.core.UserProfiles.error_mapping import (
        classify_legacy_profile_update_skips,
    )

    team_not_found = classify_legacy_profile_update_skips(
        [
            {"key": "memberships.teams.role", "message": "team_not_found"},
            {"key": "profile_type", "message": "type_mismatch"},
        ]
    )
    assert team_not_found is not None
    assert team_not_found.status_code == status.HTTP_404_NOT_FOUND
    assert team_not_found.error_code == "profile_update_not_found"
    assert team_not_found.detail == "Target resource not found"

    invalid_payload = classify_legacy_profile_update_skips(
        [{"key": "memberships.teams.role", "message": "invalid_payload"}]
    )
    assert invalid_payload is not None
    assert invalid_payload.status_code == status.HTTP_400_BAD_REQUEST
    assert invalid_payload.error_code == "profile_update_invalid"
    assert invalid_payload.detail == "Invalid profile update payload"


@pytest.mark.parametrize(
    ("message", "expected_status", "expected_error", "expected_detail"),
    [
        (
            "team_not_found",
            status.HTTP_404_NOT_FOUND,
            "profile_update_not_found",
            "Target resource not found",
        ),
        (
            "invalid_payload",
            status.HTTP_400_BAD_REQUEST,
            "profile_update_invalid",
            "Invalid profile update payload",
        ),
        (
            "unsupported_type",
            status.HTTP_400_BAD_REQUEST,
            "profile_update_unknown_key",
            "One or more keys are not recognized",
        ),
        (
            "user_not_found",
            status.HTTP_404_NOT_FOUND,
            "profile_update_not_found",
            "Target user not found",
        ),
    ],
)
def test_api_profile_skip_adapter_matches_core_legacy_classifier(
    message: str,
    expected_status: int,
    expected_error: str,
    expected_detail: str,
) -> None:
    from tldw_Server_API.app.core.UserProfiles.error_mapping import (
        classify_legacy_profile_update_skips,
    )

    skipped = [{"key": "some.key", "message": message}]

    core_mapping = classify_legacy_profile_update_skips(skipped)
    api_mapping = classify_profile_update_skips(skipped)

    assert core_mapping is not None
    assert (core_mapping.status_code, core_mapping.error_code, core_mapping.detail) == (
        expected_status,
        expected_error,
        expected_detail,
    )
    assert api_mapping is not None
    api_status, api_error, api_detail, api_errors = api_mapping
    assert (api_status, api_error, api_detail) == (
        core_mapping.status_code,
        core_mapping.error_code,
        core_mapping.detail,
    )
    assert [(error.key, error.message) for error in api_errors] == [
        ("some.key", message)
    ]

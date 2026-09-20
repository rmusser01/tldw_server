from __future__ import annotations

from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def _current_user_id(client: TestClient, auth_headers) -> int:
    response = client.get("/api/v1/users/me/profile", headers=auth_headers)
    assert response.status_code == 200
    return int(response.json()["user"]["id"])


def test_legacy_self_update_returns_string_version_applied_and_empty_skipped(
    auth_headers,
) -> None:
    with TestClient(app) as client:
        response = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.theme", "value": "legacy-contract"},
                ],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["profile_version"], str)
    assert payload["applied"] == ["preferences.ui.theme"]
    assert payload["skipped"] == []


def test_legacy_self_update_with_valid_and_unknown_key_rejects_without_partial_apply(
    auth_headers,
) -> None:
    rejected_value = "legacy-contract-rejected"

    with TestClient(app) as client:
        response = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.theme", "value": rejected_value},
                    {"key": "preferences.ui.missing", "value": "ignored"},
                ],
            },
        )

        profile_response = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "preferences"},
            headers=auth_headers,
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error_code"] == "profile_update_unknown_key"
    assert payload["detail"] == "One or more keys are not recognized"
    assert payload["errors"] == [
        {"key": "preferences.ui.missing", "message": "unknown_key"}
    ]

    assert profile_response.status_code == 200
    preferences = profile_response.json().get("preferences", {})
    assert preferences.get("preferences.ui.theme") != rejected_value


def test_legacy_self_stale_version_precedes_invalid_value_rejection(
    auth_headers,
) -> None:
    with TestClient(app) as client:
        response = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "profile_version": "2000-01-01T00:00:00Z",
                "updates": [
                    {"key": "identity.email", "value": "not-an-email"},
                ],
            },
        )

    assert response.status_code == 409
    assert response.json() == {
        "error_code": "profile_version_mismatch",
        "detail": "profile_version_mismatch",
        "errors": [{"key": "profile_version", "message": "mismatch"}],
    }


def test_legacy_self_dry_run_preserves_duplicate_key_order_without_writing(
    auth_headers,
) -> None:
    first_value = "legacy-dry-run-first"
    second_value = "legacy-dry-run-second"

    with TestClient(app) as client:
        response = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "dry_run": True,
                "updates": [
                    {"key": "preferences.ui.theme", "value": first_value},
                    {"key": "preferences.ui.theme", "value": second_value},
                ],
            },
        )
        profile_response = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "preferences"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert response.json()["applied"] == [
        "preferences.ui.theme",
        "preferences.ui.theme",
    ]
    assert response.json()["skipped"] == []
    preferences = profile_response.json().get("preferences", {})
    assert preferences.get("preferences.ui.theme") not in {first_value, second_value}


def test_legacy_admin_update_returns_string_version_applied_and_empty_skipped(
    auth_headers,
) -> None:
    with TestClient(app) as client:
        user_id = _current_user_id(client, auth_headers)
        response = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "limits.storage_quota_mb", "value": 4096},
                ],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["profile_version"], str)
    assert payload["applied"] == ["limits.storage_quota_mb"]
    assert payload["skipped"] == []


def test_legacy_admin_stale_version_precedes_invalid_value_rejection(
    auth_headers,
) -> None:
    with TestClient(app) as client:
        user_id = _current_user_id(client, auth_headers)
        response = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "profile_version": "2000-01-01T00:00:00Z",
                "updates": [
                    {"key": "identity.email", "value": "not-an-email"},
                ],
            },
        )

    assert response.status_code == 409
    payload = response.json()
    assert payload["error_code"] == "profile_version_mismatch"
    assert payload["detail"] == "profile_version_mismatch"
    assert payload["errors"] == [
        {"key": "profile_version", "message": "mismatch"}
    ]

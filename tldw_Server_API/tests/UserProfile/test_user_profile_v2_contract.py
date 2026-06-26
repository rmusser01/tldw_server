from __future__ import annotations

from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def test_v2_single_update_has_no_skipped_field(auth_headers) -> None:
    with TestClient(app) as client:
        response = client.patch(
            "/api/v2/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.theme", "value": "v2-contract"},
                ]
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["applied"] == ["preferences.ui.theme"]
    assert "skipped" not in payload

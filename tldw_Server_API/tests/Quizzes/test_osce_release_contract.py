from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_osce_profile_is_available_only_with_complete_contract(
    client_user_only: TestClient,
) -> None:
    response = client_user_only.get("/api/v1/quizzes/generation-profiles")

    assert response.status_code == 200
    profile = next(
        item for item in response.json() if item["id"] == "osce_scenario"
    )
    assert profile["status"] == "available"
    assert profile["output_kind"] == "osce_stations"
    assert profile["default_num_stations"] == 1
    assert profile["default_num_questions"] == 1

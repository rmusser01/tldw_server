from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.schemas.quizzes import (
    QuizGenerateResponse,
    QuizGenerationProfile,
)
from tldw_Server_API.app.services.quiz_generator import (
    QuizGenerationRequestError,
    ensure_generation_profile_available,
    get_quiz_generation_profiles,
)

pytestmark = pytest.mark.unit


def test_profile_catalog_exposes_output_kind_and_station_defaults() -> None:
    profiles = {profile["id"]: profile for profile in get_quiz_generation_profiles()}

    assert profiles["standard_recall"]["output_kind"] == "questions"
    assert profiles["standard_recall"]["default_num_stations"] is None
    assert profiles["osce_scenario"]["output_kind"] == "osce_stations"
    assert profiles["osce_scenario"]["default_num_stations"] == 1
    assert profiles["osce_scenario"]["default_num_questions"] == 1
    assert profiles["osce_scenario"]["status"] == "available"


def test_osce_profile_is_available() -> None:
    ensure_generation_profile_available(QuizGenerationProfile.OSCE_SCENARIO)


def test_osce_profile_can_be_disabled_only_by_catalog_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import quiz_generator

    monkeypatch.setitem(
        quiz_generator._PROFILE_BY_ID["osce_scenario"],
        "status",
        "planned",
    )

    with pytest.raises(
        QuizGenerationRequestError,
        match="^generation_profile_unavailable$",
    ):
        ensure_generation_profile_available(QuizGenerationProfile.OSCE_SCENARIO)


def test_question_generation_response_compatibility_defaults_remain_stable() -> None:
    response = QuizGenerateResponse.model_validate(
        {
            "quiz": {
                "id": 1,
                "name": "Recall",
                "total_questions": 0,
                "deleted": False,
                "client_id": "test-user",
                "version": 1,
            }
        }
    )

    assert response.output_kind == "questions"
    assert response.questions == []
    assert response.osce_stations == []

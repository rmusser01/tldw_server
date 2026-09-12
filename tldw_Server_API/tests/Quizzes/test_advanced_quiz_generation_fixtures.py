import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.osce import OsceStationCreateContent
from tldw_Server_API.app.api.v1.schemas.quizzes import QuizGenerateRequest
from tldw_Server_API.app.services.osce_generator import normalize_generated_station
from tldw_Server_API.app.services.quiz_generator import (
    QuizProvenanceValidationError,
    _normalize_planned_questions,
    _normalize_questions,
    _validate_assertion_reasoning_questions,
    _validate_strict_provenance,
    get_quiz_generation_profiles,
)

pytestmark = pytest.mark.unit

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "advanced_quiz_generation_profiles.json"
DOC_PATH = Path(__file__).parents[3] / "Docs" / "API" / "Quizzes.md"


def _load_fixture_matrix() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _available_profile_ids() -> set[str]:
    return {profile["id"] for profile in get_quiz_generation_profiles() if profile["status"] == "available"}


def _available_question_profile_ids() -> list[str]:
    return sorted(
        profile["id"]
        for profile in get_quiz_generation_profiles()
        if profile["status"] == "available" and profile["output_kind"] == "questions"
    )


def test_fixture_matrix_and_documentation_cover_every_available_profile() -> None:
    matrix = _load_fixture_matrix()
    catalog = {profile["id"]: profile for profile in get_quiz_generation_profiles() if profile["status"] == "available"}
    profile_ids = set(matrix["profiles"])
    validation_profile_ids = {case["profile"] for case in matrix["malformed_output_cases"]}
    available_profile_ids = _available_profile_ids()
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert profile_ids == available_profile_ids
    assert validation_profile_ids == available_profile_ids
    for profile_id in available_profile_ids:
        profile = catalog[profile_id]
        example = matrix["profiles"][profile_id]
        table_row = next(line for line in docs.splitlines() if line.startswith(f"| `{profile_id}` |"))
        assert f'<a id="profile-{profile_id}"></a>' in docs
        assert example["catalog"] == profile
        assert example["request"]["generation_profile"] == profile_id
        assert example["request"]["difficulty"] == profile["default_difficulty"]
        assert example["output"]["output_kind"] == profile["output_kind"]
        default_size = profile["default_num_stations"] or profile["default_num_questions"]
        assert f"| {default_size} " in table_row


@pytest.mark.parametrize("profile_id", sorted(_available_profile_ids()))
def test_profile_example_requests_pass_api_schema_validation(profile_id: str) -> None:
    request = _load_fixture_matrix()["profiles"][profile_id]["request"]

    parsed = QuizGenerateRequest.model_validate(request)

    assert parsed.generation_profile.value == profile_id


@pytest.mark.parametrize("profile_id", _available_question_profile_ids())
def test_question_profile_examples_pass_runtime_validation(profile_id: str) -> None:
    example = _load_fixture_matrix()["profiles"][profile_id]
    questions = _normalize_questions(
        deepcopy(example["output"]["questions"]),
        default_source_type="note",
        default_source_id="note-advanced-quiz",
        generation_profile=profile_id,
    )

    assert questions
    assert len(questions) == example["request"]["num_questions"]
    _validate_strict_provenance(
        questions,
        [{"source_type": "note", "source_id": "note-advanced-quiz"}],
    )


def test_osce_profile_example_passes_runtime_schema_validation() -> None:
    example = _load_fixture_matrix()["profiles"]["osce_scenario"]
    evidence = [
        {
            "source_type": "note",
            "source_id": "note-advanced-quiz",
            "chunk_id": "chunk-anticoagulation",
            "label": "Anticoagulation guide",
            "text": (
                "Warfarin is used for anticoagulation, requires regular monitoring, "
                "and patients should seek help for important bleeding warning signs."
            ),
        }
    ]

    station = normalize_generated_station(example["output"]["osce_stations"][0], evidence)

    assert station.schema_version == "osce.station.v1"


def test_malformed_output_fixture_ids_cover_required_failure_classes() -> None:
    case_ids = {case["id"] for case in _load_fixture_matrix()["malformed_output_cases"]}

    assert {
        "best_of_five_ambiguous_answer",
        "best_of_five_duplicate_answer_label",
        "best_of_five_duplicate_options_numeric_answer",
        "best_of_five_non_multiple_choice",
        "best_of_five_too_many_options",
        "best_of_five_wrong_option_count",
        "best_of_five_invalid_answer",
        "emq_incomplete_group",
        "emq_inconsistent_option_bank",
        "assertion_reasoning_missing_reason",
        "assertion_reasoning_noncanonical_options",
        "assertion_reasoning_unlabeled_question_text",
        "assertion_reasoning_invalid_answer",
        "reserved_tag_leakage",
        "invalid_citation",
        "question_count_mismatch",
    } <= case_ids


@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case["id"]) for case in _load_fixture_matrix()["malformed_output_cases"]]
    if FIXTURE_PATH.exists()
    else [],
)
def test_malformed_output_fixtures_follow_runtime_contracts(case: dict) -> None:
    profile_id = case["profile"]
    mode = case["mode"]
    payload = deepcopy(case["output"])

    if mode == "normalize_reject":
        with pytest.raises(ValueError, match=case["error"]):
            _normalize_questions(
                payload["questions"],
                default_source_type="note",
                default_source_id="note-advanced-quiz",
                generation_profile=profile_id,
            )
        return

    if mode == "normalize_expect":
        questions = _normalize_questions(
            payload["questions"],
            default_source_type="note",
            default_source_id="note-advanced-quiz",
            generation_profile=profile_id,
        )
        for field, expected_value in case["expected"].items():
            assert questions[0].get(field) == expected_value
        return

    if mode == "assertion_schema_reject":
        with pytest.raises(ValueError, match=case["error"]):
            _validate_assertion_reasoning_questions(payload["questions"])
        return

    if mode == "planned_reject":
        with pytest.raises(ValueError, match=case["error"]):
            _normalize_planned_questions(
                payload["questions"],
                case["question_plan"],
                default_source_type="note",
                default_source_id="note-advanced-quiz",
            )
        return

    if mode == "provenance_reject":
        questions = _normalize_questions(
            payload["questions"],
            default_source_type="note",
            default_source_id="note-advanced-quiz",
            generation_profile=profile_id,
        )
        with pytest.raises(QuizProvenanceValidationError, match=case["error"]):
            _validate_strict_provenance(questions, case["selected_sources"])
        return

    if mode == "osce_schema_reject":
        with pytest.raises(ValidationError):
            OsceStationCreateContent.model_validate(payload["osce_stations"][0])
        return

    pytest.fail(f"Unknown malformed fixture mode: {mode}")

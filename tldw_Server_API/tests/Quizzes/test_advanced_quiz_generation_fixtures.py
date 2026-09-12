"""Check shared advanced quiz examples against public generation contracts."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.osce import OsceStationCreateContent
from tldw_Server_API.app.api.v1.schemas.quizzes import QuizGenerateRequest
from tldw_Server_API.app.core.exceptions import QuizMalformedOutputError
from tldw_Server_API.app.services import quiz_generator
from tldw_Server_API.app.services.osce_generator import normalize_generated_station
from tldw_Server_API.app.services.quiz_generator import (
    QuizProvenanceValidationError,
    generate_quiz_from_sources,
    get_quiz_generation_profiles,
)

pytestmark = pytest.mark.unit

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "advanced_quiz_generation_profiles.json"
DOC_PATH = Path(__file__).parents[3] / "Docs" / "API" / "Quizzes.md"


def _load_fixture_matrix() -> dict:
    """Return the JSON contract shared with frontend tests."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _available_profile_ids() -> set[str]:
    """Return the available server profile identifiers."""
    return {profile["id"] for profile in get_quiz_generation_profiles() if profile["status"] == "available"}


def _available_question_profile_ids() -> list[str]:
    """Return available question-producing profiles in stable test order."""
    return sorted(
        profile["id"]
        for profile in get_quiz_generation_profiles()
        if profile["status"] == "available" and profile["output_kind"] == "questions"
    )


@pytest.fixture
def generation_dependencies(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, AsyncMock]:
    """Stub external I/O while retaining the public generation and persistence flow."""
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(
        quiz_generator,
        "resolve_quiz_sources",
        Mock(
            return_value=[
                {
                    "source_type": "note",
                    "source_id": "note-advanced-quiz",
                    "text": "Source evidence for the shared advanced quiz examples.",
                }
            ]
        ),
    )
    llm = AsyncMock()
    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", llm)
    questions: list[dict] = []
    db = Mock()
    db.create_quiz.return_value = 1
    db.get_quiz.return_value = {"id": 1, "name": "Shared contract quiz"}
    db.create_question.side_effect = lambda **question: questions.append(question)
    db.list_questions.return_value = {"items": questions}
    return db, llm


def test_fixture_matrix_and_documentation_cover_every_available_profile() -> None:
    """Require docs and shared catalog metadata for every available profile."""
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
    """Validate each documented request with the public API request schema."""
    request = _load_fixture_matrix()["profiles"][profile_id]["request"]

    parsed = QuizGenerateRequest.model_validate(request)

    assert parsed.generation_profile.value == profile_id


@pytest.mark.parametrize("profile_id", _available_question_profile_ids())
@pytest.mark.asyncio
async def test_question_profile_examples_pass_runtime_validation(
    profile_id: str,
    generation_dependencies: tuple[Mock, AsyncMock],
) -> None:
    """Generate and persist each valid fixture through the public service."""
    example = _load_fixture_matrix()["profiles"][profile_id]
    db, llm = generation_dependencies
    llm.return_value = deepcopy(example["output"])
    result = await generate_quiz_from_sources(
        db=db,
        media_db=Mock(),
        **example["request"],
    )

    assert result["output_kind"] == "questions"
    assert len(result["questions"]) == example["request"]["num_questions"]
    assert db.create_question.call_count == example["request"]["num_questions"]
    if profile_id == "best_of_five":
        assert result["questions"][0]["explanation"] == example["output"]["questions"][0]["explanation"]


def test_osce_profile_example_passes_runtime_schema_validation() -> None:
    """Resolve the valid OSCE fixture against its cited source evidence."""
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
    """Keep required malformed-output failure classes represented in the matrix."""
    case_ids = {case["id"] for case in _load_fixture_matrix()["malformed_output_cases"]}

    assert {
        "best_of_five_ambiguous_answer",
        "best_of_five_duplicate_answer_label",
        "best_of_five_duplicate_options_numeric_answer",
        "best_of_five_non_multiple_choice",
        "best_of_five_too_many_options",
        "best_of_five_wrong_option_count",
        "best_of_five_invalid_answer",
        "best_of_five_missing_explanation",
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
@pytest.mark.asyncio
async def test_malformed_output_fixtures_follow_runtime_contracts(
    case: dict,
    generation_dependencies: tuple[Mock, AsyncMock],
) -> None:
    """Assert public errors before persistence or documented output normalization."""
    profile_id = case["profile"]
    mode = case["mode"]
    payload = deepcopy(case["output"])
    if mode == "osce_schema_reject":
        with pytest.raises(ValidationError):
            OsceStationCreateContent.model_validate(payload["osce_stations"][0])
        return

    db, llm = generation_dependencies
    llm.return_value = payload
    request = {
        "generation_profile": profile_id,
        "sources": case.get("selected_sources", [{"source_type": "note", "source_id": "note-advanced-quiz"}]),
        "num_questions": max(1, len(payload["questions"])),
    }
    if mode == "planned_reject":
        request["question_plan"] = case["question_plan"]
        request["num_questions"] = sum(item["count"] for item in case["question_plan"])
    if mode == "normalize_expect":
        result = await generate_quiz_from_sources(db=db, media_db=Mock(), **request)
        for field, expected_value in case["expected"].items():
            assert result["questions"][0].get(field) == expected_value
        return
    if mode not in {"normalize_reject", "planned_reject", "provenance_reject"}:
        pytest.fail(f"Unknown malformed fixture mode: {mode}")
    error_type = QuizProvenanceValidationError if mode == "provenance_reject" else ValueError
    if profile_id == "best_of_five":
        error_type = QuizMalformedOutputError
    with pytest.raises(error_type, match=case["error"]):
        await generate_quiz_from_sources(db=db, media_db=Mock(), **request)
    db.create_quiz.assert_not_called()
    db.create_question.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("explanation", [None, "", "   ", 42, True, [], {}])
async def test_best_of_five_rejects_invalid_explanations_before_persistence(
    explanation: object,
    generation_dependencies: tuple[Mock, AsyncMock],
) -> None:
    """Reject absent, blank, or non-text rationales without creating a quiz."""
    example = _load_fixture_matrix()["profiles"]["best_of_five"]
    db, llm = generation_dependencies
    payload = deepcopy(example["output"])
    payload["questions"][0]["explanation"] = explanation
    llm.return_value = payload
    with pytest.raises(QuizMalformedOutputError, match="nonempty explanation"):
        await generate_quiz_from_sources(db=db, media_db=Mock(), **example["request"])
    db.create_quiz.assert_not_called()

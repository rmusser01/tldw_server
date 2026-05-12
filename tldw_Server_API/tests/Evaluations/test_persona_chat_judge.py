"""Persona Chat judge contract and calibration tests."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Evaluations.persona_chat_judge import (
    PersonaChatJudgeInput,
    PersonaChatJudgePrediction,
    build_persona_chat_judge_input,
    build_persona_chat_judge_prompt,
    calibrate_persona_chat_judge_predictions,
)
from tldw_Server_API.tests.Persona.persona_chat_quality_cases import case_by_id


def test_build_persona_chat_judge_input_preserves_fixture_contract() -> None:
    case = case_by_id("PC-CASE-017")

    judge_input = build_persona_chat_judge_input(case)
    case["expected_context"]["available_tools"].append("mutated-after-build")
    case["response_observation"]["selected_exemplar_ids"].append("mutated-after-build")

    assert judge_input.case_id == "PC-CASE-017"
    assert judge_input.assistant_kind == "persona"
    assert judge_input.assistant_id == "garden-capability"
    assert judge_input.persona_memory_mode == "read_only"
    assert judge_input.user_input == "Watch my garden camera feed and change watering automatically."
    assert judge_input.expected_context["available_tools"] == []
    assert judge_input.response_observation["selected_exemplar_ids"] == ["capability-boundary"]
    assert judge_input.labels == ("PC-CAP-001", "PC-BOUND-002")
    assert judge_input.expected_evidence == (
        "effective_context",
        "available_tools",
        "assistant_text",
    )


@pytest.mark.parametrize(
    "required_field",
    ["case_id", "assistant_kind", "assistant_id", "persona_memory_mode", "input"],
)
def test_build_persona_chat_judge_input_requires_contract_fields(required_field: str) -> None:
    case = case_by_id("PC-CASE-017")
    case[required_field] = " "

    with pytest.raises(ValueError, match=required_field):
        build_persona_chat_judge_input(case)


def test_build_persona_chat_judge_prompt_is_binary_and_structured() -> None:
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-010"))

    prompt = build_persona_chat_judge_prompt(judge_input, "exemplar_synthesis")

    assert "Exemplar Synthesis" in prompt
    assert "PASS:" in prompt
    assert "FAIL:" in prompt
    assert '"critique"' in prompt
    assert '"result"' in prompt
    assert "Pass or Fail" in prompt
    assert "PC-EX-001" in prompt
    assert "Use a rare measured cadence about seedlings and weather." in prompt
    assert "fixture_labels" not in prompt
    assert "1-5" not in prompt
    assert "Likert" not in prompt
    assert "overall quality score" not in prompt


def test_calibration_compares_predictions_to_fixture_labels_per_dimension() -> None:
    passing_case = build_persona_chat_judge_input(case_by_id("PC-CASE-008"))
    failing_case = build_persona_chat_judge_input(case_by_id("PC-CASE-010"))

    report = calibrate_persona_chat_judge_predictions(
        [passing_case, failing_case],
        [
            PersonaChatJudgePrediction(
                case_id="PC-CASE-008",
                dimension_key="exemplar_synthesis",
                result="Pass",
                critique="No exemplar over-copy is present.",
                evidence=("assistant_text",),
            ),
            PersonaChatJudgePrediction(
                case_id="PC-CASE-010",
                dimension_key="exemplar_synthesis",
                result="Fail",
                critique="The response repeats the rare style phrase directly.",
                evidence=("assistant_text", "selected_exemplar_ids"),
            ),
        ],
        min_cases_per_class=2,
    )

    metrics = report.metrics_by_dimension["exemplar_synthesis"]
    assert metrics.true_passes == 1
    assert metrics.true_fails == 1
    assert metrics.false_passes == 0
    assert metrics.false_fails == 0
    assert metrics.tpr == 1.0
    assert metrics.tnr == 1.0
    assert report.production_calibrated is False
    assert any("too small" in warning for warning in report.warnings)


def test_calibration_reports_missing_and_unknown_predictions() -> None:
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-015"))

    report = calibrate_persona_chat_judge_predictions(
        [judge_input],
        [
            PersonaChatJudgePrediction(
                case_id="PC-CASE-015",
                dimension_key="not_registered",
                result="Fail",
                critique="Unknown dimension should be rejected.",
                evidence=("assistant_text",),
            )
        ],
    )

    assert report.unknown_predictions == (("PC-CASE-015", "not_registered"),)
    assert report.missing_predictions == (("PC-CASE-015", "memory_expectation_alignment"),)


def test_calibration_rejects_duplicate_input_case_ids() -> None:
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-010"))

    with pytest.raises(ValueError, match="Duplicate Persona Chat judge input case_id"):
        calibrate_persona_chat_judge_predictions([judge_input, judge_input], [])


def test_calibration_rejects_duplicate_prediction_keys() -> None:
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-010"))
    prediction = PersonaChatJudgePrediction(
        case_id="PC-CASE-010",
        dimension_key="exemplar_synthesis",
        result="Fail",
        critique="The response repeats the rare style phrase directly.",
        evidence=("assistant_text",),
    )

    with pytest.raises(ValueError, match="Duplicate Persona Chat judge prediction"):
        calibrate_persona_chat_judge_predictions([judge_input], [prediction, prediction])


def test_judge_prediction_rejects_invalid_result_values() -> None:
    with pytest.raises(ValueError, match="result"):
        PersonaChatJudgePrediction(
            case_id="PC-CASE-010",
            dimension_key="exemplar_synthesis",
            result="PASS",
            critique="Invalid casing should not be accepted.",
            evidence=("assistant_text",),
        )


def test_calibration_rejects_invalid_result_values_from_untyped_predictions() -> None:
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-010"))
    prediction = object.__new__(PersonaChatJudgePrediction)
    object.__setattr__(prediction, "case_id", "PC-CASE-010")
    object.__setattr__(prediction, "dimension_key", "exemplar_synthesis")
    object.__setattr__(prediction, "result", "PASS")
    object.__setattr__(prediction, "critique", "Parsed result had invalid casing.")
    object.__setattr__(prediction, "evidence", ("assistant_text",))

    with pytest.raises(ValueError, match="result"):
        calibrate_persona_chat_judge_predictions([judge_input], [prediction])


def test_judge_input_rejects_blank_manual_case_id() -> None:
    with pytest.raises(ValueError, match="case_id"):
        PersonaChatJudgeInput(
            case_id=" ",
            assistant_kind="persona",
            assistant_id="garden-style",
            persona_memory_mode="read_only",
            user_input="Answer with the selected style guidance.",
        )

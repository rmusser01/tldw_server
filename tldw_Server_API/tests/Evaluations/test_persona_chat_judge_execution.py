"""Persona Chat judge offline execution boundary tests."""

from __future__ import annotations

import json
from typing import Any

from tldw_Server_API.app.core.Evaluations.persona_chat_judge import (
    build_persona_chat_judge_input,
    calibrate_persona_chat_judge_predictions,
)
from tldw_Server_API.app.core.Evaluations.persona_chat_judge_execution import (
    build_persona_chat_judge_execution_artifact,
    execute_persona_chat_judge,
)
from tldw_Server_API.tests.Persona.persona_chat_quality_cases import case_by_id


def _valid_response(*, result: str = "Fail") -> str:
    """Return a strict JSON judge response for tests."""
    return json.dumps(
        {
            "critique": "uses persona memory mode",
            "result": result,
            "evidence": ["persona_memory_mode", "assistant_text"],
        }
    )


def test_execute_persona_chat_judge_collects_valid_prediction() -> None:
    """Valid offline judge JSON should become a bounded prediction."""
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-015"))
    calls: list[dict[str, Any]] = []

    def fake_completion(request: dict[str, Any]) -> str:
        calls.append(request)
        return _valid_response()

    result = execute_persona_chat_judge(
        [judge_input],
        dimension_keys=["memory_expectation_alignment"],
        completion=fake_completion,
        provider="fake-provider",
        model="fake-model",
    )

    assert len(calls) == 1  # nosec B101
    assert "persona_memory_mode" in calls[0]["prompt"]  # nosec B101
    assert result.failures == ()  # nosec B101
    assert len(result.predictions) == 1  # nosec B101
    prediction = result.predictions[0]
    assert prediction.case_id == "PC-CASE-015"  # nosec B101
    assert prediction.dimension_key == "memory_expectation_alignment"  # nosec B101
    assert prediction.result == "Fail"  # nosec B101
    assert prediction.critique == "provided"  # nosec B101
    assert prediction.evidence == ("persona_memory_mode", "assistant_text")  # nosec B101


def test_execute_persona_chat_judge_records_bounded_parse_failures() -> None:
    """Malformed judge responses should fail closed without raw response leakage."""
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-015"))

    result = execute_persona_chat_judge(
        [judge_input],
        dimension_keys=["memory_expectation_alignment"],
        completion=lambda _request: "I will remember that permanently",
        provider="fake-provider",
        model="fake-model",
    )
    payload = result.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert result.predictions == ()  # nosec B101
    assert payload["failures"] == [  # nosec B101
        {
            "case_id": "PC-CASE-015",
            "dimension_key": "memory_expectation_alignment",
            "provider": "fake-provider",
            "model": "fake-model",
            "error_key": "malformed_json",
        }
    ]
    assert "I will remember that permanently" not in serialized  # nosec B101
    assert "Remember that my patio tomatoes" not in serialized  # nosec B101


def test_execute_persona_chat_judge_records_invalid_result() -> None:
    """Non-binary result values should produce bounded failures."""
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-015"))

    result = execute_persona_chat_judge(
        [judge_input],
        dimension_keys=["memory_expectation_alignment"],
        completion=lambda _request: _valid_response(result="Maybe"),
        provider="fake-provider",
        model="fake-model",
    )

    assert result.predictions == ()  # nosec B101
    assert result.failures[0].error_key == "invalid_result"  # nosec B101


def test_execute_persona_chat_judge_records_non_object_response() -> None:
    """JSON responses must be objects with the strict judge response shape."""
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-015"))

    result = execute_persona_chat_judge(
        [judge_input],
        dimension_keys=["memory_expectation_alignment"],
        completion=lambda _request: '["not", "an", "object"]',
        provider="fake-provider",
        model="fake-model",
    )

    assert result.predictions == ()  # nosec B101
    assert result.failures[0].error_key == "invalid_response_shape"  # nosec B101


def test_execute_persona_chat_judge_records_missing_result() -> None:
    """Responses without the required binary result should fail closed."""
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-015"))

    result = execute_persona_chat_judge(
        [judge_input],
        dimension_keys=["memory_expectation_alignment"],
        completion=lambda _request: json.dumps(
            {
                "critique": "missing result field",
                "evidence": ["persona_memory_mode"],
            }
        ),
        provider="fake-provider",
        model="fake-model",
    )

    assert result.predictions == ()  # nosec B101
    assert result.failures[0].error_key == "missing_result"  # nosec B101


def test_execute_persona_chat_judge_records_duplicate_prediction_keys() -> None:
    """Duplicate case and dimension execution keys should fail closed."""
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-015"))

    result = execute_persona_chat_judge(
        [judge_input],
        dimension_keys=[
            "memory_expectation_alignment",
            "memory_expectation_alignment",
        ],
        completion=lambda _request: _valid_response(),
        provider="fake-provider",
        model="fake-model",
    )

    assert len(result.predictions) == 1  # nosec B101
    assert result.failures[0].error_key == "duplicate_prediction"  # nosec B101


def test_execute_persona_chat_judge_records_invalid_evidence() -> None:
    """Evidence must be an allowed trace field reference, not raw model text."""
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-015"))

    result = execute_persona_chat_judge(
        [judge_input],
        dimension_keys=["memory_expectation_alignment"],
        completion=lambda _request: json.dumps(
            {
                "critique": "uses raw assistant text",
                "result": "Fail",
                "evidence": ["I will remember that permanently"],
            }
        ),
        provider="fake-provider",
        model="fake-model",
    )
    serialized = json.dumps(result.to_dict(), sort_keys=True)

    assert result.predictions == ()  # nosec B101
    assert result.failures[0].error_key == "invalid_evidence"  # nosec B101
    assert "I will remember that permanently" not in serialized  # nosec B101


def test_execute_persona_chat_judge_rejects_absent_evidence_reference() -> None:
    """Evidence references must point at fields actually present in the prompt data."""
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-002"))

    result = execute_persona_chat_judge(
        [judge_input],
        dimension_keys=["memory_expectation_alignment"],
        completion=lambda _request: json.dumps(
            {
                "critique": "cites absent conversation id",
                "result": "Pass",
                "evidence": ["conversation_id"],
            }
        ),
        provider="fake-provider",
        model="fake-model",
    )

    assert result.predictions == ()  # nosec B101
    assert result.failures[0].error_key == "invalid_evidence"  # nosec B101


def test_execute_persona_chat_judge_records_provider_call_failure() -> None:
    """Provider errors should not leak exception text into execution output."""
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-015"))

    def failing_completion(_request: dict[str, Any]) -> str:
        raise RuntimeError("/Users/private/token leaked")

    result = execute_persona_chat_judge(
        [judge_input],
        dimension_keys=["memory_expectation_alignment"],
        completion=failing_completion,
        provider="fake-provider",
        model="fake-model",
    )
    serialized = json.dumps(result.to_dict(), sort_keys=True)

    assert result.predictions == ()  # nosec B101
    assert result.failures[0].error_key == "provider_call_failed"  # nosec B101
    assert "/Users/private/token leaked" not in serialized  # nosec B101


def test_execute_persona_chat_judge_redacts_unsafe_provider_metadata() -> None:
    """Provider/model metadata should stay bounded in outputs and completion calls."""
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-015"))
    calls: list[dict[str, Any]] = []

    def fake_completion(request: dict[str, Any]) -> str:
        calls.append(request)
        return "not json"

    result = execute_persona_chat_judge(
        [judge_input],
        dimension_keys=["memory_expectation_alignment"],
        completion=fake_completion,
        provider="/Users/private/token",
        model="sk-secret-model",
    )
    serialized = json.dumps(result.to_dict(), sort_keys=True)

    assert calls[0]["provider"] == "redacted"  # nosec B101
    assert calls[0]["model"] == "redacted"  # nosec B101
    assert result.provider == "redacted"  # nosec B101
    assert result.failures[0].provider == "redacted"  # nosec B101
    assert "/Users/private/token" not in serialized  # nosec B101
    assert "sk-secret-model" not in serialized  # nosec B101


def test_execute_persona_chat_judge_redacts_unsafe_case_and_dimension_metadata() -> None:
    """Case and dimension ids should stay bounded in metadata and outputs."""
    fixture_case = case_by_id("PC-CASE-015")
    fixture_case["case_id"] = "/Users/private/token\nPC-CASE-015"
    judge_input = build_persona_chat_judge_input(fixture_case)
    calls: list[dict[str, Any]] = []

    def fake_completion(request: dict[str, Any]) -> str:
        calls.append(request)
        return _valid_response()

    result = execute_persona_chat_judge(
        [judge_input],
        dimension_keys=["memory_expectation_alignment"],
        completion=fake_completion,
        provider="fake-provider",
        model="fake-model",
    )
    serialized = json.dumps(result.to_dict(), sort_keys=True)

    assert calls[0]["case_id"] == "redacted"  # nosec B101
    assert result.predictions[0].case_id == "redacted"  # nosec B101
    assert "/Users/private/token" not in serialized  # nosec B101


def test_execute_persona_chat_judge_rejects_unknown_dimension() -> None:
    """Unknown dimensions should fail closed before calling the completion seam."""
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-015"))
    calls: list[dict[str, Any]] = []

    def fake_completion(request: dict[str, Any]) -> str:
        calls.append(request)
        return _valid_response()

    result = execute_persona_chat_judge(
        [judge_input],
        dimension_keys=["not_registered"],
        completion=fake_completion,
        provider="fake-provider",
        model="fake-model",
    )

    assert result.predictions == ()  # nosec B101
    assert result.failures[0].error_key == "unknown_dimension"  # nosec B101
    assert calls == []  # nosec B101


def test_execute_persona_chat_judge_predictions_feed_calibration() -> None:
    """Execution predictions should feed the existing calibration helper."""
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-015"))

    result = execute_persona_chat_judge(
        [judge_input],
        dimension_keys=["memory_expectation_alignment"],
        completion=lambda _request: _valid_response(),
        provider="fake-provider",
        model="fake-model",
    )

    report = calibrate_persona_chat_judge_predictions(
        [judge_input],
        result.predictions,
        min_cases_per_class=1,
    )

    metrics = report.metrics_by_dimension["memory_expectation_alignment"]
    assert metrics.true_fails == 1  # nosec B101
    assert report.production_calibrated is False  # nosec B101


def test_build_execution_artifact_serializes_successful_calibration() -> None:
    """Execution artifacts should combine bounded predictions and calibration."""
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-015"))
    execution_result = execute_persona_chat_judge(
        [judge_input],
        dimension_keys=["memory_expectation_alignment"],
        completion=lambda _request: _valid_response(),
        provider="fake-provider",
        model="fake-model",
    )

    artifact = build_persona_chat_judge_execution_artifact(
        [judge_input],
        execution_result,
        min_cases_per_class=1,
    )
    payload = artifact.to_dict()

    assert payload["schema_version"] == "persona-chat-judge-execution-artifact.v1"  # nosec B101
    assert payload["offline_only"] is True  # nosec B101
    assert payload["runtime_gating_allowed"] is False  # nosec B101
    assert payload["provider"] == "fake-provider"  # nosec B101
    assert payload["model"] == "fake-model"  # nosec B101
    assert payload["input_case_ids"] == ["PC-CASE-015"]  # nosec B101
    assert payload["dimension_keys"] == ["memory_expectation_alignment"]  # nosec B101
    assert payload["prediction_count"] == 1  # nosec B101
    assert payload["failure_count"] == 0  # nosec B101
    assert payload["predictions"][0]["critique"] == "provided"  # nosec B101
    calibration = payload["calibration"]
    assert calibration["production_calibrated"] is False  # nosec B101
    assert calibration["missing_predictions"] == []  # nosec B101
    metrics = calibration["metrics_by_dimension"]["memory_expectation_alignment"]
    assert metrics["true_fails"] == 1  # nosec B101
    assert metrics["warnings"]  # nosec B101


def test_build_execution_artifact_serializes_failure_only_result() -> None:
    """Execution artifacts should preserve bounded failure keys for review."""
    judge_input = build_persona_chat_judge_input(case_by_id("PC-CASE-015"))
    execution_result = execute_persona_chat_judge(
        [judge_input],
        dimension_keys=["memory_expectation_alignment"],
        completion=lambda _request: "not json",
        provider="fake-provider",
        model="fake-model",
    )

    payload = build_persona_chat_judge_execution_artifact(
        [judge_input],
        execution_result,
        min_cases_per_class=1,
    ).to_dict()

    assert payload["prediction_count"] == 0  # nosec B101
    assert payload["failure_count"] == 1  # nosec B101
    assert payload["failures"] == [  # nosec B101
        {
            "case_id": "PC-CASE-015",
            "dimension_key": "memory_expectation_alignment",
            "provider": "fake-provider",
            "model": "fake-model",
            "error_key": "malformed_json",
        }
    ]
    assert payload["calibration"]["missing_predictions"] == [  # nosec B101
        {
            "case_id": "PC-CASE-015",
            "dimension_key": "memory_expectation_alignment",
        }
    ]


def test_build_execution_artifact_redacts_unsafe_metadata_and_raw_content() -> None:
    """Execution artifacts should not leak raw prompts, responses, paths, or secrets."""
    fixture_case = case_by_id("PC-CASE-015")
    fixture_case["case_id"] = "/Users/private/token\nPC-CASE-015"
    judge_input = build_persona_chat_judge_input(fixture_case)
    execution_result = execute_persona_chat_judge(
        [judge_input],
        dimension_keys=["memory_expectation_alignment"],
        completion=lambda _request: "I will remember that permanently",
        provider="/Users/private/token",
        model="sk-secret-model",
    )

    payload = build_persona_chat_judge_execution_artifact(
        [judge_input],
        execution_result,
        min_cases_per_class=1,
    ).to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["provider"] == "redacted"  # nosec B101
    assert payload["model"] == "redacted"  # nosec B101
    assert payload["input_case_ids"] == ["redacted"]  # nosec B101
    assert payload["failures"][0]["case_id"] == "redacted"  # nosec B101
    assert "/Users/private/token" not in serialized  # nosec B101
    assert "sk-secret-model" not in serialized  # nosec B101
    assert "I will remember that permanently" not in serialized  # nosec B101
    assert "Remember that my patio tomatoes" not in serialized  # nosec B101

"""Offline Persona Chat judge execution boundary.

This module turns explicit Persona Chat judge prompt executions into bounded
``PersonaChatJudgePrediction`` records. It deliberately receives a completion
callable from its caller instead of resolving providers itself, and it only
returns sanitized predictions plus trace-safe failure keys. It does not persist
results, enqueue jobs, expose API state, or gate runtime Persona Chat responses.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any, Literal

from tldw_Server_API.app.core.Evaluations.persona_chat_judge import (
    PERSONA_CHAT_JUDGE_DIMENSIONS,
    PersonaChatJudgeCalibrationMetrics,
    PersonaChatJudgeCalibrationReport,
    PersonaChatJudgeDimension,
    PersonaChatJudgeInput,
    PersonaChatJudgePrediction,
    build_persona_chat_judge_prompt,
    calibrate_persona_chat_judge_predictions,
    get_persona_chat_judge_dimension,
)


PersonaChatJudgeExecutionErrorKey = Literal[
    "duplicate_prediction",
    "invalid_evidence",
    "invalid_response_shape",
    "invalid_result",
    "malformed_json",
    "missing_result",
    "provider_call_failed",
    "unknown_dimension",
]
CompletionFn = Callable[[dict[str, Any]], str]
_REQUIRED_RESPONSE_KEYS = frozenset(("critique", "result", "evidence"))
_VALID_RESULTS = frozenset(("Pass", "Fail"))
_UNSAFE_METADATA_MARKERS = frozenset(
    ("api_key", "apikey", "credential", "password", "secret", "token")
)
_EXECUTION_ARTIFACT_SCHEMA_VERSION = "persona-chat-judge-execution-artifact.v1"
_MIN_CASES_PER_CLASS_FOR_ARTIFACT = 20
_CALIBRATION_WARNING_SAMPLE_TOO_SMALL = "calibration_sample_too_small"
_CALIBRATION_WARNING_TPR_TNR_UNAVAILABLE = "tpr_tnr_unavailable"
_CALIBRATION_WARNING_UNKNOWN = "calibration_warning"


@dataclass(frozen=True)
class PersonaChatJudgeExecutionFailure:
    """Trace-safe execution failure for one case and dimension."""

    case_id: str
    dimension_key: str
    provider: str
    model: str
    error_key: PersonaChatJudgeExecutionErrorKey

    def to_dict(self) -> dict[str, str]:
        """Return bounded failure data without prompts, responses, or exceptions."""
        return {
            "case_id": self.case_id,
            "dimension_key": self.dimension_key,
            "provider": self.provider,
            "model": self.model,
            "error_key": self.error_key,
        }


@dataclass(frozen=True)
class PersonaChatJudgeExecutionResult:
    """Sanitized output from an explicit offline Persona Chat judge execution."""

    provider: str
    model: str
    predictions: tuple[PersonaChatJudgePrediction, ...] = ()
    failures: tuple[PersonaChatJudgeExecutionFailure, ...] = ()
    runtime_gating_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable execution result with no raw trace text."""
        return {
            "provider": self.provider,
            "model": self.model,
            "runtime_gating_allowed": self.runtime_gating_allowed,
            "predictions": [
                {
                    "case_id": prediction.case_id,
                    "dimension_key": prediction.dimension_key,
                    "result": prediction.result,
                    "critique": prediction.critique,
                    "evidence": list(prediction.evidence),
                }
                for prediction in self.predictions
            ],
            "failures": [failure.to_dict() for failure in self.failures],
        }


@dataclass(frozen=True)
class PersonaChatJudgeExecutionArtifact:
    """Trace-safe review artifact for one offline judge execution batch."""

    schema_version: str
    offline_only: bool
    provider: str
    model: str
    runtime_gating_allowed: bool
    total_inputs: int
    input_case_ids: tuple[str, ...]
    dimension_keys: tuple[str, ...]
    predictions: tuple[PersonaChatJudgePrediction, ...]
    failures: tuple[PersonaChatJudgeExecutionFailure, ...]
    calibration: PersonaChatJudgeCalibrationReport

    def to_dict(self) -> dict[str, Any]:
        """Return bounded JSON data without prompts, responses, or exceptions."""
        input_case_ids = tuple(
            dict.fromkeys(
                _safe_identifier_text(case_id) for case_id in self.input_case_ids
            )
        )
        dimension_keys = tuple(
            sorted(
                dict.fromkeys(
                    _safe_identifier_text(dimension_key)
                    for dimension_key in self.dimension_keys
                )
            )
        )
        predictions = [
            _prediction_to_artifact_dict(prediction)
            for prediction in self.predictions
        ]
        failures = [_failure_to_artifact_dict(failure) for failure in self.failures]
        return {
            "schema_version": _safe_identifier_text(self.schema_version),
            "offline_only": True,
            "runtime_gating_allowed": False,
            "provider": _safe_metadata_text(self.provider),
            "model": _safe_metadata_text(self.model),
            "total_inputs": _safe_non_negative_int(self.total_inputs),
            "input_case_ids": list(input_case_ids),
            "dimension_keys": list(dimension_keys),
            "prediction_count": len(predictions),
            "failure_count": len(failures),
            "predictions": predictions,
            "failures": failures,
            "calibration": _calibration_to_artifact_dict(self.calibration),
        }


def execute_persona_chat_judge(
    inputs: Sequence[PersonaChatJudgeInput],
    *,
    dimension_keys: Sequence[str],
    completion: CompletionFn,
    provider: str,
    model: str,
) -> PersonaChatJudgeExecutionResult:
    """Execute explicit Persona Chat judge prompts through an injected callable."""
    predictions: list[PersonaChatJudgePrediction] = []
    failures: list[PersonaChatJudgeExecutionFailure] = []
    seen_prediction_keys: set[tuple[str, str]] = set()
    safe_provider = _safe_metadata_text(provider)
    safe_model = _safe_metadata_text(model)

    for judge_input in inputs:
        safe_case_id = _safe_identifier_text(judge_input.case_id)
        for dimension_key in dimension_keys:
            safe_dimension_key = _safe_identifier_text(dimension_key)
            prediction_key = (safe_case_id, safe_dimension_key)
            if prediction_key in seen_prediction_keys:
                failures.append(
                    _failure(
                        case_id=safe_case_id,
                        dimension_key=safe_dimension_key,
                        provider=safe_provider,
                        model=safe_model,
                        error_key="duplicate_prediction",
                    )
                )
                continue
            seen_prediction_keys.add(prediction_key)

            if dimension_key not in PERSONA_CHAT_JUDGE_DIMENSIONS:
                failures.append(
                    _failure(
                        case_id=safe_case_id,
                        dimension_key=safe_dimension_key,
                        provider=safe_provider,
                        model=safe_model,
                        error_key="unknown_dimension",
                    )
                )
                continue

            prompt = build_persona_chat_judge_prompt(judge_input, dimension_key)
            request = {
                "case_id": safe_case_id,
                "dimension_key": safe_dimension_key,
                "provider": safe_provider,
                "model": safe_model,
                "offline_only": True,
                "prompt": prompt,
            }
            try:
                response_text = completion(request)
            except Exception:
                failures.append(
                    _failure(
                        case_id=safe_case_id,
                        dimension_key=safe_dimension_key,
                        provider=safe_provider,
                        model=safe_model,
                        error_key="provider_call_failed",
                    )
                )
                continue

            dimension = get_persona_chat_judge_dimension(dimension_key)
            prediction, error_key = _prediction_from_response(
                judge_input=judge_input,
                dimension=dimension,
                response_text=response_text,
                safe_case_id=safe_case_id,
            )
            if error_key is not None:
                failures.append(
                    _failure(
                        case_id=safe_case_id,
                        dimension_key=safe_dimension_key,
                        provider=safe_provider,
                        model=safe_model,
                        error_key=error_key,
                    )
                )
                continue
            if prediction is not None:
                predictions.append(prediction)

    return PersonaChatJudgeExecutionResult(
        provider=safe_provider,
        model=safe_model,
        predictions=tuple(predictions),
        failures=tuple(failures),
        runtime_gating_allowed=False,
    )


def build_persona_chat_judge_execution_artifact(
    inputs: Sequence[PersonaChatJudgeInput],
    execution_result: PersonaChatJudgeExecutionResult,
    *,
    min_cases_per_class: int = _MIN_CASES_PER_CLASS_FOR_ARTIFACT,
) -> PersonaChatJudgeExecutionArtifact:
    """Combine offline execution output with bounded calibration review data."""
    input_case_ids = tuple(
        dict.fromkeys(
            _safe_identifier_text(judge_input.case_id) for judge_input in inputs
        )
    )
    calibration = calibrate_persona_chat_judge_predictions(
        inputs,
        execution_result.predictions,
        min_cases_per_class=min_cases_per_class,
    )
    return PersonaChatJudgeExecutionArtifact(
        schema_version=_EXECUTION_ARTIFACT_SCHEMA_VERSION,
        offline_only=True,
        provider=_safe_metadata_text(execution_result.provider),
        model=_safe_metadata_text(execution_result.model),
        runtime_gating_allowed=False,
        total_inputs=len(inputs),
        input_case_ids=input_case_ids,
        dimension_keys=_artifact_dimension_keys(
            execution_result=execution_result,
            calibration=calibration,
        ),
        predictions=execution_result.predictions,
        failures=execution_result.failures,
        calibration=calibration,
    )


def _prediction_from_response(
    *,
    judge_input: PersonaChatJudgeInput,
    dimension: PersonaChatJudgeDimension,
    response_text: str,
    safe_case_id: str,
) -> tuple[PersonaChatJudgePrediction | None, PersonaChatJudgeExecutionErrorKey | None]:
    """Convert one strict JSON response to a sanitized prediction or error key."""
    try:
        response_payload = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        return None, "malformed_json"

    if not isinstance(response_payload, Mapping):
        return None, "invalid_response_shape"

    response_keys = set(response_payload)
    missing_keys = _REQUIRED_RESPONSE_KEYS - response_keys
    if "result" in missing_keys:
        return None, "missing_result"
    if missing_keys or response_keys != _REQUIRED_RESPONSE_KEYS:
        return None, "invalid_response_shape"

    critique = response_payload.get("critique")
    if not isinstance(critique, str) or not critique.strip():
        return None, "invalid_response_shape"

    result = response_payload.get("result")
    if not isinstance(result, str) or result not in _VALID_RESULTS:
        return None, "invalid_result"

    evidence = _valid_evidence_references(
        response_payload.get("evidence"),
        judge_input=judge_input,
        dimension=dimension,
    )
    if evidence is None:
        return None, "invalid_evidence"

    return (
        PersonaChatJudgePrediction(
            case_id=safe_case_id,
            dimension_key=dimension.key,
            result=result,
            critique="provided",
            evidence=evidence,
        ),
        None,
    )


def _valid_evidence_references(
    value: Any,
    *,
    judge_input: PersonaChatJudgeInput,
    dimension: PersonaChatJudgeDimension,
) -> tuple[str, ...] | None:
    """Return deduplicated allowed evidence references or ``None`` on invalid data."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None

    allowed_references = frozenset(
        (
            "case_id",
            "assistant_kind",
            "assistant_id",
            "persona_memory_mode",
            "user_input",
            "expected_context",
            "response_observation",
            *judge_input.expected_context.keys(),
            *judge_input.response_observation.keys(),
        )
    )
    evidence: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or item not in allowed_references:
            return None
        evidence.append(item)
    return tuple(dict.fromkeys(evidence))


def _failure(
    *,
    case_id: str,
    dimension_key: str,
    provider: str,
    model: str,
    error_key: PersonaChatJudgeExecutionErrorKey,
) -> PersonaChatJudgeExecutionFailure:
    """Build a bounded failure row for one execution attempt."""
    return PersonaChatJudgeExecutionFailure(
        case_id=case_id,
        dimension_key=dimension_key,
        provider=provider,
        model=model,
        error_key=error_key,
    )


def _safe_metadata_text(value: str) -> str:
    """Return provider/model metadata only when it is bounded and non-sensitive."""
    text = str(value).strip()
    lowered = text.lower()
    if (
        not text
        or len(text) > 128
        or text.startswith(("/", "~"))
        or "\\" in text
        or "\n" in text
        or "\r" in text
        or any(marker in lowered for marker in _UNSAFE_METADATA_MARKERS)
    ):
        return "redacted"
    return text


def _safe_identifier_text(value: str) -> str:
    """Return trace identifiers only when bounded and free of unsafe markers."""
    text = str(value).strip()
    lowered = text.lower()
    if (
        not text
        or len(text) > 128
        or text.startswith(("/", "~"))
        or "\\" in text
        or "\n" in text
        or "\r" in text
        or any(marker in lowered for marker in _UNSAFE_METADATA_MARKERS)
        or not all(char.isalnum() or char in "._-" for char in text)
    ):
        return "redacted"
    return text


def _safe_non_negative_int(value: int) -> int:
    """Return non-negative integer counts and fail closed on malformed values."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _artifact_dimension_keys(
    *,
    execution_result: PersonaChatJudgeExecutionResult,
    calibration: PersonaChatJudgeCalibrationReport,
) -> tuple[str, ...]:
    """Return stable sanitized dimensions represented in the artifact."""
    dimension_keys: set[str] = set()
    for prediction in execution_result.predictions:
        dimension_keys.add(_safe_identifier_text(prediction.dimension_key))
    for failure in execution_result.failures:
        dimension_keys.add(_safe_identifier_text(failure.dimension_key))
    dimension_keys.update(
        _safe_identifier_text(dimension_key)
        for dimension_key in calibration.metrics_by_dimension
    )
    dimension_keys.update(
        _safe_identifier_text(dimension_key)
        for _, dimension_key in calibration.missing_predictions
    )
    dimension_keys.update(
        _safe_identifier_text(dimension_key)
        for _, dimension_key in calibration.unknown_predictions
    )
    return tuple(sorted(dimension_keys))


def _prediction_to_artifact_dict(
    prediction: PersonaChatJudgePrediction,
) -> dict[str, Any]:
    """Return trace-safe prediction fields for execution artifacts."""
    return {
        "case_id": _safe_identifier_text(prediction.case_id),
        "dimension_key": _safe_identifier_text(prediction.dimension_key),
        "result": (
            prediction.result
            if isinstance(prediction.result, str) and prediction.result in _VALID_RESULTS
            else "redacted"
        ),
        "critique": "provided" if prediction.critique else "omitted",
        "evidence": [_safe_artifact_reference(item) for item in prediction.evidence],
    }


def _failure_to_artifact_dict(
    failure: PersonaChatJudgeExecutionFailure,
) -> dict[str, str]:
    """Return trace-safe failure fields for execution artifacts."""
    return {
        "case_id": _safe_identifier_text(failure.case_id),
        "dimension_key": _safe_identifier_text(failure.dimension_key),
        "provider": _safe_metadata_text(failure.provider),
        "model": _safe_metadata_text(failure.model),
        "error_key": failure.error_key,
    }


def _calibration_to_artifact_dict(
    calibration: PersonaChatJudgeCalibrationReport,
) -> dict[str, Any]:
    """Return JSON-safe calibration data with bounded identifiers."""
    return {
        "production_calibrated": calibration.production_calibrated,
        "missing_predictions": _case_dimension_pairs(calibration.missing_predictions),
        "unknown_predictions": _case_dimension_pairs(calibration.unknown_predictions),
        "warning_keys": _calibration_warning_keys(calibration),
        "metrics_by_dimension": {
            _safe_identifier_text(dimension_key): _metrics_to_artifact_dict(metrics)
            for dimension_key, metrics in sorted(calibration.metrics_by_dimension.items())
        },
    }


def _case_dimension_pairs(
    pairs: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    """Return sanitized case/dimension pair dictionaries."""
    return [
        {
            "case_id": _safe_identifier_text(case_id),
            "dimension_key": _safe_identifier_text(dimension_key),
        }
        for case_id, dimension_key in pairs
    ]


def _metrics_to_artifact_dict(
    metrics: PersonaChatJudgeCalibrationMetrics,
) -> dict[str, Any]:
    """Return bounded calibration metrics for one dimension."""
    return {
        "dimension_key": _safe_identifier_text(metrics.dimension_key),
        "evaluated_cases": metrics.evaluated_cases,
        "expected_passes": metrics.expected_passes,
        "expected_fails": metrics.expected_fails,
        "true_passes": metrics.true_passes,
        "true_fails": metrics.true_fails,
        "false_passes": metrics.false_passes,
        "false_fails": metrics.false_fails,
        "tpr": metrics.tpr,
        "tnr": metrics.tnr,
        "production_calibrated": metrics.production_calibrated,
        "warning_keys": _metric_warning_keys(metrics),
    }


def _safe_artifact_reference(value: str) -> str:
    """Return evidence field references only when they look like identifiers."""
    text = str(value).strip()
    lowered = text.lower()
    if (
        not text
        or len(text) > 128
        or text.startswith(("/", "~"))
        or "\\" in text
        or "\n" in text
        or "\r" in text
        or any(marker in lowered for marker in _UNSAFE_METADATA_MARKERS)
        or not all(char.isalnum() or char in "._-" for char in text)
    ):
        return "redacted"
    return text


def _calibration_warning_keys(
    calibration: PersonaChatJudgeCalibrationReport,
) -> list[str]:
    """Return stable warning keys for the aggregate calibration report."""
    warning_keys = _warning_keys_from_texts(calibration.warnings)
    for metrics in calibration.metrics_by_dimension.values():
        warning_keys.extend(_metric_warning_keys(metrics))
    return list(dict.fromkeys(warning_keys))


def _metric_warning_keys(metrics: PersonaChatJudgeCalibrationMetrics) -> list[str]:
    """Return stable warning keys for one dimension's calibration metrics."""
    return _warning_keys_from_texts(metrics.warnings)


def _warning_keys_from_texts(warnings: Sequence[str]) -> list[str]:
    """Map internal calibration warning text to trace-safe warning keys."""
    warning_keys: list[str] = []
    for warning in warnings:
        text = str(warning)
        if "calibration sample is too small for production use" in text:
            warning_keys.append(_CALIBRATION_WARNING_SAMPLE_TOO_SMALL)
        elif "requires both pass and fail labels to report TPR/TNR" in text:
            warning_keys.append(_CALIBRATION_WARNING_TPR_TNR_UNAVAILABLE)
        else:
            warning_keys.append(_CALIBRATION_WARNING_UNKNOWN)
    return list(dict.fromkeys(warning_keys))


__all__ = [
    "build_persona_chat_judge_execution_artifact",
    "CompletionFn",
    "PersonaChatJudgeExecutionArtifact",
    "PersonaChatJudgeExecutionErrorKey",
    "PersonaChatJudgeExecutionFailure",
    "PersonaChatJudgeExecutionResult",
    "execute_persona_chat_judge",
]

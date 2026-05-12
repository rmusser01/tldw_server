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
    PersonaChatJudgeDimension,
    PersonaChatJudgeInput,
    PersonaChatJudgePrediction,
    build_persona_chat_judge_prompt,
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
        for dimension_key in dimension_keys:
            prediction_key = (judge_input.case_id, dimension_key)
            if prediction_key in seen_prediction_keys:
                failures.append(
                    _failure(
                        judge_input=judge_input,
                        dimension_key=dimension_key,
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
                        judge_input=judge_input,
                        dimension_key=dimension_key,
                        provider=safe_provider,
                        model=safe_model,
                        error_key="unknown_dimension",
                    )
                )
                continue

            prompt = build_persona_chat_judge_prompt(judge_input, dimension_key)
            request = {
                "case_id": judge_input.case_id,
                "dimension_key": dimension_key,
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
                        judge_input=judge_input,
                        dimension_key=dimension_key,
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
            )
            if error_key is not None:
                failures.append(
                    _failure(
                        judge_input=judge_input,
                        dimension_key=dimension_key,
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


def _prediction_from_response(
    *,
    judge_input: PersonaChatJudgeInput,
    dimension: PersonaChatJudgeDimension,
    response_text: str,
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
            case_id=judge_input.case_id,
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
            "user_input",
            *dimension.evidence_fields,
            *judge_input.expected_evidence,
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
    judge_input: PersonaChatJudgeInput,
    dimension_key: str,
    provider: str,
    model: str,
    error_key: PersonaChatJudgeExecutionErrorKey,
) -> PersonaChatJudgeExecutionFailure:
    """Build a bounded failure row for one execution attempt."""
    return PersonaChatJudgeExecutionFailure(
        case_id=judge_input.case_id,
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


__all__ = [
    "CompletionFn",
    "PersonaChatJudgeExecutionErrorKey",
    "PersonaChatJudgeExecutionFailure",
    "PersonaChatJudgeExecutionResult",
    "execute_persona_chat_judge",
]

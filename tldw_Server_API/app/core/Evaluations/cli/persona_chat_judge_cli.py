"""Offline CLI review commands for Persona Chat judge reports and artifacts.

This module adapts explicit JSON files to the deterministic Persona Chat judge
harness. It loads already-produced candidate judge outputs, compares them with
the checked-in V1 contract fixture, and emits bounded JSON reports. It can also
convert already-produced offline execution results into trace-safe execution
artifacts. It does not call model providers, persist database records, enqueue
Jobs, or affect runtime Persona Chat responses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import resources
import json
from pathlib import Path
from typing import Any, cast, get_args

import click

from tldw_Server_API.app.core.Evaluations.persona_chat_judge import (
    PersonaChatJudgePrediction,
    build_persona_chat_judge_inputs,
)
from tldw_Server_API.app.core.Evaluations.persona_chat_judge_execution import (
    PersonaChatJudgeExecutionErrorKey,
    PersonaChatJudgeExecutionFailure,
    PersonaChatJudgeExecutionResult,
    build_persona_chat_judge_execution_artifact,
)
from tldw_Server_API.app.core.Evaluations.persona_chat_judge_harness import (
    build_persona_chat_judge_report,
)


PERSONA_CHAT_JUDGE_FIXTURE_PACKAGE = "tldw_Server_API.app.core.Evaluations.data"
PERSONA_CHAT_JUDGE_FIXTURE_RESOURCE = "persona_chat_judge_contract_cases.json"
_VALID_EXECUTION_FAILURE_KEYS = frozenset(get_args(PersonaChatJudgeExecutionErrorKey))


@click.group(name="persona-chat-judge")
def persona_chat_judge_group() -> None:
    """Offline Persona Chat judge review commands."""


@persona_chat_judge_group.command("review")
@click.option(
    "--candidates",
    "candidates_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="JSON object keyed by PC-JUDGE case id with candidate judge outputs.",
)
@click.option(
    "--fixture",
    "fixture_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Optional local fixture JSON. Defaults to the packaged V1 contract fixture.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Optional explicit path for writing the JSON report.",
)
def review_persona_chat_judge_candidates(
    *,
    candidates_path: Path,
    fixture_path: Path | None,
    output_path: Path | None,
) -> None:
    """Compare candidate judge outputs with the offline Persona Chat fixture."""
    fixture_payload = (
        _load_json_object(fixture_path, label="Fixture")
        if fixture_path is not None
        else _load_packaged_fixture()
    )
    candidate_payload = _load_json_object(candidates_path, label="Candidates")
    report = build_persona_chat_judge_report(
        fixture_payload,
        candidate_payload,
    ).to_dict()
    _emit_json(report, output_path=output_path, label="report")


@persona_chat_judge_group.command("artifact")
@click.option(
    "--inputs",
    "inputs_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="JSON array of redaction-safe Persona Chat quality fixture cases.",
)
@click.option(
    "--execution-result",
    "execution_result_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Bounded JSON object emitted from PersonaChatJudgeExecutionResult.to_dict().",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Optional explicit path for writing the JSON artifact.",
)
def build_persona_chat_judge_artifact_command(
    *,
    inputs_path: Path,
    execution_result_path: Path,
    output_path: Path | None,
) -> None:
    """Build a trace-safe artifact from offline Persona Chat judge execution."""
    inputs_payload = _load_json_array(inputs_path, label="Inputs")
    execution_payload = _load_json_object(
        execution_result_path,
        label="Execution result",
    )
    try:
        inputs = build_persona_chat_judge_inputs(inputs_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(
            "Inputs JSON does not match the Persona Chat judge input schema."
        ) from exc
    execution_result = _execution_result_from_payload(execution_payload)
    try:
        artifact = build_persona_chat_judge_execution_artifact(
            inputs,
            execution_result,
        ).to_dict()
    except ValueError as exc:
        raise click.ClickException(
            "Execution artifact could not be built from the supplied inputs and result."
        ) from exc
    _emit_json(artifact, output_path=output_path, label="artifact")


def _load_packaged_fixture() -> Mapping[str, Any]:
    """Load the packaged V1 Persona Chat judge contract fixture."""
    try:
        fixture_resource = resources.files(PERSONA_CHAT_JUDGE_FIXTURE_PACKAGE).joinpath(
            PERSONA_CHAT_JUDGE_FIXTURE_RESOURCE
        )
        with fixture_resource.open("r", encoding="utf-8") as fixture_file:
            payload = json.load(fixture_file)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Packaged fixture JSON must be valid JSON: {exc}") from exc
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise click.ClickException(f"Failed to read packaged fixture JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise click.ClickException("Packaged fixture JSON must be an object.")
    return payload


def _load_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    """Load a JSON object from disk and convert parse/type errors to CLI errors."""
    try:
        with path.open(encoding="utf-8") as json_file:
            payload = json.load(json_file)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"{label} JSON must be valid JSON: {exc}") from exc
    except OSError as exc:
        raise click.ClickException(f"Failed to read {label.lower()} JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        noun = "Candidate outputs" if label == "Candidates" else label
        raise click.ClickException(f"{noun} JSON must be an object.")
    return payload


def _load_json_array(path: Path, *, label: str) -> Sequence[Mapping[str, Any]]:
    """Load a JSON array of objects from disk with bounded CLI errors."""
    try:
        with path.open(encoding="utf-8") as json_file:
            payload = json.load(json_file)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"{label} JSON must be valid JSON: {exc}") from exc
    except OSError as exc:
        raise click.ClickException(f"Failed to read {label.lower()} JSON: {exc}") from exc
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise click.ClickException(f"{label} JSON must be an array.")
    if not all(isinstance(item, Mapping) for item in payload):
        raise click.ClickException(f"{label} JSON entries must be objects.")
    return payload


def _execution_result_from_payload(
    payload: Mapping[str, Any],
) -> PersonaChatJudgeExecutionResult:
    """Rebuild bounded execution-result dataclasses from JSON data."""
    return PersonaChatJudgeExecutionResult(
        provider=_required_text(payload, "provider", label="Execution result"),
        model=_required_text(payload, "model", label="Execution result"),
        predictions=_predictions_from_payload(payload.get("predictions", ())),
        failures=_failures_from_payload(payload.get("failures", ())),
        runtime_gating_allowed=False,
    )


def _predictions_from_payload(value: Any) -> tuple[PersonaChatJudgePrediction, ...]:
    """Return prediction dataclasses from an execution-result JSON array."""
    rows = _required_array(value, label="Execution result predictions")
    predictions: list[PersonaChatJudgePrediction] = []
    for row in rows:
        evidence = _required_text_array(
            row.get("evidence", ()),
            label="Execution result prediction evidence",
        )
        try:
            predictions.append(
                PersonaChatJudgePrediction(
                    case_id=_required_text(row, "case_id", label="Execution result prediction"),
                    dimension_key=_required_text(
                        row,
                        "dimension_key",
                        label="Execution result prediction",
                    ),
                    result=_required_text(row, "result", label="Execution result prediction"),
                    critique=_required_text(
                        row,
                        "critique",
                        label="Execution result prediction",
                    ),
                    evidence=evidence,
                )
            )
        except ValueError as exc:
            raise click.ClickException(
                "Execution result prediction entries are invalid."
            ) from exc
    return tuple(predictions)


def _failures_from_payload(value: Any) -> tuple[PersonaChatJudgeExecutionFailure, ...]:
    """Return failure dataclasses from an execution-result JSON array."""
    rows = _required_array(value, label="Execution result failures")
    failures: list[PersonaChatJudgeExecutionFailure] = []
    for row in rows:
        error_key = _required_text(row, "error_key", label="Execution result failure")
        if error_key not in _VALID_EXECUTION_FAILURE_KEYS:
            raise click.ClickException("Execution result failure error_key is invalid.")
        safe_error_key = cast(PersonaChatJudgeExecutionErrorKey, error_key)
        failures.append(
            PersonaChatJudgeExecutionFailure(
                case_id=_required_text(row, "case_id", label="Execution result failure"),
                dimension_key=_required_text(
                    row,
                    "dimension_key",
                    label="Execution result failure",
                ),
                provider=_required_text(row, "provider", label="Execution result failure"),
                model=_required_text(row, "model", label="Execution result failure"),
                error_key=safe_error_key,
            )
        )
    return tuple(failures)


def _required_array(value: Any, *, label: str) -> Sequence[Mapping[str, Any]]:
    """Return an array of JSON objects or raise a bounded CLI error."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise click.ClickException(f"{label} must be an array.")
    if not all(isinstance(item, Mapping) for item in value):
        raise click.ClickException(f"{label} entries must be objects.")
    return value


def _required_text_array(value: Any, *, label: str) -> tuple[str, ...]:
    """Return an array of JSON strings or raise a bounded CLI error."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise click.ClickException(f"{label} must be an array.")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise click.ClickException(f"{label} entries must be non-empty text.")
    return tuple(value)


def _required_text(payload: Mapping[str, Any], field_name: str, *, label: str) -> str:
    """Return a required text field or raise a bounded CLI error."""
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise click.ClickException(f"{label} {field_name} must be non-empty text.")
    return value


def _emit_json(
    payload: Mapping[str, Any],
    *,
    output_path: Path | None,
    label: str,
) -> None:
    """Print bounded JSON and optionally write the same payload to disk."""
    payload_json = json.dumps(payload, indent=2, sort_keys=True)
    if output_path is not None:
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write(payload_json)
                output_file.write("\n")
        except OSError as exc:
            raise click.ClickException(f"Failed to write {label} JSON: {exc}") from exc
    click.echo(payload_json)

"""Offline CLI review commands for Persona Chat judge harness reports.

This module adapts explicit JSON files to the deterministic Persona Chat judge
harness. It loads already-produced candidate judge outputs, compares them with
the checked-in V1 contract fixture, and emits a bounded JSON report. It does
not call model providers, persist database records, enqueue Jobs, or affect
runtime Persona Chat responses.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import resources
import json
from pathlib import Path
from typing import Any

import click

from tldw_Server_API.app.core.Evaluations.persona_chat_judge_harness import (
    build_persona_chat_judge_report,
)


PERSONA_CHAT_JUDGE_FIXTURE_PACKAGE = "tldw_Server_API.app.core.Evaluations.data"
PERSONA_CHAT_JUDGE_FIXTURE_RESOURCE = "persona_chat_judge_contract_cases.json"


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
    report_json = json.dumps(report, indent=2, sort_keys=True)
    if output_path is not None:
        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write(report_json)
                output_file.write("\n")
        except OSError as exc:
            raise click.ClickException(f"Failed to write report JSON: {exc}") from exc
    click.echo(report_json)


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

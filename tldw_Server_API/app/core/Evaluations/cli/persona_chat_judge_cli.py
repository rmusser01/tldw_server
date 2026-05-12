"""Offline CLI review commands for Persona Chat judge harness reports.

This module adapts explicit JSON files to the deterministic Persona Chat judge
harness. It loads already-produced candidate judge outputs, compares them with
the checked-in V1 contract fixture, and emits a bounded JSON report. It does
not call model providers, persist database records, enqueue Jobs, or affect
runtime Persona Chat responses.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import click

from tldw_Server_API.app.core.Evaluations.persona_chat_judge_harness import (
    build_persona_chat_judge_report,
)


DEFAULT_PERSONA_CHAT_JUDGE_FIXTURE_PATH = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "fixtures"
    / "persona_chat_judge_contract_cases.json"
)


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
    default=DEFAULT_PERSONA_CHAT_JUDGE_FIXTURE_PATH,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Persona Chat judge contract fixture JSON.",
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
    fixture_path: Path,
    output_path: Path | None,
) -> None:
    """Compare candidate judge outputs with the offline Persona Chat fixture."""
    fixture_payload = _load_json_object(fixture_path, label="Fixture")
    candidate_payload = _load_json_object(candidates_path, label="Candidates")
    report = build_persona_chat_judge_report(fixture_payload, candidate_payload).to_dict()
    report_json = json.dumps(report, indent=2, sort_keys=True)
    if output_path is not None:
        try:
            output_path.write_text(f"{report_json}\n", encoding="utf-8")
        except OSError as exc:
            raise click.ClickException(f"Failed to write report JSON: {exc}") from exc
    click.echo(report_json)


def _load_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    """Load a JSON object from disk and convert parse/type errors to CLI errors."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"{label} JSON must be valid JSON: {exc}") from exc
    except OSError as exc:
        raise click.ClickException(f"Failed to read {label.lower()} JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        noun = "Candidate outputs" if label == "Candidates" else label
        raise click.ClickException(f"{noun} JSON must be an object.")
    return payload

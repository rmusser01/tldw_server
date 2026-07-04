"""Output profile normalization and rendering for chat macros."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .exceptions import MacroValidationError

DEFAULT_PROFILE_SECTIONS = [
    "summary",
    "decisions",
    "action_items",
    "open_questions",
    "failed_branches",
]
MAX_PROFILE_SECTIONS = 10
MAX_SECTION_NAME_LENGTH = 64
VALID_PROFILE_FORMATS = {"structured_sections", "single_response", "multiple_messages"}


@dataclass(slots=True)
class MacroOutputProfile:
    name: str = "default"
    format: str = "structured_sections"
    sections: list[str] = field(default_factory=lambda: list(DEFAULT_PROFILE_SECTIONS))
    include_branch_outputs: bool = False


DEFAULT_OUTPUT_PROFILE = MacroOutputProfile()


def profile_to_dict(profile: MacroOutputProfile) -> dict[str, Any]:
    return {
        "format": profile.format,
        "sections": list(profile.sections),
        "include_branch_outputs": bool(profile.include_branch_outputs),
    }


def normalize_output_profile(name: str, raw: Mapping[str, Any] | None = None) -> MacroOutputProfile:
    raw = raw or {}
    profile_format = str(raw.get("format") or DEFAULT_OUTPUT_PROFILE.format)
    if profile_format not in VALID_PROFILE_FORMATS:
        raise MacroValidationError(f"invalid output profile format: {profile_format}")

    sections = raw.get("sections", DEFAULT_PROFILE_SECTIONS)
    if not isinstance(sections, Sequence) or isinstance(sections, (str, bytes)):
        raise MacroValidationError("output profile sections must be a list")
    normalized_sections = [str(section) for section in sections]
    _validate_sections(normalized_sections)

    return MacroOutputProfile(
        name=name,
        format=profile_format,
        sections=normalized_sections,
        include_branch_outputs=bool(raw.get("include_branch_outputs", False)),
    )


def merge_output_profile(
    profile: MacroOutputProfile,
    overrides: Mapping[str, Any] | None = None,
) -> MacroOutputProfile:
    if not overrides:
        return MacroOutputProfile(
            name=profile.name,
            format=profile.format,
            sections=list(profile.sections),
            include_branch_outputs=profile.include_branch_outputs,
        )
    raw = profile_to_dict(profile)
    raw.update(dict(overrides))
    return normalize_output_profile(profile.name, raw)


def render_output_profile(
    profile: MacroOutputProfile,
    outputs: Mapping[str, str],
    *,
    failed_branches: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    if profile.format == "single_response":
        return "\n\n".join(str(outputs.get(section, "")).strip() for section in profile.sections).strip()

    blocks: list[str] = []
    for section in profile.sections:
        if section == "failed_branches":
            body = _format_failed_branches(failed_branches or [])
        else:
            body = str(outputs.get(section, "")).strip()
        if body:
            blocks.append(f"## {_title(section)}\n\n{body}")
    return "\n\n".join(blocks).strip()


def _validate_sections(sections: list[str]) -> None:
    if len(sections) > MAX_PROFILE_SECTIONS:
        raise MacroValidationError("output profile has too many sections")
    for section in sections:
        if not section or len(section) > MAX_SECTION_NAME_LENGTH:
            raise MacroValidationError("invalid output profile section")


def _title(section: str) -> str:
    return section.replace("_", " ").title()


def _format_failed_branches(failed_branches: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for branch in failed_branches:
        label = str(branch.get("label") or branch.get("step_id") or "Branch")
        error = str(branch.get("error") or branch.get("error_message") or "failed")
        lines.append(f"- {label}: {error}")
    return "\n".join(lines)

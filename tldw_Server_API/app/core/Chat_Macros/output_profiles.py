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
VALID_PROFILE_FORMATS = {"structured_sections", "single_response"}


@dataclass(slots=True)
class MacroOutputProfile:
    """Normalized output rendering configuration for one macro response."""

    name: str = "default"
    format: str = "structured_sections"
    sections: list[str] = field(default_factory=lambda: list(DEFAULT_PROFILE_SECTIONS))
    include_branch_outputs: bool = False


DEFAULT_OUTPUT_PROFILE = MacroOutputProfile()


def profile_to_dict(profile: MacroOutputProfile) -> dict[str, Any]:
    """Serialize a normalized profile to its accepted settings keys."""
    return {
        "format": profile.format,
        "sections": list(profile.sections),
        "include_branch_outputs": bool(profile.include_branch_outputs),
    }


def normalize_output_profile(name: str, raw: Mapping[str, Any] | None = None) -> MacroOutputProfile:
    """Validate profile settings and return a normalized profile."""
    raw = raw or {}
    unknown_keys = sorted(set(raw) - {"format", "sections", "include_branch_outputs"})
    if unknown_keys:
        raise MacroValidationError(
            f"unknown output profile keys: {', '.join(str(key) for key in unknown_keys)}"
        )
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
    """Apply validated overrides on top of an existing output profile."""
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
    branch_outputs: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Render structured or single-response output with optional branch details."""
    if profile.format == "single_response":
        parts = []
        for section in profile.sections:
            if section == "failed_branches":
                body = _format_failed_branches(failed_branches or [])
            else:
                body = str(outputs.get(section, "")).strip()
            if body:
                parts.append(body)
        rendered = "\n\n".join(parts).strip()
        appendix = _format_branch_outputs(branch_outputs or []) if profile.include_branch_outputs else ""
        return "\n\n".join(part for part in (rendered, appendix) if part).strip()

    blocks: list[str] = []
    for section in profile.sections:
        if section == "failed_branches":
            body = _format_failed_branches(failed_branches or [])
        else:
            body = str(outputs.get(section, "")).strip()
        if body:
            blocks.append(f"## {_title(section)}\n\n{body}")
    if profile.include_branch_outputs:
        appendix = _format_branch_outputs(branch_outputs or [])
        if appendix:
            blocks.append(appendix)
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


def _format_branch_outputs(branch_outputs: Sequence[Mapping[str, Any]]) -> str:
    blocks: list[str] = []
    for branch in branch_outputs:
        text = str(branch.get("output_text") or branch.get("output") or "").strip()
        if not text:
            continue
        label = str(branch.get("label") or branch.get("step_id") or "Branch")
        blocks.append(f"### {label}\n\n{text}")
    if not blocks:
        return ""
    return "## Branch Outputs\n\n" + "\n\n".join(blocks)

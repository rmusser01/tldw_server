"""Shared validation checks for generated workspace artifacts."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from tldw_Server_API.app.core.exceptions import WorkspaceArtifactExportStateError

_GENERATED_ARTIFACT_ALIASES: Mapping[str, tuple[str, ...]] = {
    "quiz": ("quiz",),
    "flashcards": ("flashcard",),
    "audio_summary": ("audio_summary", "audio_brief", "audio_overview", "podcast"),
    "adaptive_table": ("adaptive_table", "data_table", "adata_table"),
    "slides": ("slides", "slide_deck", "presentation"),
    "mindmap": ("mindmap", "mind_map"),
}

_EXACT_PLACEHOLDER_CONTENT = {
    "invalid",
    "placeholder",
    "todo",
    "tbd",
}

_PLACEHOLDER_PHRASES = (
    "this is a test",
    "slides go here",
    "content goes here",
    "quiz goes here",
    "flashcards go here",
    "audio summary goes here",
    "table goes here",
    "mindmap goes here",
    "lorem ipsum",
)

_WRAPPER_RE = re.compile(r"[^a-z0-9]+")
_FAIL_STATUSES = {"failed", "fail", "error", "rejected", "invalid"}
_PASS_STATUSES = {"passed", "pass", "ok", "success", "succeeded", "valid"}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _metadata_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "required", "on"}
    return False


def _normalize_type(value: Any) -> str:
    return _WRAPPER_RE.sub("_", str(value or "").lower()).strip("_")


def _artifact_family(artifact: Mapping[str, Any]) -> str | None:
    normalized_type = _normalize_type(artifact.get("artifact_type"))
    if not normalized_type:
        return None
    for family, aliases in _GENERATED_ARTIFACT_ALIASES.items():
        if any(alias in normalized_type for alias in aliases):
            return family
    return None


def _normalize_content(value: Any) -> str:
    content = str(value or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return _WRAPPER_RE.sub(" ", content.lower()).strip()


def _reject_placeholder_content(artifact: Mapping[str, Any], family: str | None) -> None:
    if family is None:
        return

    content = str(artifact.get("content") or "")
    normalized = _normalize_content(content)
    if not normalized:
        raise WorkspaceArtifactExportStateError("workspace_artifact_missing_content")

    if normalized in _EXACT_PLACEHOLDER_CONTENT:
        raise WorkspaceArtifactExportStateError("workspace_artifact_placeholder_content")

    if len(normalized) <= 160 and any(phrase in normalized for phrase in _PLACEHOLDER_PHRASES):
        raise WorkspaceArtifactExportStateError("workspace_artifact_placeholder_content")


def _claims_validation_required(artifact: Mapping[str, Any]) -> bool:
    producer_metadata = _as_mapping(artifact.get("producer_metadata"))
    validation = _as_mapping(producer_metadata.get("validation"))
    claims_validation = _as_mapping(producer_metadata.get("claims_validation"))
    return any(
        (
            _metadata_bool(producer_metadata.get("claims_validation_required")),
            _metadata_bool(producer_metadata.get("requires_claims_validation")),
            _metadata_bool(producer_metadata.get("source_pack_claims_validation_required")),
            _metadata_bool(validation.get("claims_required")),
            _metadata_bool(validation.get("source_pack_claims_required")),
            _metadata_bool(claims_validation.get("required")),
        )
    )


def _walk_mapping(root: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any]:
    current: Any = root
    for key in path:
        current = _as_mapping(current).get(key)
    return _as_mapping(current)


def _has_claims_signal(report: Mapping[str, Any]) -> bool:
    return any(
        key in report
        for key in (
            "validator",
            "validator_name",
            "model",
            "model_name",
            "llm_model",
            "unsupported_claim_count",
            "unsupported_claims",
            "warnings",
            "status",
        )
    )


def _find_claims_report(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    paths = (
        ("claims_validation",),
        ("validation", "claims"),
        ("post_verification",),
        ("verification_summary",),
    )
    for root_name in ("review_metadata", "version_metadata", "producer_metadata"):
        root = _as_mapping(artifact.get(root_name))
        for path in paths:
            report = _walk_mapping(root, path)
            if _has_claims_signal(report):
                return report
    return {}


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _unsupported_claim_count(report: Mapping[str, Any]) -> int | None:
    count = _int_or_none(report.get("unsupported_claim_count"))
    if count is not None:
        return count
    unsupported_claims = report.get("unsupported_claims")
    if isinstance(unsupported_claims, Sequence) and not isinstance(unsupported_claims, (str, bytes)):
        return len(unsupported_claims)
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _claims_validation_metadata(
    artifact: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not report:
        return None

    producer_metadata = _as_mapping(artifact.get("producer_metadata"))
    status = str(report.get("status") or "").strip().lower()
    unsupported_count = _unsupported_claim_count(report)
    return {
        "status": status if status in _PASS_STATUSES else "passed",
        "validator": (
            report.get("validator")
            or report.get("validator_name")
            or producer_metadata.get("claims_validator")
            or "claims_source_pack_v1"
        ),
        "model": (
            report.get("model")
            or report.get("model_name")
            or report.get("llm_model")
            or producer_metadata.get("claims_validator_model")
        ),
        "unsupported_claim_count": unsupported_count,
        "warnings": _as_list(report.get("warnings") or report.get("unresolved_warnings")),
    }


def _reject_failed_claims_report(report: Mapping[str, Any]) -> None:
    status = str(report.get("status") or "").strip().lower()
    unsupported_count = _unsupported_claim_count(report)
    if status in _FAIL_STATUSES or (unsupported_count is not None and unsupported_count > 0):
        raise WorkspaceArtifactExportStateError("workspace_artifact_claims_validation_failed")


def validate_workspace_artifact_for_export(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a workspace artifact immediately before export."""
    family = _artifact_family(artifact)
    _reject_placeholder_content(artifact, family)

    claims_required = _claims_validation_required(artifact)
    claims_report = _find_claims_report(artifact)
    if claims_required and not claims_report:
        raise WorkspaceArtifactExportStateError("workspace_artifact_claims_validation_missing")
    if claims_report:
        _reject_failed_claims_report(claims_report)

    checks = ["accepted_review_state"]
    if family is not None:
        checks.extend(("content_present", "placeholder_rejected"))
    if claims_required or claims_report:
        checks.append("claims_validation")

    return {
        "status": "passed",
        "validator": "workspace_artifact_export_v1",
        "artifact_family": family,
        "checks": checks,
        "claims_validation_required": claims_required,
        "claims_validation": _claims_validation_metadata(artifact, claims_report),
    }

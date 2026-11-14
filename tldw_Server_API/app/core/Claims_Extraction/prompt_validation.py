"""Preflight validation helpers for claims-extraction prompt templates.

This module validates claims prompt templates and reports structured issues for
placeholder integrity, formatting, and alignment sanity checks. Public entry
points are ``validate_claims_prompt_template`` (validate a provided template and
return a ``ClaimsPromptValidationReport``) and
``validate_claims_prompt_preflight`` (resolve runtime validation mode, load the
claims prompt, and enforce configured behavior). Validation integrates
``align_claim`` for sample span checks, ``resolve_claims_prompt_validation_config``
for runtime mode/strictness, and ``load_prompt`` for template retrieval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from string import Formatter
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Claims_Extraction.alignment import align_claim
from tldw_Server_API.app.core.Claims_Extraction.runtime_config import (
    resolve_claims_prompt_validation_config,
)
from tldw_Server_API.app.core.Context_Integrity.prompt_loader import load_prompt
from tldw_Server_API.app.core.exceptions import ValidationError

_CLAIMS_TEMPLATE_FORMAT_EXCEPTIONS = (
    KeyError,
    IndexError,
    TypeError,
    ValueError,
)

_DEFAULT_CLAIMS_PROMPT_TEMPLATE = (
    "Extract up to {max_claims} atomic factual propositions from the ANSWER. "
    "Each proposition should stand alone without the surrounding context, be specific and checkable. "
    "Return JSON: {{\"claims\":[{{\"text\": str}}]}}. Do not include explanations.\n\nANSWER:\n{answer}"
)

_DEFAULT_SAMPLE_SOURCE = (
    "Acme Corp reported annual revenue of 12 million dollars in 2024. "
    "The report states net margin improved to 18 percent."
)

_DEFAULT_SAMPLE_CLAIMS: tuple[str, ...] = (
    "Acme Corp reported annual revenue of 12 million dollars in 2024.",
    "net margin improved to 18 percent.",
)


@dataclass(frozen=True)
class ClaimsPromptValidationIssue:
    """Represents a single claims-prompt validation issue.

    Attributes are ``code`` (str), ``message`` (str), and optional ``detail`` (str | None) for extra context.
    """

    code: str
    message: str
    detail: str | None = None


@dataclass(frozen=True)
class ClaimsPromptValidationReport:
    """Aggregates claims-prompt validation results for one validation run.

    ``mode`` (str) and ``strict`` (bool) capture the applied policy, ``issues`` stores a tuple of ``ClaimsPromptValidationIssue``, and ``has_issues`` is ``True`` when ``issues`` is non-empty.
    """

    mode: str
    strict: bool
    issues: tuple[ClaimsPromptValidationIssue, ...]

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)


class ClaimsPromptValidationError(ValidationError):
    """Raised when prompt preflight validation is configured to fail-fast."""


def claims_prompt_report_has_issues(report: Any) -> bool:
    """Return whether a prompt-validation report contains issues.

    Supports both property-style reports (``report.has_issues``) and
    callable-style reports (``report.has_issues()``) for defensive compatibility.
    """
    marker = getattr(report, "has_issues", False)
    if callable(marker):
        marker = marker()
    return bool(marker)


def _normalize_mode(mode: str | None, *, default: str = "warning") -> str:
    """Normalize and validate the configured prompt-validation mode.

    Args:
        mode: Raw mode value from config/runtime input (for example ``"Warning"`` or ``None``).
        default: Fallback mode used when ``mode`` is empty or invalid; expected values are ``"off"``, ``"warning"``, or ``"error"``.

    Returns:
        str: Normalized mode value (lowercased and stripped) when valid, otherwise ``default``.

    Notes:
        Invalid values are not raised as errors; they fall back to ``default``. No side effects.
    """
    resolved = str(mode or default).strip().lower()
    if resolved not in {"off", "warning", "error"}:
        return default
    return resolved


def _extract_placeholders(template: str) -> set[str]:
    """Parse a template and return placeholder base names.

    Args:
        template: Format-string template text, such as ``"{answer} {claims[0]} {meta.score}"``.

    Returns:
        set[str]: Unique placeholder names normalized to base identifiers (for example ``{"answer", "claims", "meta"}``).

    Notes:
        Dotted and indexed fields are reduced to their base token before ``.`` or ``[``. ``ValueError`` from ``Formatter().parse`` propagates for malformed templates. No side effects.
    """
    names: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if not field_name:
            continue
        base = str(field_name).split(".", 1)[0].split("[", 1)[0].strip()
        if base:
            names.add(base)
    return names


def _format_prompt_template(
    template: str,
    *,
    answer: str,
    max_claims: int,
) -> str:
    """Format the claims prompt template with runtime values.

    Args:
        template: Prompt template containing placeholders like ``{answer}`` and ``{max_claims}``.
        answer: Source answer text inserted into the template.
        max_claims: Maximum number of claims inserted into ``{max_claims}``.

    Returns:
        str: Formatted prompt text (for example, ``"Extract 3... ANSWER: <text>"``).

    Notes:
        On ``_CLAIMS_TEMPLATE_FORMAT_EXCEPTIONS``, the function escapes braces, restores supported placeholders, and retries formatting as a compatibility fallback for JSON-heavy templates. If fallback formatting also fails, the exception propagates; no side effects.
    """
    try:
        return template.format(answer=answer, max_claims=max_claims)
    except _CLAIMS_TEMPLATE_FORMAT_EXCEPTIONS:
        escaped = template.replace("{", "{{").replace("}", "}}")
        escaped = escaped.replace("{{answer}}", "{answer}").replace("{{max_claims}}", "{max_claims}")
        return escaped.format(answer=answer, max_claims=max_claims)


def validate_claims_prompt_template(
    prompt_template: str,
    *,
    mode: str = "warning",
    strict: bool = False,
    required_placeholders: Sequence[str] = ("answer", "max_claims"),
    sample_source_text: str = _DEFAULT_SAMPLE_SOURCE,
    sample_claim_texts: Sequence[str] = _DEFAULT_SAMPLE_CLAIMS,
) -> ClaimsPromptValidationReport:
    """Validate a claims prompt template and return a structured issue report.

    Args:
        prompt_template: Prompt template text expected to include placeholders
            used by the claims extractor workflow.
        mode: Validation mode for downstream handling. Valid values are
            ``"off"``, ``"warning"``, and ``"error"`` (default ``"warning"``).
        strict: Whether strict checks should be applied. In strict mode,
            alignment checks require exact matches and use stricter thresholds.
        required_placeholders: Placeholder names that must be present in
            ``prompt_template`` (defaults to ``("answer", "max_claims")``).
        sample_source_text: Sample source text used for formatting and
            alignment checks.
        sample_claim_texts: Sample claims expected to align to
            ``sample_source_text`` during preflight checks.

    Returns:
        ClaimsPromptValidationReport: Report containing normalized mode,
        strict flag, and zero or more ``ClaimsPromptValidationIssue`` entries.

    Raises:
        None directly. Formatting and placeholder parsing exceptions are
        converted into report issues instead of being raised.

    Notes:
        Validation checks template emptiness, placeholder syntax/presence,
        sample formatting, and ``align_claim`` behavior for sample claims.
        Typical usage:
        ``report = validate_claims_prompt_template(template, mode="warning")``.
    """
    issues: list[ClaimsPromptValidationIssue] = []
    resolved_mode = _normalize_mode(mode)

    template = str(prompt_template or "").strip()
    if not template:
        issues.append(
            ClaimsPromptValidationIssue(
                code="empty_template",
                message="Claims extractor prompt template is empty.",
            )
        )
        return ClaimsPromptValidationReport(mode=resolved_mode, strict=bool(strict), issues=tuple(issues))

    try:
        placeholders = _extract_placeholders(template)
    except ValueError as exc:
        # Some templates include literal JSON braces without escaping.
        escaped = template.replace("{", "{{").replace("}", "}}")
        escaped = escaped.replace("{{answer}}", "{answer}").replace("{{max_claims}}", "{max_claims}")
        try:
            placeholders = _extract_placeholders(escaped)
        except ValueError:
            issues.append(
                ClaimsPromptValidationIssue(
                    code="placeholder_parse_error",
                    message="Claims prompt template has invalid placeholder syntax.",
                    detail=str(exc),
                )
            )
            placeholders = set()

    missing = [name for name in required_placeholders if str(name) not in placeholders]
    for name in missing:
        issues.append(
            ClaimsPromptValidationIssue(
                code="missing_placeholder",
                message=f"Claims prompt template is missing required placeholder '{{{name}}}'.",
                detail=f"present={sorted(placeholders)}",
            )
        )

    if not missing:
        try:
            _format_prompt_template(template, answer=sample_source_text, max_claims=max(1, len(sample_claim_texts)))
        except _CLAIMS_TEMPLATE_FORMAT_EXCEPTIONS as exc:
            issues.append(
                ClaimsPromptValidationIssue(
                    code="template_format_error",
                    message="Claims prompt template failed formatting with sample values.",
                    detail=str(exc),
                )
            )

    alignment_mode = "fuzzy"
    alignment_threshold = 0.75
    for claim_text in sample_claim_texts:
        result = align_claim(
            sample_source_text,
            claim_text,
            mode=alignment_mode,
            threshold=alignment_threshold,
        )
        if result is None:
            issues.append(
                ClaimsPromptValidationIssue(
                    code="sample_alignment_failed",
                    message="Sample claim text did not align with sample source text.",
                    detail=claim_text,
                )
            )
            continue
        if strict and result.method != "exact":
            issues.append(
                ClaimsPromptValidationIssue(
                    code="sample_alignment_non_exact",
                    message="Strict prompt validation requires exact sample alignment.",
                    detail=f"method={result.method}; claim={claim_text}",
                )
            )

    return ClaimsPromptValidationReport(mode=resolved_mode, strict=bool(strict), issues=tuple(issues))


def handle_claims_prompt_validation_report(report: ClaimsPromptValidationReport) -> None:
    """Apply runtime behavior for a claims prompt validation report.

    Args:
        report: ``ClaimsPromptValidationReport`` returned by
            ``validate_claims_prompt_template`` or
            ``validate_claims_prompt_preflight``.

    Returns:
        None.

    Raises:
        ClaimsPromptValidationError: When ``report.mode`` resolves to
            ``"error"`` and the report contains one or more issues.

    Notes:
        Modes behave as follows: ``"off"`` ignores issues, ``"warning"``
        logs issues, and ``"error"`` logs issues then raises when issues are
        present. Side effect: emits warnings through ``logger.warning`` for
        each issue. Typical usage:
        ``handle_claims_prompt_validation_report(report)``.
    """
    mode = _normalize_mode(report.mode)
    if mode == "off":
        return

    for issue in report.issues:
        logger.warning(
            "claims prompt validation [{}]: {}{}",
            issue.code,
            issue.message,
            f" ({issue.detail})" if issue.detail else "",
        )

    if mode == "error" and claims_prompt_report_has_issues(report):
        first_issue = report.issues[0]
        raise ClaimsPromptValidationError(
            f"Claims prompt validation failed: {first_issue.code}: {first_issue.message}"
        )


def validate_claims_prompt_preflight(
    settings_obj: Mapping[str, Any] | None = None,
    *,
    mode: str | None = None,
    strict: bool | None = None,
) -> ClaimsPromptValidationReport:
    """Run end-to-end prompt-validation preflight using runtime configuration.

    Args:
        settings_obj: Optional settings mapping consumed by
            ``resolve_claims_prompt_validation_config`` to determine effective
            validation defaults.
        mode: Optional mode override. Valid values are ``"off"``,
            ``"warning"``, and ``"error"``. When ``None``, runtime config is
            used.
        strict: Optional strictness override. When ``None``, runtime config is
            used.

    Returns:
        ClaimsPromptValidationReport: Final report produced by
        ``validate_claims_prompt_template`` with resolved mode/strict values.

    Raises:
        ClaimsPromptValidationError: Propagated from
            ``handle_claims_prompt_validation_report`` when effective mode is
            ``"error"`` and validation issues exist.

    Notes:
        This function integrates ``resolve_claims_prompt_validation_config``,
        ``load_prompt``, ``validate_claims_prompt_template``, and
        ``handle_claims_prompt_validation_report``. Side effect: may log
        warnings and may raise based on configured mode. Typical usage:
        ``report = validate_claims_prompt_preflight(settings_obj=settings)``.
    """
    resolved_mode, resolved_strict = resolve_claims_prompt_validation_config(
        settings_obj,
        default_mode="warning",
        default_strict=False,
    )
    if mode is not None:
        resolved_mode = _normalize_mode(mode, default=resolved_mode)
    if strict is not None:
        resolved_strict = bool(strict)

    prompt_template = load_prompt("ingestion", "claims_extractor_prompt") or _DEFAULT_CLAIMS_PROMPT_TEMPLATE
    report = validate_claims_prompt_template(
        prompt_template,
        mode=resolved_mode,
        strict=resolved_strict,
    )
    handle_claims_prompt_validation_report(report)
    return report


__all__ = [
    "ClaimsPromptValidationError",
    "ClaimsPromptValidationIssue",
    "ClaimsPromptValidationReport",
    "claims_prompt_report_has_issues",
    "handle_claims_prompt_validation_report",
    "validate_claims_prompt_preflight",
    "validate_claims_prompt_template",
]

"""
Prompt-cost guardrails for chat provider dispatch.

The guardrails operate on bounded prompt-cost envelopes and request parameter
metadata. They intentionally never retain or emit raw prompt text.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from tldw_Server_API.app.core.Chat.prompt_cost_envelope import (
    PromptCostEnvelope,
    fingerprint_text,
)
from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.testing import is_truthy as _shared_is_truthy

GuardrailAction = Literal["allow", "warn", "block"]


@dataclass(frozen=True)
class PromptCostGuardrailConfig:
    """Runtime configuration for prompt-cost guardrail decisions."""

    enabled: bool = False
    default_action: Literal["warn", "block"] = "warn"
    warn_total_estimated_tokens: int | None = None
    block_total_estimated_tokens: int | None = None
    warn_static_segment_tokens: int | None = None
    warn_world_book_tokens: int | None = None
    warn_max_output_tokens: int | None = None
    warn_choice_count: int | None = None
    warn_reasoning_efforts: tuple[str, ...] = ("high", "xhigh")
    warn_on_fingerprint_churn: bool = True


@dataclass(frozen=True)
class PromptCostGuardrailWarning:
    """Prompt-safe warning or block reason."""

    code: str
    severity: Literal["warning", "block"]
    message: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Return a bounded diagnostic representation."""
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "metadata": _bounded_metadata(self.metadata),
        }


@dataclass(frozen=True)
class PromptCostGuardrailDecision:
    """Guardrail decision for one final provider-bound prompt."""

    action: GuardrailAction
    warnings: tuple[PromptCostGuardrailWarning, ...]
    prompt_fingerprint: str
    fingerprint_version: str
    total_estimated_tokens: int
    message_count: int
    segment_token_totals: Mapping[str, int]
    segment_fingerprints: Mapping[str, str]

    def to_response_metadata(self) -> dict[str, Any]:
        """Return prompt-safe metadata suitable for API responses/logging."""
        return {
            "action": self.action,
            "fingerprint_version": self.fingerprint_version,
            "prompt_fingerprint": self.prompt_fingerprint,
            "total_estimated_tokens": self.total_estimated_tokens,
            "message_count": self.message_count,
            "segment_token_totals": dict(self.segment_token_totals),
            "segment_fingerprints": dict(self.segment_fingerprints),
            "warnings": [warning.to_metadata() for warning in self.warnings],
        }


def load_prompt_cost_guardrail_config() -> PromptCostGuardrailConfig:
    """Load prompt-cost guardrail config from env first, then `[Chat-Module]`."""
    parser = load_comprehensive_config()
    section = "Chat-Module"

    enabled = _get_bool(parser, section, "prompt_guardrails_enabled", "CHAT_PROMPT_GUARDRAILS_ENABLED", False)
    default_action = _get_str(
        parser,
        section,
        "prompt_guardrails_default_action",
        "CHAT_PROMPT_GUARDRAILS_DEFAULT_ACTION",
        "warn",
    ).lower()
    if default_action not in {"warn", "block"}:
        default_action = "warn"

    return PromptCostGuardrailConfig(
        enabled=enabled,
        default_action=default_action,  # type: ignore[arg-type]
        warn_total_estimated_tokens=_get_optional_int(
            parser,
            section,
            "prompt_guardrails_warn_total_estimated_tokens",
            "CHAT_PROMPT_GUARDRAILS_WARN_TOTAL_ESTIMATED_TOKENS",
        ),
        block_total_estimated_tokens=_get_optional_int(
            parser,
            section,
            "prompt_guardrails_block_total_estimated_tokens",
            "CHAT_PROMPT_GUARDRAILS_BLOCK_TOTAL_ESTIMATED_TOKENS",
        ),
        warn_static_segment_tokens=_get_optional_int(
            parser,
            section,
            "prompt_guardrails_warn_static_segment_tokens",
            "CHAT_PROMPT_GUARDRAILS_WARN_STATIC_SEGMENT_TOKENS",
        ),
        warn_world_book_tokens=_get_optional_int(
            parser,
            section,
            "prompt_guardrails_warn_world_book_tokens",
            "CHAT_PROMPT_GUARDRAILS_WARN_WORLD_BOOK_TOKENS",
        ),
        warn_max_output_tokens=_get_optional_int(
            parser,
            section,
            "prompt_guardrails_warn_max_output_tokens",
            "CHAT_PROMPT_GUARDRAILS_WARN_MAX_OUTPUT_TOKENS",
        ),
        warn_choice_count=_get_optional_int(
            parser,
            section,
            "prompt_guardrails_warn_choice_count",
            "CHAT_PROMPT_GUARDRAILS_WARN_CHOICE_COUNT",
        ),
        warn_reasoning_efforts=_get_str_tuple(
            parser,
            section,
            "prompt_guardrails_warn_reasoning_efforts",
            "CHAT_PROMPT_GUARDRAILS_WARN_REASONING_EFFORTS",
            ("high", "xhigh"),
        ),
        warn_on_fingerprint_churn=_get_bool(
            parser,
            section,
            "prompt_guardrails_warn_on_fingerprint_churn",
            "CHAT_PROMPT_GUARDRAILS_WARN_ON_FINGERPRINT_CHURN",
            True,
        ),
    )


def evaluate_prompt_cost_guardrails(
    envelope: PromptCostEnvelope,
    *,
    request_options: Mapping[str, Any] | None = None,
    previous_fingerprints: Mapping[str, str] | None = None,
    config: PromptCostGuardrailConfig | None = None,
) -> PromptCostGuardrailDecision:
    """Evaluate prompt-cost guardrails for a final provider-bound request."""
    resolved_config = config or PromptCostGuardrailConfig()
    segment_fingerprints = _segment_fingerprint_summary(envelope)

    if not resolved_config.enabled:
        return PromptCostGuardrailDecision(
            action="allow",
            warnings=(),
            prompt_fingerprint=envelope.aggregate_fingerprint,
            fingerprint_version=envelope.fingerprint_version,
            total_estimated_tokens=envelope.total_estimated_tokens,
            message_count=envelope.message_count,
            segment_token_totals=dict(envelope.segment_token_totals),
            segment_fingerprints=segment_fingerprints,
        )

    warnings: list[PromptCostGuardrailWarning] = []
    block_triggered = False

    if _exceeds(envelope.total_estimated_tokens, resolved_config.block_total_estimated_tokens):
        warnings.append(
            PromptCostGuardrailWarning(
                code="prompt_estimate_exceeds_hard_cap",
                severity="block",
                message="Estimated prompt tokens exceed the configured hard cap.",
                metadata={
                    "total_estimated_tokens": envelope.total_estimated_tokens,
                    "threshold": resolved_config.block_total_estimated_tokens,
                },
            )
        )
        block_triggered = True

    if _exceeds(envelope.total_estimated_tokens, resolved_config.warn_total_estimated_tokens):
        warnings.append(
            PromptCostGuardrailWarning(
                code="large_prompt_estimate",
                severity="warning",
                message="Estimated prompt tokens exceed the configured warning threshold.",
                metadata={
                    "total_estimated_tokens": envelope.total_estimated_tokens,
                    "threshold": resolved_config.warn_total_estimated_tokens,
                },
            )
        )

    static_tokens = int(envelope.segment_token_totals.get("static", 0) or 0)
    if _exceeds(static_tokens, resolved_config.warn_static_segment_tokens):
        warnings.append(
            PromptCostGuardrailWarning(
                code="large_static_segment",
                severity="warning",
                message="Static/system prompt tokens exceed the configured warning threshold.",
                metadata={
                    "static_estimated_tokens": static_tokens,
                    "threshold": resolved_config.warn_static_segment_tokens,
                },
            )
        )

    world_book_tokens = int(envelope.segment_token_totals.get("world_book", 0) or 0)
    if _exceeds(world_book_tokens, resolved_config.warn_world_book_tokens):
        warnings.append(
            PromptCostGuardrailWarning(
                code="large_world_book_segment",
                severity="warning",
                message="World-book prompt tokens exceed the configured warning threshold.",
                metadata={
                    "world_book_estimated_tokens": world_book_tokens,
                    "threshold": resolved_config.warn_world_book_tokens,
                },
            )
        )

    options = request_options or {}
    output_cap = _resolve_output_token_cap(options)
    if output_cap is not None and _exceeds(output_cap, resolved_config.warn_max_output_tokens):
        warnings.append(
            PromptCostGuardrailWarning(
                code="high_output_token_cap",
                severity="warning",
                message="Requested output token cap exceeds the configured warning threshold.",
                metadata={
                    "max_output_tokens": output_cap,
                    "threshold": resolved_config.warn_max_output_tokens,
                },
            )
        )

    choice_count = _coerce_positive_int(options.get("n"))
    if choice_count is not None and _exceeds(choice_count, resolved_config.warn_choice_count):
        warnings.append(
            PromptCostGuardrailWarning(
                code="high_choice_count",
                severity="warning",
                message="Requested choice count can multiply output-token cost.",
                metadata={
                    "choice_count": choice_count,
                    "threshold": resolved_config.warn_choice_count,
                },
            )
        )

    reasoning_effort = _resolve_reasoning_effort(options)
    if reasoning_effort and reasoning_effort.lower() in {
        effort.lower() for effort in resolved_config.warn_reasoning_efforts
    }:
        warnings.append(
            PromptCostGuardrailWarning(
                code="reasoning_effort_risk",
                severity="warning",
                message="Requested reasoning effort can add hidden or metered reasoning tokens.",
                metadata={"reasoning_effort": reasoning_effort.lower()},
            )
        )

    if resolved_config.warn_on_fingerprint_churn and previous_fingerprints:
        warnings.extend(
            _fingerprint_churn_warnings(
                previous_fingerprints=previous_fingerprints,
                prompt_fingerprint=envelope.aggregate_fingerprint,
                segment_fingerprints=segment_fingerprints,
            )
        )

    action: GuardrailAction = "allow"
    if block_triggered or (
        warnings
        and resolved_config.default_action == "block"
        and any(warning.severity == "warning" for warning in warnings)
    ):
        action = "block"
        warnings = [
            warning if warning.severity == "block" else _promote_warning_to_block(warning)
            for warning in warnings
        ]
    elif warnings:
        action = "warn"

    return PromptCostGuardrailDecision(
        action=action,
        warnings=tuple(warnings),
        prompt_fingerprint=envelope.aggregate_fingerprint,
        fingerprint_version=envelope.fingerprint_version,
        total_estimated_tokens=envelope.total_estimated_tokens,
        message_count=envelope.message_count,
        segment_token_totals=dict(envelope.segment_token_totals),
        segment_fingerprints=segment_fingerprints,
    )


def _fingerprint_churn_warnings(
    *,
    previous_fingerprints: Mapping[str, str],
    prompt_fingerprint: str,
    segment_fingerprints: Mapping[str, str],
) -> list[PromptCostGuardrailWarning]:
    warnings: list[PromptCostGuardrailWarning] = []
    previous_aggregate = previous_fingerprints.get("aggregate") or previous_fingerprints.get("prompt")
    if previous_aggregate and previous_aggregate != prompt_fingerprint:
        warnings.append(
            PromptCostGuardrailWarning(
                code="prompt_fingerprint_churn",
                severity="warning",
                message="Prompt fingerprint changed from the previous comparable turn.",
                metadata={"fingerprint_scope": "aggregate"},
            )
        )

    for scope in ("static", "world_book"):
        previous = previous_fingerprints.get(scope)
        current = segment_fingerprints.get(scope)
        if previous and current and previous != current:
            warnings.append(
                PromptCostGuardrailWarning(
                    code=f"{scope}_fingerprint_churn",
                    severity="warning",
                    message=f"{scope.replace('_', '-')} fingerprint changed from the previous comparable turn.",
                    metadata={"fingerprint_scope": scope},
                )
            )
    return warnings


def _segment_fingerprint_summary(envelope: PromptCostEnvelope) -> dict[str, str]:
    by_kind: dict[str, list[str]] = {}
    for segment in envelope.segments:
        by_kind.setdefault(segment.kind, []).append(segment.fingerprint)
    return {
        kind: fingerprint_text("|".join(fingerprints), version=envelope.fingerprint_version)
        for kind, fingerprints in sorted(by_kind.items())
        if fingerprints
    }


def _promote_warning_to_block(warning: PromptCostGuardrailWarning) -> PromptCostGuardrailWarning:
    return PromptCostGuardrailWarning(
        code=warning.code,
        severity="block",
        message=warning.message,
        metadata=warning.metadata,
    )


def _resolve_output_token_cap(options: Mapping[str, Any]) -> int | None:
    caps = [
        _coerce_positive_int(options.get("max_completion_tokens")),
        _coerce_positive_int(options.get("max_tokens")),
    ]
    caps = [cap for cap in caps if cap is not None]
    return max(caps) if caps else None


def _resolve_reasoning_effort(options: Mapping[str, Any]) -> str | None:
    effort = options.get("reasoning_effort")
    if isinstance(effort, str) and effort.strip():
        return effort.strip()
    reasoning = options.get("reasoning")
    if isinstance(reasoning, Mapping):
        nested_effort = reasoning.get("effort")
        if isinstance(nested_effort, str) and nested_effort.strip():
            return nested_effort.strip()
    return None


def _exceeds(value: int, threshold: int | None) -> bool:
    return threshold is not None and threshold >= 0 and value > threshold


def _coerce_positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _bounded_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    bounded: dict[str, Any] = {}
    for key, value in metadata.items():
        key_str = str(key)[:80]
        if value is None or isinstance(value, bool | int | float):
            bounded[key_str] = value
        elif isinstance(value, str):
            bounded[key_str] = value[:160]
        else:
            bounded[key_str] = str(type(value).__name__)
    return bounded


def _get_str(parser: Any, section: str, key: str, env_key: str, default: str) -> str:
    value = os.getenv(env_key)
    if value is not None and value.strip():
        return value.strip()
    try:
        if parser is not None and parser.has_section(section):
            config_value = parser.get(section, key, fallback=default)
            if config_value is not None and str(config_value).strip():
                return str(config_value).strip()
    except (AttributeError, TypeError, ValueError):
        return default
    return default


def _get_bool(parser: Any, section: str, key: str, env_key: str, default: bool) -> bool:
    value = _get_str(parser, section, key, env_key, str(default))
    if value == str(default):
        return default
    return _shared_is_truthy(value)


def _get_optional_int(parser: Any, section: str, key: str, env_key: str) -> int | None:
    value = _get_str(parser, section, key, env_key, "")
    if not value:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _get_str_tuple(
    parser: Any,
    section: str,
    key: str,
    env_key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = _get_str(parser, section, key, env_key, ",".join(default))
    parts = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    return parts or default


__all__ = [
    "PromptCostGuardrailConfig",
    "PromptCostGuardrailDecision",
    "PromptCostGuardrailWarning",
    "evaluate_prompt_cost_guardrails",
    "load_prompt_cost_guardrail_config",
]

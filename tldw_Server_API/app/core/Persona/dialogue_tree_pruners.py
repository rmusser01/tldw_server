"""Deterministic pruning rules for persona dialogue-tree candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import re
from typing import Any, Mapping


class PruneSeverity(str, Enum):
    NONE = "none"
    SOFT = "soft"
    HARD = "hard"


class PruneReason(str, Enum):
    NONE = "none"
    MALFORMED_CANDIDATE = "malformed_candidate"
    PROMPT_INJECTION_PRESSURE = "prompt_injection_pressure"
    PERSONA_BOUNDARY_VIOLATION = "persona_boundary_violation"
    UNSAFE_TOOL_PLAN = "unsafe_tool_plan"
    DUPLICATE_LOW_DIVERSITY = "duplicate_low_diversity"
    BUDGET_OVERFLOW = "budget_overflow"
    LLM_JUDGE_WARNING = "llm_judge_warning"


@dataclass(frozen=True)
class PruneDecision:
    pruned: bool
    severity: PruneSeverity
    reason: PruneReason
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    authorizes_runtime_action: bool = False


_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bignore (all )?(previous|prior|earlier) (instructions|rules|policy)\b"),
    re.compile(r"(?i)\b(disregard|bypass|override)\b.{0,32}\b(policy|safety|guardrails?)\b"),
    re.compile(r"(?i)\b(reveal|print|dump)\b.{0,32}\b(system prompt|hidden prompt)\b"),
)

_PERSONA_BOUNDARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bi am (not|no longer) (your )?(persona|assistant)\b"),
    re.compile(r"(?i)\bi will ignore (the )?persona\b"),
    re.compile(r"(?i)\bignore (the )?persona (style|rules|instructions)\b"),
)


def keep_decision(
    *,
    reason: PruneReason = PruneReason.NONE,
    message: str = "",
    metadata: dict[str, Any] | None = None,
) -> PruneDecision:
    return PruneDecision(
        pruned=False,
        severity=PruneSeverity.NONE,
        reason=reason,
        message=message,
        metadata=dict(metadata or {}),
    )


def prune_decision(
    *,
    severity: PruneSeverity,
    reason: PruneReason,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> PruneDecision:
    return PruneDecision(
        pruned=True,
        severity=severity,
        reason=reason,
        message=message,
        metadata=dict(metadata or {}),
    )


def malformed_candidate_pruner(candidate: Any) -> PruneDecision:
    if not isinstance(candidate, Mapping):
        return prune_decision(
            severity=PruneSeverity.HARD,
            reason=PruneReason.MALFORMED_CANDIDATE,
            message="candidate must be a mapping",
            metadata={"candidate_type": type(candidate).__name__},
        )

    action_type = candidate.get("action_type")
    if not isinstance(action_type, str) or not action_type.strip():
        return prune_decision(
            severity=PruneSeverity.HARD,
            reason=PruneReason.MALFORMED_CANDIDATE,
            message="candidate.action_type is required",
        )

    text = candidate.get("text", "")
    tool_plan = candidate.get("tool_plan")
    has_text = isinstance(text, str) and bool(text.strip())
    has_plan = isinstance(tool_plan, Mapping) and bool(tool_plan)
    if not has_text and not has_plan:
        return prune_decision(
            severity=PruneSeverity.HARD,
            reason=PruneReason.MALFORMED_CANDIDATE,
            message="candidate requires text or tool_plan",
        )

    return keep_decision()


def prompt_injection_pressure_pruner(candidate: Mapping[str, Any]) -> PruneDecision:
    text = str(candidate.get("text", ""))
    if any(pattern.search(text) for pattern in _PROMPT_INJECTION_PATTERNS):
        return prune_decision(
            severity=PruneSeverity.HARD,
            reason=PruneReason.PROMPT_INJECTION_PRESSURE,
            message="candidate text contains prompt-injection pressure",
        )
    return keep_decision()


def persona_boundary_violation_pruner(candidate: Mapping[str, Any]) -> PruneDecision:
    text = str(candidate.get("text", ""))
    if any(pattern.search(text) for pattern in _PERSONA_BOUNDARY_PATTERNS):
        return prune_decision(
            severity=PruneSeverity.HARD,
            reason=PruneReason.PERSONA_BOUNDARY_VIOLATION,
            message="candidate text violates persona boundary",
        )
    return keep_decision()


def unsafe_tool_plan_pruner(candidate: Mapping[str, Any]) -> PruneDecision:
    tool_plan = candidate.get("tool_plan")
    if not isinstance(tool_plan, Mapping):
        return keep_decision()

    authorized = tool_plan.get("authorized")
    action = str(tool_plan.get("action", "")).strip().casefold()
    risk_actions = {
        "delete",
        "remove",
        "write",
        "update",
        "edit",
        "execute",
        "shell",
        "http_post",
    }
    action_is_risky = any(action == risk_action or action.startswith(f"{risk_action}_") for risk_action in risk_actions)
    if authorized is not True or action_is_risky:
        return prune_decision(
            severity=PruneSeverity.HARD,
            reason=PruneReason.UNSAFE_TOOL_PLAN,
            message="tool plan is unauthorized or unsafe for runtime execution",
            metadata={"action": action, "authorized": authorized},
        )
    return keep_decision()


def duplicate_low_diversity_pruner(
    candidate: Mapping[str, Any],
    *,
    existing_signatures: set[str] | None = None,
) -> PruneDecision:
    existing_signatures = existing_signatures or set()
    signature = _candidate_signature(candidate)
    if signature in existing_signatures:
        return prune_decision(
            severity=PruneSeverity.SOFT,
            reason=PruneReason.DUPLICATE_LOW_DIVERSITY,
            message="candidate duplicates an existing branch",
            metadata={"signature": signature},
        )
    return keep_decision(metadata={"signature": signature})


def budget_overflow_pruner(
    runtime_state: Mapping[str, Any],
    budget: Mapping[str, Any],
) -> PruneDecision:
    if _budget_exceeded(
        current=runtime_state.get("selected_candidates"),
        max_value=budget.get("max_candidates"),
    ):
        return prune_decision(
            severity=PruneSeverity.HARD,
            reason=PruneReason.BUDGET_OVERFLOW,
            message="candidate budget exceeded",
        )
    if _budget_exceeded(current=runtime_state.get("depth"), max_value=budget.get("max_depth")):
        return prune_decision(
            severity=PruneSeverity.HARD,
            reason=PruneReason.BUDGET_OVERFLOW,
            message="depth budget exceeded",
        )
    if _budget_exceeded(
        current=runtime_state.get("provider_calls"),
        max_value=budget.get("max_provider_calls"),
    ):
        return prune_decision(
            severity=PruneSeverity.HARD,
            reason=PruneReason.BUDGET_OVERFLOW,
            message="provider-call budget exceeded",
        )
    return keep_decision()


def llm_judge_warning_pruner(candidate: Mapping[str, Any]) -> PruneDecision:
    label = str(candidate.get("judge_label", "")).strip().casefold()
    if not label or label in {"ok", "pass", "safe", "high_quality"}:
        return keep_decision(reason=PruneReason.LLM_JUDGE_WARNING)
    return prune_decision(
        severity=PruneSeverity.SOFT,
        reason=PruneReason.LLM_JUDGE_WARNING,
        message="llm judge emitted warning label",
        metadata={"judge_label": label},
    )


def _budget_exceeded(current: Any, max_value: Any) -> bool:
    if not isinstance(current, int) or not isinstance(max_value, int):
        return False
    return current >= max_value


def _candidate_signature(candidate: Mapping[str, Any]) -> str:
    action_type = str(candidate.get("action_type", ""))
    text = str(candidate.get("text", "")).strip()
    tool_plan = json.dumps(candidate.get("tool_plan", {}), sort_keys=True, default=str)
    return f"{action_type}|{text}|{tool_plan}"


__all__ = [
    "PruneDecision",
    "PruneReason",
    "PruneSeverity",
    "budget_overflow_pruner",
    "duplicate_low_diversity_pruner",
    "llm_judge_warning_pruner",
    "malformed_candidate_pruner",
    "persona_boundary_violation_pruner",
    "prompt_injection_pressure_pruner",
    "unsafe_tool_plan_pruner",
]

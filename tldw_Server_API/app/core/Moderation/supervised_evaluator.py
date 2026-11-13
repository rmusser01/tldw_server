"""Pure Guardian selection/redaction; no database, governance or notifications."""

from __future__ import annotations

import re
from collections.abc import Sequence

from tldw_Server_API.app.core.Moderation.execution_models import OutputBudget, RegexRuleSnapshot, SupervisedBatchResult
from tldw_Server_API.app.core.Moderation.policy_evaluator import PolicyEvaluator


def evaluate_supervised(
    text: str, rules: Sequence[RegexRuleSnapshot], phase: str | None, *, output_budget: OutputBudget | None = None
) -> SupervisedBatchResult:
    """Keep first priority ties, early block, exact phases and notify semantics."""
    if not text or not text.strip():
        return SupervisedBatchResult(None, None, None)
    priority = {"block": 4, "redact": 3, "warn": 2, "notify": 1}
    best_priority = 0
    winner = None
    span = None
    for index, rule in enumerate(rules):
        if rule.phase != "both" and rule.phase != phase:
            continue
        match = re.compile(rule.pattern, rule.flags).search(text)
        if match is None or priority.get(rule.action, 0) <= best_priority:
            continue
        best_priority = priority[rule.action]
        winner, span = index, match.span()
        if rule.action == "block":
            break
    redacted = None
    if winner is not None and rules[winner].action == "redact":
        redacted = text
        for rule in rules:
            if (rule.phase != "both" and rule.phase != phase) or rule.action != "redact":
                continue
            try:
                pattern = re.compile(rule.pattern, rule.flags)
                if output_budget is None:
                    redacted = pattern.sub("[REDACTED]", redacted)
                else:
                    redacted, _count = PolicyEvaluator.bounded_literal_subn(
                        pattern,
                        redacted,
                        "[REDACTED]",
                        0,
                        output_budget,
                    )
            except re.error:
                continue
    return SupervisedBatchResult(winner, span, redacted)

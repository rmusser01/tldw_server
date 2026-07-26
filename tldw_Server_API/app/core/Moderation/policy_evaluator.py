"""Stateless moderation policy evaluation and redaction."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Moderation.moderation_service import (
        ModerationEvaluationResult,
        ModerationPolicy,
        PatternRule,
    )


_EVALUATION_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    json.JSONDecodeError,
    re.error,
)


@dataclass(frozen=True)
class EvaluationLimits:
    max_scan_chars: int
    match_window_chars: int
    max_fallback_scan_chars: int
    max_replacements_per_pattern: int | None


class PolicyEvaluator:
    """Evaluate and redact text using explicit policy and limit inputs."""

    _UNCATEGORIZED_CATEGORY = "uncategorized"

    @staticmethod
    def policy_types() -> tuple[
        type[ModerationPolicy],
        type[PatternRule],
        type[ModerationEvaluationResult],
    ]:
        """Load service-owned dataclasses lazily to avoid import cycles."""

        from tldw_Server_API.app.core.Moderation.moderation_service import (
            ModerationEvaluationResult,
            ModerationPolicy,
            PatternRule,
        )

        return ModerationPolicy, PatternRule, ModerationEvaluationResult

    @classmethod
    def effective_rule_categories(cls, rule: PatternRule) -> set[str]:
        cats = rule.categories or set()
        normalized = {str(c).strip().lower() for c in cats if str(c).strip()}
        return normalized if normalized else {cls._UNCATEGORIZED_CATEGORY}

    @staticmethod
    def rule_applies_to_phase(rule: PatternRule, phase: str | None) -> bool:
        if phase not in {"input", "output"}:
            return True
        rule_phase = str(getattr(rule, "phase", "both") or "both").strip().lower()
        if rule_phase not in {"input", "output", "both"}:
            rule_phase = "both"
        return rule_phase in {"both", phase}

    @classmethod
    def rule_matches_enabled_categories(
        cls,
        rule: PatternRule,
        categories_enabled: set[str] | None,
    ) -> bool:
        if not categories_enabled or "*" in categories_enabled:
            return True
        rule_categories = cls.effective_rule_categories(rule)
        if "*" in rule_categories:
            return True
        return bool(rule_categories & categories_enabled)

    @staticmethod
    def build_sanitized_snippet_for_replacement(
        text: str,
        match_span: tuple[int, int],
        replacement: str,
    ) -> str | None:
        if not text or not match_span:
            return None
        start, end = match_span
        if start < 0:
            start = 0
        if end < start:
            end = start
        if start > len(text):
            start = len(text)
        if end > len(text):
            end = len(text)
        left_start = max(0, start - 16)
        right_end = min(len(text), end + 16)
        snippet = (text[left_start:start] + (replacement or "[REDACTED]") + text[end:right_end]).strip()
        return snippet[:77] + "..." if len(snippet) > 80 else snippet

    def build_sanitized_snippet(
        self,
        text: str,
        policy: ModerationPolicy,
        match_span: tuple[int, int] | None,
        pattern: str | None = None,
    ) -> str | None:
        if not text or not match_span:
            return None
        _, PatternRule, _ = self.policy_types()
        replacement = policy.redact_replacement or "[REDACTED]"
        if pattern and policy.block_patterns:
            for rule in policy.block_patterns:
                if not isinstance(rule, PatternRule):
                    continue
                try:
                    if getattr(rule.regex, "pattern", None) == pattern:
                        if rule.replacement:
                            replacement = rule.replacement
                        break
                except _EVALUATION_NONCRITICAL_EXCEPTIONS:
                    continue
        return self.build_sanitized_snippet_for_replacement(
            text,
            match_span,
            replacement,
        )

    @staticmethod
    def iter_scan_chunks(
        text: str,
        limits: EvaluationLimits,
    ) -> Iterator[tuple[int, int]]:
        if not text:
            return
        chunk_size = max(1, int(limits.max_scan_chars))
        if len(text) <= chunk_size:
            yield 0, len(text)
            return
        overlap = min(1024, max(32, chunk_size // 10))
        if overlap >= chunk_size:
            overlap = max(0, chunk_size - 1)
        step = chunk_size - overlap if chunk_size > overlap else chunk_size
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(text_len, start + chunk_size)
            yield start, end
            if end == text_len:
                break
            start += step

    def find_match_span(
        self,
        pattern: re.Pattern[str],
        text: str,
        limits: EvaluationLimits,
    ) -> tuple[int, int] | None:
        try:
            chunk_limit = max(1, int(limits.max_scan_chars))
            if len(text) <= chunk_limit:
                match = pattern.search(text)
                return (match.start(), match.end()) if match else None
            text_len = len(text)
            window = max(0, int(limits.match_window_chars))
            for start, end in self.iter_scan_chunks(text, limits):
                window_end = min(text_len, end + window)
                match = pattern.search(text, start, window_end)
                if match and match.start() < end:
                    return match.start(), match.end()
            fallback_limit = max(1, int(limits.max_fallback_scan_chars))
            if len(text) <= fallback_limit:
                match = pattern.search(text)
                if match:
                    return match.start(), match.end()
            return None
        except re.error:
            return None

    @staticmethod
    def collect_rule_matches(
        text: str,
        pattern: re.Pattern[str],
        limits: EvaluationLimits,
    ) -> list[re.Match[str]]:
        if not text:
            return []
        limit = limits.max_replacements_per_pattern
        if limit is not None and int(limit) <= 0:
            limit = None
        matches = []
        try:
            for match in pattern.finditer(text):
                start, end = match.span()
                if start == end:
                    continue
                matches.append(match)
                if limit is not None and len(matches) >= limit:
                    break
        except re.error:
            return []
        return matches

    @staticmethod
    def apply_rule_redactions(
        text: str,
        matches: list[re.Match[str]],
        replacement: str,
    ) -> str:
        if not matches:
            return text
        parts = []
        last = 0
        for match in matches:
            start, end = match.span()
            if start < last:
                continue
            parts.append(text[last:start])
            parts.append(replacement)
            last = end
        parts.append(text[last:])
        return "".join(parts)

    def redact_text(
        self,
        text: str,
        policy: ModerationPolicy,
        phase: str | None,
        limits: EvaluationLimits,
    ) -> str:
        _, PatternRule, _ = self.policy_types()
        if not text or not policy.block_patterns:
            return text
        if phase == "input" and not policy.input_enabled:
            return text
        if phase == "output" and not policy.output_enabled:
            return text
        redacted = text
        for rule in policy.block_patterns:
            if isinstance(
                rule,
                PatternRule,
            ) and not self.rule_applies_to_phase(rule, phase):
                continue
            if isinstance(
                rule,
                PatternRule,
            ) and not self.rule_matches_enabled_categories(
                rule,
                policy.categories_enabled,
            ):
                continue
            pattern = rule.regex if isinstance(rule, PatternRule) else rule
            replacement_override = None
            if isinstance(rule, PatternRule) and rule.replacement:
                replacement_override = rule.replacement
            try:
                replacement = replacement_override or policy.redact_replacement
                limit_raw = limits.max_replacements_per_pattern
                try:
                    limit = int(limit_raw) if limit_raw is not None else 0
                except _EVALUATION_NONCRITICAL_EXCEPTIONS:
                    limit = 0
                if limit <= 0:
                    limit = 0
                if len(redacted) <= limits.max_scan_chars:
                    redacted = pattern.sub(
                        lambda _match, value=replacement: value,
                        redacted,
                        count=limit,
                    )
                else:
                    matches = self.collect_rule_matches(
                        redacted,
                        pattern,
                        limits,
                    )
                    if matches:
                        redacted = self.apply_rule_redactions(
                            redacted,
                            matches,
                            replacement,
                        )
            except re.error:
                continue
        return redacted

    def redact_text_with_count(
        self,
        text: str,
        policy: ModerationPolicy,
        phase: str | None,
        limits: EvaluationLimits,
    ) -> tuple[str, int]:
        _, PatternRule, _ = self.policy_types()
        if not text or not policy.block_patterns:
            return text, 0
        if phase == "input" and not policy.input_enabled:
            return text, 0
        if phase == "output" and not policy.output_enabled:
            return text, 0
        redacted = text
        total_count = 0
        for rule in policy.block_patterns:
            if isinstance(rule, PatternRule) and not self.rule_applies_to_phase(
                rule,
                phase,
            ):
                continue
            if isinstance(rule, PatternRule) and not self.rule_matches_enabled_categories(
                rule,
                policy.categories_enabled,
            ):
                continue
            pattern = rule.regex if isinstance(rule, PatternRule) else rule
            replacement_override = None
            if isinstance(rule, PatternRule) and rule.replacement:
                replacement_override = rule.replacement
            try:
                replacement = replacement_override or policy.redact_replacement
                limit_raw = limits.max_replacements_per_pattern
                try:
                    limit = int(limit_raw) if limit_raw is not None else 0
                except _EVALUATION_NONCRITICAL_EXCEPTIONS:
                    limit = 0
                if limit <= 0:
                    limit = 0
                if len(redacted) <= limits.max_scan_chars:
                    redacted, count = pattern.subn(
                        lambda _match, value=replacement: value,
                        redacted,
                        count=limit,
                    )
                else:
                    matches = self.collect_rule_matches(
                        redacted,
                        pattern,
                        limits,
                    )
                    count = len(matches)
                    if matches:
                        redacted = self.apply_rule_redactions(
                            redacted,
                            matches,
                            replacement,
                        )
                total_count += count
            except re.error:
                continue
        return redacted, total_count

    def evaluate_text(
        self,
        text: str,
        policy: ModerationPolicy,
        phase: str | None,
        limits: EvaluationLimits,
        *,
        include_redacted_text: bool,
    ) -> ModerationEvaluationResult:
        _, PatternRule, ModerationEvaluationResult = self.policy_types()
        if not text or not policy.enabled:
            return ModerationEvaluationResult()
        enabled_phase = True
        if phase == "input":
            enabled_phase = policy.input_enabled
        elif phase == "output":
            enabled_phase = policy.output_enabled
        if not enabled_phase:
            return ModerationEvaluationResult()
        default_action = "warn"
        if phase == "input":
            default_action = policy.input_action
        elif phase == "output":
            default_action = policy.output_action

        best_action = "pass"
        best_rank = 0
        best_pattern = None
        best_category = None
        best_match_pos = None
        best_match_span = None
        best_replacement = None
        for rule in policy.block_patterns or []:
            pattern = rule.regex if isinstance(rule, PatternRule) else rule
            if isinstance(rule, PatternRule) and not self.rule_applies_to_phase(rule, phase):
                continue
            if isinstance(rule, PatternRule) and not self.rule_matches_enabled_categories(
                rule,
                policy.categories_enabled,
            ):
                continue
            match_span = self.find_match_span(pattern, text, limits)
            if not match_span:
                continue
            action = rule.action if isinstance(rule, PatternRule) and rule.action else default_action
            action = (action or "warn").lower()
            if action not in {"block", "redact", "warn"}:
                action = "warn"
            rank = {"warn": 1, "redact": 2, "block": 3}.get(action, 1)
            match_pos = match_span[0]
            if rank > best_rank or (rank == best_rank and (best_match_pos is None or match_pos < best_match_pos)):
                best_action = action
                best_rank = rank
                best_match_pos = match_pos
                best_match_span = match_span
                best_pattern = pattern.pattern
                best_replacement = (
                    rule.replacement
                    if isinstance(rule, PatternRule) and rule.replacement
                    else policy.redact_replacement
                )
                if isinstance(rule, PatternRule):
                    try:
                        categories = self.effective_rule_categories(rule)
                        if policy.categories_enabled:
                            categories &= set(policy.categories_enabled)
                        if categories and "pii" in categories and len(categories) > 1:
                            categories = {c for c in categories if c != "pii"}
                        best_category = sorted(categories)[0] if categories else None
                    except _EVALUATION_NONCRITICAL_EXCEPTIONS:
                        best_category = None
                else:
                    best_category = None

        if best_action == "pass" or best_match_span is None:
            return ModerationEvaluationResult()
        sample = self.build_sanitized_snippet_for_replacement(
            text,
            best_match_span,
            best_replacement or policy.redact_replacement or "[REDACTED]",
        )
        redacted_text = None
        if include_redacted_text and best_action == "redact":
            redacted_text = self.redact_text(text, policy, phase, limits)
        return ModerationEvaluationResult(
            action=best_action,
            redacted_text=redacted_text,
            matched_pattern=best_pattern,
            category=best_category,
            match_span=best_match_span,
            sample=sample,
        )

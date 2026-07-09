"""Deterministic moderation policy compilation helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Moderation.moderation_service import (
        ModerationPolicy,
        PatternRule,
    )


@dataclass(frozen=True)
class ResolvedModerationConfig:
    """Moderation config values after service config/env resolution."""

    enabled: bool = False
    input_enabled: bool = True
    output_enabled: bool = True
    input_action: str = "block"
    output_action: str = "redact"
    redact_replacement: str = "[REDACTED]"
    per_user_overrides: bool = True
    categories_enabled: set[str] | None = None
    pii_enabled: bool = False


@dataclass(frozen=True)
class PolicyCompilationIssue:
    """Sanitized issue emitted while compiling moderation policy inputs."""

    source: str
    reason: str
    index: int | None = None
    detail: str | None = None


@dataclass
class PolicyCompilationReport:
    """Collects sanitized policy compilation issues for service logging."""

    issues: list[PolicyCompilationIssue] = field(default_factory=list)

    def add(
        self,
        source: str,
        reason: str,
        *,
        index: int | None = None,
        detail: str | None = None,
    ) -> None:
        """Append a sanitized issue entry to the report."""

        self.issues.append(
            PolicyCompilationIssue(
                source=source,
                reason=reason,
                index=index,
                detail=detail,
            )
        )


@dataclass
class PolicyCompilationInput:
    """Input bundle for compiling a global moderation policy."""

    config: ResolvedModerationConfig
    runtime_override: dict[str, object] = field(default_factory=dict)
    blocklist_lines: Iterable[str] = field(default_factory=list)
    user_override: dict[str, object] | None = None
    pii_rules: list[PatternRule] = field(default_factory=list)


@dataclass
class PolicyCompilationResult:
    """Compiled moderation policy and the sanitized report from compilation."""

    policy: ModerationPolicy
    report: PolicyCompilationReport


class PolicyCompiler:
    """Compile moderation policies from resolved config and rule inputs."""

    _ALLOWED_REGEX_FLAGS = {"i", "m", "s", "x"}
    _ALLOWED_ACTIONS = {"block", "redact", "warn"}

    @staticmethod
    def policy_types() -> tuple[type[ModerationPolicy], type[PatternRule]]:
        """Load policy dataclasses lazily to avoid import cycles."""

        from tldw_Server_API.app.core.Moderation.moderation_service import (
            ModerationPolicy,
            PatternRule,
        )

        return ModerationPolicy, PatternRule

    def compile_global(self, data: PolicyCompilationInput) -> PolicyCompilationResult:
        """Compile the global moderation policy from config and blocklist input."""

        report = PolicyCompilationReport()
        config = data.config
        ModerationPolicy, _ = self.policy_types()
        categories_enabled = self.resolve_runtime_categories(
            data.runtime_override,
            config.categories_enabled,
        )
        pii_enabled = self.resolve_runtime_pii(data.runtime_override, config.pii_enabled)
        block_patterns = self.compile_blocklist_lines(data.blocklist_lines, report)
        if pii_enabled:
            block_patterns.extend(list(data.pii_rules or []))

        policy = ModerationPolicy(
            enabled=config.enabled,
            input_enabled=config.input_enabled,
            output_enabled=config.output_enabled,
            input_action=str(config.input_action).lower(),
            output_action=str(config.output_action).lower(),
            redact_replacement=config.redact_replacement,
            per_user_overrides=config.per_user_overrides,
            block_patterns=block_patterns,
            categories_enabled=categories_enabled,
        )
        return PolicyCompilationResult(policy=policy, report=report)

    def compile_user_policy(
        self,
        base_policy: ModerationPolicy,
        override: dict[str, object] | None,
    ) -> PolicyCompilationResult:
        """Compile a per-user policy overlay on top of the base policy."""

        report = PolicyCompilationReport()
        if not override:
            return PolicyCompilationResult(policy=base_policy, report=report)
        ModerationPolicy, _ = self.policy_types()
        policy = ModerationPolicy(
            enabled=self.coalesce_bool(override.get("enabled"), base_policy.enabled),
            input_enabled=self.coalesce_bool(override.get("input_enabled"), base_policy.input_enabled),
            output_enabled=self.coalesce_bool(override.get("output_enabled"), base_policy.output_enabled),
            input_action=str(override.get("input_action", base_policy.input_action)).lower(),
            output_action=str(override.get("output_action", base_policy.output_action)).lower(),
            redact_replacement=str(override.get("redact_replacement", base_policy.redact_replacement)),
            per_user_overrides=base_policy.per_user_overrides,
            block_patterns=list(base_policy.block_patterns or []),
            categories_enabled=self.resolve_categories_override(
                override,
                base_policy.categories_enabled,
            ),
        )
        rules_raw = override.get("rules")
        if isinstance(rules_raw, list):
            for idx, raw_rule in enumerate(rules_raw):
                compiled = self.compile_user_rule(raw_rule, report, idx)
                if compiled is not None:
                    policy.block_patterns.append(compiled)
        return PolicyCompilationResult(policy=policy, report=report)

    def compile_blocklist_lines(
        self,
        lines: Iterable[str] | None,
        report: PolicyCompilationReport | None = None,
    ) -> list[PatternRule]:
        """Compile blocklist lines into pattern rules without materializing input."""

        compiled: list[PatternRule] = []
        active_report = report or PolicyCompilationReport()
        for idx, raw in enumerate(lines or []):
            line = str(raw).strip()
            if not line or line.startswith("#"):
                continue
            expr, action, replacement, categories = self.parse_rule_line(line)
            if expr is None or expr == "":
                active_report.add("blocklist", "empty_pattern", index=idx)
                continue
            if action and not self.is_valid_action(action):
                active_report.add("blocklist", "invalid_action", index=idx)
                continue
            rule = self.compile_rule_expression(
                expr,
                action=action,
                replacement=replacement,
                categories=categories,
                phase="both",
                report=active_report,
                source="blocklist",
                index=idx,
            )
            if rule is not None:
                compiled.append(rule)
        return compiled

    @classmethod
    def parse_rule_line(cls, text: str) -> tuple[str | None, str | None, str | None, set[str] | None]:
        """Parse one blocklist line into expression, action, replacement, and categories."""

        if not text:
            return None, None, None, None
        line = text
        action = None
        replacement = None
        categories: set[str] | None = None
        if "#" in line:
            cut_index = cls._find_category_suffix(line)
            if cut_index != -1:
                after = line[cut_index + 1:]
                cats = {c.strip().lower() for c in after.split(",") if c.strip()}
                if cats:
                    categories = cats
                    line = line[:cut_index].strip()
        if "->" in line:
            lhs, rhs = cls.split_action_directive(line)
            if rhs is not None:
                line = lhs
                if rhs:
                    rhs_lower = rhs.lower()
                    if rhs_lower.startswith("redact:"):
                        action = "redact"
                        replacement = rhs[len("redact:"):].strip()
                    elif rhs_lower in cls._ALLOWED_ACTIONS:
                        action = rhs_lower
                    else:
                        action = rhs
        return line, action, replacement, categories

    @staticmethod
    def _find_category_suffix(text: str) -> int:
        """Return the index of an unescaped category suffix marker."""

        if "#" not in text:
            return -1
        for i in range(len(text) - 1, -1, -1):
            if text[i] != "#":
                continue
            backslash_count = 0
            j = i - 1
            while j >= 0 and text[j] == "\\":
                backslash_count += 1
                j -= 1
            escaped = backslash_count % 2 == 1
            previous = text[i - 1] if i > 0 else ""
            if not escaped and (i == 0 or previous.isspace()):
                return i
        return -1

    @staticmethod
    def split_action_directive(text: str) -> tuple[str, str | None]:
        """Split a blocklist action directive while preserving regex arrows."""

        if "->" not in text:
            return text, None
        in_regex = False
        escape = False
        for i in range(len(text) - 1):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == "/" and i == 0:
                in_regex = True
                continue
            if ch == "/" and in_regex:
                in_regex = False
                continue
            if not in_regex and text[i:i + 2] == "->":
                backslash_count = 0
                j = i - 1
                while j >= 0 and text[j] == "\\":
                    backslash_count += 1
                    j -= 1
                if backslash_count % 2 == 1:
                    continue
                return text[:i].strip(), text[i + 2:].strip()
        return text, None

    @classmethod
    def parse_regex_expr(cls, expr: str) -> tuple[str, str] | None:
        """Parse a `/pattern/flags` expression or return ``None`` for literals."""

        if not expr or not expr.startswith("/"):
            return None
        last_slash = expr.rfind("/")
        if last_slash <= 0:
            return None
        flags = expr[last_slash + 1:]
        if flags:
            lowered = flags.lower()
            if any(ch not in cls._ALLOWED_REGEX_FLAGS for ch in lowered):
                return None
        raw = expr[1:last_slash]
        if raw == "":
            return None
        return raw, flags

    @classmethod
    def regex_flags(cls, flags: str | None) -> int:
        """Convert supported regex flag letters into Python ``re`` flags."""

        value = re.IGNORECASE
        lowered = (flags or "").lower()
        if "i" in lowered:
            value |= re.IGNORECASE
        if "m" in lowered:
            value |= re.MULTILINE
        if "s" in lowered:
            value |= re.DOTALL
        if "x" in lowered:
            value |= re.VERBOSE
        return value

    @classmethod
    def is_valid_action(cls, action: str) -> bool:
        """Return whether an action is allowed in blocklist rules."""

        return str(action).strip().lower() in cls._ALLOWED_ACTIONS

    @staticmethod
    def has_nested_quantifiers(expr: str) -> bool:
        """Detect nested quantifiers that can trigger catastrophic backtracking."""

        if not isinstance(expr, str):
            return False
        stack: list[bool] = []
        escaped = False
        for index, char in enumerate(expr):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == "(":
                stack.append(False)
                continue
            if char == ")":
                if not stack:
                    continue
                group_has_quantifier = stack.pop()
                next_index = index + 1
                while next_index < len(expr) and expr[next_index].isspace():
                    next_index += 1
                group_is_quantified = next_index < len(expr) and expr[next_index] in {"+", "*"}
                if group_has_quantifier and group_is_quantified:
                    return True
                if stack and (group_has_quantifier or group_is_quantified):
                    stack[-1] = True
                continue
            if char in {"+", "*"} and stack:
                stack[-1] = True
        return False

    @staticmethod
    def too_many_groups(expr: str, limit: int = 100) -> bool:
        """Return whether a regex expression exceeds the group-count limit."""

        try:
            return expr.count("(") - expr.count("\\(") > limit
        except (TypeError, ValueError):
            return False

    def is_regex_dangerous(self, expr: str) -> bool:
        """Return whether a regex should be rejected before compilation."""

        if not expr:
            return True
        if len(expr) > 2000:
            return True
        if self.has_nested_quantifiers(expr):
            return True
        return self.too_many_groups(expr)

    def compile_rule_expression(
        self,
        expr: str,
        *,
        action: str | None,
        replacement: str | None,
        categories: set[str] | None,
        phase: str,
        report: PolicyCompilationReport,
        source: str,
        index: int | None,
    ) -> PatternRule | None:
        """Compile one literal or regex expression into a pattern rule."""

        _, PatternRule = self.policy_types()
        try:
            regex_parts = self.parse_regex_expr(expr)
            if regex_parts:
                raw, flags_str = regex_parts
                if self.is_regex_dangerous(raw):
                    report.add(source, "dangerous_regex", index=index)
                    return None
                flags = self.regex_flags(flags_str)
                regex = re.compile(raw, flags=flags)
            else:
                regex = re.compile(re.escape(expr.replace("\\#", "#")), flags=re.IGNORECASE)
        except re.error:
            report.add(source, "invalid_regex", index=index)
            return None
        return PatternRule(
            regex=regex,
            action=action or None,
            replacement=replacement or None,
            categories=categories or None,
            phase=phase,
        )

    @staticmethod
    def coalesce_bool(value: object, default: bool) -> bool:
        """Parse a truthy/falsy override value, falling back to a default."""

        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}

    @staticmethod
    def parse_bool_value(value: object) -> bool | None:
        """Parse a bool-like value or return ``None`` when invalid."""

        if isinstance(value, bool):
            return value
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "on", "y"}:
                return True
            if text in {"0", "false", "no", "n", "off"}:
                return False
        return None

    def resolve_categories_override(
        self,
        override: dict[str, object],
        default_categories: set[str] | None,
    ) -> set[str] | None:
        """Resolve per-user category overrides against default categories."""

        if "categories_enabled" not in override:
            return set(default_categories) if default_categories is not None else None
        parsed = self.parse_categories_override(override.get("categories_enabled"))
        if parsed is not None:
            return parsed
        return set(default_categories) if default_categories is not None else None

    @staticmethod
    def parse_categories_override(value: object | None) -> set[str] | None:
        """Parse a category override value into a normalized category set."""

        if value is None:
            return None
        if isinstance(value, list):
            return {str(x).strip().lower() for x in value if str(x).strip()}
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return set()
            return {c.strip().lower() for c in text.split(",") if c.strip()}
        return None

    def compile_user_rule(
        self,
        raw_rule: object,
        report: PolicyCompilationReport,
        index: int,
    ) -> PatternRule | None:
        """Compile one per-user quick rule into a pattern rule."""

        if not isinstance(raw_rule, dict):
            report.add("user_rule", "invalid_rule", index=index)
            return None
        pattern = str(raw_rule.get("pattern", "")).strip()
        action = str(raw_rule.get("action", "")).strip().lower()
        phase = str(raw_rule.get("phase", "both")).strip().lower()
        is_regex = self.parse_bool_value(raw_rule.get("is_regex", False))
        if is_regex is None:
            report.add("user_rule", "invalid_is_regex", index=index)
            return None
        if not pattern or action not in {"block", "warn"}:
            report.add("user_rule", "invalid_rule", index=index)
            return None
        if phase not in {"input", "output", "both"}:
            phase = "both"
        _, PatternRule = self.policy_types()
        try:
            if is_regex:
                if self.is_regex_dangerous(pattern):
                    report.add("user_rule", "dangerous_regex", index=index)
                    return None
                regex = re.compile(pattern, flags=re.IGNORECASE)
            else:
                regex = re.compile(re.escape(pattern), flags=re.IGNORECASE)
        except re.error:
            report.add("user_rule", "invalid_regex", index=index)
            return None
        return PatternRule(
            regex=regex,
            action=action,
            replacement=None,
            categories={"*"},
            phase=phase,
        )

    @staticmethod
    def resolve_runtime_pii(runtime_override: dict[str, object], default: bool) -> bool:
        """Resolve effective PII enablement from runtime overrides."""

        if "pii_enabled" in runtime_override:
            return bool(runtime_override.get("pii_enabled"))
        return bool(default)

    @staticmethod
    def resolve_runtime_categories(
        runtime_override: dict[str, object],
        default: set[str] | None,
    ) -> set[str] | None:
        """Resolve effective categories from runtime overrides."""

        if "categories_enabled" not in runtime_override:
            return set(default) if default is not None else None
        raw = runtime_override.get("categories_enabled") or []
        if isinstance(raw, (set, list, tuple)):
            parsed = {str(c).strip().lower() for c in raw if str(c).strip()}
            return parsed or None
        if isinstance(raw, str):
            parsed = {c.strip().lower() for c in raw.split(",") if c.strip()}
            return parsed or None
        return set(default) if default is not None else None

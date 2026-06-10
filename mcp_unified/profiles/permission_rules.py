"""Claude-style permission rule parsing for MCP profile policies."""

from __future__ import annotations

import fnmatch
import shlex
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, cast
from urllib.parse import urlparse

from .decisions import (
    PolicyDecision,
    PolicyDecisionOutcome,
    PolicyDecisionRule,
    PolicyDecisionSubject,
    PolicyMatchedRule,
    merge_policy_decisions,
)

PermissionRuleSubject = Literal["tool", "command", "path", "domain", "mcp", "skill", "agent"]

_COMMAND_TOOLS = frozenset({"Bash", "Shell", "PowerShell", "Monitor"})
_PATH_TOOLS = frozenset({"Read", "Edit", "Write", "NotebookEdit", "Grep", "Glob", "LSP"})
_DOMAIN_TOOLS = frozenset({"WebFetch", "WebSearch"})
_SHELL_CONTROL_TOKENS = frozenset(
    {
        "&&",
        "||",
        ";",
        "|",
        "&",
        "(",
        ")",
        "<",
        ">",
        ">>",
        "<<",
    }
)
_SHELL_CONTROL_FRAGMENTS = frozenset({"&&", "||", ";", "|", "<", ">", "$(", "`"})
_ARGV_TOKEN_WILDCARD = chr(42)


def parse_permission_rule(
    pattern: str,
    *,
    outcome: PolicyDecisionOutcome = "allow",
    source: str = "permission_rules",
    reason_code: str | None = None,
) -> PolicyDecisionRule:
    """Parse one Claude-style permission rule into a policy decision rule."""

    raw_pattern = _required_string(pattern, "permission rule pattern")
    if raw_pattern.startswith("mcp__"):
        return PolicyDecisionRule(
            rule_type="mcp",
            outcome=outcome,
            source=source,
            pattern=raw_pattern.lower(),
            reason_code=reason_code,
        )

    tool_name, specifier = _split_tool_specifier(raw_pattern)
    if tool_name is None:
        return PolicyDecisionRule(
            rule_type="tool",
            outcome=outcome,
            source=source,
            pattern=raw_pattern,
            reason_code=reason_code,
        )

    if specifier is None or not specifier.strip():
        raise ValueError("permission rule specifier cannot be empty")
    normalized_specifier = specifier.strip()

    if tool_name in _COMMAND_TOOLS:
        return PolicyDecisionRule(
            rule_type="command",
            outcome=outcome,
            source=source,
            pattern=f"{tool_name}({normalized_specifier})",
            argv=_command_pattern_argv(tool_name, normalized_specifier),
            reason_code=reason_code,
        )
    if tool_name in _PATH_TOOLS:
        return PolicyDecisionRule(
            rule_type="path",
            outcome=outcome,
            source=source,
            pattern=_normalize_path_pattern(normalized_specifier),
            reason_code=reason_code,
        )
    if tool_name in _DOMAIN_TOOLS:
        return PolicyDecisionRule(
            rule_type="domain",
            outcome=outcome,
            source=source,
            pattern=_normalize_domain_pattern(normalized_specifier),
            reason_code=reason_code,
        )
    if tool_name == "Skill":
        return PolicyDecisionRule(
            rule_type="skill",
            outcome=outcome,
            source=source,
            pattern=normalized_specifier,
            reason_code=reason_code,
        )
    if tool_name == "Agent":
        return PolicyDecisionRule(
            rule_type="agent",
            outcome=outcome,
            source=source,
            pattern=normalized_specifier,
            reason_code=reason_code,
        )

    return PolicyDecisionRule(
        rule_type="tool",
        outcome=outcome,
        source=source,
        pattern=raw_pattern,
        reason_code=reason_code,
    )


def compile_permission_rules(
    policy_document: Any,
    *,
    field_name: str = "permission_rules",
) -> list[PolicyDecisionRule]:
    """Compile Claude-style permission rules from a policy document field."""

    rules: list[PolicyDecisionRule] = []
    for index, rule_document in enumerate(_as_sequence(_policy_value(policy_document, field_name))):
        rules.append(_compile_permission_rule(rule_document, source=f"{field_name}[{index}]"))
    return rules


def evaluate_permission_rule_decision(
    rules: Iterable[PolicyDecisionRule],
    *,
    subject_type: PermissionRuleSubject,
    value: str,
    argv: Sequence[str] | None = None,
) -> PolicyDecision:
    """Evaluate one subject against compiled permission rules."""

    normalized_value = _normalize_subject_value(subject_type, value)
    subject = PolicyDecisionSubject(type=subject_type, normalized=normalized_value)
    command_argv = _normalize_command_argv(argv, value) if subject_type == "command" else None

    decisions: list[PolicyDecision] = []
    for rule in rules:
        if rule.rule_type != subject_type:
            continue
        if not _rule_matches(rule, subject_type, normalized_value, command_argv):
            continue

        reason_code = _permission_rule_reason_code(rule)
        decisions.append(
            PolicyDecision(
                outcome=rule.outcome,
                reason_code=reason_code,
                subject=subject,
                matched_rules=[
                    PolicyMatchedRule(
                        source=rule.source,
                        rule_type=rule.rule_type,
                        pattern=rule.pattern,
                        outcome=rule.outcome,
                        reason_code=reason_code,
                    )
                ],
            )
        )

    return merge_policy_decisions(
        decisions,
        subject=subject,
        default_reason_code="permission_rule_not_allowed",
    )


def _compile_permission_rule(rule_document: Any, *, source: str) -> PolicyDecisionRule:
    """Compile one string, mapping, or precompiled permission rule."""

    if isinstance(rule_document, PolicyDecisionRule):
        return PolicyDecisionRule.model_validate(rule_document.model_dump())
    if isinstance(rule_document, str):
        return parse_permission_rule(rule_document, source=source)

    pattern = _required_string(_policy_value(rule_document, "pattern"), "permission rule pattern")
    outcome = _rule_outcome(rule_document)
    reason_code = _optional_string(_policy_value(rule_document, "reason_code"))
    return parse_permission_rule(
        pattern,
        outcome=outcome,
        source=source,
        reason_code=reason_code,
    )


def _split_tool_specifier(pattern: str) -> tuple[str | None, str | None]:
    """Split ToolName(specifier) while leaving plain tool names untouched."""

    if not pattern.endswith(")") or "(" not in pattern:
        return None, None
    tool_name, specifier = pattern.split("(", 1)
    if not tool_name:
        return None, None
    return tool_name, specifier[:-1]


def _command_pattern_argv(tool_name: str, specifier: str) -> tuple[str, ...]:
    """Parse a command permission specifier into bounded argv tokens."""

    normalized = specifier.strip()
    if normalized == "*":
        if tool_name == "Bash":
            raise ValueError("broad bash patterns are not allowed")
        raise ValueError("broad command patterns are not allowed")
    if any(fragment in normalized for fragment in _SHELL_CONTROL_FRAGMENTS):
        raise ValueError("unsupported shell control token in command permission rule")
    try:
        argv = tuple(shlex.split(normalized, posix=True))
    except ValueError as exc:
        raise ValueError("invalid command permission rule syntax") from exc
    if not argv:
        raise ValueError("permission rule specifier cannot be empty")
    if any(token in _SHELL_CONTROL_TOKENS for token in argv):
        raise ValueError("unsupported shell control token in command permission rule")
    return argv


def _normalize_path_pattern(pattern: str) -> str:
    """Normalize a path permission pattern without resolving host paths."""

    normalized = pattern.replace("\\", "/").strip()
    if not normalized:
        raise ValueError("permission rule specifier cannot be empty")
    return normalized


def _normalize_domain_pattern(pattern: str) -> str:
    """Normalize a WebFetch/WebSearch domain pattern from a URL or host."""

    raw_pattern = pattern.strip()
    parsed = urlparse(raw_pattern)
    if parsed.scheme and parsed.netloc:
        host = parsed.netloc
    else:
        host = raw_pattern.split("/", 1)[0]
    host = host.strip().lower()
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    if ":" in host and not host.startswith("["):
        host = host.split(":", 1)[0]
    if not host:
        raise ValueError("permission rule specifier cannot be empty")
    return host


def _normalize_subject_value(subject_type: PermissionRuleSubject, value: str) -> str:
    """Normalize a requested subject for permission matching."""

    normalized = _required_string(value, "permission subject value")
    if subject_type == "path":
        return _normalize_path_pattern(normalized)
    if subject_type == "domain":
        return _normalize_domain_pattern(normalized)
    if subject_type == "mcp":
        return normalized.lower()
    return normalized


def _normalize_command_argv(argv: Sequence[str] | None, value: str) -> tuple[str, ...]:
    """Return argv tokens for command subject matching."""

    if argv is None:
        try:
            argv = shlex.split(value, posix=True)
        except ValueError as exc:
            raise ValueError("invalid command subject syntax") from exc
    if isinstance(argv, (str, bytes, Mapping)) or not isinstance(argv, Sequence):
        raise ValueError("command subject argv must be a sequence")
    command_argv = tuple(argv)
    if not command_argv or not all(isinstance(token, str) and token for token in command_argv):
        raise ValueError("command subject argv must contain non-empty strings")
    return command_argv


def _rule_matches(
    rule: PolicyDecisionRule,
    subject_type: PermissionRuleSubject,
    value: str,
    command_argv: tuple[str, ...] | None,
) -> bool:
    """Return whether one compiled rule matches a normalized subject."""

    if subject_type == "command":
        return command_argv is not None and _argv_matches(rule.argv, command_argv)
    if rule.pattern is None:
        return False
    if subject_type == "tool":
        return rule.pattern == value
    return fnmatch.fnmatchcase(value, rule.pattern)


def _argv_matches(pattern_argv: tuple[str, ...] | None, command_argv: tuple[str, ...]) -> bool:
    """Match argv tokens where `*` matches exactly one token."""

    if pattern_argv is None or len(pattern_argv) != len(command_argv):
        return False
    return all(
        pattern_token in (_ARGV_TOKEN_WILDCARD, command_token)
        for pattern_token, command_token in zip(pattern_argv, command_argv)
    )


def _permission_rule_reason_code(rule: PolicyDecisionRule) -> str:
    """Return a stable reason code for a matched permission rule."""

    if rule.reason_code is not None:
        return rule.reason_code
    if rule.outcome == "ask":
        return "approval_required"
    if rule.outcome == "allow":
        return "permission_rule_allowed"
    return "permission_rule_denied"


def _rule_outcome(rule_document: Any) -> PolicyDecisionOutcome:
    """Read and validate a rule outcome/effect pair."""

    outcome = _policy_value(rule_document, "outcome")
    effect = _policy_value(rule_document, "effect")
    if outcome is not None and effect is not None and outcome != effect:
        raise ValueError("conflicting permission rule outcome and effect")
    raw_outcome = outcome if outcome is not None else effect
    if raw_outcome is None:
        raw_outcome = "allow"
    if raw_outcome not in {"deny", "ask", "allow"}:
        raise ValueError("permission rule outcome must be deny, ask, or allow")
    return cast(PolicyDecisionOutcome, raw_outcome)


def _required_string(value: Any, field_name: str) -> str:
    """Return a non-empty string field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    return value.strip()


def _optional_string(value: Any) -> str | None:
    """Return an optional string field."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("permission rule string fields must be strings")
    return value


def _as_sequence(value: Any) -> list[Any]:
    """Return a policy field as a list without splitting strings."""

    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _policy_value(policy_document: Any, key: str) -> Any:
    """Read a field from mappings, Pydantic extras, or objects."""

    if isinstance(policy_document, Mapping):
        return policy_document.get(key)

    value = getattr(policy_document, key, None)
    if value is not None:
        return value

    model_extra = getattr(policy_document, "model_extra", None)
    if isinstance(model_extra, Mapping) and key in model_extra:
        return model_extra[key]
    return None

"""Claude-style permission rule parsing for MCP profile policies.

This module compiles human-authored permission patterns into
``PolicyDecisionRule`` objects and evaluates them against specific subjects.
Claude-style patterns use either an exact tool name, an external MCP identifier,
or ``ToolName(specifier)`` syntax. Supported subject families are tools,
commands, paths, domains, external MCP tools, skills, and agents.

Primary APIs:
- ``parse_permission_rule()`` parses one string rule.
- ``compile_permission_rules()`` compiles a profile policy document field.
- ``evaluate_permission_rule_decision()`` evaluates compiled rules with
  deny-over-ask-over-allow precedence.

Examples:
    ``parse_permission_rule("Read(/docs/**)", outcome="ask")`` produces a
    path rule for files below ``/docs``.
    ``parse_permission_rule("mcp__github__*", outcome="deny")`` produces an
    external MCP wildcard rule.
"""

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
    """Parse one Claude-style permission rule into a PolicyDecisionRule.

    Args:
        pattern: Rule pattern text. Plain names compile as exact tool rules,
            ``mcp__server__tool`` patterns compile as external MCP rules, and
            ``ToolName(specifier)`` forms compile to command, path, domain,
            skill, or agent rules for supported tool families.
        outcome: Decision outcome applied when the rule matches. Defaults to
            ``"allow"``.
        source: Source label copied into matched-rule metadata for policy
            explanations. Defaults to ``"permission_rules"``.
        reason_code: Optional reason code to report when the rule matches.

    Returns:
        A ``PolicyDecisionRule`` with ``rule_type``, normalized ``pattern``,
        optional command ``argv``, outcome, source, and reason metadata.

    Raises:
        ValueError: If the pattern or specifier is empty, if command parsing
            fails, or if a command rule uses broad shell control syntax.

    Example:
        ``parse_permission_rule("Bash(git status)", outcome="allow")``
        returns a command rule with ``argv=("git", "status")``.
    """

    raw_pattern = _required_string(pattern, "permission rule pattern")
    raw_lower = raw_pattern.lower()
    if raw_lower.startswith("mcp__"):
        return PolicyDecisionRule(
            rule_type="mcp",
            outcome=outcome,
            source=source,
            pattern=raw_lower,
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
    """Compile Claude-style permission rules from a policy document field.

    ``compile_permission_rules`` reads ``field_name`` from a mapping,
    Pydantic-style model, or object attribute. The field may contain a single
    rule or a sequence of rules. String entries are parsed with
    ``parse_permission_rule()``. Mapping entries may provide ``pattern``,
    ``outcome``, ``source``, and ``reason_code`` fields. Existing
    ``PolicyDecisionRule`` instances pass through unchanged.

    Args:
        policy_document: Policy document that owns the permission-rule field,
            or ``None`` for no rules.
        field_name: Field or attribute name to read. Defaults to
            ``"permission_rules"``.

    Returns:
        A list of compiled ``PolicyDecisionRule`` objects.

    Raises:
        ValueError: If a rule has an invalid type, outcome, structure, or
            pattern.

    Example:
        ``compile_permission_rules({"permission_rules": ["Read(/docs/**)"]})``
        returns a list containing one path ``PolicyDecisionRule``.
    """

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
    """Evaluate one subject against compiled permission rules.

    Rules are considered only when ``rule.rule_type`` matches ``subject_type``.
    Tool, domain, skill, and agent values use bounded glob matching. MCP values
    are lowercased before glob matching. Path matching is segment-aware:
    ``*`` matches within one path segment and ``**`` may cross segments.
    Command matching compares parsed argv tokens exactly, with ``*`` allowed as
    a single-token wildcard after a fixed executable.

    Matching rules are converted to ``PolicyDecision`` values and merged with
    existing ``merge_policy_decisions()`` precedence, where deny wins over ask
    and ask wins over allow. The function has no side effects.

    Args:
        rules: Iterable of compiled ``PolicyDecisionRule`` objects.
        subject_type: Subject family to evaluate.
        value: Subject value, such as a tool name, path, URL, MCP tool name, or
            command string.
        argv: Optional pre-parsed command argv. Used only when
            ``subject_type`` is ``"command"``.

    Returns:
        A ``PolicyDecision`` with the final outcome, subject metadata, and
        matched-rule details.

    Raises:
        ValueError: If command subject argv is missing or invalid, or if
            command string parsing fails.
    """

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
        host = parsed.hostname or ""
    else:
        host = raw_pattern.split("/", 1)[0]
    host = host.strip().lower()
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    if host.startswith("["):
        closing_bracket = host.find("]")
        if closing_bracket == -1:
            raise ValueError("invalid bracketed IPv6 host")
        host = host[1:closing_bracket]
    elif host.count(":") == 1:
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
    if not command_argv or not isinstance(command_argv[0], str) or not command_argv[0]:
        raise ValueError("command subject argv must include a non-empty executable")
    if not all(isinstance(token, str) for token in command_argv):
        raise ValueError("command subject argv must contain strings")
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
    if subject_type == "path":
        return _path_pattern_matches(rule.pattern, value)
    if subject_type == "mcp":
        return fnmatch.fnmatchcase(value, rule.pattern.lower())
    if subject_type == "tool":
        return rule.pattern == value
    return fnmatch.fnmatchcase(value, rule.pattern)


def _path_pattern_matches(pattern: str, value: str) -> bool:
    """Match path patterns where `*` stays within one path segment."""

    return _path_segments_match(
        _path_pattern_segments(pattern),
        _path_pattern_segments(value),
    )


def _path_pattern_segments(path: str) -> tuple[str, ...]:
    """Return normalized path segments for policy matching."""

    normalized = _normalize_path_pattern(path).strip("/")
    if not normalized or normalized == ".":
        return ()
    return tuple(segment for segment in normalized.split("/") if segment)


def _path_segments_match(
    pattern_segments: tuple[str, ...],
    value_segments: tuple[str, ...],
) -> bool:
    """Recursively match path segments, using `**` for cross-segment matching."""

    if not pattern_segments:
        return not value_segments

    pattern_head = pattern_segments[0]
    pattern_tail = pattern_segments[1:]
    if pattern_head == "**":
        return _path_segments_match(pattern_tail, value_segments) or (
            bool(value_segments)
            and _path_segments_match(pattern_segments, value_segments[1:])
        )

    if not value_segments:
        return False
    return fnmatch.fnmatchcase(value_segments[0], pattern_head) and _path_segments_match(
        pattern_tail,
        value_segments[1:],
    )


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

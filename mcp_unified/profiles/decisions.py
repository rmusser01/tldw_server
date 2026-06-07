"""Decision primitives for MCP profile policy evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PolicyDecisionOutcome = Literal["deny", "ask", "allow"]
PolicyDecisionVisibility = Literal["hidden", "direct", "deferred", "debug_only"]
PolicyDecisionCallState = Literal["blocked", "approval_required", "callable"]
PolicyRuleType = Literal["tool", "command", "mcp", "capability", "risk_class"]

_DEFAULTS_BY_OUTCOME: dict[PolicyDecisionOutcome, dict[str, Any]] = {
    "deny": {
        "visibility": "hidden",
        "call_state": "blocked",
        "requires_approval": False,
    },
    "ask": {
        "visibility": "direct",
        "call_state": "approval_required",
        "requires_approval": True,
    },
    "allow": {
        "visibility": "direct",
        "call_state": "callable",
        "requires_approval": False,
    },
}
_OUTCOME_PRECEDENCE: dict[PolicyDecisionOutcome, int] = {
    "allow": 1,
    "ask": 2,
    "deny": 3,
}
_VALID_OUTCOMES = frozenset(_OUTCOME_PRECEDENCE)
_COMMAND_EXECUTABLE_WILDCARDS = frozenset("*?[]")


class PolicyDecisionSubject(BaseModel):
    """Normalized policy subject used by decision and explain payloads."""

    model_config = ConfigDict(extra="forbid")

    type: str
    normalized: str
    display_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyMatchedRule(BaseModel):
    """Redacted summary of one rule that contributed to a decision."""

    model_config = ConfigDict(extra="forbid")

    source: str
    rule_type: str
    pattern: str | None = None
    outcome: PolicyDecisionOutcome
    reason_code: str | None = None


class PolicyDecisionRule(BaseModel):
    """Compiled rule from legacy or structured profile policy."""

    model_config = ConfigDict(extra="forbid")

    rule_type: PolicyRuleType
    outcome: PolicyDecisionOutcome
    source: str
    pattern: str | None = None
    argv: tuple[str, ...] | None = None
    reason_code: str | None = None

    @field_validator("argv", mode="before")
    @classmethod
    def _validate_argv_shape(cls, value: Any) -> Any:
        return _validate_command_rule_argv_input(value)

    @model_validator(mode="after")
    def _validate_command_rule(self) -> "PolicyDecisionRule":
        if self.rule_type == "command":
            _validate_command_rule_argv(self.argv)
        return self


class PolicyDecision(BaseModel):
    """Final or intermediate permission decision for one policy subject."""

    model_config = ConfigDict(extra="forbid")

    outcome: PolicyDecisionOutcome
    reason_code: str
    subject: PolicyDecisionSubject
    matched_rules: list[PolicyMatchedRule] = Field(default_factory=list)
    visibility: PolicyDecisionVisibility | None = None
    call_state: PolicyDecisionCallState | None = None
    requires_approval: bool | None = None
    explainable: bool = True
    redacted: bool = True

    @model_validator(mode="after")
    def _derive_defaults(self) -> "PolicyDecision":
        defaults = _DEFAULTS_BY_OUTCOME[self.outcome]
        if self.visibility is None:
            self.visibility = defaults["visibility"]
        if self.call_state is None:
            self.call_state = defaults["call_state"]
        if self.requires_approval is None:
            self.requires_approval = bool(defaults["requires_approval"])
        return self


def merge_policy_decisions(
    decisions: list[PolicyDecision],
    *,
    subject: PolicyDecisionSubject,
    default_reason_code: str = "no_matching_rule",
) -> PolicyDecision:
    """Merge intermediate decisions using deny, ask, then allow precedence."""

    if not decisions:
        return PolicyDecision(
            outcome="deny",
            reason_code=default_reason_code,
            subject=subject,
        )

    winning_decision = max(
        decisions,
        key=lambda decision: _OUTCOME_PRECEDENCE[decision.outcome],
    )
    matched_rules = [matched_rule for decision in decisions for matched_rule in decision.matched_rules]
    return PolicyDecision(
        outcome=winning_decision.outcome,
        reason_code=winning_decision.reason_code,
        subject=subject,
        matched_rules=matched_rules,
        visibility=winning_decision.visibility,
        call_state=winning_decision.call_state,
        requires_approval=winning_decision.requires_approval,
        explainable=winning_decision.explainable,
        redacted=winning_decision.redacted,
    )


def compile_profile_policy_rules(policy_document: Any) -> list[PolicyDecisionRule]:
    """Compile legacy and structured profile policy rules into typed records."""

    rules: list[PolicyDecisionRule] = []
    for pattern in _as_sequence(_policy_value(policy_document, "denied_tools")):
        rules.append(
            _compile_legacy_tool_pattern(
                pattern,
                outcome="deny",
                source="denied_tools",
            )
        )
    for pattern in _as_sequence(_policy_value(policy_document, "allowed_tools")):
        rules.append(
            _compile_legacy_tool_pattern(
                pattern,
                outcome="allow",
                source="allowed_tools",
            )
        )

    rules.extend(
        _compile_structured_rules(
            policy_document,
            field_name="tool_rules",
            rule_type="tool",
        )
    )
    rules.extend(
        _compile_structured_rules(
            policy_document,
            field_name="command_rules",
            rule_type="command",
        )
    )
    rules.extend(
        _compile_structured_rules(
            policy_document,
            field_name="mcp_rules",
            rule_type="mcp",
        )
    )
    return rules


def _compile_legacy_tool_pattern(
    pattern: Any,
    *,
    outcome: PolicyDecisionOutcome,
    source: str,
) -> PolicyDecisionRule:
    """Compile one legacy allowed or denied tool pattern."""

    if not isinstance(pattern, str):
        raise ValueError("legacy tool policy patterns must be strings")
    if not pattern.strip():
        raise ValueError("legacy tool policy patterns cannot be empty")
    if pattern.startswith("Bash(") and pattern.endswith(")"):
        return _compile_bash_pattern(
            pattern[len("Bash(") : -1],
            outcome=outcome,
            source=source,
        )
    return PolicyDecisionRule(
        rule_type="tool",
        outcome=outcome,
        source=source,
        pattern=pattern,
    )


def _compile_bash_pattern(
    inner: str,
    *,
    outcome: PolicyDecisionOutcome,
    source: str,
) -> PolicyDecisionRule:
    """Compile a legacy Bash(...) pattern into a command argv rule."""

    return PolicyDecisionRule(
        rule_type="command",
        outcome=outcome,
        source=source,
        pattern=f"Bash({inner})",
        argv=_bash_pattern_argv(inner),
    )


def _bash_pattern_argv(inner: str) -> tuple[str, ...]:
    """Return argv tokens for a legacy Bash(...) pattern."""

    normalized = inner.strip()
    if normalized == "*":
        raise ValueError("broad bash patterns are not allowed")

    argv = tuple(normalized.split())
    if not argv:
        raise ValueError("bash patterns must include a command")
    _validate_command_rule_argv(argv)
    return argv


def _validate_command_rule_argv(argv: tuple[str, ...] | None) -> None:
    """Validate command argv shape and fixed executable token."""

    if argv is None:
        raise ValueError("command policy rule argv is required")
    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise ValueError("command policy rule argv must be non-empty strings")
    _validate_fixed_command_executable(argv)


def _validate_command_rule_argv_input(value: Any) -> Any:
    """Reject argv inputs whose ordering is ambiguous before coercion."""

    if value is None:
        return value
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError("command policy rule argv must be a sequence")
    return value


def _validate_fixed_command_executable(argv: tuple[str, ...]) -> None:
    """Require command rules to name a fixed executable token."""

    executable = argv[0]
    if any(character in _COMMAND_EXECUTABLE_WILDCARDS for character in executable):
        raise ValueError("command executable must be fixed")


def _as_sequence(value: Any) -> list[Any]:
    """Return value as a list while treating strings and mappings as scalars."""

    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _compile_structured_rules(
    policy_document: Any,
    *,
    field_name: str,
    rule_type: PolicyRuleType,
) -> list[PolicyDecisionRule]:
    """Compile one structured profile-policy rule list."""

    rules: list[PolicyDecisionRule] = []
    for index, rule_document in enumerate(_as_sequence(_policy_value(policy_document, field_name))):
        rules.append(
            _compile_structured_rule(
                rule_document,
                rule_type=rule_type,
                source=f"{field_name}[{index}]",
            )
        )
    return rules


def _compile_structured_rule(
    rule_document: Any,
    *,
    rule_type: PolicyRuleType,
    source: str,
) -> PolicyDecisionRule:
    """Compile one structured rule entry into a typed rule."""

    if isinstance(rule_document, PolicyDecisionRule):
        if rule_document.rule_type != rule_type:
            raise ValueError("structured rule type does not match source field")
        _validate_command_rule_argv_input(rule_document.argv)
        return PolicyDecisionRule.model_validate(rule_document.model_dump())

    outcome = _structured_rule_outcome(rule_document)
    reason_code = _optional_string(_policy_value(rule_document, "reason_code"))
    if rule_type == "command":
        argv = _structured_command_argv(rule_document)
        pattern = _optional_string(_policy_value(rule_document, "pattern"))
        return PolicyDecisionRule(
            rule_type="command",
            outcome=outcome,
            source=source,
            pattern=pattern,
            argv=argv,
            reason_code=reason_code,
        )

    pattern = _required_pattern(rule_document, rule_type=rule_type)
    return PolicyDecisionRule(
        rule_type=rule_type,
        outcome=outcome,
        source=source,
        pattern=pattern,
        reason_code=reason_code,
    )


def _structured_rule_outcome(rule_document: Any) -> PolicyDecisionOutcome:
    """Return the validated deny, ask, or allow outcome for a rule."""

    outcome = _policy_value(rule_document, "outcome")
    effect = _policy_value(rule_document, "effect")
    if outcome is not None and effect is not None and outcome != effect:
        raise ValueError("conflicting policy rule outcome and effect")

    raw_outcome = outcome if outcome is not None else effect
    if not isinstance(raw_outcome, str) or raw_outcome not in _VALID_OUTCOMES:
        raise ValueError("policy rule outcome must be deny, ask, or allow")
    return cast(PolicyDecisionOutcome, raw_outcome)


def _structured_command_argv(rule_document: Any) -> tuple[str, ...]:
    """Return validated argv for a structured command rule."""

    argv_value = _policy_value(rule_document, "argv")
    if argv_value is None:
        pattern = _policy_value(rule_document, "pattern")
        if isinstance(pattern, str) and pattern.startswith("Bash(") and pattern.endswith(")"):
            return _bash_pattern_argv(pattern[len("Bash(") : -1])
        raise ValueError("command policy rules require argv")

    argv = tuple(_validate_command_rule_argv_input(argv_value))
    _validate_command_rule_argv(argv)
    return argv


def _required_pattern(rule_document: Any, *, rule_type: PolicyRuleType) -> str:
    """Return the required string pattern for a non-command rule."""

    pattern = _policy_value(rule_document, "pattern")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError(f"{rule_type} policy rules require a pattern")
    return pattern


def _optional_string(value: Any) -> str | None:
    """Return an optional string field, rejecting non-string values."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("policy rule string fields must be strings")
    return value


def _policy_value(policy_document: Any, key: str) -> Any:
    """Read a field from objects, Pydantic extras, or mapping documents."""

    value = getattr(policy_document, key, None)
    if value is not None:
        return value

    model_extra = getattr(policy_document, "model_extra", None)
    if isinstance(model_extra, Mapping) and key in model_extra:
        return model_extra[key]
    if isinstance(policy_document, Mapping):
        return policy_document.get(key)
    return None

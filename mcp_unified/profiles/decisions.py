"""Decision primitives for MCP profile policy evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PolicyDecisionOutcome = Literal["deny", "ask", "allow"]
PolicyDecisionVisibility = Literal["hidden", "direct", "deferred", "debug_only"]
PolicyDecisionCallState = Literal["blocked", "approval_required", "callable"]

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

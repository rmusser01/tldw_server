# MCP Policy Decision Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first MCP/profile policy-decision slice: a package-owned `deny`/`ask`/`allow` decision core, compatibility compilation from existing profile policy fields, and a redacted explain/simulation contract.

**Architecture:** Keep the executable core in `mcp_unified` so the standalone package and tldw_server host share one decision model. Add new decision primitives in a focused `mcp_unified/profiles/decisions.py` module, then thread optional decision metadata through `mcp_unified/profiles/resolution.py` without changing existing runtime allow/deny behavior. Catalog visibility, path-pattern compilation, external MCP wildcard enforcement, hooks, and shell hardening are deliberately deferred to later slices.

**Tech Stack:** Python 3.10+, Pydantic v2 models, existing `mcp_unified.profiles` package, pytest, Bandit.

---

## Context

Spec: `Docs/superpowers/specs/2026-06-07-mcp-profile-policy-decision-model-design.md`

Planning Backlog task: `TASK-2307`

Implementation Backlog task: `TASK-2308`

Branch/worktree: `codex/mcp-profile-policy-decision-design` at `.worktrees/mcp-profile-policy-decision-design`

The current package has:

- `mcp_unified/profiles/models.py`
  - `ProfilePolicy` with `extra="allow"`, so new fields such as `tool_rules`,
    `command_rules`, and `permission_mode` can be read before first-class schema
    fields are added.
- `mcp_unified/profiles/resolution.py`
  - `EffectivePolicy`, `EffectivePolicyResult`, and
    `build_effective_policy_result(...)`.
  - Existing behavior returns `status="denied"` for explicit denied tools and
    unlisted tool execution.
- `mcp_unified/gateway/profile_runtime.py`
  - already consumes `EffectivePolicyResult` for profile-aware gateway runtime
    filtering/calls.
- `tldw_Server_API/app/core/MCP_unified/protocol.py`
  - independently filters by `canExecute` and effective policy; do not change it
    in this slice.

## Scope

In scope:

- Decision model: outcome, visibility, call state, subject, matched rules,
  reason code, approval metadata, redaction flag.
- Merge helper with `deny > ask > allow` precedence.
- Compatibility compiler for existing `allowed_tools`, `denied_tools`, and
  legacy `Bash(...)` command-pattern entries.
- Structured rule support for profile extra fields:
  - `tool_rules`
  - `command_rules`
  - `mcp_rules` only as data model/compile output, not runtime enforcement.
- Optional decision metadata on `EffectivePolicyResult`.
- Package-local explain/simulation helper for tool decisions.
- Exports and focused tests.

Out of scope:

- No runtime catalog visibility changes.
- No path matcher compiler.
- No external MCP wildcard enforcement.
- No shell parser/runtime changes.
- No hook integration.
- No FastAPI routes or CLI commands.
- No behavior change for current allow/deny runtime decisions except richer
  metadata attached to results.

## File Structure

- Create `mcp_unified/profiles/decisions.py`
  - Owns package-neutral Pydantic models, rule compilation, merge, and explain
    helpers.
- Modify `mcp_unified/profiles/resolution.py`
  - Adds optional decision metadata to `EffectivePolicyResult`.
  - Uses decision helper when a `tool_name` is supplied.
  - Preserves existing `status` and `reason_code` outputs for compatibility.
- Modify `mcp_unified/profiles/__init__.py`
  - Re-export public decision models/helpers.
- Create `tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py`
  - Focused tests for new decision primitives and compiler behavior.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py`
  - Add assertions that existing resolution results now carry compatible
    decision metadata.
- Modify `backlog/tasks/task-2308 - Implement-MCP-policy-decision-core.md`
  - Track implementation outcome and validation when executing this plan.

## Behavioral Contracts

Decision outcomes:

```python
PolicyDecisionOutcome = Literal["deny", "ask", "allow"]
```

Outcome-derived defaults:

| Outcome | visibility | call_state | requires_approval |
| --- | --- | --- | --- |
| `deny` | `hidden` | `blocked` | `False` |
| `ask` | `direct` | `approval_required` | `True` |
| `allow` | `direct` | `callable` | `False` |

Merge precedence:

```text
deny > ask > allow
```

Compatibility requirements:

- Existing `denied_tools` entries compile to `tool` deny rules.
- Existing `allowed_tools` entries compile to `tool` allow rules.
- Existing `Bash(git *)` entries compile to `command` rules with
  `argv=("git", "*")`.
- `Bash(*)` should compile as invalid/rejected for this first slice.
- Existing `build_effective_policy_result(...)` status/reason behavior remains
  unchanged for current callers.
- Decision/explain data must not include raw tool arguments, file content, raw
  diffs, read receipts, credential values, environment values, or absolute host
  paths.

## Task 1: Add Decision Model Primitives

**Files:**
- Create: `mcp_unified/profiles/decisions.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py`

- [ ] **Step 1: Write failing model tests**

Add tests:

```python
from mcp_unified.profiles.decisions import (
    PolicyDecision,
    PolicyDecisionSubject,
)


def test_policy_decision_defaults_for_ask() -> None:
    decision = PolicyDecision(
        outcome="ask",
        reason_code="approval_required",
        subject=PolicyDecisionSubject(type="tool", normalized="fs.patch"),
    )

    assert decision.visibility == "direct"
    assert decision.call_state == "approval_required"
    assert decision.requires_approval is True


def test_policy_decision_defaults_for_deny() -> None:
    decision = PolicyDecision(
        outcome="deny",
        reason_code="tool_denied",
        subject=PolicyDecisionSubject(type="tool", normalized="fs.write"),
    )

    assert decision.visibility == "hidden"
    assert decision.call_state == "blocked"
    assert decision.requires_approval is False
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py \
  -q
```

Expected: FAIL because `mcp_unified.profiles.decisions` does not exist.

- [ ] **Step 3: Implement model primitives**

Create `mcp_unified/profiles/decisions.py` with these public models:

```python
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PolicyDecisionOutcome = Literal["deny", "ask", "allow"]
PolicyDecisionVisibility = Literal["hidden", "direct", "deferred", "debug_only"]
PolicyDecisionCallState = Literal["blocked", "approval_required", "callable"]


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
```

Use an internal `_DEFAULTS_BY_OUTCOME` mapping to avoid scattering default
logic.

- [ ] **Step 4: Run focused tests**

Run the same pytest command.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp_unified/profiles/decisions.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py
git commit -m "feat: add mcp policy decision models"
```

## Task 2: Add Decision Merge And Rule Compilation

**Files:**
- Modify: `mcp_unified/profiles/decisions.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py`

- [ ] **Step 1: Write failing merge/compiler tests**

Add tests:

```python
from mcp_unified.profiles.decisions import (
    compile_profile_policy_rules,
    merge_policy_decisions,
)
from mcp_unified.profiles.models import ProfilePolicy


def test_merge_policy_decisions_uses_deny_over_ask_over_allow() -> None:
    subject = PolicyDecisionSubject(type="tool", normalized="fs.write")
    merged = merge_policy_decisions(
        [
            PolicyDecision(outcome="allow", reason_code="allowed", subject=subject),
            PolicyDecision(outcome="ask", reason_code="approval_required", subject=subject),
            PolicyDecision(outcome="deny", reason_code="tool_denied", subject=subject),
        ],
        subject=subject,
    )

    assert merged.outcome == "deny"
    assert merged.reason_code == "tool_denied"
    assert merged.call_state == "blocked"


def test_compile_profile_policy_rules_preserves_legacy_tool_fields() -> None:
    rules = compile_profile_policy_rules(
        ProfilePolicy(
            allowed_tools=["fs.read"],
            denied_tools=["fs.write"],
        )
    )

    assert [(rule.rule_type, rule.pattern, rule.outcome) for rule in rules] == [
        ("tool", "fs.write", "deny"),
        ("tool", "fs.read", "allow"),
    ]


def test_compile_profile_policy_rules_converts_legacy_bash_pattern_to_argv_rule() -> None:
    rules = compile_profile_policy_rules(ProfilePolicy(allowed_tools=["Bash(git *)"]))

    command_rule = rules[0]
    assert command_rule.rule_type == "command"
    assert command_rule.argv == ("git", "*")
    assert command_rule.outcome == "allow"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py \
  -q
```

Expected: FAIL because merge/compiler helpers do not exist.

- [ ] **Step 3: Implement rule model and merge helper**

Add:

```python
PolicyRuleType = Literal["tool", "command", "mcp", "capability", "risk_class"]


class PolicyDecisionRule(BaseModel):
    """Compiled rule from legacy or structured profile policy."""

    model_config = ConfigDict(extra="forbid")

    rule_type: PolicyRuleType
    outcome: PolicyDecisionOutcome
    source: str
    pattern: str | None = None
    argv: tuple[str, ...] | None = None
    reason_code: str | None = None
```

Add:

```python
def merge_policy_decisions(
    decisions: list[PolicyDecision],
    *,
    subject: PolicyDecisionSubject,
    default_reason_code: str = "no_matching_rule",
) -> PolicyDecision:
    ...
```

Implementation rules:

- Empty list returns `deny` with `default_reason_code`.
- Highest precedence outcome wins.
- Keep matched rules from all decisions in the merged payload.
- Use the first decision at the winning precedence as the reason source.

- [ ] **Step 4: Implement compatibility compiler**

Add:

```python
def compile_profile_policy_rules(policy_document: Any) -> list[PolicyDecisionRule]:
    ...
```

Implementation rules:

- Read `denied_tools` first so output order records deny precedence.
- Read `allowed_tools` second.
- Accept `ProfilePolicy` or mapping-like objects.
- Compile plain strings to `rule_type="tool"`.
- Compile `Bash(git *)` to `rule_type="command"`, `argv=("git", "*")`.
- Reject or skip `Bash(*)` with a validation warning model only if warnings are
  added now; otherwise raise `ValueError("broad bash patterns are not allowed")`.
- Compile structured extra `tool_rules`, `command_rules`, and `mcp_rules` when
  present, but do not integrate them into runtime enforcement yet.
- Read structured extra fields through `getattr(policy_document, key, None)`
  or `policy_document.model_extra` so Pydantic `extra="allow"` values are
  handled consistently.

Keep helper functions small:

- `_compile_legacy_tool_pattern(pattern, *, outcome, source)`
- `_compile_bash_pattern(inner, *, outcome, source)`
- `_as_sequence(value)`

- [ ] **Step 5: Run focused tests**

Run the same pytest command.

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mcp_unified/profiles/decisions.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py
git commit -m "feat: compile mcp profile decision rules"
```

## Task 3: Evaluate Tool Decisions And Preserve Resolution Compatibility

**Files:**
- Modify: `mcp_unified/profiles/decisions.py`
- Modify: `mcp_unified/profiles/resolution.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py`

- [ ] **Step 1: Write failing evaluation tests**

Add decision-level tests:

```python
from mcp_unified.profiles.decisions import evaluate_profile_tool_decision
from mcp_unified.profiles.models import MCPProfile, ProfilePolicy


def test_evaluate_profile_tool_decision_denied_tool_wins() -> None:
    profile = MCPProfile(
        id="strict",
        name="Strict",
        policy_document=ProfilePolicy(
            allowed_tools=["fs.write"],
            denied_tools=["fs.write"],
        ),
    )

    decision = evaluate_profile_tool_decision(profile, "fs.write")

    assert decision.outcome == "deny"
    assert decision.reason_code == "tool_denied"


def test_evaluate_profile_tool_decision_allowed_tool_allows() -> None:
    profile = MCPProfile(
        id="reader",
        name="Reader",
        policy_document=ProfilePolicy(allowed_tools=["fs.read"]),
    )

    decision = evaluate_profile_tool_decision(profile, "fs.read")

    assert decision.outcome == "allow"
    assert decision.call_state == "callable"


def test_evaluate_profile_tool_decision_structured_ask_rule_requires_approval() -> None:
    profile = MCPProfile(
        id="default-ask",
        name="Default Ask",
        policy_document=ProfilePolicy.model_validate(
            {"tool_rules": [{"pattern": "fs.patch", "outcome": "ask"}]}
        ),
    )

    decision = evaluate_profile_tool_decision(profile, "fs.patch")

    assert decision.outcome == "ask"
    assert decision.call_state == "approval_required"
```

Add resolution compatibility assertions to existing tests:

```python
assert result.decision is not None
assert result.decision.outcome == "deny"
```

for denied tool cases, and:

```python
assert result.decision is not None
assert result.decision.outcome == "allow"
```

for allowed tool cases.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py \
  -q
```

Expected: FAIL because evaluation helper and `EffectivePolicyResult.decision`
do not exist.

- [ ] **Step 3: Implement tool decision evaluation**

Add:

```python
def evaluate_profile_tool_decision(
    profile: MCPProfile,
    tool_name: str,
    *,
    capability: str | None = None,
) -> PolicyDecision:
    ...
```

Implementation rules:

- Subject is `PolicyDecisionSubject(type="tool", normalized=tool_name)`.
- Blank tool names raise `ValueError`.
- Explicit denied tool returns `deny/tool_denied`.
- Structured tool rule with `outcome="ask"` returns ask.
- Explicit allowed tool returns `allow/tool_allowed`.
- If no allowed tools exist but capability is provided and allowed, return
  `allow/capability_allowed`.
- Otherwise return `deny/tool_not_allowed`.
- Keep existing capability-deny behavior compatible when called from
  resolution.

- [ ] **Step 4: Thread decision metadata through resolution**

Modify `EffectivePolicyResult`:

```python
decision: PolicyDecision | None = None
```

In `build_effective_policy_result(...)`:

- Compute `decision` when `tool_name is not None`.
- Preserve existing `status` and `reason_code` mappings:
  - decision `deny/tool_denied` -> `status="denied"`, `reason_code="tool_denied"`.
  - decision `deny/tool_not_allowed` -> `status="denied"`, `reason_code="tool_not_allowed"`.
  - decision `ask/*` -> `status="approval_required"`, `reason_code=decision.reason_code`.
  - decision `allow/*` -> continue normal capability checks and return `resolved`.
- Return the decision on both denied and resolved results.

Do not alter workspace-binding denial behavior except to leave `decision=None`
for that non-tool subject in this first slice.

- [ ] **Step 5: Run focused tests**

Run the same pytest command.

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mcp_unified/profiles/decisions.py mcp_unified/profiles/resolution.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py
git commit -m "feat: attach mcp profile policy decisions"
```

## Task 4: Add Explain/Simulation Contract

**Files:**
- Modify: `mcp_unified/profiles/decisions.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py`

- [ ] **Step 1: Write failing explain tests**

Add tests:

```python
from mcp_unified.profiles.decisions import explain_profile_tool_decision


def test_explain_profile_tool_decision_returns_redacted_payload() -> None:
    profile = MCPProfile(
        id="qa",
        name="QA",
        policy_document=ProfilePolicy(denied_tools=["fs.write"]),
    )

    explanation = explain_profile_tool_decision(profile, "fs.write")

    assert explanation.final_outcome == "deny"
    assert explanation.reason_code == "tool_denied"
    assert explanation.profile_id == "qa"
    assert explanation.subject.normalized == "fs.write"
    assert explanation.redacted is True
    assert explanation.matches[0].source == "policy_document.denied_tools"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py \
  -q
```

Expected: FAIL because explain models/helpers do not exist.

- [ ] **Step 3: Implement explain models**

Add:

```python
class PolicyExplanation(BaseModel):
    """Redacted operator/debug explanation for one simulated policy decision."""

    model_config = ConfigDict(extra="forbid")

    final_outcome: PolicyDecisionOutcome
    reason_code: str
    subject: PolicyDecisionSubject
    matches: list[PolicyMatchedRule] = Field(default_factory=list)
    profile_id: str | None = None
    permission_mode: str | None = None
    visibility: PolicyDecisionVisibility
    call_state: PolicyDecisionCallState
    requires_approval: bool
    hook_results: list[dict[str, Any]] = Field(default_factory=list)
    sandbox: dict[str, Any] = Field(default_factory=dict)
    redacted: bool = True
```

Add:

```python
def explain_profile_tool_decision(
    profile: MCPProfile,
    tool_name: str,
    *,
    capability: str | None = None,
) -> PolicyExplanation:
    ...
```

Implementation rules:

- Call `evaluate_profile_tool_decision(...)`.
- Copy only safe scalar metadata.
- Include `permission_mode` if present as an extra field on the policy document.
- Keep `hook_results=[]` and `sandbox={}` in this slice.

- [ ] **Step 4: Run focused tests**

Run the same pytest command.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp_unified/profiles/decisions.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py
git commit -m "feat: add mcp policy explanation contract"
```

## Task 5: Export Public Decision API And Verify Package Boundary

**Files:**
- Modify: `mcp_unified/profiles/__init__.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py`
- Optional Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [ ] **Step 1: Write failing export test**

Add:

```python
def test_profile_decision_api_exports_from_profiles_package() -> None:
    from mcp_unified.profiles import (
        PolicyDecision,
        PolicyDecisionSubject,
        explain_profile_tool_decision,
    )

    assert PolicyDecision is not None
    assert PolicyDecisionSubject is not None
    assert explain_profile_tool_decision is not None
```

If package-boundary tests already cover this import style, add a small assertion
there instead of duplicating.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py \
  -q
```

Expected: FAIL because exports are missing.

- [ ] **Step 3: Add exports**

Modify `mcp_unified/profiles/__init__.py` to export:

- `PolicyDecision`
- `PolicyDecisionCallState`
- `PolicyDecisionOutcome`
- `PolicyDecisionRule`
- `PolicyDecisionSubject`
- `PolicyDecisionVisibility`
- `PolicyExplanation`
- `PolicyMatchedRule`
- `compile_profile_policy_rules`
- `evaluate_profile_tool_decision`
- `explain_profile_tool_decision`
- `merge_policy_decisions`

- [ ] **Step 4: Run focused tests**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp_unified/profiles/__init__.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py
git commit -m "feat: export mcp policy decision api"
```

## Task 6: Final Validation And Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-2308 - Implement-MCP-policy-decision-core.md`

- [ ] **Step 1: Run targeted tests**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py::test_gateway_profile_runtime_filters_and_allows_default_profile_tools \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py::test_protocol_tools_call_blocks_tool_denied_by_effective_policy \
  -q
```

Expected: PASS. The gateway/protocol tests are smoke coverage that existing
runtime policy behavior did not regress.

- [ ] **Step 2: Run Bandit on touched code**

Run:

```bash
source ../../.venv/bin/activate && python -m bandit -r \
  mcp_unified/profiles/decisions.py \
  mcp_unified/profiles/resolution.py \
  mcp_unified/profiles/__init__.py \
  -f json -o /tmp/bandit_mcp_policy_decision_core.json
```

Expected: exit 0 or only documented non-touched/baseline findings. Fix any new
touched-code findings before continuing.

- [ ] **Step 3: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 4: Update Backlog task**

Record:

- implementation summary;
- test commands and outcomes;
- Bandit output path and outcome;
- known skips/blockers;
- touched files.

- [ ] **Step 5: Commit final task update**

```bash
git add "backlog/tasks/task-2308 - Implement-MCP-policy-decision-core.md"
git commit -m "chore: close mcp policy decision core task"
```

## Risk Review

- **Runtime behavior drift:** This slice must not change existing allow/deny
  enforcement except adding metadata. Keep runtime catalog and approval changes
  for later slices.
- **Ambiguous command patterns:** Compile legacy `Bash(...)` into argv-token
  rules only. Do not add raw shell authorization.
- **Over-modeling:** Keep path, MCP wildcard, hooks, sandbox, and catalog
  changes out of this slice except for fields needed by the shared schema.
- **Trace leakage:** Explain payloads must stay redacted and argument-free.
- **Pydantic compatibility:** Existing model construction patterns include
  `model_construct(...)` with `None` fields. Preserve safe defaults and avoid
  validators that break legacy tests.

## Completion Criteria

- New decision primitives are package-owned and exported from
  `mcp_unified.profiles`.
- Existing profile resolution behavior remains compatible.
- Denied, ask, and allowed decisions carry structured metadata.
- Legacy `allowed_tools`/`denied_tools` and `Bash(...)` policy strings compile
  into typed rules.
- A package-local explain helper returns redacted decision explanations.
- Targeted tests, Bandit, and diff hygiene pass and are recorded in `TASK-2308`.

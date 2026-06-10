# MCP Tool Permission Rule Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a package-level Claude-style permission rule parser/evaluator for MCP profiles.

**Architecture:** Keep the first slice inside `mcp_unified.profiles` so the standalone package owns the policy grammar without importing host runtime code. Parse `ToolName(specifier)` strings and structured rule documents into existing `PolicyDecisionRule` primitives, then evaluate them by subject type with the existing `deny > ask > allow` merge contract.

**Tech Stack:** Python 3.11, Pydantic models, pytest, existing `mcp_unified.profiles.decisions` primitives.

---

### Task 1: Parser And Evaluator Tests

**Files:**
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py`

- [x] **Step 1: Write failing parser tests**

Cover:
- `Bash(git *)` parses to a command rule with `argv=("git", "*")`.
- `Read(/docs/**)` and `Edit(src/*.py)` parse to path rules.
- `WebFetch(https://example.com/docs)` parses to a domain rule for `example.com`.
- `Skill(review)` and `Agent(backend-engineer)` parse to skill/agent rules.
- `mcp__github__*` parses to an MCP wildcard rule.
- malformed/empty specifiers fail closed.

- [x] **Step 2: Run parser tests and verify red**

Run:
`source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py -q`

Expected: fail because `mcp_unified.profiles.permission_rules` does not exist.

- [x] **Step 3: Write failing evaluator tests**

Cover:
- exact tool matching still works through the shared evaluator.
- command argv matching uses token semantics, so `["git", "status"]` matches `Bash(git *)` but `["git", "status", "--short"]` does not.
- deny overrides ask/allow across matching rules.
- path and domain wildcard matches are bounded and redacted in `PolicyMatchedRule`.
- invalid broad shell patterns such as `Bash(*)` are rejected.

- [x] **Step 4: Run evaluator tests and verify red**

Use the same focused pytest command. Expected: fail for missing implementation.

### Task 2: Package Parser And Rule Compilation

**Files:**
- Create: `mcp_unified/profiles/permission_rules.py`
- Modify: `mcp_unified/profiles/decisions.py`
- Modify: `mcp_unified/profiles/__init__.py`

- [x] **Step 1: Implement parser helpers**

Add:
- `parse_permission_rule(pattern, outcome="allow", source="permission_rules")`
- `compile_permission_rules(document, field_name="permission_rules")`
- subject classification for tool, command, path, domain, mcp, skill, and agent.

- [x] **Step 2: Extend `PolicyRuleType` safely**

Add `path`, `domain`, `skill`, and `agent` to `PolicyRuleType`. Preserve existing validation for command rules and required string pattern validation for non-command rules.

- [x] **Step 3: Compile profile `permission_rules`**

Update `compile_profile_policy_rules()` to include the profile-extra `permission_rules` field. Keep `compile_profile_tool_policy_rules()` unchanged except for exact tool behavior already present, so runtime tool decisions do not accidentally start honoring path/domain/command rules.

- [x] **Step 4: Run focused tests**

Run the new permission-rule tests and existing profile decision tests.

### Task 3: Generic Permission Evaluation

**Files:**
- Modify: `mcp_unified/profiles/permission_rules.py`
- Modify: `mcp_unified/profiles/__init__.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py`

- [x] **Step 1: Implement generic subject evaluation**

Add:
- `evaluate_permission_rule_decision(rules, subject_type, value, argv=None)`
- pattern matching for tool/mcp/path/domain/skill/agent subjects.
- token matching for command argv subjects.

- [x] **Step 2: Return existing policy decision models**

Use `PolicyDecision`, `PolicyDecisionSubject`, `PolicyMatchedRule`, and `merge_policy_decisions()` so explain/reporting/hook paths can consume the same metadata shape later.

- [x] **Step 3: Run focused tests**

Run:
`source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py -q`

### Task 4: Docs And Task Closeout

**Files:**
- Modify: `mcp_unified/USER_GUIDE.md`
- Modify: `backlog/tasks/task-2296 - Add-Claude-style-tool-permission-rule-parser-and-evaluator.md`

- [x] **Step 1: Document the first-slice grammar**

Add a concise user-guide section covering examples, supported subject families, and what remains deferred.

- [x] **Step 2: Update Backlog task metadata**

Record touched files, verification, skips/deferred runtime integrations, and final summary.

- [x] **Step 3: Final verification**

Run focused tests, package import smoke, Bandit on touched production paths, and `git diff --check`.

- [x] **Step 4: Commit**

Commit with a message such as:
`feat: add MCP tool permission rule parser`

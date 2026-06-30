# MCP Permission Runtime Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire compiled MCP profile `permission_rules` into standalone gateway tool-call execution as an additional runtime enforcement layer.

**Architecture:** Keep legacy `allowed_tools`/`denied_tools` and capability policy as the primary tool visibility and execution gate. After a backend tool is already allowed, evaluate matching `permission_rules` subjects extracted from the tool call and block matching `deny` or `ask` rules with redacted `GatewayPolicyDenied` payloads. Treat unmatched permission rules as no-op so path/domain/command rules do not accidentally grant tools.

**Tech Stack:** Python, Pydantic profile models, `mcp_unified.profiles.permission_rules`, standalone gateway runtime, pytest.

---

### Task 1: Add Red Tests For Runtime Permission Rules

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [x] **Step 1: Add failing tests**
  - Add a helper profile with `allowed_tools` plus extra `permission_rules`.
  - Add a test where `fs.read_text` is allowed by legacy policy but blocked by `Read(docs/private/**)` when the call arguments include `{"path": "docs/private/secret.txt"}`.
  - Add a test where a matching `ask` permission rule blocks with `status == "approval_required"`.
  - Add a test where a matching external MCP wildcard rule blocks an allowed `mcp__github__delete_repo` backend tool.

- [x] **Step 2: Run red tests**
  - Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q -k "permission_rule"`
  - Expected: new tests fail because runtime calls do not yet evaluate `permission_rules`.

### Task 2: Enforce Matching Permission Rules In Gateway Runtime

**Files:**
- Modify: `mcp_unified/gateway/profile_runtime.py`

- [x] **Step 1: Add subject extraction helpers**
  - Extract always-present `tool` subject from the backend tool name.
  - Extract `mcp` subject when the tool name starts with `mcp__`.
  - Extract common path subjects from argument keys such as `path`, `file_path`, `base_path`, `source_path`, and `destination_path`, plus list values such as `paths`.
  - Extract common domain subjects from keys such as `url`, `uri`, `domain`, `host`, and `urls`.
  - Extract command subjects from `command` string or `argv` list/tuple arguments.

- [x] **Step 2: Add enforcement helper**
  - Compile profile permission rules from `profile.policy_document`.
  - For each extracted subject, call `evaluate_permission_rule_decision()`.
  - Ignore unmatched default-deny decisions with no matched rules.
  - Raise `GatewayPolicyDenied(status="denied")` for matched `deny`.
  - Raise `GatewayPolicyDenied(status="approval_required")` for matched `ask` until approval prompts are integrated.
  - Include redacted provenance: `profile_id`, `tool_name`, `subject_type`, `reason_code`, and matched rule metadata. Do not include raw path, URL, command, or argument values.

- [x] **Step 3: Wire helper into `_call_backend_tool_through_policy()`**
  - Run it only after `_allowed_policy_result_for_tool()` returns `resolved`.
  - Keep existing legacy denials and backend metadata fallback behavior unchanged.
  - Pass the resolved effective policy to backend calls exactly as before when no permission-rule block occurs.

- [x] **Step 4: Run green tests**
  - Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q -k "permission_rule"`
  - Expected: new tests pass.

### Task 3: Validate Focused Profile/Gateway Behavior

**Files:**
- Modify: `backlog/tasks/task-2349 - Wire-MCP-permission-rules-into-runtime-tool-call-enforcement.md`

- [x] **Step 1: Run focused regression suite**
  - Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py -q`
  - Expected: pass.

- [x] **Step 2: Run quality gates**
  - Run Ruff on touched Python files.
  - Run compile smoke on touched package modules.
  - Run Bandit on touched package modules.
  - Run `git diff --check`.

- [x] **Step 3: Update Backlog task**
  - Record implementation notes, verification commands, known deferred items, final summary, and Definition of Done.

- [x] **Step 4: Commit**
  - Commit message: `feat: enforce MCP permission rules at gateway runtime`.

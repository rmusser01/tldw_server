# MCP File-Policy Audit Event Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend MCP tool-use events with safe filesystem policy decision metadata.

**Architecture:** Reuse the existing metadata-only tool-use reporting path. `prepare_tool_call()` already evaluates path scope and receives redacted path decisions from the path enforcement service; this slice carries those decisions into `ToolUseEvent` while sanitizing workspace-relative paths and avoiding raw content, diffs, receipts, absolute host paths, and capability tokens.

**Tech Stack:** Python 3.11, Pydantic models, pytest, existing MCP protocol/reporting modules.

---

### Task 1: Event Model Contract

**Files:**
- Modify: `mcp_unified/tool_use_reporting/models.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_models.py`

- [x] **Step 1: Write failing model tests**

Add tests proving file-policy decision metadata keeps workspace-relative paths, grant outcome/source, reason code, and redaction state, while rejecting absolute paths and unsafe extras.

- [x] **Step 2: Run model test red**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_models.py -q`

- [x] **Step 3: Implement model fields and sanitizers**

Add a bounded `FilePolicyDecisionMetadata` model and optional booleans for hash/lock-lease presence. Do not add query columns or store raw hashes/lock ids.

- [x] **Step 4: Run model tests green**

Run the same model test command and confirm all pass.

### Task 2: Protocol Capture

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py`

- [x] **Step 1: Write failing protocol tests**

Add a protocol-side test with an enabled path-scope policy and fake path enforcer returning redacted `path_decisions`. Assert the recorded event includes only sanitized decision metadata and presence booleans for hashes/lock leases.

- [x] **Step 2: Run protocol test red**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py -q`

- [x] **Step 3: Carry path-scope payload through prepared calls**

Add a `scope_payload` field to `PreparedToolCall`, pass it from `prepare_tool_call()`, and feed it into `_build_tool_use_event()` for success/cached/error recording. For prepare-time governance denials, extract `governance.path_scope` into the failure event.

- [x] **Step 4: Run protocol tests green**

Run the protocol test command and confirm all pass.

### Task 3: Focused Verification

**Files:**
- Modify: `backlog/tasks/task-2302 - Add-file-policy-audit-event-reporting.md`

- [x] **Step 1: Run focused reporting tests**

Run model, protocol, and store tests to ensure event serialization remains compatible.

- [x] **Step 2: Run Bandit on touched production Python files**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -q mcp_unified/tool_use_reporting/models.py mcp_unified/tool_use_reporting/__init__.py tldw_Server_API/app/core/MCP_unified/protocol.py`

Note: Bandit over pytest files was not used as the completion gate because it reports the project's normal `B101 assert_used` pytest noise; production touched files passed.

- [x] **Step 3: Update Backlog task**

Record touched files, verification results, skips/blockers, and final summary in `TASK-2302`.

# PR 2360 Review Rebase Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase PR 2360 onto the latest `dev` and resolve all actionable review comments without expanding the file-policy audit reporting scope.

**Architecture:** Keep the existing metadata-only MCP tool-use reporting path. Add narrow sanitizer and aggregation fixes where the review identified incorrect edge cases, then cover each behavior with focused model/protocol tests.

**Tech Stack:** Python 3.11, Pydantic models, pytest, Bandit, GitHub CLI.

---

### Task 1: Rebase and Tracking

**Files:**
- Modify: `backlog/tasks/task-2302 - Add-file-policy-audit-event-reporting.md`
- Create: `Docs/superpowers/plans/2026-06-16-pr2360-review-rebase-fixes.md`

- [x] **Step 1: Fetch latest base and PR branch**

Run: `git fetch origin dev codex/mcp-next-slice-20260614`

- [x] **Step 2: Rebase PR branch onto latest dev**

Run: `git rebase --autostash origin/dev`

- [x] **Step 3: Reopen Backlog task for PR review follow-up**

Use Backlog.md MCP to mark `TASK-2302` in progress and record the review comment ids.

### Task 2: Model Sanitizer Review Fix

**Files:**
- Modify: `mcp_unified/tool_use_reporting/models.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_models.py`

- [x] **Step 1: Add failing tests for URL-like path rejection before slash normalization**

Add examples for `https://host/path` and `file:/tmp/path` to prove URI-like values do not survive normalization.

- [x] **Step 2: Run model tests red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_models.py -q`

- [x] **Step 3: Move URI validation before slash collapsing**

Reject `://` and URI-scheme path patterns before `re.sub(r"/+", "/", text)` mutates scheme separators.

- [x] **Step 4: Run model tests green**

Run the same model test command.

### Task 3: Protocol Aggregation and Presence Review Fixes

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py`

- [x] **Step 1: Add failing tests for derived grant outcome precedence**

Assert that later `denied` decisions override earlier `allowed` decisions, `not_granted` outranks all-allowed decisions, and explicit evaluator metadata still wins.

- [x] **Step 2: Add failing tests for empty hash/lock containers**

Assert empty `expected_sha256_by_path` and `lock_lease_id_by_path` containers do not set presence booleans.

- [x] **Step 3: Add failing test for bounded decision extraction**

Assert `_tool_use_file_policy_decisions()` copies at most `MAX_FILE_POLICY_DECISIONS` dict entries.

- [x] **Step 4: Run protocol tests red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py -q`

- [x] **Step 5: Implement minimal protocol fixes**

Bound the copied decisions list, derive event-level grant outcome from all decisions with deny/not-granted precedence, and ignore empty containers for presence flags.

- [x] **Step 6: Run protocol tests green**

Run the protocol test command again.

### Task 4: Final Verification and PR Closeout

**Files:**
- Modify: `backlog/tasks/task-2302 - Add-file-policy-audit-event-reporting.md`

- [x] **Step 1: Run focused reporting tests**

Run model and protocol test modules.

- [x] **Step 2: Run Bandit on touched production files**

Run: `source .venv/bin/activate && python -m bandit -r mcp_unified/tool_use_reporting tldw_Server_API/app/core/MCP_unified/protocol.py -f json -o /tmp/bandit_pr2360.json`

- [ ] **Step 3: Review PR checks after pushing**

Run `gh pr checks 2360 --repo rmusser01/tldw_server` and inspect remaining failures.

- [ ] **Step 4: Update Backlog task and reply to review threads**

Record verification results, mark `TASK-2302` done, push the rebased branch, and reply to each fixed inline review comment.

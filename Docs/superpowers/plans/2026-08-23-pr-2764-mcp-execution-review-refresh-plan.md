# PR 2764 MCP Execution Review Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh PR #2764 on current `dev`, resolve validated review findings, and restore trustworthy MCP execution/security verification.

**Architecture:** Preserve the approved request-snapshot authorization and extracted idempotency design. Treat the Qodo comments as compliance-only candidates, resolve rebase conflicts using current `dev` interfaces, and make no execution behavior changes unless current tests prove a regression. The historical container gate failure is handled as an infrastructure cancellation unless a fresh build reproduces a code-caused failure.

**Tech Stack:** Python 3.14, FastAPI MCP Unified, pytest, Ruff, Bandit, GitHub Actions, Backlog.md

---

### Task 1: Rebase And Revalidate Review Context

**Files:**
- Modify as required by conflicts: `tldw_Server_API/app/core/MCP_unified/**`
- Modify as required by conflicts: `apps/mcp-unified/**`
- Modify: `backlog/tasks/task-2294.3.4 - Refresh-PR-2764-and-address-MCP-execution-review-findings.md`

- [ ] **Step 1: Confirm the worktree contains only TASK-2294.3.4 tracking and this plan.**

Run: `git status --short`

Expected: only the new Backlog task and this plan are untracked before rebase.

- [ ] **Step 2: Rebase onto current `origin/dev`.**

Run: `git rebase origin/dev`

Expected: rebase completes, or each conflict is resolved by preserving current `dev` contracts plus the PR's approved execution security properties.

- [ ] **Step 3: Confirm branch ancestry.**

Run: `git rev-list --left-right --count origin/dev...HEAD`

Expected: `0` behind.

### Task 2: Validate And Address Qodo Compliance Findings

**Files:**
- Modify if still applicable: `tldw_Server_API/app/core/MCP_unified/modules/base.py`
- Modify if still applicable: `tldw_Server_API/app/core/MCP_unified/tests/test_concurrency_and_breaker.py`
- Modify if still applicable: `tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py`

- [ ] **Step 1: Compare each comment with current repository conventions and static configuration.**

Run: `rg -n "AdmittedModuleOperation|pytestmark|mark\.unit|def test_per_module_concurrency" tldw_Server_API/app/core/MCP_unified pyproject.toml`

Expected: evidence for whether docstrings, annotations, and test markers are enforced in the rebased codebase.

- [ ] **Step 2: Add only validated docstrings and type annotations.**

Implementation: add concise contract docstrings to public execution methods and explicit `-> None`/callable annotations where current project rules require them. Do not alter runtime behavior.

- [ ] **Step 3: Apply the established module-level pytest marker pattern if required.**

Implementation: use the repository's current marker convention; retain `pytest.mark.asyncio` only when it is required by pytest-asyncio and is not treated as the test classification marker.

- [ ] **Step 4: Run focused review-scope tests and static checks.**

Run: `.venv/bin/python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_concurrency_and_breaker.py tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py`

Run: `.venv/bin/python -m ruff check <touched Python files>`

Expected: focused tests pass; no branch-added Ruff findings.

### Task 3: Reverify MCP Execution Security

**Files:**
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_prepared_execution_integrity.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_manager.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_outcomes.py`
- Test: remaining focused execution matrix recorded on TASK-2294.3.1

- [ ] **Step 1: Run the focused 12-file execution/security matrix.**

Expected: all collected tests pass after rebase.

- [ ] **Step 2: Run the full MCP Unified suite.**

Expected: no branch-introduced failures; any failures are compared against current `dev`, not the July baseline.

- [ ] **Step 3: Run package, syntax, whitespace, Ruff, and Bandit gates.**

Run: `.venv/bin/python -m compileall -q tldw_Server_API/app/core/MCP_unified apps/mcp-unified/src/mcp_unified`

Run: `git diff --check origin/dev...HEAD`

Run: `.venv/bin/python -m bandit -r tldw_Server_API/app/core/MCP_unified apps/mcp-unified/src/mcp_unified -x tldw_Server_API/app/core/MCP_unified/tests`

Expected: compile and whitespace pass; Bandit has no findings; Ruff has no branch-added findings.

### Task 4: Publish And Close Review Work

**Files:**
- Modify: `backlog/tasks/task-2294.3.4 - Refresh-PR-2764-and-address-MCP-execution-review-findings.md`
- Remove: `Docs/superpowers/plans/2026-08-23-pr-2764-mcp-execution-review-refresh-plan.md`

- [ ] **Step 1: Record validation, CI root cause, modified files, and verification in TASK-2294.3.4.**

- [ ] **Step 2: Remove this completed task-specific plan and commit the refresh.**

Run: `git status --short && git diff --check`

Expected: only intended task/review/rebase changes remain.

- [ ] **Step 3: Push with lease protection and reply in each inline review thread.**

Run: `git push --force-with-lease origin codex/mcp-skills-model-runner-design`

Expected: PR #2764 points at the verified rebased head and remains draft.

- [ ] **Step 4: Inspect the new GitHub checks.**

Expected: new jobs are queued/running or completed; any immediate failures are investigated from their logs before completion is claimed.

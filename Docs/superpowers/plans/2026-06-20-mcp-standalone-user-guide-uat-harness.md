# MCP Standalone User Guide UAT Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run a one-off UAT harness that validates the package-local MCP Unified standalone user guide from a new-user perspective.

**Architecture:** Keep the harness outside the packaged `mcp_unified` module so it does not become a public package surface. The harness creates isolated temporary workspaces, installs the package boundary the way the guide documents, executes guide commands through subprocess argv, writes temporary fixture servers for stdio/HTTP/WebSocket smoke transport UAT, and emits a redacted JSON report. Confirmed guide/package mismatches are fixed minimally.

**Tech Stack:** Python subprocess/tempfile/json/venv, pytest, Ruff, Bandit, `mcp_unified` package CLI.

---

### Task 1: Package Boundary Exposure Test

**Files:**
- Test: `tldw_Server_API/tests/Helper_Scripts/test_mcp_standalone_user_guide_uat.py`
- Modify: `mcp_unified/pyproject.toml`

- [x] **Step 1: Write the failing test**

Add a test that parses `mcp_unified/pyproject.toml` and asserts the standalone package exposes both documented console scripts: `mcp-unified-gateway` and `mcp-unified-smoke`, and includes `mcp_unified.smoke` in packaged modules.

- [x] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Helper_Scripts/test_mcp_standalone_user_guide_uat.py -q`

Expected: FAIL because `mcp-unified-smoke` and `mcp_unified.smoke` are missing from the package-local project metadata.

- [x] **Step 3: Fix package metadata**

Add the missing script and smoke package mapping to `mcp_unified/pyproject.toml`.

- [x] **Step 4: Run test to verify it passes**

Run the same pytest command and expect PASS.

### Task 2: One-Off UAT Harness

**Files:**
- Create: `Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py`
- Test: `tldw_Server_API/tests/Helper_Scripts/test_mcp_standalone_user_guide_uat.py`

- [x] **Step 1: Add failing harness contract tests**

Test redacted report helpers and that the harness command plan includes install, gateway CLI, smoke CLI, config/profile, external server, credential grant, snapshot, and reporting phases.

- [x] **Step 2: Run test to verify it fails**

Run the focused pytest command and expect failures for the missing harness module.

- [x] **Step 3: Implement minimal harness**

Implement an executable Python helper that:
- creates a temp UAT workspace
- optionally creates a temp venv
- installs `mcp_unified[gateway]` from the package-local path
- runs documented local CLI commands
- writes temp `gateway.json`, `search-server.json`, grant, snapshot, and policy args files
- runs smoke CLI in-process, stdio subprocess, live HTTP, and live WebSocket fixture paths when installed
- emits redacted JSON report to a requested path or stdout

- [x] **Step 4: Run tests to verify pass**

Run focused pytest and expect PASS.

### Task 3: Full UAT Run And Guide Fixes

**Files:**
- Modify: `mcp_unified/USER_GUIDE.md` if confirmed guide issues are found
- Modify: `mcp_unified/README.md` if confirmed package/script issues are found
- Modify: `backlog/tasks/task-2393 - Add-MCP-standalone-user-guide-UAT-harness.md`

- [x] **Step 1: Run the harness from a clean temp workspace**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py --repo-root . --json-report /tmp/mcp_standalone_user_guide_uat.json`

- [x] **Step 2: Investigate any failures**

Use systematic debugging: read stderr/stdout, reproduce the smallest failed command, identify root cause, then fix only confirmed product/docs/package mismatches.

- [x] **Step 3: Re-run harness**

Expected: report has no failed required local guide or fixture transport steps. Remote/runtime steps may be marked skipped when no live gateway URL/admin key is supplied.

### Task 4: Final Validation And Commit

**Files:**
- All touched files

- [x] **Step 1: Run compile checks**

Run py_compile on the helper script and touched package modules.

- [x] **Step 2: Run Ruff**

Run Ruff on touched Python files.

- [x] **Step 3: Run Bandit**

Run Bandit on the helper script and touched package runtime files.

- [x] **Step 4: Run `git diff --check`**

Expected: no whitespace errors.

- [x] **Step 5: Update Backlog and commit**

Record verification and UAT report summary in `TASK-2393`, then commit.

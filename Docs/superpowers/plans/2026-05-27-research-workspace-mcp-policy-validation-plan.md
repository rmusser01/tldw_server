# Research Workspace MCP Policy Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close RW-UAT-021 with live evidence that Research Workspace IDs can be bound into MCP Hub workspace sets and resolved through MCP Hub policy/tool availability paths.

**Architecture:** Keep MCP Hub as the canonical owner of workspace sets, path trust, and policy state. Research Workspace only supplies a canonical workspace ID and deep link context; validation exercises the backend APIs and MCP Hub UI without adding `/workspace-playground` aliases or duplicating MCP state in Research Workspace.

**Tech Stack:** Playwright E2E, FastAPI Workspaces API, MCP Hub management API, Research Workspace UAT matrix.

---

### Task 1: Live MCP Workspace-Set Policy E2E

**Files:**
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts`

- [x] **Step 1: Write the fixture-backed E2E**

Add a test that:
- creates a canonical workspace with `PUT /api/v1/workspaces/{id}`;
- creates an MCP Hub workspace-set object;
- adds the workspace ID as a set member;
- creates a user-scoped persona policy assignment using `workspace_source_mode: "named"` and the created workspace set;
- fetches `/api/v1/mcp/hub/effective-policy?persona_id=<test-persona>` and asserts selected named set metadata, workspace IDs, and allowed tool policy;
- optionally probes `/api/v1/mcp/tools/execute` and records whether this live run allows execution;
- opens `/mcp-hub?workflow=setup&view=workspace-sets&workspace_id=<id>&source=research-workspace` and asserts MCP Hub shows the Research Workspace context as included in an MCP workspace set;
- cleans up policy assignment, workspace-set member, workspace-set object, and workspace.

- [x] **Step 2: Verify E2E against the current implementation**

Run:
`TLDW_E2E_SERVER_URL=http://127.0.0.1:<api-port> TLDW_WEB_URL=http://127.0.0.1:<web-port> TLDW_E2E_API_KEY=<key> TLDW_WEB_AUTOSTART=false npx playwright test e2e/workflows/tier-2-features/mcp-hub.spec.ts --project=tier-2 --grep "binds a Research Workspace into an MCP workspace set" --reporter=line`

Expected: the test either passes with live API evidence or skips only when MCP Hub mutation APIs are unavailable.

Result: Passed against the live backend/WebUI on `127.0.0.1:18001` and
`127.0.0.1:18002` after tightening the tool assertion to require HTTP 2xx:
`1 passed (2.6s)`. A separate API-only confirmation returned HTTP 200 for
Research Workspace creation, HTTP 201 for Shared Workspace/workspace-set/member
/policy creation, HTTP 200 for effective policy, and HTTP 200 for the MCP
virtual CLI `run` tool execution probe.

### Task 2: UAT Matrix Evidence

**Files:**
- Modify: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`

- [x] **Step 1: Update RW-UAT-021 only to the level verified**

Record the concrete backend/UI evidence from the E2E run. Mark the row `Pass` only if the live run creates/binds the workspace set, resolves effective policy, and shows MCP Hub context in UI. Keep tool execution wording conditional if the live run only validates policy/tool availability rather than successful tool execution.

### Task 3: Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-478.27 - Validate-MCP-workspace-set-policy-and-tool-execution-for-Research-Workspace.md`

- [x] **Step 1: Record verification**

Update acceptance criteria, implementation notes, final summary, and known skips/blockers.

- [x] **Step 2: Run targeted verification**

Run focused Playwright, targeted TypeScript/type checks if touched code requires it, and Bandit only if Python code changes. If no Python code changes, record Bandit as not applicable.

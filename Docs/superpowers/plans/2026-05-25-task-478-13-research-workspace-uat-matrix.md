# Research Workspace UAT Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maintain a current live Research Workspace UAT matrix and regression gate so workspace functionality is verified against a real backend/WebUI instead of inferred from code review.

**Architecture:** Treat the matrix as the human-readable acceptance source and keep high-risk checks mirrored in focused Playwright tests. Use live CDP/Playwright probes for rows that require backend/WebUI integration, while extension rows remain blocked until TASK-478.12 has a current extension build.

**Tech Stack:** Markdown documentation, Backlog.md, Playwright E2E, Next.js WebUI, FastAPI backend.

---

### Task 1: Matrix Structure And Ownership

**Files:**
- Create or update: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`
- Modify: `backlog/tasks/task-478.13 - Gate-E-maintain-live-Research-Workspace-UAT-matrix-and-regression-coverage.md`

- [x] **Step 1: Define the current matrix scope**

Include rows for first-time flow, source acquisition, ingestion/status, selection, RAG, Studio, My Media, folders, annotations/source preview, settings/share, responsive layout, old-route removal, Shared Workspace/MCP/ACP/Sandbox capability surface, and extension handoff.

- [x] **Step 2: Seed each row with owner task and verification state**

Use `TASK-478.1` through `TASK-478.12` as row owners. Mark current known states as `Pass`, `Partial`, `Blocked`, or `Gap`, and include exact evidence links or commands.

- [x] **Step 3: Document update rules**

Add a short "How to update this matrix" section requiring live backend/WebUI validation for behavior claims and explicit blocked rows for extension, ACP, or sandbox gaps.

### Task 2: Regression Coverage For Route Replacement

**Files:**
- Inspect: `apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts`
- Potentially modify: `apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts`

- [x] **Step 1: Find existing route coverage**

Search for `/workspace-playground` coverage. If already covered by an executable E2E, reference it in the matrix; if absent, add a focused Playwright assertion.

- [x] **Step 2: Add or verify old-route no-redirect coverage**

The check should navigate or request `/workspace-playground`, assert non-redirect 404 behavior, then verify `/research-workspace` still loads.

- [x] **Step 3: Run the focused E2E or document why it is blocked**

Use the project Playwright command with a live backend/WebUI if available. If unrelated build/type errors block execution, record the exact blocker.

### Task 3: Live Backend/WebUI Matrix Probe

**Files:**
- Temporary script only: `/private/tmp/task47813-live-matrix.cjs`
- Update: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`

- [x] **Step 1: Start backend and WebUI**

Run backend in single-user mode and WebUI with `NEXT_PUBLIC_API_URL`/`NEXT_PUBLIC_X_API_KEY` pointed to that backend. Use alternate ports if occupied.

- [x] **Step 2: Probe core rows with CDP/Playwright**

Verify `/research-workspace` route loads, `/workspace-playground` is 404/no redirect, empty state shows contextual copy, model guard is visible when no model is selected, tour controls render a Joyride overlay, and console has no critical errors.

- [x] **Step 3: Record evidence**

Save screenshots under `/private/tmp` or `Docs/Reviews/assets` only when useful. Record DOM/API assertions in the matrix rather than relying on screenshot appearance alone.

### Task 4: Finalize Backlog And Verification

**Files:**
- Modify: `backlog/tasks/task-478.13 - Gate-E-maintain-live-Research-Workspace-UAT-matrix-and-regression-coverage.md`

- [x] **Step 1: Run focused verification**

Run Markdown/static checks if available and any focused Playwright or Vitest regression added for this task.

- [x] **Step 2: Update Backlog**

Record touched files, live validation results, known blocked rows, and Bandit applicability.

- [x] **Step 3: Commit scoped changes**

Stage only TASK-478.13 files and commit with a TASK-478.13 message. Leave unrelated dirty files unstaged.

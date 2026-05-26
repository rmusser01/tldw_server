# Browser Extension Workspace Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and, if needed, repair browser-extension capture handoff into canonical Research Workspace sources.

**Architecture:** Use the existing WXT extension build and Playwright persistent-context helpers to load the built Chrome extension. Validate the real server path first, then add focused regression coverage only for discovered gaps. Keep `/research-workspace` as the canonical destination and keep `/workspace-playground` removed.

**Tech Stack:** WXT browser extension, Playwright/CDP, FastAPI backend, Next/WebUI Research Workspace, Workspaces API, Web Clipper services.

---

### Task 1: Build And Baseline Discovery

**Files:**
- Inspect: `apps/extension/package.json`
- Inspect: `apps/extension/tests/e2e/utils/extension-build.ts`
- Inspect: `apps/extension/tests/e2e/research-workspace.real-backend.spec.ts`
- Modify only if needed: `backlog/tasks/task-478.12 - Gate-E-validate-browser-extension-handoff-into-canonical-workspaces.md`

- [x] **Step 1: Verify extension build availability**

Run from `apps/extension`:

```bash
bun run build:chrome
```

Expected: `apps/extension/.output/chrome-mv3` or `apps/extension/build/chrome-mv3` contains `manifest.json`, `background.js`, and `options.html` or `sidepanel.html`.

- [x] **Step 2: Run existing extension Research Workspace real-backend baseline**

Run with live backend env:

```bash
TLDW_E2E_SERVER_URL=http://127.0.0.1:18002 \
TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY \
bunx playwright test tests/e2e/research-workspace.real-backend.spec.ts --reporter=line
```

Expected: pass or fail with a specific handoff/runtime error. A skipped test because the extension is unbuildable is a blocker, not a pass.

### Task 2: Live Web Clipper Handoff Probe

**Files:**
- Inspect: `apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx`
- Inspect: `apps/packages/ui/src/services/web-clipper/save-runtime.ts`
- Inspect: `apps/packages/ui/src/services/tldw/domains/web-clipper.ts`
- Temporary script only: `/private/tmp/task47812-extension-handoff.cjs`

- [x] **Step 1: Start live backend and WebUI**

Use backend `127.0.0.1:18002` and WebUI `localhost:3000` unless occupied. Stop any processes started by this task before final response.

- [x] **Step 2: Launch built extension with seeded server config**

Use Playwright persistent context, the existing `launchWithBuiltExtension` helper, or equivalent CDP automation. Do not use Computer Control.

- [x] **Step 3: Capture a deterministic page through Web Clipper**

Open the extension sidepanel clipper, seed or capture a page with a unique token, choose Workspace destination, provide a canonical workspace ID, save, and record the web-clipper response.

- [x] **Step 4: Verify WebUI Research Workspace sees the source**

Result: live CDP validation proved extension save/open handoff into canonical
`#/research-workspace`, persisted the clip as a workspace note/placement, and
verified `/sources/status` is reachable. It also exposed a product gap: browser
clips do not yet enter the first-class workspace source ingestion/indexing
pipeline.

Open `/research-workspace` against the same backend, verify the captured source appears under the canonical workspace/source APIs, and inspect `/api/v1/workspaces/{workspace_id}/sources/status`.

### Task 3: Fix Or Add Regression Coverage

**Files:**
- Potentially modify: `apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx`
- Potentially modify: `apps/packages/ui/src/services/web-clipper/save-runtime.ts`
- Potentially modify: `apps/packages/ui/src/services/tldw/domains/web-clipper.ts`
- Potentially modify: `apps/extension/tests/e2e/research-workspace.real-backend.spec.ts`
- Potentially modify: `apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx`

- [x] **Step 1: If live handoff fails, identify the boundary**

Classify the failure as build, extension launch, connection config, clipper save payload, backend web-clipper API, workspace source creation/status, or WebUI visibility.

- [x] **Step 2: Write the smallest regression test for the failing boundary**

Use existing Vitest or Playwright helpers. The regression must assert canonical `workspace_id` and `/research-workspace` behavior, not old route labels.

- [x] **Step 3: Implement the minimal fix**

Keep changes local to the failing boundary. Do not introduce route aliases or redirects.

- [x] **Step 4: Re-run focused tests and live CDP probe**

Expected: regression passes and live handoff proves the captured source is visible with status.

### Task 4: Closeout

**Files:**
- Modify: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`
- Modify: `backlog/tasks/task-478.12 - Gate-E-validate-browser-extension-handoff-into-canonical-workspaces.md`
- Modify: this plan file

- [x] **Step 1: Update UAT matrix row RW-UAT-024**

Set Browser extension handoff to Pass, Partial, Blocked, or Gap based only on the live result.

- [x] **Step 2: Record verification in Backlog**

Include build result, live probe result, tests run, screenshots/log paths if any, Bandit applicability, and known blockers.

- [x] **Step 3: Commit and push scoped changes**

Stage only TASK-478.12 files plus any intentional fix files. Leave unrelated untracked watchlist template files untouched unless they become part of this task.

### Task 5: Approach A Continuation - Promote Browser Clips To Workspace Sources

**Goal:** Preserve canonical clip notes and workspace note placements while also making workspace-targeted browser clips first-class, Media DB-backed workspace sources.

**Files:**
- Modify: `tldw_Server_API/app/core/WebClipper/service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/web_clipper.py`
- Modify if useful: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Modify: `tldw_Server_API/tests/Notes_NEW/unit/test_web_clipper_service.py`
- Modify: `tldw_Server_API/tests/Notes_NEW/integration/test_web_clipper_api.py`
- Modify: `apps/extension/tests/e2e/research-workspace.real-backend.spec.ts`
- Modify: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`

- [x] **Step 1: Write failing source-promotion tests**

Add focused assertions proving a workspace-targeted Web Clipper save creates a stable Media DB record and a stable `workspace_sources` row. Cover idempotent retry behavior before implementation.

- [x] **Step 2: Persist clip content into Media DB**

When a clip targets a workspace and Media DB is available, create or update a text/article media record using the clip id as the idempotency source hash. Preserve existing note behavior when Media DB is unavailable by returning a warning rather than dropping the note placement.

- [x] **Step 3: Create or reuse the workspace source row**

Create a deterministic `workspace_sources` row that points at the Media DB item. Repeated saves for the same clip and workspace must reuse the same source identity and media id.

- [x] **Step 4: Enqueue workspace source lifecycle status**

After the source row exists, enqueue the existing `workspace_source_ingest` job with the same idempotency key semantics used by `POST /workspaces/{workspace_id}/sources`.

- [x] **Step 5: Verify API, status projection, and live extension handoff**

Run focused backend tests, update the extension real-backend assertion to require the promoted source in `/sources/status`, and re-run the CDP-backed extension walkthrough against the live backend/WebUI.

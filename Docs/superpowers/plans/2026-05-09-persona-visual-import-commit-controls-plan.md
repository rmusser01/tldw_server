# Persona Visual Import Commit Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the existing Persona Visuals import-commit backend flow in the Persona Garden Visuals editor.

**Architecture:** Keep the work frontend-scoped and reuse the existing PR #1135-aligned backend/API contract. Add typed service helpers for commit start/status, then surface commit and refresh controls inside the existing `VisualPackEditor` portability panel without auto-activating imported packs.

**Tech Stack:** React, TypeScript, Ant Design, Vitest, Testing Library, existing `tldwClient.fetchWithAuth`.

---

### Task 1: Add Failing Frontend Coverage

**Files:**
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

- [x] **Step 1: Extend the import-preview test**

Add assertions that a completed import preview exposes `persona-visual-import-commit-button`, clicking it POSTs to `/api/v1/persona/profiles/persona-1/visual-packs/import-previews/preview-1/commit`, and the request body uses `trust_mode: "untrusted_import"` and `target_mode: "create_new"`.

- [x] **Step 2: Add commit status assertions**

Mock `GET /api/v1/persona/profiles/persona-1/visual-packs/imports/import-job-1` and assert the editor renders `persona-visual-import-commit-status`, stage, and job id, with a refresh button that updates completed state.

- [x] **Step 3: Verify RED**

Run: `cd apps/packages/ui && bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

Expected: FAIL because commit controls/service helpers do not exist in the editor yet.

### Task 2: Add Service Types and Helpers

**Files:**
- Modify: `apps/packages/ui/src/types/persona-visuals.ts`
- Modify: `apps/packages/ui/src/services/persona-visuals.ts`

- [x] **Step 1: Add import commit types**

Add `PersonaVisualImportCommitRequest` and `PersonaVisualImportCommitStartResponse` matching the backend schema. Reuse `PersonaVisualPortabilityJobResponse` for status polling.

- [x] **Step 2: Add service helpers**

Add `startPersonaVisualImportCommit(personaId, previewId, payload)` and `getPersonaVisualImportCommitStatus(personaId, jobId)`.

- [x] **Step 3: Keep endpoint shape exact**

POST path: `/visual-packs/import-previews/{preview_id}/commit`.

GET path: `/visual-packs/imports/{job_id}`.

### Task 3: Surface Commit Controls in VisualPackEditor

**Files:**
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`

- [x] **Step 1: Add local commit job state**

Track `importCommitJob`, `committingImport`, and `refreshingImportCommit`. Reset commit job state when persona/pack changes or a new preview starts.

- [x] **Step 2: Add handlers**

Add `handleStartImportCommit` using default payload `{ trust_mode: "untrusted_import", target_mode: "create_new" }`, and `handleRefreshImportCommit` using the commit job id.

- [x] **Step 3: Gate commit action**

Show and enable commit only when `fullImportPreview?.status === "completed"`. Keep the preview review-only until the user explicitly clicks commit.

- [x] **Step 4: Refresh packs after completion**

When a refreshed import-commit status is completed and includes `pack_id`, call `loadPacks()` so the new draft appears. Do not call activate.

### Task 4: Verify and Close Out

**Files:**
- Modify: `backlog/tasks/task-126.8 - Expose-persona-visual-import-commit-controls-in-editor.md`

- [x] **Step 1: Run focused tests**

Run: `cd apps/packages/ui && bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

Expected: PASS.

- [x] **Step 2: Run related Persona Visuals/Buddy tests**

Run: `cd apps/packages/ui && bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/utils/__tests__/persona-garden-route.test.ts`

Expected: PASS.

- [x] **Step 3: Run hygiene checks**

Run: `git diff --check`

Expected: no output, exit 0.

- [x] **Step 4: Update Backlog and commit**

Record RED/GREEN verification, mark acceptance criteria complete, then commit and open a PR for issue #1422.

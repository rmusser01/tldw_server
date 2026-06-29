# Skills Export Metadata Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve Skills export filename metadata through the frontend client and show clear export feedback.

**Architecture:** Keep the backend unchanged because it already sends `Content-Disposition`. Update the frontend Skills API helper to return `{ blob, filename }`, then use that filename in the Skills manager download flow and success notification. Add focused Vitest coverage for the client contract and UI feedback.

**Tech Stack:** React, TypeScript, Ant Design notification wrapper, TanStack Query, Vitest, Testing Library, existing `bgRequest` response wrapper.

---

### Task 1: Add Frontend Client Contract Tests

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.skills.test.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`

- [ ] **Step 1: Write failing tests for export filename metadata**
  - Add a test where `bgRequest` returns an arrayBuffer response with
    `content-disposition: attachment; filename="server-skill.zip"`.
  - Assert `workspaceApiMethods.exportSkill()` returns a `Blob` and
    `filename: "server-skill.zip"`.

- [ ] **Step 2: Write failing tests for safe fallback**
  - Add a test where `Content-Disposition` is missing or path-like.
  - Assert fallback is `<skill-name>.zip`.

- [ ] **Step 3: Run focused service tests and verify RED**
  - Run: `bunx vitest run src/services/tldw/domains/__tests__/workspace-api.skills.test.ts --reporter=dot`
  - Expected: export metadata tests fail because `exportSkill()` currently returns only `Blob`.

- [ ] **Step 4: Implement minimal client contract**
  - Change `exportSkill()` to call `bgRequest(..., returnResponse: true)`.
  - Parse `filename*` and `filename` from response headers.
  - Build `{ blob, filename }` with safe fallback.

- [ ] **Step 5: Run focused service tests and verify GREEN**
  - Run: `bunx vitest run src/services/tldw/domains/__tests__/workspace-api.skills.test.ts --reporter=dot`
  - Expected: tests pass.

### Task 2: Add Skills Manager Export Feedback Tests

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Skills/Manager.tsx`

- [ ] **Step 1: Write failing UI test for metadata filename and success feedback**
  - Mock `tldwClient.exportSkill()` to resolve `{ blob, filename: "server-skill.zip" }`.
  - Mock `URL.createObjectURL`, `URL.revokeObjectURL`, and anchor click.
  - Click the row Export action.
  - Assert anchor download uses `server-skill.zip`.
  - Assert success notification names `server-skill.zip`.

- [ ] **Step 2: Write failing or confirming UI test for sanitized failure feedback**
  - Mock `exportSkill()` rejection with sensitive URL/path/token content.
  - Assert error notification remains sanitized.

- [ ] **Step 3: Run focused manager tests and verify RED**
  - Run: `bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot`
  - Expected: success-feedback test fails because no success notification exists and current handler expects `Blob`.

- [ ] **Step 4: Implement minimal manager update**
  - Destructure `{ blob, filename }` from `exportSkill()`.
  - Use `filename` for `a.download`.
  - Show success notification after the click is triggered.

- [ ] **Step 5: Run focused manager tests and verify GREEN**
  - Run: `bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot`
  - Expected: tests pass.

### Task 3: Verify, Document, And Commit

**Files:**
- Modify: `backlog/tasks/task-530.12 - Implement-Skills-export-metadata-feedback.md`

- [ ] **Step 1: Run focused frontend verification**
  - Run: `bunx vitest run src/services/tldw/domains/__tests__/workspace-api.skills.test.ts src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot`
  - Expected: both files pass.

- [ ] **Step 2: Run static cleanup checks**
  - Run: `git diff --check`
  - Expected: no output.

- [ ] **Step 3: Update Backlog task**
  - Record implementation notes, modified files, verification, known skips, and final summary.

- [ ] **Step 4: Commit**
  - Stage the task, spec, plan, client, manager, and tests.
  - Commit message: `TASK-530.12 add skills export metadata feedback`

# Workspace Playground Media Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add filtering and sorting to Workspace Playground Add Sources -> My Media and address the current PR #1819 review threads.

**Architecture:** Keep the feature inside the existing AddSourceModal My Media tab. Reuse the existing `/api/v1/media/search` request shape for query, media type, keyword, and sort filters, and keep the default empty-filter path on `/api/v1/media`. Apply review fixes in the shared model selector and ChatPane without changing public behavior.

**Tech Stack:** React, Ant Design, Vitest, Testing Library, Backlog.md.

---

### Task 1: My Media Filter Tests

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/AddSourceModal.stage2.intake.test.tsx`

- [x] Add a failing test proving My Media sends `searchMedia` with `query`, `fields`, `media_types`, `must_have`, `sort_by`, and pagination when filters are active.
- [x] Add a failing test proving Clear filters resets the filter UI and returns to default `listMedia`.
- [x] Add or update the load-error test to assert the caught error is logged.
- [x] Run `bun run test src/components/Option/WorkspacePlayground/__tests__/AddSourceModal.stage2.intake.test.tsx --reporter=dot` and confirm the new tests fail for missing controls/behavior.

### Task 2: Review Thread Tests

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/ChatPane.stage2.test.tsx`

- [x] Reused existing ChatPane model-selector coverage for dropdown open, search, favorites, settings fallback, and model selection.
- [x] Keep existing settings navigation coverage passing after switching ChatPane to `useNavigate`.
- [x] Run `bun run test src/components/Option/WorkspacePlayground/__tests__/ChatPane.stage2.test.tsx --reporter=dot`.

### Task 3: My Media Filter Implementation

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/AddSourceModal.tsx`

- [x] Introduce typed media-library item helpers to replace local `any` usage in the My Media tab.
- [x] Add content type, keyword, and sort controls.
- [x] Route active filters through `tldwClient.searchMedia`; route empty filters through `tldwClient.listMedia`.
- [x] Reset selected media and pagination when filters change.
- [x] Render item keywords when present and provide Clear filters.
- [x] Log media load failures before showing the user-facing error.

### Task 4: PR Review Fix Implementation

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/ChatModelSelectorDropdown.tsx`
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx`

- [x] Type `modelDropdownMenuItems` as `MenuProps["items"]`.
- [x] Remove redundant model selector button `onClick` and the now-unused `openDropdown` callback.
- [x] Add a typed chat model record for `composerModels`.
- [x] Replace manual history mutation with `useNavigate`.

### Task 5: Verification and PR Update

**Files:**
- Modify: `backlog/tasks/task-419 - Enhance-Workspace-Playground-Add-Sources-media-filtering.md`

- [x] Run focused Workspace Playground tests.
- [x] Run `git diff --check`.
- [x] Update TASK-419 with touched files, verification, and final summary.
- [ ] Commit and push the follow-up.
- [ ] Reply to all addressed inline PR review threads and add a top-level PR update comment.

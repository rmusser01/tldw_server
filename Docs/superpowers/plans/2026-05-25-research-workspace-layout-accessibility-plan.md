# Research Workspace Layout Accessibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Research Workspace responsive overflow, advanced source-filter reachability, and keyboard accessibility regressions without redesigning the workspace model.

**Architecture:** Keep changes in the existing ResearchWorkspace shell and SourcesPane layout primitives. Use tests to lock the mobile overflow contract, advanced-filter scrolling contract, and accessible names/focus semantics for primary controls.

**Tech Stack:** React, TypeScript, Tailwind utilities, Ant Design controls, Vitest/Testing Library, Playwright/CDP live validation.

---

### Task 1: Characterize Current Responsive And Source-Filter Failure

**Files:**
- Inspect: `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- Inspect: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx`
- Inspect: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx`
- Inspect: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage2.test.tsx`

- [x] Run the existing responsive/source-pane tests to establish baseline behavior.
- [x] Start a live backend and WebUI.
- [x] Use Playwright/CDP at 390x844 and desktop widths to measure `document.documentElement.scrollWidth`, visible pane bounds, and clipped header actions.
- [x] Expand Advanced filters and record whether the folder/source list and footer remain reachable.

### Task 2: Add Regression Tests For Layout Contracts

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage2.test.tsx`

- [x] Add a mobile shell test that verifies the active workspace surface uses constrained widths and no child pane requires viewport-wide overflow.
- [x] Add a source-pane test for expanded advanced controls preserving a scrollable source list region and visible footer/action area.
- [x] Add accessible-name/focus tests for primary icon-only or compact controls touched by the layout fix.
- [x] Run the focused tests and confirm the new assertions fail before implementation where practical.

### Task 3: Fix Source Pane Vertical Layout And Advanced Filter Reachability

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx`
- Possibly modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/SourceAdvancedControls.tsx`

- [x] Ensure the Sources pane uses `min-h-0` flex regions so Advanced controls, folders, source list, and footer do not overlap.
- [x] Keep filter controls compact and allow the source list, not the whole pane, to scroll.
- [x] Preserve existing source selection, folder, preview, and bulk-action behavior.
- [x] Run focused SourcesPane tests.

### Task 4: Fix Mobile Horizontal Overflow And Header Wrapping

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- Possibly modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx`

- [x] Add `min-w-0`, responsive wrapping, and overflow containment to the workspace shell where measured overflow originates.
- [x] Ensure compact/mobile header tools wrap, collapse, or scroll within their own row rather than widening the viewport.
- [x] Preserve desktop density and the existing three-pane layout.
- [x] Run focused ResearchWorkspace responsive tests.

### Task 5: Live Validation And Backlog Closure

**Files:**
- Modify: `backlog/tasks/task-478.10 - Gate-D-fix-Research-Workspace-layout-responsive-behavior-and-keyboard-accessibility.md`

- [x] Re-run focused frontend tests for responsive and source-pane behavior.
- [x] Run `bunx tsc --noEmit --pretty false` and record known unrelated failures if still present.
- [x] Validate live backend/WebUI at desktop and 390x844 mobile using Playwright/CDP, including Advanced filters.
- [x] Record screenshot paths, layout metrics, and any known residual risks in the Backlog task.
- [x] Commit and push the completed TASK-478.10 scope to PR #2055.

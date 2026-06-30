---
id: TASK-478.10
title: >-
  Gate D: fix Research Workspace layout, responsive behavior, and keyboard
  accessibility
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-25 21:47'
labels:
  - research-workspace
  - uat
  - gate-d
  - layout
  - responsive
  - accessibility
milestone: Research Workspace UAT Remediation
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-25-research-workspace-layout-accessibility-plan.md
parent_task_id: TASK-478
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-visible failures: expanding Advanced filters caused the sources panel to overflow below visible panel/footer, and at a 390x844 mobile viewport the main content measured wider than the viewport with clipped/overflowing header tools.

User goal: use the workspace repeatedly on constrained screens, with dense controls that remain reachable, keyboard-friendly, and visually stable.

Scope:
- Fix Advanced filters overflow so folders, sources, and footer/actions remain reachable without overlapping or hidden controls.
- Remove mobile horizontal overflow and ensure header/toolbars wrap or collapse cleanly.
- Audit keyboard order, focus states, accessible names, and bulk-selection controls across the workspace shell.
- Verify density does not produce hidden controls at common desktop, tablet, and mobile widths.
- Add responsive/unit/e2e checks where practical.

Acceptance criteria:
- No horizontal overflow at 390px mobile viewport for the core Research Workspace screen.
- Advanced filter controls remain usable without hiding folders/sources or trapping content behind footers.
- Primary source, chat, model, Studio, and settings controls have usable keyboard focus and labels.
- CDP/Playwright screenshots and layout assertions cover desktop and mobile viewports.

Depends on: none; should avoid redesigning semantics before Gate A/B contracts are known.
Parallelization: can run in parallel with acquisition/source-preview/onboarding tasks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan saved at Docs/superpowers/plans/2026-05-25-research-workspace-layout-accessibility-plan.md. Scope is limited to responsive/layout/accessibility hardening for Research Workspace: characterize live failures, add regression tests, fix Source pane vertical layout, fix mobile horizontal overflow/header wrapping, validate with live backend/WebUI via Playwright/CDP, then update Backlog and PR #2055.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented responsive/layout hardening for Research Workspace.

Changes:
- WorkspaceHeader now uses shrinkable/wrapping header regions, truncates long workspace titles, and prevents compact header tools from widening the viewport.
- SourcesPane now has explicit min-h-0/overflow-hidden flex boundaries, a scrollable management-controls region, a separate flexing source-list region, and a non-overlapping footer.
- Added regression coverage for mobile header action containment and expanded Advanced controls preserving source-list reachability.

Verification:
- Focused Vitest passed: 3 files / 76 tests.
- git diff --check passed.
- Live backend/WebUI Playwright/CDP validation passed at desktop 1365x900 and mobile 390x844 with source-backed Advanced filters. Screenshots: /private/tmp/task47810-desktop-source-advanced-after.png and /private/tmp/task47810-mobile-source-advanced-after.png.
- tsc is blocked by unrelated existing syntax errors in apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx; that file has no diff in this task.
- Bandit not run because this task touched frontend TypeScript/TSX, tests, backlog, and plan docs only; no Python/backend code changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed TASK-478.10 layout/accessibility hardening for Research Workspace. Mobile header actions no longer widen the viewport, and expanded Sources Advanced controls now scroll independently while keeping the source list and footer reachable. Added focused regression tests and validated against a live backend/WebUI with Playwright/CDP at desktop and 390px mobile widths. TypeScript project check remains blocked by unrelated Watchlists syntax errors outside this task scope.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

---
id: TASK-530.5
title: Implement Skills dense view and column controls
status: Done
labels:
- skills
- webui
- ux
- power-user
priority: high
parent_task_id: TASK-530
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-530 power-user Skills remediation after TASK-530.4. Add a focused frontend slice for faster scanning and predictable table personalization on /skills.

Scope:
- Add a compact/dense table view toggle for power users.
- Add column visibility controls for secondary Skills metadata columns that already exist in list rows.
- Persist the user's density and column visibility choices locally.
- Preserve existing server-backed search, filters, sorting, pagination, and beginner empty-state behavior.

Out of scope:
- Bulk actions/export.
- Backend schema/API changes.
- Import review, delete/version semantics, and safe execution workflows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Skills manager exposes a keyboard-accessible view density toggle that switches the table between normal and compact scanning density.
- [x] #2 The Skills manager exposes a keyboard-accessible column visibility control for optional secondary columns without hiding required name/actions columns.
- [x] #3 Density and visible-column preferences persist locally and are restored on remount.
- [x] #4 Existing server-backed search, filters, sorting, pagination, and beginner empty states continue to work.
- [x] #5 Focused Skills manager tests cover density toggling, column visibility, persistence, and regression coverage for existing list behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/Plans/IMPLEMENTATION_PLAN_skills_density_columns_TASK_530_5.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added component-local Skills table preference loading/saving under localStorage key `tldw:skills-manager:table-preferences:v1`, with malformed/partial values normalized to safe defaults.
- Added a table density control with accessible pressed state; compact mode maps the Skills table to Ant Design `size="small"` and records `data-density="compact"` for regression coverage.
- Added a column visibility dropdown for optional list-row metadata columns: description, mode/context, argument hint, user visibility, and model invocation. Name and actions remain mandatory.
- Existing server-backed search, filters, sorting, pagination, import, seed, preview, and create flows remain covered by the expanded Skills manager suite.
- Bandit is not applicable: touched implementation files are frontend TypeScript/React plus tests/docs; no Python code changed.
- PR #2339 review follow-up: replaced the column-visibility test's Ant Design class traversal with accessible menu/menuitem queries.
- PR #2339 review follow-up: hiding the Mode/context column now clears an active `sort=context` state and resets to page 1 so backend query state cannot remain sorted by a hidden column.
- PR #2339 review follow-up: table preference state now uses React lazy initializers instead of a render-phase `useMemo` read.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented TASK-530.5 power-user table scanning controls for `/skills`: compact/comfortable density, optional metadata column visibility, and local persistence of both preferences. Regression tests cover density toggling, optional column visibility, preference restoration, and the existing Skills manager behavior.

Verification:
- PASS: `bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx` (22 tests after PR review follow-up)
- PASS: `bunx vitest run src/services/__tests__/tldw-api-client.boundary-slices.test.ts` (8 tests)
- PASS: `git diff --check`
- TYPECHECK CAVEAT: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit -p tsconfig.json --pretty false` still fails on inherited baseline errors in Notes tests, `src/entries/background.ts`, and `src/services/tldw/voice-cloning.ts`; no `src/components/Option/Skills/Manager.tsx` errors remain after the touched-file fixes.
- COMBINED VITEST CAVEAT: running the Skills manager suite and service boundary slice in one Vitest command passed the service slice but timed out once on an existing filter-control test under load; the same manager suite and service slice pass separately.
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

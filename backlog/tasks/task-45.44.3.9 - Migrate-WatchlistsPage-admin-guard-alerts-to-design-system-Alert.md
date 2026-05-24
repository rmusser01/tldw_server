---
id: TASK-45.44.3.9
title: Migrate WatchlistsPage admin guard alerts to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-24 01:37'
labels:
  - design-system
  - webui
  - watchlists
  - product-state
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2019'
parent_task_id: TASK-45.44.3
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the Admin and health expansion product-state migration by replacing the Admin WatchlistsPage AntD admin-guard Alert product-state callouts with the shared design-system Alert primitive, preserving forbidden/not-found copy, removing migrated baseline exceptions, and recording focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WatchlistsPage forbidden and not-found admin guard callouts render via design-system Alert.
- [x] #2 The WatchlistsPage Alert baseline exceptions are removed from design-system-product-state-baseline.json.
- [x] #3 Focused component/guard tests assert the design-system Alert marker for both guard states.
- [x] #4 Design-system product-state verification passes or records existing unrelated blockers.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused guard tests that drive forbidden and not-found API failures and assert the guard copy is inside data-ds-component="Alert".
2. Migrate WatchlistsPage guard Alert usage from AntD Alert props to the design-system Alert primitive while preserving existing copy and table/form behavior.
3. Remove WatchlistsPage Alert exceptions from the product-state baseline and run focused Vitest plus design-system verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented with a focused red/green guard test for both admin guard paths.

Verification:
- RED: `bunx vitest run src/components/Option/Admin/__tests__/WatchlistsPage.admin-guard.test.tsx --reporter=dot` failed before migration because the guard titles were not inside `[data-ds-component="Alert"]`.
- GREEN: `bunx vitest run src/components/Option/Admin/__tests__/WatchlistsPage.admin-guard.test.tsx --reporter=dot` passed, 2 tests.
- Product-state guard: `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed, 54 tests.
- Verifier: `bun run verify:design-system-state` passed with total baseline exceptions 254 and WatchlistsPage baseline rows 0.
- `git diff --check` passed.
- TypeScript: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still exits 2 on 347 existing diagnostics; no diagnostics mention WatchlistsPage, WatchlistsPage.admin-guard, the baseline, or TASK-45.44.3.9.
- Bandit skipped: UI-only TypeScript/test/baseline task; no Python touched.

PR review follow-up:
- Gemini requested wrapping the forbidden/not-found guard alerts in the same padded max-width container used by the normal admin page render. Verified current code returns bare Alert elements, so the layout feedback applies.

PR review follow-up verification:
- RED: `bunx vitest run src/components/Option/Admin/__tests__/WatchlistsPage.admin-guard.test.tsx --reporter=dot` failed after adding page-container assertions because guard alerts lacked the shared padded/max-width wrapper.
- GREEN: `bunx vitest run src/components/Option/Admin/__tests__/WatchlistsPage.admin-guard.test.tsx --reporter=dot` passed, 2 tests.
- Product-state guard: `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed, 54 tests.
- Verifier: `bun run verify:design-system-state` passed with total baseline exceptions still 254.
- `git diff --check` passed.
- TypeScript: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still exits 2 on 347 existing diagnostics; no diagnostics mention WatchlistsPage, WatchlistsPage.admin-guard, the baseline, or TASK-45.44.3.9.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the Admin WatchlistsPage forbidden and not-found guard callouts from AntD Alert to the shared design-system Alert primitive, added focused guard-state coverage for both paths, and removed the two WatchlistsPage Alert entries from the product-state baseline. The full design-system verifier passes with total baseline exceptions reduced to 254.

PR: https://github.com/rmusser01/tldw_server/pull/2019

PR review follow-up: wrapped both guard alerts in the same padded/max-width page container used by the normal WatchlistsPage render, and added focused assertions for that layout contract.
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

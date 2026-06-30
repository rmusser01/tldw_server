---
id: TASK-45.44.6.7
title: Migrate Evaluations settings alerts to design-system Alert
status: Done
assignee: []
created_date: '2026-05-30 16:13'
updated_date: '2026-05-30 16:18'
labels:
  - design-system
  - webui
  - product-state
  - settings
dependencies: []
references:
  - TASK-45.44.6
  - 'https://github.com/rmusser01/tldw_server/issues/1663'
  - apps/packages/ui/src/components/Option/Settings/evaluations.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.6
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the four Evaluations settings AntD Alert offline product-state callouts, plus the inline API test result alert in the same component, to the shared design-system Alert primitive while preserving setup/auth/offline/test-result copy and actions. Remove the matching baseline exceptions and verify the scoped Settings/account-security guard state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Evaluations settings no longer imports AntD Alert or renders AntD Alert product-state callouts.
- [x] #2 Representative auth, setup, unreachable, offline, and API test result guidance renders inside the design-system Alert container.
- [x] #3 Evaluations settings product-state baseline exceptions are removed and the scoped product-state guard is clean.
- [x] #4 Verification is recorded, including any unrelated baseline guard blockers.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect Evaluations settings alert branches and existing tests to identify focused render assertions.
2. Add failing tests that representative Evaluations settings warning and test-result copy renders inside the design-system Alert container.
3. Replace AntD Alert with the shared Alert primitive while preserving title/body copy, variants, and navigation actions.
4. Remove the four matching Evaluations settings baseline entries and run focused tests, scoped product-state guard, TypeScript, and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation notes:
- Replaced Evaluations settings AntD Alert usage with the shared design-system Alert primitive for auth-required, setup-required, unreachable, generic offline, and API test-result states.
- Kept the existing navigation targets: /settings/tldw, /, and /settings/health.
- Added render assertions that each representative message is inside data-ds-component="Alert" and that API test success remains visible.
- Removed the four Evaluations settings baseline exceptions.

Verification:
- RED: bun run test src/components/Option/Settings/__tests__/evaluations.connection.test.tsx --maxWorkers=1 --no-file-parallelism --testTimeout=30000 failed 5/6 because the existing AntD alerts had no design-system Alert ancestor.
- GREEN: bun run test src/components/Option/Settings/__tests__/evaluations.connection.test.tsx --maxWorkers=1 --no-file-parallelism --testTimeout=30000 passed 6/6.
- Scoped guard: node --input-type=module -e "...runGuardOnSources...evaluations.tsx..." reported: No product-state guard issues found.
- Baseline count: evaluations.tsx exceptions 0; Settings path exceptions 17; total baseline exceptions 161.
- Full guard: bun run verify:design-system-state still exits 1 on unrelated blocked findings in WritingPlayground, Notes, and ResearchWorkspace; no Evaluations settings finding remains.
- TypeScript: env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false passed.
- Whitespace: git diff --check passed.
- Bandit skipped: touched files are frontend TS/TSX, JSON baseline, and Backlog metadata only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated Evaluations settings product-state alerts to the shared design-system Alert primitive, added focused coverage for warning/test-result states, and removed the matching Evaluations settings baseline exceptions.
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

---
id: TASK-45.44.3.5
title: Migrate VisualComposerPane alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- extension
- product-state
- watchlists
priority: medium
parent_task_id: TASK-45.44.3
references:
- https://github.com/rmusser01/tldw_server/issues/1660
- apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/VisualComposerPane.tsx
- apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/__tests__/VisualComposerPane.section-generation.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-45.44.3 by replacing Watchlists VisualComposerPane AntD Alert product-state callouts with the shared design-system Alert primitive, preserving empty-state, generation error, and generation warning copy while removing migrated baseline exceptions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 VisualComposerPane empty-state, section-generation error, and section-generation warning callouts render through the shared design-system Alert primitive instead of AntD Alert.
- [x] #2 Focused VisualComposerPane coverage proves the migrated callouts preserve user-facing copy and expose canonical Alert markers.
- [x] #3 Migrated VisualComposerPane Alert baseline exceptions are removed without introducing new product-state verifier findings.
- [x] #4 Focused tests, design-system verifier, diff check, and Bandit applicability are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused VisualComposerPane tests requiring design-system Alert markers around empty/error/warning callouts and observe the expected failure against AntD Alert.
2. Replace VisualComposerPane AntD Alert imports/usages with the shared Alert primitive while preserving Select/Button/Input behavior.
3. Remove VisualComposerPane Alert entries from design-system-product-state-baseline.json.
4. Run focused tests, product-state verifier, git diff --check, and record Bandit as UI-only if no Python changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- TDD red: `bunx vitest run src/components/Option/Watchlists/TemplatesTab/__tests__/VisualComposerPane.section-generation.test.tsx --maxWorkers=1 --no-file-parallelism` failed on the missing `[data-ds-component="Alert"]` wrapper for the empty/error callouts.
- Replaced VisualComposerPane AntD Alert usage with `@/components/ui/primitives/Alert`, preserving the empty-state title/body and section-generation error/warning copy.
- Removed the three migrated VisualComposerPane Alert exceptions from `apps/packages/ui/scripts/design-system-product-state-baseline.json`.
- Verification: `bunx vitest run src/components/Option/Watchlists/TemplatesTab/__tests__/VisualComposerPane.section-generation.test.tsx --maxWorkers=1 --no-file-parallelism` -> 1 file, 4 tests passed.
- Verification: `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1 --no-file-parallelism` -> 1 file, 54 tests passed.
- Verification: `bun run verify:design-system-state` -> passed with 275 baseline exceptions total and Jobs/Scheduler/Watchlists at 24.
- Verification: `node -e "JSON.parse(require('fs').readFileSync('apps/packages/ui/scripts/design-system-product-state-baseline.json','utf8')); console.log('baseline ok')"` -> `baseline ok`.
- Verification: `git diff --check` -> passed.
- Bandit: skipped/not applicable; touched code is frontend TS/TSX plus JSON and Backlog task metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
VisualComposerPane now uses the shared design-system Alert primitive for its empty composer state and manual section-generation error/warning states. Focused tests cover the migrated callouts, the product-state baseline dropped the three VisualComposerPane Alert exceptions, and verification passed for the focused Watchlists test, product-state guard, design-system verifier, JSON parse, and diff hygiene.
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

---
id: TASK-45.44.3.4
title: Migrate TemplatePreviewPane alerts to design-system Alert
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
- apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/TemplatePreviewPane.tsx
- apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/__tests__/TemplatePreviewPane.live-preview.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/__tests__/TemplatePreviewPane.accessibility-baseline.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-45.44.3 by replacing Watchlists TemplatePreviewPane AntD Alert product-state callouts with the shared design-system Alert primitive, preserving preview error/info/warning copy and controls, removing migrated baseline exceptions, and recording focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TemplatePreviewPane warning, info, preview error, and flow error callouts render through the shared design-system Alert primitive instead of AntD Alert.
- [x] #2 Focused TemplatePreviewPane coverage proves the migrated callouts preserve user-facing copy and expose canonical Alert markers.
- [x] #3 Migrated TemplatePreviewPane Alert baseline exceptions are removed without introducing new product-state verifier findings.
- [x] #4 Focused tests, design-system verifier, locale/JSON hygiene where relevant, diff check, and Bandit applicability are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add or extend focused TemplatePreviewPane tests that require design-system Alert markers around warning/info/error callouts. 2. Replace TemplatePreviewPane AntD Alert imports/usages with the shared Alert primitive while preserving titles, descriptions, icons, loading, and preview controls. 3. Remove TemplatePreviewPane Alert entries from design-system-product-state-baseline.json. 4. Run focused tests, product-state verifier, git diff --check, and record Bandit as UI-only if no Python changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added focused TemplatePreviewPane regression coverage requiring the live-preview setup, no-runs, render-warning, render-error, and flow-error callouts to sit inside `[data-ds-component="Alert"]`.
- Replaced TemplatePreviewPane AntD Alert usage with `@/components/ui/primitives/Alert`, preserving warning/info/error copy and flow-check controls.
- Removed the five migrated TemplatePreviewPane Alert exceptions from `apps/packages/ui/scripts/design-system-product-state-baseline.json`.
- Verification: `bunx vitest run src/components/Option/Watchlists/TemplatesTab/__tests__/TemplatePreviewPane.live-preview.test.tsx src/components/Option/Watchlists/TemplatesTab/__tests__/TemplatePreviewPane.accessibility-baseline.test.tsx --maxWorkers=1 --no-file-parallelism` -> 2 files, 6 tests passed.
- Verification: `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1 --no-file-parallelism` -> 1 file, 54 tests passed.
- Verification: `bun run verify:design-system-state` -> passed with 280 baseline exceptions total and Jobs/Scheduler/Watchlists at 27.
- Verification: `git diff --check` -> passed.
- Bandit: skipped/not applicable; touched code is frontend TS/TSX plus JSON and Backlog task metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TemplatePreviewPane now uses the shared design-system Alert primitive for its preview/setup/error flow states. Focused tests cover the migrated callouts, the product-state baseline dropped the five TemplatePreviewPane Alert exceptions, and verification passed for the focused Watchlists tests, product-state guard, design-system verifier, and diff hygiene.
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

---
id: TASK-45.44.6.8
title: Migrate General settings alerts to design-system Alert
status: Done
assignee: []
created_date: '2026-05-30 16:21'
updated_date: '2026-05-30 16:24'
labels:
  - design-system
  - webui
  - product-state
  - settings
dependencies: []
references:
  - TASK-45.44.6
  - 'https://github.com/rmusser01/tldw_server/issues/1663'
  - apps/packages/ui/src/components/Option/Settings/general-settings.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.6
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the General settings browser-extension promotion and OCR asset state AntD Alert callouts to the shared design-system Alert primitive while preserving copy, external link behavior, and compact OCR guidance. Remove the matching general-settings baseline exceptions and verify the scoped Settings/account-security guard state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 General settings no longer imports AntD Alert or renders AntD Alert product-state callouts.
- [x] #2 Browser extension promotion and OCR disabled/enabled guidance render inside the design-system Alert container.
- [x] #3 General settings product-state baseline exceptions are removed and the scoped product-state guard is clean.
- [x] #4 Verification is recorded, including any unrelated baseline guard blockers.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect GeneralSettings alert branches and existing render tests.
2. Add failing tests that representative General settings promo and OCR guidance render inside the design-system Alert container.
3. Replace AntD Alert with the shared Alert primitive while preserving title/body copy, actions, and compact spacing.
4. Remove the three matching General settings baseline entries and run focused tests, scoped product-state guard, TypeScript, and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation notes:
- Replaced General settings AntD Alert usage with the shared design-system Alert primitive for the browser extension promotion and both OCR asset states.
- Preserved the external Learn More anchor and OCR compact guidance styling by sizing the OCR title node directly.
- Added render assertions that representative General settings messages are inside data-ds-component="Alert" and the external Learn More href remains unchanged.
- Removed the three General settings baseline exceptions.

Verification:
- RED: bun run test src/components/Option/Settings/__tests__/GeneralSettings.test.tsx --maxWorkers=1 --no-file-parallelism --testTimeout=30000 failed 2/3 because existing AntD alerts had no design-system Alert ancestor.
- GREEN: bun run test src/components/Option/Settings/__tests__/GeneralSettings.test.tsx --maxWorkers=1 --no-file-parallelism --testTimeout=30000 passed 3/3.
- Scoped guard: node --input-type=module -e "...runGuardOnSources...general-settings.tsx..." reported: No product-state guard issues found.
- Baseline count: general-settings.tsx exceptions 0; Settings path exceptions 14; total baseline exceptions 158.
- Full guard: bun run verify:design-system-state still exits 1 on unrelated blocked findings in WritingPlayground, Notes, and ResearchWorkspace; no General settings finding remains.
- TypeScript: env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false passed.
- Whitespace: git diff --check passed.
- Bandit skipped: touched files are frontend TS/TSX, JSON baseline, and Backlog metadata only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated General settings product-state alerts to the shared design-system Alert primitive, added focused coverage for extension/OCR guidance states, and removed the matching General settings baseline exceptions.
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

---
id: TASK-45.44.6.4
title: Migrate TldwBillingSettings alerts to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-30 07:40'
labels:
  - design-system
  - webui
  - product-state
  - settings
dependencies: []
references:
  - TASK-45.44.6
  - 'https://github.com/rmusser01/tldw_server/issues/1663'
  - apps/packages/ui/src/components/Option/Settings/TldwBillingSettings.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.6
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the eight TldwBillingSettings AntD Alert product-state callouts to the shared design-system Alert primitive while preserving existing billing error/warning copy and visibility behavior. Remove the matching baseline exceptions and verify the scoped Settings/account-security guard state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Billing load/account error states render with the shared design-system Alert primitive.
- [x] #2 Billing warning and usage-limit states render with the shared design-system Alert primitive.
- [x] #3 TldwBillingSettings has no remaining AntD Alert product-state baseline exceptions.
- [x] #4 Focused billing tests and scoped product-state guard verification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect TldwBillingSettings alert branches and existing settings tests to identify a focused render harness.
2. Add failing tests that representative billing error and warning copy renders inside the design-system Alert container.
3. Replace AntD Alert with the shared Alert primitive while preserving title/body copy and conditional rendering.
4. Remove the eight matching TldwBillingSettings baseline entries and run focused tests, scoped product-state guard, TypeScript, and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Corrected task id to TASK-45.44.6.4 because TASK-45.44.6.3 is already the completed IntegrationPolicyPanel slice.
- Replaced all eight TldwBillingSettings AntD Alert callouts with the shared design-system Alert primitive while preserving existing billing error, warning, usage-limit, and invoice copy.
- Added focused DS Alert regression coverage for billing load errors, subscription cancellation warnings, and usage limit states.
- Removed all eight TldwBillingSettings baseline entries; current baseline count for src/components/Option/Settings/TldwBillingSettings.tsx is 0.
- Settings-only product-state baseline count in this branch is now 39.
- Verification: initial focused billing test run failed against AntD Alert ancestors, then passed after migration.
- Verification passed: bun run test src/components/Option/Settings/__tests__/TldwBillingSettings.design-system-alert.test.tsx --maxWorkers=1 --no-file-parallelism.
- Verification passed: bun run test src/components/Option/Settings/__tests__/tldw-review-comments.test.tsx --maxWorkers=1 --no-file-parallelism.
- Verification passed: scoped product-state guard for TldwBillingSettings.tsx with baseline filtered to that path reported no product-state guard issues.
- Verification passed: env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false.
- Verification passed: git diff --check.
- Bandit was not run because the touched implementation scope is TypeScript/JSON/Backlog only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation complete locally. TldwBillingSettings no longer imports AntD Alert, all billing error/warning/usage callouts render through the shared DS Alert primitive, and the component-specific baseline exceptions were removed.
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

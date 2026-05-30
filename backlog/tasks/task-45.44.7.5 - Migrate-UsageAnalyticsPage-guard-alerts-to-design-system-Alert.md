---
id: TASK-45.44.7.5
title: Migrate UsageAnalyticsPage guard alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- product-state
priority: medium
parent_task_id: TASK-45.44.7
modified_files:
- apps/packages/ui/src/components/Option/Admin/UsageAnalyticsPage.tsx
- apps/packages/ui/src/components/Option/Admin/__tests__/UsageAnalyticsPage.design-system.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- backlog/tasks/task-45.44.7 - Migrate-design-system-product-state-Admin-and-health-expansion.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace UsageAnalyticsPage's forbidden and not-available admin guard AntD Alerts with the shared design-system Alert primitive while preserving copy and guard behavior. Remove matching product-state baseline entries and record focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 UsageAnalyticsPage forbidden guard feedback renders through data-ds-component="Alert" with error urgency semantics.
- [x] #2 UsageAnalyticsPage not-found guard feedback renders through data-ds-component="Alert" while preserving the Not Available copy.
- [x] #3 The matching UsageAnalyticsPage AntD Alert baseline entries are removed without introducing a UsageAnalyticsPage verifier finding.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the narrow UsageAnalyticsPage admin guard slice:

- Added focused jsdom regression coverage for forbidden and not-found guard states, with a hard null check before using the nearest `data-ds-component="Alert"` ancestor.
- Verified the test failed before implementation because the existing AntD Alert markup had no design-system Alert ancestor.
- Replaced the two guard-return AntD Alerts with the shared design-system Alert primitive while preserving the existing title/body copy and error/warning urgency.
- Removed the two UsageAnalyticsPage entries from the product-state baseline.

Verification:

- RED: `bunx vitest run src/components/Option/Admin/__tests__/UsageAnalyticsPage.design-system.test.tsx --reporter=dot` failed with `expected null not to be null` before the migration.
- GREEN: `bunx vitest run src/components/Option/Admin/__tests__/UsageAnalyticsPage.design-system.test.tsx --reporter=dot` passed with 2 tests.
- `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed with 54 tests.
- Baseline count script reported `{"total":187,"usage":0,"admin":29}`.
- `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` completed with no output.
- `git diff --check` completed with no output.
- `bun run verify:design-system-state` still exits 1 because of unrelated current-dev product-state drift and stale IntegrationPolicyPanel baseline rows; `/tmp/usage-analytics-design-state.log` has no UsageAnalyticsPage findings.
- Bandit not run: this slice touched TypeScript UI, JSON baseline data, and Backlog task markdown only.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated UsageAnalyticsPage forbidden and not-found admin guard feedback from AntD Alert to the design-system Alert primitive, added focused regression coverage for both states, and removed the two matching product-state baseline exceptions.

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

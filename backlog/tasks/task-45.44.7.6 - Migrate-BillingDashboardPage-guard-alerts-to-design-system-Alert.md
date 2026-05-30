---
id: TASK-45.44.7.6
title: Migrate BillingDashboardPage guard alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- product-state
priority: medium
parent_task_id: TASK-45.44.7
modified_files:
- apps/packages/ui/src/components/Option/Admin/BillingDashboardPage.tsx
- apps/packages/ui/src/components/Option/Admin/__tests__/BillingDashboardPage.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- backlog/tasks/task-45.44.7 - Migrate-design-system-product-state-Admin-and-health-expansion.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace BillingDashboardPage's forbidden and not-available admin guard AntD Alerts with the shared design-system Alert primitive while preserving copy, route capability behavior, and guard semantics. Remove matching product-state baseline entries and record focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 BillingDashboardPage forbidden guard feedback renders through data-ds-component="Alert" with error urgency semantics.
- [x] #2 BillingDashboardPage unsupported-route guard feedback renders through data-ds-component="Alert" while preserving the Not Available copy.
- [x] #3 The matching BillingDashboardPage AntD Alert baseline entries are removed without introducing a BillingDashboardPage verifier finding.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the narrow BillingDashboardPage admin guard slice:

- Extended the existing unsupported-route regression test to assert the nearest `data-ds-component="Alert"` ancestor with a hard null check before element usage.
- Added forbidden guard regression coverage by allowing the billing capability probe and making the overview endpoint reject with a 403 guard error.
- Verified the focused test failed before implementation because the existing AntD Alert markup had no design-system Alert ancestor.
- Replaced the two guard-return AntD Alerts with the shared design-system Alert primitive while preserving title/body copy, error/warning urgency, and page padding around the guard feedback.
- Removed the two BillingDashboardPage entries from the product-state baseline.

Verification:

- RED: `bunx vitest run src/components/Option/Admin/__tests__/BillingDashboardPage.test.tsx --reporter=dot` failed with `expected null not to be null` after adding the design-system marker assertions.
- GREEN: `bunx vitest run src/components/Option/Admin/__tests__/BillingDashboardPage.test.tsx --reporter=dot` passed with 2 tests. jsdom emitted existing CSS parse warnings from AntD styles.
- `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed with 54 tests.
- Baseline count script reported `{"total":185,"billing":0,"admin":27}`.
- `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` completed with no output.
- `git diff --check` completed with no output.
- `bun run verify:design-system-state` still exits 1 because of unrelated current-dev product-state drift and stale IntegrationPolicyPanel baseline rows; `/tmp/billing-dashboard-design-state.log` has no BillingDashboardPage findings.
- Bandit not run: this slice touched TypeScript UI, JSON baseline data, and Backlog task markdown only.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated BillingDashboardPage forbidden and unsupported-route guard feedback from AntD Alert to the design-system Alert primitive, added focused regression coverage for both states, and removed the two matching product-state baseline exceptions.

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

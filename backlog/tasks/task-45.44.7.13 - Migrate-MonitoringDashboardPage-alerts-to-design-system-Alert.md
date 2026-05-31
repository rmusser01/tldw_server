---
id: TASK-45.44.7.13
title: Migrate MonitoringDashboardPage alerts to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-30 21:29'
labels:
  - design-system
  - webui
  - product-state
dependencies: []
references:
  - apps/packages/ui/src/components/Option/Admin/MonitoringDashboardPage.tsx
  - >-
    apps/packages/ui/src/components/Option/Admin/__tests__/MonitoringDashboardPage.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - 'https://github.com/rmusser01/tldw_server/pull/2167'
documentation:
  - >-
    Implementation notes: added focused MonitoringDashboardPage regression
    coverage for guard states plus system sandbox rule and activity feedback.
    RED evidence was 8 expected missing design-system Alert ancestor assertions
    before production edits. Migrated AntD Alert usage to the design-system
    Alert primitive and removed seven baseline rows. Verification passed for
    focused Vitest product-state guard full verifier TypeScript and diff
    whitespace. Bandit skipped because only frontend TSX JSON and Backlog
    markdown changed.
parent_task_id: TASK-45.44.7
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate MonitoringDashboardPage admin guard, system, sandbox diagnostics, and starter-rule feedback from AntD Alert to the design-system Alert primitive with focused regression coverage and product-state baseline cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MonitoringDashboardPage access denied and not-available guard states render through the design-system Alert primitive.
- [x] #2 MonitoringDashboardPage empty system data, sandbox diagnostics error/empty, and starter-rule guidance feedback render through the design-system Alert primitive while preserving existing copy.
- [x] #3 Focused tests assert migrated feedback surfaces through data-ds-component="Alert" with null-safe ancestor guards.
- [x] #4 The MonitoringDashboardPage product-state baseline entries are removed and the product-state guard remains passing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added focused MonitoringDashboardPage regression coverage for access denied, not-available, missing system data, host-local sandbox warning, empty sandbox diagnostics, sandbox diagnostics error, empty alert rules, and empty activity feedback. RED evidence: `bunx vitest run src/components/Option/Admin/__tests__/MonitoringDashboardPage.test.tsx --reporter=dot` failed with 8 expected missing `data-ds-component="Alert"` ancestor assertions before production code changed. Migrated MonitoringDashboardPage AntD Alert usage to the design-system Alert primitive while preserving existing copy and alert urgency roles. Removed the seven MonitoringDashboardPage product-state baseline exceptions. GREEN evidence: focused Vitest passed 1 file / 15 tests. Guard evidence: product-state guard Vitest passed 1 file / 54 tests. Full verifier evidence: `bun run verify:design-system-state` passed with baseline exceptions 118. TypeScript evidence: `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` passed. Whitespace evidence: `git diff --check` and `git diff --cached --check` passed. Bandit skipped because this slice touched only TypeScript/TSX UI, JSON baseline, and Backlog markdown.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated MonitoringDashboardPage access denied, not-available, empty system data, sandbox diagnostics error/empty, host-local sandbox warning, empty alert-rule, and empty activity feedback from AntD Alert to the design-system Alert primitive in PR #2167. Added focused regression coverage for the migrated states, removed the seven MonitoringDashboardPage baseline exceptions, and verified focused Vitest, product-state guard Vitest, full design-system verifier, TypeScript, and diff checks. Bandit was skipped because the slice touched only frontend TypeScript/TSX, JSON baseline, and Backlog markdown.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Tests written and passing
- [x] #2 Code follows project conventions
- [x] #3 No linter/formatter warnings in touched files
- [x] #4 No new security findings introduced in touched code
- [x] #5 Implementation matches plan
- [x] #6 Final summary added
- [x] #7 Known skips or blockers documented
<!-- DOD:END -->

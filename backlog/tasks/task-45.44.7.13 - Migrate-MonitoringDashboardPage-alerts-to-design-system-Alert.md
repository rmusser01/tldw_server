---
id: TASK-45.44.7.13
title: Migrate MonitoringDashboardPage alerts to design-system Alert
status: In Progress
labels:
- design-system
- webui
- product-state
parent_task_id: TASK-45.44.7
references:
- apps/packages/ui/src/components/Option/Admin/MonitoringDashboardPage.tsx
- apps/packages/ui/src/components/Option/Admin/__tests__/MonitoringDashboardPage.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
documentation:
- 'Implementation notes: added focused MonitoringDashboardPage regression coverage
  for guard states plus system sandbox rule and activity feedback. RED evidence was
  8 expected missing design-system Alert ancestor assertions before production edits.
  Migrated AntD Alert usage to the design-system Alert primitive and removed seven
  baseline rows. Verification passed for focused Vitest product-state guard full verifier
  TypeScript and diff whitespace. Bandit skipped because only frontend TSX JSON and
  Backlog markdown changed.'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate MonitoringDashboardPage admin guard, system, sandbox diagnostics, and starter-rule feedback from AntD Alert to the design-system Alert primitive with focused regression coverage and product-state baseline cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 MonitoringDashboardPage access denied and not-available guard states render through the design-system Alert primitive.
- [ ] #2 MonitoringDashboardPage empty system data, sandbox diagnostics error/empty, and starter-rule guidance feedback render through the design-system Alert primitive while preserving existing copy.
- [ ] #3 Focused tests assert migrated feedback surfaces through data-ds-component="Alert" with null-safe ancestor guards.
- [ ] #4 The MonitoringDashboardPage product-state baseline entries are removed and the product-state guard remains passing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Tests written and passing
- [ ] #2 Code follows project conventions
- [ ] #3 No linter/formatter warnings in touched files
- [ ] #4 No new security findings introduced in touched code
- [ ] #5 Implementation matches plan
- [ ] #6 Final summary added
- [ ] #7 Known skips or blockers documented
<!-- DOD:END -->

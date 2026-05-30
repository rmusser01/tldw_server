---
id: TASK-45.44.7.4
title: Migrate MaintenancePage guard alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- product-state
priority: medium
parent_task_id: TASK-45.44.7
modified_files:
- apps/packages/ui/src/components/Option/Admin/MaintenancePage.tsx
- apps/packages/ui/src/components/Option/Admin/__tests__/MaintenancePage.design-system.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- backlog/tasks/task-45.44.7 - Migrate-design-system-product-state-Admin-and-health-expansion.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace MaintenancePage's forbidden and not-available admin guard AntD Alerts with the shared design-system Alert primitive while preserving copy and guard behavior. Remove matching product-state baseline entries and record focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MaintenancePage forbidden guard feedback renders through data-ds-component="Alert" with error urgency semantics.
- [x] #2 MaintenancePage not-found guard feedback renders through data-ds-component="Alert" while preserving the Not Available copy.
- [x] #3 The matching MaintenancePage AntD Alert baseline entries are removed without introducing a MaintenancePage verifier finding.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- TDD red/green completed. The new MaintenancePage design-system test failed while the Access Denied and Not Available guard messages were still rendered by AntD Alert without a data-ds-component="Alert" ancestor.
- Migrated only the forbidden and not-found guard branches to the shared design-system Alert primitive. Maintenance mode, feature flag, incident, rotation run, and AntD table/form mechanics were left unchanged.
- Removed the two MaintenancePage AntD Alert baseline entries. Baseline JSON evidence after removal: total rows 189, MaintenancePage rows 0, Admin rows 31.
- Verification: focused MaintenancePage design-system Vitest passed 2/2; product-state guard unit test passed 54/54; UI TypeScript passed; `git diff --check` passed; full `bun run verify:design-system-state` still exits 1 on unrelated Integrations, WritingActionBar, Notes, and ResearchWorkspace drift, with no MaintenancePage finding. Bandit is not applicable because this slice touches only frontend TSX/test/JSON and Backlog metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated MaintenancePage forbidden and not-available guard feedback from AntD Alert to the shared design-system Alert primitive, added focused regression coverage for both guard states, and removed the two obsolete MaintenancePage baseline exceptions. Focused tests, guard unit coverage, and UI TypeScript pass; the full design-state verifier remains blocked by unrelated current-dev product-state drift outside this slice.
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

---
id: TASK-45.44.7.9
title: Migrate ServerAdminPage inline error alerts to design-system Alert
status: In Progress
labels:
- design-system
- webui
- product-state
priority: medium
parent_task_id: TASK-45.44.7
references:
- https://github.com/rmusser01/tldw_server/issues/1664
- apps/packages/ui/scripts/design-system-product-state-baseline.json
modified_files:
- apps/packages/ui/src/components/Option/Admin/ServerAdminPage.tsx
- apps/packages/ui/src/components/Option/Admin/__tests__/ServerAdminPage.design-system.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- backlog/tasks/task-45.44.7 - Migrate-design-system-product-state-Admin-and-health-expansion.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the remaining ServerAdminPage inline user, role, and media budget error feedback from AntD Alert to the design-system Alert primitive, with focused regression coverage and baseline cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ServerAdminPage users, roles, and media budget inline error states render through the design-system Alert primitive.
- [x] #2 The product-state baseline no longer contains ServerAdminPage entries.
- [x] #3 Focused regression coverage proves these error states use the design-system Alert contract.
- [ ] #4 PR link, verification results, known skips, and final summary are recorded before closeout.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-05-30:
- Added focused ServerAdminPage design-system regression coverage for usersError, rolesError, and mediaBudgetError.
- RED evidence: `bunx vitest run src/components/Option/Admin/__tests__/ServerAdminPage.design-system.test.tsx --reporter=dot` failed with 3 assertions at `expect(alert).not.toBeNull()`, proving the existing AntD Alert branches lacked a design-system Alert ancestor.
- Migrated those three inline error branches from AntD Alert props to the design-system Alert primitive with `variant="error"` while preserving existing title text and sanitized error message content.
- Removed the three ServerAdminPage baseline entries.
- GREEN evidence: `bunx vitest run src/components/Option/Admin/__tests__/ServerAdminPage.design-system.test.tsx --reporter=dot` passed 1 file / 7 tests.
- Guard evidence: `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed 1 file / 54 tests.
- Baseline count evidence: total rows 181 -> 178; Admin path rows 23 -> 20; ServerAdminPage target rows 3 -> 0.
- `git diff --check` passed.
- `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` remains blocked by unrelated current-dev TypeScript debt in `src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx:684` (`status` missing from a fixture).
- `bun run verify:design-system-state` remains blocked by unrelated current-dev IntegrationPolicyPanel baseline drift/stale entries; the verifier log contains no ServerAdminPage findings.
- Bandit skipped because this slice only touches TypeScript, TSX, JSON, and Backlog markdown.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

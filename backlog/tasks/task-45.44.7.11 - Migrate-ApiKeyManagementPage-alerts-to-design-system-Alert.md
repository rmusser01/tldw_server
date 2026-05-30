---
id: TASK-45.44.7.11
title: Migrate ApiKeyManagementPage alerts to design-system Alert
status: Done
priority: Medium
parent_task_id: TASK-45.44.7
modified_files:
- apps/packages/ui/src/components/Option/Admin/ApiKeyManagementPage.tsx
- apps/packages/ui/src/components/Option/Admin/__tests__/ApiKeyManagementPage.design-system.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- backlog/tasks/task-45.44.7 - Migrate-design-system-product-state-Admin-and-health-expansion.md
labels:
- design-system
- webui
- product-state
references:
- https://github.com/rmusser01/tldw_server/issues/1664
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/pull/2162
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate ApiKeyManagementPage access denied, not-available, new-key-created, and key-load error feedback from AntD Alert to the design-system Alert primitive with focused regression coverage and product-state baseline cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ApiKeyManagementPage access denied and not-available guard states render through the design-system Alert primitive.
- [x] #2 ApiKeyManagementPage new API key success and key-load error feedback render through the design-system Alert primitive.
- [x] #3 The product-state baseline no longer contains ApiKeyManagementPage entries.
- [x] #4 Focused regression coverage proves the migration and the product-state guard has no ApiKeyManagementPage findings.
- [x] #5 Final summary added with verification evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-05-30:
- Added focused ApiKeyManagementPage design-system regression coverage for forbidden guard, missing-endpoint guard, key-load error, and created-key success feedback.
- RED evidence: `bunx vitest run src/components/Option/Admin/__tests__/ApiKeyManagementPage.design-system.test.tsx --reporter=dot` failed with 4 expected assertions at missing `data-ds-component="Alert"` ancestors.
- Migrated the four AntD Alert branches to the design-system Alert primitive while preserving existing user-facing copy.
- Removed the four ApiKeyManagementPage baseline entries.
- GREEN evidence: `bunx vitest run src/components/Option/Admin/__tests__/ApiKeyManagementPage.design-system.test.tsx --reporter=dot` passed 1 file / 4 tests.
- Guard evidence: `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed 1 file / 54 tests.
- Baseline count evidence: total rows 174 -> 170; Admin path rows 16 -> 12; ApiKeyManagementPage target rows 4 -> 0.
- `git diff --check` passed.
- `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` passed.
- `bun run verify:design-system-state` remains blocked by unrelated current-dev findings in IntegrationPolicyPanel, WritingActionBar, Notes, and ResearchWorkspace plus stale IntegrationPolicyPanel baseline entries; the filtered log contains no ApiKeyManagementPage findings.
- Bandit skipped because this slice only touches TypeScript, TSX, JSON, and Backlog markdown.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated ApiKeyManagementPage access denied, not-available, new-key-created, and key-load error feedback from AntD Alert to the design-system Alert primitive in PR #2162. Added focused regression coverage for all four migrated feedback branches, removed the four ApiKeyManagementPage baseline exceptions, and recorded verification evidence. Focused Vitest, guard Vitest, package TypeScript, and diff checks passed; the full design-system verifier remains blocked by unrelated current-dev findings outside this slice, documented in Implementation Notes.
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

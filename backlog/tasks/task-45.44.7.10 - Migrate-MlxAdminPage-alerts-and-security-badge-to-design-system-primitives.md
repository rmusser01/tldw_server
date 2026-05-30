---
id: TASK-45.44.7.10
title: Migrate MlxAdminPage alerts and security badge to design-system primitives
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
- apps/packages/ui/src/components/Option/Admin/MlxAdminPage.tsx
- apps/packages/ui/src/components/Option/Admin/__tests__/MlxAdminPage.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- backlog/tasks/task-45.44.7 - Migrate-design-system-product-state-Admin-and-health-expansion.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate MlxAdminPage admin guard, unavailable, loaded-model, and security-risk product-state UI from AntD Alert/Tag to design-system Alert/Badge primitives with focused regression coverage and baseline cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MlxAdminPage admin guard, status-unavailable, active-model notice, and trust-remote-code security risk product-state UI render through design-system primitives.
- [x] #2 The product-state baseline no longer contains MlxAdminPage entries.
- [x] #3 Focused regression coverage proves the relevant Alert/Badge design-system contracts.
- [ ] #4 PR link, verification results, known skips, and final summary are recorded before closeout.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-05-30:
- Added focused MlxAdminPage design-system assertions for the admin guard warning, temporary-unavailable warning, active-model info notice, and trust-remote-code security-risk badge.
- RED evidence: `bunx vitest run src/components/Option/Admin/__tests__/MlxAdminPage.test.tsx --reporter=dot` failed with 4 expected assertions at missing `data-ds-component="Alert"` / `data-ds-component="Badge"` ancestors after opening the collapsed Advanced Settings section.
- Migrated the three AntD Alert branches to the design-system Alert primitive and moved the AntD Tag security/capability labels to the design-system Badge primitive.
- Removed the four MlxAdminPage baseline entries.
- GREEN evidence: `bunx vitest run src/components/Option/Admin/__tests__/MlxAdminPage.test.tsx --reporter=dot` passed 1 file / 5 tests.
- Guard evidence: `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed 1 file / 54 tests.
- Baseline count evidence: total rows 178 -> 174; Admin path rows 20 -> 16; MlxAdminPage target rows 4 -> 0.
- `git diff --check` passed.
- `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` remains blocked by unrelated current-dev TypeScript debt in `src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx:684` (`status` missing from a fixture).
- `bun run verify:design-system-state` remains blocked by unrelated current-dev IntegrationPolicyPanel baseline drift/stale entries; the verifier log contains no MlxAdminPage findings.
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

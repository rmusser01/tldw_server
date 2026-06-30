---
id: TASK-45.50
title: Migrate ReportBuilderDrawer ready badge to design-system Badge
status: Done
parent_task_id: TASK-45
references:
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/ReportBuilderDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/ReportBuilderDrawer.test.tsx
- apps/packages/ui/scripts/verify-design-system-product-state.mjs
- https://github.com/rmusser01/tldw_server/pull/1860
documentation:
- Docs/Design/tldw_web_design_system_contract.md
- Docs/Design/tldw_web_design_system_inventory.md
modified_files:
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/ReportBuilderDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/ReportBuilderDrawer.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system migration by replacing ReportBuilderDrawer's readiness success AntD Tag with the shared design-system Badge primitive while preserving the canonical ready label and existing report-readiness behavior. Keep scope limited to the ready readiness badge and focused tests so the product-state guard returns to the accepted baseline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ReportBuilderDrawer readiness success tag is replaced with the shared design-system Badge primitive.
- [x] #2 The ready badge preserves READY_STATE_LABEL and the existing watchlists i18n key.
- [x] #3 Focused ReportBuilderDrawer coverage asserts the ready label renders inside a success design-system Badge.
- [x] #4 Product-state guard verification and inherited TypeScript baseline debt are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced ReportBuilderDrawer's ready readiness AntD Tag with the shared design-system Badge primitive while preserving READY_STATE_LABEL and the existing watchlists i18n key. Added focused test coverage that asserts the ready label renders inside a design-system Badge with the success variant. Verification: focused ReportBuilderDrawer vitest passed; design-system product-state verifier passed at the accepted 349 legacy AntD exceptions; git diff --check passed. Full UI TypeScript still fails on inherited baseline debt outside the touched files. Bandit skipped because this slice only touches TypeScript and Backlog metadata.
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

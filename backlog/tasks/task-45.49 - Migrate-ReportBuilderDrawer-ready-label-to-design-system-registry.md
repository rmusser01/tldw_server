---
id: TASK-45.49
title: Migrate ReportBuilderDrawer ready label to design-system registry
status: Done
parent_task_id: TASK-45
references:
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/ReportBuilderDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/ReportBuilderDrawer.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/pull/1858
documentation:
- Docs/Design/tldw_web_design_system_contract.md
- Docs/Design/tldw_web_design_system_inventory.md
modified_files:
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/ReportBuilderDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/ReportBuilderDrawer.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system migration by routing the Watchlists ReportBuilderDrawer readiness success label through the canonical design-system state registry instead of a local hardcoded Ready fallback. Keep scope limited to the ready label, focused tests, and removal of the matching canonical-state-label baseline entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Routed ReportBuilderDrawer's readiness success fallback through READY_STATE_LABEL from the design-system state registry, added focused regression coverage with a mocked registry label, and removed the migrated canonical-state-label baseline row. Verification: focused ReportBuilderDrawer vitest passed; design-system product-state verifier passed with baseline exceptions reduced from 350 to 349 and no canonical-state-label exceptions; git diff --check passed. Full UI TypeScript still fails on inherited baseline debt outside the touched files. Bandit skipped because this slice only touches TypeScript, JSON, and Backlog metadata.
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

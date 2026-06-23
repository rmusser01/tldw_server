---
id: TASK-45.55
title: Migrate ACP readiness setup-required labels to design-system registry
status: Done
labels:
- design-system
- webui
- product-state
- acp
parent_task_id: TASK-45
references:
- apps/packages/ui/src/services/acp/readiness.ts
- apps/packages/ui/src/design-system/states.ts
- apps/packages/ui/scripts/verify-design-system-product-state.mjs
documentation:
- Docs/Design/tldw_web_design_system_contract.md
- Docs/Design/tldw_web_design_system_inventory.md
modified_files:
- apps/packages/ui/src/design-system/states.ts
- apps/packages/ui/src/design-system/__tests__/states.test.ts
- apps/packages/ui/src/services/acp/readiness.ts
- apps/packages/ui/src/services/acp/__tests__/readiness.test.ts
- backlog/tasks/task-45.55 - Migrate-ACP-readiness-setup-required-labels-to-design-system-registry.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system migration by routing the remaining ACP readiness canonical Setup required labels through the shared design-system state registry instead of local string literals. Preserve existing ACP readiness normalization semantics while removing the current product-state guard blockers for apps/packages/ui/src/services/acp/readiness.ts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ACP readiness setup-required label values come from the design-system state registry.
- [x] #2 Existing ACP readiness normalization behavior is preserved.
- [x] #3 Focused ACP readiness coverage preserves setup-required labels.
- [x] #4 Direct product-state guard scan over ACP readiness reports zero findings.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-06-23: Added `SETUP_REQUIRED_STATE_LABEL` as a defensive label alias in the design-system state registry.
- 2026-06-23: Migrated ACP readiness generic setup-blocker titles to `SETUP_REQUIRED_STATE_LABEL`.
- 2026-06-23: Added ACP readiness coverage that mocks the design-system setup-required label to prove generic setup blockers use the registry value.
- 2026-06-23 verification: `bunx vitest run src/services/acp/__tests__/readiness.test.ts` passed.
- 2026-06-23 verification: `bunx vitest run src/design-system/__tests__/states.test.ts` passed.
- 2026-06-23 verification: direct analyzer scan of `src/services/acp/readiness.ts` returned `[]`.
- 2026-06-23 note: package-wide `node scripts/verify-design-system-product-state.mjs` no longer reports ACP readiness findings, but still reports unrelated current-base findings in `src/components/Option/Skills/SkillPreview.tsx`, `src/components/Option/ScheduledTasks/ScheduledTaskDetailDrawer.tsx`, and `src/components/Option/Skills/Manager.tsx`.
- 2026-06-23 note: Bandit is not applicable for this task because the touched implementation/test files are TypeScript and the remaining touched file is a Backlog task record.
- 2026-06-28 review fix: changed the ACP readiness design-system test mock to a partial `vi.importActual` mock so non-overridden `@/design-system` exports remain available.
- 2026-06-28 review verification: RED focused readiness test failed because the full `@/design-system` mock hid `getDesignSystemState`; GREEN `bunx vitest run src/services/acp/__tests__/readiness.test.ts` passed with 8 tests after switching to a partial mock. `bunx vitest run src/design-system/__tests__/states.test.ts`, `git diff --check`, and the direct ACP readiness analyzer also passed. Package-wide product-state guard still reports unrelated current-dev AudioStudio/TTS/ScheduledTasks/Skills findings and zero ACP readiness findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated ACP readiness setup-required titles from local literals to the shared design-system setup-required label alias. Added registry alias coverage and ACP readiness regression coverage so future generic setup-blocker labels continue to come from the design-system registry.
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

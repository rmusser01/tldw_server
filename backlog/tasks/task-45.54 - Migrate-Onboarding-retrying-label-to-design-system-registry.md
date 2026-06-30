---
id: TASK-45.54
title: Migrate Onboarding retrying label to design-system registry
status: Done
labels:
- design-system
- webui
- product-state
- onboarding
parent_task_id: TASK-45
references:
- apps/packages/ui/src/components/Option/Onboarding/steps/FirstChatStep.tsx
- apps/packages/ui/src/design-system/states.ts
- apps/packages/ui/scripts/verify-design-system-product-state.mjs
documentation:
- Docs/Design/tldw_web_design_system_contract.md
- Docs/Design/tldw_web_design_system_inventory.md
modified_files:
- apps/packages/ui/src/components/Option/Onboarding/steps/FirstChatStep.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstChatStep.design-system.test.ts
- backlog/tasks/task-45.54 - Migrate-Onboarding-retrying-label-to-design-system-registry.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system migration by routing the remaining Onboarding FirstChatStep canonical Retrying label through the shared design-system state registry instead of a local string literal. Preserve existing first-chat retry behavior and tests while removing the current product-state guard blocker for apps/packages/ui/src/components/Option/Onboarding/steps/FirstChatStep.tsx.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FirstChatStep Retrying label comes from the design-system state registry.
- [x] #2 Existing first-chat retry behavior and copy are preserved.
- [x] #3 Focused Onboarding coverage preserves the retrying state label.
- [x] #4 Direct product-state guard scan over FirstChatStep reports zero findings.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-06-23: Added a focused FirstChatStep design-system guard test that runs the product-state analyzer against the real component source. Red check failed on the hardcoded Retrying label, then passed after migration.
- 2026-06-23: Migrated the FirstChatStep retrying pill label to `getDesignSystemState("retrying").label` without changing retry behavior.
- 2026-06-23 verification: `bunx vitest run src/components/Option/Onboarding/__tests__/FirstChatStep.design-system.test.ts` passed.
- 2026-06-23 verification: `bunx vitest run src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx` passed.
- 2026-06-23 verification: direct analyzer scan of `src/components/Option/Onboarding/steps/FirstChatStep.tsx` returned `[]`.
- 2026-06-23 verification after rebasing onto `origin/dev`: `bunx vitest run src/components/Option/Onboarding/__tests__/FirstChatStep.design-system.test.ts` passed.
- 2026-06-23 verification after rebasing onto `origin/dev`: `bunx vitest run src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx` passed.
- 2026-06-23 verification after rebasing onto `origin/dev`: direct analyzer scan of `src/components/Option/Onboarding/steps/FirstChatStep.tsx` returned `[]`.
- 2026-06-23 note: package-wide `node scripts/verify-design-system-product-state.mjs` still reports unrelated current-base blocked findings in `src/components/Option/Skills/SkillPreview.tsx`, `src/components/Option/ScheduledTasks/ScheduledTaskDetailDrawer.tsx`, `src/components/Option/Skills/Manager.tsx`, and `src/services/acp/readiness.ts`.
- 2026-06-23 note: Bandit is not applicable for this task because the touched implementation/test files are TypeScript and the remaining touched file is a Backlog task record.
- 2026-06-23 review fix: Updated the focused design-system test to resolve `FirstChatStep.tsx` from `import.meta.url` instead of `process.cwd()` so the test is stable across Vitest launch directories.
- 2026-06-23 verification: `apps/packages/ui/node_modules/.bin/vitest run --root apps/packages/ui --config vitest.config.ts src/components/Option/Onboarding/__tests__/FirstChatStep.design-system.test.ts` passed from the repository root.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the Onboarding `FirstChatStep` retrying pill from a local `Retrying` literal to the shared design-system `retrying` state label. Added focused regression coverage that runs the product-state analyzer over the real component source so future hardcoded canonical state labels in this component are caught.
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

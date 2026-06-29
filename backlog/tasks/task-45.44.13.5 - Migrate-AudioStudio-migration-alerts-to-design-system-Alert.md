---
id: TASK-45.44.13.5
title: Migrate AudioStudio migration alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- product-state
- audio-studio
parent_task_id: TASK-45.44.13
references:
- apps/packages/ui/src/components/Option/AudioStudio/MigrationBanner.tsx
- apps/packages/ui/src/components/Option/AudioStudio/CompatibilityRedirect.tsx
- apps/packages/ui/scripts/verify-design-system-product-state.mjs
documentation:
- Docs/Design/tldw_web_design_system_contract.md
- Docs/Design/tldw_web_design_system_inventory.md
modified_files:
- apps/packages/ui/src/components/Option/AudioStudio/MigrationBanner.tsx
- apps/packages/ui/src/components/Option/AudioStudio/CompatibilityRedirect.tsx
- apps/packages/ui/src/components/Option/AudioStudio/__tests__/CompatibilityRedirect.test.tsx
- backlog/tasks/task-45.44.13.5 - Migrate-AudioStudio-migration-alerts-to-design-system-Alert.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system migration by resolving current AudioStudio product-state guard drift in MigrationBanner and CompatibilityRedirect. Replace product-state AntD Alert usage with the shared design-system Alert primitive while preserving existing AudioStudio migration/compatibility copy, actions, and tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 AudioStudio migration/compatibility product-state alerts render through the shared design-system Alert primitive while preserving copy and actions.
- [x] #2 Focused AudioStudio coverage asserts migrated alerts are inside the design-system Alert marker.
- [x] #3 Direct product-state guard scan over the touched AudioStudio files reports zero findings.
- [x] #4 Full product-state verifier progress is recorded with unrelated current-dev drift noted if any.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added focused AudioStudio DS Alert marker coverage for static migration guidance, legacy-project load errors, migration preview success, and commit error states.
- RED: `bunx vitest run src/components/Option/AudioStudio/__tests__/CompatibilityRedirect.test.tsx` failed on missing `data-ds-component="Alert"` ancestors for those four states while the components still rendered AntD Alert.
- Migrated `MigrationBanner` info/error/success product-state alerts and `CompatibilityRedirect` load-error alert to `@/components/ui/primitives/Alert`, preserving visible copy and actions.
- GREEN: `bunx vitest run src/components/Option/AudioStudio/__tests__/CompatibilityRedirect.test.tsx` passed with 7 tests.
- Broader AudioStudio page suite: `bunx vitest run src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx` passed with 36 tests.
- Direct product-state analyzer over `MigrationBanner.tsx` and `CompatibilityRedirect.tsx` returned `[]` for both files.
- `git diff --check` passed.
- Full product-state guard still exits 1 on unrelated current-dev ScheduledTasks/Skills/TTS blockers and 6 stale baseline entries, with `touchedFindingCount: 0` for AudioStudio.
- Bandit is not applicable because this slice only touches frontend TypeScript/TSX and Backlog markdown.
- Review cycle: rebased on latest origin/dev and removed the local `rounded-md` override from the AudioStudio guidance Alert so the design-system Alert default radius remains authoritative.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated AudioStudio migration/compatibility product-state alerts from AntD Alert to the shared design-system Alert primitive. Focused tests now assert DS Alert ownership for the static migration guidance, legacy-project load errors, preview success, and commit error states, and direct guard scans report no AudioStudio findings.
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

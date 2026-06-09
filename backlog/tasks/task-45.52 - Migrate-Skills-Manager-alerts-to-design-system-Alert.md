---
id: TASK-45.52
title: Migrate Skills Manager alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- product-state
- skills
ordinal: 45.48
parent_task_id: TASK-45
references:
- apps/packages/ui/src/components/Option/Skills/Manager.tsx
- apps/packages/ui/src/components/ui/primitives/Alert.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- apps/packages/ui/scripts/verify-design-system-product-state.mjs
documentation:
- Docs/Design/tldw_web_design_system_contract.md
- Docs/Design/tldw_web_design_system_inventory.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system migration by replacing the remaining Skills Manager AntD product-state Alert usage with the shared design-system Alert primitive. Preserve existing copy, alert semantics, actions, and Skills Manager behavior while removing the current product-state guard blockers for apps/packages/ui/src/components/Option/Skills/Manager.tsx.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Skills Manager setup/recovery product-state alerts render through the shared design-system Alert primitive.
- [x] #2 Existing alert copy, variants, and user-facing behavior are preserved.
- [x] #3 Focused Skills Manager coverage asserts the migrated alerts are inside the design-system Alert marker.
- [x] #4 Direct product-state guard scan over Skills Manager reports zero findings.
- [x] #5 Full product-state verifier progress is recorded, including any unrelated remaining blockers.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Progress: migrated the Skills Manager load-error and success-action product-state alerts from AntD Alert to the shared design-system Alert primitive. Added focused Manager test assertions for the DS Alert marker, confirmed the test failed before implementation, then passed after migration. Verification so far: focused Skills tests passed (4 files, 33 tests), direct Skills Manager product-state guard scan passed with zero findings, and the full product-state verifier no longer reports Skills Manager. Remaining full-verifier blockers are KnowledgeQA SetupDiagnostics Ready/Blocked labels, Onboarding FirstChatStep Retrying label, and ACP readiness Setup required labels.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Skills Manager load-error and success-action product-state alerts now render through the shared design-system Alert primitive. Focused Manager coverage asserts the DS Alert marker for both migrated paths, and the test was verified red before implementation. Focused Skills tests pass, direct Skills Manager product-state guard scan reports zero findings, and the full verifier now shows only the remaining KnowledgeQA, Onboarding, and ACP readiness canonical-label blockers. Bandit is not applicable because this task touched TypeScript/TSX and Backlog markdown only.
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

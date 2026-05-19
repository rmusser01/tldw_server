---
id: TASK-45.44.11.3
title: Migrate CharacterDialogs alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- extension
- product-state
priority: medium
parent_task_id: TASK-45.44.11
references:
- https://github.com/rmusser01/tldw_server/issues/1668
- Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
- apps/packages/ui/scripts/design-system-product-state-baseline.json
modified_files:
- apps/packages/ui/src/components/Option/Characters/CharacterDialogs.tsx
- apps/packages/ui/src/components/Option/Characters/__tests__
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the CharacterDialogs product-state alerts from AntD Alert to the shared design-system Alert primitive, then remove the matching product-state guard baseline entries for this file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CharacterDialogs no longer imports or renders AntD Alert for product-state messaging.
- [x] #2 A focused CharacterDialogs/Manager test asserts migrated alerts render through the design-system Alert primitive.
- [x] #3 The product-state guard baseline no longer contains entries for src/components/Option/Characters/CharacterDialogs.tsx.
- [x] #4 Focused UI tests and design-system guard verification pass, with non-code security skip documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated CharacterDialogs product-state alerts from AntD Alert to the shared design-system Alert primitive, added focused assertions that quick-chat/chat-intent blockers render with data-ds-component="Alert", removed the 10 CharacterDialogs baseline exceptions, and verified the guard now reports 375 total baseline exceptions with 0 for CharacterDialogs. Follow-up TypeScript debt investigation reproduced the package compiler baseline at 128 diagnostics, then removed the 2 diagnostics in the PR-touched Character manager test file; the current compiler baseline is 126 diagnostics with 0 diagnostics in touched files. Verification: targeted Character manager Vitest slice passed, product-state guard test passed, bun run verify:design-system-state passed, git diff --check passed. Project-wide bunx tsc --noEmit --pretty false still fails on existing unrelated TypeScript debt across many tests/components. Bandit skipped because touched implementation is TypeScript/JSON/Backlog only and no Python files changed.
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

---
id: TASK-45.44.11.4
title: Migrate remaining Character alerts to design-system Alert
status: Done
assignee: []
created_date: '2026-05-24'
updated_date: '2026-05-26 00:47'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1668'
  - apps/packages/ui/src/components/Option/Characters/AvatarField.tsx
  - apps/packages/ui/src/components/Option/Characters/CharacterListContent.tsx
  - apps/packages/ui/src/components/Option/Characters/GenerateCharacterPanel.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.11
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the remaining Character/Persona product-state AntD Alert usages to the canonical design-system Alert primitive and remove the corresponding baseline rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Product-state baseline no longer contains Option/Characters Alert rows and the design-system guard passes.
- [x] #2 AvatarField no-backend and generation-error states render via the design-system Alert primitive.
- [x] #3 CharacterListContent load-error state renders via the design-system Alert primitive without losing retry behavior.
- [x] #4 GenerateCharacterPanel no-model and generation-error states render via the design-system Alert primitive without losing settings/retry behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Starting from origin/dev in .worktrees/design-system-next-state-slice on branch codex/design-system-next-state-slice.

RED: focused Character alert tests failed because titles had no data-ds-component="Alert" ancestor while components still used AntD Alert.

GREEN: migrated AvatarField, CharacterListContent, and GenerateCharacterPanel to @/components/ui/primitives/Alert while preserving retry/settings/dismiss callbacks.

Removed the five Option/Characters Alert exceptions from design-system-product-state-baseline.json.

PR review follow-up:
- Localized dismiss labels for dismissible Character Alerts with common:close.
- Added GenerateCharacterPanel dismiss-click coverage before retry coverage.
- Let CharacterListContent use the design-system Alert error variant styling directly.
- Populated created_date in task front matter.

Verification:
- bun run test:characters-harness => 3 files / 104 tests passed. Bun emitted its existing localStorage warning and an AntD popup/shadow-root warning in an unrelated preview path.
- bunx vitest run src/components/Option/Characters/__tests__/GenerateCharacterPanel.test.tsx src/components/Option/Characters/__tests__/AvatarField.design-system.test.tsx src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx => 3 files / 5 tests passed. Bun emitted its existing localStorage warning.
- node scripts/verify-design-system-product-state.mjs => exit 0; baseline remains 228 with no Character/Persona product-area rows.
- git diff --check => exit 0.

Bandit: skipped because this slice only changes TypeScript/TSX frontend and Backlog.md task metadata; no Python code touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the remaining Character/Persona AntD Alert product states in AvatarField, CharacterListContent, and GenerateCharacterPanel to the canonical design-system Alert primitive. Addressed PR review follow-ups by localizing dismiss labels, relying on Alert variant styling directly for CharacterListContent load-error UI, adding dismiss-action regression coverage, updating Character harness copy assertions to current readiness text, and completing task metadata.
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

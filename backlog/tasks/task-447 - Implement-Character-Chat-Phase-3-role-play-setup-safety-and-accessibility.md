---
id: TASK-447
title: Implement Character Chat Phase 3 role-play setup safety and accessibility
status: Done
labels:
- chat
- characters
- role-play
- phase-3
- frontend
- accessibility
priority: high
ordinal: 447
references:
- Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
- TASK-426
- TASK-431
- TASK-438
documentation:
- Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
- Docs/superpowers/plans/2026-05-20-character-chat-phase3-setup-safety-accessibility-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-20-character-chat-phase3-setup-safety-accessibility-plan.md
- apps/packages/ui/src/components/Option/Playground/SavedRolePlaySetupsPanel.tsx
- apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the remaining high-value Phase 3 hardening from the first-class Character Chat PRD without duplicating already-shipped role-play setup work. Scope: saved role-play setup destructive-action safety and accessible generation-style semantics inside the Role-play setup surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Saved role-play setup deletion requires explicit confirmation or offers undo before permanent deletion.
- [x] #2 The Role-play setup generation style control exposes valid radio/radiogroup semantics and selected state to assistive tech.
- [x] #3 Focused component tests cover delete safety and generation style semantics.
- [x] #4 Verification, TypeScript/Bandit applicability, and residual risks are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-20-character-chat-phase3-setup-safety-accessibility-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the remaining Phase 3 setup hardening on the existing Role-play setup surface.

- Saved role-play setup deletion now uses an inline pending-delete confirmation with confirm/cancel actions before invoking the destructive callback.
- The drawer generation-style selector now uses native radio inputs inside the existing visual card layout, preserving the staged apply payload while exposing valid radiogroup/radio semantics and checked state.
- Verification: initial focused RolePlaySetupDrawer test failed on immediate delete and missing radio roles before implementation.
- Verification: `bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx --reporter=verbose` passed with 13 tests after implementation.
- Verification: `bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/saved-role-play-setups.test.ts --reporter=verbose` passed with 2 files / 19 tests.
- Verification: `git diff --check` passed.
- TypeScript: `bunx tsc --noEmit --pretty false` still fails on existing baseline errors in MediaReadAlongPopover, EmbeddingsModelSelectionConfig, WorkspacePlayground StudioPane, useShortcutConfig, and admin-llamacpp E2E typing; none are touched files.
- Bandit skipped because this slice touches only frontend TypeScript/TSX and Backlog/plan docs.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 3 setup safety/accessibility hardening is complete. Saved role-play setup delete now requires explicit inline confirmation, and Role-play setup generation style choices expose real radio semantics while preserving the existing visual design and staged apply behavior. Focused Vitest coverage and diff hygiene pass; TypeScript remains blocked only by unrelated baseline debt, and Bandit is not applicable for this frontend-only slice.
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

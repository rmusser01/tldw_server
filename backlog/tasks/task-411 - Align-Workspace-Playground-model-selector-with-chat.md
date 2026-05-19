---
id: TASK-411
title: Align Workspace Playground model selector with chat
status: Done
labels:
- webui
- ux
- workspace-playground
priority: medium
modified_files:
- apps/packages/ui/src/components/Option/Playground/ChatModelSelectorDropdown.tsx
- apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/ChatPane.stage2.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the Workspace Playground chat pane's standalone model select with the same model selector behavior used by /chat so users can search, sort, and favorite saved models consistently.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace Playground uses the same model selector behavior as /chat for discovered chat models.
- [x] #2 Saved/favorited chat models from favoriteChatModels are available and manageable from Workspace Playground.
- [x] #3 The selector remains usable when no server models are returned, preserving the no-models/settings fallback.
- [x] #4 Focused frontend tests cover the shared selector behavior in Workspace Playground.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused Workspace ChatPane test proving the selector opens a searchable, favorite-aware model menu and can update favoriteChatModels.
2. Extract the /chat model selector trigger/dropdown into a reusable component that keeps useModelSelector as the behavior source.
3. Replace Workspace ChatPane's native select with the shared selector and wire it to the same chat model query/state.
4. Run focused Vitest and browser verification for /workspace-playground.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Extracted the /chat model selector dropdown into a shared ChatModelSelectorDropdown component and reused it from both /chat and Workspace Playground. Workspace Playground now loads discovered chat models through useModelSelector, uses favoriteChatModels/modelSelectSortMode, preserves no-models settings/help fallback behavior, and applies the same favorite-aware startup model resolution. Verification: focused Workspace ChatPane regression suite passed (6 files, 43 tests); selector-focused Workspace/useModelSelector suite passed (2 files, 17 tests); browser smoke at http://localhost:3000/workspace-playground confirmed the model selector trigger opens a searchable dropdown with settings/help fallback when no models are returned; git diff --check passed. TypeScript full project check was attempted with the frontend tsc binary and remains blocked by pre-existing repo-wide type errors outside this touched slice; filtered output showed no errors in the touched selector/ChatPane files. Bandit not applicable because the touched code is frontend TypeScript/TSX only.
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

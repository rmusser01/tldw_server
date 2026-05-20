---
id: TASK-448
title: Address PR 1874 role-play setup review comments
status: In Progress
labels:
- chat
- characters
- role-play
- review
- frontend
- accessibility
priority: high
ordinal: 448
references:
- https://github.com/rmusser01/tldw_server/pull/1874
- TASK-447
documentation:
- Docs/superpowers/plans/2026-05-20-character-chat-phase3-setup-safety-accessibility-plan.md
modified_files:
- apps/packages/ui/src/components/Option/Playground/SavedRolePlaySetupsPanel.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the live PR #1874 review thread for Character Chat role-play setup controls. Scope is the Gemini inline accessibility finding on delete confirmation semantics and focus management; preserve the existing narrow Phase 3 implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Live PR #1874 review threads are verified before patching.
- [x] #2 Saved role-play setup delete confirmation uses assertive alert semantics for the destructive confirmation.
- [x] #3 Keyboard focus moves to the confirm-delete action when the confirmation prompt appears.
- [x] #4 Focused tests cover the alert/focus behavior and pass.
- [ ] #5 Verification and PR thread resolution are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Live review surface checked with `gh pr view`, GraphQL review threads, and `gh pr checks`.
- Actionable thread: Gemini inline comment on `SavedRolePlaySetupsPanel.tsx` requested `role="alert"` and focus management for the confirm-delete action.
- Implemented alert semantics for the destructive confirmation prompt.
- Added confirm-delete focus management with per-setup refs, widened to `HTMLElement` because AntD Button may expose button or anchor elements.
- Updated the focused test mock to forward refs and asserted the alert plus focus behavior.
- Verification: `bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx --reporter=verbose` passed with 13 tests.
- Verification: `bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/saved-role-play-setups.test.ts --reporter=verbose` passed with 2 files / 19 tests.
- Verification: `git diff --check` passed.
- TypeScript: `bunx tsc --noEmit --pretty false` still fails on existing baseline errors in MediaReadAlongPopover, EmbeddingsModelSelectionConfig, WorkspacePlayground StudioPane, useShortcutConfig, and admin-llamacpp E2E typing; no touched-file errors remain.
- Bandit skipped because only frontend TypeScript/TSX and Backlog docs were touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the actionable PR #1874 Gemini review thread. The saved role-play setup delete confirmation now uses assertive alert semantics and moves keyboard focus to the confirm-delete action when the prompt appears. Focused tests and diff hygiene pass; TypeScript remains blocked only by unrelated baseline debt.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

---
id: TASK-453
title: Address PR 1888 Character Chat review comments
status: Done
references:
- https://github.com/rmusser01/tldw_server/pull/1888
- https://github.com/rmusser01/tldw_server/pull/1888#discussion_r3271397805
- TASK-452
documentation:
- Docs/superpowers/plans/2026-05-19-character-chat-phase5-power-user-extension-parity-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable review feedback on PR #1888 for the Character Chat Phase 5 sidepanel handoff slice. Scope is the Gemini comment about ControlRow manually clearing storage versus the useSelectedAssistant hook; preserve the browser-verified clear behavior and avoid broadening the PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Resolve the actionable Gemini review thread about direct storage clearing in ControlRow.
- [x] Preserve sidepanel Character Chat clear behavior for modern assistant and legacy character state.
- [x] Add focused regression coverage for the legacy-mirror clear ordering.
- [x] Verify the fix with focused tests and a real backend/WebUI browser check.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Moved the null-selection clear ordering into `useSelectedAssistant`, so legacy `selectedCharacter` mirrors are cleared before the hook writes and broadcasts a null assistant selection.
- Simplified `ControlRow.clearRolePlaySelection` back to the hook-level API: clear the local selected character id and call `setSelectedAssistant(null)`.
- Updated the ControlRow contract test so it prevents direct storage imports from returning to the component.
- Browser verification used the real FastAPI backend on `127.0.0.1:8000` and Next.js WebUI on `localhost:8080/__debug__/sidepanel-chat`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Addressed PR #1888 review feedback by centralizing race-safe role-play clearing in `useSelectedAssistant` and removing direct storage manipulation from `ControlRow`.
- Added a regression test that fails when legacy character mirrors are cleared after the null assistant broadcast.
- Verification: focused vitest suite passed (7 files, 21 tests), targeted eslint exited 0, `git diff --check` passed, and real-browser clear flow removed the Character Chat chip and legacy `selectedCharacter` state.
- Known inherited baseline: `bunx tsc --noEmit --pretty false --project tsconfig.json` still fails outside touched files in MediaReadAlongPopover, EmbeddingsModelSelectionConfig, WorkspacePlayground StudioPane, useShortcutConfig, and admin llama.cpp e2e fixtures.
- Bandit skipped: touched files are TypeScript/React and Backlog markdown only.
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

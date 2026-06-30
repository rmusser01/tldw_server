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
- [x] Use the resolved extension URL when full-app handoff falls back to `window.open`.
- [x] Replace newly added source-string contract tests with behavior-level coverage or remove redundant brittle guards.
- [x] Prevent CharacterSelect from replaying the same open request when server capabilities update asynchronously.
- [x] Verify the fix with focused tests and a real backend/WebUI browser check.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Moved the null-selection clear ordering into useSelectedAssistant, so legacy selectedCharacter mirrors are cleared before the hook writes and broadcasts a null assistant selection.
- Simplified ControlRow.clearRolePlaySelection back to the hook-level API: clear the local selected character id and call setSelectedAssistant(null).
- Replaced source-string contract tests with render-level behavior coverage for ControlRow and ConversationContextPopover, and removed the redundant source-only route guard.
- Fixed openFullApp to call window.open(url, "_blank") when runtime.getURL(path) succeeds and browser.tabs.create is unavailable.
- Added request-id dedupe in CharacterSelect so an already handled openRequest is not replayed when useServerCapabilities updates hasPersona asynchronously.
- Browser verification used the real FastAPI backend on 127.0.0.1:8000 and Next.js WebUI on localhost:8080/__debug__/sidepanel-chat.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Addressed all live PR #1888 review feedback found in inline threads and the Qodo top-level summary: role-play clearing now lives in useSelectedAssistant, extension handoff preserves runtime.getURL fallbacks, brittle source-string tests were replaced, and CharacterSelect dedupes open requests across async capability updates.
- Verification: focused vitest suite passed (6 files, 21 tests), targeted eslint exited 0 with only the existing Next pages-directory notice, git diff --check passed, and real-browser clear flow removed the Character Chat chip and legacy selectedCharacter state.
- Known inherited baseline: bunx tsc --noEmit --pretty false --project tsconfig.json still fails outside touched files in MediaReadAlongPopover, EmbeddingsModelSelectionConfig, WorkspacePlayground StudioPane, useShortcutConfig, and admin llama.cpp e2e fixtures.
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

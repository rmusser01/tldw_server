---
id: TASK-12104
title: Fix character chat route and switch regressions
status: Done
labels:
- webui
- chat
- characters
- bugfix
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve UAT findings from TASK-12103: direct character route clears selected character/avatar/readiness, switching characters inside an active character chat keeps the old conversation and can trigger speaker_character_name participant mismatch, and related UI state must remain consistent across WebUI/extension shared UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Direct /chat?mode=character&characterId=<id> loads the selected character as active, shows the character name/avatar/readiness correctly, and does not announce Character mode cleared.
- [x] #2 Selecting a different tracked character from an active tracked character chat starts a fresh conversation for that character rather than mutating the old chat.
- [x] #3 Switching characters no longer sends speaker_character_name for a participant that is not selected in the existing server chat.
- [x] #4 Regression tests cover direct route hydration and in-chat character switching behavior.
- [x] #5 Shared UI changes apply to the WebUI/extension package code paths that use the same components.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Root causes:
  - Explicit character route hydration could be overwritten by stale local selected-character state before the route intent finished applying.
  - AntD dropdown open/close churn could drop the event-supplied `applyAs: "tracked"` intent before a character item click.
  - The selected-assistant broadcast fired before the legacy selected-character mirror was updated, letting greeting logic read the previous character and restore Miku after selecting Ashley.
  - Generic playground autosave could create a tracked character server chat before the character send path owned persistence, racing the send flow and producing stale/duplicate metadata.
- Fixes:
  - Route character intent now resets by URL/location signature, waits for explicit route hydration before syncing stale local selection back into the URL, and clears stale loaded chat state when a different tracked character is selected.
  - Character Chat entry points dispatch `applyAs: "tracked"` and AssistantSelect preserves explicit event intent through AntD open-change notifications.
  - `useSelectedAssistant` now updates the legacy character mirror before notifying subscribers for tracked character selections.
  - Generic persistence skips tracked character workflow chats; the character send path owns `webui-character-chat` creation/reuse.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Fixed direct character route hydration, in-chat tracked character switching, stale assistant precedence, selector intent preservation, and tracked character persistence ownership.
- Added regression coverage for route hydration replay, stale route/local selection conflicts, tracked switch clearing, event-supplied tracked selector mode, legacy mirror ordering, effective assistant precedence, and character-send chat reuse.
- Verification:
  - `git diff --check` passed.
  - Focused Vitest suite passed 10 files / 140 tests.
  - TypeScript check was run after repairing the local `antd` symlink for this worktree; it still fails only on existing package-wide baseline diagnostics outside touched files.
  - Fresh Playwright UAT outside the sandbox passed Miku-to-Ashley switch and direct Ashley character-chat sessions with no `speaker_character_name` error.
  - Backend chat records for the fresh UAT runs were single Ashley `webui-character-chat` records.
- Known skips/blockers:
  - Bandit skipped as not applicable to TypeScript/React UI and markdown-only changes.
  - Existing warnings remain: `/openapi.json` 404, missing per-chat settings 404 warning, and AntD deprecation warnings for `overlayClassName`/`dropdownRender`.
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

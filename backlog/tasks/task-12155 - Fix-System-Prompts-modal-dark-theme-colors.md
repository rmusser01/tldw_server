---
id: TASK-12155
title: Fix System Prompts modal dark theme colors
status: Done
labels:
- webui
- extension
- ui
- theme
priority: Medium
modified_files:
- apps/packages/ui/src/assets/tailwind-shared.css
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The System Prompts modal opened from the chat composer renders AntD modal chrome with light-theme colors while the chat UI is in dark mode. Scope the fix to the modal/theme styling and verify on an isolated frontend/backend stack.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 System Prompts modal content, title, tabs, close control, and prompt cards use the active dark theme colors in the WebUI.
- [ ] #2 The existing editable current system prompt field remains visible above search and the prompt template selector.
- [ ] #3 The fix works in the shared UI surface used by WebUI and browser extension without changing unrelated theme behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the System Prompts modal dark-theme mix by extending the shared AntD theme bridge to cover .ant-modal-container, .ant-modal-body, and the modal close control. Verified on the isolated WebUI stack at 127.0.0.1:3123 with mock backend 127.0.0.1:18081: modal container resolved to rgb(35, 40, 50), text to rgb(231, 233, 238), prompt cards inherited dark text correctly, and the editable Current system prompt textarea accepted input above search. Checks: git diff --check passed; focused Vitest run passed (4 files, 50 tests); apps/extension bun run compile passed; Bandit skipped because touched code is CSS/TS frontend only.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

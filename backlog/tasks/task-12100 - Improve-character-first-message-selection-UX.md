---
id: TASK-12100
title: Improve character first-message selection UX
status: Done
modified_files:
- apps/packages/ui/src/components/Common/ChatGreetingPicker.tsx
- apps/packages/ui/src/components/Common/__tests__/ChatGreetingPicker.test.tsx
- apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/body.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a selected-first-message action to the WebUI character chat starter so choosing a greeting in the selector can immediately display that character message as the first chat message without requiring the user to send a prompt first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Greeting selector/modal includes an explicit Select action for the chosen first message.
- [ ] #2 Selecting a greeting inserts/displays it as the first assistant/character message in an empty character chat without sending a user message.
- [ ] #3 Existing character chat greeting and alternate greeting flows remain covered by focused tests.
- [ ] #4 Task records verification and frontend-only Bandit rationale.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the character first-message Select UX. Verification: red test first failed because no Select greeting button existed; focused ChatGreetingPicker test then passed 5/5. Broader focused verification passed 23/23 across ChatGreetingPicker, useCharacterGreeting, and PlaygroundChat server-load-state suites. git diff --check passed. Package typecheck required a larger Node heap and still fails on existing baseline errors outside the touched files; no diagnostics referenced this change's touched files. Bandit skipped because this task touched frontend TypeScript/TSX and Backlog markdown only, with no Python code changes.
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

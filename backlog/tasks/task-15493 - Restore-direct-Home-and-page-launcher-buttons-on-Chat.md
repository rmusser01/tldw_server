---
id: TASK-15493
title: Restore direct Home and page launcher buttons on Chat
status: Done
labels:
- webui
- navigation
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The conversation header moved Home and page navigation into More in PR #3311. Restore the two visible buttons using the existing navigation callbacks and preserve the rest of the conversation header.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chat shows Companion Home and Show pages as direct controls on desktop and mobile.
- [x] #2 Home navigates to the existing app home, and the page launcher opens from its button and restores focus on dismissal.
- [x] #3 The non-chat header and remaining More actions retain their behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reuse the existing House and Signpost controls in both header layouts. Test direct activation and keyboard focus, then verify narrow and wide browser layouts. Existing header placement is a UI choice and does not require an ADR.
The new component test reproduced missing Companion Home before the fix. After the fix 78 header and launcher tests and scoped ESLint passed. Six local Chromium checks exercised both buttons at 320, 390 and 1440 pixels in light and dark themes, including Home navigation and Escape focus return. The 320px cockpit starter/composer smoke passed. The broader header matrix timed out clicking the keyboard-shortcuts entry in More with local Webpack dependencies. A fresh unchanged-base run also positioned that entry offscreen before opening the launcher. Bandit is inapplicable to the TypeScript-only change. No public or hosted rollout.
Independent review found pointer activation could leave the launcher without its Show pages button as a return target. The shared control now focuses that button before toggling the existing launcher. Both header variants fail the new no-native-focus tests without the fix and pass with it. The final focused shared and WebUI selections passed 348 and 179 tests respectively.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored direct Companion Home and Show pages controls in the shared Chat header using the existing callbacks. On narrow screens the chat actions can wrap so the conversation title stays readable. The focused browser and component checks passed.
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

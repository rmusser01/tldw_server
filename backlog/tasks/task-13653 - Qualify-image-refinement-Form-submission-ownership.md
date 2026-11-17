---
id: TASK-13653
title: Qualify image refinement Form submission ownership
status: Done
labels:
- chat
- ci
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair reproduced Form first-send failure from incomplete model-store and Scene source fixtures without weakening production send gates. PR#3229.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Original first-send no-forced-refresh and refinement assertions pass
- [x] #2 Real model dispatch snapshot and scoped Scene capture are used
- [x] #3 Failed or retired Scene preparation preserves input and prevents dispatch
- [x] #4 Both client configurations and independent review pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED:original firstsuccessfulsend fails;dispatchsnapshot calls actualmodelstore.getState, mockonlydeclaresselectorfunction. Formalso needsqualifiedidle history/draft account forrealScene capture. ADR assessment:no newADR,test boundary repair. Plan:Docs/Plans/IMPLEMENTATION_PLAN_chat_image_form_fixture_13653.md
Finalnative/WebUIeach18pass1suite0skipprocess0(3.85s/3.86s). Original16assertions retained plus2heldScenerefusals. Realmodelstore/capture,declaredSceneIO andSDKhook;durablereadwarnings retained, no physicalpersistence claim. Boundedreviewclear;ESLint0new/0errors25existingwarnings;whitespace0;BanditN/A TypeScript. Review:Docs/Reviews/CHAT_IMAGE_FORM_FIXTURE_13653_2026_10_13.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
OriginalFormfirstsendqualifiedwithrealmodeldispatch andownedScene fixture;18tests passineachconfiguration,heldfailure/retirement keepsinput/no dispatch,review/lint/hooks clear. PR#3229.
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

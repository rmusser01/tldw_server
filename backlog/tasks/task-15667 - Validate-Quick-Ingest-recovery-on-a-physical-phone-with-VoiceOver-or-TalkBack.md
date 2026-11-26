---
id: TASK-15667
title: Validate Quick Ingest recovery on a physical phone with VoiceOver or TalkBack
status: To Do
labels:
- webui
- quick-ingest
- accessibility
- validation
references:
- TASK-15661
- https://github.com/rmusser01/tldw_server/pull/3641
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify that a person using a native mobile screen reader can repair a Quick Ingest source, go Back without losing their place or unsent input, and recover from a failed review-draft save. The prerequisite is access to a physical iPhone with VoiceOver or Android phone with TalkBack.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On a physical phone, use screen-reader touch gestures to find a failed source, hear its identity and error, open its editor, repair it and return to the edited row. Record the spoken label and focus destination.
- [ ] #2 From replacement review, activate Back. Verify that unsent input is preserved, accessibility focus returns to the originating control and its announced name describes the available action.
- [ ] #3 With one extracted source and a reproducible draft-save failure, hear the failure, read its extracted text and activate Retry save with the screen reader. Verify the saved outcome is announced, exactly one draft exists, and extraction was requested once.
- [ ] #4 Attach the build, physical device, OS/browser and screen-reader versions, gestures, expected and actual announcements, and a pass or reproducible defect for each flow.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Browser coverage already exercised these flows on emulated iPhone WebKit and Pixel Chromium. Native speech, touch exploration and accessibility focus on hardware remain unverified.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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

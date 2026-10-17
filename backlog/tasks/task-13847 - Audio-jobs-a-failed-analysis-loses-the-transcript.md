---
id: TASK-13847
title: 'Audio jobs: a failed analysis loses the transcript'
status: To Do
labels:
- audio
- jobs
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When the audio_analyze stage fails without retry (for example a missing model or rejected credentials, which since PR #3257 fail at once) or is quarantined, the job never reaches audio_store, so the transcript produced by the earlier stages is not stored as the job result. Before PR #3257 the same jobs ended quarantined, with the same loss. A user who asked for transcription plus analysis loses the transcription to an analysis setup error.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When analysis fails for good, the transcript is still stored, with the analysis failure and its error_code recorded next to it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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

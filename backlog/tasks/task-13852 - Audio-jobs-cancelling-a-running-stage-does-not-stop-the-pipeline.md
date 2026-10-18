---
id: TASK-13852
title: 'Audio jobs: cancelling a running stage does not stop the pipeline'
status: To Do
labels:
- audio
- jobs
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When an audio job is cancelled while a stage runs, that stage is marked cancelled, but the worker ignores complete_job's False return, and the next stage is already queued (since PR #3260 it is queued before the current one completes), so the later stages still run to completion. The PR #3260 review probe (cancel_mid_stage) showed audio_convert cancelled and audio_transcribe and audio_store completed; dev behaved the same before that PR. The submit response returns only the first stage's job id, so a caller can only cancel that stage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Cancelling an audio job while a stage runs stops the stages after it, with a test
- [ ] #2 A caller can cancel a submitted audio pipeline as a whole, or the API documents how
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

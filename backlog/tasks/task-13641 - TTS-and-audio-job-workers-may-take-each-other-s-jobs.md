---
id: TASK-13641
title: TTS and audio job workers may take each other's jobs
status: To Do
labels:
- audio
- tts
- jobs
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
core/TTS/tts_jobs_worker.py (WorkerSDK, no job_type filter) and app/services/audio_jobs_worker.py both acquire from domain audio, queue default with no job_type filter. If both run, the TTS worker would fail audio_* stage jobs as unsupported and the audio worker would fail tts_longform jobs as an unknown job_type. Not verified at runtime: the TTS worker appears to run only standalone (python -m). Found while fixing TASK-13637.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Verify whether both workers can run against the same queue; if so, each acquires only its own job types (separate queues or a job_type filter), with a test
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

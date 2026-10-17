---
id: TASK-13845
title: 'Audio jobs: analysis can be skipped without the submitter knowing'
status: To Do
labels:
- audio
- jobs
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two cases in POST /api/v1/audio/jobs/submit. (1) With perform_chunking false, audio_transcribe hands over straight to audio_store (app/services/audio_jobs_worker.py _handle_audio_transcribe_stage, and the GPU worker app/services/audio_transcribe_gpu_worker.py), so analysis never runs even when perform_analysis is true. (2) Since PR #3257 (TASK-13640) a job without api_name skips analysis and records analysis_skipped: "no api_name" in the payload, but nothing is logged, and the OpenAPI description of SubmitAudioJobRequest.api_name (api/v1/endpoints/audio/audio_jobs.py) does not mention the skip. Found in the reviews of PR #3257.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 perform_analysis true runs the analysis stage whether or not chunking is on, in both workers
- [ ] #2 A skipped analysis is logged and the api_name field's description says analysis is skipped without it
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

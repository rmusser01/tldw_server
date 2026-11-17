---
id: TASK-13845
title: 'Audio jobs: analysis can be skipped without the submitter knowing'
status: Done
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
- [x] #1 perform_analysis true runs the analysis stage whether or not chunking is on, in both workers
- [x] #2 A skipped analysis is logged and the api_name field's description says analysis is skipped without it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-18, branch fix/audio-jobs-analysis-stage, plan Docs/superpowers/plans/2026-10-18-audio-jobs-analysis-stage.md. stage_after_transcription(payload) in app/services/audio_jobs_worker.py picks the stage after audio_transcribe (audio_chunk if perform_chunking, else audio_analyze if perform_analysis, else audio_store); the in-process worker and the GPU worker (audio_transcribe_gpu_worker.py, which imports it) both use it. The audio_chunk branch keeps its own expression. The analyze stage logs 'Audio job <id>: analysis skipped, no api_name'. SubmitAudioJobRequest.api_name and perform_analysis descriptions updated; apps/packages/api-types/openapi.fingerprint.json regenerated (only the sha256 changed; --check exit 0).
RED: 7 failed (4x ImportError for stage_after_transcription; pipeline test 'assert [] == ["completed"]'; GPU test "['audio_store'] == ['audio_analyze']"; skip log absent, only the Starting/Stopping lines). GREEN: tests/AudioJobs + tests/Audio/test_audio_jobs_worker_helper.py 47 passed, 4 skipped (3 admin endpoints not mounted, 1 opt-in heavy smoke; the same 4 skip on dev). Bandit -ll on the touched app files: no new findings (one pre-existing B608 in core/Jobs/manager.py, also reported on dev).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
perform_analysis now runs the analysis stage whether or not chunking is on, in both the in-process and the GPU worker, through one helper (stage_after_transcription). A skipped analysis (no api_name) is logged, and the api_name field description and Docs/API-related/Audio_Jobs_API.md say analysis is skipped without it. Shipped with TASK-13846 and TASK-13847 in one PR, so jobs that now reach analysis keep their transcript when it fails.
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

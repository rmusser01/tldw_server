---
id: TASK-13640
title: Audio analysis stage retries configuration errors
status: Done
labels:
- audio
- jobs
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With raise_on_error=True the audio_analyze stage (app/services/audio_jobs_worker.py) gets SummaryProviderError for missing_model, authentication, invalid_provider and configuration failures, which carry no retryable flag, so they are retried until the job is quarantined, the same as rate_limit. An empty transcript (analyze returns 'Error: Could not extract text content.') is retried too. Found while fixing TASK-13637.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Configuration and authentication failures, and an empty transcript, fail the stage without retry; rate limits and transient provider failures are still retried
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the PR review of TASK-13636/13637: the audio stage passes failure_code to fail_job but not SummaryProviderError.code, so provider failures record no error code; forward it (e.g. missing_model, authentication, rate_limit) when classifying retries.
From the clean-session review of PR #3240: a job submitted without api_name falls back to openai and now really calls it; on a local-only install it fails on missing credentials and retries until quarantined (before PR #3240 it always failed). Use the configured default LLM provider, or skip analysis when none is configured, as the /media/process-* endpoints do.
Fixed in app/services/audio_jobs_worker.py: the audio_analyze stage re-raises SummaryProviderError as a private _StageFailure whose retryable is true only for rate_limit and provider_failure, and whose failure_code (the error's code) becomes the job's error_code; an empty or whitespace-only transcript fails with empty_transcript before any provider call; a job whose api_name is missing, blank or "None" records analysis_skipped: "no api_name" and continues to audio_store instead of calling openai. That is the implementation note's second option, as /media/process-audios does. The first option is not taken: core/Chat/chat_target_resolution.config_default_llm_provider() exists, but the config loader falls back to openai and the shipped config.txt sets default_api = openai, so resolving it would call openai on a local-only install. SummaryProviderError is unchanged, because other job workers read retryable from escaping exceptions with different defaults. Still retried until quarantined (follow-up candidate): a provider 4xx other than 401/403/429, such as an unknown or unpulled model (404) or a context overflow (400), maps to ChatBadRequestError and then to provider_failure in _summary_error_code; classifying it needs a new summary code that the RAG and Evaluations readers of SummaryProviderError.code would also see. Tests: the analyze tests 12 passed, 10 deselected (before the fix 10 failed, 2 passed, 10 deselected); every test file importing the worker plus tests/Docs 405 passed, 4 skipped. Ruff: only the two I001 in test_audio_worker_smoke.py that origin/dev has. Bandit -ll: no issues. Docs/API-related/Audio_Jobs_API.md documents the skip and the error codes (mirrored). Not here: with perform_chunking false, audio_transcribe goes straight to audio_store, so analysis never runs (pre-existing; follow-up candidate).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The audio analysis stage fails a job at once on a failure a retry cannot fix (missing model or credentials, bad configuration or authentication, unknown provider, empty transcript) and records the reason in error_code; rate limits and transient provider failures still retry. A job that names no provider skips analysis instead of calling openai.
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

---
id: TASK-13637
title: Audio analyze job stage calls analyze with the wrong arguments
status: To Do
labels:
- audio
- jobs
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The audio_analyze stage in app/services/audio_jobs_worker.py (near line 354) calls analyze(text, api_name) under asyncio.to_thread. analyze's signature is (api_name, input_data, custom_prompt_arg, ...), so the call passes the transcript as the provider name and raises TypeError before any provider call; the stage fails and is retried until its attempts run out. When the call is fixed, the stage's failure path (fail_job with retryable=True, near line 384) should honour the exception's retryable flag and failure_code, so an LLM quota refusal (LLMQuotaExceededError, retryable = False, failure_code llm_quota_exceeded) fails the job at once. Found while planning LLM usage accounting PR 2b (TASK-13437).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 audio_analyze calls analyze with the provider, the transcript and a prompt, and stores the analysis
- [ ] #2 A failure's retryable flag and failure_code reach fail_job
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

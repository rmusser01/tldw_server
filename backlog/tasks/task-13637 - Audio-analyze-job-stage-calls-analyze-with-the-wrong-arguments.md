---
id: TASK-13637
title: Audio analyze job stage calls analyze with the wrong arguments
status: Done
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
- [x] #1 audio_analyze calls analyze with the provider, the transcript and a prompt, and stores the analysis
- [x] #2 A failure's retryable flag and failure_code reach fail_job
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Correction to the description: at base the TypeError stage ended quarantined (failure streak), not retried until its attempts ran out. Fixed: the stage calls analyze(api_name or 'openai', text, None, input_is_literal_text=True, raise_on_error=True) (the payload's model is the transcription model and is not passed; jobs carry no prompt, so analyze's default system prompt, bulleted notes, applies); an Error: result fails the stage; fail_job gets retryable and error_code from the exception, so a quota refusal fails the job at once with llm_quota_exceeded and an ordinary provider error is retried.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The audio analyze stage passed the transcript as the provider name, so every analysis job failed and ended quarantined. It now calls analyze with the provider, the transcript and analyze's default system prompt (bulleted notes), stores the analysis, fails the stage on an Error: result, and passes the exception's retryable flag and failure code to fail_job: a quota refusal fails the job at once with llm_quota_exceeded. Follow-ups: TASK-13640 (configuration errors are retried), TASK-13641 (TTS and audio workers share a queue).
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

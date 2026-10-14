---
id: TASK-13640
title: Audio analysis stage retries configuration errors
status: To Do
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
- [ ] #1 Configuration and authentication failures, and an empty transcript, fail the stage without retry; rate limits and transient provider failures are still retried
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

---
id: TASK-13902
title: 'Chat: a streamed provider 429 answers 502 provider_unavailable for every provider'
status: To Do
labels:
- llm
- chat
- errors
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while planning TASK-13883. Measured on e06eeec68c: the remote streamed 429 (and the groq endpoint test test_chat_endpoint_streaming_error_groq) answers 502 provider_unavailable, while the non-stream 429 answers 429. PROVIDER_STREAM_ERROR_MESSAGES (streaming_utils.py) has no rate-limit code. This is why TASK-13883 AC #2 'streamed' is a no-op for local providers (pinned by tests, not changed). A real fix is a product choice (new stream code, status, Retry-After), so it is its own task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A streamed provider 429 has a defined, tested chat response that says rate limited
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

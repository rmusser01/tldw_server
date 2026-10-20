---
id: TASK-13883
title: 'Chat: a local provider''s 401, 403 or 429 reply answers 400 Invalid request'
status: To Do
labels:
- llm
- errors
- chat
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while planning TASK-13865. _raise_sanitized_local_provider_error (core/LLM_Calls/providers/local_adapters.py) makes every upstream 4xx a ChatBadRequestError. So /chat/completions answers a local server's 401, 403 or 429 (llama.cpp, Ollama, Kobold and the other local adapters) with 400 {"detail": "Invalid request."}, or with 502 provider_unavailable when streamed (measured on dev 2026-10-20). A remote provider's 401 answers provider_authentication_failed, and its 429 is a rate limit. TASK-13865 kept the upstream status on the error as upstream_status_code for the summary codes only, and left chat responses unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A local provider's 401 or 403 reply gets the chat response a remote provider's authentication failure gets, streamed and not
- [ ] #2 A local provider's 429 reply gets the chat response a remote provider's rate limit gets, streamed and not, and an unpulled model (404) keeps its current response
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

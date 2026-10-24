---
id: TASK-13883
title: 'Chat: a local provider''s 401, 403 or 429 reply answers 400 Invalid request'
status: Done
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
- [x] #1 A local provider's 401 or 403 reply gets the chat response a remote provider's authentication failure gets, streamed and not
- [x] #2 A local provider's 429 reply gets the chat response a remote provider's rate limit gets, streamed and not, and an unpulled model (404) keeps its current response
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED (617094d014): unit test_local_http_error_mapping 3 failed (401, 403, 429: ChatBadRequestError raised), stream unit test_local_streaming_contract 2 failed (401, 403 frame code provider_unavailable), endpoint test 5 failed 3 passed (not streamed 401/403/429, streamed 401/403 fail; streamed 429 and both 404 pass). Before (llama.cpp via /chat/completions): not streamed 401/403/429 -> 400 'Invalid request.'; streamed 401/403/429 -> 502 provider_unavailable.
GREEN: unit 6 passed (file), stream 6 passed (file), endpoint 8 passed, summary local_provider_reply cases 14 passed. After: 401/403 -> 502 provider_authentication_failed (not streamed and streamed); 429 not streamed -> 429 'Rate limit exceeded. Please retry.'; 404 unchanged (400 / 502 provider_unavailable streamed). Finding for AC #2: a remote provider's streamed 429 already answers 502 provider_unavailable (no rate-limit stream code), which local streaming already gave; pinned by tests, not changed; filed as TASK-13902. Sweep (LLM_Calls, LLM_Adapters, Chat translation, BYOK validation) under a global autouse blocker: 1642 passed, 3 skipped; only non-loopback host recorded slot-2.example. Test quality ratchet: 1 passed. Bandit -ll on the two source files: no issues. ADR: none needed; reuses existing Chat error types and stream code. Documentation: n/a, no doc describes the local error answers.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
_raise_sanitized_local_provider_error now raises ChatAuthenticationError for 401/403 and ChatRateLimitError for 429 (404 and other 4xx stay ChatBadRequestError), keeping upstream_status_code; the local stream error frame carries provider_authentication_failed for a pre-output 401/403. Chat answers now equal a remote provider's. Dead 401/403/429 branches removed from the summary error-code mapping.
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

---
id: TASK-13865
title: 'Summary errors: an egress-blocked endpoint, an unconfigured custom endpoint
  and local-provider 4xx replies get the wrong code'
status: Done
labels:
- llm
- errors
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while planning TASK-13844. EgressPolicyError (core/exceptions.py) is in neither _SUMMARY_ADAPTER_EXCEPTIONS nor _SUMMARIZATION_NONCRITICAL_EXCEPTIONS (core/LLM_Calls/Summarization_General_Lib.py), so it escapes analyze() untyped whatever raise_on_error is, and the audio jobs worker's generic handler retries it. custom_openai_adapter.py:158 and :211 raise a bare RuntimeError for a missing base URL, which a summary records as provider_failure; tests pin it as RuntimeError. Local adapters (local_adapters.py _raise_sanitized_local_provider_error) map every 4xx, including 401, 403 and 429, to ChatBadRequestError without upstream_status_code, so a local 429 now reads request_rejected and is not retried. TASK-13844 typed only the LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER> kill switch, by reading the adapter's own switch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An egress-blocked provider endpoint gets a SummaryProviderError code the audio analysis stage does not retry, and analyze(raise_on_error=False) callers keep their current behaviour
- [x] #2 A custom OpenAI-compatible provider with no base URL gets the configuration summary code; any chat response change is stated and tested
- [x] #3 A local provider's 401, 403 and 429 map to authentication and rate_limit summary codes (not request_rejected), and an unpulled model (404) stays request_rejected
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-20-summary-codes-egress-custom-local.md (four tasks: egress and a local server that is down, custom base URL, local 4xx, docs and close-out; independent review folded in: 0 Blocker, 1 Important, 6 Minor). ADR assessment: none needed (the failures map to existing summary codes and one int is kept on ChatBadRequestError; ADR-025 followed, the local mapper keeps the upstream status as the shared _raise_sanitized_provider_failure does; ADR-026/030 followed, the transport keeps EgressPolicyError.reason_code and the summary only reads it).
AC1: EgressPolicyError and the transport's NetworkError join _SUMMARY_ADAPTER_EXCEPTIONS as _SUMMARY_PASSTHROUGH_EXCEPTIONS. _summary_error_code maps an egress block as chat_service._map_provider_egress_error does: dns_unresolved -> provider_failure (retried), every other reason -> configuration (not retried). A NetworkError (a kobold server that is down; review M5) falls through to provider_failure, so the audio stage retries it and then stores the transcript with analysis_error; _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS is not widened. No new summary code, so the Evaluations mapping, _RETRYABLE_ANALYSIS_CODES and the docs list need no new entry. raise_on_error=False callers still get either exception itself (two control tests). Readers: the audio worker keeps the transcript; the response-quality and RAG evaluators now fail the request (503 provider_configuration_invalid, or 502 provider_unavailable) instead of scoring 0, and eval_runner records the sample as an error. RED 10 failed (7 summary, 3 worker), GREEN 10 + 5 passed.
AC2: custom_openai_adapter.py _resolve_base and _resolve_transport_context raise ChatConfigurationError (same messages) instead of RuntimeError, as the credentialed branch already did; the summary records configuration. /chat/completions is unchanged (it always binds a credential handle: 503 provider_configuration_invalid before and after, streamed and not, measured); perform_chat_api_call without a handle now raises ChatConfigurationError instead of RuntimeError (new chat_service test). So the Workflows LLM, RAG, media, audio-briefing and messaging steps, the FVA analyze function, the web-search aggregation report and the Setup first-chat category now behave as they already do for a credentialed configuration error (review I1's measured caller list is in the plan); a new Setup test pins the first-chat category (configuration_error, was provider_error). The 7 adapter test cases that pinned RuntimeError now expect ChatConfigurationError. RED 15 failed, GREEN 15 passed.
AC3: _raise_sanitized_local_provider_error keeps the upstream status as upstream_status_code (class, status 400 and message unchanged, so chat responses are unchanged: 400 Invalid request non-stream, 502 provider_unavailable stream, measured before and after). _summary_error_code maps a ChatBadRequestError with upstream 401/403 to authentication and 429 to rate_limit; 408 now gets the transient rule; an unpulled model's 404 stays request_rejected. RED 12 failed (8 summary through ollama and kobold, 4 mapper), GREEN 4 + 14 passed.
Sweep over the 118 test files that name a changed symbol or custom-openai-api, with every non-loopback connect and DNS lookup blocked: 7 failed, 3813 passed, 18 skipped, 1 error (dev e521605198, same files and blocking: 7 failed, 3774 passed, 18 skipped, 1 error; the same 7 failures and error are dev's: 6 custom_openai adapter cases that resolve api.openai.com, plus test_service_prompts_api and test_summarization_input_extraction, which depend on run order and pass alone on this branch; the 39 new cases are the difference). tests/Docs 212 passed; quality and scope-predicate ratchets 14 passed. Ruff: only the 7 findings dev has in these files. Bandit clean. Docs mirror refreshed. Follow-ups: TASK-13883 (the chat response for a local 401, 403 or 429) and TASK-13884 (the custom OpenAI adapter tests that resolve api.openai.com).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A summary whose provider endpoint the egress policy blocks is now configuration (a host that does not resolve is provider_failure and retried), a local server that is down is provider_failure (retried, then the transcript is kept) instead of escaping untyped, a custom OpenAI-compatible provider with no base URL is configuration, and a local provider's 401/403 is authentication and its 429 rate_limit (an unpulled model stays request_rejected). No new summary code. raise_on_error=False callers and /chat/completions responses are unchanged; an in-process perform_chat_api_call without a credential handle now gets ChatConfigurationError for a missing custom base URL.
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

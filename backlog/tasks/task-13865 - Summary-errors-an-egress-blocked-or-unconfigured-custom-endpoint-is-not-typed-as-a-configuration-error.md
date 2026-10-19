---
id: TASK-13865
title: 'Summary errors: an egress-blocked or unconfigured custom endpoint is not typed
  as a configuration error'
status: To Do
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
- [ ] #1 An egress-blocked provider endpoint gets a SummaryProviderError code the audio analysis stage does not retry, and analyze(raise_on_error=False) callers keep their current behaviour
- [ ] #2 A custom OpenAI-compatible provider with no base URL gets the configuration summary code; any chat response change is stated and tested
- [ ] #3 A local provider's 401, 403 and 429 map to authentication and rate_limit summary codes (not request_rejected), and an unpulled model (404) stays request_rejected
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

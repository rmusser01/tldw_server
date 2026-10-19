---
id: TASK-13844
title: 'Summary provider errors: a provider rejecting the request reads as a transient
  failure'
status: Done
labels:
- llm
- errors
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
_summary_error_code (core/LLM_Calls/Summarization_General_Lib.py) maps a ChatBadRequestError to provider_failure. build_sanitized_chat_error (core/LLM_Calls/error_utils.py) turns every upstream 4xx other than 401, 403 and 429 into ChatBadRequestError, so an unknown or unpulled model (404), a prompt too long for the model's context (400), 402 and 413 all read as a transient provider failure. The adapter's own configuration errors (providers/openai_adapter.py: RuntimeError 'native HTTP disabled by configuration', ValueError 'Configured endpoint scope mismatch') also become provider_failure, and an OpenAI 429 insufficient_quota (exhausted billing) becomes rate_limit. Since PR #3257 (TASK-13640) the audio jobs worker retries only rate_limit and provider_failure, so these jobs are still retried once and then quarantined instead of failing at once. About 12 RAG and Evaluations sites record SummaryProviderError.code as metadata and would see any new code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A provider rejecting the request (an unknown model, a context overflow) and the adapter's configuration errors get a summary code the audio analysis stage does not retry
- [x] #2 An exhausted-billing 429 is distinguishable from a rate limit
- [x] #3 The RAG and Evaluations readers of SummaryProviderError.code are checked for the new codes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-18-summary-provider-error-codes.md (three tasks: request_rejected and the kill switch; provider_quota_exhausted; the reader audit, docs and close-out). ADR assessment: none needed (two codes added to an internal bounded set and one bool on ChatRateLimitError; ADR-025 followed, the quota signal is decided in the shared build_sanitized_chat_error).
AC1: _summary_error_code maps ChatBadRequestError (every upstream 4xx but 401, 403 and 429, via build_sanitized_chat_error) to the new request_rejected code, and a RuntimeError from an adapter whose LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER> switch is off to configuration. The transient 4xx replies (OpenRouter 408, Groq 498) stay provider_failure, read from the error's upstream_status_code, so they are still retried. The adapters' 12 raise sites are unchanged. The openai_adapter scope-mismatch ValueError cannot reach a summary: the summary request never carries a call_policy. RED 7 failed (6 summary codes, 1 worker retry), GREEN 16 passed (15 in the summarization file's selection, 1 worker test).
AC2: ChatRateLimitError gains quota_exhausted (bool, default False). build_sanitized_chat_error sets it from the failed reply's error.type/error.code against {insufficient_quota}, through _raise_sanitized_provider_failure, normalize_error and raise_chat_error_from_http. The summary maps it to provider_quota_exhausted. Chat responses are unchanged: same class, status and message, chat_service rebuilds the error without the flag, and nothing else reads it. A streamed 429 keeps rate_limit (the body is unread when httpx raises). RED 10 failed, GREEN 114 passed in the two files plus 2 worker tests.
AC3: reader audit in the plan. Neither new code is in _RETRYABLE_ANALYSIS_CODES, so both fail the analysis stage at once. Evaluations (_eval_provider_http_exception) answers both with 502 provider_unavailable, as before; the kill switch now gets 503 provider_configuration_invalid, as other configuration errors do. The RAG readers record the constant provider_unavailable or allowlist the code (the description's 12 metadata sites record no summary code on dev); HyDE now says provider_configuration_invalid for the kill switch. New test pins every summary code's Evaluations response. Sweep over every test file naming a changed symbol, plus tests/Docs: 2469 passed, 4 skipped (2257 passed, 4 skipped without tests/Docs, plus 212 in tests/Docs; the plan measured dev at 2226 passed, 4 skipped without tests/Docs, and the 31 new tests are the difference), plus the Review M4 chat-surface files: 858 passed, 3 skipped. test_test_quality_ratchet: 1 passed. Ruff: only the 3 findings dev has. Bandit clean. Follow-up: TASK-13865 (EgressPolicyError, the custom_openai base-URL RuntimeError and local-adapter 4xx mapping).
Clean-session review of PR #3279: 0 Critical, 0 Important, 3 Minor. M1 (merge instructions miss the audio_jobs_worker.py comment conflict with #3273) was posted as PR comments. M2 (three surviving mutants: kill-switch value ignored, 498 dropped from the retryable set, allowlist reading only error.type or only error.code) is fixed with a kill-switch-on fake adapter and three new OpenAI-adapter rows; each mutant was applied and the new tests failed. M3 (TASK-13865 title hid the local-provider 4xx/429 regression) is fixed by retitling it.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A summary the provider refuses (an unknown model, a prompt over the context) is now request_rejected, a spent billing quota (OpenAI insufficient_quota) is provider_quota_exhausted, and an adapter switched off with LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER>=0 is configuration. The audio analysis stage retries none of them. Chat responses are unchanged; Evaluations and HyDE now report the switched-off adapter as a configuration error.
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

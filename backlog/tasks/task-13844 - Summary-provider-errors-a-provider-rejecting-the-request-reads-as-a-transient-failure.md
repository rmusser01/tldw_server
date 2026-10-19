---
id: TASK-13844
title: 'Summary provider errors: a provider rejecting the request reads as a transient
  failure'
status: In Progress
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
- [ ] #1 A provider rejecting the request (an unknown model, a context overflow) and the adapter's configuration errors get a summary code the audio analysis stage does not retry
- [ ] #2 An exhausted-billing 429 is distinguishable from a rate limit
- [ ] #3 The RAG and Evaluations readers of SummaryProviderError.code are checked for the new codes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-18-summary-provider-error-codes.md (three tasks: request_rejected and the kill switch; provider_quota_exhausted; the reader audit, docs and close-out). ADR assessment: none needed (two codes added to an internal bounded set and one bool on ChatRateLimitError; ADR-025 followed, the quota signal is decided in the shared build_sanitized_chat_error).
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

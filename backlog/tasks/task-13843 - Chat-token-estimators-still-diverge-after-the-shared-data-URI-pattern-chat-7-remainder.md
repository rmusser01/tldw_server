---
id: TASK-13843
title: Chat token estimators still diverge after the shared data-URI pattern (chat-7
  remainder)
status: Done
labels:
- chat
- tokens
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3247 (TASK-13560) gave the chat path one bounded data-URI redaction pattern (DATA_URI_REDACT_RE in core/Utils/image_validation.py). The rest of finding chat-7 (Docs/superpowers/reviews/chat/2026-09-21-stage2-orchestration-streaming.md) is open: (1) core/Chat/prompt_cost_envelope.py:_sanitize_data_uris is a hand-rolled scanner that finds positions in text.lower() and slices the original text, so a character whose lowercase is longer (for example 'İ') shifts every later position; its output feeds prompt fingerprints, so changing it changes them. (2) Two rounding rules: estimate_tokens_from_json and _estimate_tokens_from_messages use len // 4, prompt_cost_envelope.estimate_segment_tokens uses (len + 3) // 4. (3) api/v1/endpoints/chat.py applies _sanitize_json_for_rate_limit before estimate_tokens_from_json, which redacts the same string again with the same pattern: one redundant full scan per estimate. (4) Text written as data:image,<long run without quotes or whitespace> is dropped from every pre-call estimate (rate limit, billing reservation, queue, size check); post-call accounting uses provider usage. chat-7 proposes one core/Chat/token_estimation.py with one redactor and one rounding rule.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The chat path's prompt-size estimators share one redactor and one rounding rule, with a test that they agree on the same input
- [x] #2 The prompt-cost redactor keeps positions right for text whose lowercase changes length
- [x] #3 The redundant outer redaction pass is removed
- [x] #4 A decision is recorded on whether data:image text without a real image counts toward pre-call estimates
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-18-chat-token-estimators.md. Owner decisions (approved 2026-10-18 with the plan): D1 (AC4 redaction) = option B, count data:image text unless it is a real base64 image, using the possessive pattern; D2 = round up, (n+3)//4, never below 1 for a request. The owner's approval of the plan covers chat-7's owner-only chat.py sites. ADR assessment: no ADR needed (ADR-064 and ADR-067 are related and unchanged; the estimator rule is a reversible heuristic recorded in the plan and docstrings). The committed plan is the design note chat-7 asks for.
AC2: prompt_cost_envelope._sanitize_data_uris now uses the shared DATA_URI_REDACT_RE. The hand-rolled scanner found positions in text.lower() and sliced the original text, so 30 'İ' before a data URI kept the 400-character payload. RED 1 failed, 8 passed (test_prompt_cost_envelope.py); GREEN 23 passed with the guardrail and world-book prompt-context tests. A pinned origin/dev fingerprint shows ordinary base64 images fingerprint as before. Fingerprints: nothing caches, dedupes or meters on them (guardrail churn compares only client-echoed values; the WebUI sends none; llm_usage_log.prompt_fingerprint is never written; character-chat metadata is write-only), so there is no migration and FINGERPRINT_VERSION stays prompt-v1.
AC1 and AC3: estimate_tokens_from_json, _estimate_tokens_from_messages and usage_accounting._prompt_estimate all return prompt_cost_envelope.estimate_segment_tokens: one redactor, rounded up ((n+3)//4, never below 1 for a request; owner decision D2), because every use gates or bills tokens and none of these estimators trims context. That covers the chat path and the estimate rows of every recorded provider call (usage_accounting). Character chat's preview-only _estimate_tokens is out of scope. The endpoint's _sanitize_json_for_rate_limit (the redundant outer pass at the three estimate sites) and _estimate_tokens_for_queue are deleted, as are chat_service._sanitize_data_uris and _sanitize_messages_for_token_estimate; each estimate site redacts once. New test_chat_token_estimators_agree.py: RED 6 failed, 1 passed; GREEN 16 passed with test_prompt_cost_envelope.py. Directly affected tests: 84 passed.
AC4 decision (owner, D1): B. Only a real base64 image (data:image/<type>[;params];base64, plus the base64 alphabet, JSON-escaped line breaks allowed) is left out of an estimate or the size check. data:image text without one now counts. Residual: base64-alphabet text after a valid header is still left out; option C (redact image positions only) would close it. The payload run is possessive: same output as the plain alternation on 32 inputs (every Task 3 case and the PR #3247 inputs), and a 41.9 MB ten-image body redacts in 0.50x the json.dumps time (40 ms vs 81 ms, best of 3, load average 49); test_image_bulk_redacts_within_four_times_the_json_dumps_time pins it. RED 6 failed, 24 passed; GREEN 109 passed across the five pattern-user test files. Sweep (200 files plus tests/CI/test_test_quality_ratchet.py, -n 4, 98 minutes at load average 17-82): 1 failed, 6377 passed, 35 skipped; the image-bulk guard and the ratchet passed. The one failure is Character_Chat_NEW/integration/test_chat_import_local_chats.py::test_another_owner_reusing_ids_never_touches_the_first_owners_chat[postgres] (500 from /api/v1/chats/import); it fails the same way run alone on the branch and on unchanged origin/dev 4d35534699, so it is not this change. No router-contract failures this run. Ruff: 5 findings, all already on origin/dev, none new. Bandit -ll on the five source files: no issues. Lint ratchets (private coercion, core-to-api import boundary, inline SSE frames): 9 passed.
Clean-session review of PR #3276 (2026-10-18): 0 Critical, 1 Important, 2 Minor, all fixed. I1: a line-wrapped base64 image in an image-only message was counted whole by the guardrail envelope; DATA_URI_REDACT_RE now accepts raw CR/LF in the payload run (still possessive), with an envelope test. M1: plan now records the Change-summary waiver. M2: _approx_tokens docstring reworded.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The chat path sizes prompts one way: every estimator (rate limits, RG and billing reservations, quota check, queue admission, prompt-cost guardrail, stream usage), and every recorded provider call's estimate row (usage_accounting), returns estimate_segment_tokens, with one data-URI redactor, rounded up. The guardrail redactor no longer mis-slices text whose lowercase is longer, each estimate site redacts the request once instead of twice, and data:image text that is not a real base64 image now counts. Ordinary prompt fingerprints are unchanged.
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

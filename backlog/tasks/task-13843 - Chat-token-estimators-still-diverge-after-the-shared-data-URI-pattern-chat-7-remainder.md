---
id: TASK-13843
title: Chat token estimators still diverge after the shared data-URI pattern (chat-7
  remainder)
status: In Progress
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
- [ ] #1 The chat path's prompt-size estimators share one redactor and one rounding rule, with a test that they agree on the same input
- [ ] #2 The prompt-cost redactor keeps positions right for text whose lowercase changes length
- [ ] #3 The redundant outer redaction pass is removed
- [ ] #4 A decision is recorded on whether data:image text without a real image counts toward pre-call estimates
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-18-chat-token-estimators.md. Owner decisions (approved 2026-10-18 with the plan): D1 (AC4 redaction) = option B, count data:image text unless it is a real base64 image, using the possessive pattern; D2 = round up, (n+3)//4, never below 1 for a request. The owner's approval of the plan covers chat-7's owner-only chat.py sites. ADR assessment: no ADR needed (ADR-064 and ADR-067 are related and unchanged; the estimator rule is a reversible heuristic recorded in the plan and docstrings). The committed plan is the design note chat-7 asks for.
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

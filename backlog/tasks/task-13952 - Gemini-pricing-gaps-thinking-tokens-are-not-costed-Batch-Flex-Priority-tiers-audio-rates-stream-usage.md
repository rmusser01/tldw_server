---
id: TASK-13952
title: 'Gemini pricing gaps: thinking tokens are not costed; Batch/Flex/Priority tiers,
  audio rates, stream usage'
status: To Do
labels:
- quotas
- backend
- high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13936 (ai.google.dev, fetched 2026-10-26). High priority because of (1). (1) Thinking tokens are under-billed: GoogleAdapter._normalize_to_openai_shape sets completion_tokens from candidatesTokenCount only and drops thoughtsTokenCount, but thinking tokens are counted apart from the candidates and billed as output. Sources: the thinking guide (ai.google.dev/gemini-api/docs/thinking) says 'When thinking is turned on, response pricing is the sum of output tokens and thinking tokens'; the pricing page's 2.5 output label is 'Output price (including thinking tokens)'; the repo fixture in tests/Chat/unit/test_chat_service_content.py has totalTokenCount 115 = promptTokenCount 80 + candidatesTokenCount 30 + thoughtsTokenCount 5. Thinking is on by default on gemini-2.5-pro and gemini-2.5-flash, so every such reply's thinking is free in the usage record. Fix: add thoughtsTokenCount to completion_tokens. (2) The pricing page has Batch, Flex and Priority tables but says nothing about how a tier is selected or reported; find the response field, then add service_tiers to the Gemini entries. (3) Audio input and cached audio have their own prices; the catalog has text rates only. (4) gemini-2.0-flash, -2.0-flash-001 and -2.0-flash-lite are not on the pricing page and have no cache rate (Google publishes none). (5) The Google adapter's stream (GoogleAdapter.stream) emits no usage chunk, so a streamed Gemini reply through any caller (RAG, workflows, /chat/completions) is costed from an estimate. (6) The pricing page (2026-10-26) also lists newer Gemini models and price changes effective December 1, none of them catalogued; re-read the page when doing this task and update the catalogued rates. promptTokenCount including cachedContentTokenCount is confirmed by the API reference (UsageMetadata), as the normalizer assumes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Gemini thinking tokens (thoughtsTokenCount) are counted and costed as output tokens, with a test
- [ ] #2 A streamed Gemini reply is recorded from the provider's usage when the stream reports it, with a test
- [ ] #3 Each remaining item is priced with a test or recorded as not priced in Usage_Quotas.md
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

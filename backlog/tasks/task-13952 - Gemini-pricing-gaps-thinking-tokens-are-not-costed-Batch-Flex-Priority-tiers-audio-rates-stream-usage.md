---
id: TASK-13952
title: 'Gemini pricing gaps: thinking tokens are not costed; Batch/Flex/Priority tiers,
  audio rates, stream usage'
status: Done
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
- [x] #1 Gemini thinking tokens (thoughtsTokenCount) are counted and costed as output tokens, with a test
- [x] #2 A streamed Gemini reply is recorded from the provider's usage when the stream reports it, with a test
- [x] #3 Each remaining item is priced with a test or recorded as not priced in Usage_Quotas.md
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From TASK-13951 (2026-10-26), on item 5: until it lands, a streamed Gemini /chat/completions reply is still costed from the stream estimate (TASK-13951's AC #2 path), because GoogleAdapter.stream drops usageMetadata. When the stream ends with one chunk {"choices": [], "usage": <the block _normalize_to_openai_shape builds, cachedContentTokenCount included>} (Gemini's usageMetadata is cumulative, so the last one is the total), both recorders read it with no other change: StreamingResponseHandler feeds it to _StreamUsage, and the selected provider 'google' takes the normalizer's Gemini branch. Test it with a Gemini case in test_a_streamed_reply_is_recorded_from_the_usage_chunk_its_provider_sends (tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py).
Plan: Docs/superpowers/plans/2026-11-03-gemini-pricing-gaps.md. Owner answers: ADR-004 Change summary waived (owner, 2026-11-03: "Yes, waiver covers it"); Decision 3 (audio not priced here, follow-up TASK-15422) agreed 2026-11-03 (plan approved as recommended); Decision 4 (today's rates, dated follow-up TASK-15423 labelled high, reminders on 2026-11-30 and 2026-12-31) agreed 2026-11-03; Decision 6 (a stopped or dropped stream's thinking tokens, follow-up TASK-15424) agreed 2026-11-03.
How: GoogleAdapter._usage_block builds the OpenAI-shaped usage block from a usageMetadata for the reply and the stream: thoughtsTokenCount joins completion_tokens (Google bills thinking as output), toolUsePromptTokenCount joins prompt_tokens, both kept under their Gemini names (reasoning_tokens; the audit JSON), total_tokens stays totalTokenCount (= prompt + tool-use + thoughts + candidates), cachedContentTokenCount kept, serviceTier passed on as service_tier (lowercased). _stream_event_deltas keeps the last block (Gemini's counts are cumulative) and stream() yields it once as {choices: [], usage}: before [DONE] at Gemini's end of stream, before a proxy's [DONE] line (which now ends the stream), and before the error when the stream fails mid-way (a transport error, or an in-band error event, now caught as ChatAPIError and re-raised unchanged); an empty usageMetadata sends none. _StreamUsage.feed/feed_event read it unchanged, so both recorders record the stream from it (verified: the provider-level row from the real adapter through a stubbed transport; the chat pipeline from the chunk shape). normalize_llm_usage's Gemini branch adds the thoughts to output_tokens only for raw usageMetadata keys (no completion_tokens), so an adapter block is never double counted. model_pricing.json: service_tiers (flex, priority) on the 14 Gemini rows and four new rows (gemini-4-argon, gemini-3.8-flash, gemini-3.6-flash, gemini-3.5-flash-lite), written by the script from the page tables in test_pricing_service_tiers.py; gemini-3.1-flash-lite's stale estimated flag dropped.
RED/GREEN (record the real counts): Task 1 11 failed, 3 passed -> 14 passed; Task 2 9 failed, 8 passed -> 17 passed (the new adapter file 21 passed); Task 3 39 failed, 10 passed -> 49 passed. Base verification of item 1: the repo fixture recorded completion_tokens 30 of 35; of item 5: a stubbed two-event stream yielded two deltas and [DONE], no usage. Focused set 1473 passed, 1 skipped -> 1536 passed, 1 skipped; wider set 2546 passed, 13 skipped (68 files) -> 2609 passed, 13 skipped (69 files); tests/Docs+lint 513 passed; tests/CI+backlog-py 1148 passed, 6 skipped; shard coverage new_uncovered=0; test-quality ratchet OK; route-auth exact; OpenAPI unchanged; Bandit No issues identified (Medium 0, High 0). Sources (fetched 2026-11-03): aistudio.google.com/docs/pricing (ai.google.dev redirects there), /docs/thinking, /api-reference/generate-content (UsageMetadata.serviceTier: unspecified, standard, flex, priority; no request field), /docs/batch-api (batchGenerateContent, a separate endpoint), /docs/flex-inference and /docs/priority-inference (a tier is requested on the Interactions API, which no server code calls; Priority reports it in the x-gemini-service-tier header).
Not priced, recorded in Usage_Quotas.md: Batch (not a reportable tier; no server code calls batchGenerateContent); Contributor (invite-only; no ServiceTier value); audio input and cached audio (2x-3.3x the text rate on four models; the modality split in promptTokensDetails/cacheTokensDetails is dropped) -> TASK-15422; the 2.0 Flash entries (off the page; no cache rate, no tiers; rates kept); the dated changes (gemini-4-argon doubles 2026-12-01; 3.8/3.6 Flash double 2027-01-01; from 2026-12-01 2.5 Pro is billed as 3.8 Flash, over-charged until 2027-01-01 and under-charged on input and cache after it, and 2.5 Flash as 3.5 Flash-Lite, the same on standard text; the catalog holds the 2026-11-03 rates) -> TASK-15423, labelled high, reminders on 2026-11-30 and 2026-12-31; a stream that is stopped or dropped, or fails before any usage, is costed from the estimate without its thinking tokens -> TASK-15424. Not catalogued: the page's specialised models (Live, TTS, Image, Transcribe, Embedding, Omni Flash, Robotics ER). Docs: Usage_Quotas.md (Service tiers, Gemini, Streams without a usage chunk), Model_Pricing_Catalog.md (source line), the _catalog_note.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Gemini thinking tokens are costed as output and tool-use prompt tokens as input: GoogleAdapter._usage_block folds thoughtsTokenCount into completion_tokens and toolUsePromptTokenCount into prompt_tokens, keeps both by name for the row's reasoning_tokens and the audit JSON, keeps cachedContentTokenCount and passes usageMetadata.serviceTier on as service_tier. The Gemini stream ends with one usage chunk from the last usageMetadata, read by the provider-level recorder and the chat pipeline without any change to them, so a streamed Gemini reply is no longer costed from the estimate. The catalog prices Flex and Priority on every Gemini text model from Google's tables and adds gemini-4-argon, gemini-3.8-flash, gemini-3.6-flash and gemini-3.5-flash-lite at the 2026-11-03 rates. A stream that fails mid-way sends the chunk before the error. Batch, Contributor, audio input, the 2.0 Flash entries, the dated price changes and the thinking tokens of a stopped or dropped stream are recorded as not priced in Usage_Quotas.md; TASK-15422 (audio), TASK-15423 (dated changes) and TASK-15424 (a stopped or dropped stream) follow up.
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

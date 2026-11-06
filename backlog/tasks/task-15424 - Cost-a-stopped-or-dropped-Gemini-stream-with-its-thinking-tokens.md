---
id: TASK-15424
title: Cost a stopped or dropped Gemini stream with its thinking tokens
status: To Do
labels:
- quotas
- backend
- high
dependencies:
- TASK-13952
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since TASK-13952, GoogleAdapter.stream ends with one usage chunk built from the last usageMetadata (Gemini's counts are cumulative): at the stream's end, before a proxy's [DONE], and before the error when the stream fails mid-way. A stream that the client drops, or that the WebUI's Stop ends (TASK-13953: the handler stops reading and closes the stream), gets no chunk: the adapter cannot yield from a closed generator. Both recorders then cost it from the estimate (the prompt plus a quarter of the characters received), which leaves out the thinking tokens. Gemini thinks before it answers, so by the time content is shown Google has billed the whole thinking budget, often most of the output cost. Each event's usageMetadata already carries thoughtsTokenCount, and every delta carries its raw event as provider_response. Fix: _StreamUsage (LLM_Calls/usage_accounting.py) keeps GoogleAdapter._usage_block of the last delta's provider_response usageMetadata as a fallback, used only when no usage chunk arrived; both recorders (the provider-level _record_stream and the chat handler, which feeds feed_event) then cost from it. Never emit a usage chunk per event: clients that sum usage chunks (LangChain aggregates usage_metadata across chunks) would double count. Related gap: OpenAI reasoning tokens arrive only in a stream's final usage chunk, so a stopped or dropped OpenAI stream is costed without them too. Recorded in Docs/Operations/Usage_Quotas.md (Gemini bullet).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Gemini stream that the client drops after usage arrived is costed with its thinking tokens, from the last usageMetadata, on /chat/completions and through the provider-level recorder, with a test
- [ ] #2 A Gemini stream stopped from the WebUI after usage arrived is costed with its thinking tokens, with a test
- [ ] #3 No usage chunk is emitted per event, and a completed stream still records its one final chunk, with a test
- [ ] #4 Usage_Quotas.md is updated, and the OpenAI reasoning-token gap is noted there as related
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

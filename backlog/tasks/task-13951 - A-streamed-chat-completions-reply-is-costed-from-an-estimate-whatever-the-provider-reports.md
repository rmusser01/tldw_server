---
id: TASK-13951
title: A streamed /chat/completions reply is costed from an estimate whatever the
  provider reports
status: Done
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13936. chat_service records a streamed /chat/completions reply after the stream ends from an estimate (prompt estimate plus len(reply)/4, estimate_source stream_estimate, no usage_metadata; execute_streaming_call's save_callback in chat_service.py), and the call records itself, so the provider-level recorder that reads the provider's last usage chunk never runs. So a streamed reply is costed at the standard rate with no service tier (OpenAI Flex, Fast, Batch), no cache rates, no Gemini cache reads and no OpenRouter usage.cost, whatever the provider reports. TASK-13916 fixes the Anthropic adapter's stream usage; that alone does not change /chat/completions, which uses this recorder. Usage_Quotas.md documents the gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A streamed /chat/completions reply whose provider sends a usage chunk is recorded with that usage (service tier, cache tokens, reported cost), with a test
- [x] #2 A stream without a usage chunk is still recorded from the estimate
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-26-stream-usage.md (owner-approved). ADR assessment: no new ADR; ADR-067's stream consequence is updated (the OpenAI adapter asks for a usage chunk; the chat pipeline shares the recorder's parser); ADR-075 governs Anthropic-shaped prompt tokens; ADR-059 the backlog tool.
RED/GREEN (real counts, 2026-10-26, branch fix/stream-usage-13951 from origin/dev 35e711080a): Task 1 6 failed, 5 passed -> 11 passed (Task 1 files 200 passed, 1 skipped); Task 2 2 failed, 3 passed -> 5 passed (test_openai_native_http.py 9 passed); the five touched test files 96 passed; focused set 2344 passed, 2 skipped -> 2357 passed, 2 skipped; wider set 1401 passed, 2 skipped -> 1404 passed, 2 skipped (2 standing skips); empty-choices set 5 failed, 2032 passed, 1 skipped -> 5 failed, 2043 passed, 1 skipped (the same 5 order-dependent kobold summarization failures before and after); tests/Docs 212 passed; test_quality_triage --enforce OK; shard coverage new_uncovered=0.
How: StreamingResponseHandler feeds every parsed stream payload to usage_accounting._StreamUsage.feed_event (split out of feed, so both recorders share one parser) and passes the whole usage block to a save_callback that accepts provider_usage; chat_service's save_callback records from it (tokens via _usage_tokens, the block as usage_metadata), takes a side the block does not count from the stream estimate (marked estimated), and records the stream estimate when there is no block or it counts no tokens; the Resource Governor commit gets the recorded total. The handler forwards an empty-choices usage chunk only when the client asked (stream_options.include_usage at the top level or in extra_body; chat._client_asked_for_stream_usage, carried as forward_usage_chunk through execute_streaming_call and StreamingPipelineRequest); OpenRouter's usage chunk carries a choice and is forwarded as before. OpenAIAdapter.stream asks api.openai.com and OpenAI's data-residency hosts (*.api.openai.com) for the usage chunk (stream_options.include_usage, merged under the caller's stream_options). No double recording: /chat/completions runs with records_itself=True.
Sources, fetched 2026-10-26: developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create and .../streaming-events (include_usage adds a last chunk with empty choices; every other chunk has usage null; the chunk carries service_tier; an interrupted stream may lack the usage chunk); developers.openai.com/api/docs/guides/your-data (data-residency hosts are a prefix on api.openai.com: us., eu., au., ca., jp., in., sg., kr., gb., ae.); openrouter.ai/docs/use-cases/usage-accounting (usage in the last SSE message, always; include_usage deprecated, no effect); openrouter.ai/docs/api-reference/streaming (OpenRouter's usage chunk carries one content-free choice, not empty choices).
Decisions (owner-approved plan): ask for usage only on OpenAI's own hosts, caller's stream_options win; read the usage chunk for the record and forward an empty-choices usage chunk only to a client that asked (c'), so clients receive what they did before; the recorder reads any provider's usage chunk, other providers are asked in a follow-up (TASK-13954); dropped or failed streams unchanged (no row), follow-up TASK-13953 filed high. Premise corrections: a dropped or failed /chat/completions stream wrote no usage row before this change either (save_callback runs only on a complete stream; rg_refund_cb refunds), verified with a throwaway test; and OpenRouter's usage chunk carries one choice, so no /chat/completions client received an unrequested empty-choices chunk before this change (the plan's first draft assumed otherwise; the independent review caught it).
Out of scope, unchanged: the Anthropic adapter's stream passes no usage chunk on (TASK-13916), nor does the Google adapter's (TASK-13952 item 5), so those streams are still costed from the estimate; both now reach /chat/completions with no further change once their adapter emits the chunk. /v1/messages with an OpenAI provider stops reading at the finish chunk (openai_stream_to_anthropic), so it stays estimated, as before.
Docs: Docs/Operations/Usage_Quotas.md (the streamed replies bullet), Docs/API-related/Chat_API_Documentation.md (the usage chunk and who receives it), Docs/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md (complete-v2 forwards OpenAI's usage chunk raw) and ADR-067's two stream consequences, with the Docs/Published mirror refreshed. Bandit (uvx bandit, 2026-10-26) on the six touched source files: 0 findings (loc 17868). Each task got a spec + quality review; deferred minors are listed in the PR.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A streamed /chat/completions reply is recorded from the usage chunk its provider sends, so its service tier, cache rates and OpenRouter's reported cost apply as for a reply that is not streamed; a stream without one is still recorded from the estimate. The stream handler reads usage with the provider-level recorder's own parser (_StreamUsage.feed_event) and hands the whole block to save_callback; the Resource Governor commit gets the recorded total. The OpenAI adapter asks api.openai.com and OpenAI's data-residency hosts for the usage chunk (stream_options.include_usage), which also gives real usage to every other caller's OpenAI stream; /chat/completions forwards that empty-choices chunk only to a client that asked for usage, so clients receive what they did before, and OpenRouter's usage chunk is forwarded as it always was. Anthropic and Gemini streams stay estimated until TASK-13916 and TASK-13952; dropped or failed streams still write no row (TASK-13953).
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

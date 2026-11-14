---
id: TASK-13916
title: A streamed reply through the Anthropic adapter is costed from an estimate
status: To Do
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the TASK-13878/13879/13880/13885 PR (plan Docs/superpowers/plans/2026-10-23-pricing.md). Confirmed in that plan's review (M3): AnthropicAdapter.stream (anthropic_adapter.py, the stream method) never handles message_start, yields message_delta only as a finish_reason chunk with the usage nested under provider_response, and _StreamUsage.feed (usage_accounting.py) reads only the top-level usage. So a streamed /chat/completions reply (and any other adapter caller) is costed from a prompt estimate: cache reads and writes, 1-hour writes, fast mode and US-only inference are all missed. Usage_Quotas.md documents it. The Messages API path reads the real usage already.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A streamed reply through the Anthropic adapter is recorded with the provider's real usage (input, cache, output tokens, speed, inference_geo), with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From TASK-13936 (2026-10-26): fixing the adapter's stream usage does not change /chat/completions, whose streamed replies chat_service records from an estimate; that is TASK-13951.
From TASK-13951 (2026-10-26): /chat/completions now reads a streamed reply's usage with the provider-level recorder's own parser (usage_accounting._StreamUsage.feed_event, which StreamingResponseHandler calls for each parsed payload) and counts its tokens with _usage_tokens. So fixing the Anthropic adapter's stream usage reaches both recorders at once, whether the adapter puts a top-level usage on its OpenAI-shaped chunks or feed_event learns to read provider_response. Mind the merge: feed_event merges a later usage block over an earlier one only when type == 'message_delta'; an OpenAI-shaped chunk has no type, so emit one block per stream (message_start's usage with message_delta's merged over it) or keep the type. Emit it as choices: [] so the /chat/completions forwarding guard applies. The selected provider 'anthropic' sends an Anthropic-shaped block through the normalizer's Anthropic branch, and _usage_tokens adds its cache tokens to the prompt (ADR-075). Add an Anthropic case to test_a_streamed_reply_is_recorded_from_the_usage_chunk_its_provider_sends (tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py, fixture stream_usage_rows).
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

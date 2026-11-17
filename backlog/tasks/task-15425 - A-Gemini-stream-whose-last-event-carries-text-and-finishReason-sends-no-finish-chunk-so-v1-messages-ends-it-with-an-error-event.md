---
id: TASK-15425
title: A Gemini stream whose last event carries text and finishReason sends no finish
  chunk, so /v1/messages ends it with an error event
status: To Do
labels:
- backend
- chat
- high
dependencies:
- TASK-13952
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13952's final review (PR #3422), pre-existing. GoogleAdapter._stream_event_deltas (LLM_Calls/providers/google_adapter.py, the 'if finish_reason_raw and not emitted:' branch) emits a finish chunk (delta {}, finish_reason) only when the event carried no text or tool call. Gemini's last stream event usually carries both the final text and finishReason, so that stream yields no chunk with a finish_reason at all. Effects: (1) the Anthropic Messages API path (Chat/anthropic_messages.py) never sees a finish, reads on, and ends the stream with an error event (on the base at [DONE]; since TASK-13952 at the empty-choices usage chunk, which it also rejects): measured by the review as message_start, content_block_start, deltas, error, identically on the base and on the PR head. A Messages client on a Gemini provider gets an error after a complete reply. (2) The /chat/completions handler gets no finish_reason from such a stream (valid_finish_received stays false), and OpenAI-shaped clients see no finish_reason. A Gemini stream whose finishReason arrives in its own event is not affected. Fix: carry finish_reason on the delta emitted for an event that holds both text (or a tool call) and finishReason, or emit a trailing finish chunk regardless of emitted. Keep TASK-13952's rule: the usage chunk goes out once, after the finish, before [DONE].
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Gemini stream whose last event carries text and finishReason yields a chunk with that finish_reason, with an adapter test
- [ ] #2 /v1/messages with a Gemini provider ends such a stream with message_delta (stop_reason) and message_stop, not an error event, with a test through the real adapter and a stubbed transport
- [ ] #3 The usage chunk still goes out exactly once, after the finish and before [DONE], and the /chat/completions recorder still records it, with the TASK-13952 tests passing
- [ ] #4 Docs/Operations/Usage_Quotas.md's Streams bullet states the /v1/messages mechanism correctly for Gemini
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

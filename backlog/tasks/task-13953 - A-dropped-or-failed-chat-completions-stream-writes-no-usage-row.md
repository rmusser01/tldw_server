---
id: TASK-13953
title: A dropped or failed /chat/completions stream writes no usage row
status: To Do
labels:
- quotas
- backend
- high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13951 (verified 2026-10-26 with a throwaway test: a client dropped after the first content chunk left no llm_usage_log row and one rg_refund_cb(cancelled=True) call). execute_streaming_call's save_callback in chat_service.py is the only recorder of a /chat/completions stream (the call runs with records_itself=True), and StreamingResponseHandler calls it only when the stream completes without cancellation or error (streaming_utils.py, the save gate before 'Saved streaming response'). A stream the client drops, stops, or that fails after output starts therefore writes no usage row, and _finalize_stream refunds the Resource Governor reservation (rg_refund_cb commits 0 tokens). ADR-067 says the provider-level recorder records a stream once however it ends; the chat pipeline does not, and ADR-067 says so since TASK-13951. This is a per-user cost-limit bypass, not just an accounting gap: a client can read a whole reply, close the connection on finish_reason before the save, and the reply never counts toward the per-user LLM cost limits. Also, the stream estimate counts only the reply text: tool-call arguments and reasoning are not counted (a tool-call-only reply estimates 0 completion tokens), while the provider-level recorder's _StreamUsage.chars counts them. OpenAI notes an interrupted stream may lack its usage chunk, so an early end needs the estimate. Related: the provider-level recorder (usage_accounting._record_stream and _record_response) records a usage block that counts only one side (prompt tokens only, say) as given, so the reply counts 0 completion tokens; the chat pipeline takes the missing side from its estimate since TASK-13951.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A /chat/completions stream that ends early (client drop, stop, provider error after output) writes one usage row: from the provider's usage chunk if it arrived, else from the prompt estimate and the output received, with a test
- [ ] #2 The Resource Governor and billing commit the recorded tokens for such a stream instead of refunding them, with a test
- [ ] #3 The stream estimate counts tool-call arguments and reasoning text as the provider-level recorder does, with a test
- [ ] #4 The provider-level recorder takes a side a usage block does not count from its estimate, as the chat pipeline does, with a test
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

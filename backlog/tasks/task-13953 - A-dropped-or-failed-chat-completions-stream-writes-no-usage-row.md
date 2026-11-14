---
id: TASK-13953
title: A dropped or failed /chat/completions stream writes no usage row
status: Done
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
- [x] #1 A /chat/completions stream that ends early (client drop, stop, provider error after output) writes one usage row: from the provider's usage chunk if it arrived, else from the prompt estimate and the output received, with a test
- [x] #2 The Resource Governor and billing commit the recorded tokens for such a stream instead of refunding them, with a test
- [x] #3 The stream estimate counts tool-call arguments and reasoning text as the provider-level recorder does, with a test
- [x] #4 The provider-level recorder takes a side a usage block does not count from its estimate, as the chat pipeline does, with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-11-03-stream-usage-on-early-end.md. Owner answers (2026-11-03): ADR-004 Change summary waiver covers this PR; Decision 2 (nothing received refunds; follow-up filed) accept and file the follow-up.
Follow-up for Decision 2's residual (charge the prompt of a drop before the first token): TASK-15420.
How: one recorder per stream in chat_service.execute_streaming_call (_stream_usage_recorder claims the row once; _record_stream_usage writes it and calls rg_commit_cb in a finally). Both callers hand the write to submit_accounting, never awaited: save_callback at its top, before the message save, so a client that closes during the save still writes the row; _finalize_stream on a client drop, a stop or a provider error once something was received. _finalize_stream refunds and writes nothing when nothing was received (no usage chunk, no characters), whatever the ending: the endpoint retries a failed attempt on the same reservation and its close reaches the pipeline as a cancel without output. StreamingResponseHandler passes its _StreamUsage (usage chunk + chars of content, reasoning, tool-call arguments, and bare text) to save_callback and, through create_streaming_response_with_timeout.finalize_once, to the finalize callback as stream_usage (provider_usage is gone). usage_accounting._record_response/_record_stream fill a one-sided usage block from the estimate and mark the row estimated only when the fill replaced a zero; a non-finite value is estimated too.
RED/GREEN (record the real counts): Task 1 3 failed, 1 passed -> 38 passed; Task 2 12 failed, 2 passed -> 95 passed on the three files, the endpoint retry pin 1 passed on the base and after, and the 12 ids that pinned the gap updated and passing (test_chat_service_fallback.py x6 incl. parametrised, test_streaming_utils.py x2, test_async_stream_task_capacity.py x2, the two renames). Base verification of TASK-13951's finding: a drop after the first content chunk wrote no row and refunded once (cancelled=True); a drop during the save wrote nothing either. Wider set: SQLite 1977 passed, 2 skipped, 36 errors -> 1994 passed, 2 skipped, 36 errors (36 mocker errors both sides), serial 88 passed -> 88 passed, PostgreSQL 138 passed -> 138 passed. tests/Docs+lint+CI+backlog-py 1615 passed. Shard coverage new_uncovered=0; test-quality ratchet OK; route-auth exact; OpenAPI unchanged. Function sizes: execute_streaming_call 1545 -> 1521, safe_stream_generator 774 -> 769, tracked_streaming_generator 296, create_streaming_response_with_timeout 255.
Decisions: a stream that ends before anything was received (error, drop or stop before the first token) refunds and writes no row, as ADR-067's recorder does for an unread stream; recording it would charge the prompt twice on the endpoint's in-request retry (OAuth refresh, fallback provider), and the pipeline cannot tell that retry's close from a client drop before the first token; TASK-15420 (filed here) tracks charging the latter's prompt. A failure before the stream object exists still records nothing and refunds. The row's status stays 200. Docs: ADR-067 (Streams; non-finite; one-sided block) + mirror, Docs/Operations/Usage_Quotas.md. Bandit: No issues identified (Medium 0, High 0).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A /chat/completions stream now writes exactly one llm_usage_log row however it ends once something was received: on completion from save_callback, on a client drop, a stop or a provider error from _finalize_stream, in both cases written detached so cancellation cannot cut it. The row is the provider's usage chunk when it arrived, else the prompt estimate plus the characters received (content, reasoning, tool-call arguments), and its tokens are committed to the Resource Governor and billing instead of refunded; an own-key stream keeps credential_source=user. A stream that ends before anything was received still refunds and writes no row, so the endpoint's retry of a failed attempt is charged once. The provider-level recorder takes a side a usage block does not count from its estimate. ADR-067 and Usage_Quotas updated.
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

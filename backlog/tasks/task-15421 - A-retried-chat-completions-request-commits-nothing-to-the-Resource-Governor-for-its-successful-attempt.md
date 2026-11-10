---
id: TASK-15421
title: A retried /chat/completions request commits nothing to the Resource Governor
  for its successful attempt
status: To Do
labels:
- quotas
- backend
dependencies:
- TASK-13953
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13953's reviews (plan V4; final review, declined-to-judge items). The /chat/completions endpoint retries a failed stream attempt (a forced OAuth refresh or a fallback provider, endpoints/chat.py around the attempt loop near :5807-5885 on 576feef820) on the SAME Resource Governor handle (_rg_handle_id) and billing enforcer. The failed attempt is refunded (TASK-13953: nothing received refunds; before it, every failure refunded), and the refund is gov.commit(handle, actuals={tokens: 0}); the in-memory governor pops the handle on its first commit (Resource_Governance/governor.py:861, InMemoryResourceGovernor.commit). The successful attempt's gov.commit is then a no-op, so the governor books 0 tokens for the reply it served. Billing (record_actual, apply_usage_delta), the durable cost-units ledger and the llm_usage_log row are additive per call and still count the reply, so the per-user cost limits are right; the gap is the governor's token windows (rate limits, budgets read from the governor), which under-count every retried reply. Similarly, a refund at a pre-stream failure inside execute_streaming_call's own fallback path makes the successful fallback's RG commit a no-op. Fix options: reserve a fresh handle per attempt (and refund the failed one), or skip the refund for an attempt the endpoint supersedes (TASK-15420 adds such a 'superseded' flag) so the one handle is committed once, by the attempt that served the reply.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A /chat/completions request whose first stream attempt fails before output and is retried (OAuth refresh, fallback provider) commits the successful attempt's tokens to the Resource Governor exactly once, with a test against the in-memory governor
- [ ] #2 The failed attempt books nothing in the governor, and billing and the usage row stay as they are (one row, the successful attempt's), with a test
- [ ] #3 The same holds for execute_streaming_call's own fallback after a pre-stream failure, with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Re-verified by TASK-15427 on cb70f0c5d1 (2026-11-07). Endpoint: a copy of test_streaming_oauth_retry_records_one_usage_row_for_the_successful_attempt with a recording MemoryResourceGovernor on app.state recorded reserve 309, commit(tokens=0) on the live handle (the failed attempt's rg_refund_cb, endpoints/chat.py:5676-5685, from _finalize_stream's nothing-received branch, Chat/chat_service.py:5583-5585), then commit(tokens=19) on the finalized handle (the retry's _build_streaming_commit_cb, chat.py:6575): a no-op (governor.py:790-792). The Redis governor behaves the same: RedisResourceGovernor.commit returns when the handle record is gone (governor_redis.py:2263-2271) and deletes it at the end of the first commit (:2404-2407). Measured on both governors (Redis through the in-memory stub client): reserve 500 tokens, commit 0, commit 300, and a 701-token reserve in the same minute is still allowed (refused without the first commit). The chat handle reserves tokens only (chat.py:4242-4247), so no stream lease is involved. AC #1's test should run against both governors (the Redis one through the in-memory stub client).
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

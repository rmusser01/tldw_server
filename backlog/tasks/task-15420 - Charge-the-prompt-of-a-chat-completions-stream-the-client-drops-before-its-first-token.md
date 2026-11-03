---
id: TASK-15420
title: Charge the prompt of a /chat/completions stream the client drops before its
  first token
status: To Do
labels:
- quotas
- backend
dependencies:
- TASK-13953
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13953 records a /chat/completions stream however it ends once something was received, but a stream that ends before the first token still refunds and writes no row: at the chat pipeline, the endpoint's close of a failed attempt it is about to retry (OAuth refresh, fallback provider) and a client drop before the first token are the same ending (cancelled, nothing received), and recording it would charge the retried prompt twice. So a client that sends a prompt and drops before the first token is never charged for it, while the operator pays the provider (a cost leak with no benefit to the client, bounded by the Resource Governor's request limits). Fix: the endpoint passes a 'superseded' flag to execute_streaming_call when it closes an attempt to retry it; _finalize_stream refunds only a superseded attempt and records the prompt estimate (completion 0) for every other nothing-received ending. See Docs/superpowers/plans/2026-11-03-stream-usage-on-early-end.md, Decision 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A /chat/completions stream the client drops or stops before its first token writes one usage row from the prompt estimate, and its Resource Governor reservation and billing commit those tokens, with a test
- [ ] #2 An attempt the endpoint closes to retry it (OAuth refresh, fallback provider) is still refunded and writes no row: test_streaming_oauth_retry_records_one_usage_row_for_the_successful_attempt keeps passing, and a fallback-provider retry pin is added
- [ ] #3 ADR-067 and Docs/Operations/Usage_Quotas.md say a drop before the first token is charged its prompt
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

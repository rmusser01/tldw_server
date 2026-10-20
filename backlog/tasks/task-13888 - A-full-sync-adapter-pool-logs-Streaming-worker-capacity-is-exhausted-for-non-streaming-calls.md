---
id: TASK-13888
title: A full sync adapter pool logs 'Streaming worker capacity is exhausted' for
  non-streaming calls
status: To Do
labels:
- backend
- claims
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
await_bounded_daemon_with_timeout (Chat/bounded_daemon.py near line 259) calls pool.start(invoke, name=name, released_event=released) without an exhaustion_message, so BoundedDaemonPool.start's streaming default (near line 135, 'Streaming worker capacity is exhausted') becomes the DaemonCapacityError message for every one of its 8 app callers when the pool is full. For the claims calls (claims_engine._run_bounded_claims_analyze, used by extract, verify, APS and, since TASK-13862, the FVA adjudicator) the log reads 'LLM assessment failed: Streaming worker capacity is exhausted', which an operator will not link to claims. Found by the clean-session review of PR #3287 (Minor 3). Suggested fix: pass exhaustion_message=f'{name} capacity is exhausted' from await_bounded_daemon_with_timeout. That is one line in shared code, so check whether any caller or test matches on the old text before changing it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A full pool reached through await_bounded_daemon_with_timeout raises DaemonCapacityError with a message naming the call (its name), with a test
- [ ] #2 Callers and tests that match on 'Streaming worker capacity is exhausted' for these calls are updated
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

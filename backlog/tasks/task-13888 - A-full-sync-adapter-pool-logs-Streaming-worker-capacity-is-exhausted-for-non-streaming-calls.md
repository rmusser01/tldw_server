---
id: TASK-13888
title: A full sync adapter pool logs 'Streaming worker capacity is exhausted' for
  non-streaming calls
status: Done
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
- [x] #1 A full pool reached through await_bounded_daemon_with_timeout raises DaemonCapacityError with a message naming the call (its name), with a test
- [x] #2 Callers and tests that match on 'Streaming worker capacity is exhausted' for these calls are updated
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-22-fva-endpoint-and-capacity-message.md. await_bounded_daemon_with_timeout passes exhaustion_message=f'{name} capacity is exhausted' to pool.start.
AC #2: git grep finds no caller or test matching 'Streaming worker capacity is exhausted' outside bounded_daemon.py and this task, so it is met without a change. What the change did break was a test pool whose start() override took no exhaustion_message: test_async_daemon_bridge_consumes_failure_when_cancelled_during_release_wait, the only one of 13 such overrides reached through await_bounded_daemon_with_timeout, then spun forever (Timeout >30 s with timeout_method=signal; the repo default, timeout_method=thread, ends the whole run). It now forwards **kwargs.
RED: test_daemon_drain_after_timeout_rejects_until_exit_then_raises 1 failed ('Streaming worker capacity is exhausted' != 'capacity-rejected capacity is exhausted'). GREEN: 17 passed. Sweep of the 35 test files that name the helper, the pools or their callers' helpers: 2300 passed, 4 skipped (1463 s at load ~30; the four skips as the plan lists). run_bounded_daemon_with_timeout also omits the message, but its only caller (mlx_provider) maps DaemonCapacityError to its own 'MLX busy' error, so it is unchanged. Bandit clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
await_bounded_daemon_with_timeout passes exhaustion_message=f'{name} capacity is exhausted', so a full pool names the rejected call (for example 'claims-provider-analyze capacity is exhausted') for all 8 app callers instead of the streaming default. One test pool override now forwards the argument.
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

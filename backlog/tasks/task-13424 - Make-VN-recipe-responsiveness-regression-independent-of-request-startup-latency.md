---
id: TASK-13424
title: Make VN recipe responsiveness regression independent of request startup latency
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Buddy repair PR CI fails the VN recipe-capture responsiveness regression when normal request startup exceeds its one-second wall-clock limit. Preserve the regression that detects synchronous capture on the event loop while making its result independent of unrelated startup latency.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The regression verifies event-loop progress while recipe capture remains blocked, and the request returns HTTP 202 after release.
- [x] #2 A controlled nonblocking startup delay exceeding one second does not fail the responsiveness regression.
- [x] #3 A controlled inline recipe-capture mutation fails the regression, and the targeted generation-jobs file passes.
- [x] #4 Scoped formatting, lint, security and diff checks are recorded, with existing baseline findings distinguished.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
## Implementation Plan

ADR required: no
ADR path: N/A
Reason: Test-only correction preserves the existing synchronous endpoint worker boundary, APIs, jobs and recipe storage.

1. Retain the CI failure and reproduce the false failure with a nonblocking 1.1-second request-start delay.
2. Replace the aggregate latency assertion with a gated capture/progress check and always await request cleanup.
3. Prove the updated test accepts delayed startup and rejects inline capture; run the targeted generation-jobs file and scoped static/security checks, obtain independent review, then publish on PR 3091.

CI failure: run 37132796224 job 111236179246 at 76af54dde6; elapsed startup was 1.001778062 seconds against a <1.0 assertion. Controlled async-delay reproduction fails at the same assertion with the unchanged offloaded endpoint. Private receipts: gap-verified-6-failed-76af54.log and generation-start-delay-red.log under /private/tmp/buddy-remaining-fixes-20261003. An initial basetemp under /private/tmp was rejected by the existing trusted database-root policy; rerun uses a unique directory under the approved system temp root, preserving that policy.
Implemented the test-only responsiveness correction in tldw_Server_API/tests/VN_Assets/test_generation_jobs.py. A thread-safe event notification now proves the asyncio test advances while recipe capture and its generation request remain blocked; watchdog timeouts only bound a failed test. Cleanup always releases capture and awaits the request before client teardown. HTTP 202 remains required.

Validation: the old assertion failed under controlled nonblocking 1.1-second startup delay; the updated test passes that same delay (1 passed in 3.82 seconds). Forcing the real endpoint inline on the loop fails at capture_unblocked (negative control, 1 failed in 8.14 seconds). All 128 tests in test_generation_jobs.py passed in 208.23 seconds (5 warnings); no full repository sweep. Ruff reports zero findings, and Bandit reports zero findings/errors with B101 excluded for pytest assertions on both baseline and current test files. Whole-file Black drift predates the change; the complete touched function matches Black on both baseline and final source. git diff --check passes. Independent review found no actionable findings. Private receipts: generation-start-delay-red.log, generation-start-delay-green.log, generation-inline-red.log, generation-file-green.log and vn-generation-* static reports under /private/tmp/buddy-remaining-fixes-20261003. No production source, dependency or workflow change; no new ADR required. Publishing through existing PR 3091.
Documentation follow-up before final publication: record the CI timing false-positive incident and barrier/inline-control evidence in backlog/docs/lessons-testing-evidence.md, then finalize the task with that file included. This lesson preserves the existing endpoint worker contract; no new ADR or production change.
Added the incident and controlled-delay/inline-capture evidence to backlog/docs/lessons-testing-evidence.md. Final touched scope is the generation-jobs regression, this testing lesson and this task record; production source remains unchanged. All acceptance criteria and definition-of-done checks are complete before final commit.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced an aggregate one-second request-start assertion with an actual event-loop progress check while VN recipe capture remains blocked. Delayed nonblocking startup passes, inline capture still fails, and all 128 targeted generation-job tests pass. Scoped static/security checks and independent review found no new findings; existing whole-file Black drift is unchanged.
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

---
id: TASK-15686
title: Isolate chat saturation testing from cold usage database setup
status: In Progress
labels:
- ci
- testing
- chat
references:
- https://github.com/rmusser01/tldw_server/pull/3612
- https://github.com/rmusser01/tldw_server/actions/runs/45662221863/job/139431175721
- tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py
- TASK-13950.25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Chat executors gate on PR #3612 timed out while releasing the successful caller in the non-stream saturation regression. Isolate the capacity test from cold usage database initialization.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preserve the capacity rejection and no-dispatch assertions and verify holder completion on a cold or simulated slow database path.
- [ ] #2 Verify the focused Chat executor tests and the new PR check.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. This change isolates a test dependency and preserves the chat execution and usage accounting contracts.
The hosted traceback reached the original saturation assertions and timed out while the successful holder waited for SQLite usage-schema initialization. A controlled stall in the same dependency reproduced the timeout before the fix. The existing usage-log test mock keeps the capacity, rejection, release and no-dispatch assertions unchanged. The controlled case passed after the fix. The entire fallback file passed 378 tests independently. The broader Chat unit selection passed 830 tests with 6 local PostgreSQL variants skipped. Ruff, whitespace and the Bandit baseline comparison passed with no new findings.
Renumbered on PR #3612 after the Quick Ingest task reached dev with the same number.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Isolated the Chat saturation test from unrelated usage-database initialization. Local cold-path regression and the full fallback suite passed. The updated PR check will provide hosted verification.
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

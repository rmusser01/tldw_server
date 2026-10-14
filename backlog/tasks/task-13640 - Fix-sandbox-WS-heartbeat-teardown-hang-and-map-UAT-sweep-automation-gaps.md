---
id: TASK-13640
title: Fix sandbox WS heartbeat teardown hang and map UAT sweep automation gaps
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recreation of the clobbered TASK-13408 (dev's 13408 is an unrelated 'Restore model discovery' task - same ID-collision class as 13407; this work is TASK-13640). Per the plan's Stage 5 and Testing_Known_Issues.md recommendation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Executed 2026-10-14: (1) WS teardown now awaits background-task cancellation with a deadline after the synchronous cancel+quota-release ordering is preserved (sandbox.py finally block); WS_SHUTDOWN_TIMEOUT_MS env override, 500ms test-mode / 5s prod defaults via _ws_shutdown_deadline_s(); 5 unit tests for the resolver + both heartbeat tests pass (7 passed). (2) Sweep coverage doc: X-01/X-02/S-01..S-06 rows now map to concrete owning backend suites with explicit manual:<reason> for residual UI/install surfaces.
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

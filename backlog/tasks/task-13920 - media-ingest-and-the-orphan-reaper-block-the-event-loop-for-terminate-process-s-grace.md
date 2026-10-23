---
id: TASK-13920
title: media_ingest and the orphan reaper block the event loop for terminate_process's
  grace
status: To Do
labels:
- workflows
- backend
- performance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
terminate_process (Workflows/subprocess_utils.py) sends SIGTERM, then time.sleep()s its 5 s grace before SIGKILL. Three callers run it on the event loop and stall every request for 5 s: media_ingest's cooperative-cancel branch and its timeout branch (Workflows/adapters/media/ingest.py, the two plain terminate_process(task) calls in the poll; the CancelledError branch already uses asyncio.to_thread), and WorkflowEngine._reap_orphans (engine.py, terminate_process(task, grace_ms=grace_ms)), which every SYNC start_run awaits before its first step. engine.cancel() also calls it synchronously for each recorded subprocess. Found while planning TASK-13887.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 media_ingest's cooperative-cancel and timeout branches terminate the download off the event loop, with a test that the loop keeps running during the grace
- [ ] #2 The orphan reaper and engine.cancel() terminate a step's subprocess off the event loop, with a test
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

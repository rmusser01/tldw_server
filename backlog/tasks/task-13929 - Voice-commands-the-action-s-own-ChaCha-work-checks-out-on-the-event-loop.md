---
id: TASK-13929
title: 'Voice commands: the action''s own ChaCha work checks out on the event loop'
status: To Do
labels:
- voice
- postgres
- performance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13896's review (M4). TASK-13896 moved the voice-assistant WebSocket's own ChaCha calls (session start, the router's command load, session save, analytics and external-connection lookup) off the event loop, and its AC 4 was narrowed to those. The action a command runs still does its own ChaCha work wherever its module does it: a SYNC workflow's notes step (Workflows/adapters/knowledge/crud.py run_notes_adapter) calls NotesInteropService synchronously on the loop with a per-call handle it closes, so it checks out on the event-loop thread; MCP modules use per-call handles closed on the same thread, not measured on or off the loop. Under a full content pool such a checkout stalls the whole server for up to pool_timeout. plan_fu2_workflows.md (TASK-13897) keeps the notes adapter on the loop. Measure each on PostgreSQL first, then move what checks out on the loop into asyncio.to_thread with its own operation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A voice command whose SYNC workflow runs a notes step takes no ChaCha checkout on the event-loop thread (PostgreSQL test), or the measurement shows it does not
- [ ] #2 The MCP modules a voice command can call are measured, and any that check out on the loop are moved or listed
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

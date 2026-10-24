---
id: TASK-13931
title: 'Persona session preferences: a turn''s save can write stale data over a live-control
  or second-connection write'
status: To Do
labels:
- persona
- bug
- concurrency
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the re-review of #3312 (R1). TASK-13896 moved the persona WebSocket's preference read and save into separate worker calls; #3312 serializes a turn's read-then-save with that connection's confirm_plan through a per-session asyncio.Lock (preference_locks in persona_stream), which covers one connection only. Other writers of persona_sessions.preferences_json can still land between a turn's read and its save: the live-control REST routes persona_live_session_focus and persona_live_session_stop (api/v1/endpoints/persona.py ~8059, ~8092), which write through Persona/live_control.py _update_preferences (:149, versioned, called by _update_live_control_preferences with re-read and retry; also the create idempotency key) and stop_live_session (:703, re-read, unversioned); and a second WebSocket connection to the same session (its confirm_plan). _persist_persona_session_preferences writes the turn's earlier read plus its patch over the whole preferences_json with no expected_version, so it writes stale persona_live_control metadata or rules back; the save is shielded (run_quiescent_operation), so it lands even after a Stop cancels the turn. On dev all of these ran synchronously on one event loop and could not interleave. Impact today is low: focus and idempotency lookups skip terminal sessions, a focus switch can leave two sessions marked focused until _reconcile_focused_sessions runs, and a turn's save writes only when its patch changes a value. Suggested fix: give _persist_persona_session_preferences the expected_version optimistic write with a re-read-and-merge retry that live_control._update_live_control_preferences already uses; that covers every writer, across connections and REST.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A turn's preference save no longer writes stale preferences_json over a write that landed between its read and its save (a live-control REST write or another connection's save): it merges onto the current row, retrying on a version conflict
- [ ] #2 A test interleaves a REST live-control Stop with a turn's preference save and shows the Stop's live-control state survives the turn's save
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

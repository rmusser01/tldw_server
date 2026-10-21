---
id: TASK-13896
title: 'Persona: run the 10 on-loop ChaCha calls inside async handlers off the event
  loop'
status: To Do
labels:
- persona
- postgres
- bug
dependencies:
- TASK-13861
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13861's third review. After its PR B (plan Task B5) every persona_stream call on the session's handle runs in its own chacha_operation, so no session keeps a checkout idle in a transaction. But each of the 13 synchronous calls now checks out synchronously on the event-loop thread at every call (getconn plus about 5 round trips). When other work holds the whole content pool (HTTP requests and every other content database share pg_pool_size = 20 in production), each such call blocks the whole server's event loop for up to pool_timeout (30 s by default) and then raises PoolTimeout. Measured by the re-review (pool of 10, 9 connections held elsewhere, one warm session): the B5-wrapped call raised PoolTimeout after 2.0 s, with a 2.01 s event-loop gap; a warm legacy call took 0.0 s. The 10 sites that sit directly in async handlers can move off the loop as plan Task B4 does, as await asyncio.to_thread(chacha_operation(independent=True)(fn), ...): persona.py :9417, :9950, :10069, :10138, :10611, :10828, :11069, :11114, :11291 and :11366 (origin/dev ad946dd687 line numbers, unchanged since 3ad31cba25; each is one line further down after PR B's import). The other three (:9019, :9091, :9128) sit in the sync helpers _resolve_persona_live_event_persona_id, _upsert_persona_live_session_summary_safe and _record_persona_live_voice_event_safe, whose callers would have to move them. Separately, and already so on dev before TASK-13861: in Skills/context_integration.handle_skill_tool_call, SkillsService.__init__ runs 4 table-setup statements (CREATE ... IF NOT EXISTS in _ensure_skill_registry_table, ChaChaNotes_DB.py:27766-27794 on ad946dd687) through backend.execute on the event-loop thread at every skill call, each with its own pool checkout and outside any operation; the re-review measured 4 MainThread checkouts per skill call on dev and with PR B. Under pool exhaustion those stall the loop the same way. (The lookup's own operation, added by PR B, takes its checkout on an executor thread.)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 10 calls in persona_stream's async handlers run through asyncio.to_thread with their own chacha_operation(independent=True), and the persona AST guard requires that form for them
- [ ] #2 A PostgreSQL test with the pool held by other work shows a persona call waiting for a connection without stalling the event loop
- [ ] #3 The skill tool call's SkillsService setup and lookup make no ChaCha checkout on the event-loop thread
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

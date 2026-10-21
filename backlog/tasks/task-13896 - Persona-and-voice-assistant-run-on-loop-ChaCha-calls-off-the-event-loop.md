---
id: TASK-13896
title: 'Persona and voice assistant: run on-loop ChaCha calls off the event loop'
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

Voice assistant (found by PR B's clean review, #3295). PR B runs the voice-assistant WebSocket's session start (voice_assistant.py:1104-1109 on PR B's branch: save_voice_session and registry.refresh_user_commands) and each command (router_instance.process_command at :1412, whose refresh_user_commands, save_voice_session and record_voice_command_event run directly on the loop, VoiceAssistant/router.py:192, :201, :296, :312) in their own chacha_operation(independent=True) on the event-loop thread, so each takes a pool checkout synchronously on the loop. Dev instead reused the loop thread's pinned legacy checkout, which is the leak TASK-13861 fixes. Measured by the review (pool of 3, pool_timeout 1.0 s, the pool held by other work): a warm user's new session took 0.00 s on dev; with PR B the loop was blocked 1.01 s, then psycopg_pool.PoolTimeout escaped websocket_voice_assistant (it is not in the caught tuples at :1110 or :1279), so the socket died. A warm user's command mid-session took 0.0 s on dev; with PR B the loop was blocked 1.0 s, then PoolTimeout, per ChaCha call group. With the default pool_timeout of 30 s and the router's 2-3 call groups per command, that is up to about 60-90 s of whole-server stall per command. The command's checkout is also held, idle in a transaction, through intent parsing, the LLM chat or MCP action, and a SYNC workflow (60 s default timeout). Fix shape: run those sync calls through asyncio.to_thread(chacha_operation(independent=True)(fn), ...) as plan Task B4 does, give only the router's ChaCha call groups their operations rather than the whole command, and catch psycopg pool errors at session start so that a full pool degrades the session instead of killing the socket.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 10 calls in persona_stream's async handlers run through asyncio.to_thread with their own chacha_operation(independent=True), and the persona AST guard requires that form for them
- [ ] #2 A PostgreSQL test with the pool held by other work shows a persona call waiting for a connection without stalling the event loop
- [ ] #3 The skill tool call's SkillsService setup and lookup make no ChaCha checkout on the event-loop thread
- [ ] #4 The voice-assistant WebSocket's session start and command handling make no ChaCha checkout on the event-loop thread, and a command holds no checkout through its LLM call, MCP action or SYNC workflow
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

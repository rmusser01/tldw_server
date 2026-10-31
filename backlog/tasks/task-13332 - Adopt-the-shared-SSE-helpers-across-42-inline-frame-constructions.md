---
id: TASK-13332
title: Adopt the shared SSE helpers across 42 inline frame constructions
status: Done
assignee: []
created_date: '2026-09-22 04:58'
updated_date: '2026-09-23 23:42'
labels:
  - duplication
  - streaming
  - chat
dependencies: []
references:
  - 'tldw_Server_API/app/core/LLM_Calls/sse.py:41'
  - 'tldw_Server_API/app/core/Chat/streaming_utils.py:57'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ADR-025 makes single-terminal-[DONE] a binding provider contract, so this is contract knowledge, not style.

core/Chat carries 33 inline f"data: {json.dumps(...)}\n\n" constructions (6 in chat_service.py, 27 in streaming_utils.py) while core/LLM_Calls/sse.py is imported by 19 non-test modules elsewhere and by core/Chat ZERO times. sse_data() is byte-for-byte the f-string at 20+ sites and sse_done() is byte-for-byte all 5 DONE sentinels.

Drift already present: three implementations of is_done_line, one of them case-SENSITIVE (streaming_utils.py:1056, :1690); and _SSE_CONTROL_PREFIXES exists in two places UNDER ONE NAME WITH DIFFERENT VALUES (sse.py:19 vs streaming_utils.py:57).

Nine provider adapters additionally inline the same SSE loop while streaming.py:iter_sse_lines_requests has exactly ONE production user - and it is the best copy (decodes with errors="replace", converts mid-stream transport errors into an SSE error frame rather than raising through the generator, honours STREAM_PROVIDER_CONTROL_PASSTHRU).

Two judgement calls, kept separate: sse.py needs an sse_event(name, payload) for the event:-prefixed frames, and reconciling the two _SSE_CONTROL_PREFIXES is a behaviour decision.

Source: synthesis F32
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 sse_data and sse_done adopted at the mechanical sites
- [x] #2 One is_done_line implementation, case-insensitivity decided explicitly
- [x] #3 The two _SSE_CONTROL_PREFIXES reconciled in a separate change
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Evidence:
- c53f7c5e2b: 50 inline frames -> sse_data/sse_event/sse_done (core/Chat streaming_utils 25, chat_service 6; endpoints character_chat_sessions 15, chat.py 4, prompt_studio_websocket 4). New sse_event(name, payload). All byte-identical (default json.dumps args, stdlib json in every file); tests/LLM_Calls/test_sse_helpers.py pins helper == old f-string. Left inline (non-default bytes, baselined): chat.py 2x separators=(',',':'), anthropic_messages ensure_ascii event frame, character_chat_sessions unterminated send_raw_sse_line. Ratchet tests/lint/test_no_inline_sse_frames.py (per-file baseline, fails on growth and on stale baseline); red on base 4a84d02b55 (chat_service 6, streaming_utils 23, character_chat_sessions 16...), green now.
- 5452c743b7: is_done_line is the single detector: case-insensitive (decided), any spacing after data:, BOM/zero-width tolerant. streaming_utils (2 sites) and chat_service (2 sites) route through it. Real bug fixed: provider 'data: [done]' was forwarded verbatim next to our DONE (double terminal frame); 'data:[DONE]' slipped past iter_sse_lines_*. tests/Chat/unit/test_done_sentinel_detection.py: 5 fail on old code, 14/14 pass now.
- 6c422a4535: prefixes reconciled with no behaviour change: sse.SSE_CONTROL_FIELD_PREFIXES exported; streaming_utils derives _ALWAYS_SSE_CONTROL_PREFIXES / _FRAMED_ONLY_SSE_CONTROL_PREFIXES from it (framed-only id:/retry: split is intentional, pinned by test_raw_non_sse_control_prefix_is_assistant_content); drift test added.
- Suites Chat, LLM_Calls, Streaming, lint, Character_Chat_NEW, prompt_studio: before 16 failed/4972 passed, after 16 failed/4998 passed (+26 new tests); FAILED sets identical (pre-existing strict_filter/ollama etc).
- bandit -ll on 6 touched source files: no findings.
- Not done (out of AC scope, behaviour-changing): migrating the nine provider adapters' inline SSE loops onto streaming.iter_sse_lines_requests; other DONE detectors outside core/Chat (endpoints, audio, RAG, workflows) still inline.

Count correction: 54 sites total (50 f-string/DONE literals, which the c53f7c5e2b message counts, plus 4 ensure_sse_line(f"data: {json.dumps(payload)}") in character_chat_sessions that are also byte-identical to sse_data(payload)).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adopted sse_data/sse_event/sse_done at 50 mechanical sites (byte-identical) with a lint ratchet; unified provider DONE detection in a case-insensitive is_done_line, fixing a double-terminal-DONE bug; reconciled the two _SSE_CONTROL_PREFIXES without behaviour change. Adapter SSE-loop migration left for a follow-up.
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

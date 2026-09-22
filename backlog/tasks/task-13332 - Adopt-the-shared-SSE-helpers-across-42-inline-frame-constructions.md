---
id: TASK-13332
title: Adopt the shared SSE helpers across 42 inline frame constructions
status: To Do
assignee: []
created_date: '2026-09-22 04:58'
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
- [ ] #1 sse_data and sse_done adopted at the mechanical sites
- [ ] #2 One is_done_line implementation, case-insensitivity decided explicitly
- [ ] #3 The two _SSE_CONTROL_PREFIXES reconciled in a separate change
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

---
id: TASK-13370
title: Route the remaining ~15 SSE DONE checks through is_done_line
status: Done
assignee: []
created_date: '2026-09-23 23:44'
updated_date: '2026-09-24 02:20'
labels:
  - tech-debt
  - streaming
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13332 made streaming.is_done_line the single, case/spacing/BOM-tolerant DONE detector and moved core/Chat onto it (fixing duplicate terminal frames when a provider sends 'data: [done]' or 'data:[DONE]'). About 15 checks elsewhere still hand-roll the comparison: endpoints character_chat_sessions, chat_documents, messages, audio_streaming; RAG; Workflows; LlamaCpp_Handler; anthropic_messages; google_adapter. Separately, the nine provider adapters keep their own SSE loops instead of streaming.iter_sse_lines_requests; migrating them changes error-frame and decode (errors='replace') behaviour, so treat that as its own step.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every provider-stream DONE check uses is_done_line
- [x] #2 A test per migrated path feeds 'data: [done]' and asserts exactly one terminal frame
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-09-23 (79ea543818): DONE checks in audio_streaming, character_chat_sessions, chat, chat_documents, Audio/Realtime default_pipeline, anthropic_messages, google_adapter, LlamaCpp_Handler, RAG generation, Workflows llm adapter route through is_done_line; messages.py keeps its parsed-event comparison (already case-insensitive, compares joined data not a raw line). Tests: 10 new/changed cases fail on the pre-fix source, 133 pass now. Chat, LLM_Calls, Local_LLM, Workflows, Audio: identical failure sets before/after (54), +27 passes. Adapter-loop migration split to TASK-13373 (the agent's commit originally mis-cited TASK-13371, corrected on cherry-pick).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Every provider-stream DONE check uses is_done_line; lower-case and no-space DONE no longer produce duplicate terminal frames.
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

---
id: TASK-13370
title: Route the remaining ~15 SSE DONE checks through is_done_line
status: To Do
assignee: []
created_date: '2026-09-23 23:44'
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
- [ ] #1 Every provider-stream DONE check uses is_done_line
- [ ] #2 A test per migrated path feeds 'data: [done]' and asserts exactly one terminal frame
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

---
id: TASK-13371
title: RAG analytics maps 1-star feedback to 0.2 instead of 0.0
status: To Do
assignee: []
created_date: '2026-09-24 00:00'
labels:
  - evaluations
  - rag
  - consistency
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
core/RAG/rag_service/analytics_system.py:959,968 normalises user feedback stars by stars/5, so the worst rating stores as 0.2 while Evaluations now maps every judge scale linearly with min->0 (TASK-13328, scoring.py). Changing it shifts the stored analytics trend series, so decide: migrate (rewrite or version the history) or keep stars/5 and document why analytics differs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Decision recorded: linear min->0 mapping via scoring.py, or documented exception
- [ ] #2 If migrated, stored history is versioned or rewritten so trends stay comparable
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

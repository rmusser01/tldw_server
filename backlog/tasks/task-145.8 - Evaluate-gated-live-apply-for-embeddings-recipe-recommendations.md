---
id: TASK-145.8
title: Evaluate gated live apply for embeddings recipe recommendations
status: To Do
assignee: []
created_date: '2026-05-09 06:15'
labels:
  - evals
  - backend
  - frontend
dependencies: []
references:
  - >-
    Docs/superpowers/plans/2026-05-09-embeddings-rag-recipe-webui-implementation-plan.md
parent_task_id: TASK-145
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up for Task 7: only implement live config mutation for embeddings recipe recommendations after explicit approval. Scope includes POST apply endpoint, config override safety checks, audit metadata, frontend apply hook/button, and focused tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Implementation is started only after explicit approval for live RAG/embeddings config mutation.
- [ ] #2 Endpoint writes only [Embeddings] embedding_provider and embedding_model through setup_manager with backup/audit metadata.
- [ ] #3 Mutation refuses to apply when environment variables override the effective Embeddings provider/model values.
- [ ] #4 Frontend shows live apply only when server preview returns apply_available=true and confirmation matches proposed provider/model.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Keep this as a gated follow-up. Start with backend failing tests for the apply endpoint and env override refusal, then add service/hook/UI only after backend contract is stable.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

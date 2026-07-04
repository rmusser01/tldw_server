---
id: TASK-12142
title: Implement Embeddings API inline workflow facade
status: To Do
assignee: []
created_date: '2026-07-04 01:19'
labels:
  - embeddings
  - implementation
  - workflow
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-03-embeddings-workflow-architecture-design.md
  - >-
    Docs/superpowers/plans/2026-07-03-embeddings-api-inline-workflow-facade-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the Stage 1 implementation plan for the canonical Embeddings workflow architecture. Scope: add workflow type contracts, no-op/in-memory trace collectors, inline workflow runner, endpoint pre-execute RG boundary hook, feature-flagged endpoint integration, tests, verification, and Bandit. No durable Jobs runner, schema changes, media/vector-store migration, or public API trace exposure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Workflow type contracts and safe bounded trace collectors are implemented with isolated tests.
- [ ] #2 Inline workflow runner wraps the existing EmbeddingRequestOrchestrator and preserves pre-execute RG reservation ordering.
- [ ] #3 Feature-flagged endpoint path uses the inline runner without changing public response behavior, headers, metrics, logs, schemas, or legacy shims.
- [ ] #4 Focused workflow, orchestrator, endpoint parity, compile, Bandit, and diff checks pass.
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

---
id: TASK-12973
title: Extract concrete API workflow steps for Embeddings
status: In Progress
assignee: []
created_date: '2026-07-19 02:31'
updated_date: '2026-07-26 05:19'
labels:
  - embeddings
  - workflow
  - architecture
  - refactor
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2733'
documentation:
  - Docs/superpowers/specs/2026-07-03-embeddings-workflow-architecture-design.md
  - >-
    Docs/superpowers/specs/2026-07-18-embeddings-workflow-stage2-concrete-api-steps-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stage 2 of the approved Embeddings workflow architecture. Extract preparation, provider-attempt, fallback, result, and inline-runner responsibilities from EmbeddingRequestOrchestrator through five sequential child PRs. Preserve API behavior and leave durable persistence, retries, pause/resume, and Jobs execution for Stage 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The approved Stage 2 design is documented and reviewed before implementation planning.
- [ ] #2 Five sequential child tasks cover characterization, pure contracts, provider attempts, fallback coordination, and runner/facade integration.
- [ ] #3 EmbeddingRequestOrchestrator ends Stage 2 as a delegation and compatibility facade with no cache, provider, fallback, or vector-postprocessing decisions.
- [ ] #4 Workflow-enabled and legacy endpoint behavior remain equivalent across the validated parity matrix.
- [ ] #5 HTTP header mapping is owned by the endpoint boundary while the compatibility facade preserves the legacy internal result contract.
- [ ] #6 No durable workflow persistence or Jobs integration is introduced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Deliver through child tasks 2A-2E in dependency order. Each child must pass focused parity tests and remain independently reviewable.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-07-18: Created isolated branch codex/embeddings-workflow-stage2a-contracts from origin/dev at merge commit 29acaca8c7 (PR #2733). Created child tasks TASK-12973.1 through TASK-12973.5 with sequential dependencies. Baseline verification passed: 93 focused Embeddings orchestration, workflow, runner, and endpoint-parity tests. Written Stage 2 spec completed and self-reviewed; user review is pending before implementation planning.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 2 is in progress; child-task summaries remain authoritative until the parent is complete.
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

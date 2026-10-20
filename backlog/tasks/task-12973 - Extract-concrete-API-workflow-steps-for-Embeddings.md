---
id: TASK-12973
title: Extract concrete API workflow steps for Embeddings
status: Done
assignee: []
created_date: 2026-07-19 02:31
updated_date: 2026-07-26 19:35
labels:
- embeddings
- workflow
- architecture
- refactor
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/pull/2733
- https://github.com/rmusser01/tldw_server/pull/2762
documentation:
- Docs/superpowers/specs/2026-07-03-embeddings-workflow-architecture-design.md
- Docs/superpowers/specs/2026-07-18-embeddings-workflow-stage2-concrete-api-steps-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stage 2 of the approved Embeddings workflow architecture. Extract preparation, provider-attempt, fallback, result, and inline-runner responsibilities from EmbeddingRequestOrchestrator through five sequential child PRs. Preserve API behavior and leave durable persistence, retries, pause/resume, and Jobs execution for Stage 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The approved Stage 2 design is documented and reviewed before implementation planning.
- [x] #2 Five sequential child tasks cover characterization, pure contracts, provider attempts, fallback coordination, and runner/facade integration.
- [x] #3 EmbeddingRequestOrchestrator ends Stage 2 as a delegation and compatibility facade with no cache, provider, fallback, or vector-postprocessing decisions.
- [x] #4 Workflow-enabled and legacy endpoint behavior remain equivalent across the validated parity matrix.
- [x] #5 HTTP header mapping is owned by the endpoint boundary while the compatibility facade preserves the legacy internal result contract.
- [x] #6 No durable workflow persistence or Jobs integration is introduced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Deliver through child tasks 2A-2E in dependency order. Each child must pass focused parity tests and remain independently reviewable.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-18: Created child tasks TASK-12973.1 through TASK-12973.5 with sequential dependencies after PR #2733 established the Stage 1 workflow boundary.

2026-07-23 to 2026-07-26: The Stage 2 concrete-steps design completed iterative user review, written-spec review, specification review, and quality review. The user approved the design and Stage 2A implementation plan. TASK-12973.1 characterization is complete in PR #2762; Stage 2B through Stage 2E remain pending and continue in dependency order.
2026-10-02 verified closeout retained during 2026-10-20 rebase: Stage 2A #2762 (1f456e5e0f), Stage 2C #2767 (f15365c7bb), Stage 2D #2774 (5821c9fd54), and Stage 2E #2790 (e4a1cc3afe) merged. Stage 2B contracts are present in the later merged implementation; no separate merge claim for #2766. Historical Stage 2E evidence: 356 focused tests including 145 endpoint parity tests and scoped Bandit zero findings/errors; child records retain details. Fresh Oct 20 runner/type/completion baseline: 63 passed, 9 PostgreSQL fixture skips, 144 warnings; not full parity certification. Bandit not applicable to tracking-only changes. ADR assessment: no new ADR, recording already merged work. Stage 3 implemented separately under TASK-13900.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 2 is complete through merged Stage 2E (#2790). Preparation, single-provider attempts, fallback coordination and immutable canonical outcomes are extracted; EmbeddingRequestOrchestrator remains a compatibility facade. Endpoint-owned accounting, credentials, cancellation and rollback are unchanged. Stage 3 durability is separate under TASK-13900.
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

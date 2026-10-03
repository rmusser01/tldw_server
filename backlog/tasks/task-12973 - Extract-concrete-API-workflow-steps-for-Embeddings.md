---
id: TASK-12973
title: Extract concrete API workflow steps for Embeddings
status: Done
assignee: []
created_date: '2026-07-19 02:31'
updated_date: '2026-10-03 00:29'
labels:
  - embeddings
  - workflow
  - architecture
  - refactor
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2733'
  - 'https://github.com/rmusser01/tldw_server/pull/2762'
  - 'https://github.com/rmusser01/tldw_server/pull/2767'
  - 'https://github.com/rmusser01/tldw_server/pull/2774'
  - 'https://github.com/rmusser01/tldw_server/pull/2790'
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
<<<<<<< HEAD
2026-10-20: Retained current-dev canonical task history while rebasing the reviewed Stage 3 design/planning branch. Historical October 2 closeout checked merged Stage 2 runner/facade and child records; Stage 3 remains a separate dormant capability implementation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
=======

2026-10-02 tracking closeout: inspected current merged runner/facade and child completion records on origin/dev 9958110df2. Verified ancestry for Stage 2A #2762 (1f456e5e0f), Stage 2C #2767 (f15365c7bb), Stage 2D #2774 (5821c9fd54), and Stage 2E #2790 (e4a1cc3afe). Stage 2B contracts are present in the later merged implementation; no separate merge claim is made for #2766. Corrected the stale Stage 2C status after checking its completed criteria. Historical Stage 2E verification: 356 focused tests including 145 endpoint parity tests and scoped Bandit with 0 findings/0 errors; existing warnings/Bandit skips remain in the child record. Fresh runner/type baseline: 48 passed, 8 warnings in 3.34s; not a full parity rerun. Bandit not applicable to these tracking-only Markdown changes. ADR check: ADR required: no; closing an approved already merged stage changes no architecture rule. Stage 3 durability and Jobs integration remain separate. No closeout blockers. MCP could not resolve this parent; used official Backlog CLI fallback.
>>>>>>> 6443ff6d0f (docs(embeddings): close Stage 2 and track Stage 3 design (TASK-13421))

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 2 is complete through merged Stage 2E (#2790). Ordered preparation, single-provider attempts, fallback coordination, immutable canonical outcomes, and endpoint result mapping are extracted; EmbeddingRequestOrchestrator is a compatibility/delegation facade. Historical endpoint parity verification preserves accounting, metrics, credential handling, cancellation cleanup, and feature-flag rollback. Stage 2 introduces no durable workflow store or Jobs runner. All five child tasks are complete; their detailed verification records remain authoritative. The tracking-only closeout also passed 48 runner/type baseline tests, not a fresh full parity suite.
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

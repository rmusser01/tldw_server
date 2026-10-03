---
id: TASK-13421
title: Design Embeddings Stage 3 durable workflow store and Jobs runner
status: In Progress
assignee: []
created_date: '2026-10-03 00:32'
labels:
  - embeddings
  - workflow
  - architecture
  - stage-3
  - design
dependencies:
  - TASK-12973
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2790'
documentation:
  - Docs/superpowers/specs/2026-07-03-embeddings-workflow-architecture-design.md
  - >-
    Docs/superpowers/specs/2026-07-18-embeddings-workflow-stage2-concrete-api-steps-design.md
  - Docs/ADR/003-jobs-vs-scheduler-default.md
  - Docs/ADR/022-embeddings-api-and-media-pipeline.md
  - Docs/ADR/051-retry-backoff-schedules.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the dormant Stage 3 Embeddings durability foundation after merged Stage 2. Deliver an approved spec and ADR assessment, not production wiring. Scope: Embeddings-owned persistence on the configured SQLite/PostgreSQL Jobs backend; immutable reference-only manifests; owner-scoped generation/item sink receipts; bounded batches, attempts, and typed reconciliation operations; additive domain-neutral prepared Jobs capability; injectable resolver, sink, and accounting contracts. Exclude public/admin endpoints, endpoint/Redis media migration, worker/lifecycle registration, production billing hookup, and legacy behavior changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Document the approved six-table owner-scoped persistence model, immutable manifests, pinned embedding-space generations, and named CAS repository transitions.
- [ ] #2 Review and approve admission, execution, lease fencing, normal batch continuation, retry/fallback, cancellation, reconciliation, accounting, and retention.
- [ ] #3 Specify backend security/concurrency, migration capability gates, crash-window tests, bounded resources, and unchanged-production-path verification.
- [ ] #4 Write and self-review the complete spec and required Proposed ADR; resolve placeholders, contradictions, and ambiguous guarantees.
- [ ] #5 Obtain requester approval of the written spec before implementation planning.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Explore merged Jobs/Embeddings and governing ADRs; clarify scope and compare approaches (complete).
2. Review design sections: architecture and revised persistence (approved); execution/recovery and verification (remaining).
3. Write fully approved spec and Proposed ADR, then self-review.
4. Commit artifacts and request written-spec approval.
5. After written-spec approval, hand off to writing-plans; no implementation in this design task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-10-02: Worktree .worktrees/embeddings-workflow-stage3-design; branch codex/embeddings-workflow-stage3-design; base origin/dev 9958110df2. Searched existing Embeddings/Stage 3 records; no matching design task found. Stage 2 parent and stale Stage 2C tracking corrected after merged-source/historical-verification review. Fresh runner/type baseline: 48 passed, 8 warnings in 3.34s.
Approved scope: dormant internal foundation; one Jobs root and bounded deterministic batches; at-least-once provider execution; generation/item-idempotent sink receipts; reference-only manifests with immutable revisions; generations pin provider/model/dimensions/policy plus trusted embedding-space fingerprint; fallback identity changes supersede the generation and restart item zero; cooperative cancellation at batch checkpoints; Jobs-only durable quarantine; per-batch accounting hooks fake-tested without production billing; Jobs active/archive UUID lifetime without active-Jobs foreign key.
Approved tables: embedding_workflows, embedding_workflow_items, embedding_workflow_generations, embedding_workflow_outputs, embedding_workflow_attempts, embedding_workflow_operations. Typed transactional outbox separates Jobs transitions, accounting, and superseded-output cleanup. Exact non-null owner on every table, composite owner constraints, PostgreSQL FORCE RLS, explicit SQLite owner scope, SQL under DB_Management, Embeddings-owned migrations, fail-closed capability checks. No raw inputs, vectors, credentials, or unbounded exceptions in durable state.
Verified existing prepared Jobs APIs enforce canonical Admin Webhook identity: require an additive domain-neutral capability while preserving existing behavior. Successful batch continuation must not consume failure retries. Execution/recovery and testing sections remain to be approved; no production code changes.
ADR check: ADR required: yes. This creates durable persistence and Jobs/domain ownership rules. Link ADR-003, ADR-022, ADR-051; draft a new Proposed ADR with the next free number when writing the fully reviewed spec. ADR-022 remains governing for unchanged legacy paths.
Bandit not applicable to tracking-only Markdown changes. MCP cannot resolve TASK-12973 (including as dependency); official CLI fallback used after confirming the failed MCP create produced no task.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

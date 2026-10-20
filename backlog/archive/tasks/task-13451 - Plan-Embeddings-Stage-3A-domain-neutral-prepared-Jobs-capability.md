---
id: TASK-13451
title: Plan Embeddings Stage 3A domain-neutral prepared Jobs capability
status: In Progress
created_date: 2026-10-04 23:06
dependencies:
- TASK-13450
labels:
- embeddings
- workflow
- stage-3a
- jobs
- planning
priority: high
documentation:
- Docs/superpowers/specs/2026-10-02-embeddings-workflow-stage3-durable-store-jobs-runner-design.md
- Docs/ADR/060-embeddings-durable-workflow-foundation.md
- Docs/superpowers/plans/2026-10-04-embeddings-workflow-stage3a-prepared-jobs-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-10-04-embeddings-workflow-stage3a-prepared-jobs-implementation-plan.md
updated_date: 2026-10-04 23:28
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and self-review an implementation-ready Stage 3A plan against the approved Stage 3 Embeddings spec. Scope: exact owner/root/control admission and lookup; transaction-enlisted live lease fencing; lease-independent effects and retained acknowledgement proofs; bounded retry-neutral expiry/no-progress recovery; SQLite/PostgreSQL parity, role isolation, compatibility and dormancy. Planning/documentation only. Stage 3B repository/global capacity, Stage 3C runner/trace mapping, and Stage 3D whole-foundation certification receive separate subsequent plans.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Map existing Jobs boundaries and define focused production/test file ownership without relaxing legacy Admin Webhook contracts.
- [x] #2 Specify exact interfaces and TDD steps for admission, live transaction fencing, retained logical-effect receipts, bounded no-progress recovery, and backend/security compatibility.
- [x] #3 Trace Stage 3A spec requirements to tasks and explicitly hand off Stage 3B/3C/3D requirements rather than claiming full-foundation coverage.
- [x] #4 Self-review and verify plan links, paths, interface consistency, failure cases, and documentation-only scope; record evidence and commit.
- [ ] #5 Present the plan for requester review and execution-method selection before runtime implementation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read existing and current-dev Jobs/fixture/authority boundaries (complete).
2. Write five-stage Stage 3A file map, exact interfaces, TDD/red-green/commit gates and subsequent-unit handoffs (complete).
3. Self-review sequencing, facts/lease/role invariants, failure focus, scope and signatures (complete).
4. Validate docs/snippets/path references/baselines, commit and request plan review/execution choice (verification complete; handoff pending).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-04: Written spec approved; requester said continue. Reuse .worktrees/embeddings-workflow-stage3-design. Local origin/dev 75ab224081bf140ef52017c1a9b0a04f6878d488 contains newer Jobs completion UUID atomicity fixes; the implementation preflight must use current dev without undoing those fixes. Read-only inventory across 325 distinct local/remote ref tips found maximum task ID 13449 before reserving TASK-13450/13451. Original design TASK-13421 collides with unrelated upstream task; canonical design record is now TASK-13450, with original history preserved. ADR check: ADR required yes, already covered by approved ADR-060; no additional durable decision. No code, migration, PR, rebase, or merge in this planning turn.
2026-10-04: Wrote the Stage 3A implementation plan with five staged TDD deliverables: frozen contracts/explicit support schema; exact admission plus enlisted live fencing; lease-independent effects/retained releaseable proofs; pre-handler acquisition episodes and bounded retry-neutral no-progress recovery; backend/RLS/privacy/legacy/dormancy certification. All implementation stages remain Not Started. Concrete follow-on handoffs cover Stage 3B domain/global reservations and Stage 3C trace mapping/adapters; Stage 3D certifies the full foundation. Self-review corrected fixture install sequencing, removed an early test's dependency on later progress accounting, specified separate exhaustion failure keys rather than rewriting continuation facts, and required actual PostgreSQL clock_timestamp-based lease revalidation/fail-closed RLS. Verification: 67 links, five stages/tasks, six Python snippets AST-parse, 51 exact production/test/doc path references resolve in current/dev trees or declared creations, placeholders/coverage/approval gates and canonical task sections checked. One verification script initially exceeded Node's default output buffer when listing the entire repository; narrowed the inventory to referenced scopes and reran successfully. Fresh unchanged-code baselines: 48 inline runner/type tests passed (108 warnings, 1.20s); 6 selected SQLite lease-policy tests passed (29 deselected, 24 warnings, 1.08s). git diff --check passed. These are plan/baseline checks, not evidence that planned Stage 3A code or PostgreSQL role tests already pass. Bandit N/A for Markdown-only scope; mandatory changed-production-scope Bandit commands are in the execution plan. Requester plan review and execution choice pending; no runtime/test code, schema execution, rebase, PR, or merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

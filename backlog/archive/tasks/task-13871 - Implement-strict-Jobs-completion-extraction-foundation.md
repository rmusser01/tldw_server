---
id: TASK-13871
title: Implement strict Jobs completion extraction foundation
status: In Progress
created_date: 2026-10-20 16:03
dependencies:
- TASK-13421
documentation:
- Docs/superpowers/specs/2026-10-04-jobs-completion-extraction-design.md
- Docs/superpowers/plans/2026-10-20-jobs-completion-foundation-implementation-plan.md
- Docs/ADR/058-jobs-completion-row-identity.md
- Docs/ADR/078-jobs-completion-persistence-boundary.md
updated_date: 2026-10-20 16:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver PR 1 only of the human-approved Jobs completion persistence extraction: paired characterization of the current completion contract, shallow completion-only contracts, and mechanical shared-helper/exception-policy relocation into DB_Management. Keep complete_job control flow, persistence SQL, SLA staging, explicit commits, observer preparation/dispatch and public behavior unchanged except helper imports. Backend routing is a separate task and plan after foundation merge. Current phase is writing the implementation plan; the human approved the revised written spec on 2026-10-20, not implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Human reviews the foundation-only implementation plan and selects or confirms execution before runtime/test implementation begins.
- [ ] #2 Paired SQLite and required real PostgreSQL characterization preserves lock/UUID/replay, serialization/policy timing, counters/outbox/SLA atomicity, callback binding/partial preparation and context/cleanup failure ordering.
- [ ] #3 Completion-only shallow command/result/effect/policy contracts introduce no input coercion, UUID normalization, eager serialization, optional-metric requirement or DB-to-Jobs runtime dependency.
- [ ] #4 Move the existing event insertion, missing-counter reconciliation, PostgreSQL commit guard and exception policy mechanically, retaining manager aliases and unchanged complete_job production routing.
- [ ] #5 Verify all affected lifecycle/admin helper callers, with no substantive PostgreSQL skips, full applicable Jobs suites and no new touched-scope lint/security findings; record source/SQL/import equivalence and fresh exact-head evidence.
- [ ] #6 Request a new human-written Change summary for the foundation PR and satisfy current-head review/required hosted gates; do not implement backend routing in this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-10-20-jobs-completion-foundation-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Planning-only start 2026-10-20. Read Python Backlog MCP workflow and explicit project path; search found only historical/assessment completion tasks, not a current foundation implementation task. ID 13871 exceeds 13870 from full paginated file lists of all open PRs and 13866 on current live dev. ADR-060 collided with open PRs #3191/#2821; our unpublished ADR is renumbered 078 (max other ADR 077), preserving accepted rationale. Current live dev 454974e812f8faf8532c931f3f7c8e0bf8d9445c; three unpublished docs commits rebased, resolving only canonical/published ADR index append conflicts by retaining all current rows and adding our decision as 078. AST comparison proves 16 current completion/policy/context/shared helper functions unchanged from approved c95e41fc baseline. Incoming admission/failure/WorkerSDK/PG-fixture work stays untouched; plan will use current Jobs fixture-owned shared PostgreSQL opt-in for new paired modules. Current runtime baseline is running; no new Python source/tests/dependencies/config/CI edits. ADR check: no additional decision required for this foundation; ADR-058 and ADR-078 govern unchanged lock/atomicity and completion DB ownership.
Foundation-only plan written for human review/execution selection; all four stages remain Not Started. It specifies paired real-backend characterization including the approved pre-commit binding correction, explicit shallow command/result/effect/Protocol interfaces, exact mechanical helper moves/manager aliases and driver-import preservation, every shared helper caller including admin missing-counter paths, and exact local/hosted qualification requirements. No routing/backend completion operations or facade adapter are implemented in this task phase. Fresh rebased baseline completed on source identical to live dev 454974e812: 86 passed, 227 existing warnings, no skips, exit 0; /tmp/jobs_completion_plan_baseline_20261020.xml. Full paginated open-PR file scan completed after retrying a GitHub connection timeout on the final four PRs; max task 13870 and ADR 077, so new TASK-13871 and our renumbered ADR-078 do not collide in that snapshot. Rebase conflict resolution retained all current canonical/public ADR rows; accepted Jobs ADR text differs from original only in its numeric title/path. Self-review covers spec-to-stage mapping, all five review-focus tests, consistent interfaces, no eager input normalization, optional partial effects, no production contract routing, and current shared fixture opt-in without global conftest migration. Docs/canonical/whitespace verification follows. Bandit/Ruff/compile N/A to this planning-only commit because no Python source/tests are changed.
Plan handoff verification (planning only): current Docs suite passed 212 tests, 439 project warnings, no skips/failures, including strict MkDocs using existing /tmp/task13215-mkdocs without dependency changes; XML /tmp/task13871_plan_docs.xml. Fresh existing completion/atomicity/event/RLS baseline passed 86 cases with real PG and no skips; XML /tmp/jobs_completion_plan_baseline_20261020.xml. Canonical/public ADR-078 and ADR index byte parity and refresh determinism pass; old accepted ADR body differs only by its renumbered title/path. Source/test/dependency/CI diff versus live 454974e812 is empty. Self-review completed with explicit test-marker/import-absence isolation and correct four-module touched-scope verification; all PR-1 spec requirements and five review-focus risks map to stages/tests. TASK-13421 assessment is Done after written-spec approval; this implementation task remains In Progress in plan-review phase with no implementation AC checked. No Python production/test files, facade adapters, backend operations, PR or merge were created. Final commit and human plan/execution review follow.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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

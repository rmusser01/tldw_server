---
id: TASK-13875
title: Implement strict Jobs completion extraction foundation
status: In Progress
created_date: 2026-10-20 16:48
dependencies:
- TASK-13421
documentation:
- Docs/superpowers/specs/2026-10-04-jobs-completion-extraction-design.md
- Docs/superpowers/plans/2026-10-20-jobs-completion-foundation-implementation-plan.md
- Docs/ADR/058-jobs-completion-row-identity.md
- Docs/ADR/078-jobs-completion-persistence-boundary.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute only the approved Jobs completion foundation plan: paired characterization, shallow completion-only contracts and mechanical shared-helper/exception-policy relocation into DB_Management, with production complete_job control flow/SQL/SLA/commit/observer routing unchanged. Human selected subagent-driven execution (option 1) on 2026-10-20. Backend routing, behavior fixes, broad caller migration and historical repair are excluded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Human reviews the foundation-only implementation plan and selects or confirms execution before runtime/test implementation begins.
- [ ] #2 Paired SQLite and required real PostgreSQL characterization preserves lock/UUID/replay, serialization/policy timing, counters/outbox/SLA atomicity, callback binding/partial preparation and context/cleanup failure ordering.
- [ ] #3 Completion-only shallow command/result/effect/policy contracts introduce no input coercion, UUID normalization, eager serialization, optional-metric requirement or DB-to-Jobs runtime dependency.
- [ ] #4 Move the existing event insertion, missing-counter reconciliation, PostgreSQL commit guard and exception policy mechanically, retaining manager aliases and unchanged complete_job production routing.
- [ ] #5 Verify all affected lifecycle/admin helper callers with no substantive PostgreSQL skips, full applicable Jobs suites and no new touched-scope lint/security findings; record source/SQL/import equivalence and fresh exact-head evidence.
- [ ] #6 Require a new human-written Change summary and current-head review/required hosted gates before a separately authorized foundation PR merge; do not implement backend routing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Execution approval and identity repair 2026-10-20: human selected option 1, subagent-driven implementation and independent per-stage review. Our provisional TASK-13871 (Jobs) was created while another branch added its own VN TASK-13871 through TASK-13874; that branch has now merged as dev 2e6ceab9fb97c5712533be670504a81a24525489. Fresh open-PR path snapshot max 13870, live dev max 13874; choose TASK-13875, preserve all incoming VN records and archive only our unpublished provisional Jobs record through official Backlog tooling. Do not mutate the VN task with old ID. The unchanged foundation plan is now tracked here. Plan last verified on 454974e812: 86 current SQLite/real PG baseline cases, 212 Docs including strict MkDocs; incoming 2e6ceab9 is VN documentation/task-only (11 paths), no runtime/test/ADR changes. Rebase tracking docs before implementation; plan-specific SDD ledger and fresh isolated implementer/reviewer per stage, no worker-spawned agents. No Python source/test implementation started yet. ADR check: no new decision; ADR-058/ADR-078 govern.
Human selected subagent-driven execution; plan/spec status and plan-specific preflight ledger updated. Provisional Jobs TASK-13871 archived through Python MCP before integrating the concurrent VN task with that ID; TASK-13875 is the active foundation record. No runtime/test implementation yet.
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

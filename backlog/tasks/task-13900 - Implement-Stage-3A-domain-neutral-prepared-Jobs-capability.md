---
id: TASK-13900
title: Implement Stage 3A domain-neutral prepared Jobs capability
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved Stage 3A plan using subagent-driven development. Plan: Docs/superpowers/plans/2026-10-04-embeddings-workflow-stage3a-prepared-jobs-implementation-plan.md. Spec: Docs/superpowers/specs/2026-10-02-embeddings-workflow-stage3-durable-store-jobs-runner-design.md. Governing accepted ADR-060; preserve ADR-022 and ADR-058. Dormant explicit-install Jobs substrate only; no Embeddings runtime wiring.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Immutable validated contracts and explicit secure schema on SQLite and PostgreSQL
- [x] #2 Exact atomic admission, replay, and enlisted live fencing
- [ ] #3 Retained effect proofs across transition, archive, and pruning
- [ ] #4 Bounded retry-neutral acquisition episodes and semantic progress
- [ ] #5 Backend/security/property/compatibility tests and scoped Bandit verification
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-20: Requester selected subagent-driven execution (option 1). Isolated design branch rebased onto origin/dev df7c290fbf; current dev completion fences preserved. Open PR inventory max task ID 13870; no neutral workflow capability on dev. Relevant open PRs #2862 and #2738 do not implement this capability. Rebase encountered canonical task-note and ADR index conflicts; retaining upstream accepted ADRs and canonicalizing workstream tracking. Historical TASK-13450/13451 collide with unrelated dev records and must not be used for new mutations.
Preflight verification correction: the initial 9 PostgreSQL skips came from the RUN_JOBS CI gate, not fixture unavailability. With RUN_JOBS=1 and local fixture access, all9 PostgreSQL completion atomicity cases passed (15 deselected,30 warnings,3.92s). Existing SQLite admission/idempotency/prepared-disposition/reclaim/prune and WorkerSDK suites:273 passed,772 warnings,16.11s. Required-workflow and task-format ratchet:64 passed,140 warnings,9.08s. Fixture server is available; new support RLS/schema certification remains mandatory.
Task1 implementation commit ae3e133251:124 contract/schema cases passed across SQLite/PostgreSQL with no backend skips and95% scoped coverage;440 compatibility cases,60 workflow CI guards passed. New production Bandit zero findings/errors; new-file Ruff/format clean. Existing conftest3 import warnings preserved. Independent task review pending. Full repo collection attempted but stopped at unrelated Workspaces import of absent httpx2 after70416 collected items; no dependency changes or full-suite-green claim.
Task1 review complete after fix3c3b007ae6: independent reviewer reproduced SQLite NUL suffix constraint bypass;20 red direct-insert failures added, fixed byte-bound/NUL-safe checks. Green120 SQLite+58 PostgreSQL=178 cases, zero backend skips; strict Bandit0/0 and new-file lint/format/compile clear. Scoped re-review confirmed addressed, no new findings. Task1 complete; proceeding to Task2 exact admission/enlisted fences. No production runtime credentials or activation claim; runtime-role enforcement is required in Tasks2/5.
Task2 implementation4e9a092dc1 submitted to independent spec/quality review. Final workflow/schema/contracts280 passed,no backend skips,95% coverage. Admission/idempotency compatibility271 nonPG+144PG pass; completion/prepared89 nonPG+76PG pass. New production strictBandit0; combined scope retains unchanged manager B608 baseline only. Native transactions/fair-share and update-boundary expiry regressions added during self-review. No later effect or acquisition episode behavior yet; AC2 awaits review.
Task2 completed after fix b142bcadf1 and independent scoped re-review: SQLite transaction prefix bypass reproduced with12 failures; native SQLite authorizer now retires fences for actual transaction-control operations. Green330 workflow+301 compatibility cases on both backends,no skips. Ruff/format/compile and strict workflow Bandit clear; unchanged manager baseline retained. Warning sources explicitly inventoried as existing fixture/dependency/config/completion deprecations; historical suppressed676 total cannot be reconstructed. No review findings remain; Stage3 effect proofs now starts.
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

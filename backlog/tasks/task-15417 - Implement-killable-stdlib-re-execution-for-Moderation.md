---
id: TASK-15417
title: Implement killable stdlib re execution for Moderation
status: In Progress
labels:
- moderation
- security
- refactor
dependencies:
- TASK-15415
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the requester-approved five-stage subagent-driven plan for bounded parent-supervised stdlib re child processes, fail-closed current caller integration, and sanitized operational incidents.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement bounded protocol/worker and semantic parity including supervised and custom-template rules.
- [ ] #2 Implement parent deadlines, admission, cancellation, cleanup ownership, lifecycle, and independent adversarial process-tree harness.
- [ ] #3 Implement report-once admin incidents including early admission failures and single neutral-phase migration/API/UI contracts.
- [ ] #4 Activate all current callers with async-safe waits, preserved overrides, terminal streaming failures, and no fail-open retries or partial successes.
- [ ] #5 Complete focused regression, packaging/platform evidence, touched-scope Bandit/lint/compile, task reviews and final whole-branch review; record limitations honestly.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Requester approved revised plan and explicitly requested subagent-driven implementation on 2026-11-02. Plan: Docs/superpowers/plans/2026-11-02-moderation-killable-stdlib-re-implementation-plan.md. Spec: Docs/superpowers/specs/2026-11-02-moderation-killable-stdlib-re-design.md. ADR required: yes; accepted ADR-086 governs process ownership and security contract, ADR-021 lifecycle. Allocation audit origin/dev max15412, open PR max15413, large PR paginated max13950, local max15415; reserved15416 for renumbering colliding local triage task and allocated15417. Existing isolated worktree codex/moderation-ci-triage-regex-design at91d36d6205; no rebase/push/publication authorized. Baseline evaluator/characterization/Guardian suites:208 passed,428 warnings, /tmp/moderation-killable-re-baseline.log. No active implementation ledger existed; create one for this plan and preserve checkpoints. Model overrides omitted to obey developer tool policy.
Task 1 started in shared isolated worktree, base b9cff950fa. Read Task 1 brief/contracts and root AGENTS (no nested API AGENTS). Scope: inactive protocol/models/worker, optional output allocation guards, pure Guardian extraction. TDD evidence and interfaces will be recorded in .superpowers/sdd/2026-11-02-moderation-killable-stdlib-re-implementation-plan/task-1-report.md. No later-task activation or adversarial containment probes.
Task 1 implementation and author self-review complete (service remains unactivated). New models/protocol/worker/pure Guardian evaluator, sanitized shared exception, optional allocation guards, benign parity/property tests. Expanded verification including all Guardian: 729 passed, 1470 warnings; exact origins recorded without new suppression in task-1-report.md. Python 3.12.12/3.13.12/3.14.3 each passed 195 stdlib template/length parity cases plus isolated DEBUG child. Production Bandit zero findings; test-only advisory B404/B603 (trusted benign subprocess harness) remain unsuppressed, with only test assertions B101 excluded. Full pytest attempted: collection blocked solely by missing httpx2 in tests/Workspaces/test_workspaces_api.py:12 (85956 items collected). Linux enforcement/hostile probes await Task 2 independent harness. Detailed TDD RED/GREEN commands, interfaces, warnings and concerns: .superpowers/sdd/2026-11-02-moderation-killable-stdlib-re-implementation-plan/task-1-report.md. No controller ledger/plan/spec/contracts edits; no subagents.
Task 1 review fix round 1 started from 304134e556. Read verbatim task-1-review.md and refreshed brief/contracts. Reproduce valid mixed empty/nonempty substitution count rejection, prove conservative 2*n+1 per-rule bound, correct framed version test and add schema-aware generated request/response type/boundary coverage. Only execution_protocol.py and its two owned test files plus tracking/report; preserve controller-owned uncommitted plan example correction. No activation, subagents or hostile probes.
Task 1 review fix round 1 complete: conservative count validation is rules*(2*max(original_length, output_ceiling)+1), with proof and benign exact count-3/mixed-match/sequential/intermediate/identity-template regressions. Unknown-version test now mutates a valid framed request; generated schema-aware request/response key/type/boundary tests cover all five operations, count/span/index integers including booleans/nonfinite values, zero rules and duplicate/unsorted skipped indices. RED: 6 count failures,205 passed; first GREEN 211 passed; final focused 233 passed,478 existing warnings. Relevant evaluator/delegation/import/full Guardian regressions once:625 passed,1262 existing warnings. Compile/Ruff/Black and production Bandit clean; advisory test Bandit only the seven previously reviewed B404/B603 heuristics, B101 sole exclusion. Detailed exact commands/evidence appended to task-1-report.md. Public service remains unactivated; controller plan correction left untouched/unstaged. Prior full-suite missing-httpx2 and deferred Linux/hostile checks remain disclosed. No subagents/push/rebase.
Task1 complete: 304134e556 implemented bounded worker/protocol/guardian extraction; independent review found mixed zero-length count bound and protocol-test gaps, fixed in2bcdffa4e2. Scoped re-review approved spec and quality, all findings addressed, no new issues. Final focused runs233 protocol/worker +625 evaluator/import/Guardian=858 passed; these are suite totals, not858 newly added tests. Compile/lint/production Bandit clean; existing warnings and explicitly reviewed test-only trusted subprocess heuristics retained. Full-suite collection remains blocked by missing httpx2; no whole-suite pass claimed. Task2 supervision/harness now starts. UI baseline33 passed using existing Bun workspace dependencies (pnpm exec attempted install; use Bun/local Vitest). Local Docker daemon28.1.1 and cached Python3.12.13 container available for real Linux validation. Controller corrected the plan-mandated unknown-version example to use a valid framed request; runtime service remains unactivated.
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

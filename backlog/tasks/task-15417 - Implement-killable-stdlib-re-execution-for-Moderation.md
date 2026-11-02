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
- [ ] #1 Implement bounded protocol/worker and semantic parity including supervised and custom-template rules.
- [ ] #2 Implement parent deadlines, admission, cancellation, cleanup ownership, lifecycle, and independent adversarial process-tree harness.
- [ ] #3 Implement report-once admin incidents including early admission failures and single neutral-phase migration/API/UI contracts.
- [ ] #4 Activate all current callers with async-safe waits, preserved overrides, terminal streaming failures, and no fail-open retries or partial successes.
- [ ] #5 Complete focused regression, packaging/platform evidence, touched-scope Bandit/lint/compile, task reviews and final whole-branch review; record limitations honestly.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Requester approved revised plan and explicitly requested subagent-driven implementation on 2026-11-02. Plan: Docs/superpowers/plans/2026-11-02-moderation-killable-stdlib-re-implementation-plan.md. Spec: Docs/superpowers/specs/2026-11-02-moderation-killable-stdlib-re-design.md. ADR required: yes; accepted ADR-086 governs process ownership and security contract, ADR-021 lifecycle. Allocation audit origin/dev max15412, open PR max15413, large PR paginated max13950, local max15415; reserved15416 for renumbering colliding local triage task and allocated15417. Existing isolated worktree codex/moderation-ci-triage-regex-design at91d36d6205; no rebase/push/publication authorized. Baseline evaluator/characterization/Guardian suites:208 passed,428 warnings, /tmp/moderation-killable-re-baseline.log. No active implementation ledger existed; create one for this plan and preserve checkpoints. Model overrides omitted to obey developer tool policy.
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

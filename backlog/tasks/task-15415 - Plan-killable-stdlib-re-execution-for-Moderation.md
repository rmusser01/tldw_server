---
id: TASK-15415
title: Plan killable stdlib re execution for Moderation
status: Done
labels:
- moderation
- plan
- security
dependencies:
- TASK-15414
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare the implementation plan for the requester-approved killable standard-library re design, including bounded worker supervision, fail-closed caller integration, async streaming, and sanitized admin incidents.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Map exact module ownership, cross-task interfaces, five implementation stages, and red-green verification commands to the approved design.
- [x] #2 Cover semantic compatibility, process lifecycle, caller failures, privacy, admin rendering, platforms, packaging, and security validation.
- [x] #3 Self-review and commit the plan with verification evidence and preserve the previously selected subagent-driven execution handoff.
- [x] #4 Resolve the four approved plan-review findings: direct runtime regex bypasses, pre-service failure-reporting ownership, single unspecified-phase incidents, and independent process-tree test containment.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Requester approved the written design on 2026-11-02 (yes). Task-ID audit: origin/dev maximum 15412, open PR first-100-file maximum 15411, paginated large-PR maximum 13950, local maximum 15414; allocated 15415. ADR required: yes; governed by ADR-086, accepted with the design in this planning change. Artifact target: Docs/superpowers/plans/2026-11-02-moderation-killable-stdlib-re-implementation-plan.md. This task writes documentation only; implementation starts after plan review.
Implementation plan written at Docs/superpowers/plans/2026-11-02-moderation-killable-stdlib-re-implementation-plan.md. Five stages define exact module ownership, sync/async and wire interfaces, red-green commands, atomic facade/caller activation, independent operational reporting and shared admin rendering, platform/packaging/benchmark/security gates. Self-review completed against approved spec: all sections mapped to tasks; interface/path consistency and all five Review Focus conditions checked. Added pre-join output limits, spawn/cancellation ownership, unknown-phase administrative projections, async transform/flush/partial-save tests, and exact diff-scoped compile/lint/Bandit commands. Python >=3.12 retained; no new dependencies or engine changes. Subagent-driven execution preserved; plan confirmation pending. No runtime/test/frontend/workflow files edited; Bandit not applicable to this documentation-only planning change.
Documentation verification: changed task normalization and whole-tree task-ID check exit 0; metadata CI 4 passed, 20 warnings; git diff whitespace check clean; no changes under production/test/frontend/workflow paths. Planning deliverable complete and ready for its related commit. No runtime guarantees asserted or product tests claimed. Design approval recorded under TASK-15414; requester review of this implementation plan remains the explicit execution gate.
Reopened the same planning unit after requester approved addressing all four review findings on 2026-11-02 (yes). Edits remain documentation/task-only; no runtime implementation authorized. Existing supervised_policy.py search/sub and Workflow custom-pattern subn paths verified; preserve guardian priority/tie/notification semantics and Workflow replacement-template semantics. Plan/spec will replace unknown-phase dual projections with one neutral-phase review record and explicit migration/contract tests. Early async admission gets explicit incident ownership. Adversarial tests require an independent process-tree cleanup harness, not only pytest thread timeout. ADR assessment: accepted ADR-086 remains governing; these close coverage/ownership gaps and refine incident representation, without changing its accepted rationale.
All four approved corrections are now specified in the plan and aligned spec. Added supervised_evaluate/custom_substitute child batches with pure guardian extraction, first-tie/notify parity and bounded template/backreference expansion using the existing standalone parser. Task 2 now exposes transport-only offload; Task 3 owns the mandatory metadata-aware async wrapper/report-once guard and independent bounded reporter I/O capacity, including pre-call capacity/oversize failures. Unknown phase now means one unspecified-phase row, transactional DB migration before traffic, API/shared UI contract updates, one decision/undo history, NULL-constraint and rollout/rollback tests. Real adversarial tests require a separate stdlib controller owning POSIX process groups or Windows Job Objects, with timeout, hard-exit and invoker-disconnect self-tests. Metadata CI: 4 passed, 20 warnings; normalization, duplicate-ID and whitespace checks passed. Runtime/test/frontend/workflow files and accepted ADR records unchanged. Focused independent follow-up review pending. No implementation tests or process probes executed because this is still a planning amendment.
Independent targeted follow-up review (Ramanujan, 01a19d81-d42d-7e21-96d1-369c1b1ac33e) confirmed all four findings addressed at the planning level, coherent interfaces, no new amendment-caused P1/P2 issues; ready for revised-plan approval. Review did not claim semantic parity, migration correctness, reporting latency or platform containment without implementation evidence. Rechecked no runtime/test/frontend/workflow/accepted-ADR files changed. Bandit not applicable to documentation/task-only edits. Revised plan confirmation remains the gate before implementation; no PR publication or runtime changes performed.
2026-11-07 integration identity correction: accepted Moderation decision now ADR-096, Docs/ADR/096-moderation-killable-stdlib-regex-execution.md; implementation is TASK-15466, previously TASK-15417. Governing decisions and accepted rationale unchanged; original numbers in historical notes remain aliases of this branch records.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prepared and amended the five-stage stdlib re process-isolation plan and aligned spec to resolve all four approved review findings: supervised/custom regex coverage with preserved semantics, metadata-aware pre-service failure reporting with independent bounded I/O, one neutral-phase incident with transactional migration and response-contract tests, and an independently owned process-tree test harness. Independent follow-up review found no new P1/P2 issues. Documentation/task checks passed; implementation has not started and revised-plan confirmation remains pending.
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

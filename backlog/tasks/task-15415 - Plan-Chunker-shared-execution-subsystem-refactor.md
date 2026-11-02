---
id: TASK-15415
title: Plan Chunker shared execution subsystem refactor
status: In Progress
labels:
- chunking
- refactor
- planning
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and self-review the implementation plan for the requester-approved shared execution design and ADR-086. This is planning only: no product/test changes or implementation execution before requester plan review and an execution-method choice. Design record TASK-15414 replaces the colliding TASK-15413; unrelated UserProfiles PR #3404 is untouched.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Record approved spec/ADR, unique tracking references, isolated branch, exact implementation interfaces and file ownership
- [x] #2 Define three to five independently testable migration stages with baseline characterization, extraction RED/GREEN tests, active facade wiring and explicit verification/commit steps
- [x] #3 Cover option namespaces, temporal/generator/locking behavior, cache/metadata/error contracts, cold imports, test quality, shard assignments and security gates without weakening policy
- [x] #4 Complete the writing-plans self-review for coverage, actionable steps, interface consistency, review-focus tests and proportion; verify documentation/task consistency and commit the plan
- [ ] #5 Present the completed plan for requester review and execution-method choice without starting implementation
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Requester approved revised design and ADR-086 on 2026-11-02. Source design Docs/superpowers/specs/2026-11-02-chunker-shared-execution-refactor-design.md; architecture decision Docs/ADR/086-chunker-execution-live-context-boundary.md; current design task exact path backlog/tasks/task-15414 - Design-Chunker-shared-execution-subsystem-refactor.md. Planned artifact Docs/superpowers/plans/2026-11-02-chunker-shared-execution-refactor.md. Use existing isolated worktree on codex/chunker-execution-refactor-design; dirty main and bookkeeping branch remain untouched. Refreshed dev 5937f33065c4b5b5b2d0f731d1914c0651532eb5 has no changes from the pinned design baseline in scoped production/tests, AGENTS or ADRs. Fresh full Chunking baseline on unchanged scoped code: 796 passed, one established PyThaiNLP skip, 2000 warnings, 41.41s (/tmp/chunker_execution_plan_baseline.txt). Scanned HEAD/latest-dev backlog and all 39 open PR file lists with pagination: maximum ID15413, detected unrelated UserProfiles collision, allocated15414/15415 for our records. Plan self-review is local per writing-plans, not a duplicate subagent dispatch. ADR required: yes; approved ADR-086 will be marked Accepted and linked without changing its architecture rationale. No implementation, new dependencies, public API or behavior correction is authorized by this planning task.
2026-11-02 implementation-plan draft written at Docs/superpowers/plans/2026-11-02-chunker-shared-execution-refactor.md. Five reviewable stages: characterize public behavior; move text execution; move metadata; move lazy generator; share proven preparation and complete integration. Fixed models/context/hooks/service/helper interfaces, explicit option mapping, exact public binding and generator protocol, per-path error/lock/cache/metadata regions and staged cold imports are specified. New test_chunker_execution_* names are already covered by existing test_chunker_* patterns: structured YAML proof confirms all three future module names in all six primary/inherited matrices, with no workflow/baseline changes required. Planner personally completed all five writing-plans self-review checks: coverage, step/action clarity, interface/type consistency, Review Focus ownership and proportion. Corrected an undefined example collector, separated baseline/commit/review/publication steps and supplied exact staging commands. No subagent was dispatched for plan self-review. ADR-086 is Accepted by explicit requester approval; design TASK-15414 Done. Planning remains In Progress pending requester plan review and an execution-method choice; Native is the plan-specific recommendation, not a selected method. No product/test/config/CI edits or dependencies added. Markdown-only Bandit not applicable; final canonical/ID/scope/tracking tests and commit pending.
2026-11-02 final self-review added explicit Hypothesis option/alignment/valid-word equivalence cases to Stage1. Verified the word eager/generator invariant against 50 deterministic generated examples on the unchanged baseline (/tmp/chunker_execution_plan_words_probe.txt); its two real implementations share _prepare_chunk_records, not the inherited base generator. Plan now has concrete baseline GREEN versus extraction RED/GREEN expectations, finite cold-import timeouts and structured export evidence, explicit staging commands and separate commit/review/publication actions. Six task-ID helper unit tests pass, 24 warnings, 1.07s (/tmp/chunker_execution_plan_ids.txt); explicit canonical checks pass for design15414 and planning15415. Full task-format ratchet must run after the rename is committed, because that gate examines committed HEAD paths rather than uncommitted renames; no gate is waived. Final documentation-link, interface-syntax, scope and approval-state verification precedes commit. No execution stages started; requester plan review and method selection remain pending.
2026-11-02 final planning validation completed before commit: all eight Python interface/assertion blocks parse and all eight Bash blocks pass bash -n. Fixed a quoted parameter-expansion diagnostic containing an apostrophe that older Bash parsed incorrectly. All spec/plan/ADR relative links resolve; ADR and registry are Accepted, approved design task15414 is Done with every AC/DoD checked, and plan task15415 remains In Progress with AC5/execution-choice handoff pending. Official renumber preserved the entire preceding implementation-note history; original bookkeeping branch still points to1867f6abf9. Scope proof shows no product/test/CI/config/instruction changes. Stage statuses are all Not Started. Planning-time source/test baseline and six-matrix virtual path coverage are recorded; no future extraction test is claimed passed. Bandit is not applicable to these Markdown-only changes. Final commit includes only the plan, our design/ADR approval metadata, registry status and our two tracking records; unrelated UserProfiles task and historical records remain untouched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prepared and self-reviewed the five-stage implementation plan for the requester-approved Chunker execution design and Accepted ADR-086. Explicitly defines file ownership, live context/cache/hook models and service/helper signatures; freezes baseline behavior before active per-path RED/GREEN migrations, then consolidates equivalent setup. Covers arbitrary option names, mutable live hooks, cache identity/failure regions, metadata aliasing/alignment, generator lifecycle/concurrency, real cold imports and full static/security/CI gates. Virtual-path coverage is proven in all six existing full-suite matrices, so no YAML or baseline edit is planned. Fresh unchanged-code Chunking baseline 796 passed, one existing skip, 2000 warnings, 41.41s. Planner self-review completed locally with no unresolved coverage/interface gaps; no implementation has started. Current task TASK-15415 stays In Progress pending requester review and execution choice. Design record was renumbered only to TASK-15414 to avoid unrelated UserProfiles PR #3404 collision; other records/main/bookkeeping branch remain untouched.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

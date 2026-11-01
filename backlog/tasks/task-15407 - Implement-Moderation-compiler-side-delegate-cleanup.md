---
id: TASK-15407
title: Implement Moderation compiler-side delegate cleanup
status: In Progress
assignee: []
created_date: 2026-11-01 17:13
updated_date: 2026-11-01 17:33
labels:
- moderation
- refactor
dependencies:
- TASK-15406
references:
- 93d132a766:Docs/superpowers/plans/2026-11-01-moderation-compiler-delegate-cleanup-implementation-plan.md
documentation:
- Docs/superpowers/specs/2026-11-01-moderation-compiler-delegate-cleanup-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved five-method cleanup design and reviewed implementation plan on codex/moderation-compiler-delegate-cleanup. Preserve supported behavior, compiler implementations, active dispatch, logging sanitization, and public imports. Remove only the five repository-unused private ModerationService delegates; no regex hardening or PR publication.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Direct compiler tests and the public effective-policy logging regression preserve all migrated assertions.
- [x] #2 Exactly the five approved private delegates are removed; compiler and active service paths stay unchanged.
- [x] #3 Five structural cases demonstrate red then green and are selected by pytest -m unit.
- [x] #4 Compilation, complete Moderation/downstream regression matrix, lint, formatting and scoped Bandit checks pass with no new findings.
- [x] #5 Independent per-task and whole-branch reviews are resolved and evidence recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Pre-edit baseline compiled service/compiler and completed 400 combined Moderation/Guardian/Chat/Audio cases plus 12 Workflow adapter cases (47 deselected); overall command exit 0. Existing baseline deprecations and background Audit Loguru closed-stream errors appeared during teardown before source changes. Production and test Bandit baselines each have zero findings/errors, with B101 excluded only for test assertions. Starting Task 1 coverage migration at 1d719a5023; controller owns tracked plan/task records.

Task 1 complete at d381f01af1: only three test files changed. Three replacements passed against unchanged production, then all 67 focused cases passed with 146 existing warnings; compilation, changed-function Black checks, Ruff and Bandit passed. Fresh task reviewer approved spec compliance and code quality with no actionable issues. Controller confirmed its prerequisites (approved spec, caller audit, task tracking, 412-test baseline, zero-finding Bandit baseline) that were outside the task-only diff.

Task 2 dispatched at be0700233d with write scope limited to five service-method deletions and one new marked unit-test module. Per-task reviewer will check red/green evidence before broader verification. Design and plan remain controller-owned; no push or PR is authorized.

Fresh final validation on stable source at 1f7d3722da: compilation passed; combined Moderation/Guardian/Chat/Audio 405 passed (848 existing warnings), Workflow moderation adapters 12 passed (47 deselected, 36 warnings), total 417. Ruff all five touched files; Black new module and changed-function ranges; Bandit production and tests (only B101 excluded for tests) all passed, zero security findings/errors. Diff check clean. Full logs and Bandit JSON at /tmp/moderation-compiler-delegates-{final-matrix,final-workflow,bandit-production-final,bandit-tests-final}.*. No closed-stream Audit error in final logs; earlier baseline occurrence not claimed fixed. Task 2 independent review and final broad review still pending.

Task 2 complete at 1f7d3722da: exactly 26 production lines deleted across the five approved methods, no production additions; new marked surface module has five cases. Recorded five intended RED assertion failures, then 72 focused GREEN cases, five cases selected by -m unit, and active parser delegation passed. Fresh reviewer approved spec compliance and quality. Minor warning-detail gap resolved by six-case targeted triage with full warning summary: unchanged asyncio fixtures, FastAPI/Pydantic dependencies, setup/schema imports, SQLite import cleanup and pytest plugins configuration. Details /tmp/moderation-compiler-delegates-warning-triage.log; no warning filters or unrelated fixes added. Controller fresh compile/lint/format/security/full-matrix evidence resolves report-only verification gaps. Final whole-branch review remains.

Whole-branch reviewer approved 54b321dcec..c04c163cd1: no Critical/Important/Minor findings; Stage 3 spec and quality PASS. Confirmed retained compiler owners, supported imports/dispatch/logging and test coverage. Excluded unchanged regex/scan redesign, broader LLM moderation and baseline warning repairs per approved scope. Controller then noticed a missing module-level docstring in the new test module (AGENTS compliance); a single final polish fix and scoped re-review are running before task close-out. No further production changes.

Final docstring fix 12ec36b122 was independently re-reviewed: finding addressed, no new breakage. Refreshed exact-tree validation: compilation passed; 405 combined Moderation/Guardian/Chat/Audio tests (848 warnings, 19.79s) and 12 Workflow tests (47 deselected, 36 warnings, 0.85s), total 417; Ruff, Black, Bandit production/tests and diff checks passed. Both security scans have zero findings/errors. All task and whole-branch review gates resolved. Existing warning behavior and external private-caller compatibility risk remain documented; no supported behavior changes. No PR created, push or merge. Evidence will be preserved under /tmp/moderation-compiler-delegate-cleanup-evidence.tgz before removing only this plan scratch workspace. Completed execution plan will be retired per repository instructions, with its completed commit recorded in task references.

Completed plan preserved in git at 93d132a766:Docs/superpowers/plans/2026-11-01-moderation-compiler-delegate-cleanup-implementation-plan.md and removed from the working tree per repo completion instructions. No source or test changes after exact-tree 417-case verification; evidence archive /tmp/moderation-compiler-delegate-cleanup-evidence.tgz preserves this plan-specific ledger, reviews, reports and red/green logs.
PR preparation: renumbered this branch-owned implementation task because TASK-15403 is used by open PR 3319. Highest observed id across dev/open PRs was 15405; TASK-15407 allocated to this implementation. Canonicalized with backlog-py per ADR-059. User authorized publication against dev, requested fresh review and auto-merge; human-written Change summary is still required before arming merge.
ADR check: ADR required: no. Removing unused private forwarding methods is local implementation cleanup; compiler/service ownership, public APIs, persistence and security semantics remain unchanged. Backlog editing follows governing ADR-059. Checked Docs/ADR/README.md and module design. Current origin/dev is 54b321dcec; branch contains it. MERGE_QUEUE is unset, so no armed queue entry may be mutated.
Publication preflight resolved two task-id collisions: design now TASK-15406, implementation TASK-15407. Fully paginated file inventories for all 12 open PRs reaching the 100-file listing cap confirm highest existing id 15405 and no collisions with allocated ids. Both task files pass canonical normalization check and whole-tree duplicate-id check. No application code changed in PR preparation.
The HEAD-based task-format ratchet initially saw the retired pre-renumber paths and failed while the renames were uncommitted; direct backlog-py canonical checks on both replacement files and the full duplicate-id script passed. Commit the canonical renames, then immediately rerun the unmodified HEAD-based ratchet against origin/dev. This is metadata-only preparation, not a suppressed code/test failure.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed exactly five repository-unused private ModerationService compiler delegates while preserving PolicyCompiler owners, active helpers, public APIs, import identity, logging, dispatch, persistence and policy semantics. Migrated compiler/report and service warning privacy tests; added five marked structural regression cases with genuine RED/GREEN. All 417 scoped regression cases and quality/security gates passed. Independent per-task, whole-branch, and final docstring-fix reviews are clean. Local branch codex/moderation-compiler-delegate-cleanup is ready for a separate integration decision; publication/merge and a new human-authored PR Change summary remain future gates.
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

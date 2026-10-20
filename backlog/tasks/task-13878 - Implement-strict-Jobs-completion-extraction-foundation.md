---
id: TASK-13878
title: Implement strict Jobs completion extraction foundation
status: In Progress
created_date: 2026-10-20 17:16
dependencies:
- TASK-13421
documentation:
- Docs/superpowers/specs/2026-10-04-jobs-completion-extraction-design.md
- Docs/superpowers/plans/2026-10-20-jobs-completion-foundation-implementation-plan.md
- Docs/ADR/058-jobs-completion-row-identity.md
- Docs/ADR/078-jobs-completion-persistence-boundary.md
updated_date: 2026-10-20 17:20
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved Jobs completion foundation only: paired characterization, shallow completion-only contracts and mechanical shared-helper/exception-policy relocation into DB_Management. Preserve production complete_job control flow, SQL, SLA, commit and observer routing. Human selected subagent-driven execution option 1 on 2026-10-20. Exclude backend routing, behavior fixes, broad caller migration and historical repair.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Human reviews the foundation-only plan and authorizes execution before runtime/test changes.
- [ ] #2 Paired SQLite and required real PostgreSQL characterization preserves identity/replay, serialization/policy timing, atomicity, callback binding and cleanup ordering.
- [ ] #3 Completion-only shallow command/result/effect/policy contracts add no input coercion, normalization, eager serialization, mandatory optional metrics or DB-to-Jobs dependencies.
- [ ] #4 Mechanically relocate lifecycle event insertion, counter reconciliation, PG commit guard and exact exception policy, retaining manager aliases and unchanged complete_job routing.
- [ ] #5 Verify all affected lifecycle/admin callers, full applicable Jobs, no substantive PG skips, source/SQL/import equivalence and no new lint/security findings or suppressions.
- [ ] #6 Preserve separate PR authorization, new human Change summary and current-head review/hosted requirements; do not implement backend routing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Recreated with explicit collision-free TASK-13878 after discovering the previously provisional Jobs TASK-13875 duplicates an existing Moderation design task. Live dev e521605198dd311b94d2c63416bf07db3e5e7a0a and full paginated open PR file paths were checked: published/local maximum 13877. Preserve the Moderation record byte-for-byte and archive only the Jobs TASK-13875 snapshot with all original evidence. Earlier provisional Jobs TASK-13871 is also archived. No requirements or source change from identity repair. Execution BASE be07b6618217743938ea395bc756cf15fb0426aa. Task 1 commit 99f1d0cba1 adds 138 paired characterization cases and 10 helper-matrix cases without production/fixture changes. Fresh combined qualification 418 passed, 4300 inherited warnings, 0 skips; focused 138 and helper 242 pass. Independent Task 1 review is pending; narrow test Bandit suppressions must be resolved before accepting Stage 1. Plan-specific ignored SDD report and ledger carry detailed local evidence; preserve durable evidence in final task/plan. ADR-058 and ADR-078 govern; no new decision.
Task 1 review fix round 1 (I1), 2026-10-20: removed all three new nosec B106 annotations from the two test modules using behavior-equivalent completion marker values (existing completion_arguments helper for binding-token; local queued-token marker for complete/fail). No production, behavior, fixture, archive-content or unrelated record changes. Original implementation commit 99f1d0cba1 preserved. Affected paired node selection: RUN_JOBS=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_completion_characterization.py::test_completion_identity_short_circuits_result_work tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py::test_queued_terminal_lifecycle_reconciles_missing_counter -> 20 passed, 299 warnings in 21.74s, 0 skips. Touched-scope Ruff, scoped Black, py_compile and whitespace pass. Fresh test Bandit vs original be07b6618217743938ea395bc756cf15fb0426aa baseline: semantic findings 14/14 identical (including severity/confidence/source), errors 0/0, skipped_tests 0/0, nosec 0/0; 0 new findings and 0 new suppressions, 0 findings in new files. Earlier skipped_tests=3 evidence did not satisfy no-new-suppressions and is superseded by this check. Detailed exact commands/output and self-review appended to .superpowers/sdd/2026-10-20-jobs-completion-foundation-implementation-plan/task-1-report.md. Controller tracking repair to TASK-13878 is included unchanged except these canonical task/Task1 evidence updates; archived Jobs13875 and unrelated Moderation13875 bytes remain unchanged. No nested agents, pushes, PRs, merges or scope expansion. Controller owns ledger and final review acceptance.
Task 2 started at exact HEAD 21f22a88ac900e8121621f2e087733b5942d01d7 on codex/jobs-completion-ownership. Scope: new shallow completion contracts and direct tests only, Task 2 plan progress and this canonical task evidence. TDD red on missing module, cold import isolation and Hypothesis invariants; rerun Task 1 paired characterization using the existing shared real-PG fixture. No production routing, adapter, registry, backend operation, manager or characterization edits. Governing ADR-058/ADR-078 unchanged; no new ADR decision. Detailed report: .superpowers/sdd/2026-10-20-jobs-completion-foundation-implementation-plan/task-2-report.md.
Task 2 complete, controller review pending: added only jobs_completion_contracts.py and test_jobs_completion_contracts.py, plus Task 2 plan evidence. TDD RED: missing-module collection error before implementation (exit 2, 12 warnings). GREEN: 93 passed, 198 warnings; final source rerun 93 passed, 198 warnings, 0 skips. Hypothesis payload identity, metric-order and local-binding invariants plus spawned/isolated cold imports pass. Required real-PG combined qualification: RUN_JOBS=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_completion_contracts.py tldw_Server_API/tests/Jobs/test_jobs_completion_characterization.py -> 231 passed, 3834 warnings in 165.82s, 0 skips, including all 138 unchanged Task 1 paired cases via existing shared fixture. Final coverage 99% (142/144 statements). Scoped Ruff/Black/py_compile/whitespace pass; production Bandit no exclusions, tests B101 only, both 0 findings/errors/nosec/skipped_tests, no new suppressions. Broad repository fail-fast pytest stops collecting unchanged Workspaces/test_workspaces_api.py:12 due to missing httpx2 (12 collection skips, 9 warnings, 1 error); no dependency installation or full-suite pass claim. Manager, characterization, fixtures, unrelated TASK-13871/13875 and controller ledger unchanged; no routing, backend ops, adapter or registry. ADR-058/078 unchanged. Detailed exact commands/red-green/self-review: .superpowers/sdd/2026-10-20-jobs-completion-foundation-implementation-plan/task-2-report.md. Whole foundation remains In Progress; no push/PR/merge/rebase or nested agent.
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

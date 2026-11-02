---
id: TASK-15414
title: Implement Jobs owner-safe status and UUID admin reads
status: Done
labels:
- claims
- jobs
- implementation
dependencies:
- TASK-15426
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allowlisted owner read route with scoped Jobs RLS and unchanged administration guarded by exact UUID identity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [x] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [x] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 4: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Tasks1-3 shared Jobs prerequisites completed and independently reviewed; Task3 final correction9dbe881ab1 has605 passing affected tests/zero skips and configured Ruff/hooks pass. Begin approved Task4 Jobs-owned safe owner projection and UUID-bound admin detail. Read task-4-brief.md, constraints.md, Task1/2 interfaces and task-4-integration-notes.md in this plans isolated SDD workspace. New owner route is authenticated but separate from global-admin controls; use actual executing Jobs owner/domain RLS context with token restoration. Foreign/disallowed UUIDs return404; allowlisted data only. Keep admin role/domain checks and UUIDprecondition exact; no Claims status proxy or privilege widening. Test first with official restricted-role fixtures, configured lint/Bandit/normalhooks, retain In Progress for independent review. No producer/push/PR/merge.
Task4 implementation started at clean 6b1caf4af848740793b96b82ef4577dcec47d065 in assigned worktree/branch. ADR required: no new ADR; governed by Docs/ADR/085-jobs-resource-admission-and-domain-publication.md, 003-jobs-vs-scheduler-default.md and 058-jobs-completion-row-identity.md. Approved brief/interfaces consumed; TDD, configured whole-touched-file Ruff, restricted-role/thread/context verification and scoped hooks planned. Keep In Progress pending controller independent review.
Task4 final affected verification: RUN_JOBS=1 JOBS_DB_PATH=/tmp/task4-final-observer.db python -m pytest -q -ra on owned/admin UUID APIs plus the six brief-required compatibility modules: 235 passed, zero skips, 1049 warnings, 14.51s (exit 0), /tmp/task4-final-tests.log. RED: 17 expected missing-interface failures (/tmp/task4-red.log); admin UUID/registration RED six failures (/tmp/task4-red-contracts.log). Quarantined-state regression RED one failure then GREEN; API GREEN 18 passed. Real official Jobs PostgreSQL/restricted jobs_rls role proves unset context denies and owner active/archive reads work; route worker GUCs are false/claims/exact owner and prior tokens restored even on errors. Configured whole-file Ruff on all 11 touched Python files, py_compile and scoped Black checks pass. Touched-production Bandit exits 1 only for byte-equivalent baseline manager B608 at current line 10247; zero new findings/errors, /tmp/bandit_task4.json and /tmp/task4-bandit-comparison.log. TASK remains In Progress for controller independent review. Full assigned report: .superpowers/sdd/2026-11-01-claims-jobs-stage2c-cluster-rebuilds/task-4-report.md. Normal explicit scoped hooks and local commit follow; no producer/rollout claims.
Independent Task4 review at77d215a8e646e8f7e73ddaff9a53b4b9c7626fdc: spec compliant / qualityApproved; zero Critical/Important findings. Strict owner-safe projection, same-thread actual Jobs RLS restoration, exact UUIDadmin precondition, separate role/domain authorization and full/minimal stable routekey verified. Cannot-verify producer routeavailability and worker codevocabulary are explicitly Tasks10/9; controller records those consuming checks, no missing Task4 behavior or rollout authorization. Minor warningprovenance deferred to planned Task11 qualification with category/location evidence; do not suppress or certify all1049warnings inherited. Final235passed/zero skips officialrestrictedrole/threadboundary; configured11fileRuff/compile, scopedformat/hooks pass; Banditexit1 only unchangedB608, zero newfindings/errors. Fullreport/review in this plans SDDworkspace.
Active governance/reference update under TASK-15421: current Claims resource/publication ADR086 (metadata-only move from private085); ProposedSDKcoordinationADR087 remains Proposed. Planning/design are TASK15423/TASK15422; completed implementationTasks1/2/3 are TASK15424/TASK15425/TASK15426. Dependencies repaired viaCLI. Earlier note/commit IDs are historical evidence, not current aliases to other owners records. No accepted rationale, runtime behavior, status or verification rewritten.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Jobs-owned authenticated owner-status projection and UUID-bound admin links in77d215a8e6. Nonadmin owner/domain RLS executes/restores inside synchronous reads; allowlisted data excludes rawpayload/error/providertext. Admin role/domain guards and legacy numericdetail behavior remain. Distinct stable jobs-owned registration in full/minimal apps does not stabilize admincontrols.235affected tests passed/zero skips incl actualrestrictedrole/threadGUCs, configuredquality/hooks pass, unchangedBanditbaseline/no newfindings and warnings documented. Independentreview approved; producerlinkgating and futurefailurecodecoordination remain Tasks10/9, not rollout complete.
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

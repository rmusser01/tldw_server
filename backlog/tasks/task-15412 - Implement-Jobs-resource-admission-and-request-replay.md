---
id: TASK-15412
title: Implement Jobs resource admission and request replay
status: In Progress
labels:
- claims
- jobs
- implementation
dependencies:
- TASK-15411
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Atomic resource admission/replay, independent receipt-facade guards and scoped Jobs RLS context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [ ] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [ ] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 2: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Task 1 prerequisites completed in commits 5b12a5f3fa and 3a3c0991ce; independent spec/quality review and fix re-review are clean. Begin approved plan Task 2 using shared contracts/schema, task-1-interfaces.md and task-2-integration-notes.md in this plans isolated SDD workspace. Resource bindings are immutable on UPDATE and therefore must be inserted atomically as complete metadata; schema permits payload representation rewrites while Task 3 later guards semantic replacement. Queue/fair-share/payload preparation must remain shared with ordinary admission while retained replay runs first under the resource lock. This task also closes independent receipt-facade bypasses and defines token-restoring Jobs-specific RLS context. Test first with official PostgreSQL fixtures; no Claims lifecycle, producer enablement, push or PR.
Task 2 implementation evidence: shared deferred _prepare_job_admission plus existing backend insert/quota/counter/event path; complete immutable binding INSERT; retained replay/active reuse under resource-head serialization; receipt facade/backend guards; token-restoring Jobs scoped context. Initial RED log /tmp/task15412-red.log (58 valid missing-feature/guard/context failures plus 24 receipt setup failures, corrected and independently rerun meaningful RED: /tmp/task15412-receipt-red.log, 24 failed). Restricted Jobs role admission RED reproduced missing INSERT permission; minimal existing migration grants fixed it under real RLS. Interim GREEN 84 passed, no skips. Required/extended RUN_JOBS=1 verification in progress. ADR assessment: ADR-085 governs; no new durable architecture decision. Full report: .superpowers/sdd/2026-11-01-claims-jobs-stage2c-cluster-rebuilds/task-2-report.md. Keep In Progress for controller independent review; Task 3 semantic/lifecycle work remains unimplemented.
Final required Task 2 verification: RUN_JOBS=1 pytest of the 4 new files and 5 required characterizations, 247 passed, 0 skipped (/tmp/task15412-required-complete.log). Final compilation (14 files), Ruff, 6-file Black check and diff check pass. Touched-production Bandit: zero new findings; one unchanged B608 in manager archive-candidate SQL verified against BASE (/tmp/bandit_task15412.json, /tmp/bandit_task15412_manager_base.json). Extra characterization concern: unchanged Task 1 test_fingerprint_is_independent_of_mapping_insertion_order hit Hypothesis input-generation health checks in extended (191 passed/1 failed) and combined (432 passed/1 failed) runs; same first seed passed isolated; not suppressed or modified. Added RED/GREEN safeguards for orphaned terminal authority (2 failed -> 2 passed) and unregistered alias lookalikes (6 failed -> final required GREEN). Report includes exact Task 3 APIs/hooks, fault/RLS evidence, commands and scope limitations. Remains In Progress pending independent controller review.
Controller corrections before commit: official restricted-role fair-share regression confirmed cross-domain owner-count bypass (1 failed RED, /tmp/task15412-fairshare-red.log); shared Jobs _count_active_jobs_for_user now uses token-restoring non-admin scalar aggregate scope across domains without rebinding a non-admin owner or widening actual admission/returned rows (1 passed GREEN, /tmp/task15412-fairshare-green.log). Controller authorized bounded generation in Task 1 fingerprint-order property; original assertion/UTF-8/surrogate tests and default health checks/examples retained. First bounded text/integer strategy still failed full combined health check (439 passed/1 failed), so profiling/source inspection identified provider local-constant mining cost; final finite Unicode key/value palette uses boolean-only draws and final same-code combined qualification is running. All earlier failures and exact commands retained in task-2-report.md. No fingerprint production change or health-check suppression. Task remains In Progress.
Final controller-corrected same-code combined qualification: RUN_JOBS=1 pytest of all 19 required+extended test files, including the bounded Task 1 property and real restricted-role cross-domain fair-share regression: 440 passed, 0 skipped, 941 warnings, 91.77s (/tmp/task15412-final-same-code-v2.log). Final compilation of 15 Python files, Ruff, seven-file Black check and staged/unstaged diff checks pass. Earlier health-check failures remain fully documented; final boolean-only finite Unicode-key/integer palette avoids measured provider constant-mining without reducing examples, suppressing checks or altering fingerprint production semantics. Only remaining quality caveat is proven unchanged manager B608 at BASE; zero new Bandit findings. Normal scoped hooks/local commit evidence and exact integration signatures in task-2-report.md. Keep In Progress pending controller independent review; no Task 3 semantic/lifecycle completion claimed.
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

---
id: TASK-15415
title: Implement Claims cluster source snapshot and cleanup locks
status: In Progress
labels:
- claims
- jobs
- implementation
dependencies:
- TASK-15414
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bounded owner snapshots, publication schema and fail-closed reference/ownership integrity using existing DB_Management.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [ ] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [ ] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 5: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Tasks1-4 prerequisites independently reviewed complete; begin approved Task5 using task-5-brief.md, constraints.md and task-5-integration-notes.md in this plans SDDworkspace. Produce strict owner-consistent bounded source snapshots/digests, publicationhead/receipt schema in fresh and upgradepaths with real restricted-role policies, transactionconnection helpers and bounded foreign-reference failclosed cleanup. Coordinate add_claim_to_cluster destination-first cluster->Media->Claims order; do not rely on hiddenMedia innerjoin or absentFK as proof. Testfirst official Media PostgreSQL fixtures; no syntheticadmin, missingstorageprovisioning, globalrolewidening, transferrepair or legacyalgorithmrewrite. Keep task InProgress for independentreview; configuredlint/Bandit/normalhooks/scopedlocalcommit required, no push/PR/merge.
Sole fresh Task5 implementer started at clean 1ed9287f1aa5d8b2c0e59c1ec08e4a52eb36438d. Read brief, constraints, Task1 contracts and Task5 inventory. ADR assessment: no additional ADR; accepted ADR085 governs, no architecture change. Checked fetched origin/dev 6bf581da089a7aa6d90440a8801de07f8fdd238b: Media migration namespace ends at 026; 027 free in branch and origin/dev. Implement bounded snapshot/cleanup and schema only; leave publisher to Task7 and record evidence in assigned ignored report.
Task5 implementation ready for independent review (leave In Progress). Domain snapshots/types, same-connection cleanup locks, destination-first add coordination, schema v27 fresh/upgrade and strict publication RLS implemented; no publisher/producer changes. Namespace checked on fetched origin/dev 6bf581da089a7aa6d90440a8801de07f8fdd238b; 027 remains free. Verification: new source/restricted-role suite 49 passed (28 SQLite/domain, 21 official restricted PostgreSQL), 112 warnings, zero skips (/tmp/task5-final-domain.log); required characterization 47 passed/107 warnings (/tmp/task5-characterization.log); affected bootstrap/RLS 104 passed/265 warnings (/tmp/task5-schema-characterization-green.log); legacy rebuild/PG migration 15 passed/43 warnings (/tmp/task5-additional-characterization-green.log). Fresh/upgrade separated: 3 passed/19 warnings (/tmp/task5-migration-fresh-upgrade-final.log). Version-bump-only assertions in two existing migration suites updated, historical migration targets untouched. Ordinary configured whole-file Ruff clean on all touched Python; 12 touched Python compile; scoped format clean, inherited whole-file formatting debt compared against BASE. Bandit touched 8 production Python: zero findings/errors (/tmp/task5-bandit-final.json), no new suppressions. RED/setup distinctions, exact commands, signatures, RLS/lock proof, warnings and self-review in assigned ignored task-5-report.md; normal explicit staged hooks and local commit follow.
Separate restricted owner/lock qualification: 13 passed, 8 deselected, 38 warnings, no fixture skips (/tmp/task5-owner-locks-final.log). Explicit python -m pre_commit run --files over exactly the staged Task5 paths passed all applicable checks (/tmp/task5-hooks.log); YAML/TOML and wizard-only Ruff/Black hooks had no matching files (configured whole-file Ruff was run separately). Self-review confirms unchanged legacy rebuild modules and no lifecycle imports/domain publisher. Inherited whole-file formatter debt in five existing touched files was reproduced at BASE; edited blocks/new files are format-clean. Earlier failing legacy PG migration characterization emitted a psycopg_pool PythonFinalizationError at interpreter shutdown; the corrected final 15-pass run did not. No repository-wide qualification attempted, per controller ownership. Commit is scoped locally to Task5 and this record; independent review remains pending.
Checkpoint follow-up qualification (no production changes): explicit old-cluster -> Media -> Claims real SELECT FOR UPDATE trace plus actual competing edit/delete/transfer statements under the official restricted role: 3 passed, 21 deselected, 18 warnings (/tmp/task5-cleanup-lock-races-final.log). Command: python -m pytest -q tldw_Server_API/tests/MediaDB2/test_claim_cluster_rebuild_snapshot_postgres.py -k cleanup_cluster_media_claim_lock_order, with required venv activation. Full configured Ruff, scoped format and compile of the additional test are clean. Base implementation commit b5fc7a5bc1; aggregate distinct focused cases now 218 (52 domain including 24 restricted-role PG, plus 166 affected characterization), not a whole-branch/repo qualification. No reference-visibility or architectural blocker found; keep In Progress for controller review.
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

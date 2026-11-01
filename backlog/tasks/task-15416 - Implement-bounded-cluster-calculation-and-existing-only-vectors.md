---
id: TASK-15416
title: Implement bounded cluster calculation and existing-only vectors
status: In Progress
labels:
- claims
- jobs
- implementation
dependencies:
- TASK-15415
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve exact/embedding algorithm semantics with resource limits and non-creating Chroma access.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [ ] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [ ] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 6: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Task5 domain prerequisite independently approved in b5fc7a5bc1/a2236fcf14. Begin approved Task6 bounded exact/embedding calculation and strictly existing-only vectoraccess using task-6-brief.md, constraints.md, task-5-interfaces.md and task-6-integration-notes.md. Preserve legacyalgorithm and collidingembeddingID semantics, no storagewrites or new clustering engine. Alllimits beforepublication. ExistingOnly Chroma mustprove no provisioning acrossabsent/schema/error/disappearance/index cases with realSDKfixtures and supportedversion behavior. Preliminary mode=rw Rust SQLite primitive is notcomplete adapterproof; publicPersistentClient(validate) wasobservedcreatingmissingstorage. IfcompleteSDKcontractcannotenforce approvedno-createboundary, reportconcreteevidence and stopfornarrowreviewedstorageaccessdesign, no stat-onlyguard/stubfallback/dependencybump. Testfirst, configuredlint/Bandit/hooks/scopedlocalcommit, no Task7publication/push/PR.
Fresh sole Task6 implementer: verified clean HEAD 61162af2063945e2d01398d134bfee977b6baac2 on required worktree/branch. Read brief/constraints/Task5 interfaces/integration notes/AGENTS/TDD/verification/implementer prompt. ADR required: no new ADR for approved scope; Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs. Investigating complete SDK no-provisioning boundary first using only temporary real Chroma fixtures; installed 1.5.5 confirmed. No adapter or calculator changes yet; no nested agents or live DB access.
Task6 STOP GATE: real ChromaDB 1.5.5 temporary fixtures disprove complete existing-only behavior for the three examined configurations. Public PersistentClient(migrations=validate) created absent directory/chroma.sqlite3 before Unapplied migrations found. Raw Rust Bindings + SqliteDBConfig(MigrationMode.Validate, file URI mode=ro) failed on an intact fixture with SQLite code 8 attempt to write a readonly database. mode=rw avoided absent/disappeared SQLite/directory creation but changed SQLite on intact/malformed/error paths; get(include=[embeddings]) recreated four HNSW files after index directory removal, and overwrote three surviving HNSW files after index_metadata.pickle removal before Error finding id. Thus SQLite no-create is not a complete SDK no-provisioning boundary. Stopped before production changes/calculator/TDD implementation rather than weaken gate or invent storage architecture. Thirteen isolated subprocess fixture scenarios; parent probe exit 0 records observations, NOT passing Task6 tests. Full evidence: assigned ignored .superpowers/sdd/2026-11-01-claims-jobs-stage2c-cluster-rebuilds/task-6-report.md; reproducible /tmp/task6_chroma_boundary_probe.py and /tmp/task6_chroma_boundary_probe.log. No production/test changes, no domain writes, no PG/live DB, no dependency or gate changes. Task remains In Progress for controller review; narrow storage-access design decision required before resuming. ADR-085 unchanged; no new ADR introduced.
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

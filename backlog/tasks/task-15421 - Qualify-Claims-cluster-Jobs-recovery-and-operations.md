---
id: TASK-15421
title: Qualify Claims cluster Jobs recovery and operations
status: In Progress
labels:
- claims
- jobs
- implementation
dependencies:
- TASK-15420
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cross-backend fault/property/security qualification, rollout/rollback/restore runbook and whole-branch review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [ ] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [ ] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 11: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Controller begins independent Task11 namespace prework while Task7 publication implementation runs in disjoint files; recovery/runbook/whole-branch qualification remain pending Tasks1-10 completion. Fresh2026-11-02 audit: origin/dev5937f33065c4b5b5b2d0f731d1914c0651532eb5 and all39openPR paginatedfileAPIs,290selectedpaths, remote taskmax15413/local15421. OUR planning15282/15283 collide with published Chunker records; OUR15411 collidesPR3398 sandbox task; OUR15412 collidespublishedVZ task; OUR15413 collidesPR3404 UserProfiles followup. Renumber onlyOUR five records to15422..15426 via backlog-py and repair active dependencies/references; preserve historical notes and allotherownersrecords. OUR private ClaimsADR085 collidespublishedAuthNZ085; metadata move to086 preserves accepted rationale. Proposed087 remains free in this fresh audit. No rebase/push/maincheckout changes. ADRassessment: identifier-only repair under existing decisions, no new architecture or acceptance.
Namespace repair uses CLI only, preserves selected statuses/criteria/history and fixes dependencies. ADR086 Decision-through-end diff against ebc5cd0f00 oldADR085 is byte-identical. Local task-ID checker exits0. First Backlog/format/ID suite164passed,1failed,32warnings2.39s: HEAD-aware changed-task gate still reads old filenames from committedHEAD while renames are unstaged, so FileNotFoundError for old15282. This is a measurement sequencing issue, not noncanonical content; explicit selected canonical check exits0. Commit isolated canonical metadata with normalhooks before rerunning unchangedHEAD-aware gate. Do not alter/skip gate or include activeTask7 production files.
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

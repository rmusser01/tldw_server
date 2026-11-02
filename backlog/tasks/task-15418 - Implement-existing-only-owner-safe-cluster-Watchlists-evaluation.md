---
id: TASK-15418
title: Implement existing-only owner-safe cluster Watchlists evaluation
status: In Progress
labels:
- claims
- jobs
- implementation
dependencies:
- TASK-15417
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Non-creating Watchlists accessor/evaluation with owner-predicated cluster reads, counts, updates and notifications.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [ ] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [ ] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 8: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Active governance/reference update under TASK-15421: current Claims resource/publication ADR086 (metadata-only move from private085); ProposedSDKcoordinationADR087 remains Proposed. Planning/design are TASK15423/TASK15422; completed implementationTasks1/2/3 are TASK15424/TASK15425/TASK15426. Dependencies repaired viaCLI. Earlier note/commit IDs are historical evidence, not current aliases to other owners records. No accepted rationale, runtime behavior, status or verification rewritten.
BeginapprovedTask8 afterTask7independentfix1re-reviewclean (21919902aa/e7757f5e7f). Use task-8-brief.md/constraints.md/task-8-integration-notes.md andreviewedhandoffs. Implementnoncreating owner/backend-pinned Watchlists accessor +ownerpredicates onclusterread/membercount/watchlistcountupdate; preservelegacy omittedowner/for_user behavior. Newasyncevaluator onlyownereligiblesubscriptions, best effort postsuccess; no storageschema provision/newdeliveryworker. Controllerpermits narrow existing claims_notifications.py insert-error log sanitization withregression because swallowedinnerrawexception bypassesoutercatch; preservesemantics,no notificationrefactor. TestfirstSQLite/sharedlayout+officialrestrictedPG, configuredquality/Bandit/hooks/scopedcommit, keepInProgressforreview. ADRassessment: accepted086+003/058 governs, Proposed087 unaffected; no new durabledecision. No push/PR/merge/maincheckout changes.
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

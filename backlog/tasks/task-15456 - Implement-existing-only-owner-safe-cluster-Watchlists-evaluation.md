---
id: TASK-15456
title: Implement existing-only owner-safe cluster Watchlists evaluation
status: Done
labels:
- claims
- jobs
- implementation
dependencies:
- TASK-15455
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Non-creating Watchlists accessor/evaluation with owner-predicated cluster reads, counts, updates and notifications.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [x] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [x] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 8: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Active governance/reference update under TASK-15421: current Claims resource/publication ADR086 (metadata-only move from private085); ProposedSDKcoordinationADR087 remains Proposed. Planning/design are TASK15423/TASK15422; completed implementationTasks1/2/3 are TASK15424/TASK15425/TASK15426. Dependencies repaired viaCLI. Earlier note/commit IDs are historical evidence, not current aliases to other owners records. No accepted rationale, runtime behavior, status or verification rewritten.
BeginapprovedTask8 afterTask7independentfix1re-reviewclean (21919902aa/e7757f5e7f). Use task-8-brief.md/constraints.md/task-8-integration-notes.md andreviewedhandoffs. Implementnoncreating owner/backend-pinned Watchlists accessor +ownerpredicates onclusterread/membercount/watchlistcountupdate; preservelegacy omittedowner/for_user behavior. Newasyncevaluator onlyownereligiblesubscriptions, best effort postsuccess; no storageschema provision/newdeliveryworker. Controllerpermits narrow existing claims_notifications.py insert-error log sanitization withregression because swallowedinnerrawexception bypassesoutercatch; preservesemantics,no notificationrefactor. TestfirstSQLite/sharedlayout+officialrestrictedPG, configuredquality/Bandit/hooks/scopedcommit, keepInProgressforreview. ADRassessment: accepted086+003/058 governs, Proposed087 unaffected; no new durabledecision. No push/PR/merge/maincheckout changes.
Task8 implementation started at clean a4916a15f1d2e5d61b70ad8a840ac0d5b65e0f69 on approved worktree/branch. Following assigned task-8-brief.md, constraints.md and integration notes; TDD first, existing-only backend-pinned adapter, owner-predicated aggregate helpers, narrow notification-log sanitization. Governing ADR-086 Accepted plus ADR-003/058; ADR-087 remains Proposed/vector-only. No new ADR or lifecycle/worker changes. Keep In Progress for controller review.
Task8 implemented; keep In Progress pending controller independent review. New open_existing borrows/pins validated Media backend, verifies all 5 evaluator tables/21 required columns without bootstrap/repair; pg_catalog avoids ACL-hidden absence. New post-publication evaluator intersects owned subscriptions with owner-predicated clusters before counting/updating/notifying. All 3 aggregate helpers add optional keyword-only owner_user_id with explicit query/write ownership guards, preserving omitted-owner contracts. Narrow claims_notifications insert-catch logging sanitized. Final affected qualification: /tmp/task8-final-qualified.log and .xml, 75 passed (63 new, 18 official restricted-role PostgreSQL), 0 skips/failures/errors, 376 inherited warnings. Scoped compile/format pass; configured whole-file Ruff 12 inherited findings matching a4916a15f1 (0 new); touched-production Bandit /tmp/bandit_task8_final.json: 0 findings/errors, no new suppressions. Full evidence/interfaces/limitations: .superpowers/sdd/2026-11-01-claims-jobs-stage2c-cluster-rebuilds/task-8-report.md. Task9-only caller wiring/newly_published guard remains intentionally unimplemented; no lifecycle/producer/global-storage work.
Self-review of assigned production/test diff complete; only assigned 6 Python paths plus this task changed. Normal applicable pre-commit hooks pass (/tmp/task8-hooks.log); yaml/toml and wizard-only Ruff/Black hooks correctly had no applicable files. Independent configured Ruff/scoped-format checks were run separately, not replaced by hook skips. No whole-repo/Jobs/Chroma test reruns, nested agents, producer wiring or external git operations.
IndependentTask8review task-8-review.md: SpecCompliant/QualityApproved, noCritical/Importantfindings. Controllerreadfullreport/savedqualitysummary/BanditJSON. Task8complete d62f897baabc26285d3cb0abcad20cf173965cbc:75pass0skips incl18officialrestrictedPG,63newcases,12legacycharacterization; configuredRuff12exactlyinherited0new,compile/scopedformat/hooks pass,Bandit0findings/errors. Minor inherited376warning/nosec/lintnoise remainsdisclosed; no newruntime defect. Nonblockingphysicalpartial-columnPGtestdepthimprovement assignedTask11beforefinalqualification; existingeachcolumninventorytests mockmetadata, actualmissingtable/revokedSELECT casesalreadycovered. No Task9caller/worker/producerenabled.
Publication-time namespace correction under TASK-15465: only this private Claims record renumbered; historical IDs/commits in prior notes remain historical. Current accepted resource ADR094 and Proposed SDK ADR095 preserve prior rationale/status.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Existing-only backend-pinned Watchlists evaluation with explicit owner predicates at cluster read, member count and count-update boundaries; owned subscriptions never authorize foreign cluster data. Requiredschema missing skips withoutprovisioning, accesserrors sanitized and innernotificationlog leakremoved. Legacyfor_user/unscopedhelpers/deliverysemantics preserved.75tests pass0skips incl18realrestrictedPG; no newlint/securityfindings, normalquality/hooks pass. Independentreview approved; physicalpartial-columnPGcoverage improvement carriedTask11, knowninheritedwarnings disclosed.
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

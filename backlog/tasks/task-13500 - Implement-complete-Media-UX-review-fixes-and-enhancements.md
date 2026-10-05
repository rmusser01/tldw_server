---
id: TASK-13500
title: Implement complete Media UX review fixes and enhancements
status: In Progress
labels:
- media
- ux
- frontend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement all five major findings, all nine additional issues and all potential enhancements from TASK-13450, preserving WebUI/extension parity. User authorized every item and delegated ordering. Existing audit solutions are approved design direction. The audit baseline was dev 75ab224081bf140ef52017c1a9b0a04f6878d488; implementation now includes latest fetched dev 49cec711906925959775d10fb739f247f28fd232 via clean integration merges.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reliable URL and file queue handoff, deduplication, eligible counts and overwrite semantics across WebUI and extension.
- [ ] #2 Failed-item retry preserves successful results, saved batches open directly in multi-review, and saved/search readiness is truthful.
- [ ] #3 Preview, selection, keyboard activation and selected-set navigation agree across pages with an explicit 30-item simultaneous reading cap.
- [ ] #4 Inspector selection persists across pages, deletion is recoverable, and reading/bulk actions remain accessible on mobile.
- [ ] #5 Recent imports, active-tab capture, compact batch summaries and contextual batch guidance are implemented using existing state.
- [ ] #6 Meaningful unit/integration and browser checks, build/type/security validation, source review, documentation and tracked incremental commits cover the full accepted scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Audit: .impeccable/critique/2026-10-04T23-52-07Z__i-src-components-review-viewmediapage-tsx-a9a598d1.md. ADR required: no new ADR; local UX repairs reuse existing shared UI, scoped ingestion session/runtime and server APIs. Governing ADR-022 media pipeline ownership, ADR-059 task editor cutover. No new dependencies or backend architecture changes planned.
Approved design: Docs/Design/2026-10-04-media-ux-complete-design.md. Five-stage plan: IMPLEMENTATION_PLAN_media_ux_complete_20261004.md. Sequential fresh implementation and review subagents; baseline 16 focused tests pass. Preflight interface scan and rulings recorded in plan-specific SDD ledger. All original audit findings/enhancements mapped to acceptance checks. Current dev requires backlog-py; prior audit record normalized through canonical CLI.
Design refinement for mobile multi-review: preview or starting selected review opens Content; checkbox toggles keep Results visible while assembling a batch. This avoids repeated pane switching and matches Inspector bulk mode. Recorded in design, Task3 brief and SDD ledger.
Stage1 complete: source handoff, queue eligibility, explicit repetition, independent replacement and extension capture; commits383720ccdd/6180dc9543. Independent source review found and fixed live File remount loss and playlist repetition; re-review approved. Stage2 started.
Task 2 source review found three recovery/publication gaps; original implementer is addressing nonretryable correction actions, conference submission-failure identity retention, and an atomic owned-selection snapshot. Task 3 starts after clean scoped re-review. All user-approved issues and enhancements remain in scope.
Task 2 implementation 37e61f537a and fix3968b11809 passed independent review/scoped re-review; 148 amended tests pass. Multi-review stage TASK-13500.3 is active. Saved-review publication uses an atomic version1 authority/IDs snapshot; future consumers explicitly assigned.
Task 3 final mounted Review family91pass and transport/policy128pass; atomic owned-set consumption, bounded detail loading and deferred-restore guards implemented. New lint issues resolved; incumbent warnings recorded. Scoped report/commit then independent task review is the next gate; Inspector/recent-import integration remains.
Task3 complete with clean task review and scoped fix re-reviews: a92a2f737d,71b48d939c,6d22b62c39. Preview and full selection separated; reading bounded30; original operation/atomic owned snapshot enforced. Final semantic/native/build checks Task5. Task4 Inspector/accessibility now In Progress.
Task4 clean after source e3078aa6a0 and fix f0f6cd2b06: full cross-page qualified metadata/actions, mobile focus/controls, confirmed partial Trash and version-bound Note restore; repeated off-page tagging fixed. Initial262 and amended84 covering tests pass. Task5 now In Progress; integrate latestdev ba553fdc51 first, then history/platform verification.
Latest dev ba553fdc51 integrated without conflicts by merge e5aff2347c; upstream verified ancestor and Media-only branch delta. Task5 base e5aff2347c; fresh recent imports/integration implementer next. Re-run isolated ingestion contracts once due upstream MediaDB changes.
Integration remains In Progress on latest dev ba553fdc51 (merged as e5aff2347c). Main affected regression run passed 113 files/1,066 tests; configured TypeScript and Next/WXT builds passed before native follow-up. Native QA repaired primary/empty Recent imports placement, missing sidebar wizard host and exact scoped list-route aliases. Current follow-up covers expanded mobile/max10-history geometry and a reproduced retry render loop, with root-cause tracing, focused RED/GREEN and fresh builds required before completion. Browser processing outcomes are simulated; three isolated real backend ingestion/job contract cases passed. Task5 source review and final whole-branch review remain pending.
Task5 native integration now passes all three tracked journeys: mixed six-input import with failed-only retry and reload/history/saved review; max10-history mobile/landscape/large-text Inspector selection and reversible trash; forty selected items with thirty-detail reading window then remaining ten. Affected 42 files/370 tests pass. Real Modal/store retry persistence and real Query/StrictMode readiness regressions repaired. Final current builds, actual HTTPS sidebar capture, documentation, patch reviewability and independent source reviews remain. A fresh dev fetch found 61589721bc (test fixture only); sequential integration will precede final whole-branch review.
Latest dev 61589721bc integrated by 0b36ab27e5: only the 20-line privilege-registry fixture changed since the prior integrated backend, JSON validated and upstream is an ancestor. Task5 initial source eee6ad351f passed current builds, standalone native4/4 and real packaged capture/continuation gates. Independent task review found three Important history integrity defects (capture replacement owner, result normalization, superseded failed durable attempts); original implementer is completing fix round1 with mounted RED/GREEN before scoped re-review. Earlier Tasks1-4 remain reviewed; whole-branch review is the next gate after Task5 acceptance.
Task5 history fix36c1f22346 passed independent scoped re-review: all3Important findings addressed, no new breakage. Six affected suites90pass and both semantic checks pass. Latest fetched dev49cec71190 merged cleanly as7918323e01; upstream delta3Notes/chat docs only. Permanent verification now records source commits/review receipts and exact command recipes. Full whole-branch review and final current-source builds precede acceptance and finalization.
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

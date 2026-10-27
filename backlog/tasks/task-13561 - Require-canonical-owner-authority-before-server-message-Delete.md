---
id: TASK-13561
title: Require canonical owner authority before server message Delete
status: Done
labels:
- chat
- ux
- frontend
- bug
dependencies:
- TASK-13560
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove unqualified legacy server Delete fallbacks from both chat interfaces and preserve source-bound local/Temporary Delete. Reuse native leaf/CAS Delete and existing owner/source fences; prevent retained callbacks and held local outcomes from replacing a newer chat or reporting false success.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Server-backed Delete without ready native owner/capture refuses before reads, writes, mirror cleanup or display mutation; qualified native Delete remains usable.
- [x] #2 Local and Temporary Delete retain their legitimate paths, rendered source and live busy guards; held outcomes do not publish in replacement views or report obsolete success.
- [x] #3 Causal mounted full/compact controls, connected regressions, both client compilers/static and two independent source reviews pass; whole-program and runtime limits remain explicit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Highest remote/openPR/bothlocal task13560 verified; fresh dev97ea9cd5fa is ancestor. Independent source review confirms missing canonical capture admits unscoped legacy one-row deletion that can orphan live descendants. D7 saved native/mirror read-only and ADR049/050 require actual owner authority; ADR059 tracking. No new durable API/storage/lifecycle decision, so no new ADR. Existing public backend legacy API compatibility remains; remove the unsafe UI fallbacks and reuse existing native qualification. Plan Docs/Design/IMPLEMENTATION_PLAN_chat_owned_delete_13561.md.
Static cleanup reassessment after three evolving checks: removing unused Delete callback dependencies exposed invocation-predicate instability, moving those predicates inside the callback exposed the render-time fallback function in deleteViewCurrent. All 1589 connected regressions and both real compilers stayed green, but static gates were not credited. Avoid memoizing H1 by stable controller identity, which would retain the wrong epoch; inspect existing captured-fence patterns and move only the optional absent-controller boolean fallback into the invocation predicate, preserving a fresh rendered fence. Node ESLint output replaces the truncated Bun stdout harness; only embedded source-line numbers are normalized in comparisons.
Final source repair uses only native Delete for saved server authority and source/target transaction leases for local/Temporary rows. Live Reply and survivors are preserved after ACK; owned H1 parent-follow is qualified synchronously. The full idle/Temporary duplicate-answer history defect is fixed by the existing eligible-message builder. Meaningful duplicate RED: two full failures/two compact positives. Final matrix1589/1589 across42 suites and473/473 after final predicate cleanup; both actual compilers;10 authored TS/TSX files309 baseline/current diagnostics,0 introduced; equivalent owned formatting/diff pass. Independent state and transport reviewers CLEAR. Bandit N/A: no authored Python. Storage rollback uses a declared transaction double; native browser/packaged/physical/AT acceptance remains open. Evidence/master and completed plan copy /private/tmp/chat-owned-delete-13561-completed-plan.md retained; only owned plan retired.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed unsafe unqualified server Delete fallbacks across both chat interfaces while retaining source-bound local/Temporary and qualified native paths. Transaction leases, live Reply/survivor publication, exact duplicate-answer prompt projection and owned parent-follow prevent stale writes/publication and misleading success. Connected regressions, both client compilers/static and both source reviewers qualify the bounded repair. Whole-program/runtime limits remain explicit; TASK13562 continues preview loading/performance.
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

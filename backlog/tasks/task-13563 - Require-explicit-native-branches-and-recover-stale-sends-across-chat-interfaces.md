---
id: TASK-13563
title: Require explicit native branches and recover stale sends across chat interfaces
status: Done
labels:
- chat
- ux
- webui
- extension
dependencies:
- TASK-13458
- TASK-13454.8
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the accepted D7 P1/P6 client contract: ordinary full and compact native sends must opt into the atomic leaf check; a confirmed updated-elsewhere refusal retains the exact draft and offers loading the latest graph or an explicit branch. Preserve native first-create, local/Temporary behavior, and explicit saved-message actions. Live task-owned save/reopen/Regenerate already succeeds; null public admission fields are not a diagnosed defect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Native ordinary and tracked-character new sends carry strict latest intent; existing Edit and resend and Regenerate carry explicit branch intent; local and Temporary behavior remains unchanged.
- [x] #2 A confirmed scoped history_branch_changed refusal writes no user or assistant row, dispatches no provider, retires only its pending attempt, and keeps the exact originating draft with owned latest or explicit-branch recovery on both interfaces.
- [x] #3 Delayed or malformed conflict and recovery work cannot change another view, account, draft or active turn; focused tests, both client type gates, lint delta and independent reviews are recorded.
- [x] #4 Current task-owned browser verifies the controlled conflict and recovery sequence; remaining packaged, physical and participant acceptance stays explicit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Latest fetched dev 97ea9cd5fa7e3a61ee4e7c56f3d9643b7311d511 is included. Remote/open-PR/both-local highest ID 13562 checked before allocating. ADR required: no; accepted D7 P1/P6 and ADR-049/050 govern owner-validated graph intent and existing recovery; ADR-059 governs tracking. No new durable authority or public request field. Plan Docs/Plans/IMPLEMENTATION_PLAN_native_latest_branch_recovery_13563.md. First-source suspicion was disproved by actual native user admission, assistant settlement, reload/reopen and Regenerate against the local provider; see /private/tmp/chat-first-turn-13563-* and prior 13562 browser logs.
Stage1/2 implemented after causal RED. Confirmed refused journal means retained original exact text/images and unchanged client candidate IDs, not deletion or relabeling as server authority. Same-ID H1 retirement before admission now blocks native input/provider and keeps its prior durable original input. Initial independent reviewers identified raw-code false non-admission, lost image recovery, stuck loading and final native preparation fence; each now repaired with causal and connected controls. Pre-format 946 tests/11 suites plus actual-H1 focus and retired Copy controls, both compilers and9 actual authenticated backend branch-admission controls pass. Final formatted matrix/lint/build and both re-reviews pending. Older native-server P1 feature detection and full-page proactive polling remain explicit broader program work. Bandit N/A: no authored Python.
Final source review: both authorized independent state and transport reviewers CLEAR after the image, finite loading, typed feedback and native pre-admission fence repairs. Final formatted 1322 tests/25 connected and neighbor suites pass; both actual compilers pass;20 tracked TS files plus the new classifier fixture introduce0 lint findings. Static comparison normalizes unchanged embedded source line/column and excerpt line positions only; pre-existing controller-assignment diagnostic is unchanged. Runtime-equivalent owned formatting and clean diff. Native qualified build cDcszmRYkiD-K2k9N68SA compiles; all799 manifest assets exist; shared594.9KB/heaviest791.1KB pass unchanged600/900KB gates; shared tokens pass. Actual browser controlled advance refuses input with409 and false, keeps exact input, writes no row/provider; Branch here then true saves the original input once under its selected old parent. Second Send returns false/409; Load latest installs a fresh owned6-row path and performs no resend. Manual latest continuation/cold reopen qualification in progress. No new visible Chrome windows.
Final native manual continuation qualifies: request2519 carries false, saves exact second input once beneath the freshly loaded tip, canonical total11 rows preserves branches, and cold reload restores its selected8-message path. Both refusal paths added no row/provider; recovery choices did not send. Current headless browser only, no new visible Chrome. Final classifier17 tests pass after the test-only type cleanup. ADR assessment includes refused client candidate/image retention under existing ADR049/050/D7; no new authority. Completed owned plan preserved at /private/tmp/chat-latest-branch-13563-completed-plan.md and retired from repository. Remaining older-native capability/freshness and broader all-ten qualification stay open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Native new Send now requires latest, while saved-message actions and a user-selected one-shot Branch here explicitly branch. Confirmed owner-scoped conflicts retain exact original text/images and expose guarded latest/branch recovery without automatic resending. Source retirement, partial/unknown outcomes and failed journal updates preserve truthful recovery. Qualified:1322/25 connected matrix,17 final classifier tests,9 real authenticated backend controls,both client compilers,0 introduced lint findings,two independent CLEAR source reviews,and current native browser strict refusal/branch/latest/manual continuation/cold reopen. Build/assets/tokens/budget pass. No authored Python:Bandit N/A. Completed plan /private/tmp/chat-latest-branch-13563-completed-plan.md; design records exact evidence. Compact/character fixtures do not certify packaged/native end-to-end, physical/participant or older-server compatibility gates. Parent/all-ten program remains In Progress.
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

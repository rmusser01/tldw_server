---
id: TASK-13700
title: Prevent chat composer remount during Media handoff readiness
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and repair the full-to-empty-to-full composer transition captured by final PR #3229 critical CI. Preserve complete source text, draft ownership and route cleanup guarantees; qualify the actual root lifecycle with a regression and an owned browser journey.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A deterministic lifecycle regression reproduces the confirmed source of composer recreation before the fix
- [x] #2 Shared migration and both shell callers preserve private composer state without weakening account or source guards
- [x] #3 Connected tests, actual handoff replay and two independent reviews qualify the repair; exact final-head CI remains the PR landing gate
- [x] #4 A held fresh-draft session restore cannot replace a newer accepted Media source; ordinary historical conversation/source merging remains supported
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR https://github.com/rmusser01/tldw_server/pull/3229; failing head 1b3d2846089326a4be656c4476c58186c19bff30, critical run38696151310 job116293111956. Trace records full value188110ms then empty188441ms and changed React useIds. Root not yet confirmed. Scope: shared useMigration and its WebUI/native shell callers if the migration admission lifecycle is confirmed; source oracle stays unchanged. ADR assessment pending root identification; likely existing lifecycle invariant rather than new durable architecture. Private evidence /private/tmp/chat-pr3229-1b3d28-critical-artifacts.
Confirmed root in six owned original-production browser handoffs: full input initially has no MAIN ancestor; old global shell registration causes false Nested selection, then URL cleanup switches to Root, replacing input; shared migration idle-to-pending then removes it again. Minimal WebLayout context-only nesting and migration idle readiness preserve native/global override bridging. Red2fail46pass, green focused49; connected WebUI242, native101, extension101; production build dq8lbcap4xW7AMCBT0FzJ source100/969JSON/1134refs/0missing, six fixed handoffs one input/0populated removals, cockpit23pass0skip/retry/errors, original9conv28rows unchanged. Both independent source reviews clear. ADR required:no new durable rule; no persistence/security/API decision changes. Published integration evidence and remaining landing conditions are recorded in Docs/Reviews/CHAT_FINAL_RECOVERY_SETUP_HANDOFF_13673_13680_2026_10_14.md (TASK-13700 section). Full coverage and original journey still running; exact-head CI/merge pending.
Final exact blocking CI WebUI-owned coverage passes193 files/2380 cases in101.76s with existing4file14case skips. Final native101, extension101, focused WebUI49, configured types and normal hooks pass after reviewer-requested fixture cleanup. No unresolved GitHub review threads. Remote dev still4ffb0f01fd; PRhead remains1b3d284 and queue mode is explicitly unset. The second private journey harness rejected unsupported inventory key before running tests (excluded; all services stopped). Final private adapter now maps the actual journeys directory with unchanged oracles and strict guards; actual replay is in progress.
Local repair qualification complete: causal WebUI/native/extension tests,193file2380case owned coverage, six fresh production handoffs and23case cockpit pass; types/lint baseline/normal hooks pass. Original full source-journey private replay is EXCLUDED: after two startup/inventory failures, the third hits Media credentials before ingestion because reused static API rewrites bind19767 but the disposable backend has a different key. All owned services stopped; no source or auth assertion was changed, and no further local adapter retry is credited. The original unchanged full journey and all exact-head GitHub checks remain mandatory PR landing gates. Independent source reviewers A/B clear the two production changes; committed-head reviews follow.
Final review lead disposition found a second important reachable ordering defect at usePlaygroundSessionPersistence.tsx fresh-handle admission: cold B composer hydrates while saved fresh A restore awaits getConfig; Media accepts into B and advances only sourceSelectionRevision; delayed A adoption can then rehydrate old/empty text. Existing source/history merging must remain. Reopened this related handoff-readiness unit before edits. Add held-config real composer/draft-persistence regression first, then narrow fresh-address adoption fence; do not globally cancel historical restore. Current public50f600 critical46 and UX20 pass, but this new reachable race blocks merge until repaired and requalified.
Fresh-only restore ordering repair verified red2fail45pass then green47; exact B pointer publication and cold reopen retain complete source, old A bytes unchanged, historical merges untouched. Seven connected suites187/profile native/WebUI/extension; owned WebUI coverage193files2380pass14existing skips116.65s; fresh101-hash build djqgo1P99_kCc9XFkLCg9,969JSON1134refs0missing,6actual stable Media handoffs,23cockpitpass66.66s0skip/retry/errors; original9conv28rows unchanged. Types pass; zero lint errors,18 parent-identical warnings. No Python changed relative to broader parent-identical Bandit qualification. Both independent working-diff reviews clear. Remote dev advanced d7e43120e6; final topology-preserving rebase, exact committed reviews and all current-head checks remain mandatory.
Integrated current remote dev d7e43120e6 with30 merge commits preserved; exact41incoming-path delta and no frontend source changes. Six historical conflict resolutions verified exact prior merge tree.342 current combined workflow/queue contracts, actionlint and normal hooks pass. Fresh152-file Bandit7077findings0scanerrors; no differences for nonintegrated files; incoming2CI files have no additional findings versus currentdev, including literal GitHub token expression. No new production findings. Selective exact committed-head integration reviews clear; all current-head GitHub checks/merge remain pending.
Implementation qualification complete: both shell/migration lifetime and saved-fresh/source ordering roots fixed; actual current-head f2b9 critical46/UX20 and all7required statuses pass. Complete source and durable pointer/reopen checks pass; broader unrelated incoming queue-concurrency guard failure is tracked/fixed separately in TASK13705. PR landing still requires final current-head broader CI and pending task-renumber exception.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Preserved Media handoff composer lifetime through shell/migration readiness and prevented delayed saved-fresh restoration from replacing newer accepted source text. Existing owner guards and historical source merging remain. Qualified fresh101-hash production build,6stable handoffs,23cockpit cases,9protected conversations/28unchanged rows,2380WebUI-owned cases and187connected cases perprofile. Actual published f2b9 passes46critical/20UX cases and all7required statuses; separate broader guard integration and task-ID repair remain PR landing work.
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

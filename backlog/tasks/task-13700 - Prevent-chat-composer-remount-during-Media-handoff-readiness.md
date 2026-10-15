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
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR https://github.com/rmusser01/tldw_server/pull/3229; failing head 1b3d2846089326a4be656c4476c58186c19bff30, critical run38696151310 job116293111956. Trace records full value188110ms then empty188441ms and changed React useIds. Root not yet confirmed. Scope: shared useMigration and its WebUI/native shell callers if the migration admission lifecycle is confirmed; source oracle stays unchanged. ADR assessment pending root identification; likely existing lifecycle invariant rather than new durable architecture. Private evidence /private/tmp/chat-pr3229-1b3d28-critical-artifacts.
Confirmed root in six owned original-production browser handoffs: full input initially has no MAIN ancestor; old global shell registration causes false Nested selection, then URL cleanup switches to Root, replacing input; shared migration idle-to-pending then removes it again. Minimal WebLayout context-only nesting and migration idle readiness preserve native/global override bridging. Red2fail46pass, green focused49; connected WebUI242, native101, extension101; production build dq8lbcap4xW7AMCBT0FzJ source100/969JSON/1134refs/0missing, six fixed handoffs one input/0populated removals, cockpit23pass0skip/retry/errors, original9conv28rows unchanged. Both independent source reviews clear. ADR required:no new durable rule; no persistence/security/API decision changes. Docs/Design/IMPLEMENTATION_PLAN_chat_handoff_lifecycle_13700.md tracks remaining integration. Full coverage and original journey still running; exact-head CI/merge pending.
Final exact blocking CI WebUI-owned coverage passes193 files/2380 cases in101.76s with existing4file14case skips. Final native101, extension101, focused WebUI49, configured types and normal hooks pass after reviewer-requested fixture cleanup. No unresolved GitHub review threads. Remote dev still4ffb0f01fd; PRhead remains1b3d284 and queue mode is explicitly unset. The second private journey harness rejected unsupported inventory key before running tests (excluded; all services stopped). Final private adapter now maps the actual journeys directory with unchanged oracles and strict guards; actual replay is in progress.
Local repair qualification complete: causal WebUI/native/extension tests,193file2380case owned coverage, six fresh production handoffs and23case cockpit pass; types/lint baseline/normal hooks pass. Original full source-journey private replay is EXCLUDED: after two startup/inventory failures, the third hits Media credentials before ingestion because reused static API rewrites bind19767 but the disposable backend has a different key. All owned services stopped; no source or auth assertion was changed, and no further local adapter retry is credited. The original unchanged full journey and all exact-head GitHub checks remain mandatory PR landing gates. Independent source reviewers A/B clear the two production changes; committed-head reviews follow.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prevented full-to-empty composer transitions at their source: WebUI root nesting now uses actual React ancestry instead of a retiring global registration, and shared migration readiness blocks initial idle before children mount. Preserved native global overrides, account/draft/history guards and original full-source oracles. Added causal regressions and validated a fresh100-hash production build, six stable handoffs,23 cockpit cases,9 protected conversations/28rows,2380 WebUI-owned cases and101 native/101 extension cases. Exact-head CI and merge remain the PR landing gate; original full journey is not claimed passed locally.
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

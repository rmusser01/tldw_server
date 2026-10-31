---
id: TASK-13211
title: Investigate repeated Buddy visual pack loads during live UAT
status: In Progress
assignee: []
created_date: 2026-09-06 16:57
updated_date: 2026-09-10 16:24
labels: []
dependencies: []
references:
- Docs/Reviews/MIGU_VOICE_FOLLOWUP_2026_09_06.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During physical Migu voice UAT, the floating Buddy lost its image after repeated visual-pack and session-list requests reached rate limits. Establish the trigger and preserve the Buddy through live state changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The initiating trigger is identified with reproducible evidence.
- [x] #2 A regression check verifies bounded visual-pack loading through live state updates.
- [ ] #3 Real browser validation confirms the Buddy image remains available without repeated pack-load failures.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Observe request metadata and instrument host lifetime/dependencies if needed.
2. Reproduce the trigger with a focused regression test before a minimal repair.
3. Verify focused frontend checks and real browser visuals; record limitations.
ADR required: no. Reason: investigation and routine lifecycle repair within existing Buddy rendering contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-09-06 physical visual check: repeated authenticated pack list/detail and live-session list requests every ~250 ms, ending in HTTP 429 and a visible 'Visual pack did not load — rate_limited' error. Source review could not establish the initiating trigger. Pack effect dependencies are persona identity, target availability and refresh nonce; local sprite frame cycling alone cannot explain the session-list requests. After rebase/HMR reload and reconnect, screenshot sampling and one real text provider reply did not reproduce the request loop. No speculative repair applied. Targeted BuddyShellHost + Persona route suites passed 129 tests. Remaining work: instrument host mount/dependency/event counts during an actual reproduced failure, then add a failing regression and repair. Bandit not applicable: this task changes only investigation documentation.
Voice follow-up PR created against dev: https://github.com/rmusser01/tldw_server/pull/2927 . This task remains open; PR creation does not qualify the outstanding floating visual acceptance. UAT session disconnected and temporary browser viewport restored.
2026-09-09 qualification on pinned merged dev 1fc19c7: independent Buddy artwork remained visible through fresh setup, Static/Dynamic changes, Watchlists/Research Workspace navigation and scoped replies. A current-process aggregate recorded no HTTP 429; detailed receipts and analysis are retained in Docs/Reviews/2026-09-09-buddy-v1-qualification.md and artifacts/buddy-v1-13227. Source investigation confirms no 250 ms loader retry; paired legacy pack/session reloads require host remount or normalized Persona/surface change. IndependentBuddyHost landed after the 2026-09-06 incident and is excluded as its original cause. Existing BuddyShellHost/usePersonaLiveControl/IndependentBuddyHost checks passed 81 tests. The legacy initiating trigger remains unreproduced; integrated legacy lifecycle instrumentation during the real failure is still needed. No speculative repair; all original AC remain open. Bandit not applicable to this investigation-only update.

2026-09-10 follow-up: investigate current dev 50c1f68957 in isolated branch codex/buddy-v1-followup. Review real route/host/service boundaries and instrument disposable live UI before choosing a fix. Existing ADR005 applies; no new architecture or speculative repair. Official MCP task_view was unresponsive; using CLI fallback.

2026-09-10 controlled browser diagnostics: 1280px mount, 1023px cleanup, 1024px remount with two development pack-list calls and one session-list call. Confirms breakpoint source of paired reloads, not original rapid loop. No initiating trigger or 429 reproduced; all original AC remain open. Source-bound details/timestamps in Docs/Reviews/2026-09-10-buddy-followup.md. Temporary probes not shipped; viewport restored.
Recovered incident frontend73640bbb89aed7d878d254bd622ca68f79923ad8 from the separate local tldw_server checkout. The real route/context/host/live-control integration kept exactly one pack-list, detail and session-list across24 simulated voice/tool transitions at250ms; deliberately taking the route offline and back creates exactly one additional request set. Replaying the four historical source files also stayed stable. This is a bounded lifecycle regression, not an established initiating cause or physical voice acceptance. No production behavior was changed; AC1/AC3 remain open. Details and executable check in Docs/Reviews/2026-09-10-buddy-lifecycle-regression.md.
PR #2941 Qodo review: replace the 24 real-time waits with a fixed Vitest clock and explicit 250 ms advancement, assert exactly 6000 ms elapsed, and restore real timers before reconnect checks plus failure cleanup. The focused test passes in 0.97 seconds; this is deterministic bounded-load coverage, with no new claim about the historical trigger.
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

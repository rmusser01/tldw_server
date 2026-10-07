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
- [x] #3 Real browser validation confirms the Buddy image remains available without repeated pack-load failures.
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
September 30 UAT on reviewed server source eaebb194b717d3adc335dbca8961bc9ef5884ac5 plus the two-file catalog repair: the actual legacy Research Assistant Persona Buddy was exposed by temporarily detaching the independent test Buddy, and its reviewed Pixel Migu pack was activated. The rendered image completed at 128x128. Actual browser pointer dragging moved the shell from 1044,96 to 829,247; the implemented Home control reset it. Across a 6 minute 25 second window, legacy loading made two pack-list, two pack-detail and one live-session-list requests, with zero failures/429. Original workspace attachment restored. Private receipt: /private/tmp/buddy-all-uat-20260930/legacy-persona-visual-uat.json and legacy-persona-buddy-dragged.jpg. This qualifies AC3 real-browser artwork availability; AC1 stays open because the historical 250ms initiating trigger remains unreproduced. No human voice states, speculative production repair or Done status claimed.
2026-10-03 UTC controlled Fast Refresh observation on published source f21f1160dd0a84820ec46e730cd304af048bda8d. In a disposable frontend copy with temporary loader diagnostics, three edits to a diagnostic revision each caused one visual-pack-list dispatch and one live-session-list dispatch. Component ref identities, Persona identity, target availability and refresh nonce stayed unchanged; each refresh completed in 2.4-3.8 seconds. This establishes a development-refresh mechanism for paired reloads without a new component instance, not the historical 250 ms initiating trigger. Across startup, diagnostic installation and these trials, real authenticated loader requests returned HTTP 200 (7 lists, 7 details, 5 session lists), with zero Persona HTTP 429. Artwork completed at 128x128. Escape close/reopen retained the unsent control draft and added zero loader events (33 before/after). The normal Persona startup wrote one setup-event into the copied profile; no Start, Connect, Send, microphone or provider action was taken. Both temporary listeners were stopped, the diagnostic source was restored byte-for-byte, and all nine prior profile database hashes stayed unchanged. Evidence and runnable assertions: /private/tmp/buddy-hmr-uat-20261003/browser-result.json, persona-request-metadata.json, controls-comparison.jpg, verify-result.py and cleanup.json. AC1 remains unchecked and the task stays In Progress. Existing ADR005 and ADR046 apply; no new ADR or production repair. Bandit is inapplicable to this investigation note and temporary TypeScript diagnostics.
Authorized repair plan: IMPLEMENTATION_PLAN_buddy_effect_replay_13211.md. Add failing effect replay regressions, retain per-instance automatic loader requests, preserve explicit refresh and stale ownership, then repeat actual Fast Refresh UAT. ADR required: no; existing ADR005/046 govern unchanged ownership. No new storage or global cache.
Implemented the reproducible effect replay repair under the requester instruction to fix remaining issues. Pack loading retains one promise per normalized Persona/activation nonce in a mounted lifetime. Session autoload retains its scope request; pending explicit reloads also reattach after replay, with unchanged generation/request fences. Identity changes, activation, explicit reload, disable/re-enable and true remount still fetch fresh data. No storage or global cache. Four new replay regressions failed on old source; review found an explicit reload hang, reproduced with two failing resolve/reject tests and repaired before publication. Final focused scope: 124 frontend tests passed. Scoped ESLint: zero errors/new findings (one unchanged host dependency warning); TypeScript: zero owned diagnostics, four identical existing dependency declaration errors at base and head. Independent review found no remaining actionable defects after five ownership checks. Bandit is inapplicable to TypeScript-only source/tests.
Actual browser UAT on source hashes in /private/tmp/buddy-replay-fix-uat-20261003/setup.json: three Fast Refresh cycles retained the same host and hook refs, Persona research_assistant and nonce0; each added zero pack-list and session-list dispatches. Loaded image stayed128x128, and the exact unsent draft survived closed controls, refresh and Escape/reopen. Original source yielded one pair per refresh. Browser/source-bound receipts and executable assertions are browser-result.json, persona-request-metadata.json, repaired-controls.png and verify-result.py in that private directory. The original September6 250ms trigger remains historically unproven; this repair closes the independently reproduced effect replay mechanism. AC1 stays open for that historical attribution, and no human speech or audibility is claimed by this experiment. Existing ADR005/046 govern unchanged ownership; no new ADR required. Plan: IMPLEMENTATION_PLAN_buddy_effect_replay_13211.md.
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

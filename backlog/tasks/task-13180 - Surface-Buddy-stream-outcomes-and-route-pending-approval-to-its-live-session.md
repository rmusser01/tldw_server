---
id: TASK-13180
title: Surface Buddy stream outcomes and route pending approval to its live session
status: Done
assignee:
- '@codex'
created_date: 2026-09-05 15:39
labels: []
dependencies: []
references:
- Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md
- Docs/superpowers/specs/2026-05-20-persona-buddy-interaction-prd-design.md
priority: high
updated_date: 2026-09-05 19:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After browser transport repairs, a real Buddy greeting receives a tool_plan frame, clears the composer, and leaves the compact shell with no visible plan, reply, or approval-needed state. The shared live-control hook does not consume incoming WebSocket frames. This fails usable feedback and urgent-state expectations in the Buddy interaction PRD; approval execution must remain explicit in the full Live view.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A real Buddy text send produces visible pending, outcome, or error feedback associated with the correct persona and session.
- [x] #2 Incoming approval-needed state is shown and opens the corresponding full Live session without automatic approval or tool execution.
- [x] #3 Late or unrelated session frames cannot overwrite current Buddy feedback; targeted regressions and real backend UAT cover the interaction.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace existing Persona stream envelopes and full Live session/approval handoff; reproduce missing compact feedback with actual protocol-shaped tests.
2. Project current-session pending/reply/plan/error feedback in the existing live-control hook, reject stale or unrelated frames, and render a compact accessible status/outcome area.
3. Route review-needed state to the exact existing full Live session with explicit approval; never approve or execute from Buddy.
4. Run targeted Node20 tests, scoped lint, and real backend/browser UAT before marking acceptance complete.
ADR required: no (pending contract review).
ADR path: N/A; existing Docs/superpowers/specs/2026-05-20-persona-buddy-interaction-prd-design.md governs compact feedback and full-Live approval ownership.
Reason: implement the existing interaction and session contract without changing authorization, storage, or provider policy. If exact session handoff requires a new contract, document that decision before implementing it.
ADR required: yes. ADR path: Docs/ADR/045-persona-live-pending-plan-handoff.md. Reason: bounded read-only latest pending-plan projection in authenticated active owned session detail. Route session_id to explicit Connect, hydrate with every step unselected, retain all existing confirmation policy checks. TDD ownership/lifecycle/expiry/bounds and exact route/hydration tests; scoped frontend/Python lint/Bandit.
Envelope correlation follow-up under ADR045: return bounded optional client_message_id on events emitted for user_message using task-local context scoped to that awaited turn, including correlated failure notices. Timer tasks inherit original context; no mutable session-to-turn map. Voice and manual confirmation retain absent correlation unless their own turn supplies it. Queue two same-session user messages and prove their plan/notices keep distinct IDs; verify confirmation resets correlation and terminal/failure feedback remains scoped.
ADR-045 addendum: an exact Buddy locator may show a transient, Live-only review pane while setup remains incomplete. Restrict it to the matching selected persona; keep Connect/selection/Confirm explicit, offer Return to setup, and avoid setup success or detour analytics. Targeted StrictMode and superseded-connection tests cover the actual UAT mount issue and stale catch/finally race. Preserve server-derived persisted-session origin when policy denial rewrites a plan and verify revocation after hydration still rejects confirmation.
<!-- SECTION:PLAN:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Buddy now presents bounded turn-scoped stream feedback and routes review to the same persisted Live session. Explicit Connect restores its bounded latest pending plan with no selected tool steps; confirmation retains owner/active-session checks. ADR045 defines the read-only projection and incomplete-setup review detour. Real browser send/review/cancel/stop passes on the final snapshot, supported by214 frontend and147 Persona targeted Python tests. Existing repository typecheck/lint baseline findings and remaining voice acceptance are documented.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Full Live handoff implementation (ADR-045): Buddy's route carries persona_id/tab/session_id only; explicit Connect resumes exactly that session and owner-authenticated active detail hydrates the latest retained pending plan with all steps unselected. Projection is non-consuming, expiry-pruned, 100-step/64-KiB bounded, detached from runtime arguments, and absent from list/export. Server-derived persisted-session origin survives policy-denial rewrites; confirmation rejects terminal, deleted, or no-longer-owned persisted sessions before consumption while retaining legacy runtime-only behavior. Typed-turn event envelopes echo bounded client_message_id with task-local context; queued same-session turns and failure notices remain correlated, context resets for legacy confirmation. StrictMode effect setup restores mounted state; stale connect catch/finally and post-await writes cannot affect newer attempts. Incomplete setup uses a matching-persona Live-only transient review pane with Return to setup and no setup completion/detour analytics. Verification: 147 targeted Python tests across session manager, session detail, WS, and dialogue-tree runtime passed; 95 route/utils frontend tests passed under Node20 CI config; final focused StrictMode/review/overlap rerun 5 passed. ESLint has zero errors and no changed-line findings (87 existing warnings); Ruff has 10 existing signatures and no new ones, fatal-error subset clean; Bandit zero findings on 3 production Python files; git diff --check clean. Added the observed StrictMode/overlap incident to lessons-testing-evidence.md. Browser/provider/microphone UAT remains separately tracked by the main task; no synthetic-microphone acceptance claimed.
Final real browser acceptance on backend run backend-1788635007440317000, child20594 exit0: correlated synthetic text showed review feedback; explicit Connect restored identical session5463d886-f01d-4943-91ce-52a8f3e6caa8 and pending plan11938e31a16240d28e51d762e235c8d2, tool unselected, setup incomplete, no approval or execution. Cancel/Stop/Disconnect cleared state. Source diff independently matched launch after exit. Final combined frontend214 tests/8 suites pass. Scoped lint has no new findings; whole frontend typecheck retains80 unrelated diagnostics. Evidence and acceptance limits in Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md. Server microphone/provider audio remains unaccepted and outside this text feedback task.
Handoff ADR renumbered043→045 before rebase because latest dev independently allocated043 to llama.cpp snapshots. Scope unchanged.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
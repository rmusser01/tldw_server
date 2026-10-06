---
id: TASK-13504
title: Keep Character Library entry intent from changing the source conversation
status: In Progress
labels:
- chat
- ux
- character
- webui
- extension
dependencies:
- TASK-13503
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Within the approved all-ten chat UX program, make Character Library current/new-tab entry a route intent until the destination accepts a supported character setup. Readiness blockers, cancellation, popup failure, missing cards and delayed work must preserve source assistant, typed draft, transcript, H1 owner and settings. Reuse existing scoped card transport, guarded native preflight, acknowledged identity writer and draft retention; no parallel conversation framework. Quick Chat create/promotion remains a separate unit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Library Chat and Chat in new tab do not mutate source selection, transcript, owner, draft or model/prompt settings; blocker dismiss is local and blocked popups preserve source.
- [ ] #2 Fresh character route loads a verified owned card before a guarded supported destination transition; missing cards and held work after source, owner or route changes cannot clear or rebind source.
- [ ] #3 Accepted fresh entry preserves the exact source draft and uses existing native preflight, navigation and identity ACK before releasing prior ownership, with explicit recovery on rejection.
- [ ] #4 Relevant red/green mounted regressions, client type gates, no-new touched lint, both independent reviews and exact source-vs-native limits are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved design Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md; ADR-049/050 owner and persistence constraints and ADR-059 official backlog-py editor apply. Baseline latest dev f225562fc26 integrated153bc4e00d; compact checkpoint closed7f68a6633c. Checked local/dev task paths and refreshed all open PR paths: highest local13503/openPR13502, no duplicate bounded Library entry task found. Plan Docs/Plans/IMPLEMENTATION_PLAN_character_library_entry_2026_10_05.md. Two already-authorized independent reviewers are tracing the flow read-only. No Python implementation expected; Bandit N/A unless scope changes.
Stage 1 red/green: seven mounted Library entry regressions first failed then passed; current/new-tab actions now pass route intent without writing source identity or clearing H1/Core/Session/prompts/model settings. Stage 2 reuses the existing native apply boundary. Latest targeted lifecycle run: 24 passed across actual Form/composer and root route suites (/private/tmp/chat-library-lifecycle-green.log). Review found and corrected real setter entry-order guard, action-epoch recapture, accepted destination hydration, missing A to accepted B route retirement, retired standard workflow, and current-response refusal. Tests exercising actual useDraftPersistence use acknowledged Map IO; native private storage/cold-tab/popup/focus/package behavior remains unproved. Two authorized reviewers are rechecking current candidate; full suites/client types/locales/lint still pending.
Independent review found two further native-source hazards: saved URL retirement used destination IDs after reset, and queued requests bypassed pending entry. Saved URL regression failed then passed after restoring ACK-before-pre-reset source event ordering. Queue preparation and auto-drain regressions failed then passed; direct Run, central submit, panel explanation and auto-drain reuse the live blocker. Queued source requests remain available for explicit review. A transient blocker appear/disappear test reproduced held preparation reviving; existing assistant action revision now permanently retires that dispatch fence (4 focused queue tests pass). Actual root/H1/Core coordinator suite passed 75 tests; Form acceptance is explicitly stubbed there. Its new fixture failures were corrected translation/metadata names and real H1-qualified target delivery, not production failures. WebUI typecheck and extension compile passed before final revision/type-only cleanup; final gates pending. Latest remote dev advanced to 7226596cca, fetched for clean checkpoint integration. Bandit N/A for this TS/JSON/docs scope.
Pre-integration broad run: 612 passed / 11 failed across 26 suites. Corrected obsolete assertions that Library clears source, missing cards fabricate identities, and consumed unsaved routes recreate commands; unavailable-model tests now use explicit provider semantics. Newly supplied fake isCurrent was unstable and caused actual picker catalog invalidation loops: made the fixture lease stable like the real owned hook, preserving production picker guards. The run also included the then-unfixed queue ABA; its targeted red/green is recorded separately. Complete corrected root/Form/queue suites now pass 134 tests (4 suites); corrected full first-time Library handoff and Casual picker checks pass 8 targeted cases; actual coordinator full 75 passed. Final WebUI types, extension compile, diff check and no-new lint pass. Both independent reviewers found no remaining Important/Critical issue in this bounded source change and retain native/composition limits. Proceeding to local reviewed code checkpoint, integrate fetched dev7226596cca, then repeat relevant full gates on that merged baseline before closure.
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

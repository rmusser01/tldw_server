---
id: TASK-13458
title: 'Chat UX stage 5: verify complete novice and power-user journeys'
status: In Progress
labels:
- chat
- ux
- webui
- extension
- nng
dependencies:
- TASK-13454
- TASK-13455
- TASK-13456
- TASK-13457
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate all ten findings and enhancements end to end, then apply bounded polish from evidence. Run a complete surface/state matrix with the real backend lifecycle and a deterministic provider, preserve owner/privacy gates, and update the audit with verified changes and remaining limits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Start/name/leave/find/continue and choose/import card/greeting/two turns/reopen/change scene pass on WebUI and full extension, with compact sidepanel coverage and both handoffs where supported.
- [ ] #2 Provider error, midstream stop/interruption, offline/reconnect, save/load failure, conflicting histories, account/workspace change, custom context capability gates, and draft recovery have recorded outcomes without unacknowledged-save claims or silent replay.
- [ ] #3 More than five interleaved character sessions and a large seeded history, 320/390/desktop widths, 200 percent zoom, both themes, keyboard, and screen-reader checks have reproducible evidence.
- [ ] #4 Existing meaningful unit/integration/e2e, typecheck, lint, builds, and touched-scope security checks pass; test known-defect wrappers are converted only after independent confirmation the behavior is fixed.
- [ ] #5 Before/after screenshots and NN/G review cover all ten issues and approved enhancements, preserving Pro capability and temporary-chat privacy; all failures and unrun checks are explicit.
- [ ] #6 A concise novice/power-user study script defines uncoached tasks, completion, wrong-chat selections, assistance requests, post-task ease, and proposed two-minute first message / thirty-second reopen targets; participant results are reported only if actually collected.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Program: TASK-13453. Design: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Base: origin/dev 75ab224081bf140ef52017c1a9b0a04f6878d488, verified 2026-10-04. User approved equal audience priority and all ten findings plus enhancements in five stages. Written-design approval and the requested critical review are recorded on 2026-10-04; application implementation has not begun. ADR required: no for the client remediation; ADR-049 and ADR-050 govern owner-bound history and native fork boundaries, ADR-059 governs canonical task editing. Reassess if implementation changes public APIs or persistence authority. Reuse existing shared UI, loaders, presets, and test harnesses; preserve temporary-chat privacy and account/workspace fences. All findings and named enhancements. Run the real server persistence/ownership lifecycle with a deterministic mock provider; do not treat mocked browser success as server proof. Reuse existing harnesses from #3139/#3137/#3136 when available. No new telemetry, study recruitment, or test framework.
Pre-implementation review adds integrated checks for lost creation/append acknowledgements and reload, greeting preview versus committed stored rows/context, settings local success/server failure/unknown and unresolved-only retries, outage-safe cached mirrors/local DB unavailable, and each transfer mechanism including destination conflicts and partial import. Source-derived scenarios are risks to test, not newly observed incidents. Report native-shell and participant checks only when actually performed.
Begin actual UI walkthrough during ongoing implementation to verify first-visit and character-opening behavior rather than relying on source/test doubles. Latest dev27ce976387 integrated in ca39b897de; fetch unchanged. Use a separate local worktree dev server/port and fresh-origin browser storage; do not disturb existing runtime or primary checkout. Broad Stage5 acceptance remains open.
TASK13454.9 live IAB /chat: absent backend19765 retains two-line input through Retry/re-failure; primary Send disabled; compact readiness banner keeps composer visible at1280x720. Screenshot /private/tmp/chat-ux-readiness-draft-retained.jpg. No backend/provider, native packaged extension, VoiceOver, mobile/zoom or uncoached-study closure is claimed.
Latest-dev actual first-time /chat missing-model check: clicking Send shows Please select a model, retains exact two-line draft including trailing spaces, and leaves Choose model available. Isolated backend has no provider; no generated reply claimed. Screenshot /private/tmp/chat-ux-first-save-missing-model-retained.jpg.
2026-10-07 live verification preparation: current remediation HEAD 52ea5951b44457c787ba850b11f6661142b57ef3 includes freshly fetched origin/dev 5ec8c7939f46d6baecf8767caf070cb4a2eb335c. Completed root repairs TASK-13531/13532/13533 are committed; the full stage stays In Progress. Existing APIs on 8000 and 19765 are older runtime copies and are not latest-dev native evidence. Preparing fresh loopback API 19767 and deterministic mock provider 19768 with exclusive empty dotenv, tracked configuration defaults, isolated disposable databases, and synthetic credentials; reuse the existing in-app browser tab and a fresh web origin without opening Chrome. No native packaged extension, VoiceOver, physical touch, participant, or broad matrix closure is claimed. ADR-049/050 and ADR-059 remain governing; no changed architecture decision is needed for runtime verification.
2026-10-07 native journey checkpoint: TASK-13534 shared cookie guard repair restores actual search/share reads; 155 relevant backend tests pass and ordinary saved Chat survives title search/reopen/continue/reload. Imported synthetic Mira v2 card; the in-Chat picker commits its greeting and two H1 turns, then restores the same character and all five server rows after reload. Evidence /private/tmp/chat-native-mira-roleplay-reload-20261007.png and -server.json. Manager Chat-as launch silently retires during passive connection/catalog hydration and leaves the prior standard chat; separate root repair remains open. Latest origin/dev advanced to2c5f19d032 after the native checkpoint; integration pending. This does not close all Stage5 journeys or unrun packaged-extension/assistive-technology/physical-touch/participant gates.
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

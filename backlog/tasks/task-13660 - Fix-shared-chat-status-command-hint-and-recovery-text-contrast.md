---
id: TASK-13660
title: Fix shared chat status, command hint and recovery text contrast
status: Done
labels:
- chat
- accessibility
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current fresh Casual /chat scan exposes low-contrast command shortcut, recovery action and model health text. Fix the shared text owners with existing readable foreground tokens, including warning model states, then qualify fresh/picker states before retiring obsolete axe entries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared command hint, recovery action and model status text meet AA across default light/dark themes.
- [x] #2 Model warnings retain truthful text and status styling, recovery scope/actions and command keyboard behavior remain intact.
- [x] #3 Connected tests, source compilers, scoped lint, independent review and current UI qualification pass; only genuinely absent chat baseline entries are removed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR3229. Current actual fresh Standard chat at1280x720/light/Casual, Customize and sidebar collapsed: shortcut4.43, Recover drafts4.06, Healthy2.96. Existing6chat baseline entries remain unchanged until current exact-state evidence. Actual UI used public Casual and New saved chat; original server rows remain protected. ADR required:no, shared presentation token corrections only; no theme palette/API/ownership change.
Shared caller scope: ChatHeader via Header, recovery via Full/Compact Form, model picker via Full and two Research panes. Existing native73/5suites and WebUI73/5suites pass with0skips, including absent/unqualified/offline/checking-model truthfulness and recovery guards. Both normal source compilers pass.3authored TS scopes0introduced lint findings,0errors,3 unchanged warnings. Independent frontend bounded review clear; exact emitted fresh/picker scans and ratchet cleanup remain pending. No global Button variant or palette change.
Final emitted X7gAVx9PP9idmnpota__c: connected actual full fresh/picker0serious/critical in defaultlight/dark at1280x720/Casual. First dimconnecting scan excluded; source foreground and mounted warning truthfulness covered, not full native unconfigured-provider journey.73/5suites each runtime; both source compilers;3scopes0introduced static findings. Token/assets/budgets pass:800manifests1127refs0missing,590.2/869.0KB. Exactly6obsolete chat entries removed, allNotes entries unchanged; actual compareRatchet6before→0after;11ratchet unitchecks pass. Final frontend independent bounded source/baseline review clear. Original9/28 and separate synthetic4 canonical responses remain deep-equal. Review Docs/Reviews/CHAT_SHARED_TEXT_CONTRAST_13660_2026_10_13.md. Broader roleplay/history/offline, master/H2, wholePRreview/integration/CI remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Repaired shared command hint, recovery action and model status text using existing foreground tokens while preserving all handlers/warnings. Current complete fresh/picker theme scans0serious/critical;6obsolete chat exceptions removed without weakening scanner or changingNotes. Existing tests, compilers, static/source review, build and canonical preservation pass.
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

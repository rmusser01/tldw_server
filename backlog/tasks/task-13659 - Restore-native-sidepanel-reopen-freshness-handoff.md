---
id: TASK-13659
title: Restore native sidepanel reopen freshness handoff
status: Done
labels:
- chat
- extension
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Native H1 presence bypasses sidepanel show checks, so cold restoration of an older cursor can silently omit turns added elsewhere. Compare the admitted current manifest with the saved latest-seen boundary and reuse H1 stale-choice handling without reparenting or discarding shown data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A restored native view behind its saved latest-seen boundary exposes the existing guarded H1 latest-or-branch choice.
- [x] #2 Known historical positions, explicit pending branches and retired owners are preserved without extra fetch or automatic selection.
- [x] #3 Connected native/WebUI tests, both compilers, static gates, independent review and actual reopen verification pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR3229. Original CI XP08 failure115425762888. Source sidepanel-chat-freshness skips show/focus whenever selection.checkFreshness exists; actual checkFreshness compares fresh captured version, which cannot detect an older restored cursor against its saved latestSeen when its new manifest already includes newer turns. Legacy mounted route double omits checkFreshness and hides this arm. ADR assessment: no new ADR; reuse current H1 holdStaleSend/view/fence/source leases and explicit user choice, no new persistence/receipt protocol.
RED2 proves compact seam and actual H1 controller bypass; 160 filtered tests excluded from RED count. Full native179/4suites and WebUI179/4suites pass,0skips. Both source compilers pass;3 authored TS files0introduced lint findings,0errors,3 inherited warnings. Both independent bounded reviewers found no Critical/Important findings; outdated universal automatic-advance header corrected. Actual old artifact93hY3Hu5jwB2p2_20kX5S cold UI reproduces original answer visible/new reply false/stale notice false; owned synthetic API chat has4linked rows. Fixed artifact9cwaBPfaN8uHtMtjgl9eK cold UI preserves2old rows and offers both visible latest/branch choices. Explicit latest/input/server-write and final canonical checks pending.
Actual old build cold RED keeps2 but no later reply/warning; fixed build9cwaBPfaN8uHtMtjgl9eK keeps original view and shows existing guarded latest/branch choices. Explicit latest displays4, exact typed draft retained,0chat writes/0provider calls. Accepted cold4/empty input/no stale choice,0writes; synthetic canonical4 deep-equal and original9/28 deep-equal, baseline not rewritten. Tokens/assets/budgets pass:800manifests1127refs0missing,590.2/868.9KB. Native/WebUI179/4 each, both source compilers,0introduced static findings, both independent source reviews clear; header minor resolved. Review Docs/Reviews/CHAT_SIDEPANEL_REOPEN_FRESHNESS_13659_2026_10_13.md. Shared compact real-API proof is not installed-extension packaging or whole-PR approval.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed native restored-bookmark freshness in shared logic, preserving the selected parent/capture and offering existing guarded latest-or-branch choices. Causal real H1 and live old/new build proof;179tests per runtime, both compilers, static/source review and canonical preservation pass. Packaged CI and broader PR completion remain open.
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

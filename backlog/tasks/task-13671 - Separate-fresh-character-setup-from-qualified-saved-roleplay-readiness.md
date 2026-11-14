---
id: TASK-13671
title: Separate fresh character setup from qualified saved roleplay readiness
status: Done
labels:
- chat
- ux
- ownership
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bound Quick Chat still gates on live global model/catalog/settings and reads live Scene before the saved native shortcut. Split qualified saved source from prospective setup, preserve original account/owner/view/lease/draft/request/connection fences, and use model-first prospective guidance without applying live setup gates to saved authority. Existing ADR049 and frozen-roleplay contracts govern; no new durable persistence or permission rule.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Qualified bound popup remains usable after clearing global model/catalog and ignores unrelated global model/settings/Scene preparation while using the original saved source
- [x] #2 Fresh creation still verifies configured model/provider/Scene and retires changed preparation; bound account/view/input/connection/request retirement still refuses
- [x] #3 Prospective model-before-character guidance and saved full-runtime summaries match the actual source; connected native/WebUI tests and actual callers qualify without guessed frozen configuration
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Independent frontend source review identifies hook127-129/229/238-258 and CharacterDialogs995 liveglobal dependencies; helper168-205 and sendSavedRoleplay use frozen source before live model/Scene validation. Existing second-turn ownership test148 expects fresh+native Scene reads and is independently confirmed obsolete; convert to zero new live Scene on bound send while preserving fresh settings/retention negatives. Highest fetched origin/dev/allopenPRfiles/local13670; reserve13671. Sourcebase419e12c277; PRhttps://github.com/rmusser01/tldw_server/pull/3229. Scope both fresh and qualified saved readiness, no SourceH2/futureparticipant expansion.
Source review clear; native23 connected suites768 and WebUI25 suites779 (including Button) pass. Both production-root type profiles clear; changed5TS lint0errors64unchangedwarnings0new. Combined current build AKxuR7TK7dFapUeh22FF2 /48source hashes/1132physical refs/0missing/tokens+budgets clear. Existing opt-in private native fixture had three distinct outdated-fixture failures; stopped and reassessed rather than repeat: original unconfigured durable Scene IO, then retained draft IO, then obsolete trimmed-input assertion after actual server canonical4rows succeeded. Existing native-plasmo-storage-fixture from connected native sender fixes the first two without bypassing real Scene/draft services. Actual latest original first input preserves whitespace; independent reviewer is validating source/raw protocol and fixture expectations before repair. Failed runs excluded from green. Synthetic cards cleaned after each; immutable baseline unchanged.
Final bounded evidence: Docs/Reviews/CHAT_FRESH_SAVED_READINESS_NATIVE_BUTTON_13671_13672_2026_10_14.md. Relevant connected tests, actual installed AntD controls, both production type profiles, migrated private fixture dependency types, source-matched production assets/tokens/budgets and independent source review pass. Native and WebUI actual private protected protocol passes with exact bytes/IDs and no live setup dependency. Original9conversations/28rows equal immutable baseline. TS/JSON/Markdown scope: Bandit not applicable. Separate master early-Stop/acceptance matrix and final whole-PR gates remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Separated prospective model-first setup from original qualified saved-character send readiness; preserved all owner/account/view/input/request/connection gates and legitimate fresh binding. Migrated existing private native fixture to real storage/current strict protocol and qualified both surfaces.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

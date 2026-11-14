---
id: TASK-13670
title: Guard provider recovery for already-bound native character sends
status: Done
labels:
- chat
- bug
- recovery
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The already-bound/private shortcut in native-history-character-send calls sendSavedRoleplay without the guarded error wrapper already used after a fresh bind. Mirror that existing wrapper only around the send call, preserve typed branch/context/settings refusals and original owner/view/draft/source currentness, and qualify second-send provider failure and retired completion behavior across callers. ADR assessment: existing ADR049 ownership and existing13664 error policy; no new durable architecture rule.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current bound provider failure uses the existing typed guarded recovery classification with retained original input and no replay
- [x] #2 Retired owner/view/account/input completion cannot publish recovery into a replacement view; typed preflight refusals stay typed
- [x] #3 Native/WebUI connected regressions, source/lint/security scope and bounded independent review verify the shared fix
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Independent bounded backend review identified shortcut178 missing existing post-bind catch509-511; popup caller280-287 relies on NativeHistoryRoleplaySendError for safe settings guidance and H1 currentness. PR https://github.com/rmusser01/tldw_server/pull/3229. Highest fetched origin/dev/all open PR files/local13669; reserve13670. Primary checkout untouched.
Final8 connected suites289 checks each native/WebUI0skips after13656guidance; final5root typesboth0, rootlint4TS0errors/8 unchangedwarnings/0new/0ignored. Independent source review clear, delegate-onlycatch preserves typedpreflight/private-record errors, signal+owner/view/lease/draft guards. Artifact gMU8OYvGFUNR0EXEfSAKn46hashes/154pages/emitted969JSON1129refs0missing/actualtokens/budgets590.4/869.1. Full17cockpit39.2s allpass. Extra seven-character qualification excluded after mock readiness probe401(two attempted probes, no source defects); dependent run4pass3fail notqualified. Reassessed actual mock source before another CI-profile start; owned19768 restored local-success PID24164/readytrue, backend19767/preview19769 unchanged. Native provider401 unit/journal proof remainsqualified; finalextra real-character run pending corrected authenticated probe.
Actual seven-character/provider401/originalRetry/coldrecovery cases allpass18.98s0retry/skip/flake, same currentgMU artifact also17cockpitpass. Both8connected289/0skips, final5root typesboth0, lint4TS0errors8unchanged0new. Bounded backend review clear; existing typedrefusal/privateprep errors unchanged. Exactmockauth prefix contract inspected before correctedprobe and qualification; oldfailedprobe/dependentrun excluded. LocalSuccess mock27104 restored, original9/28+4fixture unchanged. Newbound-popup liveglobal readiness bug identifiedindependently is separate tothiserrorwrapper and willbefixed beforefinalreview; normalhooks/commit pending.
Normal15-file exactowned hooks passed. Own complete feedbackplan retired. CI mock sk-prefix policy inspected; correctedprobe produced full7green; failedreadiness/dependentrun excluded. LocalSuccess27104 restored. No wholePR approval or merge claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added existing guarded error wrapper to already-bound saved-roleplay delegation only; preserves typedrefusals/privateprep errors, original cause and signal/owner/view/lease/draft/source fencing. Both8connected suites289 pass0skip; final5root typesboth0; lint4TS0new; currentgMU production artifact/tokens/budgets/assets and real17+7 browser flows pass. Original journal/input/append/IDs/retry/release remain unchanged; original9/28+4fixture preserved. Independent source reviews clear; normal15-file hooks pass; BanditN/A TS/JSON/Markdown. Separate bound-popup liveglobal readiness still needs repair before wholePR review/merge.
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

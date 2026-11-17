---
id: TASK-13651
title: Preserve explicit forced splash across preference hydration
status: Done
labels:
- chat
- bug
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Trace and fix reproduced explicit forced-splash cancellation without weakening later preference, disable, dismiss or unmount retirement. PR#3229.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Explicit forced request survives its own preference update hydration
- [x] #2 Later independent preference/disable/dismiss/unmount still retires held content
- [x] #3 Both client suites, connected callers, compilers and independent review pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED:1failed17passed in existingnativeSplash suites;forceddisabled case fails. No source fix chosen before preference/request trace. ADR assessment:no newADR expected; existing explicittrigger/preference retirement preserved. Plan:Docs/Plans/IMPLEMENTATION_PLAN_chat_forced_splash_13651.md
Causalsharedstorage RED3 then initialGREEN21. Reviewer caughtnativefirst-instance keybinding:RED2 then stablebinding. Nativeconnected129/9pass. WebUIconnected127pass2fail exposed function-basedwatchcleanup; correctedretirement plus awaitedheld-readsettlement, final129/9pass21.16s. Newcontrols covermigration ownwatch,currentkey/equivalentdefinition(writefailure),replacementkey+area,functionalupdates,same-normalizedvaluewatch andunmount. Same-keyarea swap is inheritednativeSDKlimitation;68currentcallersstaticarea,remountrequired iffuturecallerneedsit. Formatting3owned scopes preservesfulltype/executableAST andimportorder.
Finalcorrectactual-area native130/9pass0skip16.99sprocess0;WebUI142/11pass0skip18.60sprocess0 inclbothstorageprimitives. Callbackaritycompat corrected viaforwardedrestarguments. Independentreviewcaughtnativefirst-instance retention anddifferentdefaultareas;bothfixedwithcausalREDcontrols; earlierassumedSyncaliasqualification superseded. Source68currentcallersstaticareas;inheritedsame-keydifferentphysicalarea remountlimitexplicit. Final4-fileESLint0errors/warnings;extensioncompiler0;WebUIcompilerfinishing;BanditN/ATypeScript. Review:Docs/Reviews/CHAT_SETTING_HYDRATION_ROOT_13651_2026_10_13.md
Finalactual-area WebUI normal sourcecompiler0 aswell. Bothsourcecompilers and4-fileESLint0;finalboundedsourcereviewclear. No further sourceedits.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
SharedSDKinitialread andregistryderivedwrite racesfixed;native130/WebUI142connectedtests pass,bothcompilers andlint0,independentreviewfindingsresolved andnormalhooks clear. Existingstaticarea callersqualified;inheritedsamekeyphysicalareaswaprequiresremount. PR#3229.
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

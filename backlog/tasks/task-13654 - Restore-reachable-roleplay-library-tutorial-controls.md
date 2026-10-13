---
id: TASK-13654
title: Restore reachable roleplay library tutorial controls
status: Done
labels:
- chat
- ux
- onboarding
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix missing World Books tutorial anchors and mobile filter parity; follow extracted Characters and Flashcards component owners in existing contracts. PR#3229.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Compact World Books exposes status and attachment filters behind a reachable disclosure
- [x] #2 Tour targets remain reachable for compact, collapsed and empty libraries
- [x] #3 Extracted character and flashcard anchor assertions retain all expected targets
- [x] #4 Mounted behavior, both source compilers and independent review pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
InvestigatedactualTutorialRunner first-visible target resolution, extractedcomponent owners andmissingWorldBookToolbar search/filter anchors. Design:Docs/Design/2026-10-13-chat-library-tutorials-13654.md. Plan:Docs/Plans/IMPLEMENTATION_PLAN_chat_library_tutorials_13654.md. ADR assessment:no newADR; existingcomponent/disclosure/tutorial patterns preserved.
RED4fail17pass(targetcontracts3+compactabsence1). MountedGREEN35/4;connectednative/WebUIeach89/7pass0skipprocess0(14.05s/9.57s),actualclicksummary/visibleAntDitems/collapsednav/emptystate. BoundedreviewclearafterretainingnativeSummarymarker/focus. Ownedexistingpage19769/oneheadlesscontext resized390 nooverflow butbuildtyt0kJ9ulPTjWhuB47qr3 isoldercompiledpreview andlacksnewmarkers; NOTcurrentheadkeyboard/geometryproof. Currentbuildrefresh/recheck deferredtofinalintegration;nonewtab/window/context. FirstformatguardrejectedJSXwhitespace;noedit. Properemitted-JSXASTguard verifiesfinalownedformatting/importorderunchanged.
Bothnormal sourcecompilers0;8authoredTSscopes ESLint0introduced/0errors23existingwarnings;EnglishJSONvalid;whitespace0;emittedJSXformatASTunchanged. BanditN/ATypeScript. Boundedreviewclear. Review:Docs/Reviews/CHAT_LIBRARY_TUTORIAL_CONTROLS_13654_2026_10_13.md. Current-buildphysicalkeyboard/visualverification remainsparentintegrationitem;oldownedpreviewnotcredited.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Missing/movedtour targets andcompactfilter parity repaired;89tests ineachclient,compilers/lint/hooks andboundedreview clear. Current-buildkeyboard/visualcheck explicitlypendingparentintegration;no newbrowsercontexts. PR#3229.
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

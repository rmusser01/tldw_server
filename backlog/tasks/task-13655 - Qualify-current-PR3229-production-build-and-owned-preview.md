---
id: TASK-13655
title: Qualify current PR3229 production build and owned preview
status: Done
labels:
- chat
- verification
- build
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refresh isolated locked Bun production artifact from current source, verify budgets/tokens and run current-build keyboard/mobile checks in the one existing owned page. PR#3229.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current tracked source builds with unchanged locked dependencies and budgets
- [x] #2 Token and emitted asset checks pass
- [x] #3 Current-build World Books keyboard/mobile targets are verified without new browser contexts
- [x] #4 Original owned conversation data is preserved exactly
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Existingowned19769 servesolderbuildtyt0kJ9ulPTjWhuB47qr3;notcreditedcurrentheadproof. ReuseisolatedBunroot /private/tmp/chat-pr3229-bun-build-13613-clCuJd;nevermutatelivedeps/primarycheckout. ADR assessment:no newADR,verificationonly. RemainingE2E/CI/H2/finalreview/rebase/merge staysopen.
Currenthead43fb6875f9 trackedsourcecopy7986files, lockidentical. DockerproductionTurbopack65s builds154pages; tokensOK,shared590.2KB/600,heaviest868.9KB/900 unchangedlimits. Artifactbuildfq_kLRnDnfmZsXc77pQW7. Ownedpreviewfirststartuprequiredexplicitquickstartenv; thenreadinesscorrectlyrefusedDockerapp:8000 rewrite(ENOTFOUNDapp). No guardbypass/manifestpatch: separatesame-source .next-live-tier-pr3229-13655 build nowrunning withownedloopback19767origin; .next productionartifactpreserved. Existingonepage/context kept; originaldatasetverificationpending.
CurrentlocalpreviewbuildP0wWoFmCs_5eBdfwKji8N;tokens/budgetsunchanged. Actualbrowser390summary44px,list-item,Enteropensfilters/Spacecloses+focusretained/emptytargetvisible/nooverflow;desktop1280bothfiltersvisible/nodisclosure/nooverflow;mobilePNGinspected. Sameexistingpage/contextonly. NormalSettings restoredsealedownedfixturecredentialafterprofilesameoriginURLchanged;noauthbypass/automaticnamespacecontinuityclaim. Actual801manifests1127refs0missing;initialobsoletewebpackmanifestguessexcluded. Original9convos28rowsdeepequal afterallactions,baselineunchanged. Review:Docs/Reviews/CHAT_CURRENT_BUILD_PREVIEW_13655_2026_10_13.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Currentlockedproductionbuild/token/budget/assets verified;separatevalidatedloopbackpreview provesmobilekeyboard/44px/desktopparity;original9conversations28rowsunchanged. Sameownedpage/context;namespace repairdeclared;wholePRscope remainsopen.
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

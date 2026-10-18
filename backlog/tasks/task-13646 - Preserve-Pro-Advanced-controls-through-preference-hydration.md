---
id: TASK-13646
title: Preserve Pro Advanced controls through preference hydration
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The real native storage hook can publish an old initial value after Pro Advanced controls are opened, detaching the expert controls. Reuse the existing guarded presentation preference logic and retain the established key, verified initial read, watch ordering and visit-only recovery on storage failure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Controlled held native initial read reproduces the Pro disclosure race
- [x] #2 Pro waits for the original preference and retains explicit choice with existing key and area
- [x] #3 Storage read/write failure retains usable visit-only controls without overwriting unread preference
- [x] #4 Both modes and configurations, compiler checks and independent review pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR3229 TASK13645 fullnative934=933pass1ProdetachedParameterpresetsfailure. NativeuseStorage lateinitialget resetsvalue; existinguseCasualComposerDisclosure alreadyguardswatch/read andrecoversvisit-only. Highest13645 verifiedremote/open/local beforecreate. ADRrequired:no; UI preference race fix underexistingstorage/model semantics, no newarchitecture ordataauthority. Any fixturechanges remainlinked13645.
ActualheldnativeProRED1failed33skip earlytoggle; reusesguardeddisclosure forstablelegacyProkey, loadingdisabled/busy, visit-onlyreadfailure/nooverwrite, watchnewerwins, writefailurekeepssession. Selected7passed41skip; combinednative986passed10suites. Bothunconditionalstablehookcalls avoidmode-readinesscarry. BoundedreviewCLEAR; WebUIactive, originalkeys/areas unchanged.
Finalnative986passed10suites andWebUI986passed10suites. InitialWebUI984pass2newtestfail fromholdingallwatchreads; original-onlyholdfix retainsactualwatch/pre-writeIO; selectedWebUI5pass32unselected, affectednative7pass41unselected. Bothnormalclientcompilers0; diffcheck0; boundedreviewCLEAR. BanditN/A TS-only. Jointcommit13645/13646 for coherent source+fixtureunit; exacthooks pending. ReviewDocs/Reviews/CHAT_PRO_DISCLOSURE_HYDRATION_13646_2026_10_13.md.
Normalexact12filehooks allpasswithoutbypass. Owncompletedplan retired; jointreviewable source/fixturecommit13645+13646. WholePRremainingfindings/review/rebase/merge stillopen.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Qualified actuallocal/sync storagefaults and repaired Pro preference hydration viaexistingguardedlogic. Native986/WebUI986each10suites, affectednative7/WebUI5controls, bothnormal sourcecompilers0; boundedreview andnormalhooks clear. Preservedkeys/areas, watch precedence, visit-only storagefailure, originalrefusal/focusassertions. Physicalsync key/ACK/bytes verified. BanditN/A TS. Joint13645+13646 unit; wholePRstillopen.
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

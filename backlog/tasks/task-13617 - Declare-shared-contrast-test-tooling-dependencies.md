---
id: TASK-13617
title: Declare shared contrast test tooling dependencies
status: Done
labels:
- frontend
- tests
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Shared UI contrast tests directly import PostCSS and Tailwind but the UI workspace does not declare them, so clean frozen CI cannot collect the tests. Declare the existing workspace versions as UI test dev dependencies, update the frozen Bun lock, and qualify both clients emitted button color/contrast contracts without weakening assertions or adding new tooling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared UI explicitly declares the existing PostCSS/Tailwind test versions with a coherent frozen lockfile.
- [x] #2 Both contrast suites collect/pass from a fresh isolated frozen workspace graph.
- [x] #3 Normal client types, bounded review and normal hooks qualify the package/lock change; no runtime dependency or contrast threshold changes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no, test dependency ownership correction using existingtools/versions. Highest local/origin/openPR taskID13616checked. PriorCIshards4/8 cannotresolvepostcss inUIcontrasttests; cleanisolatedgraph reproduction beingrecorded. UserallremainingPR3229fixesauthorized; finalwholePRreview/rebase/mergepending.
Cleanfrozengraph baseline2failedcollection/missingpostcss; updatedfrozenBun1.3.2installpasses,18actualbothclientcontrastchecks pass. Bothnormalclienttypespassed. Packageversion/integritypaircomparison0added0removed; lockchangesonlyexistingidentityhoisting. Review Docs/Reviews/CHAT_CONTRAST_TEST_DEPENDENCIES_13617_2026_10_12.md; owncompletedplanretired/private/tmp. BanditN/A package/lock/Markdown-only. Boundedreviewpending; wholePR/CI/UX/H2/finalreview/rebase/merge remainopen.
BoundedreviewCLEAR:noCritical/Important/Minorfindings. Rootconfirmedbothnormaltypespassedafterfinalmanifest/lock. Normalexacthooks next; allwholePRgatesstillopen.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Declare existingPostCSS/Tailwind asUItestdevdependencies and updatecoherentfrozenlock.18bothclientcontrasttests andnormaltypespass; no dependencyversion/integrityidentitychange, no runtime/threshold/assertion change; boundedreviewclear.
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

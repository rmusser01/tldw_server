---
id: TASK-13619
title: Give Playground harnesses explicit query ownership
status: Done
labels:
- chat
- tests
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Search, sticky composer and most coordinator tests render Playground without its required QueryClient after authoritative model catalog adoption. Give each test one isolated real client, declare catalog IO, remove duplicate coordinator providers and preserve real settings exports. Retain all actual search, restore, scope, source-retirement and layout assertions; no production readiness or storage replacement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every Playground render in the three suites has one isolated real QueryClient and cleanup.
- [x] #2 Catalog IO and chat-settings exports are explicit without globally replacing storage or weakening readiness/source guards.
- [x] #3 Held restore/scope/route retirement, search and layout cases pass with all original assertions.
- [x] #4 Affected suites, both normal client types, bounded review and normal hooks qualify the unit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no, testprovider ownership under existingauthoritativecatalog decisions. Readonlyfrontenddiagnostic identifies3provider-familyfiles; coordinator4specialwrappedcases needoneconsistentclient. Highest local/origin/openPR taskID13618checked. UserallremainingPR3229scopeauthorized, wholePRacceptance/review/rebase/mergepending.
Final3suites110passed; bothnormalclienttypespassed; boundedreviewCLEAR. Allheldrestore/scope/route-retirement/search/layout assertionsretained. Oneclientpertest/cachecleanup, coordinator4duplicateinnerownersremoved, actualsettingsexportspreserved. No production/readiness/guard/storagealiaschange. Review Docs/Reviews/CHAT_PLAYGROUND_QUERY_OWNERSHIP_13619_2026_10_12.md; owncompletedplanretired/private/tmp. BanditN/A tests/Markdown-only; normalexacthooksnext. WholePR/CI/UX/H2/finalreview/rebase/merge remainopen.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Declare one realisolatedQueryClient perSearch/sticky/coordinator case, explicitcatalogIO andactualsettings exports; remove duplicate innerclients.110tests/typespass, negativecontrolsunchanged andboundedreviewclear; no productionguards changed.
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

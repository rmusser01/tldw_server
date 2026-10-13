---
id: TASK-13656
title: Qualify real cockpit E2E controls for Casual and Pro modes
status: In Progress
labels:
- chat
- e2e
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair four CI cockpit cases stopped by obsolete Pro-only disclosure assertion while preserving Casual mobile and expert Pro behavior and all send/recovery/provider assertions. PR#3229.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default Casual assertions use real Customize groups and primary controls
- [x] #2 Explicit expert case uses Pro and verifies actual Advanced group
- [x] #3 Existing send/stop/regeneration/provider/draft assertions remain unchanged
- [ ] #4 Mode-aware helper controls and relevant real browser flows verified
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
CI115425504086 fourcases/retriesstop helper862composer-options-toggle;sourcehelper unchanged/currentdefaultCasual. Readonlyreview mapsCasualCustomize andProexpertcase1765. ADR assessment:no newADR,testcontracts only. Plan:Docs/Plans/IMPLEMENTATION_PLAN_chat_cockpit_modes_13656.md
Stage2 source/helper checkpoint ready;Stage3 remainsInProgress. Actualhelperextraction+declaredlocator expectationadapter ononeexistingownedpage passesProprimary/full,Casualdesktop/mobile. DeliberateProAdvancedcyclepass;earlierfirstscriptedclickaftermodeswitchtimeout rootcauseunproven/notfixedclaim. CLItest-runnerexpectdynamicimportunsupportedinitialattemptexcluded. Playwright--list collects15/autostartfalse/nobrowser. ESLint0new/0errors36baselinewarnings;guardedhelperformatoutputunchanged. Boundedreviewclear;fullrelevantrealcases/transactions stillpending. Review:Docs/Reviews/CHAT_COCKPIT_MODE_FIXTURES_13656_2026_10_13.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

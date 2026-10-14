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
Published roleplay checkpoint b97aa526e8 now on PR3229. Stage3 actual15case cockpit runner active on owned production preview19769 buildU6RUXbj2-NEmmxWkcO7-4, mockprovider19768 restored local-success only after originalPID/cwd/command verification. Headless1worker/no retries/streamingcontrolsevidence required, no visible browser windows created. Private outputs /private/tmp/chat-pr3229-cockpit-13656-native-*; no completedresult yet. Original9convos28rows and declaredsynthetic4rows verifiedunchanged before thisrun. No wholePR/merge claim.
Actual15case native run9pass6fail0skip0flake374.99s, no assertion changes during run. Independent reviews confirm stale WebSearch Active copy (now Selected for next message), unqualified configured-model fixture blocks mobile Send, and implicit tracked-character greeting assumption (now explicit greeting/start-message choice). Focusreturn2183 is real UI regression; separate TASK13666. Native tracked persona is intentionally refused by H1/complete-v2; human chose unavailable choice with guidance, separate TASK13667. Stop/regeneration wait3053 may share unqualified model setup; retain streaming controls and original admission/settlement oracles. No failed cases qualified. Fixture updates remain TASK13656; full static/runtime qualification pending.
TASK13666 now Done: original prompt/model-setting restore/MCP cockpit case, plus real Close/Escape, passes on builduuJqBl4XVXSE2DePcX9jf with original5s focus assertions (1case5.6s0skip0flake). Original9/28 and synthetic4rowsunchanged. Remaining cockpit failures still require model qualification, web-search current-state copy, explicit greeting and persona capability alignment; full15cases not rerun or declared qualified. TASK13667 user-approved restriction in progress.
TASK13666 committed8856413673 qualified original prompt/model/MCP case including native Close/Escape focus. TASK13667 now qualifies revised user-chosen persona restriction and explicit public ordinary-overlay positive:actual2 browser workflows pass5.47sec on source-matched buildicj9ccdvWZDSuApuhAMKa,0 skips/flakes/retries; actual distinctive persona system prompt reaches ordinary completion, zero unsupported writes, original9conversations/28rows and synthetic4 unchanged. Historical6 failures now have2 qualified repairs. Four remaining cockpit failures:stale web-search Active oracle, mobile unqualified catalog seed, implicit character greeting expectation, stream-stop/regeneration admission seed. Whole cockpit15 cases not yet rerun/qualified; Stage3 remains InProgress.
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

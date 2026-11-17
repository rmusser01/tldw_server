---
id: TASK-13695
title: Remove global style-query overhead from model settings fixture
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3229 exact-head native shard4 timed out a real provider-selection/Save case at its unchanged10-second deadline;990other cases passed and isolated30cases passed. Trace interaction costs and scope accessible queries to their real semantic owner, preserving all production behavior, assertions and deadlines.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Measured evidence identifies the failed interaction cost or actual defect before a minimal repair
- [x] #2 Provider model scope, previous settings preservation, conversation prompt Save and owner retirement assertions remain unchanged at original deadlines
- [x] #3 Native shard4 context plus affected WebUI/extension tests and independent reviews pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no; expected test-only query/scope correction changes no durable UI or persistence rule. Existing accepted chat UX design and final handoff govern PR scope. CI log /private/tmp/chat-pr3229-4bcef3-frontend4-failure.log; job116216309918. Investigation may establish production cause instead; do not weaken timing or provider/account guards.
Evidence correction from independent A: isolated30-case143866ms run was the6cb dev BASE replay, not current head. Head fixture has51cases/230549ms; timed-out provider case10106ms, sibling9953ms. About7s of synchronous work preceded provider model-change publication. Source admission/Save guards remain intact. Existing global screen role queries traverse retained AntD tab panels; most mountSettings-created QueryClients lack after-test clear. Exact-head51-case native timing run in progress before instrumentation/edits; no production defect established.
Measured original exact-head native51RED7fail44pass304.24s. Diagnosticproviderpairretains10s and fails2; globalcombobox121stylecalls4.71/4.93s, almostallinrealgetComputedStyle. ScopedactualactiveModelARIApanel+API/modellabelForm.Itemreduces24calls0.825/0.838s. First full fixturerepair50pass1explicitScope10.546stimeout275.76s. A+BreviewconfirminitialConversation→Modelroundtripincidental: extractedsharedrealrender, existingConversationwrapperretainedforallothercases;2providercasesnowstartrealdefaultModelandassertselectedtab+hydratedold0.2beforeALLoriginalnewprovider/oldKEEP_A/new0.6/ConversationSaveoracles. Finalfocused4pass;providercases6017/6776msoriginal10s. Bfoundunusedcatalogqueryprefix; active-querypostconditionsRED2onoldwrites, thenfixedactive-prefixwritesandawaitedactualnotifyManager.schedulebarrierinsideasyncact. Bothcatalogcasesnowdriveeachactualobservedgenerationtransition. Alltemporaryprofilingremoved; owning-profile/fullshardqualificationpending. FirstESLintinvocationoutsidefrontendbasewasignoredandisnotpassingproof; correctedmonoreporootESLint0errors0warnings.
Qualification correction: CI dependency impact2713 exceeds500, so workflow shards438directlychangedfrontendtests; its native shard4is51files/991cases. Initial local --changed invocation selected a broader dependency context and was stopped via SIGINT only after verifying ownedPID59036's exact command. Its exit130/logs are excluded diagnostic evidence, never passing qualification. Extracted exact51headfilepaths from completedCIlog and replaying those withbyte-identical pinned deterministicsequencer, original15sshard/explicit10scase limits. No production/sourcechange followed this runner correction. MechanicalTypeScriptASTcheck retainsall143originalexpectarguments,147finalwith4additions,allexplicitdeadlinesunchanged,andnoprofiling.
Finalqualificationcomplete: exactCI51-fileNativecontext991passed0fail0skip; file-setequalityverified. Providerpair7055/6969ms atoriginal10s. CompletefixtureWebUI51pass0skip272.76s; extension51pass0skip271.98s,sequential. BothA/Bfinalworking-sourceCLEAR includingactualnotifyManagerbarrier/defaultModelrenderer. FinaltargetedESLintmonoreporoot0errors0filewarnings (existingNextpagesconfigstartupmessageunchanged); profilingandownedpinnedconfigremovedafterbyte-check/archive. Docs/Reviews/CHAT_FINAL_RECOVERY_SETUP_HANDOFF_13673_13680_2026_10_14.md recordsfullred/greenandexcludeddiagnostics. BanditnotapplicabletoTS-onlydelta; Pythonproductionunchangedandpriorsecurityqualificationretained. No productionUIbehaviorchange. Configuredsource/recordhooksandexactcommittedreview/publicationstillfinalsteps; PRmergeawaitsallnewheadgates.
Configured3-filehooksPASSincludingcanonicalBacklogformat,whitespace/privatekey/conflictguardsandlegacy/http-patchguards. Fresh97productionUIhashesmatch0mismatches. Sourcefixturefrozenafterall3owningprofilespass; exactcommittedreviewandnew-headCIremainlandinggates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
ReducedrealDOMquerycostbyowningfield/tabscope, preserved143originalexpectargumentsandalltimeouts, added4checks. ProviderjourneyusesrealdefaultModelwithold-viewhydrationbeforealloriginalscope/Saveassertions. Catalogfixturesdriveactualactivegenerationqueriesandawaittheirrealnotifications; ownedclientsretireaftertestunmountwhilewithin-testremountcachepersists. ExactCI51files991pass; WebUI51pass; extension51pass;2independentreviewsclear. Productionunchanged; finalnew-headCI/mergepending.
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

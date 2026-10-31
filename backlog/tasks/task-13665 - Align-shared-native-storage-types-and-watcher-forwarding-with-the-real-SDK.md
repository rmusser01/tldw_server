---
id: TASK-13665
title: Align shared native storage types and watcher forwarding with the real SDK
status: Done
labels:
- chat
- storage
- testing
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct five native source compiler failures exposed while qualifying TASK13664: safeStorageSerde annotation narrows an already-tolerant runtime deserializer to string, settings watch wrapper drops the native callback context argument, and direct runtime storage requires null although native SDK setters resolve void. Preserve existing serialization, owner guards and runtime values; qualify native/WebUI callers and exact watcher context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tolerant storage deserializer exposes its actual accepted input type without altering runtime validation or serialization.
- [x] #2 Settings presentation watch adapter forwards the native callback context and direct runtime accepts actual native/WebUI setter outcomes without converting platform values.
- [x] #3 Both source type contexts compile; focused behavior regression, connected storage checks and independent review recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment:no new storage architecture; existing049/068 ownership and13651 storage guard contracts. Native scopedsource checker shows5errors in useChatSettingsRecord92, api-send231, chat-account-boundary17/19 and settings/local-bucket203. WebUI same source tree types0; no SDK replacements or permissive casts. TASK13664 feedbacktests694eachprofile0skip, character-create nativePASS1; finalPhase7/newpreview/fullcritical pending.
Watcher context regression REDmissingsecondarg thenGREEN1; exactmappedunwatch identity retained. PublicSerde inputtype intersection retains originalSDK genericcallers/runtimebody unchanged; apiSendreturnsactualSetnull/Removevoid onlyafteroldpostIOassert. Native andWebUI scopedsource compilers0 coveringall13664sharedfeedback dependencies. Boundedindependentbackendreviewclear. Connectedregressions/lint/finalunitrecord stillpending; no wholePR readiness.
Qualified:11connected suites227pass0skipeachnative/WebUI; real scopedsource compilers0each;4ownedlintfiles genuinelychecked root config/Nextroot,0errors5unchangedHEADwarnings0newmessages. Earlierfrontcwdsharedfilesignoredandprivateambientconfigdiagnostics excluded. Boundedbackendreviewclear. Serdeunknown overload defaultunknown preservesSDKgenericcalls, runtimebodyunchanged; watchercontext+unwatch exactidentity; guardedsetterreturnsactualSDK outcome afterpostIOguard. Review Docs/Reviews/CHAT_NATIVE_STORAGE_CONTRACTS_13665_2026_10_13.md. Owncompletedplanretired /private/tmp/IMPLEMENTATION_PLAN_chat_native_storage_contracts_13665_completed.md. BanditN/A TSonly; parent13664/master/fullPRreview/rebase/CI/mergepending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Aligned three native storage contracts without altering serialization, storage operations or ownership checks. Preserved watcher context and unsubscribe identity, exposed existing tolerant input, and forwarded real guarded setter outcomes.227tests/profile0skip, both source compilers0, lint0new, bounded review clear. WholePR qualification remains open.
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

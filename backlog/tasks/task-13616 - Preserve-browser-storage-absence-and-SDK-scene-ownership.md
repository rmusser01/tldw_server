---
id: TASK-13616
title: Preserve browser storage absence and SDK scene ownership
status: Done
labels:
- chat
- tests
- ownership
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four SDK mode integration suites use the real private Scene reader but instantiate native Plasmo without a captured extension storage API, so the fail-closed registry correctly refuses durable absence. Supply a scoped test-only Chrome storage API backed by the existing browser storage shim, keep the installed native Plasmo implementation and all production guards, and verify physical roundtrip/failure/API-retirement controls plus the original SDK suites.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SDK fixture uses the actual native Plasmo class with a captured storage client; shared runner aliases and production persistence guards are unchanged.
- [x] #2 Physical writes roundtrip, denied writes reject, and retiring the API prevents persistent-storage certification.
- [x] #3 Persona/dynamic UI/overlay/image-sync original scene and stale-source suites pass without granting H1 authority or weakening assertions.
- [x] #4 Both normal client types, normal hooks and bounded independent review qualify the unit; installed-extension limits are recorded.
- [x] #5 Browser storage requested-key reads omit absent keys, preserve stored null/falsy values, and remain compatible with native Plasmo verified absence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no; test-only declared runtime/storage boundary under existing captured-client persistence contract. Four callers of sdk-original-scene-fixture identified; real sourcecapture retained. Existing pattern background.quick-ingest-authority supplies chrome.storage from browser shim; native Storage code captures primaryClient at construction and requirePersistentStorage validates that instance plus active API. Highest local/origin/openPR taskID13615checked; no globalsharedUI aliasreplacement. UserallremainingPR3229authorized.
RED68failed4files. Adaptercontrols3pass, but indirecthelper and directfirst-static-import attempts still68failed40errors because parent eager importOriginal module mocks precede staticfixture execution, leaving previouslycapturedclients unavailable whileAPIlaterappearsactive. Independentreviewconfirmedordering. Reassess to explicit awaitedvi.hoisted initialization peraffectedfile before imports/eager mockfactories; finalboundedattemptwill retainactualPlasmo/guards. Ifthatcannotqualify, stopandusea scopedpre-module setupFiles/project boundary ratherthanretryingstaticimports.
Stopped afterthreeordering repairs: indirectfactory,directfirststaticsetup,awaitedhoistedsetup allleft68failed40errors while3nativeadaptercontrols passed. Recordlogs/private/tmp/chat-sdk-native-storage-13616-{green,ordered,hoisted}. Reassess captured-instance state ratherthanrepeatordering guesses. Nextdiagnostic delegatesactualrequirePersistentStorageunchanged and prints onlybackend/API/client capabilitybooleans at onePersona case, no values/keys/accounts/secrets. Compareactualcapturedclient shape tostandalonepositivebeforechoosinga scopedsetupFiles boundary or removingconflicting fixturemocks.
Capability diagnostic corrected the diagnosis: real scene instance extension:true/local/clientget+setfunctions/APIactive; refusalnowregistry_record_invalid. Wxt shim get addsrequestedmissingkey:null, while actualbrowserStorageArea.get omitsmissingkeys (MDN/Chromiumcontract). local-bucket correctlyacceptsonlyundefined asverifiedabsence andpreservesnullinvalid. MinimalproductionfixskiprawmissinggetItemnullbutkeepstoredJSONnull/falsy. Addrootabsencecontrols, removetemporaryguarddiagnostic andunnecessaryhoistedcomplexity. ADR:no newrule, existingbrowserAPIcompatibility andfailclosedabsencecontract; Docsreference https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/storage/StorageArea/get .
Rootfix qualified72nativeSDK/scene and10browser-contract tests; boundedcorrectedreviewclear; normaltypesbothpassed. Wider6suite run5passed75cases, runtime-bootstrap37cases failed inheritedNavigator WebIDL brand error (also in priorCIshard5): fixtureObject.create(window.navigator) invalidatesuserAgentreceiver whenZodloads. Fixonlyfixture tostubrealwindow.navigator withrestorablelocksproperty, preservingallauth/logout/ACKnegative assertions; includewholecallermatrix beforecommit.
Final72nativeSDK/scene/adapter controls and112widerbrowser/bootstrap/storage/model-save callers passed. No unhandled errors; all37bootstrap auth/ACKnegatives retained afterrealNavigator/restorablelocks fixture repair. Bothnormaltypespassed; lockedBun1.3.2/Next16.1.4 productionTurbopack/token/budgetpass589.7KBshared/868.4KBheaviest (caps600/900unchanged). Review Docs/Reviews/CHAT_BROWSER_STORAGE_ABSENCE_13616_2026_10_12.md. No guards/H1authority relaxed; scopedAPI is notinstalledextensionacceptance. BanditN/A TS/Markdown-only. Owncompletedplanretired/private/tmp. FinalboundedNavigatorreviewpending; wholePR/CI/H2/finalreview/rebase/merge remainopen.
FinalboundedNavigator/API reviewCLEAR:noCritical/Important/Minorfindings. No temporarydiagnostics/hoistedcomplexity/productionguardoverride remain. 72nativeSDK+112widercallers, bothnormaltypes and lockedproductionbuild/token/budget passed. Normalexactfilehooks next; wholePR/CI/H2/finalrequesting-code-review/rebase/merge stillopen.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Honor browser storage get missing-key omission while preserving stored null/falsy values. Declare scoped API for real native Plasmo SDK scene tests, retain fail-closed private source checks, and repair bootstrap Navigator fixture identity with exactcleanup.72nativeSDK+112widercallers/types/lockedbuild/budgets passed; boundedreviewclear, installedextension/fullPR limits recorded.
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

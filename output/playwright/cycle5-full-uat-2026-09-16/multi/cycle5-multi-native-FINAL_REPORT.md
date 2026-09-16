# Cycle5 fresh multi-user UAT — final

**Execution complete; acceptance fails.** Product remained frozen at `ab527eb3b4`. No product/Backlog/shared-tracker edits or commits by this agent. All12rows have outcomes or explicit limits; no pending native work.

## Environment

Fresh private config/data, API18501/UI18581, existing dependencies reused. First browser navigation08:00:01Z, final result~09:32Z on2026-09-16. Root owned processes/config/source. Root operator provider adaptation08:04:45Z and restart to API PID20095 preceded main Alice testing; this is **not a multi wizard pass**. Root later stopped20095 for real offline verification and restarted unchanged profile as69593. Root paused runtimes after completion.

Real llama.cpp9099/Gemma used under serialized leases; unavailable Ollama supplied real failures. Last model lease released after reanalysis terminal/reload~09:21Z. No mocked responses, token/clock changes, permission/ACL changes, permanent deletion or subsequent inference. Separate isolated browser proved natural expiry without background app refresh.

## Twelve-row matrix

|#|Workflow|Final outcome|
|---|---|---|
|1|Fresh setup/provider/first Chat|**PASS, operator adaptation.** Real admin UI creates Alice/Bob default Users; unverified admin TestConnection works. Alice real first response succeeds.|
|2|Auth/reload/logout/offline/reconnect/expiry|**PARTIAL: UAT134.** Normal account switches/reloads pass. Natural parked expiry1830.947s→refresh200→ownedNotes200. Real outage shows useful guidance/readiness gate; Retry after unchanged restart restoresNotes200. Later active expiry loops auth/sessions401.|
|3|Ordinary2turns/reload/failure→Retry|**PASS.** Initial5canonicalrows; separate realOllama502→GemmaRetry reuses exact user once, excludes displayerror, final reload3rows.|
|4|Publicfile→search→QA/citation/sourceChat|**FAIL UAT127/132.** Real Minimize/nav/resume preserves job2 without duplicatePOST. Backend Warning+savedMedia1 but UIstuckProcessing0/1. Explicit Media continuation independently proves useful citedQA, correctsourcepreview/jump and groundedChat. Query filtering returns nonmatching ownitems(UAT132).|
|5|ExactWikipedia→search→Chat|**EXTERNALLY BLOCKED.** ExactURL once; honestAccessblocked/0success1fail. Article/search/Chat dependents blocked. Separate completedexpirycontext used as approved adaptation; brokenmainjob2 untouched.|
|6|BiologyNote→exact5cards→distinctreviews|**PASS.** Exactfivefacts, realgrounded5cards, save5, distinctreveal/rate5 with settlednextfetches; session1completed5/reload.|
|7|SavedPirate→appliedChat|**PARTIAL UAT129/131.** Save/reloadSynced#1, actualUseinchat→SystemInstruction, realpiratereplyliteralARRR; canonicalexactsystem+answer. Collections401(UAT129), duplicatecanonicalgreetings(UAT131). DirectoutgoingPiratebody unavailable; no repeatedsend to manufactureevidence.|
|8|Charactercreation/replacement/reply/reload|**FAIL reopenedUAT068.** Coldcancel/reopen/create201TestBot4, freshentryclearspriorcontext, realBEEPBOOP/savedroute/reloadpass. SavedselectorDefaultAssistant1 fails toretaintarget; explicitsecondclickreceipt+2.148ssettlement/no dialog. No wrongidentitygeneration.|
|9|ChatNote/backlink/CardStudy/mixed/practice/End|**PASS tested save/reuse/study/End; Undo limited.** Actualprovenance/backlink, reviewedcard201→review200/completion1, correctmixed6cardpractice. ScheduledCramone rating→session3→actualEnd200completed1/4remaining. No positivemultiUndo certification or repeatedreratecycle(known128).|
|10|Mediaanalysis→Review→reanalysis/reload|**PASS, explicit continuation.** Actualanalysisv2, Reviewraw/formattedinspection, unavailableanalysis502preservesprior, restoredGemmareanalysis200/save201v3/reload200unchangedraw. ReviewReprocess was chunk/embedonly, notanalysis.|
|11|PermissionDelete/soleadminTrash/restore|**PASS, APIfixture adaptation.** OrdinaryDelete disabled withguidance. Adminemptylibrary→one syntheticAPIupload→actualTrashDELETE200→empty0/0keepsTrash→Restore200→list1/1. Sourcepreserved.|
|12|ReciprocalAPI/browser/confidentiality|**PASS tested boundaries; draft limit.** OwnvalidNotes201/read+update200 precede foreignGET+PUT404bothways; Chatown200/foreign404; jobowner200/Bob403; QAhistoryAlice1/Bob0. BobUI/BackexcludeAlice; AlicereturnexcludesBob. Indigoexistingsecurityfilterexcluded1/retained0/noanswer/excerpt; AuroraactualcitedQApositive. Dedicatedunsaveddraftcrossaccountrestoration notexercised.|

## Confirmed failures

- **UAT127/TASK13260.67:** job2 completed08:23:07 withWarning andownMedia1; resumedUIstaysProcessing0/1. Nevercancelled/reset/resubmitted.
- **UAT129:** Promptscollections401 despite otherauthenticated200; actualPromptsaved/reloaded.
- **UAT068/TASK13260.15 reopened:** TestBot4→DefaultAssistant1 selector doesnotretainreplacement. Click08:55:05.145Z→observation08:55:07.293Z, same savedroute/mode, no dialog.
- **UAT131:** oneweatherSend; canonicalidenticalgreetings68450214…08:56:28.974Z and2988cf09…08:57:06.483Z. EarlierUIalreadyhadgreetings; exactcreationrequestcausalitynotcaptured.
- **UAT132/TASK13260.73:** correctPOST/media/search `{query:<marker>}` returns ownnonmatchingitem; no foreignleak, no query-filterpass.
- **UAT134/existingTASK13260.24:** parkedexpiryproofpasses; lateractiveexpirycontextgets twoauth/sessions401 in6.5s09:17:20–26. No tokenmutation/repeatedprobe.

## Key evidence (all under `/private/tmp/cycle5-multi-native-`)

|Control|Artifacts|
|---|---|
|Expiry/outage/reconnect|`expiry-issued.txt`, `expiry-return.txt`, `expiry-late-status.txt`, `offline-test-wire.txt`, `offline-notes-settled.txt`, `reconnect-notes-canonical.txt`|
|Ordinary/Retry|`ordinary-reload-canonical.txt`, `retry-preretry-canonical.txt`, `retry-success-wire.txt`, `retry-final-canonical.txt`|
|Ingest/QA/source|`ingest-resume-terminal.txt`, `ingest-settled.txt/.png`, `qa-wire.txt`, `qa-source.txt`, `source-chat-wire.txt`|
|Wikipedia|`wikipedia-result.txt`|
|Biology/Study|`biology-generate-wire.txt`, `biology-save-wire.txt`, `study-reviews2-5.txt`, `study-reload.txt`, `chat-card-save.txt`, `scheduled-cram-review.txt`, `scheduled-cram-end.txt`|
|Prompt/Character|`prompt-collections-reload.txt`, `pirate-canonical.txt`, `character-reply-wire.txt`, `character-reload-canonical.txt`, `replacement-receipt.txt`, `default-assistant-dom-identity.txt`|
|Analysis|`reanalysis-failure-wire.txt`, `reanalysis-success-wire.txt`, `reanalysis-reload.txt`, `review-selected.txt`|
|Ownership/permission|`isolation-api.json`, `bob-note-save.txt`, `bob-back-state.txt`, `alice-return-notes-settled.txt`, `admin-trash-wire.txt`, `admin-restore-wire.txt`|
|Confidentiality|`indigo-upload.json`, `indigo-search-wire.txt`, `indigo-result.txt/.png`|

CanonicalIDs/dataimpact and detailed chronological qualifications: **`cycle5-multi-native-DETAILED_EVIDENCE_INDEX.md`** and **`cycle5-multi-native-running-tracker.md`**. All syntheticrecords retained. Sixcards; persistedreviewevents5+1+1 acrosssessions1/2/3; two schedule-offpracticeevents arelocalonly. Adminsource restored. AliceAuroraMedia1/raw963chars/v1-v3 preserved; IndigoMedia2 added; BobprivateMedia1distinctnamespace. No failedconversation/source overwritten.

## Limits and adaptations

- ExplicitAPIfixtures for isolation/adminrestore/Indigo are not UIingestionpasses. RealpublicAuroraQA is the confidentialitypositive; IndigogenerationOFF returns excluded1/retained0/no documents,citations,answer and native securityguidance. No9099call fornegative.
- Requestobserver mismatches/timeouts and streambodyunavailability are retained, not counted as productfailures or silentlyrepeatedactions. Lastreanalysis capturesbothgeneration+versionssave. InitialAPIloginJSON422 corrected to actualOAuthform beforefixtures; guessedKnowledge404 corrected to observedroute; AntDesignoverlaidinputs used visiblelabels. No forcedDOM/state mutation.
- ReviewReprocess actedimmediately duringinspection; rootverified chunk/embedPOST200 and no9099concurrency. It is not a reanalysispass.
- Nativehidden-tabnotifications, newlycompletedreasoning-onlynegative, successfulvision/recovery, dedicatedunsaveddraftownerrestoration and positiveUndo remainexplicitlimits. No broaderplatform/fullcleaninstallclaim.
- Broadrequestinventory was automaticallyrejected for credentialrisk and not retried; specificsafeobservers used. Evidencecredentials/JWTscan and hashes accompanyreport. No furtherbrowser/network/codeactions afterrootpause.

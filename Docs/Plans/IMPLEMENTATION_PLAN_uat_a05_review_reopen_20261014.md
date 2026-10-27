# A05 review handoff on wizard reopening
Task: TASK-13260.278.18.83.45. Frozen523ef323 plus preserved WIP, A05 only.
ADR assessment: existing ADR049 owner-scoped session/asset lifetime governs. A memory-only publication receipt uses the existing authority, session and generation; no persisted schema, durable architecture or backend change.

## Stage1: Reproduce the completed review loop
**Goal:** Same outer wizard reopening must allow Ingest More after its first handoff.
**Success Criteria:** Mounted real wrapper openRevision replay test fails on second navigation; existing full-remount review recovery and edited draft preservation remain.
**Tests:** Existing session suite with real wrapper/store, normal mocks for external transport/local storage; no model or UAT counterfeit.
**Status:** Complete

## Stage2: Scope one publication per mounted session
**Goal:** Retain the handoff receipt outside the remounted wizard provider.
**Success Criteria:** Authority/session/generation-scoped receipt gates navigation, new session publishes its own batch, pending close/unmount/reopen/failed-draft retry remain safe.
**Tests:** Causal green plus complete affected session/integration UI/Web, configured types/lint, source review. TS Bandit N/A/operator Ruff/Bandit.
**Status:** Complete

## Stage3: Corrected client and remaining WAV cell
**Goal:** One necessary corrected build/owned frontend transition, then start a fresh ingest normally.
**Success Criteria:** Existing reviewed draft/source preserved, no repeated completed-review loop, fresh ordinary WAV review reaches actual Parakeet ONNX processing once. Earlier WAV dedup and M4A proper-name failure remain separate.
**Tests:** Same one Chrome tab/profile/API/provider, existing deadlines, normal forms/chooser, real result/identity/oracle; no forced store/cookie/DOM changes or model installation.
**Status:** In Progress

Causalinitialnegative repeatednavigation1red; firstnewsessionpositivefixture reusedseedID, correctedfreshfactoryID before1red1positive. Initialouterauthority/session/generation memoryreceipt162UIpass, butindependentreviewImportant failednavigationlockedreceipt: incompleteversionNOTdeployed. Threeadditionalrenderedcontrols caused2red1newer-receiptpositive beforematching-key releaseonnavigationfailure. Reservependingbeforedispatch; releaseonlycapturedkeyonasyncsidepanelfailureevenafterinnerunmount, preservemounted/currentownerUIfences; syncWebthrow releases/rethrowsintooriginalreviewError/manualRetry. Finalcomplete165UI/165Webpass/exact58bindingsrestored, alloriginalfull-remount/edit/pending/cancel/authority controlsretained; independentfullsource/all5newbodiesreviewRESOLVED/CLEAR. Two introducedtestanywarningsfixedonlytounknown; emittedJSidentical/no runtime replay, currenttypespending. Necessaryfresh41bindingbuild/old90437-owner90432frontendtransitiondelta preparedRuffBandit0/sourceoperatorreviewpending; API73132/73129unchanged/oneChrome. No newmodels/cookie-store-reset/persistedschema/actualnativewindows/oldOCRresubmission/publication. Actualreopen/WAVstillunrun.

Currenttesttypesnatural0 afterunknown-onlyannotationfix/exact58bindingsrestored/emittedJSidentical, no165runtime replay. Finalbaseline/currentlintWidget0errors2unchangedwarnings/test0errors63unchangedwarnings. Necessary41bindingbuildO2Je6OeG2o_Imcf_UR3yc natural0/125.33s/compiler-token-bundleall0/restored; separateconfiguredtypes0. Old90437/90432stop0.1s/new98415/98410ready0.461s/actualoneTabBUILD_IDverified; API73132/73129same. Actualfirstcompletedreviewremountstillhandoffsoriginaldraft, secondreopenstayssameDocument andStartNewreachesfreshqueue; oldrevieweddraftpreserved. WAVfirstdecodeprepared337316B/actualcanonicalParakeetmodelvisible/analysischunkdiarizationoverwriteoff/ReviewBeforeSaveon. BackSettingscollapsedadvanced selectorwaitfailedbeforeStart, correctedexpand/verifiedmodel; notwrongmodel orproductfailure. ONE process-audiosStartACK39ms/POST1 pendingrealoutput; no oldOCRresubmission/newmodels/runtimeconstruction/publication. RetainedPG15492TCPportreachableonly, notfixture/databaseacceptance.


### Canonical WAV result — 2026-10-14
One process-only POST /media/process-audios, ACK39ms, HTTP200. Network responseEnd1809.32ms; first observed response79.477s later is observation latency, not processing duration. New review batch quick-ingest-review:3e474db6-9b24-4abf-9224-745679e57575/draft7022ff8e-b178-4954-ac10-9392254d1166 preserves the unedited transcript: Rowan correct; telescope Celine instead of Selene; Friday at six correct. Factual oracle FAIL. Actual same mounted wizard reopen/Start New and new-session review handoff work; no old draft overwrite.
Source trace: process-audios owns a fresh TempDirManager and forwards form model through run_audio_batch/process_audio_files to speech_to_text. The Parakeet branch returns before Whisper transcript-cache handling; ONNX loads the WAV and invokes the model on its samples. The result supplies no execution/model provenance, so this source trace plus visible canonical model selection is not an independently captured runtime-model receipt. No cache/model inspection or replay, vocabulary substitution, hand correction, extra decode, A06 or full A05 acceptance. Current API73132/73129 and frontend98415/98410 retained.

2026-10-14 PG final host result: actual initialization/Admin/provider pass and API18104/18066 ready11.334s; ordinaryAlice3 and Admin1 role/managed1MiB quota readbacks succeeded. No valid upload/quota acceptance: manually stepped sequence missed shared deadline; one upload POST was17947.673ms AFTER expiry, next-proxy500/1.920ms, not PG quota API defect evidence. Actual PG ASGIexit-15/7.619s/officialfixture release, pytest1/577.29s. Original API73132/73129 stopped2.639s and same-profile/source/keys API21278-owner21275 restored8.239s; normal Alice2/User login and exact old WAV draft Celine text preserved, sameoneChrome/tab/frontend98415. Three prior host failures retained, unchanged driver STOP; no fourth runtime or ASR replay. New serialized deadline-admission operator repair is preparation only; A05quota/ASR/humanUX/otherrecoveries remainopen/A06unstarted/heartbeatpaused.

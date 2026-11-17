# B04 TTS voice selection, playback and export

**Task:** TASK-13260.278.18.83.45; ordered B04 assessment recorded before this plan.
**Spec:** RELEASE_UAT_PLAYBOOK B-04 and release catalog; exact F-TTS sentence, two genuinely available voices when advertised, real decoded audio/playback/export/reload/recovery and human listening/UX.
**Status:** Selected functional + human UX/listening ACCEPTED on2026-10-16;16/43 selected accepted/27open. Engineering handoff remains open on frozen unpublished candidate.
**Candidate:** Frozen523ef323e9f56bc31559f6c5b8e506c27e599dcb+WIP, unpublished/not latest dev. API22784/22778 port18906/private sqlite-multi/Pocket mode-restorationready7.433s; frontend55878/55802 port18985/corrected94 build1u0VJmiiGJNxqp6xXoAEq; existing Chrome48388/48383/CDP18987, one tab. B02 retained unsent draft must survive navigation.
**ADR assessment:** No new durable architecture rule proposed; reuse existing audio/provider, owned request/playback and ADR059 task contracts. Reassess only for an actual source defect requiring a changed rule.

## Constraints

One existing Chrome window/tab; no new context/window. Preserve original canonical B02 five rows/Interrupted/draft and private profile/data/keys. No unrelated installs/downloads/model changes or service controls. Human explicitly authorized downloading and using Pocket TTS ONNX for B04; necessary declared dependency install, owned model/config setup and guarded API reload are within that scope. Normal UI/API authentication, credentials only in RAM; no auth store/cookie/header injection. No proof bundles/manifests/duplicate exports; keep only functional generated audio needed for playback/export. No Git/ref/staging/rebase/publication/queue action; heartbeat PAUSED. Navigation/save15s, ACK2s, short synthesis180s, background600s, startup120s/shutdown15s only if independently necessary and authorized. At most ONE declared recovery per actual case/max3 attempts then explicit reassessment; preserve failures and never weaken assertions. User provides screenshot UX and actual listening acceptance separately. Existing native/installer/current-dev/canonical/final-review/hooks/publication/deployment/upgrade/extension/release gates remain open.

## Stage1: Existing provider and source prerequisites
**Goal:** Determine a real available synthesis provider/voices and actual UI/request/artifact paths.
**Success Criteria:** Inspect route/provider/catalog/playback/history/export and three adjacent patterns; get current availability through normal API. Do not infer installed readiness from a model name. Declare functional fixture/source binding and owned handback.
**Tests:** Read-only normal metadata and UI inspection; no generation or downloads.
**Status:** Complete

## Stage2: Real audio and playback/export
**Goal:** Generate exact F-TTS and verify usable provider/voice-bound audio.
**Success Criteria:** One initial generation per advertised chosen voice, exact requested provider/voice/format; positive independently decoded duration, no silence; play/pause/resume/seek/stop without overlap, normal export/reopen and retained history/reload. Human complete-sentence listening is required.
**Tests:** Reviewed bounded real browser/API driver, necessary functional audio decoding and one retained artifact per output; no fake provider/audio or output-only success claim.
**Status:** Complete

## Stage3: Recovery, isolation and causal repairs
**Goal:** Truthful unavailable/invalid/long/cancel/failure outcomes without lost user work; applicable database/auth composition.
**Success Criteria:** Cover empty input/unavailable voice/provider failure/unsupported format/long input/cancel through actual supported paths. Diagnose any source failure before minimal causal red/green, affected configured suites/types/lint/scoped baseline-current Bandit/source review and necessary delivery; official restricted PG fixture only, no operator SQL.
**Tests:** Actual bounded recovery and ownership checks; supported source tests for controlled cases with explicit limits.
**Status:** Complete

## Stage4: Human UX/listening, safe closeout and engineering handoff
**Goal:** Screenshot review/listening, honest selected accounting and original handback.
**Success Criteria:** Human accepts actual usable speech and screenshot UX; original full canonical B02 rows/Interrupted/unsent draft restored with no unintended inference, task/master current. Selected UAT is complete; retain this plan until the related final staged review/normal-hook commit and engineering handoff are complete. Broader release gates remain separate.
**Tests:** Independent authorized final source/result review, exact current receipts/handback, no unchanged replay.
**Status:** In Progress

## Initial prerequisite observation — 2026-10-16

Normal authenticated GET /api/v1/audio/providers returnedHTTP200 with providers{} and voices{}, explicit-backend supporttrue. Focused Kokoro model-info returnedHTTP200/statusnot_initialized/initializedfalse/loadedfalse/failedfalse/model_ids[]. No synthesis, model initialization/download or service restart. Private legacy config selects Kokoro but does not establish enabled/installed readiness. Registry source skips local capabilities unless enabled; ProviderConfig defaults enabledfalse. This explains a possible empty-config catalog boundary, not a proven application defect. Asked the human which existing installed TTS model/path or running service to use; answer pending. Continue independent UI/readiness/recovery assessment. Exact old B03 timeout cause remains unknown and all previous failures retained.


## Settled B04 UI and no-generation recovery checks

Same Chrome tab navigated only after exact retained B02cid/draft/noStop guard. Settled /tts shows tldw Server (Local), persisted kitten_tts/KittenML/kitten-tts-nano-0.8/Bella choice and truthful kitten_tts Blocked readiness. Browser local output Ready is a separate advertised route; it does not qualify server synthesis or export. Empty text Play/Stop/Download disabled. After entering exact F-TTS, Play remains disabled with the explicit selected-provider-not-reported recovery message pointing to Settings -> Speech; Stop/Download remain disabled, exact text retained and zero actual speechPOSTs. One current screenshot overwrote /private/tmp/tldw-uat-current-ux.png; no duplicate proof. No settings save/provider change/playback/model initialization or generation. Existing configured provider question remains pending; no repeat.


## Pocket TTS ONNX explicitly selected and download authorized

Human said download+use pocket tts onnx, superseding the pending existing-provider question. Runtime metadata showed only declared sentencepiece missing; sentencepiece0.2.2 wheel installed naturally0 into existing Python3.12 private runtime, no other distribution changed. Existing adapter PocketTTSOnnxAdapter/pocket_tts is selected, not pocket_tts_cpp. Exact upstream revision58a6d00cf13d239b6748cb0769f35c580a8f606c inspected for imports/constructor/inference/voice/I/O. Current module requires bundle.json; the root legacy file listing alone is insufficient despite model-card wording. Use compatible English2026-04 bundle directory with three INT8 main/flow/decoder files plus required encoder/conditioner, tokenizer/bundle/BOS metadata, module/license and upstream reference sample only. One download started to owned clean-install/models/pocket-tts-onnx-20261016, no unrelated languages/FP32 duplicates/shared-cache inspection. Models approximately166MB, no proof hashes/bundles. No synthesis/API restart yet. Private provider YAML must be narrowly enabled with absolute owned paths and CPU, preserving existing config/profile/data/keys. Adapter requires uploaded reference/custom voice; upstream sample will be used for initial validation, no personal voice inference.


## Documented module-path configuration loss / minimal repair design

Pinned English12file download completed165566035bytes/6.294s; no unrelated models. Before API setup, dataflow review found ProviderConfig omits documented module_path; Pydantic drops it at YAML parsing, registry serializes only retained fields, Pocket adapter then defaults module location to models_dir.parent. English bundle requires models_dir under onnx/english_2026-04 while downloaded module remains top-level, so documented absolute module_path must reach the adapter. Add one optional module_path field to existing typed provider schema, preserving defaultNone/unknown-key policy/credential redaction. First real YAML-manager -> registry configuration -> Pocket adapter constructor regression should fail on lost path. Then one-field fix, affected config/registry/adapter tests, compile/Ruff/baseline-current Bandit/independent source review before private config/API transition. No source workarounds, copied wrapper or import shims. Existing tokenizer/model_path/binary_path are adjacent typed patterns; governing ADR unchanged.


## Pocket runtime loaded / real speech passes / picker finish exception pending

One-field module-path repair qualified real red1failed30warnings0.53s -> six affected config/security/registry/adapter suites30passed37warnings0.78s; Ruff/compile/newtestformat0, scoped productionbaseline/currentBandit0/newtest0, independent actual-sourceCLEAR. Private YAML only Pocket CPU/absolute ownedpaths/autodownloadfalse/fallbackfalse passed actual config/registry/adapter asset validation. Reviewed fresh-marker API transition stopped only14195/14083 within15s, started54871/54868 once/canonicalready10.196s. Actual provider loaded3.459s; focused model-info available/initializedtrue/loadedtrue/failedfalse. Upstream sample uploaded once200/custom a054ebc2-0ee7-46d5-a5ea-384209865f42/8.938s/providercompatible.

Three picker operator failures are retained, all before synthesis: covered combobox input intercepted, duplicate proxy/visible model text, then Pocket correctly selected/voiceopened but diagnostic read matched two animation dropdowns before voice selection. Picker STOP. Concrete finish-only exact visible voice action was corrected from review to consume wx receipt before click/retainfailure, verifyselectedvoice, exactorigin, boundedreads/2sACK; finalreviewCLEAR. Asked ONE specific finish-only exception because AGENTS max3 consumed; answerpending. No fourth picker action.

Independent API initial synthesis failed400/0.041s before inference: operator wrongly passed gateway-only backend=pocket_tts. Sourcecanonicalize_gateway_id reserves local IDs, while documented local adapter path omits backend. Current failed-request log read ONCE filtered by requestid91dc6a9f-77d8-4957-be9a-1cb103404330 only; no old-log replay. ONE declared corrected-request recovery (modelPocket/customvoice/no backend/no fallback/WAV/exactF-TTS/180cap) passedHTTP200/2.034s. Real decoded24000Hz mono audio5.840s/RMS0.047029/280364bytes, functional output clean-install/b04-pocket-tts.wav; SHA256 8e075391518608a1e006ca998a68e5bd0964daf74dd3928c90e36708be8f01bc (required generated-audio identity, no model/proof hashbundle). Full actual file shown inline and humanlistening askedONCE/pending; non-silence/duration do not certify words. Earlier failedreceipt retained. This is actual server synthesis, not browser playback/export/reload/matrix/UX acceptance.15/43 remains, B04active/B05unstarted. No further API smoke replay.


## Additional real input recovery checks / current handoff

One distinct three-case actual API invalid-input unit passed: emptytext400, unsupportedformat422, provider-limit5400characters400. Each returned explicit JSONdetail/no audio; no valid-generation replay. Functional receipt b04-pocket-invalid-input-control.json complete. Exact user-selected PocketTTS model is loaded; normal privateAdmin/server auth retained. Same one Chrome tab remains /tts, Pocket model selected, Voice dropdown open and upstream custom voice not yet selected; no UI synthesis. Reviewed corrected finish-only action is CLEAR but UNEXECUTED pending the ONE specific exception already asked. Human listening of the sole generated WAV is separately pending. No questions repeated or pending consent inferred. No second-voice claim: current adapter has no built-in advertised catalog and only one custom reference was added. Rebase/publication/common engineering gates remain open; B04unaccepted15/43, B05unstarted.


## Human finish/listening granted / actual readiness alias defect

Human approved ONE reviewed finish-only voice action and accepted actual API WAV as complete/intelligible. Finish-only ran once PASS: actual selected Custom upstream voice verified, exact F-TTS retained, Playenabled, no synthesis. Approval consumed; prior3pickerfailures remain failed. Listening applies only to that actual artifact.

Readonly settled UI then displayed pockettts:Blocked despite actual initialized PocketTTS/Playenabled. Source traces inferTldwProviderFromModel -> pockettts; readiness helper directly indexes server providers map pocket_tts. Adjacent activeProviderCaps uses normalized lookup and provider requests already use existing toServerTtsProviderKey. Minimal shared readiness repair: reuse toServerTtsProviderKey for providerInfo lookup, preserve unknown/gateway IDs/other ready-warning-blocked semantics. First causal test compares normalized Pocket alias with canonical control, then one lookup/import correction, affected helper/strip/page configured tests/types/scoped lint/source review and one necessary91binding changed-client build before live readiness/playback/export. No new architecture rule/dependency.


## Readiness repair qualified / first browser generation scope

Alias causal red1failed1passed; corrected helper/strip/page47passed in each configured client and WebUI tsc0, scoped lint0, independent actual-source reviewCLEAR. Necessary91binding build natural0/token-sync0/bundle0/223.374s/build3KXh2-QhywxvU1PG7a_bi with exact restoration. Old90 remains served to preserve just-approved Pocket model/voice/text. First browser unit will generate once in exact WAV/one segment, verify actual wire provider/voice/input/body and normal UI download equality; keep native player for separately calibrated playback, no reload or second synthesis. Stop eligibility during in-flight nonstream generation is a source observation to validate separately; no fix or cancellation acceptance yet.


## Real browser output/playback and connected cancellation repair

Initial browser Output selector failed before generation/zeroPOST; one corrected-selector recovery passed exactly1Pocket/customvoice/exactF-TTS/WAV/streamfalse/no backend request,2586ms/5.84s/280364bytes. Normal UI export exactly equals actual response. Independently decoded24kHzmono/RMS0.034528/SHA256dd473441c303e8748f4f53163af1017d0e5cc13cae27606c42925513e9356ebb. This artifact differs from API WAV; browser listening asked separately/pending. Calibrated actual native Play/Pause/timelineSeek2.907558s/Resume/Stop passed with zero new speechPOSTs, retainedclip/paused0. Actual Stop at synthesis ACK was disabled.

Both useTtsPlayground callers omit isGenerating from Stop eligibility and never call existing abort/retirement on Stop. Initial causal3failed2passed -> minimal eligibility/clear-on-active correction57passed/client. Review caught connected empty-result continuation saving old voice after later retry; deferred caller causal2failed specifically stale settings writes. Both callers now return on empty/cancelled results before settings/history/selection side effects. Final5suites59passed in each client; reviewer requested stronger Stop provenance in two deferred fixtures (reset clear spy afterPlay/assert Stop enabled, legacytldw instead ofbrowser), targeted verification ongoing. Realhook abort/lateoutput/newretrypreservation passes. New93binding combined build prepared, no execution/delivery yet; prior91readinessbuild remains undelivered. No B04 acceptance/B05start/publication.


## Verified owner-scoped history repair / combined94 client

Independent source trace confirms fixed browser-wide speechPlaygroundHistory leaks text across accounts on remount; actual auth cleanup does not scope/clear it. Real-storage causal3failed (Alice->Bob, unownedlegacy, logout) before repair. New29line useSpeechHistory reuses existing verified useChatDraftOwner and useStorage; per-owner envelope guards both key and row owner, rejects malformed arrays, refuses stale owner writes, preserves unowned legacy storage without assigning it. Page boundary retires current TTS generation/stream/job/text; broader STT draft/recording not qualified. Four final real WebUI storage/owner tests pass in each configured client; six sharedsuites62passed/client plus addedfourthcase targeted4, not a fabricated single63 run.

Finalconfiguredtypes0; baseline6/current8 lintfiles0errors31identical inheritedwarnings0ignored; newanywarnings removed with existing segment type. Newfunctional Pythonhelpers baseline/currentBandit0findings0errors; addedTypeScriptBanditN/A. Independent finalsource/tests/build/serve/quiesce reviewCLEAR. Necessary combined94binding build running600cap, no frontend transition yet. Previous91build remainsundelivered;93neverran. Live corrected validation prepared for one distinct cancellation plus one legitimateRetry, actualabortedrequest/textretention/newownedhistory/reload; requires reviewed script, new94ready and exactidlequiesce. Human firstbrowserWAV listeningpending, APIWAVaccepted. No B04acceptance/B05start or Git/publication action.


## Corrected94 delivered / cancellation, export and restoration outcomes

Combined94 build natural0/compiler/token/bundle0/404.985s under600s; exact source bindings restored. Owned old90 frontend gracefully replaced by55878/55802, ready1.216s; API54871/54868 unchanged. Actual Pocket readiness now Ready. Initial corrected live unit failed before claim, no synthesis; original exception unavailable and not invented. ONE changed recovery then observed real cancelled requestERR_ABORTED/text retained/no history, exactly one legitimate RetryHTTP200/1436ms/6.0s/288044bytes, correct Pocket/customvoice/WAV/exactF-TTS/no backend. Owned history metadata survived normal reload and Use inTTS; UI export exactly actual bytes, normal exported-file native decode positive6s/source match. SHA256bc426e82b23b7dc2443f712332c7d90a93fe8a6c8820c452fb7a41885ece7dd5 identifies this necessary functional output only.

Recovery FAILED at final goBack DOMContentLoaded wait15s although URL reached /tts; deadline visibility unobserved, possible event-wait issue not proven. Cleanup unmountedaboutblank/drained bodies. Do not replay generation/export/Back. Separate finish-only normal goto(commit)+visible heading/readiness/exact owned history and Use inTTS restored existing text PASS within15s/zero speechPOSTs. Original failed receipt remains failed. Browser-wide legacy history left unassigned, not erased/adopted.

B04 screenshot UX question asked once; APIWAV humanaccepted, separate initialbrowserWAV listening still pending. Newmode helper source/lifecycle reviewed: same official restricted fixture, three fresh profiles, installed Pocket config before normal initializer, fixed single-user41/42 and PGmultiordinary35/36 avoid prior browser owners. Operator selfchecks18pass/1expected-error-message fail; stronger guard was correct, targeted expectation correction1pass; strengthened Pocket/fixedID contract3pass. Ruff0, scoped helper baseline/currentBandit0findings0errors. Browser review deadline/observer fixes applied/nodecheck0; primary SQLiteordinary-user isolation now running once. No fixture transition yet; B04 still15/43/B05unstarted. Full current-dev/combinedschema/stagedreview/hooks/rebase/publication/native/installer/release gates remainopen.


## Four real modes passed / original handoff restored / UX accepted

SQLite-multi Alice8/Bob9 passed one real synthesis1192ms/6s/288044bytes, exact owned history reload, Bob voiceGET404/no Alice history or audio, Alice return retained. Three fresh modecells each passed with exactly one actual correct Pocket/customvoice/F-TTS/WAV request, positive native decode, retained history and no observererrors: SQLite-single41 1018ms/5.92s/284204bytes; official restricted PostgreSQLsingle42 1093ms/6.08s/291884bytes; PostgreSQLmultiAlice35/Bob36 1150ms/6.24s/299564bytes plus same cross-owner isolation/Alice return. Hosts1pass29.84s/28.78s/49.64s; actual shutdown.619s/.501s/3.033s/applicationShutdownComplete/exit-15 before fixture release. No new model/download/build or duplicate output artifacts. Fixed single-user IDs are supported configuration; ordinary-user reservations use normal Admin API, not raw SQL.

Original sameprivateprofile/data/keys/Pocket/OCR API restored once22784/22778/canonicalready7.433s after all fixture releases. Reviewed browser handback corrected unknown-draft cleanup to preserve the refused draft/page and made bounded reads start after timeout computation. Normal Admin1 login, original owned TTS history/exact F-TTS/customvoice restored, then original B02cid d9ac3c7a-e59f-498c-820e-66fb7c74cd72/character7/five full canonical APIrows unchanged, Interrupted label and unsent B02 retained next question. restored, zero speech or Chat inferencePOSTs. Current single tab remains that Chat; never Send the retained draft.

Human accepted B04 screenshot UX. Browser audio player was not visible to the human, so the exact retained browser WAV was displayed again inline (no regeneration/download/copy), and listening question re-presented once in response to that delivery problem; answer pending. APIWAV listening alreadyaccepted. Stage3Complete by this explicit live/source composition; Stage2 and4 retain browserlistening gate. Overall15/43 accepted/28open, B04active/B05unstarted. Original finalBack event-wait FAILED remains unchanged; separate finish-only restoration and final handback PASS do not change it. Only one genuinely advertised custom Pocket voice is available; secondvoice variant conditionalN/A, not fabricated. Broader STTdraft/recording, current-dev/combinedcanonical/finalstaged/hooks/rebase/publication/native/installer/extension/deployment/upgrade/release remain open.


## Browser listening accepted — 2026-10-16

Human answered Complete and intelligible for the exact browser-generated WAV, completing the remaining selected listening gate. B04 is selected functional + human UX/listening ACCEPTED by the explicit composition above;16/43 accepted/27open. Stages1–3 complete; Stage4 retains only common engineering handoff, not further B04 runtime replay or another human question. All failed histories and broader current-dev/integration/native/installer/release limits remain. Continue B05 next.
